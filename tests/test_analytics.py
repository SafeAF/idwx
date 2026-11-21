"""Independent expectations for the public analytics contract and its boundaries."""
import hashlib
import json
from dataclasses import replace

import pandas as pd
import pytest

from idwx.analytics import (
    metric_policy,
    public_daily,
    seasonal_summary,
    summarize_daily,
    validate_daily,
)
from idwx.config import StationMeta, load_config
from idwx.web_export import _csv, _json, export_web_data
from idwx.web_verify import verify_web


def test_calendar_gaps_per_metric_missingness_and_public_field_selection():
    frame = pd.DataFrame([
        {"date": "2024-02-28", "tmin_c": -2, "tmean_c": 1, "tmax_c": 4, "n_obs": 24, "secret": "private"},
        {"date": "2024-03-01", "tmin_c": -1, "tmean_c": None, "tmax_c": 5, "n_obs": 24},
        {"date": "2024-03-02", "tmin_c": -8, "tmean_c": -2, "tmax_c": 4, "n_obs": 3, "quality_flags": "insufficient_coverage"},
    ])
    daily = public_daily(frame)
    rows = daily["2024"]
    assert len(rows) == 366
    assert rows[59] == {"date": "2024-02-29", "tmin_c": None, "tmax_c": None, "tmean_c": None, "status": "missing_record", "n_obs": None}
    assert rows[60]["status"] == "invalid_temperature" and rows[60]["tmin_c"] == -1
    assert rows[61]["tmin_c"] is None and rows[61]["n_obs"] == 3
    assert all("secret" not in row and "quality_flags" not in row for row in rows)
    validate_daily(2024, rows)
    summary = summarize_daily(daily)
    feb = summary["months"][1]
    assert feb["expected_days"] == 29 and feb["complete"] is False
    assert feb["tmin_c"] == {"valid_days": 1, "mean": -2.0, "min": -2.0, "max": -2.0}
    rows[59]["tmin_c"] = 0
    with pytest.raises(ValueError, match="Missing daily"):
        validate_daily(2024, rows)


def test_complete_seasons_and_missing_day_do_not_create_totals_or_runs():
    dates = pd.date_range('2023-01-01', '2024-12-31')
    daily = public_daily(pd.DataFrame({"date": dates, "tmin_c": -10, "tmean_c": 12, "tmax_c": 20}))
    policy = metric_policy(load_config('configs/idaho.yml'))
    result = seasonal_summary(daily, policy)[1]
    assert result["gdd"] == 214 * 2 and result["through"] == '2024-10-31'
    assert result["hdd"] == 91 * 6
    assert result["cold_days"] == result["cold_run"] == 91
    # One missing growing-season day truncates the continuous subtotal.
    daily['2024'][date_index('2024-04-03')]['tmean_c'] = None
    # A missing winter day prevents a confirmed total/max run.
    daily['2024'][date_index('2024-01-15')]['tmin_c'] = None
    result = seasonal_summary(daily, policy)[1]
    assert result['gdd'] is None and result['gdd_subtotal'] == 4
    assert result['through'] == '2024-04-02'
    assert result['hdd'] is None and result['cold_run'] is None and result['cold_days'] is None
    assert result['winter_valid'] == 90


def date_index(value):
    stamp = pd.Timestamp(value)
    return (stamp - pd.Timestamp(stamp.year, 1, 1)).days


@pytest.fixture
def publication(tmp_path):
    cfg = replace(load_config('configs/idaho.yml'), backtest={'start_year': 2000, 'end_year': 2022})
    root = tmp_path / 'site'
    manifest = export_web_data(cfg, [StationMeta('demo', 'Demo')], root, demo=True)
    return cfg, root, manifest


def test_daily_tampering_cannot_be_hidden_by_an_updated_checksum(publication):
    _, root, manifest = publication
    relative = f"data/{manifest['release_id']}/analytics/demo/daily/2020.json"
    path = root / relative
    chunk = json.loads(path.read_text())
    chunk['days'][0]['date'] = '2020-01-02'
    path.write_bytes(_json(chunk))
    manifest['files'][relative] = hashlib.sha256(path.read_bytes()).hexdigest()
    (root / 'data/manifest.json').write_bytes(_json(manifest))
    with pytest.raises(ValueError, match='daily analytics row'):
        verify_web(root, allow_demo=True)


def test_monthly_summary_is_recomputed_not_just_checksum_checked(publication):
    _, root, manifest = publication
    relative = f"data/{manifest['release_id']}/analytics/demo/summary.json"
    path = root / relative
    summary = json.loads(path.read_text())
    summary['months'][0]['tmean_c']['mean'] += 1
    path.write_bytes(_json(summary))
    manifest['files'][relative] = hashlib.sha256(path.read_bytes()).hexdigest()
    (root / 'data/manifest.json').write_bytes(_json(manifest))
    with pytest.raises(ValueError, match='summary disagrees'):
        verify_web(root, allow_demo=True)


def test_old_schema_can_be_active_or_retained_beside_new_schema(publication):
    _, root, manifest = publication
    current = root / f"data/{manifest['release_id']}"
    doc = json.loads((current / 'stations/demo.json').read_text())
    content = {k: v for k, v in doc.items() if k not in {'schema_version', 'data_kind', 'methodology', 'release_id', 'analytics_url'}}
    common = {k: manifest[k] for k in ('schema_version', 'data_kind', 'methodology')}
    common['schema_version'] = '1.0.0'
    release = hashlib.sha256(_json({**common, 'stations': [content]})).hexdigest()
    common['release_id'] = release
    prefix = f'data/{release}'
    files = {
        'stations.json': _json({**common, 'stations': [{'station': content['station'], 'source_coverage': content['source_coverage'], 'url': f'{prefix}/stations/demo.json'}]}),
        'stations/demo.json': _json({**common, **content}),
        'downloads/frost-history.csv': _csv([content], common),
    }
    for name, data in files.items():
        destination = root / prefix / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(data)
    assert verify_web(root, allow_demo=True)['retained_releases_verified'] == 1
    previous = {**manifest, **common, 'stations_url': f'{prefix}/stations.json', 'history_csv_url': f'{prefix}/downloads/frost-history.csv',
                'files': {f'{prefix}/{name}': hashlib.sha256(data).hexdigest() for name, data in files.items()}}
    (root / 'data/manifest.json').write_bytes(_json(previous))
    report = verify_web(root, allow_demo=True)
    assert report['release_id'] == release and not report['data_explorer_included']
    assert report['retained_releases_verified'] == 1
