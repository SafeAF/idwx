"""Precipitation has its own coverage and reporting-day contract."""
import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest

from idwx.config import StationMeta, load_config
from idwx.precipitation import load_archive, reduce_hours, validate_precip
from idwx.web_export import _json, export_web_data
from idwx.web_verify import verify_web


def test_hour_ending_at_midnight_belongs_to_previous_reporting_day():
    hours = pd.date_range('2020-01-02T08:00Z', periods=24, freq='h')
    result = reduce_hours(hours, [1] * 24)
    assert result == [['2020-01-02', 24.0, 24, 24]]
    assert reduce_hours(hours, [0] * 24) == [['2020-01-02', 0.0, 24, 24]]
    missing = [1.] * 24
    missing[3] = float('nan')
    assert reduce_hours(hours, missing) == [['2020-01-02', None, 23, 24]]
    with pytest.raises(ValueError, match='unique ordered'):
        reduce_hours(hours.insert(0, hours[0]), [1] * 25)


@pytest.mark.parametrize('row', [
    {'precip_mm': 0, 'precip_valid_hours': 23, 'precip_expected_hours': 24, 'precip_status': 'complete'},
    {'precip_mm': -1, 'precip_valid_hours': 24, 'precip_expected_hours': 24, 'precip_status': 'complete'},
    {'precip_mm': 0, 'precip_valid_hours': None, 'precip_expected_hours': None, 'precip_status': 'not_reviewed'},
])
def test_incomplete_negative_or_unreviewed_precipitation_cannot_become_zero(row):
    with pytest.raises(ValueError):
        validate_precip(row)


def test_curated_archive_coverage_and_independent_monthly_reference_totals():
    archive = load_archive(Path('data/precipitation/twin_falls.json'))
    complete = [row for row in archive['days'] if row[1] is not None]
    assert complete[0][0] == '1980-01-01' and complete[-1][0] == '2022-12-30'
    assert len(complete) == 15705
    assert archive['days'][0][1:] == [None, 8, 24]
    assert archive['days'][-1][1:] == [None, 16, 24]
    # Independently calculated by Python csv + datetime, not the pandas reducer.
    for month, expected in [('1980-01', 71.18), ('2020-02', 20.70), ('2022-11', 46.59)]:
        assert sum(row[1] for row in complete if row[0].startswith(month)) == pytest.approx(expected)


def test_publication_rejects_false_complete_precipitation_even_with_matching_digest(tmp_path):
    cfg = replace(load_config('configs/idaho.yml'), backtest={'start_year': 2000, 'end_year': 2022})
    root = tmp_path / 'site'
    manifest = export_web_data(cfg, [StationMeta('demo', 'Demo')], root, demo=True)
    name = next(name for name in manifest['files'] if name.endswith('/daily/2020.json'))
    path = root / name
    document = json.loads(path.read_text())
    document['days'][0]['precip_valid_hours'] = 23
    path.write_bytes(_json(document))
    manifest['files'][name] = hashlib.sha256(path.read_bytes()).hexdigest()
    (root / 'data/manifest.json').write_bytes(_json(manifest))
    with pytest.raises(ValueError, match='complete precipitation'):
        verify_web(root, allow_demo=True)
