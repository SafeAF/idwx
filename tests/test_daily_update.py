"""Publishing must preserve both live bytes and refresh state on failure."""
import hashlib
import shutil
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pytest
import yaml

import idwx.daily_update as job
from idwx.config import load_config, load_stations
from idwx.precipitation import _bytes, extend_archive, load_archive
from idwx.web_export import export_web_data


@pytest.fixture
def deployment(tmp_path):
    cfg = load_config('configs/idaho.yml')
    root = tmp_path / 'srv'
    previous = root / 'builds' / 'previous'
    previous.mkdir(parents=True)
    db = tmp_path / 'seed.sqlite3'
    shutil.copyfile(cfg.database_path, db)
    archive = load_archive(cfg.precipitation_archives['twin_falls'])
    archive['days'] = [['2026-11-06', 0., 24, 24], ['2026-11-07', 0., 24, 24]]
    archive.pop('content_sha256')
    archive['content_sha256'] = hashlib.sha256(_bytes(archive)).hexdigest()
    precip = tmp_path / 'precipitation.json'
    precip.write_bytes(_bytes(archive))
    effective = replace(cfg, database_path=db, precipitation_archives={'twin_falls': precip})
    export_web_data(effective, load_stations(cfg.stations_file), previous)
    (root / 'current').symlink_to(previous)
    raw = yaml.safe_load(Path('configs/idaho.yml').read_text())
    raw.update(database_path=str(db), stations_file=str(cfg.stations_file),
               precipitation_archives={'twin_falls': str(precip)})
    config = tmp_path / 'config.yml'
    config.write_text(yaml.safe_dump(raw))
    return root, previous, config


def fake_source(destination, start, end):
    stamps = pd.date_range(start, end, freq='h')
    pd.DataFrame({'timestamp': stamps,
                  'coordinates (lat,lon)': '[42.4634, -114.4975]',
                  'model (name)': 'era5', 'model elevation (surface)': 1318.38,
                  'utc_offset (hrs)': -7, 'temperature (degC)': 5.,
                  'total_precipitation (mm of water equivalent)': .5}).to_csv(destination, index=False)


def provider(monkeypatch):
    calls = []
    monkeypatch.setattr(job, 'era5_available_end', lambda key: datetime(2026, 11, 9, 15, tzinfo=timezone.utc))
    def fetch(key, destination, **kwargs):
        calls.append(kwargs)
        fake_source(destination, kwargs['start'], kwargs['end'])
    monkeypatch.setattr(job, 'fetch_era5_hourly_csv', fetch)
    return calls


def run(deployment, **kwargs):
    root, _, config = deployment
    return job.update(config, root, api_key='test-key', source_revision='test-revision',
                      health_check=kwargs.pop('health_check', lambda release: None), **kwargs)


def test_refresh_publishes_matching_snapshot_and_then_skips_weather_download(monkeypatch, deployment):
    calls = provider(monkeypatch)
    root, previous, _ = deployment
    original = (previous / 'data/manifest.json').read_bytes()
    result = run(deployment)
    assert result['status'] == 'published'
    assert result['last_complete_date'] == result['precipitation_last_complete_date'] == '2026-11-08'
    assert (root / 'current').resolve() != previous
    assert (previous / 'data/manifest.json').read_bytes() == original
    state = root / 'automation/state/current'
    archive = load_archive(state / 'precipitation.json')
    assert archive['version'] == 2 and len(archive['refresh_sources']) == 1
    assert next(r for r in archive['days'] if r[0] == '2026-11-08')[1] == 12.
    assert run(deployment)['status'] == 'already_current'
    assert len(calls) == 1
    assert (state / 'hourly-source.csv.gz').is_file()
    assert not any(p.is_dir() for p in state.resolve().glob('hourly-*'))
    assert (root / 'current/data/manifest.json').stat().st_mode & 0o777 == 0o644


def test_failed_provider_preserves_site_and_never_creates_active_state(monkeypatch, deployment):
    provider(monkeypatch)
    def fail(*args, **kwargs):
        raise ValueError('Provider failed')
    monkeypatch.setattr(job, 'fetch_era5_hourly_csv', fail)
    root, previous, _ = deployment
    with pytest.raises(ValueError, match='Provider failed'):
        run(deployment)
    assert (root / 'current').resolve() == previous
    assert not (root / 'automation/state/current').exists()
    assert list((root / 'builds').iterdir()) == [previous]


def test_failed_served_manifest_rolls_back_both_site_and_snapshot(monkeypatch, deployment):
    provider(monkeypatch)
    run(deployment)
    root, _, _ = deployment
    previous_site = (root / 'current').resolve()
    previous_state = (root / 'automation/state/current').resolve()
    def fail(release):
        raise ValueError('HTTP manifest mismatch')
    # Force a code update with already-current inputs; no extra weather fetch.
    _, _, config = deployment
    with pytest.raises(ValueError, match='HTTP manifest mismatch'):
        job.update(config, root, api_key='test-key', source_revision='next-revision', health_check=fail)
    assert (root / 'current').resolve() == previous_site
    assert (root / 'automation/state/current').resolve() == previous_state


def test_verify_only_never_calls_provider(monkeypatch, deployment):
    def fail(*args, **kwargs):
        pytest.fail('Verify-only must not contact Oikolab')
    monkeypatch.setattr(job, 'era5_available_end', fail)
    assert run(deployment, verify_only=True)['status'] == 'valid_static_package'


def test_refresh_audit_rejects_wrong_offset_and_missing_hours(tmp_path):
    source = tmp_path / 'source.csv'
    fake_source(source, '2026-11-06T00:00Z', '2026-11-09T15:00Z')
    frame = pd.read_csv(source)
    frame.loc[0, 'utc_offset (hrs)'] = -6
    frame.to_csv(source, index=False)
    archive = load_archive(Path('data/precipitation/twin_falls.json'))
    with pytest.raises(ValueError, match='offset'):
        extend_archive(archive, source)
    frame['utc_offset (hrs)'] = -7
    frame.drop(index=3).to_csv(source, index=False)
    with pytest.raises(ValueError, match='continuous ordered'):
        extend_archive(archive, source)


def test_independent_reference_covers_bootstrap_boundary_month(tmp_path):
    source = tmp_path / 'source.csv'
    fake_source(source, '2022-12-01T00:00Z', '2023-02-01T15:00Z')
    archive = extend_archive(load_archive(Path('data/precipitation/twin_falls.json')), source)
    references = job.audit_months(source, archive)
    assert references['2022-12'] == {'hours': 744, 'total_mm': 372.}
    assert references['2023-01'] == {'hours': 744, 'total_mm': 372.}
    for row in archive['days']:
        if row[0] == '2023-01-02':
            row[1] = 0.
    with pytest.raises(ValueError, match='independent|Independent'):
        job.audit_months(source, archive)


def test_another_publisher_holding_lock_prevents_a_second_run(deployment):
    import fcntl
    root, previous, _ = deployment
    with (root / 'publish.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(BlockingIOError):
            run(deployment)
    assert (root / 'current').resolve() == previous
