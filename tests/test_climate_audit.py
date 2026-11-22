"""Admission audit rejects ambiguous identities and records full-file quality."""
import csv
import hashlib
import json
from pathlib import Path

import pytest
import yaml

from idwx.climate_audit import audit_archive, load_sources, write_report
from idwx.climate_metrics import METRICS, metric_catalog


@pytest.fixture
def archive(tmp_path):
    root = tmp_path / 'archive'
    root.mkdir()
    source = root / '42.4634--114.4975-test.csv'
    values = {m.column: (m.minimum + m.maximum) / 2 for m in METRICS.values()}
    header = ['timestamp', 'coordinates (lat,lon)', 'model (name)', 'model elevation (surface)', 'utc_offset (hrs)', *values]
    rows = [{'timestamp': f'2000-02-29 {h:02}:00:00', 'coordinates (lat,lon)': '(42.4634, -114.4975)',
             'model (name)': 'era5', 'model elevation (surface)': 1300, 'utc_offset (hrs)': -7, **values} for h in range(24)]
    manifest = tmp_path / 'sources.yml'
    def write(updated=rows, **overrides):
        with source.open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=header)
            writer.writeheader(); writer.writerows(updated)
        entry = {'station_id': 'twin_falls', 'file': source.name, 'latitude': 42.4634, 'longitude': -114.4975,
                     'name': 'Twin Falls', 'label_status': 'existing_reviewed_label', 'reporting_utc_offset_hours': -7,
                     'model': 'era5', 'sha256': hashlib.sha256(source.read_bytes()).hexdigest()}
        entry.update(overrides)
        manifest.write_text(yaml.safe_dump({'version': 1, 'adapter': 'oikolab_era5_archive_v1', 'timestamp_convention': 'naive_utc',
                                             'start_utc': '2000-02-29T00:00:00Z', 'end_utc': '2000-02-29T23:00:00Z', 'sources': [entry]}))
    write()
    return manifest, root, source, rows, write


def test_complete_full_file_audit_does_not_self_admit(archive):
    manifest, root, _, _, _ = archive
    report = audit_archive(manifest, root, chunksize=7)
    assert report['status'] == 'review_required' and not report['publication_ready']
    source = report['sources'][0]
    assert source['complete'] and source['rows'] == 24 and source['issues'] == []
    assert len(source['metrics']) == 27
    assert all(m['valid'] == 24 and m['admission'] == 'review_required' for m in source['metrics'].values())
    assert len(metric_catalog()['metrics']) == 27
    assert METRICS['snow_depth'].reducers == ('min', 'mean', 'max')
    assert 'mean' not in METRICS['wind_direction'].reducers
    assert 'sum' not in METRICS['surface_solar_radiation'].reducers
    assert METRICS['surface_pressure'].scale == .01


@pytest.mark.parametrize('mutation,issue', [
    ('duplicate', 'duplicate_or_reversed_steps'), ('gap', 'missing_hours'),
    ('coordinate', 'invalid_metadata_rows'), ('offset', 'invalid_metadata_rows'),
    ('model', 'invalid_metadata_rows'), ('elevation', 'invalid_metadata_rows'), ('timestamp', 'invalid_timestamps'),
    ('off_hour', 'off_hour_timestamps')])
def test_identity_continuity_and_chunk_boundary_failures(archive, mutation, issue):
    manifest, root, _, rows, write = archive
    if mutation == 'duplicate': rows[7]['timestamp'] = rows[6]['timestamp']
    elif mutation == 'gap': rows.pop(7)
    elif mutation == 'coordinate': rows[7]['coordinates (lat,lon)'] = '(42.0, -114.4975)'
    elif mutation == 'offset': rows[7]['utc_offset (hrs)'] = -8
    elif mutation == 'model': rows[7]['model (name)'] = 'gfs'
    elif mutation == 'elevation': rows[7]['model elevation (surface)'] = 1200
    elif mutation == 'timestamp': rows[7]['timestamp'] = 'bad'
    elif mutation == 'off_hour': rows[7]['timestamp'] = '2000-02-29 07:30:00'
    write(rows)
    report = audit_archive(manifest, root, chunksize=7)
    assert report['status'] == 'blocked'
    assert issue in report['sources'][0]['issues']
    assert all(m['admission'] == 'unavailable' for m in report['sources'][0]['metrics'].values())


def test_invalid_and_missing_are_distinct_never_clamped_or_zero_filled(archive):
    manifest, root, _, rows, write = archive
    for index, value in enumerate(['', 'nan', 'inf', -0.01, 1.01]): rows[index]['relative_humidity (0-1)'] = value
    write(rows)
    stats = audit_archive(manifest, root, chunksize=3)['sources'][0]['metrics']['relative_humidity']
    assert (stats['valid'], stats['missing'], stats['invalid']) == (19, 2, 3)
    assert stats['observed_minimum'] == -.01 and stats['observed_maximum'] == 1.01
    assert stats['admission'] == 'unavailable' and len(stats['invalid_examples']) == 3


def test_missing_unlisted_changed_and_duplicate_sources(archive):
    manifest, root, source, _, _ = archive
    (root / 'extra.csv').write_text('private')
    source.write_text(source.read_text() + '\n')
    report = audit_archive(manifest, root)
    assert report['status'] == 'blocked' and report['unlisted_files'] == ['extra.csv']
    assert 'source_fingerprint_mismatch' in report['sources'][0]['issues']
    source.unlink()
    assert audit_archive(manifest, root)['missing_files'] == [source.name]
    value = yaml.safe_load(manifest.read_text()); value['sources'].append(value['sources'][0]); manifest.write_text(yaml.safe_dump(value))
    with pytest.raises(ValueError, match='duplicate'): load_sources(manifest)


def test_source_changes_during_read_are_never_admitted(archive, monkeypatch):
    import idwx.climate_audit as audit
    manifest, root, source, _, _ = archive
    original = audit.digest
    calls = 0
    def changing(path):
        nonlocal calls
        if path == source:
            calls += 1
            if calls == 2: return 'f' * 64
        return original(path)
    monkeypatch.setattr(audit, 'digest', changing)
    assert 'source_changed_during_audit' in audit_archive(manifest, root)['sources'][0]['issues']


def test_report_write_is_private_and_atomic(archive, tmp_path, monkeypatch):
    import idwx.climate_audit as audit
    manifest, root, _, _, _ = archive
    report = audit_archive(manifest, root)
    with pytest.raises(ValueError, match='private'): write_report(report, root / 'report.json', root)
    site = tmp_path / 'site'; (site / 'data').mkdir(parents=True); (site / 'data/manifest.json').write_text('{}')
    with pytest.raises(ValueError, match='static publication'): write_report(report, site / 'report.json', root)
    destination = tmp_path / 'private/report.json'; write_report(report, destination, root)
    previous = destination.read_bytes()
    assert json.loads(previous)['private_report']
    def fail(*args): raise OSError('simulated replacement failure')
    monkeypatch.setattr(audit.os, 'replace', fail)
    with pytest.raises(OSError): write_report(report, destination, root)
    assert destination.read_bytes() == previous
    assert list(destination.parent.iterdir()) == [destination]


def test_repository_allowlist_accounts_for_exactly_21_unique_locations():
    manifest = load_sources(Path('configs/climate_sources.yml'))
    assert len(manifest['sources']) == 21
    assert sum(s['reporting_utc_offset_hours'] == -8 for s in manifest['sources']) == 3
    assert next(s for s in manifest['sources'] if s['station_id'] == 'twin_falls')['name'] == 'Twin Falls'
    assert all(s['name'] == f"{s['latitude']:.6f}, {s['longitude']:.6f}" for s in manifest['sources'] if s['station_id'] != 'twin_falls')


def test_changed_manifest_blocks_the_whole_audit(archive, monkeypatch):
    import idwx.climate_audit as audit
    manifest, root, _, _, _ = archive
    original = audit.digest
    calls = 0
    def changing(path):
        nonlocal calls
        if path == manifest:
            calls += 1
            if calls == 2: return 'f' * 64
        return original(path)
    monkeypatch.setattr(audit, 'digest', changing)
    report = audit_archive(manifest, root)
    assert report['status'] == 'blocked' and report['manifest_changed_during_audit']
    assert report['publication_ready'] is False
