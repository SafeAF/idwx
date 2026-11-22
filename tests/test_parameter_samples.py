"""Unchanged source values and units must survive import, publication and refresh."""
import csv
import hashlib
import shutil
from dataclasses import replace
from pathlib import Path

import pytest
import yaml

from idwx.climate_metrics import METRICS
from idwx.config import StationMeta, load_config
from idwx.parameter_samples import build_snapshot, encoded, read_json, validate_snapshot
from idwx.web_export import export_web_data
from idwx.web_verify import package_web, verify_web


@pytest.fixture
def samples(tmp_path):
    raw = tmp_path / 'raw'; raw.mkdir()
    source = raw / '42.4634--114.4975-test.csv'
    values = {m.column: (m.minimum + m.maximum) / 2 for m in METRICS.values()}
    values.update({'relative_humidity (0-1)': 1.72, 'surface_pressure (Pa)': 100000., 'surface_solar_radiation (W/m^2)': 250.5,
                   'snow_depth (mm of water equivalent)': 3.14})
    stamps = ['2000-02-29 23:00:00', '2000-03-01 00:00:00', '2000-03-01 01:00:00', '2000-03-01 02:00:00']
    rows = [{'timestamp': stamp, 'coordinates (lat,lon)': '(42.4634, -114.4975)', 'model (name)': 'era5',
             'model elevation (surface)': 1300., 'utc_offset (hrs)': -7, **values} for stamp in stamps]
    rows[2]['surface_pressure (Pa)'] = ''
    with source.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    manifest = tmp_path / 'sources.yml'
    manifest.write_text(yaml.safe_dump({'version': 1, 'adapter': 'oikolab_era5_archive_v1', 'timestamp_convention': 'naive_utc',
        'start_utc': '2000-02-29T23:00:00Z', 'end_utc': '2000-03-01T02:00:00Z', 'sources': [{
        'station_id': 'twin_falls', 'file': source.name, 'latitude': 42.4634, 'longitude': -114.4975, 'name': 'Twin Falls',
        'label_status': 'existing_reviewed_label', 'reporting_utc_offset_hours': -7, 'model': 'era5',
        'sha256': hashlib.sha256(source.read_bytes()).hexdigest()}]}))
    output = tmp_path / 'snapshots'
    def build(kind='historical_reanalysis'):
        return build_snapshot(manifest, raw, output, data_kind=kind, chunksize=2)
    return source, manifest, output, rows, build


def test_all_27_values_keep_source_units_and_unverified_intervals(samples):
    _, _, output, rows, build = samples
    receipt = build()
    catalog = validate_snapshot(Path(receipt['path']))
    assert receipt['parameters'] == 27 and receipt['samples'] == 4
    assert (output / 'current').resolve() == Path(receipt['path'])
    assert all(m['value_interval'] is None and m['reducer'] == 'none' for m in catalog['metrics'])
    metrics = {m['id']: m for m in catalog['metrics']}
    assert metrics['surface_pressure']['unit'] == 'Pa'
    assert metrics['surface_solar_radiation']['unit'] == 'W/m^2'
    assert metrics['relative_humidity']['unit'] == '0-1'
    flat = [row for entry in catalog['stations'][0]['months'] for row in read_json(Path(receipt['path']) / entry['path'])['values']]
    for provided, original in zip(flat, rows):
        for index, metric in enumerate(METRICS.values()):
            assert provided[index] == (None if original[metric.column] == '' else original[metric.column])
    quality = catalog['stations'][0]['quality']
    assert quality['relative_humidity']['outside_screening_range'] == 4
    assert quality['relative_humidity']['maximum'] == 1.72
    assert quality['surface_pressure']['missing'] == 1
    assert quality['surface_pressure']['provided'] == 3
    assert len(catalog['stations'][0]['months']) == 2
    assert build()['snapshot_id'] == receipt['snapshot_id']


def test_changed_source_preserves_previous_snapshot_and_pointer(samples):
    source, _, output, _, build = samples
    old = Path(build()['path'])
    source.write_text(source.read_text().replace('250.5', '251.5'))
    with pytest.raises(ValueError, match='identity/continuity'): build()
    assert (output / 'current').resolve() == old
    assert not list(output.glob('.pending-*'))


def mutate_chunk(root, mutate):
    catalog = read_json(root / 'catalog.json')
    entry = catalog['stations'][0]['months'][0]
    path = root / entry['path']; chunk = read_json(path); mutate(chunk); path.write_bytes(encoded(chunk))
    entry['sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
    catalog['snapshot_id'] = hashlib.sha256(encoded({k: v for k, v in catalog.items() if k != 'snapshot_id'})).hexdigest()
    (root / 'catalog.json').write_bytes(encoded(catalog))


def test_rehashed_false_quality_is_rejected(samples):
    _, _, _, _, build = samples
    root = Path(build()['path'])
    mutate_chunk(root, lambda chunk: chunk['values'][0].__setitem__(list(METRICS).index('relative_humidity'), .5))
    with pytest.raises(ValueError, match='quality summary'): validate_snapshot(root)


def test_boolean_values_and_unsafe_paths_are_rejected(samples):
    _, _, _, _, build = samples
    root = Path(build()['path'])
    mutate_chunk(root, lambda chunk: chunk['values'][0].__setitem__(0, True))
    with pytest.raises(ValueError, match='sample value'): validate_snapshot(root)


def test_repeated_immutable_chunks_reuse_proof_but_always_check_bytes(samples, tmp_path, monkeypatch):
    import idwx.parameter_samples as parameters
    root = Path(samples[-1]()['path'])
    parameters._MONTH_PROOFS.clear()
    original = parameters.validate_month
    calls = []
    def checked(*args):
        calls.append(args[1]['month'])
        return original(*args)
    monkeypatch.setattr(parameters, 'validate_month', checked)
    catalog = validate_snapshot(root)
    assert len(calls) == 2
    copied = tmp_path / 'copied'
    shutil.copytree(root, copied)
    assert validate_snapshot(copied) == catalog
    assert len(calls) == 2  # Same bytes and expected identity, across roots.
    chunk = copied / catalog['stations'][0]['months'][0]['path']
    chunk.write_bytes(chunk.read_bytes() + b' ')
    with pytest.raises(ValueError, match='checksum'): validate_snapshot(copied)
    assert len(calls) == 2


def test_cached_proof_does_not_admit_changed_expected_identity(samples):
    root = Path(samples[-1]()['path'])
    catalog = validate_snapshot(root)  # Warm the proof cache.
    catalog['stations'][0]['months'][0]['count'] = 2
    catalog['snapshot_id'] = hashlib.sha256(encoded({k: v for k, v in catalog.items() if k != 'snapshot_id'})).hexdigest()
    (root / 'catalog.json').write_bytes(encoded(catalog))
    with pytest.raises(ValueError, match='sample count'): validate_snapshot(root)


def test_parameter_json_retains_utf8_only_decoding(tmp_path):
    path = tmp_path / 'parameter.json'
    path.write_bytes('{"value":1}'.encode('utf-16'))
    with pytest.raises(UnicodeError): read_json(path)


def test_publication_and_zip_bind_the_snapshot_and_preserve_legacy_data(samples, tmp_path):
    _, _, _, _, build = samples
    snapshot = Path(build('synthetic_demo')['path'])
    cfg = replace(load_config('configs/idaho.yml'), backtest={'start_year': 2000, 'end_year': 2022})
    stations = [StationMeta('demo', 'Demo')]
    output = tmp_path / 'site'
    old = export_web_data(cfg, stations, output, demo=True)
    old_contents = (output / old['stations_url']).read_bytes()
    current = export_web_data(cfg, stations, output, demo=True, parameter_snapshot=snapshot)
    assert current['schema_version'] == '2.0.0'
    assert (output / old['stations_url']).read_bytes() == old_contents
    report = verify_web(output, allow_demo=True)
    assert report['retained_releases_verified'] == 1
    package = package_web(output, tmp_path / 'site.zip', allow_demo=True)
    assert package['release_id'] == current['release_id']
    # A failed future export cannot replace the current active manifest.
    saved = (output / 'data/manifest.json').read_bytes()
    mutate_chunk(snapshot, lambda chunk: chunk['values'][0].__setitem__(0, None))
    with pytest.raises(ValueError): export_web_data(cfg, stations, output, demo=True, parameter_snapshot=snapshot)
    assert (output / 'data/manifest.json').read_bytes() == saved


def test_explicit_demo_snapshot_cannot_enter_a_real_publication(samples, tmp_path):
    _, _, _, _, build = samples
    snapshot = Path(build('synthetic_demo')['path'])
    cfg = load_config('configs/idaho.yml')
    with pytest.raises(ValueError, match='Synthetic parameter'): export_web_data(cfg, [StationMeta('twin_falls', 'Twin Falls')], tmp_path / 'site', parameter_snapshot=snapshot)
