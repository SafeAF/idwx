"""Bounded full-file audit of an explicit archive allowlist; never publishes data."""
from __future__ import annotations

import ast
import csv
import hashlib
import json
import os
import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from idwx.climate_metrics import METRICS, metric_catalog

META = ['coordinates (lat,lon)', 'model (name)', 'model elevation (surface)', 'utc_offset (hrs)']
SOURCE_KEYS = {'station_id', 'file', 'latitude', 'longitude', 'name', 'label_status',
               'reporting_utc_offset_hours', 'model', 'sha256'}


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def load_sources(path: Path) -> dict:
    manifest = yaml.safe_load(path.read_text())
    if (not isinstance(manifest, dict) or set(manifest) != {'version', 'adapter', 'timestamp_convention', 'start_utc', 'end_utc', 'sources'}
            or manifest['version'] != 1 or manifest['adapter'] != 'oikolab_era5_archive_v1'
            or manifest['timestamp_convention'] != 'naive_utc' or not isinstance(manifest['sources'], list) or not manifest['sources']):
        raise ValueError('Unsupported climate source manifest')
    start, end = pd.Timestamp(manifest['start_utc']), pd.Timestamp(manifest['end_utc'])
    if start.tzinfo is None or end.tzinfo is None or start.utcoffset().total_seconds() or end.utcoffset().total_seconds() or start > end or start != start.floor('h') or end != end.floor('h'):
        raise ValueError('Invalid source timestamp bounds')
    ids, names, coords = set(), set(), set()
    for source in manifest['sources']:
        if not isinstance(source, dict) or set(source) != SOURCE_KEYS:
            raise ValueError('Invalid climate source fields')
        sid, name = source['station_id'], source['file']
        coord = (source['latitude'], source['longitude'])
        if (not isinstance(sid, str) or not re.fullmatch(r'[a-z0-9]+(?:_[a-z0-9]+)*', sid) or sid in ids
                or not isinstance(name, str) or Path(name).name != name or not name.endswith('.csv') or name in names
                or not all(type(v) in (int, float) and np.isfinite(v) for v in coord) or coord in coords
                or not -90 <= coord[0] <= 90 or not -180 <= coord[1] <= 180
                or source['model'] != 'era5' or type(source['reporting_utc_offset_hours']) is not int
                or source['reporting_utc_offset_hours'] not in (-7, -8)
                or not isinstance(source['sha256'], str) or not re.fullmatch(r'[a-f0-9]{64}', source['sha256'])
                or not isinstance(source['name'], str) or not source['name'].strip()
                or source['label_status'] not in ('coordinate_label', 'existing_reviewed_label')):
            raise ValueError('Invalid or duplicate climate source identity')
        # Filename identity must agree as well as the embedded metadata.
        expected = re.fullmatch(r'([0-9.]+)--([0-9.]+)-.*\.csv', name)
        if not expected or (float(expected[1]), -float(expected[2])) != coord:
            raise ValueError('Source filename coordinate mismatch')
        if sid == 'twin_falls' and coord != (42.4634, -114.4975):
            raise ValueError('Twin Falls coordinate identity changed')
        ids.add(sid); names.add(name); coords.add(coord)
    return manifest


def audit_file(path: Path, source: dict, start: pd.Timestamp, end: pd.Timestamp, *, chunksize=50000) -> dict:
    result = {'station_id': source['station_id'], 'file': source['file'], 'sha256': None,
              'complete': False, 'issues': [], 'rows': 0, 'missing_hours': 0, 'duplicate_or_reversed_steps': 0,
              'invalid_timestamps': 0, 'invalid_metadata_rows': 0, 'off_hour_timestamps': 0,
              'first_utc': None, 'last_utc': None, 'metrics': {}}
    before = path.stat()
    expected_rows = int((end - start).total_seconds() // 3600) + 1
    for key in METRICS:
        result['metrics'][key] = {'valid': 0, 'missing': 0, 'invalid': 0, 'minimum': None, 'maximum': None,
                                  'observed_minimum': None, 'observed_maximum': None, 'invalid_examples': [],
                                  'admission': 'unavailable', 'reason': METRICS[key].reason}
    result['sha256'] = digest(path)
    if result['sha256'] != source['sha256']:
        result['issues'].append('source_fingerprint_mismatch')
    with path.open(newline='') as stream:
        header = next(csv.reader(stream), [])
    expected_columns = {m.column for m in METRICS.values()}
    if (len(header) != 32 or header[0] not in ('', 'timestamp') or len(set(header)) != len(header)
            or set(header[1:]) != set(META) | expected_columns):
        result['issues'].append('unsupported_or_duplicate_header')
        return result
    previous = None
    elevation_value = None
    for frame in pd.read_csv(path, chunksize=chunksize, low_memory=False):
        stamps = pd.to_datetime(frame.iloc[:, 0], format='%Y-%m-%d %H:%M:%S', utc=True, errors='coerce')
        result['rows'] += len(frame)
        result['invalid_timestamps'] += int(stamps.isna().sum())
        valid = stamps.dropna()
        if not valid.empty:
            result['first_utc'] = result['first_utc'] or valid.iloc[0].isoformat()
            result['last_utc'] = valid.iloc[-1].isoformat()
            differences = valid.diff().dt.total_seconds().dropna()
            if previous is not None:
                differences = pd.concat([pd.Series([(valid.iloc[0] - previous).total_seconds()]), differences])
            result['missing_hours'] += int(((differences[differences > 3600] / 3600) - 1).sum())
            result['duplicate_or_reversed_steps'] += int((differences <= 0).sum())
            result['off_hour_timestamps'] += int((valid != valid.dt.floor('h')).sum())
            if ((valid < start).any() or (valid > end).any()) and 'timestamp_outside_declared_range' not in result['issues']:
                result['issues'].append('timestamp_outside_declared_range')
            previous = valid.iloc[-1]
        coordinate_ok = {}
        for value in frame[META[0]].drop_duplicates():
            try:
                parsed = ast.literal_eval(str(value))
                coordinate_ok[str(value)] = tuple(parsed) == (source['latitude'], source['longitude'])
            except (ValueError, SyntaxError, TypeError):
                coordinate_ok[str(value)] = False
        elevation = pd.to_numeric(frame[META[2]], errors='coerce')
        if elevation_value is None and np.isfinite(elevation).any():
            elevation_value = float(elevation[np.isfinite(elevation)].iloc[0])
        metadata = (frame[META[0]].astype(str).map(coordinate_ok).fillna(False)
                    & frame[META[1]].eq(source['model'])
                    & pd.to_numeric(frame[META[3]], errors='coerce').eq(source['reporting_utc_offset_hours'])
                    & np.isfinite(elevation) & elevation.between(-500, 9000) & elevation.eq(elevation_value))
        result['invalid_metadata_rows'] += int((~metadata).sum())
        for key, metric in METRICS.items():
            raw = frame[metric.column]
            numeric = pd.to_numeric(raw, errors='coerce')
            missing = raw.isna()
            if not pd.api.types.is_numeric_dtype(raw):
                missing = missing | raw.astype(str).str.strip().eq('')
            good = np.isfinite(numeric) & numeric.between(metric.minimum, metric.maximum)
            stats = result['metrics'][key]
            stats['missing'] += int(missing.sum())
            stats['invalid'] += int((~good & ~missing).sum())
            stats['valid'] += int(good.sum())
            observed = numeric[np.isfinite(numeric)]
            if not observed.empty:
                lo, hi = float(observed.min()), float(observed.max())
                stats['observed_minimum'] = lo if stats['observed_minimum'] is None else min(stats['observed_minimum'], lo)
                stats['observed_maximum'] = hi if stats['observed_maximum'] is None else max(stats['observed_maximum'], hi)
            for index in frame.index[~good & ~missing][:max(0, 5 - len(stats['invalid_examples']))]:
                stats['invalid_examples'].append({'timestamp': str(frame.iloc[:, 0].loc[index]), 'value': str(raw.loc[index])})
            if good.any():
                lo, hi = float(numeric[good].min()), float(numeric[good].max())
                stats['minimum'] = lo if stats['minimum'] is None else min(stats['minimum'], lo)
                stats['maximum'] = hi if stats['maximum'] is None else max(stats['maximum'], hi)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns, before.st_ino) != (after.st_size, after.st_mtime_ns, after.st_ino) or digest(path) != result['sha256']:
        result['issues'].append('source_changed_during_audit')
    if result['rows'] != expected_rows or result['first_utc'] != start.isoformat() or result['last_utc'] != end.isoformat():
        result['issues'].append('declared_timestamp_span_or_count_mismatch')
    for key in ('invalid_timestamps', 'missing_hours', 'duplicate_or_reversed_steps', 'invalid_metadata_rows', 'off_hour_timestamps'):
        if result[key]:
            result['issues'].append(key)
    for stats in result['metrics'].values():
        if not result['issues'] and not stats['invalid'] and stats['valid']:
            stats['admission'] = 'review_required'
        elif stats['invalid']:
            stats['reason'] = 'Out-of-range or nonfinite source values require investigation; values were not clamped.'
    result['complete'] = True
    result['source_elevation_m'] = elevation_value
    return result


def audit_archive(manifest_path: Path, root: Path, *, chunksize=50000) -> dict:
    manifest_digest = digest(manifest_path)
    manifest = load_sources(manifest_path)
    root = root.expanduser().resolve(strict=True)
    expected = {s['file'] for s in manifest['sources']}
    actual = {p.relative_to(root).as_posix() for p in root.rglob('*.csv')}
    report = {'version': 1, 'private_report': True, 'publication_ready': False,
              'generated_at_utc': datetime.now(timezone.utc).isoformat(),
              'manifest_sha256': manifest_digest, 'manifest_changed_during_audit': False, 'catalog': metric_catalog(),
              'unlisted_files': sorted(actual - expected), 'missing_files': sorted(expected - actual),
              'sources': [], 'status': 'review_required',
              'limitations': ['Automated range checks do not establish provider provenance, licensing, timestamp semantics or independent numeric correctness.',
                             'Coordinate labels are intentional; no unverified place names are assigned.',
                             'No metric is admitted to publication by this audit. Existing Twin Falls products remain authoritative.']}
    start, end = pd.Timestamp(manifest['start_utc']), pd.Timestamp(manifest['end_utc'])
    for source in manifest['sources']:
        path = root / source['file']
        if not path.is_file() or path.is_symlink():
            report['sources'].append({'station_id': source['station_id'], 'complete': False, 'issues': ['missing_or_symlink_source']})
            continue
        try:
            report['sources'].append(audit_file(path, source, start, end, chunksize=chunksize))
        except (OSError, ValueError, pd.errors.ParserError) as exc:
            report['sources'].append({'station_id': source['station_id'], 'complete': False,
                                      'issues': ['read_failure'], 'error_type': type(exc).__name__})
    report['manifest_changed_during_audit'] = digest(manifest_path) != manifest_digest
    blocked = report['manifest_changed_during_audit'] or report['unlisted_files'] or report['missing_files'] or any(not s['complete'] or s['issues'] for s in report['sources'])
    report['status'] = 'blocked' if blocked else 'review_required'
    report['unavailable_metric_sources'] = sum(m['admission'] == 'unavailable' for s in report['sources'] for m in s.get('metrics', {}).values())
    return report


def write_report(report: dict, output: Path, source_root: Path) -> None:
    output = output.resolve()
    if output.is_relative_to(source_root.expanduser().resolve()) or output.suffix != '.json':
        raise ValueError('Audit output must be private JSON outside the source root')
    repository = Path(__file__).resolve().parents[1]
    if output.is_relative_to(repository / 'web') or output.is_relative_to(repository / 'data'):
        raise ValueError('Audit output cannot be under public assets or curated data')
    for parent in (output.parent, *output.parents):
        if (parent / 'data/manifest.json').exists() or (parent / 'explore.html').exists():
            raise ValueError('Audit output cannot be inside a static publication')
    content = (json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + '\n').encode()
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.climate-audit-', dir=output.parent) as stage:
        pending = Path(stage) / 'report.json'
        pending.write_bytes(content)
        os.replace(pending, output)
