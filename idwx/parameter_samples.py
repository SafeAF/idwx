"""Original-unit archive samples, without inferred intervals or aggregation.

The sampling cadence describes timestamps, not a parameter's accumulation period.
Finite values outside screening bounds remain supplied values with quality flags.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import tempfile
import uuid
from collections import OrderedDict
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from idwx.climate_audit import audit_file, digest, load_sources
from idwx.climate_metrics import METRICS

VERSION = 'idwx.parameter-samples.1'
METHOD = 'source_samples_unchanged_v1'
SID = r'[a-z0-9]+(?:_[a-z0-9]+)*'
CHUNK_PATH = rf'hourly/{SID}/\d{{4}}-\d{{2}}\.json'
SHA = re.compile(r'[a-f0-9]{64}')
# Reuse semantic proofs only within this process, after hashing the exact bytes
# again and checking the same expected chunk identity. Encoded summaries cannot
# be mutated by callers; the bound covers the full archive without retaining rows.
_MONTH_PROOFS: OrderedDict[tuple, bytes] = OrderedDict()
_MAX_MONTH_PROOFS = 16384


def encoded(value) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False, ensure_ascii=False) + '\n').encode()


def supplied_metrics() -> list[dict]:
    return [{'id': m.id, 'label': m.label, 'source_column': m.column, 'unit': m.source_unit,
             'quantity': m.quantity, 'layer': m.layer, 'screening_minimum': m.minimum,
             'screening_maximum': m.maximum, 'value_interval': None, 'reducer': 'none'} for m in METRICS.values()]


def read_json(path: Path) -> dict:
    return _decode_json(path.read_bytes())


def _decode_json(content: bytes) -> dict:
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError('Duplicate parameter JSON field')
            result[key] = value
        return result
    def invalid(value):
        raise ValueError('Nonfinite parameter JSON value')
    value = json.loads(content.decode('utf-8'), object_pairs_hook=pairs, parse_constant=invalid)
    _require(isinstance(value, dict), 'Invalid parameter JSON document')
    return value


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def utc(value: str) -> datetime:
    _require(isinstance(value, str) and re.fullmatch(r'\d{4}-\d{2}-\d{2}T\d{2}:00:00Z', value), 'Invalid parameter UTC timestamp')
    return datetime.fromisoformat(value.replace('Z', '+00:00'))


def _empty_stats() -> dict:
    return {key: {'provided': 0, 'missing': 0, 'outside_screening_range': 0, 'minimum': None, 'maximum': None} for key in METRICS}


def _statistics(values: np.ndarray) -> dict:
    result = _empty_stats()
    for index, (key, metric) in enumerate(METRICS.items()):
        numbers = values[:, index]
        finite = np.isfinite(numbers)
        good = numbers[finite]
        result[key] = {'provided': int(finite.sum()), 'missing': int((~finite).sum()),
                       'outside_screening_range': int(((good < metric.minimum) | (good > metric.maximum)).sum()),
                       'minimum': float(good.min()) if len(good) else None,
                       'maximum': float(good.max()) if len(good) else None}
    return result


def _merge_stats(total: dict, part: dict) -> None:
    for key, stats in part.items():
        target = total[key]
        for field in ('provided', 'missing', 'outside_screening_range'):
            target[field] += stats[field]
        if stats['minimum'] is not None:
            target['minimum'] = stats['minimum'] if target['minimum'] is None else min(target['minimum'], stats['minimum'])
            target['maximum'] = stats['maximum'] if target['maximum'] is None else max(target['maximum'], stats['maximum'])


def validate_month(doc: dict, entry: dict, station: dict, kind: str) -> dict:
    _require(set(doc) == {'version', 'data_kind', 'station_id', 'month', 'start_utc', 'sampling_interval_seconds', 'metric_ids', 'values'}, 'Invalid parameter chunk fields')
    _require(doc['version'] == VERSION and doc['data_kind'] == kind and doc['station_id'] == station['station_id']
             and doc['month'] == entry['month'] and doc['start_utc'] == entry['start_utc']
             and type(doc['sampling_interval_seconds']) is int and doc['sampling_interval_seconds'] == 3600
             and doc['metric_ids'] == list(METRICS), 'Parameter chunk identity mismatch')
    values = doc['values']
    _require(isinstance(values, list) and len(values) == entry['count'] and 0 < len(values) <= 744, 'Invalid parameter sample count')
    _require(all(isinstance(row, list) and len(row) == len(METRICS)
                 and all(v is None or type(v) in (int, float) and math.isfinite(v) for v in row) for row in values), 'Invalid parameter sample value')
    start = utc(doc['start_utc'])
    end = start + timedelta(hours=len(values) - 1)
    _require(start.strftime('%Y-%m') == doc['month'] == end.strftime('%Y-%m'), 'Parameter chunk crosses UTC month')
    return _statistics(np.asarray(values, dtype=float))


def validate_snapshot(root: Path, *, allow_demo=False) -> dict:
    root = root.resolve(strict=True)
    catalog = read_json(root / 'catalog.json')
    _require(set(catalog) == {'version', 'method', 'data_kind', 'sampling_interval_seconds', 'timestamp_basis', 'metrics', 'stations', 'snapshot_id'}, 'Invalid parameter catalog fields')
    _require(catalog['version'] == VERSION and catalog['method'] == METHOD and encoded(catalog['metrics']) == encoded(supplied_metrics())
             and catalog['sampling_interval_seconds'] == 3600 and catalog['timestamp_basis'] == 'UTC', 'Unsupported provided-value contract')
    kind = catalog['data_kind']
    _require(kind in ('historical_reanalysis', 'synthetic_demo') and (allow_demo or kind != 'synthetic_demo'), 'Synthetic parameter samples require explicit admission')
    expected_id = hashlib.sha256(encoded({k: v for k, v in catalog.items() if k != 'snapshot_id'})).hexdigest()
    _require(catalog['snapshot_id'] == expected_id, 'Parameter snapshot identity mismatch')
    stations = catalog['stations']
    _require(isinstance(stations, list) and stations, 'No parameter locations')
    ids, paths = set(), {'catalog.json'}
    for station in stations:
        _require(isinstance(station, dict) and set(station) == {'station_id', 'name', 'latitude', 'longitude', 'reporting_utc_offset_hours', 'source_sha256', 'first_utc', 'last_utc', 'count', 'quality', 'months'}, 'Invalid parameter location fields')
        sid = station['station_id']
        _require(isinstance(sid, str) and re.fullmatch(SID, sid) and sid not in ids, 'Duplicate/invalid parameter location')
        ids.add(sid)
        _require(isinstance(station['name'], str) and 0 < len(station['name']) <= 120
                 and type(station['latitude']) in (int, float) and -90 <= station['latitude'] <= 90
                 and type(station['longitude']) in (int, float) and -180 <= station['longitude'] <= 180
                 and type(station['reporting_utc_offset_hours']) is int and station['reporting_utc_offset_hours'] in (-7, -8)
                 and isinstance(station['source_sha256'], str) and SHA.fullmatch(station['source_sha256']), 'Invalid parameter source metadata')
        _require(isinstance(station['months'], list) and station['months'], 'Missing parameter month index')
        total, count, next_stamp = _empty_stats(), 0, utc(station['first_utc'])
        for entry in station['months']:
            _require(isinstance(entry, dict) and set(entry) == {'month', 'path', 'sha256', 'start_utc', 'count'}, 'Invalid parameter month index fields')
            path = entry['path']
            _require(isinstance(entry['month'], str) and re.fullmatch(r'\d{4}-\d{2}', entry['month'])
                     and path == f'hourly/{sid}/{entry["month"]}.json' and path not in paths
                     and utc(entry['start_utc']) == next_stamp
                     and type(entry['count']) is int and 0 < entry['count'] <= 744, 'Parameter month continuity mismatch')
            paths.add(path)
            file = root / path
            _require(file.is_file() and not file.is_symlink() and file.resolve().is_relative_to(root), 'Unsafe/missing parameter chunk')
            content = file.read_bytes()
            _require(isinstance(entry['sha256'], str) and SHA.fullmatch(entry['sha256'])
                     and hashlib.sha256(content).hexdigest() == entry['sha256'], 'Parameter chunk checksum mismatch')
            proof_key = (entry['sha256'], sid, entry['month'], entry['start_utc'], entry['count'], kind)
            proof = _MONTH_PROOFS.pop(proof_key, None)
            if proof is None:
                proof = encoded(validate_month(_decode_json(content), entry, station, kind))
            _MONTH_PROOFS[proof_key] = proof
            if len(_MONTH_PROOFS) > _MAX_MONTH_PROOFS:
                _MONTH_PROOFS.popitem(last=False)
            _merge_stats(total, json.loads(proof))
            count += entry['count']; next_stamp += timedelta(hours=entry['count'])
        _require(type(station['count']) is int and station['count'] == count
                 and utc(station['last_utc']) + timedelta(hours=1) == next_stamp, 'Parameter source coverage mismatch')
        _require(encoded(station['quality']) == encoded(total), 'Parameter quality summary mismatch')
    actual = {p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file()}
    _require(actual == paths and not any(p.is_symlink() for p in root.rglob('*')), 'Unexpected/private parameter snapshot contents')
    return catalog



def _write_month(staged, station, month, start, pieces, data_kind):
    if not pieces:
        return
    values = np.concatenate(pieces)
    doc = {'version': VERSION, 'data_kind': data_kind, 'station_id': station['station_id'],
           'month': month, 'start_utc': start, 'sampling_interval_seconds': 3600,
           'metric_ids': list(METRICS), 'values': [[None if math.isnan(v) else float(v) for v in row] for row in values]}
    content = encoded(doc)
    relative = f'hourly/{station["station_id"]}/{month}.json'
    destination = staged / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(content)
    station['months'].append({'month': month, 'path': relative, 'sha256': hashlib.sha256(content).hexdigest(),
                              'start_utc': start, 'count': len(values)})
    station['count'] += len(values)
    _merge_stats(station['quality'], _statistics(values))


def build_snapshot(manifest_path: Path, source_root: Path, output: Path, *, station_ids: list[str] | None = None,
                   data_kind='historical_reanalysis', chunksize=50000) -> dict:
    """Stage a complete private sample snapshot; source bytes/units stay unchanged."""
    manifest_digest = digest(manifest_path)
    manifest = load_sources(manifest_path)
    source_root = source_root.expanduser().resolve(strict=True)
    output = output.resolve()
    _require(not output.is_relative_to(source_root) and not source_root.is_relative_to(output), 'Snapshot output must be separate from raw inputs')
    _require(data_kind in ('historical_reanalysis', 'synthetic_demo'), 'Unsupported parameter data kind')
    sources = manifest['sources']
    if station_ids is not None:
        _require(station_ids and len(station_ids) == len(set(station_ids)) and set(station_ids) <= {s['station_id'] for s in sources}, 'Unknown/duplicate parameter station selection')
        sources = [s for s in sources if s['station_id'] in station_ids]
    output.mkdir(parents=True, exist_ok=True)
    start, end = pd.Timestamp(manifest['start_utc']), pd.Timestamp(manifest['end_utc'])
    with tempfile.TemporaryDirectory(prefix='.pending-parameters-', dir=output) as temporary:
        staged = Path(temporary) / 'snapshot'; staged.mkdir()
        catalog = {'version': VERSION, 'method': METHOD, 'data_kind': data_kind, 'sampling_interval_seconds': 3600,
                   'timestamp_basis': 'UTC', 'metrics': supplied_metrics(), 'stations': []}
        for source in sources:
            path = source_root / source['file']
            _require(path.is_file() and not path.is_symlink(), 'Missing or symlink parameter source')
            checked = audit_file(path, source, start, end, chunksize=chunksize)
            _require(checked['complete'] and not checked['issues'], f'Parameter source identity/continuity check failed: {source["station_id"]}')
            before = path.stat()
            station = {k: source[k] for k in ('station_id', 'name', 'latitude', 'longitude', 'reporting_utc_offset_hours')}
            station.update(source_sha256=source['sha256'], first_utc=start.strftime('%Y-%m-%dT%H:00:00Z'),
                           last_utc=end.strftime('%Y-%m-%dT%H:00:00Z'), count=0, quality=_empty_stats(), months=[])
            pending_month, pending_start, pieces = None, None, []
            for frame in pd.read_csv(path, chunksize=chunksize, low_memory=False):
                stamps = pd.to_datetime(frame.iloc[:, 0], format='%Y-%m-%d %H:%M:%S', utc=True, errors='raise')
                months = stamps.dt.strftime('%Y-%m')
                numeric = frame[[m.column for m in METRICS.values()]].apply(pd.to_numeric, errors='raise').to_numpy(dtype=float)
                _require(not np.isinf(numeric).any(), 'Nonfinite provided value cannot be published unchanged')
                for month in months.drop_duplicates():
                    subset = months.eq(month).to_numpy()
                    if pending_month != month:
                        _write_month(staged, station, pending_month, pending_start, pieces, data_kind); pending_month = month; pending_start = stamps[months.eq(month)].iloc[0].strftime('%Y-%m-%dT%H:00:00Z'); pieces = []
                    pieces.append(numeric[subset])
            _write_month(staged, station, pending_month, pending_start, pieces, data_kind)
            after = path.stat()
            _require((before.st_size, before.st_mtime_ns, before.st_ino) == (after.st_size, after.st_mtime_ns, after.st_ino)
                     and digest(path) == source['sha256'], 'Parameter source changed while copying samples')
            catalog['stations'].append(station)
        _require(digest(manifest_path) == manifest_digest, 'Parameter source manifest changed during import')
        catalog['snapshot_id'] = hashlib.sha256(encoded(catalog)).hexdigest()
        (staged / 'catalog.json').write_bytes(encoded(catalog))
        validate_snapshot(staged, allow_demo=data_kind == 'synthetic_demo')
        destination = output / catalog['snapshot_id']
        if destination.exists():
            _require(validate_snapshot(destination, allow_demo=data_kind == 'synthetic_demo') == catalog, 'Existing immutable parameter snapshot differs')
        else:
            os.replace(staged, destination)
        link = output / 'current'
        _require(not link.exists() or link.is_symlink(), 'Preserve unexpected parameter current path')
        temporary_link = output / ('.current-' + uuid.uuid4().hex)
        try:
            temporary_link.symlink_to(destination.name); os.replace(temporary_link, link)
        finally:
            temporary_link.unlink(missing_ok=True)
    return {'snapshot_id': catalog['snapshot_id'], 'path': str(destination), 'stations': len(sources), 'parameters': len(METRICS),
            'samples': sum(s['count'] for s in catalog['stations'])}


def copy_chunks(snapshot: Path, target: Path, catalog: dict, reuse: Path | None = None) -> None:
    for station in catalog['stations']:
        for entry in station['months']:
            source = snapshot / entry['path']; destination = target / entry['path']
            destination.parent.mkdir(parents=True, exist_ok=True)
            existing = reuse / entry['path'] if reuse else None
            if (existing and existing.is_file() and not existing.is_symlink()
                    and existing.resolve().is_relative_to(reuse.resolve()) and digest(existing) == entry['sha256']):
                try:
                    os.link(existing, destination)
                except OSError:
                    shutil.copyfile(source, destination)
            else:
                shutil.copyfile(source, destination)
            _require(digest(destination) == entry['sha256'], 'Parameter snapshot changed during publication')
