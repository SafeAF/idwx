"""Reviewed hourly precipitation reduction, separate from temperature coverage."""
from __future__ import annotations

import ast
import hashlib
import json
import math
import sqlite3
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

PRECIP_FIELDS = {"precip_mm", "precip_valid_hours", "precip_expected_hours", "precip_status"}
METHOD = "ERA5 hourly increments ending at timestamp; sum over source fixed UTC-07 reporting day; all 24 hours required."


def _bytes(value) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


def reduce_hours(timestamps, values, offset_hours=-7) -> list[list]:
    stamps = pd.DatetimeIndex(pd.to_datetime(timestamps, utc=True, errors="raise"))
    numeric = pd.to_numeric(pd.Series(values), errors="coerce")
    if stamps.has_duplicates or not stamps.is_monotonic_increasing or not stamps.equals(stamps.floor("h")):
        raise ValueError("Precipitation requires unique ordered whole-hour UTC timestamps")
    if not len(stamps) or not -23 <= offset_hours <= 23:
        raise ValueError("Invalid precipitation timestamps/reporting offset")
    if len(numeric) != len(stamps):
        raise ValueError("Precipitation value/timestamp count mismatch")
    # An accumulation ending at local midnight belongs to the preceding day.
    local = (stamps - pd.Timedelta(hours=1) + pd.Timedelta(hours=offset_hours)).strftime("%Y-%m-%d")
    grouped = pd.DataFrame({"date": local, "value": numeric}).groupby("date", sort=True)
    result = []
    for day, group in grouped:
        values = [float(v) for v in group.value if pd.notna(v) and math.isfinite(v) and v >= 0]
        total = 0.0
        for value in values:
            total += value
        result.append([day, total if len(values) == 24 else None, len(values), 24])
    return result


def build_archive(source: Path, *, expected_sha256: str) -> dict:
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    if digest != expected_sha256:
        raise ValueError("Hourly source differs from the reviewed database source fingerprint")
    frame = pd.read_csv(source, usecols=[0, 1, 2, 4, 19])
    required = {"coordinates (lat,lon)", "model (name)", "utc_offset (hrs)", "total_precipitation (mm of water equivalent)"}
    if not required.issubset(frame.columns) or not frame['model (name)'].eq('era5').all() or not frame['utc_offset (hrs)'].eq(-7).all():
        raise ValueError("Unsupported precipitation source units/model/reporting-day convention")
    if set(frame['coordinates (lat,lon)'].astype(str)) != {'(42.4634, -114.4975)'}:
        raise ValueError("Precipitation source is not the allowlisted Twin Falls coordinate")
    stamps = pd.to_datetime(frame.iloc[:, 0], utc=True)
    if not pd.DatetimeIndex(stamps).equals(pd.date_range(stamps.min(), stamps.max(), freq='h')):
        raise ValueError("Reviewed precipitation source has missing hourly timestamps")
    days = reduce_hours(stamps, frame['total_precipitation (mm of water equivalent)'])
    result = {"version": 1, "station_id": "twin_falls", "source_sha256": digest, "data_kind": "historical_reanalysis",
              "reporting_utc_offset_hours": -7, "method": METHOD, "days": days}
    return {**result, "content_sha256": hashlib.sha256(_bytes(result)).hexdigest()}


def load_archive(path: Path) -> dict:
    archive = json.loads(path.read_text())
    fields = {'version', 'station_id', 'source_sha256', 'data_kind', 'reporting_utc_offset_hours', 'method', 'days', 'content_sha256'}
    if archive.get('version') == 2:
        fields.add('refresh_sources')
    if set(archive) != fields or archive['version'] not in (1, 2) or archive['station_id'] != 'twin_falls' or archive['data_kind'] != 'historical_reanalysis' or archive['reporting_utc_offset_hours'] != -7 or archive['method'] != METHOD:
        raise ValueError('Unsupported reviewed precipitation archive')
    expected = hashlib.sha256(_bytes({k: v for k, v in archive.items() if k != 'content_sha256'})).hexdigest()
    if archive['content_sha256'] != expected:
        raise ValueError('Precipitation archive content digest mismatch')
    if archive['version'] == 2:
        sources = archive['refresh_sources']
        if not isinstance(sources, list) or not sources:
            raise ValueError('Missing precipitation refresh provenance')
        for source in sources:
            if set(source) != {'sha256', 'start_utc', 'end_utc'}:
                raise ValueError('Invalid precipitation refresh provenance')
            digest = source['sha256']
            if not isinstance(digest, str) or len(digest) != 64 or any(c not in '0123456789abcdef' for c in digest):
                raise ValueError('Invalid precipitation refresh digest')
            start, end = pd.Timestamp(source['start_utc']), pd.Timestamp(source['end_utc'])
            if start.tzinfo is None or end.tzinfo is None or start > end:
                raise ValueError('Invalid precipitation refresh bounds')
    if not isinstance(archive['days'], list) or not archive['days']:
        raise ValueError('Empty precipitation archive')
    previous = None
    for row in archive['days']:
        if not isinstance(row, list) or len(row) != 4:
            raise ValueError('Invalid precipitation archive row')
        day, value, valid, expected = row
        current = date.fromisoformat(day)
        if previous is not None and current != previous + timedelta(days=1):
            raise ValueError('Precipitation archive calendar gap')
        validate_precip({'precip_mm': value, 'precip_valid_hours': valid, 'precip_expected_hours': expected,
                         'precip_status': 'complete' if valid == expected else 'incomplete'})
        previous = current
    return archive


def extend_archive(archive: dict, source: Path) -> dict:
    """Audit a continuous Oikolab ERA5 fragment and extend reviewed daily amounts."""
    frame = pd.read_csv(source)
    required = {'timestamp', 'model (name)', 'coordinates (lat,lon)', 'utc_offset (hrs)',
                'total_precipitation (mm of water equivalent)'}
    if not required.issubset(frame.columns) or not frame['model (name)'].eq('era5').all() or not frame['utc_offset (hrs)'].eq(-7).all():
        raise ValueError('Unsupported precipitation refresh model/units/reporting offset')
    coordinates = {str(value) for value in frame['coordinates (lat,lon)']}
    if not coordinates or any(tuple(ast.literal_eval(value)) != (42.4634, -114.4975) for value in coordinates):
        raise ValueError('Precipitation refresh coordinate mismatch')
    stamps = pd.DatetimeIndex(pd.to_datetime(frame['timestamp'], utc=True))
    if not len(stamps) or not stamps.equals(pd.date_range(stamps.min(), stamps.max(), freq='h')):
        raise ValueError('Precipitation refresh must have continuous ordered hours')
    amounts = pd.to_numeric(frame['total_precipitation (mm of water equivalent)'], errors='coerce')
    if any(not math.isfinite(v) or v < 0 for v in amounts):
        raise ValueError('Invalid precipitation refresh amounts')
    rows = {row[0]: row for row in archive['days']}
    for row in reduce_hours(stamps, amounts):
        if row[0] not in rows or row[2] >= rows[row[0]][2]:
            rows[row[0]] = row
    provenance = {'sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
                  'start_utc': stamps[0].isoformat(), 'end_utc': stamps[-1].isoformat()}
    sources = [*archive.get('refresh_sources', [])]
    if provenance not in sources:
        sources.append(provenance)
    result = {**{k: v for k, v in archive.items() if k != 'content_sha256'},
              'version': 2, 'days': [rows[key] for key in sorted(rows)], 'refresh_sources': sources}
    return {**result, 'content_sha256': hashlib.sha256(_bytes(result)).hexdigest()}


def validate_precip(row: dict) -> None:
    value, valid, expected, status = (row[k] for k in ('precip_mm', 'precip_valid_hours', 'precip_expected_hours', 'precip_status'))
    if status == 'not_reviewed':
        if value is not None or valid is not None or expected is not None:
            raise ValueError('Unreviewed precipitation exposes values')
        return
    if status not in {'complete', 'incomplete'} or type(valid) is not int or type(expected) is not int or expected != 24 or not 0 <= valid <= expected:
        raise ValueError('Invalid precipitation coverage')
    if status == 'complete':
        if valid != expected or type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            raise ValueError('Invalid complete precipitation total')
    elif valid == expected or value is not None:
        raise ValueError('Incomplete precipitation exposes a total')


def attach_precipitation(daily: dict, cfg, station_id: str, *, demo=False) -> dict:
    archive = None
    candidate = cfg.precipitation_archives.get(station_id)
    if not demo and candidate is not None:
        reviewed = load_archive(candidate)
        if reviewed['station_id'] != station_id:
            raise ValueError('Reviewed precipitation station does not match configuration')
        if cfg.database_path is not None:
            with sqlite3.connect(cfg.database_path.resolve().as_uri() + '?mode=ro', uri=True) as connection:
                sources = {row[0] for row in connection.execute('SELECT sha256 FROM source_files WHERE station_id = ?', (station_id,))}
                if reviewed.get('refresh_sources'):
                    refreshes = {(r[0], r[1], r[2]) for r in connection.execute(
                        'SELECT sha256, requested_start_utc, requested_end_utc FROM source_refreshes WHERE station_id = ?', (station_id,))}
                    if any((r['sha256'], r['start_utc'], r['end_utc']) not in refreshes for r in reviewed['refresh_sources']):
                        raise ValueError('Precipitation refresh provenance does not match database')
            if reviewed['source_sha256'] in sources:
                archive = reviewed
    mapping = {r[0]: r for r in archive['days']} if archive else {}
    complete_dates = []
    for days in daily.values():
        for row in days:
            item = mapping.get(row['date'])
            if demo:
                amount = round((int(row['date'][-2:]) % 7) * .4, 2)
                item = [row['date'], amount, 24, 24]
                if row['date'].endswith('06-01') and row['date'].startswith('2021'):
                    item = [row['date'], None, 23, 24]
            row.update(precip_mm=None, precip_valid_hours=None, precip_expected_hours=None, precip_status='not_reviewed')
            if item:
                row.update(precip_mm=item[1], precip_valid_hours=item[2], precip_expected_hours=item[3],
                           precip_status='complete' if item[2] == item[3] else 'incomplete')
                if row['precip_status'] == 'complete':
                    complete_dates.append(row['date'])
    return {'product': 'total_precipitation_water_equivalent', 'method': METHOD,
            'reporting_utc_offset_hours': -7, 'first_complete_date': min(complete_dates) if complete_dates else None,
            'last_complete_date': max(complete_dates) if complete_dates else None,
            'source_sha256': archive['source_sha256'] if archive else None,
            'review_status': 'synthetic_demo' if demo else 'reviewed_hourly_archive' if archive else 'not_reviewed'}


def validate_method(method: dict, daily: dict) -> None:
    fields = {'product', 'method', 'reporting_utc_offset_hours', 'first_complete_date', 'last_complete_date', 'source_sha256', 'review_status'}
    if set(method) != fields or method['method'] != METHOD or method['product'] != 'total_precipitation_water_equivalent' or method['reporting_utc_offset_hours'] != -7:
        raise ValueError('Invalid precipitation methodology')
    dates = [r['date'] for rows in daily.values() for r in rows if r['precip_status'] == 'complete']
    if method['first_complete_date'] != (min(dates) if dates else None) or method['last_complete_date'] != (max(dates) if dates else None):
        raise ValueError('Precipitation coverage/methodology mismatch')
    status = method['review_status']
    digest = method['source_sha256']
    if status not in {'synthetic_demo', 'reviewed_hourly_archive', 'not_reviewed'} or (status == 'not_reviewed' and dates):
        raise ValueError('Invalid precipitation review status')
    if status == 'reviewed_hourly_archive':
        if not isinstance(digest, str) or len(digest) != 64 or any(c not in '0123456789abcdef' for c in digest):
            raise ValueError('Invalid precipitation source digest')
    elif digest is not None:
        raise ValueError('Unexpected precipitation source digest')
