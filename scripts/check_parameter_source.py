"""Independent csv/datetime fidelity check for original-unit parameter samples."""
from __future__ import annotations

import argparse
import csv
import json
import math
from datetime import datetime, timezone
from pathlib import Path

from idwx.climate_audit import digest, load_sources
from idwx.parameter_samples import read_json


def check(snapshot: Path, source_root: Path, manifest: Path, stations: list[str], months: list[str]) -> dict:
    catalog = read_json(snapshot.resolve() / 'catalog.json')
    configured = {s['station_id']: s for s in load_sources(manifest)['sources']}
    results = []
    for sid in stations:
        station = next(s for s in catalog['stations'] if s['station_id'] == sid)
        source = source_root.expanduser() / configured[sid]['file']
        if digest(source) != station['source_sha256']:
            raise ValueError('Source fingerprint differs from the snapshot')
        expected = {}
        for entry in station['months']:
            if entry['month'] not in months:
                continue
            doc = read_json(snapshot.resolve() / entry['path'])
            start = datetime.fromisoformat(doc['start_utc'].replace('Z', '+00:00')).timestamp()
            for i, row in enumerate(doc['values']):
                expected[start + i * 3600] = row
        counts = {month: 0 for month in months}
        values_checked = 0
        with source.open(newline='') as stream:
            reader = csv.DictReader(stream)
            for row in reader:
                stamp = datetime.strptime(row[reader.fieldnames[0]], '%Y-%m-%d %H:%M:%S').replace(tzinfo=timezone.utc)
                month = stamp.strftime('%Y-%m')
                if month not in months:
                    continue
                actual = expected.pop(stamp.timestamp())
                for index, metric in enumerate(catalog['metrics']):
                    # Derive the unit directly from the original CSV header.
                    column = metric['source_column']
                    source_unit = column.rsplit(' (', 1)[1][:-1] if ' (' in column else ''
                    if metric['unit'] != source_unit:
                        raise ValueError('Provided unit changed from the source header')
                    text = row[column].strip()
                    value = None if not text or text.lower() == 'nan' else float(text)
                    if value is not None and not math.isfinite(value):
                        raise ValueError('Unrepresentable source numeric value')
                    if value != actual[index]:
                        raise ValueError(f'Provided value mismatch: {sid}, {month}, {metric["id"]}')
                    values_checked += 1
                counts[month] += 1
        if expected or not all(counts.values()):
            raise ValueError('Incomplete independent parameter comparison')
        results.append({'station_id': sid, 'utc_offset_hours': station['reporting_utc_offset_hours'],
                        'months': counts, 'values_compared': values_checked})
    return {'status': 'source_values_and_units_match', 'snapshot_id': catalog['snapshot_id'], 'references': results}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snapshot', type=Path, required=True)
    parser.add_argument('--source-root', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, default=Path('configs/climate_sources.yml'))
    parser.add_argument('--station', action='append', required=True)
    parser.add_argument('--month', action='append', required=True)
    args = parser.parse_args()
    print(json.dumps(check(args.snapshot, args.source_root, args.manifest, args.station, args.month), sort_keys=True))


if __name__ == '__main__':
    main()
