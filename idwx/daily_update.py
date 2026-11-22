"""Locked, verified Oikolab batch refresh and static-site publication."""
from __future__ import annotations

import argparse
import calendar
import csv
import fcntl
import gzip
import hashlib
import json
import math
import os
import shutil
import sqlite3
import tempfile
import uuid
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from idwx.config import load_config, load_stations
from idwx.oikolab import era5_available_end, fetch_era5_hourly_csv
from idwx.precipitation import extend_archive, load_archive
from idwx.sqlite_store import refresh_sqlite_database
from idwx.web_export import export_web_data
from idwx.web_verify import verify_web


def _write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n')


def _link(target: Path, link: Path) -> None:
    temporary = link.with_name('.' + link.name + '-' + uuid.uuid4().hex)
    temporary.symlink_to(target)
    try:
        os.replace(temporary, link)
    finally:
        temporary.unlink(missing_ok=True)


def _target(link: Path, parent: Path) -> Path:
    if not link.is_symlink():
        raise ValueError(f'Expected deployment symlink: {link}')
    target = link.resolve(strict=True)
    if target.parent != parent.resolve() or not target.is_dir():
        raise ValueError('Deployment target escapes its expected directory')
    return target


def _coverage(database: Path) -> tuple[str, bool]:
    with sqlite3.connect(database.resolve().as_uri() + '?mode=ro', uri=True) as connection:
        if connection.execute('PRAGMA integrity_check').fetchone() != ('ok',):
            raise ValueError('Database integrity check failed')
        row = connection.execute("SELECT date, n_obs, quality_flags FROM daily_weather WHERE station_id='twin_falls' ORDER BY date DESC LIMIT 1").fetchone()
        if not row:
            raise ValueError('Missing Twin Falls weather data')
        return row[0], row[1] == 24 and row[2] == ''


def _precip_end(archive: dict) -> str:
    return max(row[0] for row in archive['days'] if row[2] == 24)


def audit_months(source: Path, archive: dict) -> dict:
    """Independent csv/datetime reference totals, separate from pandas reduction."""
    amounts, hours = {}, {}
    with source.open(newline='') as stream:
        for row in csv.DictReader(stream):
            stamp = datetime.fromisoformat(row['timestamp'].replace('Z', '+00:00'))
            month = (stamp.astimezone(timezone.utc) - timedelta(hours=8)).strftime('%Y-%m')
            amounts[month] = amounts.get(month, 0.) + float(row['total_precipitation (mm of water equivalent)'])
            hours[month] = hours.get(month, 0) + 1
    checked = {}
    for month, total in amounts.items():
        year, number = map(int, month.split('-'))
        if hours[month] != calendar.monthrange(year, number)[1] * 24:
            continue
        days = [r for r in archive['days'] if r[0].startswith(month)]
        if len(days) != hours[month] // 24 or any(r[1] is None for r in days) or not math.isclose(sum(r[1] for r in days), total, rel_tol=1e-12, abs_tol=1e-7):
            raise ValueError('Independent precipitation monthly reference mismatch')
        checked[month] = {'hours': hours[month], 'total_mm': round(total, 8)}
    return checked


def _public_permissions(root: Path) -> None:
    for path in [root, *root.rglob('*')]:
        if path.is_symlink():
            raise ValueError('Unexpected symlink in public build')
        path.chmod(0o755 if path.is_dir() else 0o644)


def check_http(expected_release: str) -> None:
    # Loopback HTTP routes to the existing weather vhost over trusted HTTPS.
    import subprocess

    payload = subprocess.check_output([
        '/usr/bin/curl', '--fail', '--silent', '--show-error', '--noproxy', '*',
        '--resolve', 'weather.kerrco.us:443:127.0.0.1', '--max-time', '20',
        'https://weather.kerrco.us/data/manifest.json',
    ], text=True)
    if json.loads(payload)['release_id'] != expected_release:
        raise ValueError('Served manifest does not match promoted release')


def update(config: Path, root: Path, *, api_key: str, source_revision: str,
           verify_only: bool = False, health_check=check_http) -> dict:
    root = root.resolve(strict=True)
    automation = root / 'automation'
    automation.mkdir(mode=0o700, exist_ok=True)
    snapshots = automation / 'state' / 'snapshots'
    snapshots.mkdir(mode=0o700, parents=True, exist_ok=True)
    state_link = automation / 'state' / 'current'
    with (root / 'publish.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        previous = _target(root / 'current', root / 'builds')
        active = verify_web(previous)
        if verify_only:
            health_check(active['release_id'])
            return {'status': 'verified', **active}
        if not api_key.strip():
            raise ValueError('OIKOLAB_API_KEY is required for scheduled refresh')
        cfg = load_config(config)
        stations = load_stations(cfg.stations_file)
        if len(stations) != 1 or stations[0].station_id != 'twin_falls':
            raise ValueError('Scheduled update is restricted to Twin Falls')
        station = stations[0]
        old_state = _target(state_link, snapshots) if state_link.is_symlink() else None
        database = old_state / 'weather.sqlite3' if old_state else cfg.database_path
        precip = old_state / 'precipitation.json' if old_state else cfg.precipitation_archives['twin_falls']
        last_date, complete = _coverage(database)
        archive = load_archive(precip)
        available = era5_available_end(api_key)
        # The audited source uses fixed UTC-07 throughout the year. Temperatures
        # are instantaneous; precipitation is assigned to the preceding hour.
        temp_end = pd.Timestamp(available) - pd.Timedelta(hours=7)
        rain_end = temp_end - pd.Timedelta(hours=1)
        def last_complete(value):
            return (value.date() if value.hour == 23 else value.date() - timedelta(days=1)).isoformat()
        need_fetch = not complete or last_date < last_complete(temp_end) or _precip_end(archive) < last_complete(rain_end)
        old_receipt = json.loads((old_state / 'receipt.json').read_text()) if old_state else {}
        if not need_fetch and old_receipt.get('source_revision') == source_revision and old_receipt.get('release_id') == active['release_id']:
            return {'status': 'already_current', 'last_complete_date': last_date,
                    'precipitation_last_complete_date': _precip_end(archive),
                    'era5_available_end_utc': available.isoformat(), 'release_id': active['release_id']}
        run_id = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '-' + uuid.uuid4().hex[:8]
        pending = Path(tempfile.mkdtemp(prefix='.pending-', dir=snapshots))
        build = root / 'builds' / ('daily-' + run_id)
        promoted = False
        reference_months = {}
        try:
            shutil.copyfile(database, pending / 'weather.sqlite3')
            shutil.copyfile(precip, pending / 'precipitation.json')
            effective = replace(cfg, database_path=pending / 'weather.sqlite3',
                                precipitation_archives={'twin_falls': pending / 'precipitation.json'})
            if need_fetch:
                start = datetime.fromisoformat(min(last_date, _precip_end(archive))).replace(tzinfo=timezone.utc)
                if old_state is None:
                    start = start.replace(day=1)  # Full bootstrap boundary month for independent audit.
                if start > available:
                    raise ValueError('Provider metadata predates required update bounds')
                # Raw bytes remain in a private temporary directory, never public.
                with tempfile.TemporaryDirectory(prefix='hourly-', dir=pending) as temporary:
                    hourly = Path(temporary) / '42.4634--114.4975-oikolab-era5.csv'
                    fetch_era5_hourly_csv(api_key, hourly, latitude=station.lat, longitude=station.lon,
                                          start=start, end=available)
                    reviewed = extend_archive(archive, hourly)
                    reference_months = audit_months(hourly, reviewed)
                    refresh_sqlite_database(effective, station, hourly, effective.database_path,
                                            requested_start_utc=start.isoformat(), requested_end_utc=available.isoformat())
                    _write_json(pending / 'precipitation.json', reviewed)
                    load_archive(pending / 'precipitation.json')
                    archive = reviewed
                    # Retain reproducible source bytes privately, never in a site build.
                    (pending / 'hourly-source.csv.gz').write_bytes(gzip.compress(hourly.read_bytes(), mtime=0))
            updated_date, updated_complete = _coverage(effective.database_path)
            if updated_date < last_date or _precip_end(archive) < _precip_end(load_archive(precip)):
                raise ValueError('Update regressed source coverage')
            if not updated_complete or updated_date < last_complete(temp_end) or _precip_end(archive) < last_complete(rain_end):
                raise ValueError('Update did not reach provider complete-day coverage')
            build.mkdir(mode=0o700)
            # Retain immutable data for views opened before this switch. Hard links
            # avoid duplicating every older release in each new build.
            shutil.copytree(previous / 'data', build / 'data', copy_function=os.link,
                            ignore=lambda directory, names: ['manifest.json'] if Path(directory) == previous / 'data' else [])
            export_web_data(effective, stations, build)
            report = verify_web(build)
            _public_permissions(build)
            receipt = {'status': 'published', 'run_id': run_id, 'source_revision': source_revision,
                       'release_id': report['release_id'], 'previous_build': str(previous),
                       'build': str(build), 'previous_state': str(old_state) if old_state else None,
                       'last_complete_date': updated_date, 'precipitation_last_complete_date': _precip_end(archive),
                       'era5_available_end_utc': available.isoformat(),
                       'database_sha256': hashlib.sha256(effective.database_path.read_bytes()).hexdigest(),
                       'precipitation_content_sha256': archive['content_sha256']}
            receipt['independent_monthly_precipitation'] = reference_months
            _write_json(pending / 'receipt.json', receipt)
            snapshot = snapshots / run_id
            pending.rename(snapshot)
            pending = snapshot
            _link(build, root / 'current')
            promoted = True
            health_check(report['release_id'])
            _link(snapshot, state_link)
            return receipt
        except BaseException:
            if promoted:
                _link(previous, root / 'current')
            # Never delete a build that might be active after an interrupted swap.
            if (root / 'current').resolve() != build:
                shutil.rmtree(build, ignore_errors=True)
            raise
        finally:
            if not state_link.is_symlink() or state_link.resolve() != pending:
                shutil.rmtree(pending, ignore_errors=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--root', type=Path, default=Path('/srv/idwx'))
    parser.add_argument('--source-revision', required=True)
    parser.add_argument('--verify-only', action='store_true')
    args = parser.parse_args()
    try:
        report = update(args.config, args.root, api_key=os.environ.get('OIKOLAB_API_KEY', ''),
                        source_revision=args.source_revision, verify_only=args.verify_only)
        print(json.dumps(report, sort_keys=True))
    except (OSError, ValueError, TypeError, KeyError, sqlite3.Error) as exc:
        # Requests already redact upstream details; never print tracebacks or env.
        print(json.dumps({'status': 'failed', 'error_type': type(exc).__name__, 'error': str(exc)}))
        raise SystemExit(1) from None


if __name__ == '__main__':
    main()
