"""Private, read-only source audit. Automated findings never certify source claims."""
from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from collections import Counter, defaultdict
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from idwx.config import Config, StationMeta
from idwx.io import (
    HEADER_PRECIP_MM,
    discover_csvs,
    ingest_era5_hourly_csv,
    resolve_source_station,
)
from idwx.web_export import station_history

AUDIT_VERSION = 1
MANUAL_REVIEW = {
    "source_provenance": "Verify provider, dataset/product, station identity, retrieval history and licensing against source evidence.",
    "temperature_units": "Confirm actual values are Celsius; a degC header is only a declaration, not independent verification.",
    "precipitation_semantics": "If precipitation is present, verify units and accumulation/reset conventions; automatic mode selection is only a heuristic.",
    "timestamp_convention": "Confirm source times describe UTC instants; the adapter interprets zone-free times as UTC.",
    "reporting_day": "Confirm explicit source offsets or the station IANA timezone define the intended local reporting day.",
    "coverage_policy": "Review the distinct-hour minimum; meeting it does not prove the true daily minimum was observed.",
    "frost_results": "Hand-check station-years, seasonal windows, gaps and summer freezes before publishing historical results.",
    "data_vintage": "Verify observation delivery dates and historical revisions before interpreting as-issued backtests.",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _compare_daily_cache(path: Path, fresh: pd.DataFrame) -> dict:
    if not path.exists():
        return {"status": "missing"}
    cached = pd.read_parquet(path)
    columns = list(fresh.columns)
    if not set(columns).issubset(cached.columns):
        return {"status": "mismatch", "reason": "Cached daily schema differs from the current aggregation"}

    def canonical(frame):
        frame = frame[columns].copy()
        dates = pd.to_datetime(frame["date"], errors="raise")
        if dates.isna().any() or dates.dt.tz is not None or not dates.eq(dates.dt.normalize()).all():
            raise ValueError("Cached dates must be nonmissing timezone-naive local dates")
        frame["date"] = dates.dt.strftime("%Y-%m-%d")
        for column in columns:
            if column in {"date", "station_id", "quality_flags"}:
                frame[column] = frame[column].fillna("").astype(str)
            else:
                frame[column] = pd.to_numeric(frame[column], errors="raise").astype(float)
        return frame.sort_values(["station_id", "date"]).reset_index(drop=True)

    return {"status": "matches_current_aggregation" if canonical(cached).equals(canonical(fresh)) else "mismatch"}


def _minimum_gaps(daily: pd.DataFrame, start: int, end: int) -> list[dict]:
    expected = pd.date_range(f"{start}-01-01", f"{end}-12-31")
    minima = daily.set_index("date").tmin_c.reindex(expected)
    gaps = []
    previous = None
    for day in expected[minima.isna()]:
        if previous is not None and day - previous == pd.Timedelta(days=1):
            gaps[-1]["end_date"] = day.date().isoformat()
            gaps[-1]["days"] += 1
        else:
            gaps.append({"start_date": day.date().isoformat(), "end_date": day.date().isoformat(), "days": 1})
        previous = day
    return gaps


def audit_data(cfg: Config, stations: list[StationMeta], *, chunksize: int = 100_000) -> dict:
    ids = [station.station_id for station in stations]
    if len(ids) != len(set(ids)) or any(not re.fullmatch(r"[a-z0-9]+(?:_[a-z0-9]+)*", sid) for sid in ids):
        raise ValueError("Audit requires unique lowercase snake_case configured station IDs")
    if (not cfg.frost_thresholds_c or not np.isfinite(cfg.frost_thresholds_c).all()
            or len(set(cfg.frost_thresholds_c)) != len(cfg.frost_thresholds_c)):
        raise ValueError("Audit requires nonempty unique finite frost thresholds")
    issues: list[dict] = []
    invalid_metadata_stations = set()
    entries = []
    by_station: dict[str, list[dict]] = defaultdict(list)
    report = {
        "audit_version": AUDIT_VERSION, "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "audit_status": "review_required", "publication_ready": False,
        "data_kind": "source_audit_not_certified_observations", "private_report": True,
        "data_root": str(cfg.data_root.resolve()), "hourly_min_coverage": cfg.hourly_min_coverage,
        "adapter": "era5_style_hourly_celsius", "manual_review_required": MANUAL_REVIEW.copy(),
        "configured_stations": [], "files": entries, "blocking_issues": issues,
    }
    if not stations:
        issues.append({"code": "no_configured_stations", "detail": "Configure station metadata before auditing results"})
    try:
        paths = discover_csvs(cfg.data_root)
    except OSError as exc:
        paths = []
        issues.append({"code": "source_inventory_unavailable", "detail": str(exc)})
    if not paths and not issues:
        issues.append({"code": "no_csv_sources", "detail": "No source CSV files found under the configured data root"})

    # Inventory all mappings before processing. Multi-file stations are never
    # combined by joining daily aggregates, which could mishandle split days.
    for path in paths:
        entry = {"file": path.relative_to(cfg.data_root).as_posix(), "status": "blocked",
                 "station_id": None, "sha256": None, "diagnostics": {"complete": False}, "historical_review": None}
        entries.append(entry)
        try:
            sid, matched, lat, lon = resolve_source_station(path, stations)
            entry.update(station_id=sid, configured_station=matched is not None,
                         filename_coordinates={"lat": lat, "lon": lon})
            by_station[sid].append(entry)
        except ValueError as exc:
            issues.append({"file": entry["file"], "code": "station_mapping_error", "detail": str(exc)})

    for sid, files in by_station.items():
        if len(files) > 1:
            issues.append({"station_id": sid, "code": "multiple_station_sources",
                           "detail": "Multiple CSVs map to one station. Consolidate hourly records with duplicate checks; do not join daily aggregates."})
    for station in stations:
        files = by_station.get(station.station_id, [])
        metadata = asdict(station)
        for field in ("lat", "lon", "elevation_m"):
            if metadata[field] is not None:
                try:
                    value = float(metadata[field])
                    if not np.isfinite(value):
                        raise ValueError("nonfinite")
                    metadata[field] = value
                except (TypeError, ValueError):
                    metadata[field] = None
                    invalid_metadata_stations.add(station.station_id)
                    issues.append({"station_id": station.station_id, "code": "invalid_station_metadata",
                                   "detail": f"Review nonnumeric or nonfinite {field}"})
        report["configured_stations"].append({**metadata, "effective_timezone": station.timezone or cfg.timezone,
            "availability": "missing_source" if not files else "multiple_sources" if len(files) > 1 else "source_found",
            "source_files": [entry["file"] for entry in files]})
        if not files:
            issues.append({"station_id": station.station_id, "code": "missing_station_source", "detail": "No source file matches this configured station"})

    for path, entry in zip(paths, entries, strict=True):
        if entry["station_id"] is None:
            continue
        try:
            before = path.stat()
            entry.update(size_bytes=before.st_size, sha256=_sha256(path))
            if len(by_station[entry["station_id"]]) > 1 or entry["station_id"] in invalid_metadata_stations:
                continue
            sid, daily, precip_mode = ingest_era5_hourly_csv(path, cfg, stations, chunksize=chunksize, diagnostics=entry["diagnostics"])
            after = path.stat()
            if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                raise ValueError("Source file changed during audit; retry against a stable snapshot")
            if daily.empty:
                raise ValueError("Source contains no daily records")
            precipitation_present = HEADER_PRECIP_MM in entry["diagnostics"]["header"]
            entry["precipitation_present"] = precipitation_present
            entry["precipitation_mode_heuristic"] = precip_mode if precipitation_present else None
            if not entry["configured_station"]:
                issues.append({"file": entry["file"], "code": "unconfigured_source", "detail": "No configured station matches source coordinates; review metadata"})
                continue
            station = next(station for station in stations if station.station_id == sid)
            start = int(cfg.backtest.get("start_year", daily.date.dt.year.min()))
            end = int(cfg.backtest.get("end_year", daily.date.dt.year.max()))
            entry["historical_review"] = station_history(daily, station, cfg, (start, end), min_years=10)
            entry["status"] = "computed_for_review"
            annual = entry["historical_review"]["annual"]
            entry["review_summary"] = {
                "event_status_counts": {name: dict(Counter(row[name]["status"] for row in annual))
                                        for name in ("last_spring_frost", "first_fall_frost")},
                "entirely_missing_years": sorted(set(range(start, end + 1)) - set(daily.date.dt.year)),
                "missing_daily_minimum_ranges": _minimum_gaps(daily, start, end),
                "summer_freeze_records": [{"season_year": row["season_year"], "threshold_c": row["threshold_c"]}
                                          for row in annual if row["interval"]["summer_freeze_observed"]],
            }
            try:
                entry["daily_cache_comparison"] = _compare_daily_cache(cfg.cache_dir / "daily" / f"{sid}.parquet", daily)
            except (OSError, ValueError, KeyError) as exc:
                entry["daily_cache_comparison"] = {"status": "unreadable", "reason": str(exc)}
            if entry["daily_cache_comparison"]["status"] in {"mismatch", "unreadable"}:
                entry["status"] = "blocked"
                issues.append({"file": entry["file"], "code": "daily_cache_mismatch", "detail": "Rebuild the daily cache; it does not match current source aggregation"})
            for field, code in (("date_only_timestamp_rows", "unsupported_daily_timestamps"),
                                ("invalid_temperature_rows", "invalid_temperature_values")):
                if entry["diagnostics"][field]:
                    entry["status"] = "blocked"
                    issues.append({"file": entry["file"], "code": code, "detail": f"Review {entry['diagnostics'][field]} affected rows before using source results"})
        except (OSError, ValueError, KeyError) as exc:
            entry["status"] = "blocked"
            entry["historical_review"] = None
            issues.append({"file": entry["file"], "code": "source_processing_error", "detail": str(exc)})
    if issues:
        report["audit_status"] = "blocked"
    return report


def write_audit_report(cfg: Config, report: dict, output: Path) -> None:
    """Write strict JSON atomically, outside raw input and publication directories."""
    output = output.resolve()
    if output.suffix.lower() != ".json":
        raise ValueError("Audit output must be a .json report")
    if output.is_relative_to(cfg.data_root.resolve()) or output.is_relative_to((Path.cwd() / "web").resolve()):
        raise ValueError("Private audit reports must stay outside source and publication directories")
    if any((parent / "data/manifest.json").is_file() or
           ((parent / "index.html").is_file() and (parent / "js/app.js").is_file()) for parent in output.parents):
        raise ValueError("Private audit reports must stay outside publication directories")
    content = json.dumps(report, allow_nan=False, sort_keys=True, indent=2) + "\n"
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", newline="\n", dir=output.parent,
                                         prefix=".audit-", suffix=".tmp", delete=False) as temporary:
            temporary_path = Path(temporary.name)
            temporary.write(content)
        os.replace(temporary_path, output)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
