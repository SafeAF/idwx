"""Allowlisted public historical data, with immutable releases and an atomic manifest."""
from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import os
import re
import tempfile
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from idwx.config import Config, StationMeta
from idwx.frost import frost_rows, validated_daily
from idwx.history import summarize_events
from idwx.sqlite_store import load_daily_from_sqlite

SCHEMA_VERSION = "1.0.0"
WEB_ASSETS = ("index.html", "css/styles.css", "js/app.js", "js/charts.js", "js/forecast.js", "js/format.js")
EVENT_NAMES = ("last_spring_frost", "first_fall_frost")
NOTICE = (
    "Temperature-based freeze dates are a proxy for frost risk. Surface frost and crop damage "
    "also depend on local conditions. Historical quantiles are not forecasts or guarantees. "
    "Spring P90 is a later last-freeze date; fall P10 is an earlier first-freeze date."
)


def _clean(value):
    if isinstance(value, dict):
        return {key: _clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(item) for item in value]
    if value is None or value is pd.NaT or value is pd.NA:
        return None
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.date().isoformat()
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if math.isfinite(value) else None
    return value


def _json(value) -> bytes:
    return (json.dumps(_clean(value), allow_nan=False, sort_keys=True, indent=2) + "\n").encode("utf-8")


def _coverage(df: pd.DataFrame) -> dict:
    valid = df.loc[df.tmin_c.notna(), "date"]
    return _clean({
        "first_date": df.date.min(), "last_date": df.date.max(),
        "first_valid_minimum_date": valid.min(), "last_valid_minimum_date": valid.max(),
        "recorded_days": len(df), "valid_minimum_days": len(valid),
    })


def _event(row: dict) -> dict:
    return {
        "status": row["status"], "date": row["value_date"], "tmin_c": row["event_tmin_c"],
        "candidate_date": row["candidate_date"], "candidate_tmin_c": row["candidate_tmin_c"],
        "window_start": row["window_start"], "window_end": row["window_end"],
        **{key: row[key] for key in ("expected_days", "valid_days", "missing_days", "coverage_fraction")},
    }


def station_history(daily: pd.DataFrame, station: StationMeta, cfg: Config,
                    baseline: tuple[int, int], min_years: int) -> dict:
    df = validated_daily(daily, station.station_id)
    if df.empty:
        raise ValueError(f"No daily records for configured station {station.station_id}")
    start = int(cfg.backtest.get("start_year", df.index.year.min()))
    # The model backtest can intentionally retain an older information cutoff,
    # but the historical website should expose all available years. Research
    # model training remains governed by cfg.backtest elsewhere.
    end = max(int(cfg.backtest.get("end_year", df.index.year.max())), int(df.index.year.max()))
    if start > end or not cfg.frost_thresholds_c:
        raise ValueError("Export requires an ordered year range and at least one threshold")
    if not start <= baseline[0] <= baseline[1] <= end:
        raise ValueError("Baseline must be ordered and within the exported season-year range")
    annual = []
    for year in range(start, end + 1):
        rows = frost_rows(df, station.station_id, year, cfg)
        for index in range(0, len(rows), 3):
            spring, fall, interval = rows[index:index + 3]
            annual.append({
                "season_year": year, "threshold_c": spring["threshold_c"],
                EVENT_NAMES[0]: _event(spring), EVENT_NAMES[1]: _event(fall),
                "interval": {
                    "date_difference_days": interval["value_float"],
                    "frost_free_days": interval["frost_free_days"],
                    "status": interval["status"], "summer_freeze_observed": interval["summer_freeze"],
                },
            })
    summaries = []
    for threshold in cfg.frost_thresholds_c:
        records = [r for r in annual if r["threshold_c"] == threshold]
        summaries.append({
            "threshold_c": threshold,
            **{name: summarize_events(
                [{"season_year": r["season_year"], **r[name]} for r in records],
                *baseline, min_years=min_years,
            ) for name in EVENT_NAMES},
        })
    metadata = asdict(station)
    for key in ("lat", "lon", "elevation_m"):
        if metadata[key] is not None:
            metadata[key] = float(metadata[key])
    metadata["timezone"] = station.timezone or cfg.timezone
    return _clean({
        "station": metadata, "source_coverage": _coverage(df),
        "season_years": {"start_year": start, "end_year": end},
        "windows": {**cfg.windows, **cfg.station_windows.get(station.station_id, {})},
        "thresholds_c": cfg.frost_thresholds_c, "annual": annual, "summaries": summaries,
    })


def _csv(documents: list[dict], common: dict) -> bytes:
    rows = []
    for document in documents:
        for record in document["annual"]:
            row = {**{key: common[key] for key in ("schema_version", "release_id", "data_kind")},
                   "station_id": document["station"]["station_id"],
                   "season_year": record["season_year"], "threshold_c": record["threshold_c"]}
            for name in EVENT_NAMES:
                row.update({f"{name}_{key}": value for key, value in record[name].items()})
            row.update({f"interval_{key}": value for key, value in record["interval"].items()})
            rows.append(row)
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode("utf-8")


def demo_daily(station_id: str) -> pd.DataFrame:
    """Deterministic synthetic fixtures, never a replacement for real observations."""
    dates = pd.date_range("2000-01-01", "2022-12-31")
    df = pd.DataFrame({"station_id": station_id, "date": dates, "tmin_c": 8.0})
    for year in range(2000, 2023):
        spring = pd.Timestamp(year=year, month=5, day=1 + year % 20)
        fall = pd.Timestamp(year=year, month=9, day=1 + year % 25)
        df.loc[df.date.isin([spring, fall]), "tmin_c"] = -3.0
    df.loc[df.date.eq("2020-07-04"), "tmin_c"] = -3.0
    df.loc[df.date.eq("2021-06-01"), "tmin_c"] = np.nan
    df.loc[df.date.dt.year.eq(2019), "tmin_c"] = 8.0
    return df


def export_web_data(cfg: Config, stations: list[StationMeta], output: Path,
                    baseline: tuple[int, int] | None = None, min_years: int = 10,
                    demo: bool = False) -> dict:
    """Calculate from daily caches, then publish data without trusting stale target caches."""
    if not stations:
        raise ValueError("At least one configured station is required")
    ids = [s.station_id for s in stations]
    if len(set(ids)) != len(ids) or any(not re.fullmatch(r"[a-z0-9]+(?:_[a-z0-9]+)*", s) for s in ids):
        raise ValueError("Configured station IDs must be unique lowercase snake_case identifiers")
    if len(set(cfg.frost_thresholds_c)) != len(cfg.frost_thresholds_c):
        raise ValueError("Frost thresholds must be unique")
    documents = []
    data_kinds = set()
    for station in sorted(stations, key=lambda s: s.station_id):
        if demo:
            daily = demo_daily(station.station_id)
            data_kinds.add("synthetic_demo")
        elif cfg.database_path is not None:
            daily, data_kind = load_daily_from_sqlite(cfg.database_path, station)
            data_kinds.add(data_kind)
        else:
            daily = pd.read_parquet(cfg.cache_dir / "daily" / f"{station.station_id}.parquet")
            data_kinds.add("historical_observations")
        years = pd.to_datetime(daily["date"]).dt.year
        last_date = pd.Timestamp(daily["date"].max())
        latest_complete_year = last_date.year if (last_date.month, last_date.day) == (12, 31) else last_date.year - 1
        period = baseline or (
            int(cfg.backtest.get("start_year", years.min())),
            max(int(cfg.backtest.get("end_year", years.max())), latest_complete_year),
        )
        documents.append(station_history(daily, station, cfg, period, min_years))
    if len(data_kinds) != 1:
        raise ValueError("All exported stations must use the same data kind")
    kind = data_kinds.pop()
    common = {"schema_version": SCHEMA_VERSION, "data_kind": kind, "methodology": NOTICE}
    identity = hashlib.sha256(_json({**common, "stations": documents})).hexdigest()
    release_id = identity  # Full content hash avoids mutable release paths.
    common = {**common, "release_id": release_id}
    root = f"data/{release_id}"
    index = [{"station": d["station"], "source_coverage": d["source_coverage"],
              "url": f"{root}/stations/{d['station']['station_id']}.json"} for d in documents]
    files = {
        "stations.json": _json({**common, "stations": index}),
        "downloads/frost-history.csv": _csv(documents, common),
        **{f"stations/{d['station']['station_id']}.json": _json({**common, **d}) for d in documents},
    }
    manifest = {
        **common, "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "stations_url": f"{root}/stations.json", "history_csv_url": f"{root}/downloads/frost-history.csv",
        "source_coverage": {
            "first_date": min(d["source_coverage"]["first_date"] for d in documents),
            "last_date": max(d["source_coverage"]["last_date"] for d in documents),
        },
        "files": {f"{root}/{name}": hashlib.sha256(content).hexdigest() for name, content in files.items()},
    }
    data_dir = Path(output) / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    release_dir = data_dir / release_id
    if release_dir.exists():
        actual_files = {p.relative_to(release_dir).as_posix() for p in release_dir.rglob("*") if p.is_file()}
        if actual_files != set(files) or any((release_dir / name).read_bytes() != content
               for name, content in files.items()):
            raise ValueError("Existing immutable release differs from expected contents")
    else:
        with tempfile.TemporaryDirectory(prefix=".staging-", dir=data_dir) as stage:
            staged = Path(stage) / release_id
            for name, content in files.items():
                destination = staged / name
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(content)
            os.replace(staged, release_dir)
    # The manifest changes only after the complete immutable release exists.
    with tempfile.TemporaryDirectory(prefix=".manifest-", dir=data_dir) as stage:
        # Assets are allowlisted: generated data, source files, and private files
        # can never be copied into a publication by directory traversal.
        assets = Path(__file__).resolve().parent / "web_assets"
        if not (assets / "index.html").is_file():
            assets = Path(__file__).resolve().parent.parent / "web"
        for name in WEB_ASSETS:
            destination = Path(output) / name
            content = (assets / name).read_bytes()
            if destination.is_file() and destination.read_bytes() == content:
                continue
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary_asset = Path(stage) / "asset"
            temporary_asset.write_bytes(content)
            os.replace(temporary_asset, destination)
        temporary_manifest = Path(stage) / "manifest.json"
        temporary_manifest.write_bytes(_json(manifest))
        os.replace(temporary_manifest, data_dir / "manifest.json")
    return manifest
