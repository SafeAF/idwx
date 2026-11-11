from __future__ import annotations

import csv
import logging
import os
import re
import tempfile
from collections import defaultdict
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from idwx.config import Config, StationMeta
from idwx.daily import DayAccumulator, finalize_daily

LOG = logging.getLogger(__name__)

FILENAME_COORD_RE = re.compile(r"^([+-]?\d+(?:\.\d+)?)--([+-]?\d+(?:\.\d+)?)-")
HEADER_COORDS = "coordinates (lat,lon)"
HEADER_ELEV = "model elevation (surface)"
HEADER_UTC_OFFSET = "utc_offset (hrs)"
HEADER_TEMP_C = "temperature (degC)"
HEADER_PRECIP_MM = "total_precipitation (mm of water equivalent)"

def discover_csvs(data_root: Path) -> list[Path]:
    if not data_root.exists():
        raise FileNotFoundError(f"data_root does not exist: {data_root}")
    return sorted(data_root.rglob("*.csv"))


def _parse_filename_coords(path: Path) -> tuple[float, float]:
    m = FILENAME_COORD_RE.match(path.name)
    if not m:
        raise ValueError(f"Could not parse lat/lon from filename: {path.name}")
    return float(m.group(1)), float(m.group(2))


def _fmt6(v: float) -> str:
    return f"{v:.6f}"


def _default_station_id(lat: float, lon: float) -> str:
    return f"lat_{_fmt6(lat)}__lon_{_fmt6(lon)}"


def _match_station_by_coord(lat: float, lon: float, stations: list[StationMeta], tol: float = 1e-4) -> StationMeta | None:
    matches = []
    for s in stations:
        if s.lat is None or s.lon is None:
            continue
        if abs(float(s.lat) - lat) < tol and abs(float(s.lon) - lon) < tol:
            matches.append(s)
    if len(matches) > 1:
        raise ValueError("Source coordinates match multiple configured stations; resolve station metadata before ingestion")
    return matches[0] if matches else None


def resolve_source_station(path: Path, stations: list[StationMeta]) -> tuple[str, StationMeta | None, float, float]:
    lat, lon = _parse_filename_coords(path)
    if not -90 <= lat <= 90 or not -180 <= lon <= 180:
        raise ValueError(f"Invalid filename coordinates in {path.name}")
    matched = _match_station_by_coord(lat, lon, stations)
    if matched and not re.fullmatch(r"[a-z0-9]+(?:_[a-z0-9]+)*", matched.station_id):
        raise ValueError("Configured station IDs must use lowercase snake_case")
    sid = matched.station_id if matched else _default_station_id(lat, lon)
    return sid, matched, lat, lon


def _read_header(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8", newline="") as f:
        try:
            row = next(csv.reader(f))
        except StopIteration as exc:
            raise ValueError(f"Source CSV has no header: {path.name}") from exc
    return row


def _index_map(header: list[str]) -> dict[str, int]:
    if not header or len(set(header)) != len(header):
        raise ValueError("Source header must be nonempty and contain unique column names")
    idx = {"timestamp": 0}
    for i, col in enumerate(header):
        if col == HEADER_UTC_OFFSET:
            idx["utc_offset"] = i
        elif col == HEADER_TEMP_C:
            idx["temperature"] = i
        elif col == HEADER_PRECIP_MM:
            idx["precip"] = i
        elif col == HEADER_ELEV:
            idx["elevation"] = i
        elif col == HEADER_COORDS:
            idx["coords"] = i
    if "temperature" not in idx:
        raise ValueError(f"Missing required column '{HEADER_TEMP_C}' in file header")
    return idx


def _iter_hourly_chunks(
    path: Path, chunksize: int = 100_000
) -> tuple[pd.io.parsers.TextFileReader, dict[str, int], list[str]]:
    header = _read_header(path)
    idx = _index_map(header)
    usecols = sorted(set(idx.values()))
    reader = pd.read_csv(path, usecols=usecols, chunksize=chunksize, low_memory=False)
    return reader, idx, header


def _compute_local_date(ts: pd.Series, utc_offset: pd.Series | None, timezone_name: str) -> pd.Series:
    ts_utc = pd.to_datetime(ts, errors="coerce", utc=True)
    if utc_offset is not None:
        offset_td = pd.to_timedelta(pd.to_numeric(utc_offset, errors="coerce").fillna(np.nan), unit="h")
        local_ts = ts_utc + offset_td
        local_date = pd.to_datetime(local_ts.dt.date, errors="coerce")
        missing = pd.to_numeric(utc_offset, errors="coerce").isna()
    else:
        local_date = pd.Series(pd.NaT, index=ts.index)
        missing = pd.Series(True, index=ts.index)

    if missing.any():
        tz = ZoneInfo(timezone_name)
        fallback = ts_utc[missing].dt.tz_convert(tz).dt.date
        local_date.loc[missing] = pd.to_datetime(fallback, errors="coerce")

    return local_date


def ingest_era5_hourly_csv(
    path: Path,
    config: Config,
    stations: list[StationMeta],
    chunksize: int = 100_000,
    diagnostics: dict | None = None,
) -> tuple[str, pd.DataFrame, str]:
    station_id, matched, lat, lon = resolve_source_station(path, stations)
    elev_from_station = matched.elevation_m if matched else None

    LOG.info("Ingesting %s as station_id=%s", path.name, station_id)

    day_acc: dict[pd.Timestamp, DayAccumulator] = defaultdict(DayAccumulator)
    reader, idx, header = _iter_hourly_chunks(path, chunksize=chunksize)
    if diagnostics is not None:
        diagnostics.update(complete=False, header=header, rows=0, invalid_timestamp_rows=0,
                           missing_temperature_rows=0, invalid_temperature_rows=0,
                           duplicate_timestamp_rows=0, date_only_timestamp_rows=0,
                           timestamps_without_zone_suffix_rows=0, temperature_min_c=None, temperature_max_c=None,
                           explicit_offset_rows=0, invalid_offset_rows=0, fallback_timezone_rows=0, timestamp_order_violations=0,
                           first_timestamp_utc=None, last_timestamp_utc=None, source_offset_min=None, source_offset_max=None)

    elev_seen: float | None = elev_from_station
    chunk_count = 0
    row_count = 0
    seen_timestamps: set[pd.Timestamp] = set()
    covered_hours: dict[pd.Timestamp, set[pd.Timestamp]] = defaultdict(set)
    previous_timestamp = None
    for chunk in reader:
        chunk_count += 1
        row_count += len(chunk)
        rename: dict[str, str] = {chunk.columns[0]: "timestamp"}
        for key, pos in idx.items():
            if key == "timestamp":
                continue
            rename[header[pos]] = key
        chunk = chunk.rename(columns=rename)

        ts = chunk.get("timestamp")
        if ts is None:
            continue
        temp = pd.to_numeric(chunk.get("temperature"), errors="coerce")
        precip = pd.to_numeric(chunk.get("precip"), errors="coerce") if "precip" in chunk.columns else pd.Series(np.nan, index=chunk.index)
        utc_offset = pd.to_numeric(chunk.get("utc_offset"), errors="coerce") if "utc_offset" in chunk.columns else None
        timestamps = pd.to_datetime(ts, errors="coerce", utc=True)
        if diagnostics is not None:
            diagnostics["rows"] += len(chunk)
            diagnostics["invalid_timestamp_rows"] += int(timestamps.isna().sum())
            diagnostics["missing_temperature_rows"] += int(chunk["temperature"].isna().sum())
            diagnostics["invalid_temperature_rows"] += int((chunk["temperature"].notna() & ~np.isfinite(temp)).sum())
            timestamp_text = ts.astype("string")
            diagnostics["date_only_timestamp_rows"] += int(timestamp_text.str.fullmatch(r"\d{4}-\d{2}-\d{2}", na=False).sum())
            diagnostics["timestamps_without_zone_suffix_rows"] += int((~timestamp_text.str.contains(r"(?:[zZ]|[+-]\d{2}:?\d{2})$", na=False)).sum())
            explicit = utc_offset.notna() if utc_offset is not None else pd.Series(False, index=chunk.index)
            diagnostics["explicit_offset_rows"] += int(explicit.sum())
            diagnostics["fallback_timezone_rows"] += int((~explicit).sum())
            valid = timestamps.dropna()
            diagnostics["timestamp_order_violations"] += int((valid.diff() < pd.Timedelta(0)).sum())
            if len(valid):
                if previous_timestamp is not None and valid.iloc[0] < previous_timestamp:
                    diagnostics["timestamp_order_violations"] += 1
                previous_timestamp = valid.iloc[-1]
                for key, candidate, operation in (("first_timestamp_utc", valid.min(), min), ("last_timestamp_utc", valid.max(), max)):
                    old = diagnostics[key]
                    diagnostics[key] = operation(old, candidate.isoformat()) if old is not None else candidate.isoformat()
            if utc_offset is not None:
                finite_offsets = utc_offset[np.isfinite(utc_offset)]
                if len(finite_offsets):
                    for key, value, operation in (("source_offset_min", float(finite_offsets.min()), min),
                                                   ("source_offset_max", float(finite_offsets.max()), max)):
                        old = diagnostics[key]
                        diagnostics[key] = operation(old, value) if old is not None else value
            finite_temperatures = temp[np.isfinite(temp)]
            if len(finite_temperatures):
                for key, value, operation in (("temperature_min_c", float(finite_temperatures.min()), min),
                                               ("temperature_max_c", float(finite_temperatures.max()), max)):
                    old = diagnostics[key]
                    diagnostics[key] = operation(old, value) if old is not None else value
        if timestamps.isna().any():
            raise ValueError(f"Invalid or missing hourly timestamps in {path.name}; normalize the source before ingestion")
        if utc_offset is not None:
            malformed = chunk["utc_offset"].notna() & (utc_offset.isna() | ~np.isfinite(utc_offset) | utc_offset.abs().ge(24))
            if malformed.any():
                if diagnostics is not None:
                    diagnostics["invalid_offset_rows"] += int(malformed.sum())
                raise ValueError(f"Invalid source UTC offsets in {path.name}; offsets must be finite and less than 24 hours in magnitude")
        valid_timestamps = timestamps.dropna()
        duplicates = valid_timestamps.duplicated() | valid_timestamps.isin(seen_timestamps)
        if diagnostics is not None:
            diagnostics["duplicate_timestamp_rows"] += int(duplicates.sum())
        if duplicates.any():
            raise ValueError(f"Duplicate hourly timestamps in {path.name}; resolve before ingestion")
        seen_timestamps.update(valid_timestamps)
        timezone_name = matched.timezone if matched and matched.timezone else config.timezone
        local_date = _compute_local_date(ts, utc_offset, timezone_name)

        if elev_seen is None and "elevation" in chunk.columns:
            elev_vals = pd.to_numeric(chunk["elevation"], errors="coerce").dropna()
            if not elev_vals.empty:
                elev_seen = float(elev_vals.iloc[0])

        frame = pd.DataFrame({"local_date": local_date, "temperature": temp, "precip": precip,
                              "hour": timestamps.dt.floor("h")})
        frame = frame.dropna(subset=["local_date"])
        frame = frame.sort_values("local_date")

        for day, g in frame.groupby("local_date"):
            acc = day_acc[pd.Timestamp(day)]
            t = g["temperature"].to_numpy(dtype=float)
            valid_t = t[np.isfinite(t)]
            # Subhourly readings cannot make a sparse day meet hourly coverage.
            hours = g.loc[np.isfinite(g["temperature"]), "hour"].dropna()
            covered_hours[pd.Timestamp(day)].update(hours)
            acc.n_obs = len(covered_hours[pd.Timestamp(day)])
            if valid_t.size > 0:
                acc.temp_sum += float(valid_t.sum())
                acc.temp_count += int(valid_t.size)
                acc.tmin = min(acc.tmin, float(valid_t.min()))
                acc.tmax = max(acc.tmax, float(valid_t.max()))
            p = g["precip"].to_numpy(dtype=float)
            acc.precip_values.extend([float(v) for v in p if np.isfinite(v)])

    daily, precip_mode = finalize_daily(
        station_id=station_id,
        lat=lat if matched is None or matched.lat is None else float(matched.lat),
        lon=lon if matched is None or matched.lon is None else float(matched.lon),
        elevation_m=elev_seen,
        day_acc=day_acc,
        min_coverage=config.hourly_min_coverage,
    )

    LOG.info(
        "Station %s: chunks=%d rows=%d days=%d precip_mode=%s",
        station_id,
        chunk_count,
        row_count,
        len(daily),
        precip_mode,
    )
    if diagnostics is not None:
        diagnostics.update(complete=True, distinct_utc_timestamps=len(seen_timestamps),
                           distinct_covered_utc_hours=len(set().union(*covered_hours.values())) if covered_hours else 0)
    return station_id, daily, precip_mode


def build_data_cache(config: Config, stations: list[StationMeta], rebuild: bool = False) -> list[str]:
    csvs = discover_csvs(config.data_root)
    if not csvs:
        raise FileNotFoundError(f"No source CSV files found under data_root: {config.data_root}")
    sources: dict[str, Path] = {}
    for path in csvs:
        sid, _, _, _ = resolve_source_station(path, stations)
        if sid in sources:
            raise ValueError(f"Multiple CSV files map to station {sid}; consolidate and audit source records before building")
        sources[sid] = path
    cache_daily = config.cache_dir / "daily"
    cache_daily.mkdir(parents=True, exist_ok=True)

    written: list[str] = []
    mode_counts: dict[str, int] = defaultdict(int)
    # Finish all parsing/serialization before replacing any existing station cache.
    with tempfile.TemporaryDirectory(prefix=".staging-", dir=cache_daily) as stage:
        for sid, path in sources.items():
            if not rebuild and (cache_daily / f"{sid}.parquet").exists():
                LOG.info("Skipping existing daily parquet for station %s", sid)
                continue
            _, daily, mode = ingest_era5_hourly_csv(path, config=config, stations=stations)
            daily.to_parquet(Path(stage) / f"{sid}.parquet", index=False)
            mode_counts[mode] += 1
            written.append(sid)
        for sid in written:
            os.replace(Path(stage) / f"{sid}.parquet", cache_daily / f"{sid}.parquet")

    LOG.info(
        "Ingest summary: files=%d written=%d cumulative_mode=%d incremental_mode=%d",
        len(csvs),
        len(written),
        mode_counts.get("cumulative", 0),
        mode_counts.get("incremental", 0),
    )
    return written
