from __future__ import annotations

import csv
import logging
import re
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
    for s in stations:
        if s.lat is None or s.lon is None:
            continue
        if abs(float(s.lat) - lat) < tol and abs(float(s.lon) - lon) < tol:
            return s
    return None


def _read_header(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8", newline="") as f:
        row = next(csv.reader(f))
    return row


def _index_map(header: list[str]) -> dict[str, int]:
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
) -> tuple[str, pd.DataFrame, str]:
    lat, lon = _parse_filename_coords(path)
    matched = _match_station_by_coord(lat, lon, stations)
    station_id = matched.station_id if matched else _default_station_id(lat, lon)
    elev_from_station = matched.elevation_m if matched else None

    LOG.info("Ingesting %s as station_id=%s", path.name, station_id)

    day_acc: dict[pd.Timestamp, DayAccumulator] = defaultdict(DayAccumulator)
    reader, idx, header = _iter_hourly_chunks(path, chunksize=chunksize)

    elev_seen: float | None = elev_from_station
    chunk_count = 0
    row_count = 0
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
        local_date = _compute_local_date(ts, utc_offset, config.timezone)

        if elev_seen is None and "elevation" in chunk.columns:
            elev_vals = pd.to_numeric(chunk["elevation"], errors="coerce").dropna()
            if not elev_vals.empty:
                elev_seen = float(elev_vals.iloc[0])

        frame = pd.DataFrame({"local_date": local_date, "temperature": temp, "precip": precip})
        frame = frame.dropna(subset=["local_date"])
        frame = frame.sort_values("local_date")

        for day, g in frame.groupby("local_date"):
            acc = day_acc[pd.Timestamp(day)]
            t = g["temperature"].to_numpy(dtype=float)
            valid_t = t[np.isfinite(t)]
            acc.n_obs += int(valid_t.size)
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
    return station_id, daily, precip_mode


def build_data_cache(config: Config, stations: list[StationMeta], rebuild: bool = False) -> list[str]:
    csvs = discover_csvs(config.data_root)
    cache_daily = config.cache_dir / "daily"
    cache_daily.mkdir(parents=True, exist_ok=True)

    written: list[str] = []
    mode_counts: dict[str, int] = defaultdict(int)
    for p in csvs:
        sid, daily, mode = ingest_era5_hourly_csv(p, config=config, stations=stations)
        day_path = cache_daily / f"{sid}.parquet"
        if not rebuild and day_path.exists():
            LOG.info("Skipping existing daily parquet for station %s", sid)
            continue
        daily.to_parquet(day_path, index=False)
        mode_counts[mode] += 1
        written.append(sid)

    LOG.info(
        "Ingest summary: files=%d written=%d cumulative_mode=%d incremental_mode=%d",
        len(csvs),
        len(written),
        mode_counts.get("cumulative", 0),
        mode_counts.get("incremental", 0),
    )
    return written
