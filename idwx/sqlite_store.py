"""Versioned SQLite storage for application-ready daily weather data."""
from __future__ import annotations

import hashlib
import math
import os
import sqlite3
import tempfile
from pathlib import Path

import pandas as pd

from idwx.config import Config, StationMeta
from idwx.io import ingest_era5_hourly_csv

DATABASE_SCHEMA_VERSION = 1
DATABASE_APPLICATION_ID = 0x49445758  # ASCII "IDWX"
REANALYSIS_DATA_KIND = "historical_reanalysis"
DAILY_COLUMNS = (
    "station_id", "date", "tmin_c", "tmax_c", "tmean_c", "precip_mm",
    "lat", "lon", "elevation_m", "n_obs", "quality_flags",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _optional_float(value):
    return None if value is None or not math.isfinite(float(value)) else float(value)


def _daily_rows(daily: pd.DataFrame):
    ordered = daily.sort_values(["station_id", "date"])
    for row in ordered.loc[:, DAILY_COLUMNS].itertuples(index=False, name=None):
        station_id, date, *values = row
        numeric = [None if pd.isna(value) else value for value in values[:-1]]
        yield (str(station_id), pd.Timestamp(date).date().isoformat(), *numeric, str(values[-1] or ""))


def _write_database(path: Path, station: StationMeta, daily: pd.DataFrame, *, source: Path,
                    source_sha256: str, source_size: int, precip_mode: str,
                    hourly_min_coverage: int) -> None:
    with sqlite3.connect(path) as connection:
        connection.executescript(f"""
            PRAGMA page_size = 4096;
            PRAGMA application_id = {DATABASE_APPLICATION_ID};
            PRAGMA user_version = {DATABASE_SCHEMA_VERSION};
            PRAGMA foreign_keys = ON;
            CREATE TABLE metadata (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            ) STRICT;
            CREATE TABLE stations (
                station_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                latitude REAL,
                longitude REAL,
                elevation_m REAL,
                source TEXT,
                timezone TEXT NOT NULL
            ) STRICT;
            CREATE TABLE source_files (
                station_id TEXT PRIMARY KEY REFERENCES stations(station_id),
                file_name TEXT NOT NULL,
                sha256 TEXT NOT NULL CHECK(length(sha256) = 64),
                size_bytes INTEGER NOT NULL CHECK(size_bytes > 0),
                adapter TEXT NOT NULL,
                precipitation_mode TEXT NOT NULL,
                hourly_min_coverage INTEGER NOT NULL CHECK(hourly_min_coverage > 0),
                first_local_date TEXT NOT NULL,
                last_local_date TEXT NOT NULL
            ) STRICT;
            CREATE TABLE daily_weather (
                station_id TEXT NOT NULL REFERENCES stations(station_id),
                date TEXT NOT NULL CHECK(length(date) = 10),
                tmin_c REAL,
                tmax_c REAL,
                tmean_c REAL,
                precip_mm REAL,
                latitude REAL NOT NULL,
                longitude REAL NOT NULL,
                elevation_m REAL,
                n_obs INTEGER NOT NULL CHECK(n_obs >= 0),
                quality_flags TEXT NOT NULL,
                PRIMARY KEY (station_id, date)
            ) STRICT, WITHOUT ROWID;
        """)
        connection.executemany("INSERT INTO metadata(key, value) VALUES (?, ?)", (
            ("schema_version", str(DATABASE_SCHEMA_VERSION)),
            ("data_kind", REANALYSIS_DATA_KIND),
            ("date_semantics", "local_calendar_date"),
            ("temperature_unit", "degrees_celsius"),
            ("precipitation_unit", "millimeters"),
        ))
        connection.execute(
            "INSERT INTO stations VALUES (?, ?, ?, ?, ?, ?, ?)",
            (station.station_id, station.name, _optional_float(station.lat),
             _optional_float(station.lon), _optional_float(station.elevation_m),
             station.source, station.timezone or "America/Boise"),
        )
        connection.execute(
            "INSERT INTO source_files VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (station.station_id, source.name, source_sha256, source_size,
             "era5_style_hourly_celsius_v1", precip_mode, hourly_min_coverage,
             pd.Timestamp(daily["date"].min()).date().isoformat(),
             pd.Timestamp(daily["date"].max()).date().isoformat()),
        )
        connection.executemany(
            "INSERT INTO daily_weather VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            _daily_rows(daily),
        )
        connection.commit()
        result = connection.execute("PRAGMA integrity_check").fetchone()
        if result != ("ok",):
            raise ValueError(f"SQLite integrity check failed: {result}")
        connection.execute("VACUUM")


def build_sqlite_database(config: Config, station: StationMeta, source: Path,
                          output: Path) -> dict:
    """Aggregate one audited-style hourly source and atomically replace a SQLite database."""
    source = source.expanduser().resolve()
    output = output.expanduser().resolve()
    if source == output:
        raise ValueError("SQLite output must not replace the raw source file")
    before = source.stat()
    source_digest = _sha256(source)
    station_id, daily, precip_mode = ingest_era5_hourly_csv(source, config, [station])
    after = source.stat()
    if station_id != station.station_id:
        raise ValueError(f"Source coordinates do not match configured station {station.station_id}")
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError("Source file changed while the database was being built")
    if daily.empty:
        raise ValueError("Source produced no daily weather rows")

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        descriptor, name = tempfile.mkstemp(prefix=f".{output.name}.", suffix=".tmp", dir=output.parent)
        os.close(descriptor)
        temporary_path = Path(name)
        _write_database(temporary_path, station, daily, source=source,
                        source_sha256=source_digest, source_size=before.st_size,
                        precip_mode=precip_mode,
                        hourly_min_coverage=config.hourly_min_coverage)
        # The database is a checked-in deployment input, not a private raw
        # source. Make it readable by the account that packages the site.
        os.chmod(temporary_path, 0o644)
        os.replace(temporary_path, output)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)

    return {
        "database": str(output),
        "sha256": _sha256(output),
        "station_id": station.station_id,
        "daily_rows": len(daily),
        "first_date": pd.Timestamp(daily["date"].min()).date().isoformat(),
        "last_date": pd.Timestamp(daily["date"].max()).date().isoformat(),
        "valid_minimum_days": int(daily["tmin_c"].notna().sum()),
        "data_kind": REANALYSIS_DATA_KIND,
    }


def _readonly(path: Path) -> sqlite3.Connection:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"SQLite weather database does not exist: {path}")
    connection = sqlite3.connect(f"{path.as_uri()}?mode=ro", uri=True)
    connection.execute("PRAGMA query_only = ON")
    return connection


def load_daily_from_sqlite(path: Path, station: StationMeta) -> tuple[pd.DataFrame, str]:
    """Read and validate a configured station from the checked-in database."""
    with _readonly(path) as connection:
        application_id = connection.execute("PRAGMA application_id").fetchone()[0]
        user_version = connection.execute("PRAGMA user_version").fetchone()[0]
        if (application_id, user_version) != (DATABASE_APPLICATION_ID, DATABASE_SCHEMA_VERSION):
            raise ValueError("Unsupported IDWX SQLite database")
        metadata = dict(connection.execute("SELECT key, value FROM metadata"))
        if metadata.get("schema_version") != str(DATABASE_SCHEMA_VERSION):
            raise ValueError("SQLite metadata/schema version mismatch")
        if metadata.get("data_kind") != REANALYSIS_DATA_KIND:
            raise ValueError("Unsupported SQLite data kind")
        stored = connection.execute(
            "SELECT name, latitude, longitude, elevation_m, source, timezone "
            "FROM stations WHERE station_id = ?", (station.station_id,),
        ).fetchone()
        if stored is None:
            raise ValueError(f"Station {station.station_id} is not present in the SQLite database")
        expected = (station.name, _optional_float(station.lat), _optional_float(station.lon),
                    _optional_float(station.elevation_m), station.source,
                    station.timezone or "America/Boise")
        if stored != expected:
            raise ValueError(f"Configured metadata for {station.station_id} differs from the SQLite database")
        daily = pd.read_sql_query(
            "SELECT station_id, date, tmin_c, tmax_c, tmean_c, precip_mm, "
            "latitude AS lat, longitude AS lon, elevation_m, n_obs, quality_flags "
            "FROM daily_weather WHERE station_id = ? ORDER BY date",
            connection, params=(station.station_id,),
        )
        if daily.empty:
            raise ValueError(f"Station {station.station_id} has no daily rows in the SQLite database")
        daily["date"] = pd.to_datetime(daily["date"], errors="raise")
        return daily, REANALYSIS_DATA_KIND
