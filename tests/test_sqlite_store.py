import sqlite3
from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest

from idwx.config import StationMeta, load_config, load_stations
from idwx.sqlite_store import (
    DATABASE_APPLICATION_ID,
    DATABASE_SCHEMA_VERSION,
    REANALYSIS_DATA_KIND,
    build_sqlite_database,
    load_daily_from_sqlite,
    refresh_sqlite_database,
)


def _source(path: Path) -> Path:
    timestamps = pd.date_range("2020-01-01T00:00:00Z", periods=48, freq="h")
    frame = pd.DataFrame({
        "timestamp": timestamps,
        "temperature (degC)": range(48),
        "total_precipitation (mm of water equivalent)": 0.0,
        "utc_offset (hrs)": 0.0,
    })
    frame.to_csv(path, index=False)
    return path


def test_build_and_load_sqlite_daily_weather(tmp_path):
    station = StationMeta("twin_falls", "Twin Falls", 42.4634, -114.4975,
                          1318.38, "Test ERA5", "UTC")
    cfg = replace(load_config("configs/idaho.yml"), hourly_min_coverage=18,
                  database_path=tmp_path / "weather.sqlite3", timezone="UTC")
    source = _source(tmp_path / "42.4634--114.4975-source.csv")
    report = build_sqlite_database(cfg, station, source, cfg.database_path)
    assert report["daily_rows"] == report["valid_minimum_days"] == 2
    assert report["data_kind"] == REANALYSIS_DATA_KIND

    daily, data_kind = load_daily_from_sqlite(cfg.database_path, station)
    assert data_kind == REANALYSIS_DATA_KIND
    assert daily.date.tolist() == [pd.Timestamp("2020-01-01"), pd.Timestamp("2020-01-02")]
    assert daily.tmin_c.tolist() == [0.0, 24.0]
    assert daily.tmax_c.tolist() == [23.0, 47.0]
    assert daily.n_obs.tolist() == [24, 24]
    with sqlite3.connect(cfg.database_path) as connection:
        assert connection.execute("PRAGMA application_id").fetchone()[0] == DATABASE_APPLICATION_ID
        assert connection.execute("PRAGMA user_version").fetchone()[0] == DATABASE_SCHEMA_VERSION
        assert connection.execute("PRAGMA integrity_check").fetchone() == ("ok",)


def test_database_rejects_configured_station_metadata_drift(tmp_path):
    station = StationMeta("twin_falls", "Twin Falls", 42.4634, -114.4975,
                          1318.38, "Test ERA5", "UTC")
    cfg = replace(load_config("configs/idaho.yml"), database_path=tmp_path / "weather.sqlite3",
                  timezone="UTC")
    build_sqlite_database(cfg, station,
                          _source(tmp_path / "42.4634--114.4975-source.csv"),
                          cfg.database_path)
    with pytest.raises(ValueError, match="differs"):
        load_daily_from_sqlite(cfg.database_path, replace(station, lat=42.0))


def test_builder_will_not_replace_raw_source(tmp_path):
    station = StationMeta("twin_falls", "Twin Falls", 42.4634, -114.4975,
                          1318.38, "Test ERA5", "UTC")
    cfg = replace(load_config("configs/idaho.yml"), timezone="UTC")
    source = _source(tmp_path / "42.4634--114.4975-source.csv")
    original = source.read_bytes()
    with pytest.raises(ValueError, match="must not replace"):
        build_sqlite_database(cfg, station, source, source)
    assert source.read_bytes() == original


def test_refresh_upgrades_legacy_database_and_only_imports_complete_days(tmp_path):
    station = StationMeta("twin_falls", "Twin Falls", 42.4634, -114.4975,
                          1318.38, "Test ERA5", "UTC")
    cfg = replace(load_config("configs/idaho.yml"), hourly_min_coverage=18,
                  database_path=tmp_path / "weather.sqlite3", timezone="UTC")
    original = _source(tmp_path / "42.4634--114.4975-original.csv")
    build_sqlite_database(cfg, station, original, cfg.database_path)
    with sqlite3.connect(cfg.database_path) as connection:
        connection.execute("DROP TABLE source_refreshes")
        connection.execute("UPDATE metadata SET value = '1' WHERE key = 'schema_version'")
        connection.execute("PRAGMA user_version = 1")

    timestamps = pd.date_range("2020-01-02T12:00:00Z", periods=48, freq="h")
    refresh = pd.DataFrame({
        "timestamp": timestamps,
        "temperature (degC)": range(100, 148),
        "total_precipitation (mm of water equivalent)": 0.0,
        "utc_offset (hrs)": 0.0,
    })
    source = tmp_path / "42.4634--114.4975-oikolab.csv"
    refresh.to_csv(source, index=False)
    report = refresh_sqlite_database(
        cfg, station, source, cfg.database_path,
        requested_start_utc="2020-01-02T12:00:00+00:00",
        requested_end_utc="2020-01-04T11:00:00+00:00",
    )
    assert report["imported_rows"] == 1
    assert report["first_imported_date"] == report["last_imported_date"] == "2020-01-03"
    with sqlite3.connect(cfg.database_path) as connection:
        assert connection.execute("PRAGMA user_version").fetchone()[0] == DATABASE_SCHEMA_VERSION
        assert connection.execute("SELECT count(*) FROM source_refreshes").fetchone()[0] == 1
        # The two incomplete edge days do not replace existing complete data.
        assert connection.execute(
            "SELECT tmin_c FROM daily_weather WHERE date = '2020-01-02'"
        ).fetchone()[0] == 24.0
        assert connection.execute(
            "SELECT tmin_c FROM daily_weather WHERE date = '2020-01-03'"
        ).fetchone()[0] == 112.0


def test_checked_in_twin_falls_database_is_valid():
    cfg = load_config("configs/idaho.yml")
    station = load_stations(cfg.stations_file)[0]
    assert cfg.database_path == Path.cwd() / "data/idwx.sqlite3"
    daily, data_kind = load_daily_from_sqlite(cfg.database_path, station)
    assert data_kind == REANALYSIS_DATA_KIND
    assert len(daily) == 17_114
    assert daily.date.min() == pd.Timestamp("1979-12-31")
    assert daily.date.max() == pd.Timestamp("2026-11-07")
    assert daily.tmin_c.notna().sum() == 17_113
    assert set(daily.station_id) == {"twin_falls"}
