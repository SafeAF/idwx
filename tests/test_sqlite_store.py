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


def test_checked_in_twin_falls_database_is_valid():
    cfg = load_config("configs/idaho.yml")
    station = load_stations(cfg.stations_file)[0]
    assert cfg.database_path == Path.cwd() / "data/idwx.sqlite3"
    daily, data_kind = load_daily_from_sqlite(cfg.database_path, station)
    assert data_kind == REANALYSIS_DATA_KIND
    assert len(daily) == 15_707
    assert daily.date.min() == pd.Timestamp("1979-12-31")
    assert daily.date.max() == pd.Timestamp("2022-12-31")
    assert daily.tmin_c.notna().sum() == 15_705
    assert set(daily.station_id) == {"twin_falls"}
