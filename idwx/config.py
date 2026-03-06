from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class StationMeta:
    station_id: str
    name: str
    lat: float | None = None
    lon: float | None = None
    elevation_m: float | None = None
    source: str | None = None


@dataclass
class Config:
    data_root: Path
    cache_dir: Path
    models_dir: Path
    reports_dir: Path
    timezone: str
    stations_file: Path
    frost_thresholds_c: list[float]
    windows: dict[str, dict[str, str]]
    winter: dict[str, Any]
    gdd: dict[str, Any]
    backtest: dict[str, Any]
    models: dict[str, dict[str, Any]]
    hourly_min_coverage: int = 18


DEFAULT_WINDOWS = {
    "last_spring_frost": {"start": "01-01", "end": "06-30"},
    "first_fall_frost": {"start": "08-01", "end": "12-31"},
}


def _as_repo_path(path_str: str) -> Path:
    p = Path(path_str).expanduser()
    return p if p.is_absolute() else Path.cwd() / p


def load_config(path: str | Path) -> Config:
    cfg_path = Path(path)
    with cfg_path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    data_root = Path(raw.get("data_root", "~/historical_weather_data-1980-2022/idaho")).expanduser()
    cache_dir = _as_repo_path(raw.get("cache_dir", "data_cache"))
    models_dir = _as_repo_path(raw.get("models_dir", "models"))
    reports_dir = _as_repo_path(raw.get("reports_dir", "reports"))
    stations_file = _as_repo_path(raw.get("stations_file", "configs/stations.yml"))

    return Config(
        data_root=data_root,
        cache_dir=cache_dir,
        models_dir=models_dir,
        reports_dir=reports_dir,
        timezone=raw.get("timezone", "America/Boise"),
        stations_file=stations_file,
        frost_thresholds_c=[float(x) for x in raw.get("frost_thresholds_c", [0.0, -2.0])],
        windows=raw.get("windows", DEFAULT_WINDOWS),
        winter=raw.get("winter", {}),
        gdd=raw.get("gdd", {}),
        backtest=raw.get("backtest", {"start_year": 1980, "end_year": 2022, "min_train_years": 20}),
        models=raw.get("models", {}),
        hourly_min_coverage=int(raw.get("hourly_min_coverage", 18)),
    )


def load_stations(stations_file: str | Path) -> list[StationMeta]:
    p = Path(stations_file).expanduser()
    with p.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    stations_raw = raw.get("stations", [])
    stations: list[StationMeta] = []
    for row in stations_raw:
        sid = str(row["station_id"]).strip().lower()
        stations.append(
            StationMeta(
                station_id=sid,
                name=str(row.get("name", sid)),
                lat=row.get("lat"),
                lon=row.get("lon"),
                elevation_m=row.get("elevation_m"),
                source=row.get("source"),
            )
        )
    return stations
