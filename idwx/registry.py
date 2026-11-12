from __future__ import annotations

import hashlib
import json
import math
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

import pandas as pd
import yaml

from idwx.config import Config
from idwx.models.base import BaseModel
from idwx.models.conformal import CALIBRATION_VERSION

ARTIFACT_VERSION = 2


def threshold_key(threshold_c: float | None) -> str:
    if threshold_c is None:
        return "threshold_none"
    value = float(threshold_c)
    if not math.isfinite(value):
        raise ValueError("Model threshold must be finite")
    value = 0.0 if value == 0 else value
    return "threshold_" + str(value).replace("-", "m").replace("+", "p").replace(".", "p")


def _identifier(value: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9_-]+", value):
        raise ValueError(f"Invalid artifact identifier: {value!r}")
    return value

def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def hash_config(cfg: Config) -> str:
    payload = {
        "data_root": str(cfg.data_root),
        "cache_dir": str(cfg.cache_dir),
        "models_dir": str(cfg.models_dir),
        "reports_dir": str(cfg.reports_dir),
        "timezone": cfg.timezone,
        "frost_thresholds_c": cfg.frost_thresholds_c,
        "windows": cfg.windows,
        "station_windows": cfg.station_windows,
        "stations_sha256": _sha256_bytes(cfg.stations_file.read_bytes()),
        "winter": cfg.winter,
        "gdd": cfg.gdd,
        "backtest": cfg.backtest,
        "models": cfg.models,
        "hourly_min_coverage": cfg.hourly_min_coverage,
    }
    return _sha256_bytes(json.dumps(payload, sort_keys=True).encode("utf-8"))


def hash_dataset(df: pd.DataFrame) -> str:
    serial = df.sort_values(list(df.columns)).to_csv(index=False).encode("utf-8")
    return _sha256_bytes(serial)


def hash_feature_schema(columns: list[str]) -> str:
    return _sha256_bytes(json.dumps(sorted(columns)).encode("utf-8"))


def git_commit_hash() -> str:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
            .decode("utf-8")
            .strip()
        )
    except (OSError, subprocess.SubprocessError, UnicodeDecodeError):
        return "unknown"


def model_run_dir(
    config: Config, target: str, station: str, model_name: str,
    *, threshold_c: float | None, run_id: str | None = None, forecast_context: dict | None = None,
) -> Path:
    rid = run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + "_" + uuid4().hex
    base = (config.models_dir / _identifier(target) / _identifier(station)
            / _identifier(model_name) / threshold_key(threshold_c))
    if forecast_context is not None:
        base = base / _identifier(forecast_context["protocol_id"]) / _identifier(f"season_{forecast_context['season_year']}")
    return base / _identifier(rid)


def save_model_artifacts(
    model: BaseModel,
    config: Config,
    target: str,
    station: str,
    model_name: str,
    dataset: pd.DataFrame,
    threshold_c: float | None,
    metrics: dict[str, Any] | None = None,
    run_id: str | None = None,
    forecast_context: dict | None = None,
) -> Path:
    out = model_run_dir(config, target=target, station=station, model_name=model_name,
                        threshold_c=threshold_c, run_id=run_id, forecast_context=forecast_context)
    out.mkdir(parents=True, exist_ok=False)
    model.save(out)

    cfg_out = {
        "artifact_version": ARTIFACT_VERSION,
        "validation_status": "unvalidated_research",
        "forecast_context": forecast_context,
        "input_provenance": dataset.attrs if forecast_context is not None else None,
        "rf_calibration_version": CALIBRATION_VERSION if model_name == "rf" else None,
        "threshold_c": threshold_c,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit_hash": git_commit_hash(),
        "config_hash": hash_config(config),
        "dataset_hash": hash_dataset(dataset),
        "feature_schema_hash": hash_feature_schema([c for c in dataset.columns if c != "target_doy"]),
        "target": target,
        "station": station,
        "model_name": model_name,
        "training_start_year": int(dataset["season_year"].min()),
        "training_end_year": int(dataset["season_year"].max()),
    }

    with (out / "feature_schema.json").open("w", encoding="utf-8") as f:
        json.dump([c for c in dataset.columns if c != "target_doy"], f, indent=2)

    with (out / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics or {}, f, indent=2)

    # Completion marker comes last; interrupted runs are not eligible for lookup.
    with (out / ".config.yml.tmp").open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg_out, f, sort_keys=True)
    (out / ".config.yml.tmp").replace(out / "config.yml")

    return out


def latest_model_dir(
    config: Config, target: str, station: str, model_name: str,
    *, threshold_c: float | None, dataset: pd.DataFrame | None = None, forecast_context: dict | None = None,
) -> Path:
    base = model_run_dir(config, target, station, model_name, threshold_c=threshold_c,
                         run_id="lookup", forecast_context=forecast_context).parent
    if not base.exists():
        raise FileNotFoundError(f"No threshold-specific model directory. Retrain legacy models. Expected: {base}")
    runs = sorted(p for p in base.iterdir() if p.is_dir() and (p / "config.yml").is_file())
    if not runs:
        raise FileNotFoundError(f"No run directories in {base}")
    latest = runs[-1]
    try:
        with (latest / "config.yml").open(encoding="utf-8") as f:
            metadata = yaml.safe_load(f)
    except (OSError, yaml.YAMLError) as exc:
        raise ValueError(f"Unreadable model metadata: {latest}. Retrain this model.") from exc
    expected = {"artifact_version": ARTIFACT_VERSION, "target": target, "station": station,
                "model_name": model_name, "threshold_c": threshold_c, "config_hash": hash_config(config),
                "forecast_context": forecast_context}
    if model_name == "rf":
        expected["rf_calibration_version"] = CALIBRATION_VERSION
    if forecast_context is not None:
        if dataset is None:
            raise ValueError("Protocol model lookup requires the training dataset")
        expected["input_provenance"] = dataset.attrs
    if dataset is not None:
        expected["dataset_hash"] = hash_dataset(dataset)
        expected["feature_schema_hash"] = hash_feature_schema([c for c in dataset if c != "target_doy"])
    if not isinstance(metadata, dict) or any(metadata.get(key) != value for key, value in expected.items()):
        raise ValueError(f"Incompatible model artifact: {latest}. Retrain for the current configuration and dataset.")
    return latest
