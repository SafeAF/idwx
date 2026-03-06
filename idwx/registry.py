from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from idwx.config import Config
from idwx.models.base import BaseModel


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
    except Exception:
        return "unknown"


def model_run_dir(config: Config, target: str, station: str, model_name: str, run_id: str | None = None) -> Path:
    rid = run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return config.models_dir / target / station / model_name / rid


def save_model_artifacts(
    model: BaseModel,
    config: Config,
    target: str,
    station: str,
    model_name: str,
    dataset: pd.DataFrame,
    metrics: dict[str, Any] | None = None,
    run_id: str | None = None,
) -> Path:
    out = model_run_dir(config, target=target, station=station, model_name=model_name, run_id=run_id)
    out.mkdir(parents=True, exist_ok=True)
    model.save(out)

    cfg_out = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit_hash": git_commit_hash(),
        "config_hash": hash_config(config),
        "dataset_hash": hash_dataset(dataset),
        "feature_schema_hash": hash_feature_schema([c for c in dataset.columns if c != "target_doy"]),
        "target": target,
        "station": station,
        "model_name": model_name,
    }
    with (out / "config.yml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg_out, f, sort_keys=True)

    with (out / "feature_schema.json").open("w", encoding="utf-8") as f:
        json.dump([c for c in dataset.columns if c != "target_doy"], f, indent=2)

    with (out / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics or {}, f, indent=2)

    return out


def latest_model_dir(config: Config, target: str, station: str, model_name: str) -> Path:
    base = config.models_dir / target / station / model_name
    if not base.exists():
        raise FileNotFoundError(f"No model directory: {base}")
    runs = sorted([p for p in base.iterdir() if p.is_dir()])
    if not runs:
        raise FileNotFoundError(f"No run directories in {base}")
    return runs[-1]
