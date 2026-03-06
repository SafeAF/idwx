from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from idwx.config import Config
from idwx.features import add_target_lags, build_yearly_features
from idwx.targets import build_targets_for_station


def _target_doy_table(targets: pd.DataFrame, target_name: str, threshold: float | None) -> pd.DataFrame:
    t = targets[targets["target_name"] == target_name].copy()
    if threshold is not None and "threshold_c" in t.columns:
        t = t[t["threshold_c"].fillna(9999).round(4) == round(float(threshold), 4)]

    t["target_doy"] = t["value_doy"]
    if target_name in {"freeze_free_days", "wsi", "gdd_total", "heat_days_32c", "heat_days_35c"}:
        t["target_doy"] = t["value_float"]
    return t[["station_id", "season_year", "target_doy"]]


def build_station_dataset(
    daily_df: pd.DataFrame,
    station_id: str,
    cfg: Config,
    target_name: str,
    threshold: float | None,
) -> pd.DataFrame:
    target_path = cfg.cache_dir / "targets" / f"{station_id}.parquet"
    if target_path.exists():
        targets = pd.read_parquet(target_path)
    else:
        targets = build_targets_for_station(daily_df, station_id=station_id, cfg=cfg)
    y = _target_doy_table(targets, target_name=target_name, threshold=threshold)

    feats = build_yearly_features(daily_df, station_id=station_id, target_name=target_name)
    merged = feats.merge(y, on=["station_id", "season_year"], how="left")
    merged = add_target_lags(merged, merged[["season_year", "target_doy"]], "target_doy")

    # Attach lagged WSI, if available.
    wsi = _target_doy_table(targets, target_name="wsi", threshold=None)
    wsi = wsi.rename(columns={"target_doy": "wsi"}).sort_values("season_year")
    merged = merged.merge(wsi, on=["station_id", "season_year"], how="left")
    merged["wsi_lag1"] = merged["wsi"].shift(1)

    # Simple feature-imputation flags as required by contract.
    for c in ["summer_tmin_mean", "winter_tmean", "target_lag1"]:
        if c in merged.columns:
            merged[f"{c}_imputed"] = merged[c].isna().astype(int)
            merged[c] = merged[c].fillna(merged[c].median())

    return merged.sort_values("season_year").reset_index(drop=True)


def build_all_datasets(config: Config, target_name: str, threshold: float | None) -> pd.DataFrame:
    daily_dir = config.cache_dir / "daily"
    if not daily_dir.exists():
        raise FileNotFoundError(f"Daily cache dir does not exist: {daily_dir}")

    all_frames: list[pd.DataFrame] = []
    for p in sorted(daily_dir.glob("*.parquet")):
        sid = p.stem
        daily = pd.read_parquet(p)
        ds = build_station_dataset(daily, station_id=sid, cfg=config, target_name=target_name, threshold=threshold)
        all_frames.append(ds)

    if not all_frames:
        return pd.DataFrame()

    out = pd.concat(all_frames, ignore_index=True)
    datasets_dir = config.cache_dir / "datasets"
    datasets_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"thr{threshold}" if threshold is not None else "none"
    out_path = datasets_dir / f"{target_name}_{suffix}.parquet"
    out.to_parquet(out_path, index=False)
    return out


def build_future_row(history_df: pd.DataFrame) -> pd.DataFrame:
    if history_df.empty:
        raise ValueError("Cannot build future row from empty history")

    h = history_df.sort_values("season_year").reset_index(drop=True)
    latest = h.iloc[-1].copy()
    future = latest.copy()
    future["season_year"] = int(latest["season_year"]) + 1
    future["year_index"] = int(future["season_year"])

    # Update lag-driven features.
    if "target_doy" in h.columns:
        future["target_lag1"] = float(h.iloc[-1]["target_doy"]) if pd.notna(h.iloc[-1]["target_doy"]) else np.nan
        future["target_roll3"] = float(h["target_doy"].tail(3).mean())
        future["target_roll5"] = float(h["target_doy"].tail(5).mean())

    if "wsi" in h.columns:
        future["wsi_lag1"] = float(h.iloc[-1]["wsi"]) if pd.notna(h.iloc[-1]["wsi"]) else np.nan

    future["target_doy"] = np.nan
    return pd.DataFrame([future])


def load_or_build_dataset(config: Config, target_name: str, threshold: float | None) -> pd.DataFrame:
    suffix = f"thr{threshold}" if threshold is not None else "none"
    path = config.cache_dir / "datasets" / f"{target_name}_{suffix}.parquet"
    if path.exists():
        return pd.read_parquet(path)
    return build_all_datasets(config, target_name=target_name, threshold=threshold)
