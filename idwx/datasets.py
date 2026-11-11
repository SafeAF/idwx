from __future__ import annotations

import numpy as np
import pandas as pd

from idwx.config import Config
from idwx.features import (
    build_features_cache,
    build_yearly_features,
    prior_target_statistics,
)
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

    thr_key = "none" if threshold is None else str(float(threshold)).replace(".", "p").replace("-", "m")
    feat_path = cfg.cache_dir / "features" / f"{target_name}_thr{thr_key}" / f"{station_id}.parquet"
    if feat_path.exists():
        feats = pd.read_parquet(feat_path)
    else:
        feats = build_yearly_features(
            daily_df,
            station_id=station_id,
            targets_df=targets,
            target_name=target_name,
            threshold=threshold,
            start_year=int(cfg.backtest.get("start_year", 1980)),
            end_year=int(cfg.backtest.get("end_year", 2022)),
        )
    merged = feats.merge(y, on=["station_id", "season_year"], how="left")

    # Legacy flags identify values needing imputation; never fit on all years.
    for c in ["summer_tmin_mean", "winter_tmean", "target_lag1"]:
        if c in merged.columns:
            merged[f"{c}_imputed"] = merged[c].isna().astype(int)

    return merged.sort_values("season_year").reset_index(drop=True)


def build_all_datasets(config: Config, target_name: str, threshold: float | None) -> pd.DataFrame:
    daily_dir = config.cache_dir / "daily"
    if not daily_dir.exists():
        raise FileNotFoundError(f"Daily cache dir does not exist: {daily_dir}")

    build_features_cache(config, target_name=target_name, threshold=threshold, rebuild=False)

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
    if h["station_id"].nunique(dropna=False) != 1 or h["season_year"].duplicated().any():
        raise ValueError("Future rows require a single station and unique season years")
    latest = h.iloc[-1].copy()
    year = int(latest["season_year"]) + 1
    # Only static metadata and explicit prior-year values may be carried forward.
    # Current-season weather and unknown feature columns remain unavailable.
    future = {col: np.nan for col in h.columns}
    for col in ("station_id", "elevation_m"):
        if col in h:
            future[col] = latest[col]
    future["season_year"] = year
    future["year_index"] = year

    # Update lag-driven features.
    if "target_doy" in h.columns:
        stats = prior_target_statistics(h.set_index("season_year")["target_doy"], pd.Series([year]))
        future.update(stats.iloc[0].to_dict())

    if "wsi" in h.columns:
        future["wsi_lag1"] = float(h.iloc[-1]["wsi"]) if pd.notna(h.iloc[-1]["wsi"]) else np.nan

    for col in h.columns:
        if col.endswith("_lag1") and col[:-5] in h and col != "target_lag1":
            future[col] = latest[col[:-5]]
    for col in h.columns:
        if col.endswith("_imputed") and col[:-8] in future:
            future[col] = int(pd.isna(future[col[:-8]]))

    future["target_doy"] = np.nan
    return pd.DataFrame([future])


def load_or_build_dataset(config: Config, target_name: str, threshold: float | None) -> pd.DataFrame:
    suffix = f"thr{threshold}" if threshold is not None else "none"
    path = config.cache_dir / "datasets" / f"{target_name}_{suffix}.parquet"
    if path.exists():
        return pd.read_parquet(path)
    return build_all_datasets(config, target_name=target_name, threshold=threshold)
