from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from idwx.config import Config
from idwx.targets import build_targets_for_station


def _slope_vs_day(sub: pd.DataFrame, value_col: str) -> float:
    if sub.empty or sub[value_col].notna().sum() < 3:
        return np.nan
    x = sub["date"].dt.dayofyear.to_numpy(dtype=float)
    y = sub[value_col].to_numpy(dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3:
        return np.nan
    return float(np.polyfit(x[mask], y[mask], 1)[0])


def _q(s: pd.Series, p: float) -> float:
    return float(s.quantile(p)) if s.notna().any() else np.nan


def _target_series(targets: pd.DataFrame, target_name: str, threshold: float | None) -> pd.Series:
    t = targets[targets["target_name"] == target_name].copy()
    if threshold is not None and "threshold_c" in t.columns:
        t = t[t["threshold_c"].fillna(9999).round(4) == round(float(threshold), 4)]

    if target_name in {"freeze_free_days", "wsi", "gdd_total", "heat_days_32c", "heat_days_35c"}:
        vals = t[["season_year", "value_float"]].drop_duplicates("season_year").set_index("season_year")["value_float"]
    else:
        vals = t[["season_year", "value_doy"]].drop_duplicates("season_year").set_index("season_year")["value_doy"]
    return vals.sort_index()


def add_target_lags(features_df: pd.DataFrame, target_values: pd.DataFrame, target_col: str) -> pd.DataFrame:
    out = features_df.copy()
    y = target_values[["season_year", target_col]].drop_duplicates().sort_values("season_year")
    out = out.merge(y, on="season_year", how="left")
    out["target_lag1"] = out[target_col].shift(1)
    out["target_roll3"] = out[target_col].shift(1).rolling(3, min_periods=2).mean()
    out["target_roll5"] = out[target_col].shift(1).rolling(5, min_periods=3).mean()
    return out.drop(columns=[target_col])


def build_yearly_features(
    daily_df: pd.DataFrame,
    station_id: str,
    targets_df: pd.DataFrame,
    target_name: str,
    threshold: float | None,
    start_year: int | None = None,
    end_year: int | None = None,
) -> pd.DataFrame:
    df = daily_df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df[df["station_id"] == station_id].sort_values("date")
    years = sorted(df["date"].dt.year.dropna().unique().tolist())
    if start_year is not None and end_year is not None:
        years = [y for y in years if start_year <= y <= end_year]

    rows: list[dict] = []
    for y in years:
        row = {"station_id": station_id, "season_year": int(y), "year_index": int(y)}

        elev = pd.to_numeric(df.get("elevation_m", np.nan), errors="coerce").dropna()
        row["elevation_m"] = float(elev.iloc[0]) if not elev.empty else np.nan

        # First-fall oriented features.
        summer = df[(df["date"].dt.year == y) & (df["date"].dt.month.isin([6, 7, 8]))]
        row["summer_tmin_mean"] = float(summer["tmin_c"].mean()) if not summer.empty else np.nan
        row["summer_tmin_std"] = float(summer["tmin_c"].std(ddof=0)) if not summer.empty else np.nan
        row["summer_tmin_min"] = float(summer["tmin_c"].min()) if not summer.empty else np.nan
        row["summer_tmin_q10"] = _q(summer["tmin_c"], 0.10) if not summer.empty else np.nan
        row["summer_tmin_q90"] = _q(summer["tmin_c"], 0.90) if not summer.empty else np.nan

        aug_sep = df[(df["date"].dt.year == y) & (df["date"].dt.month.isin([8, 9]))]
        row["aug_sep_tmin_slope"] = _slope_vs_day(aug_sep, "tmin_c")
        if not aug_sep.empty:
            rolling_min = aug_sep["tmin_c"].rolling(window=7, min_periods=3).min()
            row["aug_sep_roll7min_q25"] = _q(rolling_min, 0.25)
            row["aug_sep_roll7min_q75"] = _q(rolling_min, 0.75)
        else:
            row["aug_sep_roll7min_q25"] = np.nan
            row["aug_sep_roll7min_q75"] = np.nan

        jan_to_dec = df[df["date"].dt.year == y]
        hit_10 = jan_to_dec[jan_to_dec["tmin_c"] <= 10.0]
        row["first_day_tmin_le_10c"] = float(hit_10["date"].dt.dayofyear.min()) if not hit_10.empty else np.nan
        sep = df[(df["date"].dt.year == y) & (df["date"].dt.month == 9)]
        row["sep_days_tmin_le_12c"] = float((sep["tmin_c"] <= 12.0).sum()) if not sep.empty else np.nan

        # Last-spring oriented features.
        winter = df[((df["date"].dt.year == y - 1) & (df["date"].dt.month == 12)) | ((df["date"].dt.year == y) & (df["date"].dt.month.isin([1, 2])))]
        row["winter_tmean"] = float(winter["tmean_c"].mean()) if not winter.empty else np.nan
        row["winter_tmin_min"] = float(winter["tmin_c"].min()) if not winter.empty else np.nan

        mar_apr = df[(df["date"].dt.year == y) & (df["date"].dt.month.isin([3, 4]))]
        row["mar_apr_tmin_slope"] = _slope_vs_day(mar_apr, "tmin_c")
        april = df[(df["date"].dt.year == y) & (df["date"].dt.month == 4)]
        row["april_nights_le_0c"] = float((april["tmin_c"] <= 0.0).sum()) if not april.empty else np.nan

        rows.append(row)

    feat = pd.DataFrame(rows).sort_values("season_year").reset_index(drop=True)
    if feat.empty:
        return feat

    # Generic lag/trend features.
    for col in ["summer_tmin_mean", "winter_tmean", "aug_sep_tmin_slope", "mar_apr_tmin_slope"]:
        feat[f"{col}_lag1"] = feat[col].shift(1)

    # Lag target features.
    y_series = _target_series(targets_df, target_name=target_name, threshold=threshold)
    y_frame = y_series.rename("target_doy").reset_index()
    feat = add_target_lags(feat, y_frame, "target_doy")

    # WSI lag feature.
    wsi_series = _target_series(targets_df, target_name="wsi", threshold=None)
    feat = feat.merge(wsi_series.rename("wsi").reset_index(), on="season_year", how="left")
    feat["wsi_lag1"] = feat["wsi"].shift(1)

    return feat


def _features_subdir(config: Config, target_name: str, threshold: float | None) -> Path:
    t = "none" if threshold is None else str(float(threshold)).replace(".", "p").replace("-", "m")
    return config.cache_dir / "features" / f"{target_name}_thr{t}"


def build_features_cache(
    config: Config,
    target_name: str,
    threshold: float | None,
    rebuild: bool = False,
) -> list[str]:
    daily_dir = config.cache_dir / "daily"
    target_dir = config.cache_dir / "targets"
    out_dir = _features_subdir(config, target_name=target_name, threshold=threshold)

    if not daily_dir.exists():
        raise FileNotFoundError(f"Daily cache dir does not exist: {daily_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)

    start_year = int(config.backtest.get("start_year", 1980))
    end_year = int(config.backtest.get("end_year", 2022))

    written: list[str] = []
    all_frames: list[pd.DataFrame] = []
    for p in sorted(daily_dir.glob("*.parquet")):
        sid = p.stem
        out_path = out_dir / f"{sid}.parquet"
        if out_path.exists() and not rebuild:
            all_frames.append(pd.read_parquet(out_path))
            continue

        daily = pd.read_parquet(p)
        target_path = target_dir / f"{sid}.parquet"
        if target_path.exists():
            targets = pd.read_parquet(target_path)
        else:
            targets = build_targets_for_station(daily, station_id=sid, cfg=config)

        feat = build_yearly_features(
            daily,
            station_id=sid,
            targets_df=targets,
            target_name=target_name,
            threshold=threshold,
            start_year=start_year,
            end_year=end_year,
        )

        # Feature-level imputation flags for model tolerance.
        for col in ["summer_tmin_mean", "winter_tmean", "target_lag1", "wsi_lag1"]:
            if col in feat.columns:
                feat[f"{col}_imputed"] = feat[col].isna().astype(int)
                feat[col] = feat[col].fillna(feat[col].median())

        feat.to_parquet(out_path, index=False)
        written.append(sid)
        all_frames.append(feat)

    if all_frames:
        pd.concat(all_frames, ignore_index=True).to_parquet(out_dir / "_all_features.parquet", index=False)

    return written


def load_feature_cache(config: Config, target_name: str, threshold: float | None) -> pd.DataFrame:
    out_dir = _features_subdir(config, target_name=target_name, threshold=threshold)
    all_path = out_dir / "_all_features.parquet"
    if all_path.exists():
        return pd.read_parquet(all_path)

    frames: list[pd.DataFrame] = []
    for p in sorted(out_dir.glob("*.parquet")):
        if p.name == "_all_features.parquet":
            continue
        frames.append(pd.read_parquet(p))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
