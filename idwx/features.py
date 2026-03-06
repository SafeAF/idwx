from __future__ import annotations

import numpy as np
import pandas as pd


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


def build_yearly_features(daily_df: pd.DataFrame, station_id: str, target_name: str) -> pd.DataFrame:
    df = daily_df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df[df["station_id"] == station_id].sort_values("date")
    years = sorted(df["date"].dt.year.dropna().unique().tolist())

    rows: list[dict] = []
    for y in years:
        row = {"station_id": station_id, "season_year": int(y), "year_index": int(y)}

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

        winter = df[((df["date"].dt.year == y - 1) & (df["date"].dt.month == 12)) | ((df["date"].dt.year == y) & (df["date"].dt.month.isin([1, 2])))]
        row["winter_tmean"] = float(winter["tmean_c"].mean()) if not winter.empty else np.nan
        row["winter_tmin_min"] = float(winter["tmin_c"].min()) if not winter.empty else np.nan

        mar_apr = df[(df["date"].dt.year == y) & (df["date"].dt.month.isin([3, 4]))]
        row["mar_apr_tmin_slope"] = _slope_vs_day(mar_apr, "tmin_c")
        april = df[(df["date"].dt.year == y) & (df["date"].dt.month == 4)]
        row["april_nights_le_0c"] = float((april["tmin_c"] <= 0).sum()) if not april.empty else np.nan

        rows.append(row)

    feat = pd.DataFrame(rows).sort_values("season_year").reset_index(drop=True)
    if feat.empty:
        return feat

    # Lag and rolling trend features.
    for col in ["summer_tmin_mean", "winter_tmean", "aug_sep_tmin_slope"]:
        feat[f"{col}_lag1"] = feat[col].shift(1)

    return feat


def add_target_lags(features_df: pd.DataFrame, target_values: pd.DataFrame, target_col: str) -> pd.DataFrame:
    out = features_df.copy()
    y = target_values[["season_year", target_col]].drop_duplicates().sort_values("season_year")
    out = out.merge(y, on="season_year", how="left")
    out["target_lag1"] = out[target_col].shift(1)
    out["target_roll3"] = out[target_col].shift(1).rolling(3, min_periods=2).mean()
    out["target_roll5"] = out[target_col].shift(1).rolling(5, min_periods=3).mean()
    return out.drop(columns=[target_col])
