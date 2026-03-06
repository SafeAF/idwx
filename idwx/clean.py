from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd


COLUMN_ALIASES = {
    "timestamp": ["timestamp", "datetime", "date_time", "time", "valid"],
    "date": ["date", "day"],
    "station_id": ["station_id", "station", "site_id", "stid"],
    "tmin": ["tmin", "temp_min", "min_temp", "minimum temperature"],
    "tmax": ["tmax", "temp_max", "max_temp", "maximum temperature"],
    "tmean": ["tmean", "temp_mean", "avg_temp", "temperature"],
    "temp": ["temp", "air_temp", "temperature", "tmp"],
    "precip": ["precip", "precipitation", "rain", "prcp"],
    "wind": ["wind", "windspeed", "wind_speed"],
    "rh": ["rh", "humidity", "relative_humidity"],
    "snow": ["snow", "snowfall"],
    "solar": ["solar", "radiation", "shortwave"],
    "pressure": ["pressure", "pres", "barometer"],
}


def canonicalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename_map: dict[str, str] = {}
    lower_cols = {c: c.lower().strip() for c in df.columns}

    for original, low in lower_cols.items():
        for target, aliases in COLUMN_ALIASES.items():
            if low == target or any(a == low for a in aliases):
                rename_map[original] = target
                break

    out = df.rename(columns=rename_map).copy()
    out.columns = [c.lower().strip() for c in out.columns]
    return out


def f_to_c(series: pd.Series) -> pd.Series:
    return (series - 32.0) * (5.0 / 9.0)


def in_to_mm(series: pd.Series) -> pd.Series:
    return series * 25.4


def mph_to_mps(series: pd.Series) -> pd.Series:
    return series * 0.44704


def ensure_numeric(df: pd.DataFrame, cols: Iterable[str]) -> pd.DataFrame:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def normalize_units(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    for c in list(out.columns):
        if c.endswith("_f"):
            base = c[:-2]
            out[base + "_c"] = f_to_c(pd.to_numeric(out[c], errors="coerce"))
        if c.endswith("_in"):
            base = c[:-3]
            out[base + "_mm"] = in_to_mm(pd.to_numeric(out[c], errors="coerce"))
        if c.endswith("_mph"):
            base = c[:-4]
            out[base + "_mps"] = mph_to_mps(pd.to_numeric(out[c], errors="coerce"))

    # Heuristic fallback for legacy ambiguous columns.
    if "tmin" in out.columns and "tmin_c" not in out.columns:
        s = pd.to_numeric(out["tmin"], errors="coerce")
        out["tmin_c"] = f_to_c(s) if s.dropna().median() > 45 else s
    if "tmax" in out.columns and "tmax_c" not in out.columns:
        s = pd.to_numeric(out["tmax"], errors="coerce")
        out["tmax_c"] = f_to_c(s) if s.dropna().median() > 45 else s
    if "tmean" in out.columns and "tmean_c" not in out.columns:
        s = pd.to_numeric(out["tmean"], errors="coerce")
        out["tmean_c"] = f_to_c(s) if s.dropna().median() > 45 else s
    if "temp" in out.columns and "temp_c" not in out.columns:
        s = pd.to_numeric(out["temp"], errors="coerce")
        out["temp_c"] = f_to_c(s) if s.dropna().median() > 45 else s

    if "precip" in out.columns and "precip_mm" not in out.columns:
        s = pd.to_numeric(out["precip"], errors="coerce")
        out["precip_mm"] = in_to_mm(s) if s.dropna().quantile(0.99) < 5 else s
    if "wind" in out.columns and "wind_mps" not in out.columns:
        s = pd.to_numeric(out["wind"], errors="coerce")
        out["wind_mps"] = mph_to_mps(s) if s.dropna().median() > 20 else s

    return out


def standardize_daily_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    keep = [
        "station_id",
        "date",
        "tmin_c",
        "tmax_c",
        "tmean_c",
        "precip_mm",
        "wind_mps",
        "rh_pct",
        "snow_mm",
        "solar_wm2",
        "pressure_hpa",
    ]
    for c in keep:
        if c not in out.columns:
            out[c] = np.nan

    if out["tmean_c"].isna().all() and ("tmin_c" in out.columns and "tmax_c" in out.columns):
        out["tmean_c"] = (out["tmin_c"] + out["tmax_c"]) / 2.0

    return out[keep]
