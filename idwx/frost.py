"""Conservative historical freeze events from local calendar-day minima."""
from __future__ import annotations

import numpy as np
import pandas as pd

from idwx.config import Config


def validated_daily(daily: pd.DataFrame, station_id: str) -> pd.DataFrame:
    df = daily.loc[daily["station_id"] == station_id].copy()
    df["date"] = pd.to_datetime(df["date"], errors="raise")
    if df["date"].isna().any() or df["date"].dt.tz is not None:
        raise ValueError("Daily dates must be nonmissing, timezone-naive local dates")
    if not df["date"].eq(df["date"].dt.normalize()).all():
        raise ValueError("Daily dates must not contain times")
    if df["date"].duplicated().any():
        raise ValueError(f"Duplicate daily dates for {station_id}; resolve before calculating frost")
    df["tmin_c"] = pd.to_numeric(df["tmin_c"], errors="coerce")
    df.loc[~np.isfinite(df["tmin_c"]), "tmin_c"] = np.nan
    if "quality_flags" in df:
        invalid = df["quality_flags"].fillna("").str.contains("insufficient_coverage", regex=False)
        df.loc[invalid, "tmin_c"] = np.nan
    return df.sort_values("date").set_index("date", drop=False)


def window_event(df: pd.DataFrame, year: int, threshold: float, window: dict, last: bool) -> dict:
    start, end = (pd.Timestamp(f"{year}-{window[key]}") for key in ("start", "end"))
    if end < start:
        raise ValueError("Frost windows must start and end within the same calendar year")
    expected = pd.date_range(start, end, freq="D")
    minima = df["tmin_c"].reindex(expected)
    valid = int(minima.notna().sum())
    events = minima[minima <= threshold]
    candidate = events.index[-1 if last else 0] if not events.empty else pd.NaT
    status = (
        "insufficient_data" if valid == 0 else
        "incomplete_window" if valid < len(expected) else
        "no_event_in_complete_window" if events.empty else "observed"
    )
    confirmed = candidate if status == "observed" else pd.NaT
    return {
        "value_date": confirmed,
        "value_doy": float(confirmed.day_of_year) if pd.notna(confirmed) else np.nan,
        "value_float": np.nan,
        "status": status,
        "quality_flags": "" if status == "observed" else status,
        "window_start": start,
        "window_end": end,
        "expected_days": len(expected),
        "valid_days": valid,
        "missing_days": len(expected) - valid,
        "coverage_fraction": valid / len(expected),
        "candidate_date": candidate,
        "candidate_tmin_c": float(minima.loc[candidate]) if pd.notna(candidate) else np.nan,
        "event_tmin_c": float(minima.loc[confirmed]) if pd.notna(confirmed) else np.nan,
    }


def frost_rows(df: pd.DataFrame, station_id: str, year: int, cfg: Config) -> list[dict]:
    windows = {**cfg.windows, **cfg.station_windows.get(station_id, {})}
    rows = []
    for threshold in cfg.frost_thresholds_c:
        if not np.isfinite(threshold):
            raise ValueError("Frost thresholds must be finite Celsius values")
        base = {"station_id": station_id, "season_year": year, "threshold_c": float(threshold)}
        spring = window_event(df, year, threshold, windows["last_spring_frost"], last=True)
        fall = window_event(df, year, threshold, windows["first_fall_frost"], last=False)
        for name, event in (("last_spring_frost", spring), ("first_fall_frost", fall)):
            rows.append({**base, "target_name": name, **event})

        interval = np.nan
        intervening = np.nan
        flags = "missing_dependency"
        summer = df.loc[(df.index.year == year) & (df.index.month == 7), "tmin_c"]
        summer_freeze = bool((summer <= threshold).any())
        if spring["status"] == fall["status"] == "observed":
            start, end = spring["value_date"], fall["value_date"]
            if end > start:
                interval = float((end - start).days)
                dates = pd.date_range(start + pd.Timedelta(days=1), end - pd.Timedelta(days=1))
                middle = df["tmin_c"].reindex(dates)
                if middle.isna().any():
                    flags = "incomplete_interval"
                elif (middle <= threshold).any():
                    flags = "intervening_freeze"
                else:
                    intervening = float(len(dates))
                    flags = ""
            else:
                flags = "invalid_event_order"
        rows.append({
            **base, "target_name": "freeze_free_days", "value_date": pd.NaT,
            "value_doy": np.nan, "value_float": interval,
            "interval_definition": "fall_date_minus_spring_date",
            "frost_free_days": intervening, "summer_freeze": summer_freeze,
            "status": "verified" if not flags else flags, "quality_flags": flags,
        })
    return rows
