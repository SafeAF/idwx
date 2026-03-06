from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class DayAccumulator:
    n_obs: int = 0
    temp_sum: float = 0.0
    temp_count: int = 0
    tmin: float = float("inf")
    tmax: float = float("-inf")
    precip_values: list[float] = field(default_factory=list)


def _is_monotonic_nondecreasing(values: list[float], eps: float = 1e-9) -> bool:
    if len(values) < 2:
        return False
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size < 2:
        return False
    return bool(np.all(np.diff(arr) >= -eps))


def detect_precip_mode(day_acc: dict[pd.Timestamp, DayAccumulator]) -> str:
    valid_days = 0
    monotonic_days = 0
    for acc in day_acc.values():
        vals = [v for v in acc.precip_values if np.isfinite(v)]
        if len(vals) < 2:
            continue
        valid_days += 1
        if _is_monotonic_nondecreasing(vals):
            monotonic_days += 1
    if valid_days == 0:
        return "incremental"
    return "cumulative" if (monotonic_days / valid_days) >= 0.7 else "incremental"


def finalize_daily(
    station_id: str,
    lat: float,
    lon: float,
    elevation_m: float | None,
    day_acc: dict[pd.Timestamp, DayAccumulator],
    min_coverage: int,
) -> tuple[pd.DataFrame, str]:
    precip_mode = detect_precip_mode(day_acc)
    rows: list[dict] = []
    for day, acc in sorted(day_acc.items()):
        has_coverage = acc.n_obs >= min_coverage
        if precip_mode == "cumulative":
            precip_valid = [v for v in acc.precip_values if np.isfinite(v)]
            precip = float(max(precip_valid) - min(precip_valid)) if len(precip_valid) >= 2 else np.nan
        else:
            precip = float(np.nansum(acc.precip_values)) if acc.precip_values else np.nan

        row = {
            "station_id": station_id,
            "date": pd.Timestamp(day),
            "tmin_c": float(acc.tmin) if has_coverage and np.isfinite(acc.tmin) else np.nan,
            "tmax_c": float(acc.tmax) if has_coverage and np.isfinite(acc.tmax) else np.nan,
            "tmean_c": float(acc.temp_sum / acc.temp_count) if has_coverage and acc.temp_count > 0 else np.nan,
            "precip_mm": float(precip) if has_coverage and np.isfinite(precip) else np.nan,
            "lat": float(lat),
            "lon": float(lon),
            "elevation_m": float(elevation_m) if elevation_m is not None and np.isfinite(elevation_m) else np.nan,
            "n_obs": int(acc.n_obs),
            "quality_flags": "" if has_coverage else "insufficient_coverage",
        }
        rows.append(row)

    out = pd.DataFrame(rows)
    if out.empty:
        out = pd.DataFrame(
            columns=[
                "station_id",
                "date",
                "tmin_c",
                "tmax_c",
                "tmean_c",
                "precip_mm",
                "lat",
                "lon",
                "elevation_m",
                "n_obs",
                "quality_flags",
            ]
        )
    return out, precip_mode
