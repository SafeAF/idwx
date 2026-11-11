from __future__ import annotations

import numpy as np
import pandas as pd

from idwx.config import Config
from idwx.frost import frost_rows, validated_daily


def _cold_snap_max_run(mask: pd.Series) -> int:
    max_run = 0
    current = 0
    for v in mask.fillna(False):
        if bool(v):
            current += 1
            max_run = max(max_run, current)
        else:
            current = 0
    return int(max_run)


def build_targets_for_station(daily_df: pd.DataFrame, station_id: str, cfg: Config) -> pd.DataFrame:
    df = validated_daily(daily_df, station_id).reset_index(drop=True)
    observed_years = df["date"].dt.year
    start_year = cfg.backtest.get("start_year", int(observed_years.min()) if not df.empty else None)
    end_year = cfg.backtest.get("end_year", int(observed_years.max()) if not df.empty else None)
    if start_year is None or end_year is None:
        raise ValueError("An explicit year range is required when daily data is empty")
    if int(end_year) < int(start_year):
        raise ValueError("end_year must be at least start_year")
    years = range(int(start_year), int(end_year) + 1)
    rows: list[dict] = []

    for y in years:
        rows.extend(frost_rows(df.set_index('date', drop=False), station_id, y, cfg))

        winter_months = cfg.winter.get("season_months", [12, 1, 2])
        winter = df[
            ((df["date"].dt.year == y - 1) & (df["date"].dt.month.isin([m for m in winter_months if m == 12])))
            | ((df["date"].dt.year == y) & (df["date"].dt.month.isin([m for m in winter_months if m in (1, 2)])))
        ].copy()

        if not winter.empty:
            base = float(cfg.winter.get("hdd_base_c", 18.0))
            cold_thr = float(cfg.winter.get("cold_snap_threshold_c", -10.0))
            hdd = (base - winter["tmean_c"]).clip(lower=0).sum(min_count=1)
            tmin_extreme = winter["tmin_c"].min()
            cold_snap_days = (winter["tmin_c"] <= cold_thr).sum()
            cold_snap_max_run = _cold_snap_max_run(winter["tmin_c"] <= cold_thr)
            precip = winter["precip_mm"].sum(min_count=1) if "precip_mm" in winter.columns else np.nan

            rows.extend(
                [
                    {
                        "station_id": station_id,
                        "season_year": int(y),
                        "threshold_c": np.nan,
                        "target_name": "winter_hdd",
                        "value_date": pd.NaT,
                        "value_doy": np.nan,
                        "value_float": float(hdd) if pd.notna(hdd) else np.nan,
                        "quality_flags": "",
                    },
                    {
                        "station_id": station_id,
                        "season_year": int(y),
                        "threshold_c": np.nan,
                        "target_name": "winter_tmin_extreme",
                        "value_date": pd.NaT,
                        "value_doy": np.nan,
                        "value_float": float(tmin_extreme) if pd.notna(tmin_extreme) else np.nan,
                        "quality_flags": "",
                    },
                    {
                        "station_id": station_id,
                        "season_year": int(y),
                        "threshold_c": np.nan,
                        "target_name": "winter_cold_snap_days",
                        "value_date": pd.NaT,
                        "value_doy": np.nan,
                        "value_float": float(cold_snap_days),
                        "quality_flags": "",
                    },
                    {
                        "station_id": station_id,
                        "season_year": int(y),
                        "threshold_c": np.nan,
                        "target_name": "winter_cold_snap_max_run",
                        "value_date": pd.NaT,
                        "value_doy": np.nan,
                        "value_float": float(cold_snap_max_run),
                        "quality_flags": "",
                    },
                    {
                        "station_id": station_id,
                        "season_year": int(y),
                        "threshold_c": np.nan,
                        "target_name": "winter_precip_mm",
                        "value_date": pd.NaT,
                        "value_doy": np.nan,
                        "value_float": float(precip) if pd.notna(precip) else np.nan,
                        "quality_flags": "",
                    },
                ]
            )

        grow = df[
            (df["date"] >= pd.Timestamp(f"{y}-{cfg.gdd.get('season_start', '04-01')}"))
            & (df["date"] <= pd.Timestamp(f"{y}-{cfg.gdd.get('season_end', '10-31')}"))
        ]
        if not grow.empty:
            gdd_base = float(cfg.gdd.get("base_c", 10.0))
            tmean = grow["tmean_c"].copy()
            cap = cfg.gdd.get("upper_cap_c")
            if cap is not None:
                tmean = tmean.clip(upper=float(cap))
            gdd_total = (tmean - gdd_base).clip(lower=0).sum(min_count=1)

            rows.extend(
                [
                    {
                        "station_id": station_id,
                        "season_year": int(y),
                        "threshold_c": np.nan,
                        "target_name": "gdd_total",
                        "value_date": pd.NaT,
                        "value_doy": np.nan,
                        "value_float": float(gdd_total) if pd.notna(gdd_total) else np.nan,
                        "quality_flags": "",
                    },
                    {
                        "station_id": station_id,
                        "season_year": int(y),
                        "threshold_c": np.nan,
                        "target_name": "heat_days_32c",
                        "value_date": pd.NaT,
                        "value_doy": np.nan,
                        "value_float": float((grow["tmax_c"] >= 32.0).sum()),
                        "quality_flags": "",
                    },
                    {
                        "station_id": station_id,
                        "season_year": int(y),
                        "threshold_c": np.nan,
                        "target_name": "heat_days_35c",
                        "value_date": pd.NaT,
                        "value_doy": np.nan,
                        "value_float": float((grow["tmax_c"] >= 35.0).sum()),
                        "quality_flags": "",
                    },
                ]
            )

    out = pd.DataFrame(rows)

    # Station-level normalization and WSI composition.
    component_names = [
        "winter_hdd",
        "winter_tmin_extreme",
        "winter_cold_snap_days",
        "winter_cold_snap_max_run",
        "winter_precip_mm",
    ]
    comp = out[out["target_name"].isin(component_names)].copy()
    if not comp.empty:
        p = comp.pivot_table(index=["station_id", "season_year"], columns="target_name", values="value_float", aggfunc="first")
        z = (p - p.mean()) / p.std(ddof=0)
        z = z.replace([np.inf, -np.inf], np.nan)

        weights = cfg.winter.get("weights", {"hdd": 0.40, "tmin_extreme": 0.25, "cold_snap_days": 0.20, "cold_snap_run": 0.15})
        include_precip = bool(cfg.winter.get("include_precip_if_available", True)) and "winter_precip_mm" in z.columns
        wsi = (
            z.get("winter_hdd", 0) * float(weights.get("hdd", 0.40))
            + z.get("winter_tmin_extreme", 0) * float(weights.get("tmin_extreme", 0.25))
            + z.get("winter_cold_snap_days", 0) * float(weights.get("cold_snap_days", 0.20))
            + z.get("winter_cold_snap_max_run", 0) * float(weights.get("cold_snap_run", 0.15))
        )
        if include_precip:
            wsi = wsi + z["winter_precip_mm"] * float(weights.get("precip", 0.0))

        rows2 = []
        for col in z.columns:
            for (sid, sy), val in z[col].items():
                rows2.append(
                    {
                        "station_id": sid,
                        "season_year": int(sy),
                        "threshold_c": np.nan,
                        "target_name": f"{col}_z",
                        "value_date": pd.NaT,
                        "value_doy": np.nan,
                        "value_float": float(val) if pd.notna(val) else np.nan,
                        "quality_flags": "",
                    }
                )
        for (sid, sy), val in wsi.items():
            rows2.append(
                {
                    "station_id": sid,
                    "season_year": int(sy),
                    "threshold_c": np.nan,
                    "target_name": "wsi",
                    "value_date": pd.NaT,
                    "value_doy": np.nan,
                    "value_float": float(val) if pd.notna(val) else np.nan,
                    "quality_flags": "",
                }
            )
        if rows2:
            out = pd.concat([out, pd.DataFrame(rows2)], ignore_index=True)

    return out.sort_values(["station_id", "season_year", "target_name", "threshold_c"], na_position="last").reset_index(drop=True)


def build_targets_cache(config: Config, rebuild: bool = False) -> list[str]:
    daily_dir = config.cache_dir / "daily"
    targets_dir = config.cache_dir / "targets"
    if not daily_dir.exists():
        raise FileNotFoundError(f"Daily cache dir does not exist: {daily_dir}")
    targets_dir.mkdir(parents=True, exist_ok=True)

    written: list[str] = []
    all_frames: list[pd.DataFrame] = []
    for p in sorted(daily_dir.glob("*.parquet")):
        sid = p.stem
        out_path = targets_dir / f"{sid}.parquet"
        if out_path.exists() and not rebuild:
            df_existing = pd.read_parquet(out_path)
            all_frames.append(df_existing)
            continue

        daily = pd.read_parquet(p)
        targets = build_targets_for_station(daily, station_id=sid, cfg=config)
        targets.to_parquet(out_path, index=False)
        written.append(sid)
        all_frames.append(targets)

    if all_frames:
        pd.concat(all_frames, ignore_index=True).to_parquet(targets_dir / "_all_targets.parquet", index=False)
    return written
