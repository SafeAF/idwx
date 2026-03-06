from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from idwx.config import Config
from idwx.models import create_model


def _metrics(df: pd.DataFrame) -> dict[str, float]:
    e = df["pred_p50"] - df["actual"]
    mae = float(np.abs(e).mean()) if len(df) else float("nan")
    rmse = float(np.sqrt((e**2).mean())) if len(df) else float("nan")
    bias = float(e.mean()) if len(df) else float("nan")
    coverage = float(((df["actual"] >= df["pred_p10"]) & (df["actual"] <= df["pred_p90"])).mean()) if len(df) else float("nan")
    return {"mae": mae, "rmse": rmse, "bias": bias, "coverage": coverage}


def walk_forward_backtest(dataset: pd.DataFrame, model_name: str, cfg: Config) -> tuple[pd.DataFrame, dict[str, float], pd.DataFrame]:
    req = ["station_id", "season_year", "target_doy"]
    for c in req:
        if c not in dataset.columns:
            raise ValueError(f"Dataset missing required column: {c}")

    bt = cfg.backtest
    start_year = int(bt.get("start_year", int(dataset["season_year"].min())))
    end_year = int(bt.get("end_year", int(dataset["season_year"].max())))
    min_train_years = int(bt.get("min_train_years", 20))

    rows: list[dict] = []
    for sid, sdf in dataset.groupby("station_id"):
        sdf = sdf.sort_values("season_year")
        for t in range(start_year, end_year + 1):
            train = sdf[sdf["season_year"] <= (t - 1)].copy()
            test = sdf[sdf["season_year"] == t].copy()
            if test.empty:
                continue
            if train["target_doy"].notna().sum() < min_train_years:
                continue

            X_train = train.drop(columns=["target_doy"])
            y_train = train["target_doy"]
            X_test = test.drop(columns=["target_doy"])

            model = create_model(model_name, cfg.models)
            model.fit(X_train, y_train, meta=train[["station_id", "season_year"]], config=cfg.models)
            pred = model.predict(X_test, cfg.models)

            for i, (_, tr) in enumerate(test.iterrows()):
                rows.append(
                    {
                        "station_id": sid,
                        "season_year": int(tr["season_year"]),
                        "actual": float(tr["target_doy"]) if pd.notna(tr["target_doy"]) else np.nan,
                        "pred_p10": float(pred.iloc[i]["p10"]),
                        "pred_p50": float(pred.iloc[i]["p50"]),
                        "pred_p90": float(pred.iloc[i]["p90"]),
                        "model_name": model_name,
                    }
                )

    yearly = pd.DataFrame(rows).dropna(subset=["actual"])
    summary = _metrics(yearly)
    per_station = yearly.groupby("station_id", as_index=False).apply(lambda g: pd.Series(_metrics(g))).reset_index()
    if "level_1" in per_station.columns:
        per_station = per_station.drop(columns=["level_1"])
    return yearly, summary, per_station


def write_eval_reports(
    yearly: pd.DataFrame,
    summary: dict[str, float],
    per_station: pd.DataFrame,
    config: Config,
    target: str,
    model_name: str,
    baseline_summary: dict[str, float] | None = None,
) -> Path:
    out = config.reports_dir / target / model_name
    out.mkdir(parents=True, exist_ok=True)

    summary_df = pd.DataFrame([summary])
    if baseline_summary:
        summary_df["baseline_mae"] = baseline_summary.get("mae")
        summary_df["delta_mae"] = summary_df["mae"] - summary_df["baseline_mae"]

    summary_df.to_csv(out / "summary.csv", index=False)
    per_station.to_csv(out / "per_station.csv", index=False)
    yearly.to_csv(out / "yearly_errors.csv", index=False)
    (out / "plots").mkdir(exist_ok=True)
    return out
