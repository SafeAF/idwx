from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from idwx.config import Config
from idwx.models import create_model
from idwx.registry import threshold_key

YEARLY_COLUMNS = [
    "station_id", "season_year", "actual", "pred_p10", "pred_p50", "pred_p90",
    "model_name", "status", "reason", "training_start_year", "training_end_year",
    "n_train_years", "n_calibration_rows",
]


def _metrics(df: pd.DataFrame) -> dict:
    evaluated = df[df["status"] == "evaluated"]
    e = evaluated["pred_p50"] - evaluated["actual"]
    summary = {
        "status": "evaluated" if len(evaluated) else "insufficient_evaluation",
        "n_folds": len(df), "n_evaluated": len(evaluated), "n_skipped": len(df) - len(evaluated),
        "mae": float(e.abs().mean()) if len(e) else np.nan,
        "rmse": float(np.sqrt((e**2).mean())) if len(e) else np.nan,
        "bias": float(e.mean()) if len(e) else np.nan,
        "coverage": float(((evaluated["actual"] >= evaluated["pred_p10"])
                           & (evaluated["actual"] <= evaluated["pred_p90"])).mean()) if len(e) else np.nan,
        "mean_interval_width": float((evaluated["pred_p90"] - evaluated["pred_p10"]).mean()) if len(e) else np.nan,
    }
    if "baseline_pred_p50" in evaluated:
        paired = evaluated[np.isfinite(pd.to_numeric(evaluated["baseline_pred_p50"], errors="coerce"))]
        summary["n_paired"] = len(paired)
        summary["paired_model_mae"] = float((paired.pred_p50 - paired.actual).abs().mean()) if len(paired) else np.nan
        summary["baseline_mae"] = float((paired.baseline_pred_p50 - paired.actual).abs().mean()) if len(paired) else np.nan
        summary["delta_mae"] = summary["paired_model_mae"] - summary["baseline_mae"]
    return summary


def summarize_backtest(yearly: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    summary = _metrics(yearly)
    rows = [{"station_id": sid, **_metrics(group)} for sid, group in yearly.groupby("station_id")]
    return summary, pd.DataFrame(rows, columns=["station_id", *summary])


def pair_baseline(yearly: pd.DataFrame, baseline: pd.DataFrame) -> pd.DataFrame:
    """Attach only matching successful baseline folds, never an aggregate from other years."""
    keys = ["station_id", "season_year"]
    valid = baseline[baseline.status == "evaluated"][keys + ["actual", "pred_p50"]]
    result = yearly.merge(valid.rename(columns={"actual": "baseline_actual", "pred_p50": "baseline_pred_p50"}),
                          on=keys, how="left", validate="one_to_one")
    matched = result.baseline_actual.notna()
    if not result.loc[matched, "actual"].eq(result.loc[matched, "baseline_actual"]).all():
        raise ValueError("Baseline and model folds have different actual targets")
    return result.drop(columns="baseline_actual")


def walk_forward_backtest(dataset: pd.DataFrame, model_name: str, cfg: Config) -> tuple[pd.DataFrame, dict, pd.DataFrame]:
    req = ["station_id", "season_year", "target_doy"]
    for c in req:
        if c not in dataset.columns:
            raise ValueError(f"Dataset missing required column: {c}")

    if dataset[["station_id", "season_year"]].isna().any().any():
        raise ValueError("Backtest station IDs and season years must be nonmissing")
    years = pd.to_numeric(dataset["season_year"], errors="raise")
    if not (np.isfinite(years) & (years == np.floor(years))).all():
        raise ValueError("Backtest season years must be finite integers")
    dataset = dataset.copy()
    dataset["season_year"] = years.astype(int)
    if dataset.duplicated(["station_id", "season_year"]).any():
        raise ValueError("Backtest requires unique station-years")
    dataset["target_doy"] = pd.to_numeric(dataset["target_doy"], errors="raise").replace([np.inf, -np.inf], np.nan)
    if dataset.empty:
        yearly = pd.DataFrame(columns=YEARLY_COLUMNS)
        summary, per_station = summarize_backtest(yearly)
        return yearly, summary, per_station

    bt = cfg.backtest
    start_year = int(bt.get("start_year", int(dataset["season_year"].min())))
    end_year = int(bt.get("end_year", int(dataset["season_year"].max())))
    min_train_years = int(bt.get("min_train_years", 20))
    if start_year > end_year or min_train_years < 1:
        raise ValueError("Backtest requires ordered year bounds and positive min_train_years")

    rows: list[dict] = []
    for sid, sdf in dataset.groupby("station_id"):
        sdf = sdf.sort_values("season_year")
        for t in range(start_year, end_year + 1):
            train = sdf[(sdf["season_year"] < t) & sdf["target_doy"].notna()].copy()
            test = sdf[sdf["season_year"] == t].copy()
            row = {col: np.nan for col in YEARLY_COLUMNS}
            row.update(station_id=sid, season_year=t, model_name=model_name, reason="",
                       n_train_years=len(train), training_start_year=train.season_year.min(),
                       training_end_year=train.season_year.max())
            if test.empty:
                row["status"] = "missing_test_year"
                rows.append(row)
                continue
            row["actual"] = test.iloc[0].target_doy
            if pd.isna(row["actual"]):
                row["status"] = "missing_actual"
                rows.append(row)
                continue
            if len(train) < min_train_years:
                row["status"] = "insufficient_training"
                rows.append(row)
                continue

            X_train = train.drop(columns=["target_doy"])
            y_train = train["target_doy"]
            X_test = test.drop(columns=["target_doy"])

            model = create_model(model_name, cfg.models)
            try:
                model.fit(X_train, y_train, meta=train[["station_id", "season_year"]], config=cfg.models)
                pred = model.predict(X_test, cfg.models)
            except ValueError as exc:
                row.update(status="model_unavailable", reason=str(exc))
                rows.append(row)
                continue
            row["n_calibration_rows"] = getattr(model, "metadata", {}).get("n_calib", np.nan)
            if len(pred) != 1 or not {"p10", "p50", "p90"}.issubset(pred.columns):
                raise ValueError("Model must return one prediction with p10/p50/p90 per test row")
            bounds = pred.iloc[0][["p10", "p50", "p90"]].to_numpy(dtype=float)
            row.update(zip(["pred_p10", "pred_p50", "pred_p90"], bounds))
            row["status"] = "evaluated" if np.isfinite(bounds).all() and bounds[0] <= bounds[1] <= bounds[2] else "invalid_prediction"
            rows.append(row)

    yearly = pd.DataFrame(rows, columns=YEARLY_COLUMNS)
    summary, per_station = summarize_backtest(yearly)
    return yearly, summary, per_station


def write_eval_reports(
    yearly: pd.DataFrame,
    summary: dict,
    per_station: pd.DataFrame,
    config: Config,
    target: str,
    model_name: str,
    threshold_c: float | None,
) -> Path:
    out = config.reports_dir / target / model_name / threshold_key(threshold_c)
    out.mkdir(parents=True, exist_ok=True)

    summary_df = pd.DataFrame([summary])

    summary_df.to_csv(out / "summary.csv", index=False)
    per_station.to_csv(out / "per_station.csv", index=False)
    yearly.to_csv(out / "yearly_errors.csv", index=False)
    (out / "plots").mkdir(exist_ok=True)
    return out
