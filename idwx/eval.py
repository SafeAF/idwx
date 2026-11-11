from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from idwx.config import Config
from idwx.forecast_protocol import (
    PROTOCOL_ID,
    issue_context,
    protocol_inputs,
    protocol_training_data,
    valid_calendar_bounds,
    validate_protocol,
)
from idwx.models import create_model
from idwx.models.conformal import InsufficientCalibrationError
from idwx.registry import threshold_key

YEARLY_COLUMNS = [
    "station_id", "season_year", "actual", "pred_p10", "pred_p50", "pred_p90",
    "model_name", "status", "reason", "training_start_year", "training_end_year",
    "n_train_years", "n_calibration_rows",
    "forecast_protocol", "issue_date", "information_cutoff_date",
    "calendar_reference_year",
    "fitting_start_year", "fitting_end_year", "calibration_start_year", "calibration_end_year",
    "n_fitting_years", "n_calibration_years", "n_embargo_rows", "nominal_coverage",
    "calibration_rank", "interval_semantics",
]


def _record_calibration(row: dict, model: object) -> None:
    metadata = getattr(model, "metadata", {})
    for column in ("fitting_start_year", "fitting_end_year", "calibration_start_year", "calibration_end_year",
                   "n_fitting_years", "n_calibration_years", "nominal_coverage", "calibration_rank", "interval_semantics"):
        row[column] = metadata.get(column, np.nan)
    row["n_calibration_rows"] = metadata.get("n_calib", np.nan)
    row["n_embargo_rows"] = metadata.get("n_embargo", np.nan)


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
    nominal = pd.to_numeric(evaluated.get("nominal_coverage", pd.Series(dtype=float)), errors="coerce")
    summary["nominal_coverage"] = float(nominal.iloc[0]) if len(e) and nominal.notna().all() and nominal.nunique() == 1 else np.nan
    summary["coverage_minus_nominal"] = summary["coverage"] - summary["nominal_coverage"]
    summary["lower_miss_rate"] = float((evaluated.actual < evaluated.pred_p10).mean()) if len(e) else np.nan
    summary["upper_miss_rate"] = float((evaluated.actual > evaluated.pred_p90).mean()) if len(e) else np.nan
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


def walk_forward_backtest(dataset: pd.DataFrame, model_name: str, cfg: Config, *, protocol: str | None = None) -> tuple[pd.DataFrame, dict, pd.DataFrame]:
    if protocol is not None:
        validate_protocol(protocol)
        protocol_inputs(dataset)
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
            train = (protocol_training_data(sdf, t) if protocol is not None else
                     sdf[(sdf["season_year"] < t) & sdf["target_doy"].notna()].copy())
            test = sdf[sdf["season_year"] == t].copy()
            row = {col: np.nan for col in YEARLY_COLUMNS}
            row.update(station_id=sid, season_year=t, model_name=model_name, reason="",
                       n_train_years=len(train), training_start_year=train.season_year.min(),
                       training_end_year=train.season_year.max())
            if protocol is not None:
                context = issue_context(t, f"{t:04d}-01-01")
                row.update(forecast_protocol=PROTOCOL_ID, issue_date=context["issue_date"],
                           information_cutoff_date=context["information_cutoff_date"],
                           calendar_reference_year=context["calendar_reference_year"])
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

            X_train = protocol_inputs(train) if protocol is not None else train.drop(columns=["target_doy"])
            y_train = train["target_doy"]
            X_test = protocol_inputs(test) if protocol is not None else test.drop(columns=["target_doy"])

            model = create_model(model_name, cfg.models)
            try:
                model.fit(X_train, y_train, meta=train[["station_id", "season_year"]], config=cfg.models)
                pred = model.predict(X_test, cfg.models)
            except InsufficientCalibrationError as exc:
                _record_calibration(row, model)
                row.update(status="insufficient_calibration", reason=str(exc))
                rows.append(row)
                continue
            except ValueError as exc:
                _record_calibration(row, model)
                row.update(status="model_unavailable", reason=str(exc))
                rows.append(row)
                continue
            _record_calibration(row, model)
            if len(pred) != 1 or not {"p10", "p50", "p90"}.issubset(pred.columns):
                raise ValueError("Model must return one prediction with p10/p50/p90 per test row")
            bounds = pred.iloc[0][["p10", "p50", "p90"]].to_numpy(dtype=float)
            row.update(zip(["pred_p10", "pred_p50", "pred_p90"], bounds))
            valid = (valid_calendar_bounds(bounds) if protocol is not None else
                     np.isfinite(bounds).all() and bounds[0] <= bounds[1] <= bounds[2])
            row["status"] = "evaluated" if valid else "invalid_prediction"
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
    protocol: str | None = None,
) -> Path:
    out = config.reports_dir / target / model_name / threshold_key(threshold_c)
    if protocol is not None:
        validate_protocol(protocol)
        out = out / protocol
    out.mkdir(parents=True, exist_ok=True)

    summary_df = pd.DataFrame([summary])

    summary_df.to_csv(out / "summary.csv", index=False)
    per_station.to_csv(out / "per_station.csv", index=False)
    yearly.to_csv(out / "yearly_errors.csv", index=False)
    (out / "plots").mkdir(exist_ok=True)
    return out
