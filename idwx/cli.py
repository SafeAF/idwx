from __future__ import annotations

import json
import logging
import math
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import typer

from idwx.config import Config, load_config, load_stations
from idwx.datasets import build_all_datasets, build_future_row, load_or_build_dataset
from idwx.eval import (
    pair_baseline,
    summarize_backtest,
    walk_forward_backtest,
    write_eval_reports,
)
from idwx.features import build_features_cache
from idwx.forecast_protocol import (
    build_protocol_dataset,
    issue_context,
    protocol_inputs,
    protocol_prediction_data,
    valid_calendar_bounds,
    validate_protocol,
)
from idwx.io import build_data_cache
from idwx.models import create_model
from idwx.registry import latest_model_dir, save_model_artifacts
from idwx.targets import build_targets_cache

app = typer.Typer(help="Idaho weather seasonal modeling CLI")
data_app = typer.Typer(help="Data cache operations")
dataset_app = typer.Typer(help="Dataset operations")
target_app = typer.Typer(help="Target table operations")
feature_app = typer.Typer(help="Feature table operations")
app.add_typer(data_app, name="data")
app.add_typer(dataset_app, name="dataset")
app.add_typer(target_app, name="target")
app.add_typer(feature_app, name="feature")


def _modeling_dataset(cfg: Config, target: str, threshold: float, protocol: str | None,
                      *, season_year: int | None = None, issue_date: str | None = None,
                      require_context: bool = False) -> pd.DataFrame:
    if protocol is None:
        if season_year is not None or issue_date is not None:
            raise typer.BadParameter("Explicit forecast context requires --protocol frost_jan01_v1")
        return load_or_build_dataset(cfg, target_name=target, threshold=threshold)
    try:
        validate_protocol(protocol)
        if require_context:
            issue_context(season_year, issue_date)
        return build_protocol_dataset(cfg, target, threshold, through_year=season_year)
    except (ValueError, FileNotFoundError) as exc:
        raise typer.BadParameter(str(exc)) from exc


@app.command("export-web")
def export_web(
    config: str = typer.Option("configs/idaho.yml", "--config"),
    output: Path = typer.Option(Path("web"), "--output"),  # noqa: B008 - Typer declares CLI options in defaults.
    baseline_start: int | None = typer.Option(None, "--baseline-start"),
    baseline_end: int | None = typer.Option(None, "--baseline-end"),
    min_history_years: int = typer.Option(10, "--min-history-years", min=1),
    demo: bool = typer.Option(False, "--demo", help="Export clearly labeled synthetic data without reading weather files."),
) -> None:
    """Export historical summaries and immutable JSON/CSV data for the static website."""
    from idwx.web_export import export_web_data

    if (baseline_start is None) != (baseline_end is None):
        raise typer.BadParameter("Supply both --baseline-start and --baseline-end")
    baseline = (baseline_start, baseline_end) if baseline_start is not None else None
    cfg = load_config(config)
    try:
        manifest = export_web_data(cfg, load_stations(cfg.stations_file), output,
                                   baseline=baseline, min_years=min_history_years, demo=demo)
    except (FileNotFoundError, ValueError) as exc:
        raise typer.BadParameter(str(exc)) from exc
    typer.echo(json.dumps({"release_id": manifest["release_id"], "data_kind": manifest["data_kind"],
                           "manifest": str(output / "data" / "manifest.json")}))


@data_app.command("build")
def data_build(
    config: str = typer.Option("configs/idaho.yml", "--config"),
    rebuild: bool = typer.Option(False, "--rebuild/--no-rebuild"),
) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg = load_config(config)
    stations = load_stations(cfg.stations_file)
    written = build_data_cache(cfg, stations, rebuild=rebuild)
    typer.echo(json.dumps({"stations_written": written, "count": len(written)}))


@dataset_app.command("build")
def dataset_build(
    target: str = typer.Option(..., "--target"),
    threshold: float = typer.Option(0.0, "--threshold"),
    config: str = typer.Option("configs/idaho.yml", "--config"),
) -> None:
    cfg = load_config(config)
    ds = build_all_datasets(cfg, target_name=target, threshold=threshold)
    typer.echo(json.dumps({"rows": len(ds), "target": target, "threshold": threshold}))


@target_app.command("build")
def target_build(
    config: str = typer.Option("configs/idaho.yml", "--config"),
    rebuild: bool = typer.Option(False, "--rebuild/--no-rebuild"),
) -> None:
    cfg = load_config(config)
    written = build_targets_cache(cfg, rebuild=rebuild)
    typer.echo(json.dumps({"targets_written": written, "count": len(written)}))


@feature_app.command("build")
def feature_build(
    target: str = typer.Option("first_fall_frost", "--target"),
    threshold: float = typer.Option(0.0, "--threshold"),
    config: str = typer.Option("configs/idaho.yml", "--config"),
    rebuild: bool = typer.Option(False, "--rebuild/--no-rebuild"),
) -> None:
    cfg = load_config(config)
    written = build_features_cache(cfg, target_name=target, threshold=threshold, rebuild=rebuild)
    typer.echo(json.dumps({"features_written": written, "count": len(written), "target": target, "threshold": threshold}))


@app.command("train")
def train(
    target: str = typer.Option(..., "--target"),
    models: str = typer.Option("climatology,trend,rf", "--models"),
    station: str = typer.Option("all", "--station"),
    threshold: float = typer.Option(0.0, "--threshold"),
    config: str = typer.Option("configs/idaho.yml", "--config"),
    protocol: str | None = typer.Option(None, "--protocol", help="Opt-in frost_jan01_v1 research inputs."),
    season_year: int | None = typer.Option(None, "--season-year"),
    issue_date: str | None = typer.Option(None, "--issue-date"),
) -> None:
    cfg = load_config(config)
    ds = _modeling_dataset(cfg, target, threshold, protocol, season_year=season_year,
                           issue_date=issue_date, require_context=True)
    if ds.empty:
        raise typer.Exit("No dataset rows available for training")

    model_names = [m.strip() for m in models.split(",") if m.strip()]
    stations = sorted(ds["station_id"].dropna().unique().tolist())
    if station != "all":
        stations = [station.strip().lower().replace(" ", "_")]

    artifacts = []
    skipped: list[dict[str, str]] = []
    for sid in stations:
        sdf = ds[ds["station_id"] == sid].sort_values("season_year")
        forecast_context = None
        if protocol is not None:
            try:
                sdf, _, forecast_context = protocol_prediction_data(sdf, season_year, issue_date)
            except ValueError as exc:
                raise typer.BadParameter(str(exc)) from exc
        if sdf["target_doy"].notna().sum() < 6:
            continue
        X = protocol_inputs(sdf) if protocol is not None else sdf.drop(columns=["target_doy"])
        y = sdf["target_doy"]

        for model_name in model_names:
            model = create_model(model_name, cfg.models)
            try:
                model.fit(X, y, meta=sdf[["station_id", "season_year"]], config=cfg.models)
            except ValueError as exc:
                skipped.append({"station_id": sid, "model": model_name, "reason": str(exc)})
                continue
            train_metrics = {
                "n_rows": len(sdf),
                "n_target_non_null": int(sdf["target_doy"].notna().sum()),
                "model_metadata": getattr(model, "metadata", {}),
            }
            out = save_model_artifacts(
                model,
                config=cfg,
                target=target,
                station=sid,
                model_name=model_name,
                dataset=sdf,
                threshold_c=threshold,
                metrics=train_metrics,
                forecast_context=forecast_context,
            )
            artifacts.append(str(out))

    typer.echo(json.dumps({"trained": artifacts, "count": len(artifacts), "skipped": skipped}))


@app.command("eval")
def evaluate(
    target: str = typer.Option(..., "--target"),
    models: str = typer.Option("climatology,trend,rf", "--models"),
    threshold: float = typer.Option(0.0, "--threshold"),
    config: str = typer.Option("configs/idaho.yml", "--config"),
    protocol: str | None = typer.Option(None, "--protocol", help="Opt-in frost_jan01_v1 research inputs."),
) -> None:
    cfg = load_config(config)
    ds = _modeling_dataset(cfg, target, threshold, protocol)
    if ds.empty:
        raise typer.BadParameter("No dataset rows available for evaluation")

    model_names = list(dict.fromkeys(m.strip() for m in models.split(",") if m.strip()))
    if not model_names:
        raise typer.BadParameter("Request at least one model")
    try:
        for model_name in model_names:
            create_model(model_name, cfg.models)  # Reject unknown models even when there are no eligible folds.
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    baseline_yearly, _, _ = walk_forward_backtest(ds, model_name="climatology", cfg=cfg, protocol=protocol)
    outputs = []
    for model_name in model_names:
        if model_name == "climatology":
            yearly = baseline_yearly
        else:
            yearly, _, _ = walk_forward_backtest(ds, model_name=model_name, cfg=cfg, protocol=protocol)
        yearly = pair_baseline(yearly, baseline_yearly)
        summary, per_station = summarize_backtest(yearly)
        report_dir = write_eval_reports(
            yearly,
            summary,
            per_station,
            config=cfg,
            target=target,
            model_name=model_name,
            threshold_c=threshold,
            protocol=protocol,
        )
        outputs.append({"model": model_name, "report_dir": str(report_dir),
                        "summary": {key: None if isinstance(value, float) and not math.isfinite(value) else value
                                    for key, value in summary.items()}})

    typer.echo(json.dumps({"target": target, "threshold_c": threshold,
                           "forecast_protocol": protocol, "validation_status": "unvalidated_research", "outputs": outputs},
                          indent=2, allow_nan=False))


@app.command("predict")
def predict(
    target: str = typer.Option(..., "--target"),
    station: str = typer.Option(..., "--station"),
    model: str = typer.Option("rf", "--model"),
    threshold: float = typer.Option(0.0, "--threshold"),
    config: str = typer.Option("configs/idaho.yml", "--config"),
    pretty: bool = typer.Option(False, "--pretty"),
    protocol: str | None = typer.Option(None, "--protocol", help="Opt-in frost_jan01_v1 research inputs."),
    season_year: int | None = typer.Option(None, "--season-year"),
    issue_date: str | None = typer.Option(None, "--issue-date"),
) -> None:
    cfg = load_config(config)
    sid = station.strip().lower().replace(" ", "_")

    ds = _modeling_dataset(cfg, target, threshold, protocol, season_year=season_year,
                           issue_date=issue_date, require_context=True)
    hist = ds[ds["station_id"] == sid].sort_values("season_year")
    if hist.empty:
        raise typer.Exit(f"No dataset history for station: {sid}")

    forecast_context = None
    if protocol is not None:
        try:
            hist, future, forecast_context = protocol_prediction_data(hist, season_year, issue_date)
        except ValueError as exc:
            raise typer.BadParameter(str(exc)) from exc
    else:
        future = build_future_row(hist)

    try:
        model_dir = latest_model_dir(cfg, target=target, station=sid, model_name=model,
                                    threshold_c=threshold, dataset=hist, forecast_context=forecast_context)
        baseline_dir = latest_model_dir(cfg, target=target, station=sid, model_name="climatology",
                                       threshold_c=threshold, dataset=hist, forecast_context=forecast_context)
    except (FileNotFoundError, ValueError) as exc:
        raise typer.BadParameter(str(exc)) from exc

    if model == "climatology":
        from idwx.models.climatology import ClimatologyModel

        mdl = ClimatologyModel.load(model_dir)
    elif model == "trend":
        from idwx.models.trend import TrendQuantileModel

        mdl = TrendQuantileModel.load(model_dir)
    elif model == "rf":
        from idwx.models.rf import RFConformalModel

        mdl = RFConformalModel.load(model_dir)
    else:
        raise typer.Exit(f"Unsupported model {model}")

    pred = mdl.predict(future.drop(columns=["target_doy"], errors="ignore"), cfg.models).iloc[0]

    from idwx.models.climatology import ClimatologyModel

    baseline_model = ClimatologyModel.load(baseline_dir)
    baseline = baseline_model.predict(future.drop(columns=["target_doy"], errors="ignore"), cfg.models).iloc[0]

    out = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "station_id": sid,
        "target": target,
        "threshold_c": threshold,
        "season_year": int(future.iloc[0]["season_year"]),
        "model": model,
        "validation_status": "unvalidated_research",
        "prediction": {"p10": float(pred["p10"]), "p50": float(pred["p50"]), "p90": float(pred["p90"])},
        "baseline_climatology": {
            "p10": float(baseline["p10"]),
            "p50": float(baseline["p50"]),
            "p90": float(baseline["p90"]),
        },
    }

    if protocol is not None:
        values = [float(pred[key]) for key in ("p10", "p50", "p90")]
        if not valid_calendar_bounds(values):
            raise typer.BadParameter("Protocol model returned invalid calendar bounds; no prediction emitted")
        out.update(forecast_context)
        out["training_period"] = {"start_year": int(hist.season_year.min()), "end_year": int(hist.season_year.max())}
        out["prediction_kind"] = "unvalidated_model_interval"
        out["prediction"] = {"lower": values[0], "point": values[1], "upper": values[2]}

    if pretty:
        typer.echo(pd.DataFrame([out["prediction"]]).to_string(index=False))
    typer.echo(json.dumps(out, indent=2, allow_nan=False))
