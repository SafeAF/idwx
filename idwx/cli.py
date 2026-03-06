from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path

import pandas as pd
import typer

from idwx.config import load_config, load_stations
from idwx.datasets import build_all_datasets, build_future_row, load_or_build_dataset
from idwx.eval import walk_forward_backtest, write_eval_reports
from idwx.io import build_data_cache
from idwx.models import create_model
from idwx.registry import latest_model_dir, save_model_artifacts
from idwx.targets import build_targets_cache

app = typer.Typer(help="Idaho weather seasonal modeling CLI")
data_app = typer.Typer(help="Data cache operations")
dataset_app = typer.Typer(help="Dataset operations")
target_app = typer.Typer(help="Target table operations")
app.add_typer(data_app, name="data")
app.add_typer(dataset_app, name="dataset")
app.add_typer(target_app, name="target")


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
    typer.echo(json.dumps({"rows": int(len(ds)), "target": target, "threshold": threshold}))


@target_app.command("build")
def target_build(
    config: str = typer.Option("configs/idaho.yml", "--config"),
    rebuild: bool = typer.Option(False, "--rebuild/--no-rebuild"),
) -> None:
    cfg = load_config(config)
    written = build_targets_cache(cfg, rebuild=rebuild)
    typer.echo(json.dumps({"targets_written": written, "count": len(written)}))


@app.command("train")
def train(
    target: str = typer.Option(..., "--target"),
    models: str = typer.Option("climatology,trend,rf", "--models"),
    station: str = typer.Option("all", "--station"),
    threshold: float = typer.Option(0.0, "--threshold"),
    config: str = typer.Option("configs/idaho.yml", "--config"),
) -> None:
    cfg = load_config(config)
    ds = load_or_build_dataset(cfg, target_name=target, threshold=threshold)
    if ds.empty:
        raise typer.Exit("No dataset rows available for training")

    model_names = [m.strip() for m in models.split(",") if m.strip()]
    stations = sorted(ds["station_id"].dropna().unique().tolist())
    if station != "all":
        stations = [station.strip().lower().replace(" ", "_")]

    artifacts = []
    for sid in stations:
        sdf = ds[ds["station_id"] == sid].sort_values("season_year")
        if sdf["target_doy"].notna().sum() < 6:
            continue
        X = sdf.drop(columns=["target_doy"])
        y = sdf["target_doy"]

        for model_name in model_names:
            model = create_model(model_name, cfg.models)
            model.fit(X, y, meta=sdf[["station_id", "season_year"]], config=cfg.models)
            out = save_model_artifacts(
                model,
                config=cfg,
                target=target,
                station=sid,
                model_name=model_name,
                dataset=sdf,
                metrics={},
            )
            artifacts.append(str(out))

    typer.echo(json.dumps({"trained": artifacts, "count": len(artifacts)}))


@app.command("eval")
def evaluate(
    target: str = typer.Option(..., "--target"),
    models: str = typer.Option("climatology,trend,rf", "--models"),
    threshold: float = typer.Option(0.0, "--threshold"),
    config: str = typer.Option("configs/idaho.yml", "--config"),
) -> None:
    cfg = load_config(config)
    ds = load_or_build_dataset(cfg, target_name=target, threshold=threshold)
    if ds.empty:
        raise typer.Exit("No dataset rows available for evaluation")

    model_names = [m.strip() for m in models.split(",") if m.strip()]

    baseline_summary = None
    outputs = []
    for model_name in model_names:
        yearly, summary, per_station = walk_forward_backtest(ds, model_name=model_name, cfg=cfg)
        if model_name == "climatology":
            baseline_summary = summary
        report_dir = write_eval_reports(
            yearly,
            summary,
            per_station,
            config=cfg,
            target=target,
            model_name=model_name,
            baseline_summary=baseline_summary if model_name != "climatology" else None,
        )
        outputs.append({"model": model_name, "report_dir": str(report_dir), "summary": summary})

    typer.echo(json.dumps({"target": target, "outputs": outputs}, indent=2))


@app.command("predict")
def predict(
    target: str = typer.Option(..., "--target"),
    station: str = typer.Option(..., "--station"),
    model: str = typer.Option("rf", "--model"),
    threshold: float = typer.Option(0.0, "--threshold"),
    config: str = typer.Option("configs/idaho.yml", "--config"),
    pretty: bool = typer.Option(False, "--pretty"),
) -> None:
    cfg = load_config(config)
    sid = station.strip().lower().replace(" ", "_")

    ds = load_or_build_dataset(cfg, target_name=target, threshold=threshold)
    hist = ds[ds["station_id"] == sid].sort_values("season_year")
    if hist.empty:
        raise typer.Exit(f"No dataset history for station: {sid}")

    model_dir = latest_model_dir(cfg, target=target, station=sid, model_name=model)

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

    future = build_future_row(hist)
    pred = mdl.predict(future.drop(columns=["target_doy"], errors="ignore"), cfg.models).iloc[0]

    baseline_dir = latest_model_dir(cfg, target=target, station=sid, model_name="climatology")
    from idwx.models.climatology import ClimatologyModel

    baseline_model = ClimatologyModel.load(baseline_dir)
    baseline = baseline_model.predict(future.drop(columns=["target_doy"], errors="ignore"), cfg.models).iloc[0]

    out = {
        "generated_at_utc": datetime.utcnow().isoformat() + "Z",
        "station_id": sid,
        "target": target,
        "threshold_c": threshold,
        "season_year": int(future.iloc[0]["season_year"]),
        "model": model,
        "prediction": {"p10": float(pred["p10"]), "p50": float(pred["p50"]), "p90": float(pred["p90"])},
        "baseline_climatology": {
            "p10": float(baseline["p10"]),
            "p50": float(baseline["p50"]),
            "p90": float(baseline["p90"]),
        },
    }

    if pretty:
        typer.echo(pd.DataFrame([out["prediction"]]).to_string(index=False))
    typer.echo(json.dumps(out, indent=2))
