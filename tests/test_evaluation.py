import json
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
from typer.testing import CliRunner

from idwx.cli import app
from idwx.config import load_config
from idwx.eval import (
    pair_baseline,
    summarize_backtest,
    walk_forward_backtest,
    write_eval_reports,
)


@pytest.fixture
def cfg(tmp_path):
    return replace(load_config("configs/idaho.yml"), reports_dir=tmp_path / "reports",
                   backtest={"start_year": 2000, "end_year": 2006, "min_train_years": 2})


def dataset():
    years = [2000, 2001, 2003, 2004, 2005, 2006]
    return pd.DataFrame({"station_id": "demo", "season_year": years, "year_index": years,
                         "target_doy": [10., np.nan, 20., 30., np.inf, 40.]})


def test_empty_and_ineligible_backtests_write_explicit_reports(cfg):
    for data in (dataset().iloc[:0], dataset().iloc[:1]):
        yearly, summary, stations = walk_forward_backtest(data, "climatology", cfg)
        paired = pair_baseline(yearly, yearly)
        paired_summary, _ = summarize_backtest(paired)
        assert paired_summary["n_paired"] == 0
        assert summary["status"] == "insufficient_evaluation"
        assert summary["n_evaluated"] == 0
        assert np.isnan(summary["mae"]) and np.isnan(summary["coverage"])
        path = write_eval_reports(yearly, summary, stations, cfg, "last_spring_frost", "climatology", 0.)
        assert path.name == "threshold_0p0"
        assert pd.read_csv(path / "yearly_errors.csv").columns.tolist() == yearly.columns.tolist()
        assert pd.read_csv(path / "summary.csv").iloc[0].status == "insufficient_evaluation"


def test_walk_forward_counts_skips_and_scores_only_valid_folds(cfg):
    yearly, summary, stations = walk_forward_backtest(dataset(), "climatology", cfg)
    by_year = yearly.set_index("season_year")
    assert by_year.status.to_dict() == {
        2000: "insufficient_training", 2001: "missing_actual", 2002: "missing_test_year",
        2003: "insufficient_training", 2004: "evaluated", 2005: "missing_actual", 2006: "evaluated",
    }
    assert summary["n_folds"] == 7 and summary["n_evaluated"] == 2 and summary["n_skipped"] == 5
    assert summary["mae"] == 17.5 and summary["bias"] == -17.5
    assert summary["coverage"] == 0 and summary["mean_interval_width"] == pytest.approx(12.)
    assert by_year.loc[2004, "n_train_years"] == 2
    assert by_year.loc[2004, "training_end_year"] == 2003
    assert stations.iloc[0].n_evaluated == 2
    later_changed = dataset()
    later_changed.loc[later_changed.season_year == 2006, "target_doy"] = 4000
    revised, _, _ = walk_forward_backtest(later_changed, "climatology", cfg)
    pd.testing.assert_series_equal(yearly.iloc[4], revised.iloc[4])


@pytest.mark.parametrize("bounds", [[np.nan, 5., 8.], [8., 5., 2.], [1., np.inf, 8.]])
def test_invalid_intervals_are_excluded_and_counted(cfg, monkeypatch, bounds):
    class InvalidModel:
        def fit(self, *args, **kwargs):
            pass

        def predict(self, *args, **kwargs):
            return pd.DataFrame([bounds], columns=["p10", "p50", "p90"])

    monkeypatch.setattr("idwx.eval.create_model", lambda *args: InvalidModel())
    yearly, summary, _ = walk_forward_backtest(dataset(), "invalid", cfg)
    assert (yearly.status == "invalid_prediction").sum() == 2
    assert summary["n_evaluated"] == 0 and np.isnan(summary["mean_interval_width"])


def test_model_insufficient_history_is_an_explicit_fold_status(cfg, monkeypatch):
    class UnavailableModel:
        def fit(self, *args, **kwargs):
            raise ValueError("Need more calibration years")

    monkeypatch.setattr("idwx.eval.create_model", lambda *args: UnavailableModel())
    yearly, summary, _ = walk_forward_backtest(dataset(), "unavailable", cfg)
    failures = yearly[yearly.status == "model_unavailable"]
    assert len(failures) == 2 and (failures.reason == "Need more calibration years").all()
    assert summary["n_evaluated"] == 0


def test_baseline_comparison_uses_identical_eligible_years():
    model = pd.DataFrame({"station_id": "demo", "season_year": [2000, 2001], "actual": 0.,
                          "pred_p10": -10., "pred_p50": [1., 9.], "pred_p90": 10., "status": "evaluated"})
    baseline = model.iloc[1:].copy()
    baseline["pred_p50"] = 3.
    paired = pair_baseline(model, baseline)
    summary, per_station = summarize_backtest(paired)
    assert summary["mae"] == 5.
    assert summary["n_paired"] == 1
    assert summary["paired_model_mae"] == 9. and summary["baseline_mae"] == 3.
    assert summary["delta_mae"] == 6. and per_station.iloc[0].delta_mae == 6.
    baseline["actual"] = 1.
    with pytest.raises(ValueError, match="different actual"):
        pair_baseline(model, baseline)


@pytest.mark.parametrize("change", ["duplicate", "missing_year", "fractional_year", "mixed_duplicate"])
def test_ambiguous_backtest_keys_fail(cfg, change):
    data = dataset()
    if change == "duplicate":
        data = pd.concat([data, data.iloc[:1]])
    elif change == "missing_year":
        data["season_year"] = np.nan
    elif change == "mixed_duplicate":
        data = pd.concat([data, data.iloc[:1].assign(season_year="2000")])
    else:
        data["season_year"] = data.season_year + .5
    with pytest.raises(ValueError):
        walk_forward_backtest(data, "climatology", cfg)


@pytest.mark.parametrize("models", ["trend", "trend,climatology", "climatology,trend"])
def test_cli_baseline_is_independent_of_requested_model_order(cfg, monkeypatch, models):
    monkeypatch.setattr("idwx.cli.load_config", lambda *args: cfg)
    monkeypatch.setattr("idwx.cli.load_or_build_dataset", lambda *args, **kwargs: dataset())
    result = CliRunner().invoke(app, ["eval", "--target", "last_spring_frost", "--models", models])
    assert result.exit_code == 0, result.output
    output = json.loads(result.output, parse_constant=lambda value: pytest.fail(f"Invalid JSON value {value}"))
    assert output["validation_status"] == "unvalidated_research"
    trend = next(item for item in output["outputs"] if item["model"] == "trend")
    assert trend["summary"]["n_paired"] == 2
    assert trend["summary"]["baseline_mae"] == 17.5
    assert pd.read_csv(cfg.reports_dir / "last_spring_frost/trend/threshold_0p0/yearly_errors.csv").baseline_pred_p50.notna().sum() == 2


def test_cli_empty_evaluation_metrics_are_json_null(cfg, monkeypatch):
    cfg = replace(cfg, backtest={**cfg.backtest, "min_train_years": 99})
    monkeypatch.setattr("idwx.cli.load_config", lambda *args: cfg)
    monkeypatch.setattr("idwx.cli.load_or_build_dataset", lambda *args, **kwargs: dataset())
    result = CliRunner().invoke(app, ["eval", "--target", "last_spring_frost", "--models", "climatology"])
    assert result.exit_code == 0, result.output
    output = json.loads(result.output, parse_constant=lambda value: pytest.fail(f"Invalid JSON value {value}"))
    assert output["outputs"][0]["summary"]["mae"] is None
