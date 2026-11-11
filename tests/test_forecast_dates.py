import json
from dataclasses import replace

import pandas as pd
import pytest
from typer.testing import CliRunner

from idwx.cli import app
from idwx.config import load_config
from idwx.eval import pair_baseline, summarize_backtest, walk_forward_backtest
from idwx.forecast_dates import (
    DATE_POLICY_VERSION,
    interpret_forecast_dates,
    window_calendar_bounds,
)
from idwx.forecast_protocol import (
    INPUT_VERSION,
    PROTOCOL_ID,
    build_protocol_dataset,
    protocol_inputs,
)

FULL_YEAR = {"start": "01-01", "end": "12-31"}


@pytest.mark.parametrize("year, bounds, dates, width", [
    (2000, [59., 60., 61.], ["2000-02-28", "2000-02-29", "2000-03-01"], 2),
    (2001, [59., 60., 61.], ["2001-02-28", "2001-03-01", "2001-03-01"], 1),
    (2001, [59.5, 60., 61.], ["2001-03-01"] * 3, 0),
    (2000, [59.1, 60., 60.9], ["2000-02-29"] * 3, 0),
    (2001, [61., 61., 61.], ["2001-03-01"] * 3, 0),
    (2023, [365., 365.5, 366.], ["2023-12-30", "2023-12-31", "2023-12-31"], 1),
    (2025, [1., 1., 1.], ["2025-01-01"] * 3, 0),
    (2025, [151., 151.5, 152.], ["2025-05-30", "2025-05-31", "2025-05-31"], 1),
])
def test_calendar_support_fractional_bounds_and_later_tie_policy(year, bounds, dates, width):
    original = list(bounds)
    result = interpret_forecast_dates(bounds, year, FULL_YEAR)
    assert [result[key] for key in ("lower", "point", "upper")] == dates
    assert result["interval_width_days"] == width
    assert bounds == original  # Numeric uncertainty bounds are never clipped or rounded in place.


@pytest.mark.parametrize("year, bounds", [(2001, [60., 60., 60.]), (2001, [59.1, 60., 60.9]),
                                         (2024, [151.1, 151.5, 151.9])])
def test_empty_discrete_date_support_is_rejected(year, bounds):
    with pytest.raises(ValueError, match="no supported date"):
        interpret_forecast_dates(bounds, year, FULL_YEAR)


@pytest.mark.parametrize("bounds", [[0., 10., 20.], [10., 20., 367.], [30., 20., 10.],
                                    [1., float("nan"), 5.], [1., 5.], [1., 5., float("inf")]])
def test_invalid_numeric_bounds_are_not_repaired(bounds):
    with pytest.raises(ValueError, match="finite ordered"):
        interpret_forecast_dates(bounds, 2024, FULL_YEAR)


@pytest.mark.parametrize("window", [{}, {"start": "01-01"}, {"start": "02-30", "end": "06-30"},
                                    {"start": "1-1", "end": "06-30"}, {"start": "12-01", "end": "01-31"}])
def test_bad_or_cross_year_windows_fail(window):
    with pytest.raises(ValueError, match="Forecast window"):
        window_calendar_bounds(window)


@pytest.mark.parametrize("year", [None, True, 2024.5, 0, 10000])
def test_invalid_forecast_years_fail(year):
    with pytest.raises(ValueError, match="integer year"):
        interpret_forecast_dates([1., 2., 3.], year, FULL_YEAR)


def test_window_boundaries_are_inclusive_and_outside_bounds_are_rejected():
    spring = {"start": "01-01", "end": "06-30"}
    assert window_calendar_bounds(spring) == (1, 182)
    dates = interpret_forecast_dates([1., 61., 182.], 2023, spring)
    assert dates["lower"] == "2023-01-01" and dates["upper"] == "2023-06-30"
    with pytest.raises(ValueError, match="outside the configured"):
        interpret_forecast_dates([1., 61., 182.01], 2023, spring)
    fall = {"start": "08-01", "end": "12-31"}
    dates = interpret_forecast_dates([214., 300., 366.], 2023, fall)
    assert dates["lower"] == "2023-08-01" and dates["point"] == "2023-10-26"
    assert dates["upper"] == "2023-12-31"


@pytest.fixture
def cfg(tmp_path):
    dates = pd.date_range("1998-01-01", "2012-12-31")
    daily = pd.DataFrame({"date": dates, "station_id": "demo", "tmin_c": 5., "tmax_c": 15., "tmean_c": 10.})
    daily.loc[((daily.date.dt.month == 3) | (daily.date.dt.month == 10)) & daily.date.dt.day.eq(1), "tmin_c"] = 0.
    stations = tmp_path / "stations.yml"
    stations.write_text("stations: [{station_id: demo, timezone: America/Boise}]", encoding="utf-8")
    cfg = replace(load_config("configs/idaho.yml"), cache_dir=tmp_path / "cache", stations_file=stations,
                  models_dir=tmp_path / "models", reports_dir=tmp_path / "reports",
                  backtest={"start_year": 2011, "end_year": 2012, "min_train_years": 6})
    (cfg.cache_dir / "daily").mkdir(parents=True)
    daily.to_parquet(cfg.cache_dir / "daily/demo.parquet", index=False)
    return cfg


def fixed_model(bounds):
    class FixedModel:
        def fit(self, *args, **kwargs):
            pass

        def predict(self, *args, **kwargs):
            return pd.DataFrame([bounds], columns=["p10", "p50", "p90"])

    return FixedModel()


def test_date_errors_and_width_use_actual_year_days_and_pair_baselines(cfg, monkeypatch):
    data = build_protocol_dataset(cfg, "last_spring_frost", 0.)
    baseline, _, _ = walk_forward_backtest(data, "climatology", cfg, protocol=PROTOCOL_ID)
    monkeypatch.setattr("idwx.eval.create_model", lambda *args: fixed_model([59., 59., 61.]))
    yearly, summary, _ = walk_forward_backtest(data, "fixed", cfg, protocol=PROTOCOL_ID)
    assert yearly.actual_date.tolist() == ["2011-03-01", "2012-03-01"]
    assert yearly.pred_point_date.tolist() == ["2011-02-28", "2012-02-28"]
    assert yearly.date_error_days.tolist() == [-1, -2]
    assert yearly.date_interval_width_days.tolist() == [1, 2]
    assert summary["mae"] == 2. and summary["date_mae_days"] == 1.5
    assert summary["date_bias_days"] == -1.5 and summary["mean_date_interval_width_days"] == 1.5
    assert summary["date_coverage"] == summary["coverage"] == 1.
    paired = pair_baseline(yearly, baseline)
    compared, stations = summarize_backtest(paired)
    assert compared["n_date_paired"] == 2
    assert compared["baseline_date_mae_days"] == 0. and compared["delta_date_mae_days"] == 1.5
    assert stations.iloc[0].date_mae_days == 1.5
    assert (yearly.date_policy_version == DATE_POLICY_VERSION).all()


def test_backtest_respects_effective_station_window_and_retains_invalid_reason(cfg, monkeypatch):
    monkeypatch.setattr("idwx.eval.create_model", lambda *args: fixed_model([183., 190., 200.]))
    data = build_protocol_dataset(cfg, "last_spring_frost", 0.)
    invalid, summary, _ = walk_forward_backtest(data, "fixed", cfg, protocol=PROTOCOL_ID)
    assert (invalid.status == "invalid_prediction").all()
    assert invalid.reason.str.contains("outside the configured").all()
    assert invalid.pred_point_date.isna().all() and summary["n_date_evaluated"] == 0
    assert summary["n_invalid_predictions"] == 2
    assert invalid.pred_p50.tolist() == [190., 190.]
    extended = replace(cfg, station_windows={"demo": {"last_spring_frost": {"start": "01-01", "end": "07-31"}}})
    data = build_protocol_dataset(extended, "last_spring_frost", 0.)
    valid, _, _ = walk_forward_backtest(data, "fixed", extended, protocol=PROTOCOL_ID)
    assert (valid.status == "evaluated").all()
    assert (valid.window_end_month_day == "07-31").all()
    assert valid.pred_point_date.tolist() == ["2011-07-08", "2012-07-08"]


def test_backtest_empty_nonleap_support_is_invalid_not_fabricated(cfg, monkeypatch):
    monkeypatch.setattr("idwx.eval.create_model", lambda *args: fixed_model([60., 60., 60.]))
    data = build_protocol_dataset(cfg, "last_spring_frost", 0.)
    yearly, summary, _ = walk_forward_backtest(data, "fixed", cfg, protocol=PROTOCOL_ID)
    assert yearly.iloc[0].status == "invalid_prediction" and "no supported date" in yearly.iloc[0].reason
    assert yearly.iloc[1].status == "evaluated" and yearly.iloc[1].pred_point_date == "2012-02-29"
    assert summary["n_evaluated"] == summary["n_date_evaluated"] == 1


@pytest.mark.parametrize("actual_date", ["2010-03-01", "2011-03-02"])
def test_backtest_rejects_inconsistent_actual_date_before_scoring(cfg, monkeypatch, actual_date):
    data = build_protocol_dataset(cfg, "last_spring_frost", 0.)
    data.loc[data.season_year == 2011, "target_date"] = actual_date
    monkeypatch.setattr("idwx.eval.create_model", lambda *args: pytest.fail("Model fitted with inconsistent actual date"))
    with pytest.raises(ValueError, match="target date and calendar coordinate disagree"):
        walk_forward_backtest(data, "climatology", cfg, protocol=PROTOCOL_ID)


@pytest.mark.parametrize("attribute, value", [("input_version", 1), ("date_policy_version", None), ("date_policy_version", 99)])
def test_old_or_unversioned_date_inputs_are_rejected(cfg, attribute, value):
    data = build_protocol_dataset(cfg, "last_spring_frost", 0.)
    data.attrs[attribute] = value
    with pytest.raises(ValueError, match="provenance"):
        protocol_inputs(data)


def test_cli_dates_context_and_prediction_baseline_failure_states(cfg, monkeypatch):
    monkeypatch.setattr("idwx.cli.load_config", lambda *args: cfg)
    runner = CliRunner()
    args = ["--target", "last_spring_frost", "--protocol", PROTOCOL_ID,
            "--season-year", "2011", "--issue-date", "2011-01-01"]
    train = runner.invoke(app, ["train", *args, "--models", "climatology"])
    assert train.exit_code == 0, train.output
    predict = ["predict", *args, "--station", "demo", "--model", "climatology"]
    result = runner.invoke(app, predict)
    assert result.exit_code == 0, result.output
    output = json.loads(result.output)
    assert output["date_policy_version"] == DATE_POLICY_VERSION
    assert output["input_version"] == INPUT_VERSION == 2
    assert output["target_window"] == {"start": "01-01", "end": "06-30"}
    assert output["prediction_dates"] == output["baseline_climatology_dates"]
    assert output["prediction_dates"]["point"] == "2011-03-01"
    assert output["prediction"]["point"] == 61.  # Original shared-calendar coordinate is retained.

    for fail_on_call, bounds, reason in [(1, [183., 190., 200.], "outside the configured"),
                                         (2, [183., 190., 200.], "outside the configured"),
                                         (1, [60., 60., 60.], "no supported date")]:
        calls = []

        def controlled_prediction(*args, _calls=calls, _bounds=bounds, _fail_on_call=fail_on_call, **kwargs):
            _calls.append(True)
            return pd.DataFrame([_bounds if len(_calls) == _fail_on_call else [61., 61., 61.]], columns=["p10", "p50", "p90"])

        monkeypatch.setattr("idwx.models.climatology.ClimatologyModel.predict", controlled_prediction)
        invalid = runner.invoke(app, predict)
        assert invalid.exit_code != 0 and reason in invalid.output
        assert '"prediction_dates"' not in invalid.output  # No partial JSON result is emitted.
