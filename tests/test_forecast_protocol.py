import json
from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest
import yaml
from typer.testing import CliRunner

from idwx.cli import app
from idwx.config import load_config
from idwx.eval import walk_forward_backtest
from idwx.forecast_protocol import (
    FEATURE_COLUMNS,
    PROTOCOL_ID,
    build_protocol_dataset,
    build_station_protocol_dataset,
    issue_context,
    protocol_inputs,
    protocol_prediction_data,
    protocol_training_data,
    valid_calendar_bounds,
)
from idwx.history import calendar_day


@pytest.fixture
def daily():
    dates = pd.date_range("1998-01-01", "2012-12-31")
    result = pd.DataFrame({"date": dates, "station_id": "demo", "tmin_c": 5., "tmax_c": 15., "tmean_c": 10.})
    for year in range(1998, 2013):
        result.loc[result.date.isin([pd.Timestamp(year, 5, 10 + year % 5),
                                    pd.Timestamp(year, 10, 1 + year % 4)]), "tmin_c"] = -2.
    return result


@pytest.fixture
def cfg(tmp_path, daily):
    stations = tmp_path / "stations.yml"
    stations.write_text("stations: [{station_id: demo, timezone: America/Boise}]", encoding="utf-8")
    cfg = replace(load_config("configs/idaho.yml"), cache_dir=tmp_path / "cache",
                  models_dir=tmp_path / "models", reports_dir=tmp_path / "reports",
                  stations_file=stations, timezone="UTC",
                  backtest={"start_year": 2000, "end_year": 2010, "min_train_years": 6})
    (cfg.cache_dir / "daily").mkdir(parents=True)
    daily.to_parquet(cfg.cache_dir / "daily/demo.parquet", index=False)
    return cfg


def test_protocol_embargo_rolling_windows_and_whitelist(cfg):
    data = build_protocol_dataset(cfg, "last_spring_frost", 0.)
    history, future, context = protocol_prediction_data(data, 2008, "2008-01-01")
    assert history.season_year.max() == 2006
    assert future.columns.tolist() == FEATURE_COLUMNS
    assert future.iloc[0].target_lag2 == calendar_day(pd.Timestamp("2006-05-11"))
    expected = data.set_index("season_year").target_doy
    assert future.iloc[0].target_roll3_lag2 == expected.loc[2002:2006].tail(3).mean()
    assert future.iloc[0].target_roll5_lag2 == expected.loc[2002:2006].mean()
    assert context["information_cutoff_date"] == context["last_observation_date"] == "2006-12-31"
    assert context["station_timezone"] == "America/Boise"
    poisoned = data.assign(wsi=9999., wsi_lag1=9999., summer_tmin_mean=-9999., unknown=1.)
    pd.testing.assert_frame_equal(protocol_inputs(data), protocol_inputs(poisoned))


@pytest.mark.parametrize("target", ["last_spring_frost", "first_fall_frost"])
def test_later_observations_cannot_change_protocol_inputs_or_predictions(cfg, daily, target):
    original = build_station_protocol_dataset(daily, "demo", cfg, target, 0.)
    changed = daily.copy()
    changed.loc[changed.date >= "2007-01-01", "tmin_c"] = -10.
    revised = build_station_protocol_dataset(changed, "demo", cfg, target, 0.)
    old_history, old_future, old_context = protocol_prediction_data(original, 2008, "2008-01-01")
    new_history, new_future, new_context = protocol_prediction_data(revised, 2008, "2008-01-01")
    pd.testing.assert_frame_equal(old_history, new_history)
    pd.testing.assert_frame_equal(old_future, new_future)
    assert old_context == new_context
    old_backtest, _, _ = walk_forward_backtest(original, "climatology", cfg, protocol=PROTOCOL_ID)
    new_backtest, _, _ = walk_forward_backtest(revised, "climatology", cfg, protocol=PROTOCOL_ID)
    pd.testing.assert_frame_equal(old_backtest[old_backtest.season_year <= 2006],
                                  new_backtest[new_backtest.season_year <= 2006])
    # The 2008 outcome can change, but its prediction must stay unchanged.
    columns = ["pred_p10", "pred_p50", "pred_p90", "training_end_year"]
    pd.testing.assert_frame_equal(old_backtest.loc[old_backtest.season_year == 2008, columns],
                                  new_backtest.loc[new_backtest.season_year == 2008, columns])


def test_partial_and_missing_years_remain_unavailable(cfg, daily):
    incomplete = daily.loc[~daily.date.eq("2006-01-10") & ~daily.date.dt.year.eq(2004)]
    data = build_station_protocol_dataset(incomplete, "demo", cfg, "last_spring_frost", 0.).set_index("season_year")
    assert data.loc[2006, "target_status"] == "incomplete_window"
    assert data.loc[2004, "target_status"] == "insufficient_data"
    assert pd.isna(data.loc[2008, "target_lag2"]) and data.loc[2008, "target_lag2_missing"] == 1
    assert pd.isna(data.loc[2008, "target_roll3_lag2"])  # Only 2005 is confirmed.


def test_protocol_normalizes_leap_calendar_and_honors_station_windows(cfg, daily):
    daily = daily.copy()
    daily["tmin_c"] = 5.
    daily.loc[daily.date.isin(pd.to_datetime(["2000-03-01", "2001-03-01"])), "tmin_c"] = 0.
    data = build_station_protocol_dataset(daily, "demo", cfg, "last_spring_frost", 0.).set_index("season_year")
    assert data.loc[2000, "target_doy"] == data.loc[2001, "target_doy"] == 61
    override = replace(cfg, station_windows={"demo": {"last_spring_frost": {"start": "01-01", "end": "07-31"}}})
    daily.loc[daily.date.eq("2000-07-01"), "tmin_c"] = 0.
    extended = build_station_protocol_dataset(daily, "demo", override, "last_spring_frost", 0.).set_index("season_year")
    assert extended.loc[2000, "target_date"] == "2000-07-01"


def test_walk_forward_uses_only_protocol_features_and_embargoed_training(cfg, monkeypatch):
    seen = []

    class SpyModel:
        def fit(self, X, y, **kwargs):
            assert X.columns.tolist() == FEATURE_COLUMNS
            self.latest = X.season_year.max()
            self.value = float(y.median())

        def predict(self, X, config):
            assert X.columns.tolist() == FEATURE_COLUMNS
            assert self.latest <= X.iloc[0].season_year - 2
            seen.append(int(X.iloc[0].season_year))
            return pd.DataFrame([[self.value - 1, self.value, self.value + 1]], columns=["p10", "p50", "p90"])

    monkeypatch.setattr("idwx.eval.create_model", lambda *args: SpyModel())
    data = build_protocol_dataset(cfg, "last_spring_frost", 0.)
    yearly, summary, _ = walk_forward_backtest(data, "spy", cfg, protocol=PROTOCOL_ID)
    assert seen and summary["n_evaluated"] > 0
    assert (yearly.forecast_protocol == PROTOCOL_ID).all()
    assert (yearly.issue_date == yearly.season_year.astype(str) + "-01-01").all()


@pytest.mark.parametrize("year, issue", [(None, None), (2008, None), (2008, "2008-02-01"),
                                         (2008, "2007-01-01"), (2008, "2008-1-1"), (1, "0001-01-01")])
def test_protocol_requires_explicit_matching_issue_context(year, issue):
    with pytest.raises(ValueError):
        issue_context(year, issue)


def test_missing_provenance_and_unsupported_targets_fail(cfg):
    data = build_protocol_dataset(cfg, "last_spring_frost", 0.)
    data.attrs = {}
    with pytest.raises(ValueError, match="provenance"):
        protocol_training_data(data, 2008)
    with pytest.raises(ValueError, match="only spring and fall"):
        build_protocol_dataset(cfg, "wsi", 0.)


def test_protocol_cli_bypasses_derived_caches_and_records_forecast_context(cfg, monkeypatch):
    monkeypatch.setattr("idwx.cli.load_config", lambda *args: cfg)
    monkeypatch.setattr("idwx.cli.load_or_build_dataset", lambda *args, **kwargs: pytest.fail("Legacy cache read"))
    # These must never be consulted by the new protocol.
    for directory in ("features", "targets", "datasets"):
        (cfg.cache_dir / directory).mkdir()
        (cfg.cache_dir / directory / "poison.parquet").write_bytes(b"not parquet")
    runner = CliRunner()
    common = ["--target", "last_spring_frost", "--protocol", PROTOCOL_ID, "--threshold", "-2"]
    context = ["--season-year", "2008", "--issue-date", "2008-01-01"]
    trained = runner.invoke(app, ["train", *common, *context, "--models", "climatology,trend"])
    assert trained.exit_code == 0, trained.output
    artifacts = json.loads(trained.output)
    assert artifacts["count"] == 2
    metadata = yaml.safe_load(Path(artifacts["trained"][0], "config.yml").read_text(encoding="utf-8"))
    assert metadata["forecast_context"]["information_cutoff_date"] == "2006-12-31"
    assert metadata["input_provenance"]["forecast_protocol"] == PROTOCOL_ID
    predicted = runner.invoke(app, ["predict", *common, *context, "--station", "demo", "--model", "climatology"])
    assert predicted.exit_code == 0, predicted.output
    output = json.loads(predicted.output)
    assert output["season_year"] == 2008 and output["issue_date"] == "2008-01-01"
    assert output["calendar_reference_year"] == 2000
    assert output["last_observation_date"] == output["information_cutoff_date"] == "2006-12-31"
    assert output["training_period"]["end_year"] == 2006
    assert output["validation_status"] == "unvalidated_research"
    assert set(output["prediction"]) == {"lower", "point", "upper"}
    wrong_year = runner.invoke(app, ["predict", *common, "--season-year", "2009", "--issue-date", "2009-01-01",
                                    "--station", "demo", "--model", "climatology"])
    assert wrong_year.exit_code != 0 and "season_2009" in wrong_year.output
    evaluated = runner.invoke(app, ["eval", *common, "--models", "climatology"])
    assert evaluated.exit_code == 0, evaluated.output
    report = json.loads(evaluated.output)["outputs"][0]
    assert report["report_dir"].endswith(PROTOCOL_ID)
    # A matching directory cannot override mismatched input-version metadata.
    metadata["forecast_context"]["input_version"] = 99
    Path(artifacts["trained"][0], "config.yml").write_text(yaml.safe_dump(metadata), encoding="utf-8")
    monkeypatch.setattr("idwx.models.climatology.ClimatologyModel.load",
                        lambda *args: pytest.fail("Incompatible protocol pickle loaded"))
    incompatible = runner.invoke(app, ["predict", *common, *context, "--station", "demo", "--model", "climatology"])
    assert incompatible.exit_code != 0 and "Incompatible model artifact" in incompatible.output


def test_cli_rejects_missing_context_before_reading_private_data(cfg, monkeypatch):
    monkeypatch.setattr("idwx.cli.load_config", lambda *args: cfg)
    monkeypatch.setattr("idwx.cli.build_protocol_dataset", lambda *args, **kwargs: pytest.fail("Read before validation"))
    result = CliRunner().invoke(app, ["train", "--target", "last_spring_frost", "--protocol", PROTOCOL_ID])
    assert result.exit_code != 0 and "explicit" in result.output


@pytest.mark.parametrize("values, valid", [([1., 61., 366.], True), ([0., 1., 2.], False),
                                          ([1., 2., 367.], False), ([3., 2., 1.], False),
                                          ([1., float("nan"), 3.], False)])
def test_protocol_rejects_invalid_calendar_bounds_without_clipping(values, valid):
    assert valid_calendar_bounds(values) is valid


def test_protocol_rf_train_predict_round_trip(cfg, monkeypatch):
    cfg = replace(cfg, models={**cfg.models, "rf": {"n_estimators": 5, "random_state": 1}})
    monkeypatch.setattr("idwx.cli.load_config", lambda *args: cfg)
    runner = CliRunner()
    args = ["--target", "last_spring_frost", "--protocol", PROTOCOL_ID,
            "--season-year", "2012", "--issue-date", "2012-01-01"]
    trained = runner.invoke(app, ["train", *args, "--models", "rf,climatology"])
    assert trained.exit_code == 0, trained.output
    assert json.loads(trained.output)["count"] == 2
    predicted = runner.invoke(app, ["predict", *args, "--station", "demo", "--model", "rf"])
    assert predicted.exit_code == 0, predicted.output
    output = json.loads(predicted.output)
    assert output["prediction_kind"] == "unvalidated_model_interval"
    assert valid_calendar_bounds(list(output["prediction"].values()))
