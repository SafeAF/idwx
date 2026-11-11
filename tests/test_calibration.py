from dataclasses import replace

import joblib
import numpy as np
import pandas as pd
import pytest
import yaml

from idwx.config import load_config
from idwx.eval import walk_forward_backtest
from idwx.models import create_model
from idwx.models.conformal import (
    CALIBRATION_VERSION,
    InsufficientCalibrationError,
    calibration_rank,
    residual_quantile,
)
from idwx.models.rf import RFConformalModel
from idwx.registry import latest_model_dir, save_model_artifacts


def inputs():
    return pd.DataFrame({"station_id": "demo", "season_year": range(2000, 2020),
                         "weather": np.arange(20, dtype=float)})


def targets():
    return pd.Series(np.arange(100., 120.))


class SpyForest:
    def __init__(self, invalid_prediction=False):
        self.fit_calls = []
        self.predict_calls = []
        self.invalid_prediction = invalid_prediction

    def fit(self, X, y):
        self.fit_calls.append((X.copy(), y.copy()))
        self.point = float(y.mean())

    def predict(self, X):
        self.predict_calls.append(X.copy())
        return np.full(len(X), np.nan if self.invalid_prediction else self.point)


@pytest.mark.parametrize("n, alpha, rank", [(4, .2, 4), (5, .2, 5), (9, .2, 8), (19, .05, 19)])
def test_finite_sample_rank_uses_order_statistic_without_interpolation(n, alpha, rank):
    assert calibration_rank(n, alpha) == rank
    actual = np.arange(n, 0, -1, dtype=float)
    assert residual_quantile(actual, np.zeros(n), alpha) == rank


def test_zero_residuals_are_a_supported_zero_width_not_a_fallback():
    assert residual_quantile(np.ones(5), np.ones(5), .2) == 0.


@pytest.mark.parametrize("n, alpha", [(0, .2), (3, .2), (5, .01)])
def test_unsupported_finite_rank_fails_instead_of_clipping(n, alpha):
    with pytest.raises(InsufficientCalibrationError):
        residual_quantile(np.ones(n), np.ones(n), alpha)


@pytest.mark.parametrize("alpha", [0., 1., -.1, float("nan"), float("inf")])
def test_invalid_alpha_is_rejected(alpha):
    with pytest.raises(ValueError, match="alpha"):
        residual_quantile(np.ones(5), np.ones(5), alpha)
    with pytest.raises(ValueError, match="alpha"):
        RFConformalModel(alpha=alpha)


@pytest.mark.parametrize("actual, predicted", [([1., 2.], [1.]), ([[1.]], [[1.]]),
                                             ([float("nan")] * 5, [1.] * 5),
                                             ([1.] * 5, [float("inf")] * 5)])
def test_bad_residual_inputs_are_not_broadcast_or_silently_dropped(actual, predicted):
    with pytest.raises(ValueError):
        residual_quantile(actual, predicted)


def test_overflowed_residuals_are_rejected():
    with pytest.raises(ValueError, match="overflowed"):
        residual_quantile(np.full(5, 1e308), np.full(5, -1e308))


def test_rf_fits_once_and_uses_only_held_out_residuals():
    model = RFConformalModel(n_estimators=5)
    spy = SpyForest()
    model.model = spy
    X, y = inputs(), targets()
    X.loc[14, "weather"] = 1e10  # Entire year 2014 is embargoed.
    y.loc[14] = 1e10
    model.fit(X, y, None, {})
    assert len(spy.fit_calls) == len(spy.predict_calls) == 1
    assert spy.fit_calls[0][0].weather.tolist() == list(range(14))
    assert spy.predict_calls[0].weather.tolist() == list(range(15, 20))
    assert model.feature_medians == {"weather": 6.5}
    assert model.qhat == 12.5  # Largest of five calibration errors about fitting mean 106.5.
    assert model.metadata["fitting_end_year"] == 2013
    assert model.metadata["calibration_start_year"] == 2015
    assert model.metadata["n_calib"] == model.metadata["n_calibration_years"] == 5
    assert model.metadata["n_embargo"] == 1 and model.metadata["calibration_rank"] == 5
    assert model.metadata["nominal_coverage"] == .8
    assert model.metadata["interval_semantics"] == "symmetric_absolute_residual_interval"


def test_calibration_changes_cannot_fit_preprocessing_or_refit_forest():
    models = []
    for changed in (False, True):
        model = RFConformalModel(n_estimators=5)
        model.model = SpyForest()
        X, y = inputs(), targets()
        if changed:
            X.loc[14:, "weather"] = -1e9
            y.loc[15:] += 100.
        model.fit(X, y, None, {})
        models.append(model)
    assert models[0].feature_medians == models[1].feature_medians
    assert models[0].model.point == models[1].model.point
    assert models[1].qhat == models[0].qhat + 100.


def test_pooled_stations_are_split_by_whole_calendar_years():
    X = pd.concat([inputs(), inputs().assign(station_id="other")], ignore_index=True)
    y = pd.Series(np.tile(targets().to_numpy(), 2))
    order = X.sample(frac=1., random_state=4).index
    model = RFConformalModel(n_estimators=5)
    model.model = SpyForest()
    model.fit(X.loc[order], y.loc[order], None, {})
    assert model.metadata["n_train"] == 28
    assert model.metadata["n_calib"] == 10
    assert model.metadata["n_calibration_years"] == 5
    assert model.metadata["n_embargo"] == 2
    assert model.metadata["fitting_end_year"] == 2013
    assert model.metadata["calibration_start_year"] == 2015


def test_calendar_gap_does_not_discard_an_extra_available_fitting_year():
    X, y = inputs().drop(index=14), targets().drop(index=14)
    model = RFConformalModel(n_estimators=5)
    model.model = SpyForest()
    model.fit(X, y, None, {})
    assert model.metadata["n_embargo"] == 0
    assert model.metadata["n_train"] == 14
    assert model.metadata["calibration_start_year"] == 2015


def test_default_split_requires_fourteen_consecutive_valid_years():
    model = RFConformalModel(n_estimators=5)
    model.model = SpyForest()
    with pytest.raises(ValueError, match="at least 8 fitting years"):
        model.fit(inputs().iloc[:13], targets().iloc[:13], None, {})
    model.fit(inputs().iloc[:14], targets().iloc[:14], None, {})
    assert model.metadata["n_fitting_years"] == 8
    assert model.metadata["n_calibration_years"] == 5
    assert model.metadata["n_embargo"] == 1


def test_bad_calibration_predictions_fail_without_training_residual_fallback(tmp_path):
    model = RFConformalModel(n_estimators=5)
    spy = SpyForest(invalid_prediction=True)
    model.model = spy
    with pytest.raises(ValueError, match="finite"):
        model.fit(inputs(), targets(), None, {})
    assert len(spy.predict_calls) == 1
    with pytest.raises(ValueError, match="held-out calibration"):
        model.predict(inputs().iloc[-1:], {})
    with pytest.raises(ValueError, match="held-out calibration"):
        model.save(tmp_path / "invalid")


@pytest.mark.parametrize("qhat", [float("nan"), float("inf"), -1.])
def test_no_fixed_width_prediction_fallback(qhat):
    model = RFConformalModel(n_estimators=5)
    model.fit(inputs(), targets(), None, {})
    model.qhat = qhat
    with pytest.raises(ValueError, match="held-out calibration"):
        model.predict(inputs().iloc[-1:], {})


def test_failed_refit_disables_old_predictions_and_save(tmp_path):
    model = RFConformalModel(n_estimators=5)
    model.fit(inputs(), targets(), None, {})
    with pytest.raises(InsufficientCalibrationError, match="held-out calibration years"):
        model.fit(inputs().iloc[:4], targets().iloc[:4], None, {})
    assert model.feature_medians == {} and np.isnan(model.qhat)
    with pytest.raises(ValueError, match="retrain"):
        model.predict(inputs().iloc[-1:], {})
    with pytest.raises(ValueError, match="retrain"):
        model.save(tmp_path / "invalid")


@pytest.mark.parametrize("kwargs", [{"calibration_fraction": 0.}, {"calibration_fraction": 1.},
                                    {"calibration_fraction": float("nan")}, {"min_calibration_years": 0},
                                    {"min_calibration_years": 2.5}, {"min_calibration_years": True}])
def test_invalid_calibration_configuration_is_rejected(kwargs):
    with pytest.raises(ValueError):
        RFConformalModel(**kwargs)


def test_factory_passes_calibration_settings_and_rejects_impossible_finite_rank():
    model = create_model("rf", {"rf": {"n_estimators": 5},
                                 "conformal": {"alpha": .01, "calibration_fraction": .25, "min_calibration_years": 6}})
    assert model.calibration_fraction == .25 and model.min_calibration_years == 6
    spy = SpyForest()
    model.model = spy
    with pytest.raises(InsufficientCalibrationError, match="required rank"):
        model.fit(inputs(), targets(), None, {})
    assert not spy.fit_calls


def test_backtest_records_calibration_partitions_and_explicit_insufficiency():
    cfg = replace(load_config("configs/idaho.yml"), backtest={"start_year": 2019, "end_year": 2019, "min_train_years": 8},
                  models={"rf": {"n_estimators": 5}, "conformal": {"alpha": .2}})
    data = inputs().assign(target_doy=targets())
    yearly, summary, _ = walk_forward_backtest(data, "rf", cfg)
    row = yearly.iloc[0]
    assert row.status == "evaluated" and row.calibration_start_year == 2014
    assert row.fitting_end_year == 2012 and row.n_calibration_rows == 5
    assert row.n_calibration_years == 5 and row.n_embargo_rows == 1
    assert summary["nominal_coverage"] == .8
    assert summary["coverage_minus_nominal"] == summary["coverage"] - .8
    cfg = replace(cfg, models={"rf": {"n_estimators": 5}, "conformal": {"alpha": .01}})
    skipped, skipped_summary, _ = walk_forward_backtest(data, "rf", cfg)
    assert skipped.iloc[0].status == "insufficient_calibration"
    assert skipped.iloc[0].n_calibration_rows == 5
    assert skipped_summary["n_evaluated"] == 0 and skipped_summary["n_skipped"] == 1


def test_calibration_model_round_trip_and_legacy_rejection(tmp_path):
    model = RFConformalModel(n_estimators=5)
    model.fit(inputs(), targets(), None, {})
    model.save(tmp_path)
    loaded = RFConformalModel.load(tmp_path)
    pd.testing.assert_frame_equal(model.predict(inputs().iloc[-1:], {}), loaded.predict(inputs().iloc[-1:], {}))
    loaded.metadata.pop("calibration_version")
    joblib.dump(loaded, tmp_path / "model.pkl")
    with pytest.raises(ValueError, match="compatible held-out calibration"):
        RFConformalModel.load(tmp_path)


def test_registry_rejects_old_rf_calibration_before_loading_pickle(tmp_path):
    cfg = replace(load_config("configs/idaho.yml"), models_dir=tmp_path / "models")
    model = RFConformalModel(n_estimators=5)
    model.fit(inputs(), targets(), None, {})
    data = inputs().assign(target_doy=targets())
    path = save_model_artifacts(model, cfg, "last_spring_frost", "demo", "rf", data, 0.)
    lookup = {"threshold_c": 0., "dataset": data}
    assert latest_model_dir(cfg, "last_spring_frost", "demo", "rf", **lookup) == path
    metadata = yaml.safe_load((path / "config.yml").read_text(encoding="utf-8"))
    assert metadata["rf_calibration_version"] == CALIBRATION_VERSION
    metadata.pop("rf_calibration_version")
    (path / "config.yml").write_text(yaml.safe_dump(metadata), encoding="utf-8")
    with pytest.raises(ValueError, match="Incompatible"):
        latest_model_dir(cfg, "last_spring_frost", "demo", "rf", **lookup)
