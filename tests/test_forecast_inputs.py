"""Leakage regressions; these synthetic checks do not validate forecasts."""
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from idwx.config import load_config
from idwx.datasets import build_future_row, build_station_dataset
from idwx.features import add_target_lags, build_features_cache, build_yearly_features
from idwx.models.rf import RFConformalModel


def test_target_lags_use_calendar_years_and_preserve_input_order():
    features = pd.DataFrame({"station_id": "demo", "season_year": [2004, 2000, 2002, 2003]})
    values = pd.DataFrame({"season_year": [2000, 2002, 2003], "target": [10., 30., 40.]})
    result = add_target_lags(features, values, "target").set_index("season_year")
    assert result.index.tolist() == features.season_year.tolist()
    assert np.isnan(result.loc[2002, "target_lag1"])
    assert np.isnan(result.loc[2002, "target_roll3"])  # Only one of the previous three years.
    assert result.loc[2003, "target_roll3"] == 20
    assert result.loc[2004, "target_lag1"] == 40
    assert result.loc[2004, "target_roll3"] == 35  # 2001 is missing; 2000 is outside the window.
    assert result.loc[2004, "target_roll5"] == pytest.approx(80 / 3)


def test_conflicting_target_years_fail_instead_of_duplicating_rows():
    with pytest.raises(ValueError, match="one value per season year"):
        add_target_lags(pd.DataFrame({"season_year": [2001]}),
                        pd.DataFrame({"season_year": [2000, 2000], "target": [10, 20]}), "target")


def sparse_daily():
    return pd.DataFrame({
        "station_id": "demo", "date": pd.to_datetime(["2000-01-01", "2001-07-01", "2003-07-01"]),
        "tmin_c": [1., 10., 20.], "tmax_c": [10., 20., 30.], "tmean_c": [5., 15., 25.],
    })


def test_weather_lags_do_not_bridge_absent_years():
    targets = pd.DataFrame(columns=["target_name", "season_year", "value_doy", "value_float"])
    features = build_yearly_features(sparse_daily(), "demo", targets, "last_spring_frost", 0)
    result = features.set_index("season_year")
    assert np.isnan(result.loc[2003, "summer_tmin_mean_lag1"])
    assert np.isnan(result.loc[2000, "elevation_m"])  # Optional station metadata.


def test_feature_and_dataset_caches_preserve_missingness(tmp_path):
    cfg = replace(load_config("configs/idaho.yml"), cache_dir=tmp_path,
                  backtest={"start_year": 2000, "end_year": 2003})
    (tmp_path / "daily").mkdir()
    sparse_daily().to_parquet(tmp_path / "daily/demo.parquet", index=False)
    build_features_cache(cfg, "last_spring_frost", 0)
    cached = pd.read_parquet(tmp_path / "features/last_spring_frost_thr0p0/demo.parquet")
    assert np.isnan(cached.iloc[0].summer_tmin_mean)
    assert cached.iloc[0].summer_tmin_mean_imputed == 1
    result = build_station_dataset(sparse_daily(), "demo", cfg, "last_spring_frost", 0)
    assert np.isnan(result.iloc[0].summer_tmin_mean)
    assert result.iloc[0].summer_tmin_mean_imputed == 1


def test_future_row_carries_only_static_and_explicit_lags():
    history = pd.DataFrame({
        "station_id": "demo", "season_year": [2000, 2002, 2003], "year_index": [2000, 2002, 2003],
        "elevation_m": 1000., "target_doy": [10., 30., 40.],
        "summer_tmin_mean": [1., 2., 3.], "summer_tmin_mean_lag1": [np.nan, np.nan, 2.],
        "summer_tmin_mean_imputed": 0, "winter_tmean": [4., 5., 6.], "wsi": [0., 1., 2.],
        "unknown_seasonal_feature": 99.,
    })
    future = build_future_row(history).iloc[0]
    assert future.season_year == future.year_index == 2004
    assert future.station_id == "demo" and future.elevation_m == 1000
    assert future.summer_tmin_mean_lag1 == 3
    assert future.wsi_lag1 == 2
    assert future.summer_tmin_mean_imputed == 1
    assert future.target_lag1 == 40
    assert future.target_roll3 == 35
    assert future.target_roll5 == pytest.approx(80 / 3)
    for col in ("summer_tmin_mean", "winter_tmean", "wsi", "unknown_seasonal_feature", "target_doy"):
        assert pd.isna(future[col])


def test_future_rolling_history_requires_minimum_year_counts():
    future = build_future_row(pd.DataFrame({"station_id": ["demo"], "season_year": [2000],
                                          "target_doy": [100.]})).iloc[0]
    assert future.target_lag1 == 100
    assert pd.isna(future.target_roll3) and pd.isna(future.target_roll5)


@pytest.mark.parametrize("stations, years", [(["a", "b"], [2000, 2001]), (["a", "a"], [2000, 2000])])
def test_future_rejects_ambiguous_history(stations, years):
    with pytest.raises(ValueError, match="single station and unique season years"):
        build_future_row(pd.DataFrame({"station_id": stations, "season_year": years}))


def test_rf_does_not_learn_medians_from_calibration_or_prediction():
    X = pd.DataFrame({"station_id": "demo", "season_year": range(2000, 2012),
                      "weather": [np.nan] * 9 + [100., 200., 300.]})
    model = RFConformalModel(n_estimators=5)
    y = pd.Series(np.arange(12, dtype=float))
    model.fit(X, y, meta=None, config={})
    assert model.feature_medians == {}  # All-null training feature uses fixed zero fallback.
    model.predict(X.iloc[-1:], {})
    assert model.feature_medians == {}
    # Refit must also replace the previous fit's feature schema and medians.
    model.fit(X.rename(columns={"weather": "new_weather"}).fillna({"new_weather": 7}), y, None, {})
    assert model.feature_cols == ["new_weather"]
    assert model.feature_medians == {"new_weather": 7.}
