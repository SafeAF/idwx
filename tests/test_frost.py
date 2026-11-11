from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from idwx.config import load_config
from idwx.targets import build_targets_cache, build_targets_for_station


@pytest.fixture
def cfg():
    return replace(load_config("configs/idaho.yml"), backtest={"start_year": 2020, "end_year": 2020})


@pytest.fixture
def daily():
    dates = pd.date_range("2020-01-01", "2020-12-31")
    df = pd.DataFrame({"date": dates, "station_id": "twin_falls", "tmin_c": 5.0,
                       "tmax_c": 15.0, "tmean_c": 10.0})
    df.loc[df.date.isin(pd.to_datetime(["2020-02-29", "2020-04-30", "2020-09-15", "2020-12-31"])), "tmin_c"] = 0.0
    return df


def records(daily, cfg, threshold=0):
    out = build_targets_for_station(daily, "twin_falls", cfg)
    return out[out.threshold_c.eq(threshold)].set_index("target_name")


def test_confirmed_events_and_interval(daily, cfg):
    rows = records(daily, cfg)
    spring, fall, interval = (rows.loc[name] for name in
                              ("last_spring_frost", "first_fall_frost", "freeze_free_days"))
    assert spring.value_date == pd.Timestamp("2020-04-30")
    assert fall.value_date == pd.Timestamp("2020-09-15")
    assert spring.event_tmin_c == 0
    assert spring.expected_days == spring.valid_days == 182
    assert spring.status == "observed"
    assert interval.value_float == 138
    assert interval.frost_free_days == 137
    assert interval.status == "verified"


@pytest.mark.parametrize("mode", ["absent", "null", "infinite", "flagged"])
def test_missing_day_withholds_confirmation(daily, cfg, mode):
    index = daily.index[daily.date.eq("2020-06-01")]
    if mode == "absent":
        daily = daily.drop(index)
    elif mode == "flagged":
        daily["quality_flags"] = ""
        daily.loc[index, "quality_flags"] = "insufficient_coverage"
    else:
        daily.loc[index, "tmin_c"] = np.nan if mode == "null" else np.inf
    rows = records(daily, cfg)
    spring = rows.loc["last_spring_frost"]
    assert spring.status == "incomplete_window"
    assert spring.missing_days == 1
    assert pd.isna(spring.value_date)
    assert pd.isna(spring.value_doy)
    assert spring.candidate_date == pd.Timestamp("2020-04-30")
    assert spring.candidate_tmin_c == 0
    assert pd.isna(rows.loc["freeze_free_days"].value_float)


def test_complete_without_event_and_exact_secondary_threshold(daily, cfg):
    rows = records(daily, cfg, -2)
    assert rows.loc["last_spring_frost"].status == "no_event_in_complete_window"
    daily.loc[daily.date.eq("2020-06-30"), "tmin_c"] = -2
    daily.loc[daily.date.eq("2020-08-01"), "tmin_c"] = -2
    rows = records(daily, cfg, -2)
    assert rows.loc["last_spring_frost"].value_date == pd.Timestamp("2020-06-30")
    assert rows.loc["first_fall_frost"].value_date == pd.Timestamp("2020-08-01")


def test_missing_year_and_empty_station(daily, cfg):
    cfg.backtest = {"start_year": 2019, "end_year": 2020}
    out = build_targets_for_station(daily, "twin_falls", cfg)
    missing = out[out.season_year.eq(2019) & out.target_name.eq("last_spring_frost")]
    assert len(missing) == 2
    assert missing.status.eq("insufficient_data").all()
    assert missing.expected_days.eq(181).all()
    assert missing.valid_days.eq(0).all()
    empty = build_targets_for_station(daily.iloc[:0], "twin_falls", cfg)
    assert empty[empty.target_name.eq("last_spring_frost")].status.eq("insufficient_data").all()


def test_unfinished_season(daily, cfg):
    rows = records(daily[daily.date.le("2020-05-01")], cfg)
    assert rows.loc["last_spring_frost"].status == "incomplete_window"
    assert rows.loc["first_fall_frost"].status == "insufficient_data"


@pytest.mark.parametrize("gap", [True, False])
def test_july_does_not_silently_become_frost_free(daily, cfg, gap):
    if gap:
        daily = daily[daily.date.ne("2020-07-04")]
    else:
        daily.loc[daily.date.eq("2020-07-04"), "tmin_c"] = -3
    interval = records(daily, cfg).loc["freeze_free_days"]
    assert interval.value_float == 138  # legacy date difference remains available
    assert pd.isna(interval.frost_free_days)
    assert interval.status == ("incomplete_interval" if gap else "intervening_freeze")
    assert bool(interval.summer_freeze) is not gap


def test_duplicate_daily_records_rejected(daily, cfg):
    with pytest.raises(ValueError, match="Duplicate daily"):
        records(pd.concat([daily, daily.iloc[:1]]), cfg)


def test_station_window_override(daily, cfg):
    cfg.station_windows = {"twin_falls": {"last_spring_frost": {"start": "01-01", "end": "07-31"}}}
    daily.loc[daily.date.eq("2020-07-04"), "tmin_c"] = -1
    assert records(daily, cfg).loc["last_spring_frost"].value_date == pd.Timestamp("2020-07-04")


def test_leap_day_event(daily, cfg):
    daily.loc[daily.date.eq("2020-04-30"), "tmin_c"] = 5
    spring = records(daily, cfg).loc["last_spring_frost"]
    assert spring.value_date == pd.Timestamp("2020-02-29")
    assert spring.value_doy == 60


def test_adjacent_events_have_zero_intervening_days(daily, cfg):
    cfg.windows = {
        "last_spring_frost": {"start": "06-30", "end": "06-30"},
        "first_fall_frost": {"start": "07-01", "end": "07-01"},
    }
    daily.loc[daily.date.isin(pd.to_datetime(["2020-06-30", "2020-07-01"])), "tmin_c"] = 0
    interval = records(daily, cfg).loc["freeze_free_days"]
    assert interval.value_float == 1
    assert interval.frost_free_days == 0


def test_target_cache_roundtrip(daily, cfg, tmp_path):
    cfg.cache_dir = tmp_path
    source = tmp_path / "daily"
    source.mkdir()
    daily[daily.date.ne("2020-06-01")].to_parquet(source / "twin_falls.parquet", index=False)
    assert build_targets_cache(cfg, rebuild=True) == ["twin_falls"]
    result = pd.read_parquet(tmp_path / "targets" / "_all_targets.parquet")
    spring = result[result.target_name.eq("last_spring_frost") & result.threshold_c.eq(0)].iloc[0]
    assert spring.status == "incomplete_window"
    assert pd.isna(spring.value_date)
    assert spring.candidate_date == pd.Timestamp("2020-04-30")
