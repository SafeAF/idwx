from dataclasses import replace

import pandas as pd
import pytest

from idwx.config import StationMeta, load_config
from idwx.io import _compute_local_date, ingest_era5_hourly_csv


def write_csv(tmp_path, timestamps):
    path = tmp_path / "42.5629---114.4609-hourly.csv"
    pd.DataFrame({"timestamp": timestamps, "temperature (degC)": -1.0}).to_csv(path, index=False)
    return path


def test_timezone_and_explicit_offset():
    ts = pd.Series(["2020-01-01T07:30:00Z", "2020-01-01T07:30:00Z"])
    dates = _compute_local_date(ts, pd.Series([-8, None]), "America/Boise")
    assert dates.tolist() == [pd.Timestamp("2019-12-31"), pd.Timestamp("2020-01-01")]


def test_station_timezone_override(tmp_path):
    cfg = replace(load_config("configs/idaho.yml"), hourly_min_coverage=1)
    path = write_csv(tmp_path, ["2020-01-01T07:30:00Z"])
    station = StationMeta("twin_falls", "Twin Falls", 42.5629, -114.4609, timezone="America/Los_Angeles")
    _, daily, _ = ingest_era5_hourly_csv(path, cfg, [station])
    assert daily.iloc[0].date == pd.Timestamp("2019-12-31")


def test_subhourly_readings_cannot_inflate_coverage(tmp_path):
    cfg = load_config("configs/idaho.yml")
    times = pd.date_range("2020-01-01T12:00:00Z", periods=18, freq="min")
    _, daily, _ = ingest_era5_hourly_csv(write_csv(tmp_path, times), cfg, [], chunksize=5)
    assert daily.iloc[0].n_obs == 1
    assert pd.isna(daily.iloc[0].tmin_c)
    assert daily.iloc[0].quality_flags == "insufficient_coverage"


def test_eighteen_distinct_hours_meet_coverage_across_chunks(tmp_path):
    cfg = load_config("configs/idaho.yml")
    times = pd.date_range("2020-01-01T07:00:00Z", periods=18, freq="h")
    _, daily, _ = ingest_era5_hourly_csv(write_csv(tmp_path, times), cfg, [], chunksize=5)
    assert len(daily) == 1
    assert daily.iloc[0].n_obs == 18
    assert daily.iloc[0].tmin_c == -1


@pytest.mark.parametrize("chunksize", [1, 100])
def test_duplicate_hourly_timestamps_rejected_across_chunks(tmp_path, chunksize):
    cfg = load_config("configs/idaho.yml")
    path = write_csv(tmp_path, ["2020-01-01T12:00:00Z"] * 2)
    with pytest.raises(ValueError, match="Duplicate hourly"):
        ingest_era5_hourly_csv(path, cfg, [], chunksize=chunksize)
