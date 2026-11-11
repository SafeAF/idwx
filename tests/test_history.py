import pandas as pd
import pytest

from idwx.history import calendar_day, summarize_events


def test_calendar_mapping_preserves_month_day_across_leap_years():
    assert calendar_day(pd.Timestamp("2019-03-01")) == calendar_day(pd.Timestamp("2020-03-01")) == 61
    assert calendar_day(pd.Timestamp("2020-02-29")) == 60


def test_summary_uses_only_confirmed_events_in_baseline():
    events = [
        {"season_year": 2018, "date": "2018-04-30", "status": "observed"},
        {"season_year": 2019, "date": "2019-03-01", "status": "observed"},
        {"season_year": 2020, "date": "2020-03-01", "status": "observed"},
        {"season_year": 2021, "date": None, "status": "incomplete_window"},
        {"season_year": 2022, "date": None, "status": "no_event_in_complete_window"},
    ]
    result = summarize_events(events, 2019, 2022, min_years=2)
    assert result["valid_year_count"] == 2
    assert result["valid_years"] == [2019, 2020]
    assert result["status_counts"] == {"observed": 2, "incomplete_window": 1, "no_event_in_complete_window": 1}
    assert result["p10"] == result["p50"] == result["p90"] == {"calendar_day": 61, "month_day": "03-01"}


def test_quantiles_are_observed_dates_with_explicit_minimum_history():
    events = [{"season_year": 2000 + i, "date": f"{2000 + i}-05-{i + 1:02}", "status": "observed"}
              for i in range(10)]
    result = summarize_events(events, 2000, 2009)
    assert result["status"] == "available"
    assert [result[key]["month_day"] for key in ("p10", "p50", "p90")] == ["05-01", "05-05", "05-09"]
    sparse = summarize_events(events[:1], 2000, 2009)
    assert sparse["status"] == "insufficient_history"
    assert sparse["earliest"]["month_day"] == "05-01"
    assert sparse["p10"] is sparse["p50"] is sparse["p90"] is None
    empty = summarize_events([], 2000, 2009)
    assert empty["earliest"] is empty["latest"] is None


@pytest.mark.parametrize("period,minimum", [((2020, 2019), 10), ((2019, 2020), 0)])
def test_invalid_summary_settings(period, minimum):
    with pytest.raises(ValueError):
        summarize_events([], *period, min_years=minimum)
