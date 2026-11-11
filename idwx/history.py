"""Historical freeze-date summaries on a shared leap-year calendar."""
from __future__ import annotations

from collections import Counter

import numpy as np
import pandas as pd

CALENDAR_YEAR = 2000


def calendar_day(date: pd.Timestamp) -> int:
    """Map month/day to 1..366 without shifting March-December in leap years."""
    return pd.Timestamp(year=CALENDAR_YEAR, month=date.month, day=date.day).day_of_year


def summarize_events(events: list[dict], start_year: int, end_year: int, min_years: int = 10) -> dict:
    if start_year > end_year or min_years < 1:
        raise ValueError("Baseline years must be ordered and minimum history must be positive")
    selected = [e for e in events if start_year <= e["season_year"] <= end_year]
    observed = [e for e in selected if e["status"] == "observed"]
    days = [calendar_day(pd.Timestamp(e["date"])) for e in observed]

    def point(day: int) -> dict:
        date = pd.Timestamp(f"{CALENDAR_YEAR}-01-01") + pd.Timedelta(days=int(day) - 1)
        return {"calendar_day": int(day), "month_day": date.strftime("%m-%d")}

    enough = len(days) >= min_years
    return {
        "kind": "historical_distribution",
        "status": "available" if enough else "insufficient_history",
        "baseline_period": {"start_year": start_year, "end_year": end_year},
        "calendar_reference_year": CALENDAR_YEAR,
        "quantile_method": "inverted_cdf",
        "minimum_valid_years": min_years,
        "valid_year_count": len(days),
        "valid_years": [e["season_year"] for e in observed],
        "status_counts": dict(sorted(Counter(e["status"] for e in selected).items())),
        "earliest": point(min(days)) if days else None,
        "latest": point(max(days)) if days else None,
        "p10": point(np.quantile(days, .1, method="inverted_cdf")) if enough else None,
        "p50": point(np.quantile(days, .5, method="inverted_cdf")) if enough else None,
        "p90": point(np.quantile(days, .9, method="inverted_cdf")) if enough else None,
    }
