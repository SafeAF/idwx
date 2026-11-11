"""Map research calendar coordinates onto supported dates without changing bounds."""
from __future__ import annotations

from calendar import isleap
from datetime import date, timedelta
from math import ceil, floor

import numpy as np

from idwx.history import CALENDAR_YEAR, calendar_day

DATE_POLICY_VERSION = 1


def valid_calendar_bounds(values: list[float] | np.ndarray) -> bool:
    bounds = np.asarray(values, dtype=float)
    return bool(bounds.shape == (3,) and np.isfinite(bounds).all()
                and 1 <= bounds[0] <= bounds[1] <= bounds[2] <= 366)


def window_calendar_bounds(window: dict[str, str]) -> tuple[int, int]:
    try:
        start, end = (date.fromisoformat(f"{CALENDAR_YEAR}-{window[key]}") for key in ("start", "end"))
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Forecast window must contain valid MM-DD start and end values") from exc
    if end < start:
        raise ValueError("Forecast window must be ordered within one calendar year")
    return calendar_day(start), calendar_day(end)


def interpret_forecast_dates(values: list[float] | np.ndarray, season_year: int, window: dict[str, str]) -> dict:
    """Represent precisely the supported discrete dates in a numeric interval.

    Endpoints select the first/last actual date whose common-calendar coordinate
    is inside the unchanged bounds. The point selects the nearest supported
    coordinate, choosing the later date on a tie. No February 29 is invented.
    """
    if not valid_calendar_bounds(values):
        raise ValueError("Forecast requires finite ordered calendar bounds within [1, 366]")
    if isinstance(season_year, bool) or not isinstance(season_year, (int, np.integer)) or not 1 <= season_year <= 9999:
        raise ValueError("Forecast date conversion requires an integer year between 1 and 9999")
    start, end = window_calendar_bounds(window)
    lower, point, upper = np.asarray(values, dtype=float)
    if lower < start or upper > end:
        raise ValueError(f"Forecast bounds are outside the configured {window['start']} through {window['end']} window")
    first = date(CALENDAR_YEAR, 1, 1)
    supported = []
    for coordinate in range(ceil(lower), floor(upper) + 1):
        reference = first + timedelta(days=coordinate - 1)
        if reference.month == 2 and reference.day == 29 and not isleap(int(season_year)):
            continue
        supported.append((date(int(season_year), reference.month, reference.day), coordinate))
    if not supported:
        raise ValueError("Forecast interval contains no supported date in the forecast year")
    point_date, _ = min(supported, key=lambda item: (abs(item[1] - point), -item[1]))
    return {"lower": supported[0][0].isoformat(), "point": point_date.isoformat(),
            "upper": supported[-1][0].isoformat(),
            "interval_width_days": (supported[-1][0] - supported[0][0]).days}
