"""Conservative, opt-in frost research inputs. This is not a publication gate."""
from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

from idwx.config import Config, load_stations
from idwx.forecast_dates import DATE_POLICY_VERSION
from idwx.forecast_dates import valid_calendar_bounds as _valid_calendar_bounds
from idwx.frost import validated_daily, window_event
from idwx.history import CALENDAR_YEAR, calendar_day

PROTOCOL_ID = "frost_jan01_v1"
INPUT_VERSION = 2
FROST_TARGETS = {"last_spring_frost", "first_fall_frost"}
FEATURE_COLUMNS = ["station_id", "season_year", "year_index", "target_lag2",
                   "target_roll3_lag2", "target_roll5_lag2", "target_lag2_missing"]


def validate_protocol(protocol: str) -> None:
    if protocol != PROTOCOL_ID:
        raise ValueError(f"Unsupported forecast protocol: {protocol!r}; expected {PROTOCOL_ID}")


def valid_calendar_bounds(values: list[float] | np.ndarray) -> bool:
    """Compatibility entry point for the shared calendar validator."""
    return _valid_calendar_bounds(values)


def issue_context(season_year: int | None, issue_date: str | None) -> dict:
    if season_year is None or issue_date is None:
        raise ValueError("This protocol requires explicit --season-year and --issue-date")
    if not 3 <= season_year <= 9999:
        raise ValueError("Forecast season year must be between 3 and 9999")
    expected = date(season_year, 1, 1).isoformat()
    if issue_date != expected:
        raise ValueError(f"{PROTOCOL_ID} requires issue date {expected} (station-local January 1)")
    return {"protocol_id": PROTOCOL_ID, "input_version": INPUT_VERSION,
            "date_policy_version": DATE_POLICY_VERSION,
            "calendar_reference_year": CALENDAR_YEAR, "season_year": season_year,
            "issue_date": issue_date, "information_cutoff_date": date(season_year - 2, 12, 31).isoformat()}


def _features(values: pd.Series, year: int) -> dict:
    # A full intervening calendar year is held back. Missing years stay missing.
    prior = values.reindex(range(year - 6, year - 1)).replace([np.inf, -np.inf], np.nan)
    lag = prior.loc[year - 2]
    return {"year_index": year, "target_lag2": lag,
            "target_roll3_lag2": prior.tail(3).mean() if prior.tail(3).notna().sum() >= 2 else np.nan,
            "target_roll5_lag2": prior.mean() if prior.notna().sum() >= 3 else np.nan,
            "target_lag2_missing": int(pd.isna(lag))}


def build_station_protocol_dataset(
    daily: pd.DataFrame, station_id: str, cfg: Config, target: str, threshold_c: float,
    *, through_year: int | None = None, timezone: str | None = None,
) -> pd.DataFrame:
    if target not in FROST_TARGETS:
        raise ValueError("The preseason protocol supports only spring and fall frost dates")
    if not np.isfinite(threshold_c):
        raise ValueError("Protocol threshold must be finite")
    df = validated_daily(daily, station_id)
    if df.empty:
        raise ValueError(f"No daily observations for configured station: {station_id}")
    observed_start, observed_end = int(df.date.dt.year.min()), int(df.date.dt.year.max())
    start = min(observed_start, int(cfg.backtest.get("start_year", observed_start)))
    end = max(observed_end, int(cfg.backtest.get("end_year", observed_end)), through_year or observed_end)
    if start < 7 or end > 9999:
        raise ValueError("Protocol input years must be between 7 and 9999")
    window = {**cfg.windows, **cfg.station_windows.get(station_id, {})}[target]
    events = {year: window_event(df, year, threshold_c, window, last=target == "last_spring_frost")
              for year in range(start, end + 1)}
    values = pd.Series({year: float(calendar_day(event["value_date"])) if event["status"] == "observed" else np.nan
                        for year, event in events.items()})
    rows = []
    for year, event in events.items():
        context = issue_context(year, f"{year:04d}-01-01")
        available = df.loc[df.date <= pd.Timestamp(context["information_cutoff_date"]), "date"]
        rows.append({"station_id": station_id, "season_year": year, **_features(values, year),
                     "target_doy": values.loc[year], "target_status": event["status"],
                     "target_date": event["value_date"].date().isoformat() if event["status"] == "observed" else None,
                     "issue_date": context["issue_date"], "information_cutoff_date": context["information_cutoff_date"],
                     "last_observation_date": available.max().date().isoformat() if len(available) else None,
                     "window_start_month_day": window["start"], "window_end_month_day": window["end"],
                     "station_timezone": timezone or cfg.timezone})
    result = pd.DataFrame(rows)
    result.attrs = {"forecast_protocol": PROTOCOL_ID, "input_version": INPUT_VERSION,
                    "date_policy_version": DATE_POLICY_VERSION,
                    "calendar_reference_year": CALENDAR_YEAR, "target": target, "threshold_c": float(threshold_c)}
    return result


def build_protocol_dataset(cfg: Config, target: str, threshold_c: float, *, through_year: int | None = None) -> pd.DataFrame:
    frames = []
    for station in load_stations(cfg.stations_file):
        path = cfg.cache_dir / "daily" / f"{station.station_id}.parquet"
        if not path.is_file():
            raise FileNotFoundError(f"Missing daily cache for protocol: {path}")
        frames.append(build_station_protocol_dataset(pd.read_parquet(path), station.station_id, cfg, target,
                                                     threshold_c, through_year=through_year, timezone=station.timezone))
    if not frames:
        raise ValueError("No configured stations for protocol")
    result = pd.concat(frames, ignore_index=True)
    result.attrs = frames[0].attrs.copy()
    return result


def protocol_inputs(dataset: pd.DataFrame) -> pd.DataFrame:
    if (dataset.attrs.get("forecast_protocol") != PROTOCOL_ID
            or dataset.attrs.get("input_version") != INPUT_VERSION
            or dataset.attrs.get("date_policy_version") != DATE_POLICY_VERSION
            or dataset.attrs.get("calendar_reference_year") != CALENDAR_YEAR):
        raise ValueError("Missing or incompatible protocol input provenance; rebuild from daily caches")
    return dataset[FEATURE_COLUMNS].copy()


def protocol_training_data(dataset: pd.DataFrame, season_year: int) -> pd.DataFrame:
    protocol_inputs(dataset)  # Check provenance before accepting precomputed features.
    values = pd.to_numeric(dataset.target_doy, errors="raise")
    return dataset.loc[(dataset.season_year <= season_year - 2) & np.isfinite(values)
                       & dataset.target_status.eq("observed")].copy()


def protocol_prediction_data(dataset: pd.DataFrame, season_year: int, issue_date: str) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    context = issue_context(season_year, issue_date)
    if dataset.station_id.nunique() != 1 or dataset.season_year.duplicated().any():
        raise ValueError("Protocol prediction requires one station and unique season years")
    future = dataset.loc[dataset.season_year == season_year].copy()
    if len(future) != 1:
        raise ValueError("Rebuild protocol inputs through the explicit forecast year")
    if (future.iloc[0].issue_date != context["issue_date"]
            or future.iloc[0].information_cutoff_date != context["information_cutoff_date"]):
        raise ValueError("Protocol row does not match the explicit issue date and information cutoff")
    history = protocol_training_data(dataset, season_year)
    if history.empty:
        raise ValueError("No confirmed history before the protocol information cutoff")
    context.update(last_observation_date=future.iloc[0].last_observation_date,
                   station_timezone=future.iloc[0].station_timezone,
                   target_window={"start": future.iloc[0].window_start_month_day,
                                  "end": future.iloc[0].window_end_month_day})
    # The actual outcome is deliberately absent from prediction inputs.
    return history, protocol_inputs(future), context
