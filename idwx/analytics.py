"""Coverage-aware public temperature analytics, independent of research targets."""
from __future__ import annotations

import calendar
import math
from datetime import date, timedelta

import pandas as pd

TEMPERATURES = ("tmin_c", "tmax_c", "tmean_c")
DAILY_FIELDS = {"date", *TEMPERATURES, "status", "n_obs"}


def public_daily(frame: pd.DataFrame) -> dict[str, list[dict]]:
    """Publish full local calendars; absent/invalid days retain explicit nulls."""
    source = {pd.Timestamp(row["date"]).date().isoformat(): row for row in frame.to_dict("records")}
    first, last = min(source), max(source)
    result = {}
    for year in range(int(first[:4]), int(last[:4]) + 1):
        rows = []
        day = date(year, 1, 1)
        while day.year == year:
            key = day.isoformat()
            row = source.get(key)
            values = {k: None for k in TEMPERATURES}
            status, n_obs = "missing_record", None
            if row is not None:
                status = "invalid_temperature"
                count = row.get("n_obs")
                if count is not None and pd.notna(count) and math.isfinite(float(count)) and float(count).is_integer() and count >= 0:
                    n_obs = int(count)
                valid = "insufficient_coverage" not in str(row.get("quality_flags", ""))
                for k in TEMPERATURES:
                    value = row.get(k)
                    if value is not None and pd.notna(value) and math.isfinite(float(value)) and valid:
                        values[k] = float(value)
                if all(v is not None for v in values.values()):
                    status = "valid"
            rows.append({"date": key, **values, "status": status, "n_obs": n_obs})
            day += timedelta(days=1)
        result[str(year)] = rows
    return result


def _total(values) -> float:
    # Explicit left-to-right binary64 addition also matches the browser and
    # avoids Python-version changes to built-in sum altering release identity.
    total = 0.0
    for value in values:
        total += value
    return total


def temperature_summary(rows: list[dict]) -> dict:
    result = {"expected_days": len(rows)}
    for key in TEMPERATURES:
        values = [r[key] for r in rows if r[key] is not None]
        result[key] = {
            "valid_days": len(values), "mean": _total(values) / len(values) if values else None,
            "min": min(values) if values else None, "max": max(values) if values else None,
        }
    result["complete"] = all(result[k]["valid_days"] == len(rows) for k in TEMPERATURES)
    return result


def summarize_daily(daily: dict[str, list[dict]], policy: dict | None = None) -> dict:
    months, years = [], []
    for year, rows in daily.items():
        years.append({"year": int(year), **temperature_summary(rows)})
        for month in range(1, 13):
            selected = [r for r in rows if int(r["date"][5:7]) == month]
            months.append({"year": int(year), "month": month, **temperature_summary(selected)})
    result = {"months": months, "years": years}
    if policy is not None:
        result.update(policy=policy, seasons=seasonal_summary(daily, policy))
    return result


def validate_daily(year: int, rows: list[dict]) -> None:
    """Strict public field/calendar validation; never trusts a matching checksum."""
    if not isinstance(rows, list) or len(rows) != 365 + calendar.isleap(year):
        raise ValueError("Invalid daily analytics calendar")
    for offset, row in enumerate(rows):
        expected = (date(year, 1, 1) + timedelta(days=offset)).isoformat()
        if not isinstance(row, dict) or set(row) != DAILY_FIELDS or row["date"] != expected:
            raise ValueError("Invalid daily analytics row")
        if row["status"] not in {"valid", "missing_record", "invalid_temperature"}:
            raise ValueError("Invalid daily analytics status")
        for key in TEMPERATURES:
            value = row[key]
            if value is not None and (type(value) not in (int, float) or not math.isfinite(value)):
                raise ValueError("Invalid daily analytics temperature")
        count = row["n_obs"]
        if count is not None and (type(count) is not int or count < 0):
            raise ValueError("Invalid daily analytics coverage")
        valid = all(row[k] is not None for k in TEMPERATURES)
        if (row["status"] == "valid") != valid:
            raise ValueError("Daily analytics status disagrees with values")
        if row["status"] == "missing_record" and (count is not None or any(row[k] is not None for k in TEMPERATURES)):
            raise ValueError("Missing daily analytics record has values")


def seasonal_summary(daily: dict[str, list[dict]], policy: dict) -> list[dict]:
    rows = {r["date"]: r for days in daily.values() for r in days}

    def window(start, end):
        start, end = date.fromisoformat(start), date.fromisoformat(end)
        return [rows.get((start + timedelta(days=i)).isoformat()) for i in range((end - start).days + 1)]

    def complete(days, key):
        return bool(days) and all(r is not None and r[key] is not None for r in days)

    result = []
    for y in daily:
        year = int(y)
        growing = window(f"{year}-{policy['gdd']['start']}", f"{year}-{policy['gdd']['end']}")
        winter = window(f"{year - 1}-12-01", f"{year}-02-{calendar.monthrange(year, 2)[1]}")
        subtotal, through, run, max_run, cold_days = 0, None, 0, 0, 0
        for row in growing:
            if row is None or row["tmean_c"] is None:
                break
            subtotal += max(0, min(row["tmean_c"], policy["gdd"]["cap_c"] if policy["gdd"]["cap_c"] is not None else math.inf) - policy["gdd"]["base_c"])
            through = row["date"]
        for row in winter:
            if row is not None and row["tmin_c"] is not None and row["tmin_c"] <= policy["cold_threshold_c"]:
                run += 1
                cold_days += 1
                max_run = max(max_run, run)
            else:
                run = 0
        winter_complete = complete(winter, "tmean_c") and complete(winter, "tmin_c")
        result.append({
            "year": year, "gdd": subtotal if complete(growing, "tmean_c") else None,
            "gdd_subtotal": subtotal if through else None, "through": through,
            "gdd_valid": sum(r is not None and r["tmean_c"] is not None for r in growing), "gdd_expected": len(growing),
            "hdd": _total(max(0, policy["hdd_base_c"] - r["tmean_c"]) for r in winter) if winter_complete else None,
            "cold_days": cold_days if winter_complete else None, "cold_run": max_run if winter_complete else None,
            "winter_valid": sum(r is not None and r["tmean_c"] is not None and r["tmin_c"] is not None for r in winter), "winter_expected": len(winter),
        })
    return result


def metric_policy(cfg) -> dict:
    if cfg.winter.get("season_months", [12, 1, 2]) != [12, 1, 2]:
        raise ValueError("Public winter analytics require December–February windows")
    policy = {"gdd": {"start": cfg.gdd.get("season_start", "04-01"), "end": cfg.gdd.get("season_end", "10-31"),
                       "base_c": float(cfg.gdd.get("base_c", 10)), "cap_c": cfg.gdd.get("upper_cap_c")},
              "hdd_base_c": float(cfg.winter.get("hdd_base_c", 18)),
              "cold_threshold_c": float(cfg.winter.get("cold_snap_threshold_c", -10))}
    validate_policy(policy)
    return policy


def validate_policy(policy: dict) -> None:
    if not isinstance(policy, dict) or set(policy) != {"gdd", "hdd_base_c", "cold_threshold_c"}:
        raise ValueError("Invalid seasonal analytics policy")
    gdd = policy["gdd"]
    if not isinstance(gdd, dict) or set(gdd) != {"start", "end", "base_c", "cap_c"}:
        raise ValueError("Invalid GDD policy")
    start, end = date.fromisoformat(f"2001-{gdd['start']}"), date.fromisoformat(f"2001-{gdd['end']}")
    if start > end:
        raise ValueError("Unordered GDD season")
    for value in [gdd["base_c"], policy["hdd_base_c"], policy["cold_threshold_c"], *([] if gdd["cap_c"] is None else [gdd["cap_c"]])]:
        if type(value) not in (int, float) or not math.isfinite(value):
            raise ValueError("Invalid seasonal analytics base/cap")
