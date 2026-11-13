"""Small, credential-safe Oikolab client used by the SQLite refresh command."""
from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

API_ROOT = "https://api.oikolab.com"
ERA5_MODEL = "era5"
PARAMETERS = ("temperature", "total_precipitation")
USER_AGENT = "idwx-weather-import/1.0"


def _request_json(api_key: str, path: str, query: list[tuple[str, str]] | None = None,
                  *, timeout: int = 180) -> dict:
    if not api_key.strip():
        raise ValueError("Oikolab API key is empty")
    url = f"{API_ROOT}/{path.lstrip('/')}"
    if query:
        url = f"{url}?{urllib.parse.urlencode(query)}"
    request = urllib.request.Request(
        url,
        headers={"api-key": api_key, "Accept": "application/json", "User-Agent": USER_AGENT},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = json.load(response)
    except urllib.error.HTTPError as exc:
        # Do not include the response body: upstream errors can reflect request
        # details and credentials must never be copied into logs.
        raise ValueError(f"Oikolab request failed with HTTP {exc.code} ({exc.reason})") from exc
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        raise ValueError(f"Oikolab request failed: {type(exc).__name__}") from exc
    if not isinstance(payload, dict):
        raise TypeError("Oikolab returned an unexpected response type")
    return payload


def era5_available_end(api_key: str) -> datetime:
    """Return the inclusive end of the currently published ERA5 dataset."""
    datasets = _request_json(api_key, "datasets", timeout=30)
    era5 = datasets.get(ERA5_MODEL)
    if not isinstance(era5, dict) or not isinstance(era5.get("end"), str):
        raise TypeError("Oikolab dataset metadata did not contain an ERA5 end timestamp")
    try:
        parsed = datetime.strptime(era5["end"], "%Y-%m-%d %H UTC").replace(tzinfo=timezone.utc)
    except ValueError as exc:
        raise ValueError("Oikolab returned an invalid ERA5 end timestamp") from exc
    return parsed


def fetch_era5_hourly_csv(api_key: str, destination: Path, *, latitude: float,
                          longitude: float, start: datetime, end: datetime) -> dict:
    """Fetch and validate Twin Falls hourly ERA5 temperature/precipitation."""
    if start.tzinfo is None or end.tzinfo is None:
        raise ValueError("Oikolab request bounds must be timezone-aware")
    start = start.astimezone(timezone.utc)
    end = end.astimezone(timezone.utc)
    if start > end:
        raise ValueError("Oikolab request start must not be after its end")

    query = [("param", parameter) for parameter in PARAMETERS]
    query.extend((
        ("start", start.strftime("%Y-%m-%dT%H:%M:%SZ")),
        ("end", end.strftime("%Y-%m-%dT%H:%M:%SZ")),
        ("lat", f"{latitude:.6f}"),
        ("lon", f"{longitude:.6f}"),
        ("model", ERA5_MODEL),
    ))
    response = _request_json(api_key, "weather", query)
    encoded = response.get("data")
    try:
        weather = json.loads(encoded) if isinstance(encoded, str) else encoded
    except json.JSONDecodeError as exc:
        raise ValueError("Oikolab returned invalid encoded weather data") from exc
    if not isinstance(weather, dict):
        raise TypeError("Oikolab weather response did not contain a data table")

    index = weather.get("index")
    columns = weather.get("columns")
    rows = weather.get("data")
    if not isinstance(index, list) or not isinstance(columns, list) or not isinstance(rows, list):
        raise TypeError("Oikolab weather table is missing index, columns, or data")
    if len(index) != len(rows) or not rows:
        raise ValueError("Oikolab weather table has no rows or inconsistent row counts")
    required = {
        "coordinates (lat,lon)",
        "model elevation (surface)",
        "utc_offset (hrs)",
        "temperature (degC)",
        "total_precipitation (mm of water equivalent)",
    }
    missing = required.difference(columns)
    if missing:
        raise ValueError(f"Oikolab response is missing required columns: {sorted(missing)}")

    if all(isinstance(value, (int, float)) for value in index):
        # Oikolab examples historically used epoch milliseconds, while the
        # current endpoint can return epoch seconds. Detect the scale without
        # relying on a stale transport convention.
        unit = "ms" if max(abs(float(value)) for value in index) >= 100_000_000_000 else "s"
        timestamps = pd.to_datetime(index, unit=unit, utc=True, errors="coerce")
    else:
        timestamps = pd.to_datetime(index, utc=True, errors="coerce")
    if timestamps.isna().any() or not timestamps.is_monotonic_increasing or timestamps.has_duplicates:
        raise ValueError("Oikolab returned invalid, unordered, or duplicate timestamps")
    expected = pd.date_range(start=start, end=end, freq="h")
    if not timestamps.equals(expected):
        missing_hours = expected.difference(timestamps)
        extra_hours = timestamps.difference(expected)
        raise ValueError(
            "Oikolab response does not provide continuous hourly coverage for the request "
            f"(expected={len(expected)}, received={len(timestamps)}, "
            f"first={timestamps[0].isoformat()}, last={timestamps[-1].isoformat()}, "
            f"missing={len(missing_hours)}, extra={len(extra_hours)})"
        )

    frame = pd.DataFrame(rows, columns=columns, index=timestamps)
    for column in ("temperature (degC)", "total_precipitation (mm of water equivalent)"):
        numeric = pd.to_numeric(frame[column], errors="coerce")
        if numeric.isna().any():
            raise ValueError(f"Oikolab returned missing or nonnumeric values in {column}")
        frame[column] = numeric

    destination = destination.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    frame.index.name = "timestamp"
    frame.to_csv(destination, date_format="%Y-%m-%dT%H:%M:%SZ")
    return {
        "rows": len(frame),
        "first_timestamp_utc": timestamps[0].isoformat(),
        "last_timestamp_utc": timestamps[-1].isoformat(),
        "parameters": list(PARAMETERS),
        "model": ERA5_MODEL,
    }
