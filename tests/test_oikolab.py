import json
from datetime import datetime, timezone

import pandas as pd

from idwx.oikolab import fetch_era5_hourly_csv


def test_fetch_accepts_current_epoch_seconds_and_writes_ingestible_csv(monkeypatch, tmp_path):
    timestamps = pd.date_range("2026-01-01T00:00:00Z", periods=3, freq="h")
    table = {
        "index": [int(value.timestamp()) for value in timestamps],
        "columns": [
            "coordinates (lat,lon)", "model (name)", "model elevation (surface)",
            "utc_offset (hrs)", "temperature (degC)",
            "total_precipitation (mm of water equivalent)",
        ],
        "data": [
            [[42.4634, -114.4975], "era5", 1318.0, -7, temperature, precipitation]
            for temperature, precipitation in ((-2.0, 0.0), (-1.0, 0.1), (0.0, 0.0))
        ],
    }
    monkeypatch.setattr("idwx.oikolab._request_json", lambda *args, **kwargs: {
        "data": json.dumps(table),
    })
    destination = tmp_path / "hourly.csv"
    report = fetch_era5_hourly_csv(
        "secret", destination, latitude=42.4634, longitude=-114.4975,
        start=datetime(2026, 1, 1, tzinfo=timezone.utc),
        end=datetime(2026, 1, 1, 2, tzinfo=timezone.utc),
    )
    assert report["rows"] == 3
    written = pd.read_csv(destination)
    assert written["timestamp"].tolist() == [
        "2026-01-01T00:00:00Z", "2026-01-01T01:00:00Z", "2026-01-01T02:00:00Z",
    ]
    assert written["temperature (degC)"].tolist() == [-2.0, -1.0, 0.0]
