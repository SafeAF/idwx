from __future__ import annotations

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from idwx.config import load_config, load_stations
from idwx.io import discover_csvs, ingest_era5_hourly_csv


def main() -> None:
    cfg = load_config("configs/idaho.yml")
    stations = load_stations(cfg.stations_file)
    files = discover_csvs(cfg.data_root)
    if not files:
        raise SystemExit(f"No CSV files found under {cfg.data_root}")

    first = files[0]
    station_id, daily, precip_mode = ingest_era5_hourly_csv(first, config=cfg, stations=stations)

    print(f"csv: {first}")
    print(f"station_id: {station_id}")
    print(f"precip_mode: {precip_mode}")

    if daily.empty:
        print("daily rows: 0")
        return

    dmin = daily["date"].min()
    dmax = daily["date"].max()
    print(f"date_range: {dmin.date()} -> {dmax.date()}")
    print(daily.head(5).to_string(index=False))


if __name__ == "__main__":
    main()
