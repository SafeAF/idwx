# Checked-in Twin Falls weather database

`idwx.sqlite3` is the compact, application-ready input for the static frost
dashboard. It is intentionally tracked so a checkout and CI deployment can
produce the real-data site without access to the raw archive.

## Scope and provenance

- Station ID: `twin_falls` (the source archive labels this coordinate as the
  Joslin field near Twin Falls)
- Coordinate: `42.4634, -114.4975`
- Product: ERA5 reanalysis supplied through Oikolab
- Base source period: 1980-01-01 through 2022-12-31 hourly timestamps
- Source file: `42.4634--114.4975-1980-01-01-2022-12-31-full-params-idaho.csv`
- Source bytes: `80044940`
- Source SHA-256: `1a439c26f6426236e943b22221a1246da8d9b5fc438d145f00b0971536311a8e`
- Oikolab refresh request: 2022-12-31 00:00 UTC through 2026-11-08 15:00 UTC
- Refresh rows/bytes: `33808` / `2425823`
- Refresh SHA-256: `7ed584a770383a794b9971032e76bade1f78da5da2631bfbe0a4b76ea8ccad01`
- Database SHA-256: `cb7f325643d26648aac229ff6d01077c3eee7246675b34a2b328df135a9e40ce`

The database has 17,114 daily rows and only the `twin_falls` station. It covers
local dates `1979-12-31` through `2026-11-07`; 17,113 dates have valid daily
minimum/maximum/mean values. The initial `1979-12-31` boundary date remains
partial. The refresh replaced the formerly partial `2022-12-31` row with a
complete local day. Temperatures are Celsius and precipitation is millimeters.

This is modeled historical reanalysis, not a physical weather-station
observation record and not a current forecast. The public exporter labels it
`historical_reanalysis`.

## Rebuild

Keep the raw 80 MB hourly CSV outside this repository. With the data checkout
next to this repository, rebuild from the repository root with:

```sh
uv run idwx data build-sqlite \
  ../idaho-weather-data/archive/42.4634--114.4975-1980-01-01-2022-12-31-full-params-idaho.csv
```

The builder reads only that explicitly named source, verifies its coordinates
against `configs/stations.yml`, fingerprints it, writes a new schema-versioned
SQLite database, runs `PRAGMA integrity_check`, and atomically replaces
`data/idwx.sqlite3`. That rebuild recreates the base archive only. Apply the
provider refresh afterward with a hidden API-key prompt:

```sh
uv run idwx data refresh-oikolab
```

The refresh requests only the missing Twin Falls ERA5 hours, validates
continuous timestamps and required columns, imports complete local days, and
stores fragment provenance in `source_refreshes`. Oikolab can revise reanalysis
values, so a future provider response is not claimed to reproduce the historical
refresh hash byte-for-byte.

Inspect the bounded contents with:

```sh
sqlite3 data/idwx.sqlite3 \
  "SELECT station_id, COUNT(*), MIN(date), MAX(date) FROM daily_weather GROUP BY station_id;"
```

Do not add another station implicitly. Adding a station requires an explicit
configuration, storage, publication, and review change.
