# Checked-in Twin Falls weather database

`idwx.sqlite3` is the compact, application-ready input for the static frost
dashboard. It is intentionally tracked so a checkout and CI deployment can
produce the real-data site without access to the raw archive.

## Scope and provenance

- Station ID: `twin_falls` (the source archive labels this coordinate as the
  Joslin field near Twin Falls)
- Coordinate: `42.4634, -114.4975`
- Product: ERA5 reanalysis supplied through Oikolab
- Source period: 1980-01-01 through 2022-12-31 hourly timestamps
- Source file: `42.4634--114.4975-1980-01-01-2022-12-31-full-params-idaho.csv`
- Source bytes: `80044940`
- Source SHA-256: `1a439c26f6426236e943b22221a1246da8d9b5fc438d145f00b0971536311a8e`
- Database SHA-256: `2b22d1447d3254ea94d01a67d504174af9b32fc525867260b0c39c44c33357ae`

The database has 15,707 daily rows and only the `twin_falls` station. The
first (`1979-12-31`) and last (`2022-12-31`) local dates are partial UTC-offset
boundary days and are retained with `insufficient_coverage`; 15,705 dates have
valid daily minimum/maximum/mean values. Temperatures are Celsius and
precipitation is millimeters.

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
`data/idwx.sqlite3`. A rebuild from the source/hash above is byte-for-byte
deterministic with the committed database.

Inspect the bounded contents with:

```sh
sqlite3 data/idwx.sqlite3 \
  "SELECT station_id, COUNT(*), MIN(date), MAX(date) FROM daily_weather GROUP BY station_id;"
```

Do not add another station implicitly. Adding a station requires an explicit
configuration, storage, publication, and review change.
