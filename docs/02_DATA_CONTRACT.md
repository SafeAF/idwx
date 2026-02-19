# Data contract

This document defines the **input assumptions**, the **canonical schema**, and the **rules** for missingness and units.

## Local raw data root

Raw CSVs live on this machine:

`~/historical_weather_data-1980-2022/idaho/`

This directory is **not committed**.

## Supported input types

The ingest layer must support:

- **Daily CSVs** (already aggregated)
- **Hourly CSVs** (aggregated to daily by this repo)

We do **not** assume consistent headers, units, or timestamp formats across stations.

## Station metadata (required)

Create:

`configs/stations.yml`

Example:

~~~yaml
stations:
  - station_id: twin_falls
    name: Twin Falls
    lat: 42.5629
    lon: -114.4609
    elevation_m: 1134
    source: NOAA
~~~

### Station ID rules

- `station_id` is lowercase snake_case
- stable forever (used in cache paths and model registry)
- do not rename without a migration step

## Canonical daily schema (internal)

The internal daily table for each station must contain these columns:

Required:

- `station_id` (string)
- `date` (YYYY-MM-DD, local date)
- `tmin_c` (float)
- `tmax_c` (float)
- `tmean_c` (float)

Optional:

- `precip_mm` (float)
- `wind_mps` (float)
- `rh_pct` (float)
- `snow_mm` (float)
- `solar_wm2` (float)
- `pressure_hpa` (float)

### Notes

- If `tmean_c` is missing, derive it as `(tmin_c + tmax_c) / 2`.
- If only hourly temps exist, daily values are derived by aggregation.

## Hourly -> daily aggregation rules

If raw input is hourly (or sub-hourly):

- `tmin_c` = minimum temperature of the day
- `tmax_c` = maximum temperature of the day
- `tmean_c` = arithmetic mean of all valid readings that day
- `precip_mm` = sum of precip readings that day (if present)

Coverage requirement:

- to compute Tmin/Tmax/Tmean for a day, require at least **18 hourly readings**
  - configurable in YAML (default 18)
- if coverage is insufficient, the daily row is still present but the affected fields are `null`

## Missingness rules (non-negotiable)

- Never interpolate frost targets.
- Never fabricate missing days.
- Missing daily values are allowed; the model must tolerate them.

Imputation is allowed only for **features**, and must create flags:

- `tmin_imputed`
- `tmax_imputed`
- `precip_imputed`

## Unit normalization

All canonical values must be stored as:

- temperature: Celsius
- precipitation: millimeters
- wind: meters per second
- relative humidity: percent (0–100)
- pressure: hPa

Raw data may be Fahrenheit/inches/mph.
Conversion must happen during ingest.

## Timezone rules

- `date` is the **local calendar date** for the station.
- Config defines the timezone (likely `America/Boise`).
- If raw data timestamps are UTC, convert before grouping into days.

## Output artifacts

After ingest, the repo writes:

- `data_cache/canonical/<station_id>.parquet`
- `data_cache/daily/<station_id>.parquet`

These are local artifacts and must be gitignored.
