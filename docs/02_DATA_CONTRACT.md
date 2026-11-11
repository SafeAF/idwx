# Data contract

This document defines the **input assumptions**, the **canonical schema**, and the **rules** for missingness and units.

## Local raw data root

The configured raw CSV location is:

`~/historical_weather_data-1980-2022/idaho/`

This directory is **not committed** and is currently absent in the implementation
workspace. Real station availability and source conventions remain unverified.

## Supported input types

The original ingestion scope includes:

- **Daily CSVs** (already aggregated)
- **Hourly CSVs** (aggregated to daily by this repo)

The implemented adapter currently supports ERA5-style hourly CSVs with an exact
Celsius header and source times interpreted as UTC. General daily CSV parsing,
alternate headers and Fahrenheit/inches/mph conversion are not implemented.
Do not assume arbitrary source formats are normalized by the existing code.
Use `idwx data audit` and a reviewed adapter/normalization step for new sources;
see `20_SOURCE_DATA_AUDIT.md`.

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
- `precip_mm` currently uses a cumulative/incremental heuristic when present;
  accumulation/reset conventions require source review before using it

Coverage requirement:

- to compute Tmin/Tmax/Tmean for a day, require at least **18 distinct UTC hours with finite temperatures**
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

Raw data may be Fahrenheit/inches/mph. Such inputs require an explicit reviewed
conversion adapter; the current Celsius-only adapter does not perform this
general normalization.

## Timezone rules

- `date` is the **local calendar date** for the station.
- Config defines the timezone (likely `America/Boise`).
- If raw data timestamps are UTC, convert before grouping into days.

## Output artifacts

The implemented hourly adapter writes:

- `data_cache/daily/<station_id>.parquet`

The separately proposed `canonical/` raw-observation cache is not currently
written. Daily caches are local artifacts and must be gitignored. Source
collisions fail before writing; parsing/serialization is staged before replacing
old caches, with atomic replacement per station. See the source audit document
for the remaining multi-file commit/consolidation limitations.
