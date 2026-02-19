# Targets

This document defines the targets (y-values) built from daily weather data.

All targets are computed per:

- `station_id`
- `season_year` (integer)

## Frost thresholds

Frost is defined using **daily Tmin**.

Thresholds are configurable (common values):

- light frost: Tmin <= 0.0°C
- killing frost: Tmin <= -2.0°C

The system must support multiple thresholds simultaneously.

## Target: last spring frost

Definition:

For season year `Y`, last spring frost is:

- the last date in the window where `tmin_c <= threshold_c`

Default search window:

- Jan 1 -> Jun 30

If no frost occurs in the window:

- target is missing (`null` / `NaN`)
- do not fabricate a date

## Target: first fall frost

Definition:

For season year `Y`, first fall frost is:

- the first date in the window where `tmin_c <= threshold_c`

Default search window:

- Aug 1 -> Dec 31

If no frost occurs in the window:

- target is missing (`null` / `NaN`)

## Derived target: freeze-free season length

If both frost dates exist:

- `freeze_free_days = (first_fall_frost - last_spring_frost).days`

If either frost date is missing:

- derived value is missing

## Winter season definition

Winter severity is computed over:

- Dec of `Y-1`
- Jan of `Y`
- Feb of `Y`

This makes the winter belong to the season year `Y`.

## Winter Severity Index (WSI)

WSI is a scalar designed to be:

- interpretable
- rankable
- stable across stations

Compute raw components:

- `hdd` (heating degree days, base configurable)
- `tmin_extreme` (minimum daily Tmin in winter)
- `cold_snap_days` (count of days with Tmin <= cold threshold)
- `cold_snap_max_run` (max consecutive run length under cold threshold)
- `winter_precip_mm` (sum, if precip exists)

Normalize components per station across years (z-score or robust scaling).

Composite:

- weighted sum of normalized components

Store both:

- raw components
- normalized components
- composite WSI

## Growing season quality

At minimum compute:

- `gdd_total` (growing degree days, base configurable)
- `heat_days_32c` (count of days with Tmax >= 32°C)
- `heat_days_35c` (count of days with Tmax >= 35°C)

If precip exists, compute a simple moisture stress proxy:

- `moisture_proxy = precip_sum - k * temp_sum`

Where:

- `k` is a configurable scalar

## Target output tables

Targets are stored as a per-year table:

- one row per (station_id, season_year, threshold_c, target_name)

Example columns:

- `station_id`
- `season_year`
- `threshold_c` (nullable for non-frost targets)
- `target_name`
- `value_date` (nullable)
- `value_doy` (nullable)
- `value_float` (nullable)
- `quality_flags`
