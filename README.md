# idwx — Idaho Seasonal Agro-Weather Models (frost, winter, growing season)

`idwx` is a local-first modeling repo for turning long historical station data into:

- **Last spring frost** date predictions (P10/P50/P90)
- **First fall frost** date predictions (P10/P50/P90)
- **Winter severity** forecasts (rank + percentile + components)
- **Growing season** metrics and forecasts (freeze-free window, GDD)

This is not daily weather forecasting.
This is seasonal / event forecasting for agriculture.

## Data location (local, not committed)

Raw CSVs live on this machine:

`~/historical_weather_data-1980-2022/idaho/`

This repo assumes you have ~50 years of daily or hourly station data for ~20 locations.

## Core principles

- **Local-first**: data stays on disk, never committed.
- **Reproducible**: configs define everything.
- **Comparable**: multiple models, same targets, same scoring.
- **Honest**: always output uncertainty intervals and baseline comparisons.

## Repo layout

