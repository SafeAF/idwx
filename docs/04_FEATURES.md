# Feature engineering

## Philosophy

We are predicting seasonal event dates.
This is a low-sample regime (~50 years per station).
Feature sets must be simple, stable, and hard to overfit.

## Feature sets

All features are per (station, year).

### A) Prior-season features

For predicting first fall frost in year Y:
- summer Tmin distribution (Jun-Aug):
  - mean, std, min, quantiles
- fall transition (Aug-Sep):
  - slope of Tmin vs date
  - rolling 7-day Tmin minima quantiles
- early cooling signals:
  - first day Tmin <= 10°C
  - count of days Tmin <= 12°C in Sep

For predicting last spring frost in year Y:
- winter coldness (Dec-Feb)
- early spring warmup (Mar-Apr):
  - slope of Tmin
  - count of nights <= 0°C in April

### B) Lag features

- last year’s frost DOY (lag1)
- last year’s WSI (lag1)
- rolling 3y mean frost DOY
- rolling 5y mean frost DOY

### C) Trend features

- year index (captures warming trend)
- optional: station elevation (if pooled model)

## Pooled vs per-station

We support two training modes:

1. Per-station model:
   - trained only on that station’s years
2. Pooled model:
   - trained on all stations
   - includes station_id as categorical and elevation as numeric

Pooled models often perform better due to sample size.
