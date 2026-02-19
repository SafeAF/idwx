# Models

## Core requirement

Every model must output:
- a point estimate (P50)
- uncertainty bounds (P10/P90 or P20/P80)
- model metadata

## Model interface

Each model implements:

- `fit(X, y, meta, config)`
- `predict(X_future, config) -> dict`
- `save(path)`
- `load(path)`

## Model zoo (v1)

### 0) Climatology baseline (mandatory)
For each station:
- P50 = median historical DOY
- P10/P90 = quantiles

This is the benchmark.

### 1) Trend baseline
Quantile regression:
- `DOY ~ year`

Outputs P10/P50/P90.
Captures climate drift.

### 2) ARIMA / ETS
Time series on DOY per station.
Mostly for comparison.

### 3) Random Forest regression
Feature-driven model.
Wrap with conformal prediction for intervals.

### 4) Gradient boosting (optional)
If you want:
- HistGradientBoostingRegressor
- or LightGBM later

## Uncertainty / intervals

Preferred method:
- conformal prediction wrapper

Why:
- works in small data regimes
- produces calibrated intervals without distribution assumptions
