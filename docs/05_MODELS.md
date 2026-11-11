# Models

## Core requirement

Every model must output:
- a point estimate (a median only when supported by the estimator)
- uncertainty bounds with explicit semantics
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
- separates fitted model behavior from held-out interval construction
- supports finite-sample marginal guarantees under exchangeability assumptions;
  those assumptions have not been established for the weather series

The implemented RF calibration contract is version 2: a chronological
calibration tail with a configurable five-year minimum, a full calendar-year
gap after at least eight fitting years, and a finite-sample residual order
statistic. Unsupported intervals fail without substituting training residuals
or fixed widths. RF's point is its regression estimate; the legacy `p10`/
`p50`/`p90` API names do not certify quantiles. See
[the calibration contract](18_RF_CALIBRATION_CONTRACT.md) for sample rules,
metadata, migration and remaining validation requirements. No forecasts are
published.
