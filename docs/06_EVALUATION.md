# Evaluation

## Non-negotiable rule

No random train/test splits.
This is time-series data.
Use walk-forward backtesting.

## Walk-forward scheme

For each station:

For year t in [start+N ... end]:
- train on years <= t-1
- test on year t

Aggregate errors across all folds.

## Metrics

### Frost date targets
- MAE in days
- RMSE in days
- bias (mean error)
- interval coverage:
  - fraction of years where truth is inside [P10, P90]
- pinball loss (if quantile models)

### Winter severity
- MAE
- Spearman rank correlation
- accuracy if bucketed into quintiles

## Reporting outputs

Write:

reports/<target>/<model_name>/
  summary.csv
  per_station.csv
  yearly_errors.csv
  plots/

Plots:
- predicted vs actual DOY scatter
- error over time
- coverage calibration plot

## Baseline comparisons

Every report must include:
- baseline MAE
- model MAE
- delta (model - baseline)

If the model does not beat baseline, it is not promoted.
