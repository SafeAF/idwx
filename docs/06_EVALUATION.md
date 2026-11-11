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
  threshold_0p0/
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

## Implemented research reporting

Reports are separated by threshold: `threshold_0p0` for 0 C,
`threshold_m2p0` for -2 C, or `threshold_none` for no threshold. Different
threshold runs do not overwrite each other. Re-running the same target/model/
threshold replaces that report. `plots/` is reserved; plots are not generated.

`yearly_errors.csv` now contains a row for every configured test year at each
station in the input dataset, including skipped folds. `status` identifies:

- `evaluated`: finite target and finite, ordered prediction bounds.
- `missing_test_year`: no dataset row exists for the test year.
- `missing_actual`: the target is missing or nonfinite.
- `insufficient_training`: too few earlier years with finite targets.
- `model_unavailable`: fitting or prediction raised a `ValueError`; `reason`
  retains its explanation, including model-specific sample requirements.
- `invalid_prediction`: nonfinite or crossed prediction bounds. These are
  excluded rather than repaired silently.

Malformed prediction schemas fail the evaluation. Duplicate station-years,
missing keys, noninteger years, reversed year bounds and nonpositive training
minimums are rejected. Training uses only finite targets from years strictly
before the test year. Fold rows include that input training span, valid training
year count, and the model's calibration row count when provided. For RF this
span includes both its internal fitting and calibration partitions.

Summary and per-station tables contain `n_folds`, `n_evaluated`, `n_skipped`,
MAE, RMSE, signed bias, coverage and mean interval width. Only `evaluated`
folds enter those metrics; coverage is not a percentage of all configured
years. Read the counts and skipped rows alongside scores. With no evaluable
folds, status is `insufficient_evaluation`, metrics are blank in CSV and null
in CLI JSON. Empty input tables with the required columns also produce
well-formed empty reports through the Python evaluation API.

The CLI always evaluates climatology, even for `--models rf` or when it is
listed last. Baseline comparison joins successful folds by station/year and
verifies matching actual targets. The yearly report includes
`baseline_pred_p50`; `n_paired`, `paired_model_mae`, `baseline_mae`, and
`delta_mae` are calculated on exactly the common successful folds.
`delta_mae = paired_model_mae - baseline_mae`; the overall `mae` can use a
larger set and must not be substituted in that subtraction. Baseline report
files are written when climatology is explicitly requested; its paired scores
are included for each requested model regardless.

All modeling output remains `unvalidated_research`. The default legacy path
still has seasonal feature leakage and full-history WSI limitations. The new
opt-in `--protocol frost_jan01_v1` excludes those predictors, uses only training
labels through test year minus two, and adds explicit issue/cutoff/calendar
columns and a protocol report subdirectory. It shares its input rules with
training and prediction; see [the protocol](17_PRESEASON_RESEARCH_PROTOCOL.md).
Calibration and real-data validation remain outstanding for every mode.
