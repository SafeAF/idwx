# RF held-out calibration contract, version 2

This corrects the RF research interval construction. It does not establish
calibrated coverage for weather data or permit forecast publication. The
static exporter remains historical-only.

## Chronological partitions

RF retains only finite targets, rejects ambiguous station/year keys and sorts
by calendar year. Let N be the number of distinct years with a valid target.
The calibration tail contains the latest
`max(min_calibration_years, ceil(N * calibration_fraction))` years, capped at
available years so insufficient history can be reported explicitly. Every
station in a selected calendar year belongs to the same partition.

With default configuration:

```yaml
models:
  conformal:
    alpha: 0.2
    calibration_fraction: 0.2
    min_calibration_years: 5
```

The first calibration year is C. Fitting uses only years **at or before C-2**.
Year C-1 is excluded from both fitting and calibration. This full calendar-year
gap respects the opt-in preseason protocol's label availability boundary for
its first calibration prediction. The same conservative split applies to
legacy RF research, although it does not fix legacy feature leakage.
If the intervening year is already absent, no extra available year is removed.

At least eight distinct fitting years and the configured minimum calibration
years are required. With consecutive annual records and defaults, the first
possible split has 14 years: eight fitting, one excluded and five calibration.
The five-year floor is an explicit research policy, not a sufficient sample
size for publication or a claim of stable calibration.

Fit feature medians and the forest once on the fitting partition only. Use
those fixed medians to transform calibration rows; all-null fitting features
retain the existing zero fallback. The forest is **not refitted** on calibration
or excluded-year targets. Neither partition may update preprocessing. For
`frost_jan01_v1`, each row's predictors already obey its own issue cutoff.

## Residual threshold

For n held-out residuals `r_i = abs(actual_i - fitted_prediction_i)`, compute

```text
k = ceil((n + 1) * (1 - alpha))
qhat = kth smallest residual, using a one-based rank
interval = [forest_point - qhat, forest_point + qhat]
```

Use the actual order statistic without interpolation. The split-conformal
finite-sample correction is described by
[Angelopoulos and Bates](https://arxiv.org/html/2107.07511v6#S1.SS1).
Its standard marginal coverage result relies on exchangeability; that is not
established for this historical weather series or pooled station residuals.
Chronological splitting and a gap are availability safeguards, not a proof of
coverage under serial dependence or climate change.

For example, alpha 0.2 and n=5 selects the largest residual (rank 5), while n=3
requires unavailable rank 4. When k exceeds n, the application withholds a
finite interval and raises `InsufficientCalibrationError`. It does not cap the
rank at n, change alpha, increase the calibration partition silently, or
serialize an infinite bound. To research smaller alpha, explicitly reserve a
sufficient calibration sample and rerun evaluation.

Empty, mismatched, multidimensional or nonfinite residual inputs fail.
Residual overflow fails too. A legitimate all-zero held-out residual sample
can produce zero width; that is an observed calculation, not a fallback.

There is no training-residual fallback and no arbitrary fixed-width fallback.
A model without compatible, finite, nonnegative held-out `qhat` cannot predict
or save. Starting a refit disables the previous fit; a failed refit cannot
leave old predictions or preprocessing usable.

## Output semantics and identity

RF's point is the forest's regression estimate, not a fitted conditional
median. Internal `p10`, `p50`, `p90` field names remain compatibility aliases
for lower bound, point and upper bound. Symmetric absolute-residual bounds do
not establish equal tail probabilities or conditional P10/P90 quantiles.
Protocol CLI predictions already use `lower`, `point`, `upper` and remain
`unvalidated_model_interval`.

Model metadata now records calibration version/method, interval semantics,
alpha and nominal coverage, requested partition settings, fitting/calibration
date ranges, row and distinct-year counts, excluded rows, residual rank and
`qhat`. `predict` JSON includes RF `interval_metadata` alongside the explicitly
unvalidated result.

Registry format remains version 2. The additional `rf_calibration_version: 2`
field is checked before an RF pickle is loaded. Earlier RF artifacts without
the marker are incompatible and must be retrained; no older-run fallback is
used. Direct `RFConformalModel.load` likewise rejects old/unfitted calibration
state. Climatology and trend require no RF calibration marker, although any
configuration change is still subject to the registry's conservative config
hash checks. The calibration correction itself did not change predictor
availability. The later date-policy follow-up requires protocol input version
2 and date policy version 1; rebuild/retrain earlier protocol artifacts.

## Evaluation diagnostics

Yearly reports include actual fitting and calibration spans, distinct-year
counts, calibration row count, excluded row count, nominal coverage, selected
rank and interval semantics. These are distinct from the outer backtest's
`training_start_year`, `training_end_year` and `n_train_years`, which describe
all eligible input rows before RF's internal partitioning.

Unsupported finite ranks or insufficient calibration years are explicit
`insufficient_calibration` folds. Too few fitting years remain
`model_unavailable`, with an explanation. Available partition diagnostics are
retained even on failed folds. No unsuccessful fold enters accuracy/coverage
metrics or paired baseline comparisons.

Summary and per-station reports add nominal coverage, observed minus nominal
coverage, lower miss rate and upper miss rate. Nominal coverage is populated
only when all evaluated folds report the same nominal level; it is blank/null
for baselines without such metadata or empty evaluations. Coverage and miss
rates are calculated on evaluated folds only, with inclusive interval bounds.
The two tail miss rates are diagnostics; no alpha/2 tail guarantee is asserted.
No confidence interval for these rates or validated promotion rule is added.

## Verification and remaining work

Thirty-eight new synthetic tests cover rank correction and unsupported ranks,
invalid inputs/configuration, residual overflow, zero residuals, chronological
whole-year splitting, pooled station partitions, natural calendar gaps,
fitting-only preprocessing, calibration perturbations, single forest fitting,
failed calibration/refits, rejection of fallback widths, metadata and report
diagnostics, serialization, and legacy RF rejection.

Full verification after the date-policy follow-up: **169 Python tests passed**, one optional live Nginx test
skipped; five frontend tests passed. Targeted Ruff and `git diff --check`
passed. These are implementation regressions, not predictive performance or
coverage measurements. The RF protocol round trip now supplies enough history
for the larger held-out partition and gap.

Real source/vintage audits, sufficient independent validation data, temporal
calibration assessment, station/threshold/window performance, paired baseline
comparisons under the implemented date policy, predetermined promotion
criteria, browser verification and production TLS remain outstanding. Do not
publish model predictions based solely on passing these tests.
