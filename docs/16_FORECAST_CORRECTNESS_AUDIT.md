# Forecast correctness audit

Audit date: November 11, 2026. This is a source-code audit with synthetic
regression tests, not a validation of station forecasts. The website continues
to export historical observations and historical distributions only. Its
synthetic mode remains explicitly labeled. No model predictions are exported.

## Input corrections implemented

- `features.py` and `datasets.py` retain missing feature values instead of
  filling them with medians calculated across the full station history.
  Otherwise future backtest years could influence earlier training inputs.
  Existing `*_imputed` flags are retained for compatibility: 1 identifies an
  unavailable value requiring preprocessing; it does not mean a value was
  filled in the cache.
- Target lags, rolling target means, weather lags, and WSI lags use calendar
  years. An absent year cannot silently become the previous available year.
  Rolling means use the previous three/five calendar years, with at least
  two/three nonmissing targets respectively. Future rows use the same rule.
- `build_future_row` initializes unknown features as missing. It carries only
  station ID/elevation, the next year index, explicit one-year lags from their
  base columns, and recomputed target history. It does not copy last year's
  seasonal weather into the upcoming year. Multiple stations or duplicate
  station-years fail instead of selecting an arbitrary latest record.
- RF medians and feature columns are fitted explicitly on its training
  partition. Calibration/prediction cannot learn medians even when every
  training feature is null. Refitting replaces the previous preprocessing.
  The existing fixed zero fallback for an entirely missing training feature
  remains; its suitability has not been validated.

These changes do **not** make the remaining forecast pipeline safe to publish.
In particular, carrying a WSI lag still carries the normalization problem below.

## Evaluation and artifact corrections implemented

The follow-up implementation separates model artifacts and evaluation reports
by exact threshold. Version 2 artifacts record threshold and configuration/
dataset/schema identity; prediction checks both the requested model and
climatology before loading. Configuration hashing now includes station windows
and station metadata bytes. Unique automatic run IDs, no-overwrite explicit
IDs, and an atomically installed completion marker prevent run collisions and
selection of interrupted saves. Legacy models require retraining.

Backtests retain skipped and invalid folds with reasons instead of crashing on
empty results or silently losing missing years. Reports include input training
span, sample counts, available calibration counts and mean interval width.
The CLI calculates climatology regardless of requested model order and compares
MAE on identical successful station-years. Missing scores serialize as JSON
null. See `06_EVALUATION.md` for statuses/denominators and `07_CLI_SPEC.md` for
artifact layout and compatibility checks.

These are research safeguards. No issue-date protocol, validated preprocessing
version, forecast promotion gate or forecast export has been added.

## Remaining blockers found in source

| Area | Evidence and consequence | Required follow-up |
| --- | --- | --- |
| Issue-date feature availability | `build_yearly_features` puts summer, August/September, September counts, March/April, and winter observations in every year's row regardless of target or issue date. `train` and `walk_forward_backtest` pass these columns to RF. Spring targets can see later summer weather; fall targets can see observations after the first fall event. | Define a forecast protocol for each target, including local issue date, observation delivery lag, allowed features and availability dates. Rebuild each fold from only information actually available then. Account for events that already occurred before issue. |
| Full-history WSI normalization | `build_targets_for_station` calculates component means and standard deviations over all years. Both current WSI and `wsi_lag1` can therefore depend on later years. | Remove WSI from frost predictors until normalization is fitted inside training folds, or use raw prior-year components with training-only transformations. Audit winter target coverage separately. |
| Forecast context | `build_future_row` still selects latest dataset year + 1. `predict` has no explicit forecast-year or issue-date arguments and does not report the last observation date or training period. Missing future weather is now honest, but backtests still use observed same-year weather. | Require explicit season year/issue date and compatible model artifacts. Apply identical availability rules to backtesting and prediction. A dataset ending in 2022 does not support an automatic current-year forecast. |
| Calibration and labeling | RF holds out the last 20% of training rows, but `residual_quantile` uses an interpolated residual quantile without a finite-sample rank correction. RF includes a training-residual fallback and a fixed-width prediction fallback. Its symmetric bounds are returned as `p10`/`p90`. | Define a chronological calibration protocol and minimum calibration sample. Reject unsupported intervals instead of substituting training residuals or arbitrary widths. Use interval-bound labels unless tail quantiles are established. Evaluate empirical coverage under temporal change; chronological splitting alone establishes no calibration guarantee. |
| Date representation | Forecast datasets use actual-year `value_doy`; website history uses the common leap-year calendar. Trend quantiles can cross, and predictions are not constrained to valid target-window dates. | Choose and version a forecast calendar contract. Test leap dates, conversion, ordering and window bounds; evaluate any correction or clipping rather than silently applying it. |
| Model identity | Threshold/configuration/dataset/schema checks, station metadata hashing and collision prevention are implemented. Protocol and feature/preprocessing provenance are still unversioned; a newly saved artifact can still originate from stale caches. | Version issue-date protocols and preprocessing, reject stale derived inputs, and verify that metadata records the effective forecast context. The registry format version is not a validation status. |
| Evaluation/reporting | Empty/skipped folds, counts, training spans, calibration counts, width and paired climatology comparisons are implemented. Upstream feature leakage, real-data results and calibration validation remain unresolved. | Re-run both frost targets with clean fold inputs and sufficient training/calibration history. Review paired error, bias, interval coverage/width, sample counts and uncertainty against predetermined promotion criteria. |

## Forecast publication requirements

Keep forecasts out of the web contract until all of the following are met:

1. Audit real source provenance, units, reporting days, station windows,
   coverage and hand-checked event dates. Define the delivery cadence.
2. Version explicit target/year/issue-date protocols and train/calibration/test
   availability rules. Tests must show that changing observations after an
   issue date cannot affect that issue's features or predictions, and changing
   future test years cannot affect earlier folds or preprocessing.
3. Rebuild targets, features, datasets and models after the corrections.
   Reproduce chronological backtests for both frost targets and each threshold
   with paired climatology comparisons. Report MAE, RMSE, signed bias, interval
   coverage and width, valid-year counts, training/calibration periods, and
   per-station results. Assess quantile losses only for outputs justified as
   quantiles.
4. Document and agree accuracy, baseline improvement, calibration and sample
   size acceptance criteria before reviewing final holdout results. No numeric
   promotion thresholds have been chosen without data and intended-use review.
   The existing evaluation specification requires a model to beat climatology
   before promotion; that alone is insufficient for forecast publication.
5. Attach validation status, forecast year, issue date, last available
   observation, training period, baseline and validated interval semantics to
   every public prediction. Add contract and UI tests for these fields before
   permitting forecast export. Failed criteria retain the historical dashboard.

The existing local `train`, `eval` and `predict` commands remain research tools;
their output must not be treated as validated forecasts or copied into the
static publication directory. This audit does not add a forecast schema.

## Cache migration and verification

Old feature/dataset caches can retain the earlier whole-history imputation.
There is no automatic migration or version rejection for those caches.
In a separate cache directory, rebuild data/targets under current coverage
rules, then run `feature build --rebuild` and `dataset build` for each target
and threshold. Retrain models in a separate artifact directory after the
remaining blockers are fixed. Do not reuse existing serialized models as
evidence of corrected preprocessing or readiness.

```powershell
.venv/Scripts/python.exe -m pytest -q
.venv/Scripts/python.exe -m ruff check idwx/features.py idwx/datasets.py idwx/models/rf.py tests/test_forecast_inputs.py
.venv/Scripts/python.exe -m ruff check idwx/eval.py idwx/registry.py idwx/cli.py tests/test_evaluation.py tests/test_registry.py
node --test tests/web.test.mjs
```

Verification after the evaluation/registry follow-up: 73 Python tests passed,
one optional live Nginx test skipped; five frontend tests passed. Targeted Ruff
and `git diff --check` passed. Nine input regression cases cover missing calendar years,
rolling counts, ambiguous histories, cache missingness, future-row construction,
and training-only RF preprocessing/refitting. Another 26 cases cover empty and
skipped evaluations, invalid intervals, paired baseline scoring, CLI ordering,
strict JSON, threshold isolation, artifact compatibility and run completion/
collision behavior, including a synthetic CLI train/predict round trip.
These are regression checks, not accuracy or calibration measurements.

## Infrastructure findings

The configured `C:/Users/kerrc/historical_weather_data-1980-2022/idaho`
path was checked directly and does not exist. No source CSVs or daily Parquet
caches were found in the tracked/nonignored workspace. The user's home
directory could not be enumerated because access was denied, so no claim is
made that data is absent everywhere on the machine.

Browser inventory reported `The browser access host is unavailable`. The native
Firefox fallback was stopped by the computer-use tool because browser URL
policy enforcement is unsupported for this Windows browser. The companion
page supplied no document session. Rendered desktop/mobile behavior remains
unverified. Production hostname, certificates and server access have not been
provided; production deployment/TLS verification remains outstanding.
