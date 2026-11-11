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

These are research safeguards. The opt-in `frost_jan01_v1` protocol is now
implemented; see `17_PRESEASON_RESEARCH_PROTOCOL.md`. It uses a fixed local
January 1 issue, a full intervening-year holdback, confirmed frost history only,
and a common leap-year calendar. Training, evaluation and prediction share the
same input allowlist and information boundary. It bypasses legacy derived
caches and versions input provenance and artifact context. This is a research
protocol; no forecast promotion gate or forecast export has been added.

RF calibration version 2 is also implemented; see
`18_RF_CALIBRATION_CONTRACT.md`. It reserves whole calendar years, leaves a
full year between fitting and calibration, applies the finite-sample residual
rank, and withholds unsupported intervals. Training-residual and arbitrary
width fallbacks are removed. Actual partitions, nominal coverage and observed
coverage diagnostics are reported. Earlier RF artifacts require retraining.
This does not establish empirical coverage or weather-series exchangeability.

## Remaining blockers found in source

| Area | Evidence and consequence | Required follow-up |
| --- | --- | --- |
| Issue-date feature availability | The new opt-in protocol excludes weather features and uses only confirmed frost targets through Y-2, with consistent per-row and outer-fold cutoffs. The legacy `build_yearly_features` path still includes later seasonal observations without an issue cutoff. | Use the protocol for new frost research. Audit source delivery and historical revisions before interpreting it as an as-issued forecast. Any protocol admitting more recent weather needs separate availability rules and validation. |
| Full-history WSI normalization | Legacy WSI and its lags still use full-history normalization. The new protocol excludes all WSI inputs. | Keep WSI outside frost predictors until normalization is fitted within training folds. Audit winter target coverage separately. |
| Forecast context | Protocol training/prediction require explicit year and matching local January 1 issue, and report cutoff, last observation and training period. Legacy prediction still chooses latest dataset year + 1. | Keep legacy output out of publication. Evaluate source freshness and the intentionally conservative holdback; an explicit research year and a reported old source date do not establish current forecast validity. |
| Calibration and labeling | RF now has a versioned whole-year chronological split, explicit sample floors, one-year gap, finite-sample rank and no fallback intervals. Metadata and protocol output label symmetric bounds explicitly; legacy internal field names remain aliases. Empirical temporal coverage is unmeasured. | Validate coverage/width and tail miss rates using sufficient historical folds and a final holdout. Review dependence, drift and pooled-station effects. Choose publication sample/calibration criteria before final holdout review; chronological splitting alone establishes no calibration guarantee. |
| Date representation | The opt-in protocol normalizes onto year 2000, records the reference, and rejects crossed/nonfinite/out-of-calendar bounds. Legacy datasets still use actual-year DOY. No forecast-year date conversion or target-window support policy is validated. | Test and validate conversion and window bounds without silent clipping. Treat protocol output as numeric calendar coordinates until that policy is defined. |
| Model identity | Protocol artifacts include issue context and input provenance, are separated by protocol/year and bypass stale derived caches. Legacy inputs remain unversioned; daily cache ingestion/vintage is not independently verified. | Add daily input provenance and source audit evidence before promotion. The registry or input version alone is not a validation status. |
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
There is no automatic migration or version rejection for legacy caches. The
opt-in protocol bypasses them and rejects missing/incompatible protocol input
provenance; daily caches still require rebuilding/auditing.
In a separate cache directory, rebuild data/targets under current coverage
rules, then run `feature build --rebuild` and `dataset build` for each target
and threshold. Retrain models in a separate artifact directory after the
remaining blockers are fixed. Do not reuse existing serialized models as
evidence of corrected preprocessing or readiness.

```powershell
.venv/Scripts/python.exe -m pytest -q
.venv/Scripts/python.exe -m ruff check idwx/features.py idwx/datasets.py idwx/models/rf.py tests/test_forecast_inputs.py
.venv/Scripts/python.exe -m ruff check idwx/eval.py idwx/registry.py idwx/cli.py tests/test_evaluation.py tests/test_registry.py
.venv/Scripts/python.exe -m ruff check idwx/forecast_protocol.py tests/test_forecast_protocol.py
.venv/Scripts/python.exe -m ruff check idwx/models/conformal.py idwx/models/rf.py idwx/models/__init__.py tests/test_calibration.py
node --test tests/web.test.mjs
```

Verification after the calibration follow-up: 132 Python tests passed,
one optional live Nginx test skipped; five frontend tests passed. Targeted Ruff
and `git diff --check` passed. Nine input regression cases cover missing calendar years,
rolling counts, ambiguous histories, cache missingness, future-row construction,
and training-only RF preprocessing/refitting. Another 26 cases cover empty and
skipped evaluations, invalid intervals, paired baseline scoring, CLI ordering,
strict JSON, threshold isolation, artifact compatibility and run completion/
collision behavior, including a synthetic CLI train/predict round trip.
The protocol adds 21 cases, including perturbing later observations without
changing permitted inputs or earlier predictions; see its document for scope.
The calibration follow-up adds 38 cases for finite ranks, explicit failures,
chronological partitions, preprocessing isolation, refits, state/serialization
and reporting. See the calibration contract for detail.
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
