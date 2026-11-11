# Opt-in preseason frost research protocol

`frost_jan01_v1` is a conservative input protocol for local research with the
existing climatology, trend and RF models. It does not validate the models,
enable forecast publication or change the static website data contract.
The default legacy modeling path remains available for compatibility and
retains the leakage limitations in `16_FORECAST_CORRECTNESS_AUDIT.md`.

## Forecast context and information boundary

The issue date is **station-local January 1 of explicit season year Y**.
Training and prediction both require `--season-year Y --issue-date Y-01-01`.
Other dates and implicit year selection are rejected for this protocol.
Evaluation uses January 1 separately for each configured test year.

Only observations and confirmed target labels through **December 31 of Y-2**
are eligible. The entire intervening calendar year is deliberately held back.
For example, the 2024 research forecast issued January 1, 2024 uses confirmed
training outcomes no later than 2022 and reports an information cutoff of
December 31, 2022. This is an intentionally conservative benchmark, not a
claim that a year of source latency is necessary or that actual delivery
latency has been measured. A protocol using more recent observations needs
its own version and availability audit.

The holdback applies to training labels as well as predictors. Each historical
training row's features are constructed relative to that row's own January 1
issue date, not the outer backtest or prediction date. Missing years cannot
be replaced by an earlier available season.

Source observation dates do not prove that records or revisions were available
at that historical issue date. Data vintage, provenance, ingestion quality,
station metadata history and delivery cadence still require real-source
auditing. The protocol cannot establish those facts from current daily caches.

## Permitted inputs and target definition

The exact model input columns are:

| Column | Meaning |
| --- | --- |
| `station_id`, `season_year` | Model identity/context; RF excludes these as predictors. |
| `year_index` | Explicit forecast year for the trend component. |
| `target_lag2` | Confirmed event date in year Y-2, otherwise missing. |
| `target_roll3_lag2` | Mean of confirmed event dates in Y-4 through Y-2, requiring at least two years. |
| `target_roll5_lag2` | Mean of confirmed event dates in Y-6 through Y-2, requiring at least three years. |
| `target_lag2_missing` | 1 when that exact year's lag is unavailable. |

No weather aggregates, WSI, WSI lags, target-year outcomes, future metadata,
or arbitrary additional numeric columns enter the model. The selection is an
explicit allowlist shared by CLI training, walk-forward evaluation and
prediction. Missing numerical values remain missing until model training
preprocessing; RF retains its existing training-only median/zero behavior.

Targets are last spring / first fall events at or below the exact selected
threshold using each station's configured window. Only complete-window
`observed` outcomes supply labels or history features. Incomplete windows,
complete windows without an event, absent years and nonfinite targets remain
unavailable. Predictions are therefore conditional date estimates; they do not
estimate the probability that an event occurs in a season.

Dates are mapped by month/day onto **leap year 2000**: March 1 is always
calendar day 61. Internal `target_doy` and model `p10`/`p50`/`p90` names are
retained for compatibility but refer to that common calendar in this protocol.
These names do not certify forecast tail probabilities. Fractional model
outputs remain numeric calendar coordinates; no rounding into a calendar date
or conversion of February 29 into a non-leap forecast year is performed.
Nonfinite, crossed or out-of-calendar [1, 366] bounds are rejected for protocol
prediction and marked `invalid_prediction` in evaluation. Target-window bounds
and other date conversion policies still require validation.

## Rebuilding inputs and artifact identity

`--protocol frost_jan01_v1` reads only configured stations' daily Parquet
caches and recomputes event records under the current frost calculation rules.
It bypasses target, feature and dataset caches, including older whole-history
imputation and WSI artifacts. Missing/empty daily station caches fail explicitly;
there is no synthetic fallback. Protocol inputs are built in memory each run.

Protocol/input version, target, exact threshold and calendar reference are
attached to the input frame. Missing or incompatible provenance is rejected.
This does not independently certify that a daily cache was made using the
current hourly ingestion rules. Rebuild old daily caches before real-data use.

Artifacts have a separate protocol and forecast-year namespace:

```text
models/<target>/<station>/<model>/threshold_0p0/frost_jan01_v1/season_2024/<run-id>/
```

The registry format remains version 2; its optional forecast context
now records protocol/input/calendar versions, explicit year/issue/cutoff,
effective station timezone and last recorded observation before the cutoff.
Input provenance is recorded separately. Both are verified alongside the
configuration, dataset and feature schema before loading a protocol model.
Legacy artifacts and artifacts for another protocol/year cannot be selected
as substitutes. Configuration/data changes inside the permitted history
require retraining. A newer observation outside the permitted cutoff does not
change the protocol's training inputs or identity.

The artifact's training span includes all input rows passed to `fit`; for RF
that includes its internal calibration partition. An old source can still
produce an explicitly requested later-year research result. The recorded last
observation and training years expose that limitation; the command does not
represent the result as current, validated or newly observed weather.

## Commands

These require real, audited daily caches; the examples do not generate demo
data. Adjust the explicit research year and config intentionally.

```powershell
# January 1 walk-forward folds over the configured evaluation years.
uv run idwx eval --target last_spring_frost --threshold 0 --protocol frost_jan01_v1 --models climatology,trend,rf
uv run idwx eval --target first_fall_frost --threshold -2 --protocol frost_jan01_v1 --models climatology,trend,rf

# Train both the selected model and its required baseline for one explicit issue.
uv run idwx train --target last_spring_frost --threshold 0 --protocol frost_jan01_v1 --season-year 2024 --issue-date 2024-01-01 --models climatology,rf
uv run idwx predict --target last_spring_frost --station twin_falls --threshold 0 --protocol frost_jan01_v1 --season-year 2024 --issue-date 2024-01-01 --model rf
```

Protocol reports are separated at
`reports/<target>/<model>/<threshold-key>/frost_jan01_v1/`. Yearly rows include
protocol, issue date, information cutoff and calendar reference, plus the
existing fold statuses, counts and paired baseline predictions.

Prediction JSON includes `validation_status: unvalidated_research`, explicit
context, training period and `prediction_kind: unvalidated_model_interval`.
Its `prediction` object uses `lower`, `point`, `upper`; the separately named
`baseline_climatology` retains the historical distribution field names.
Neither is published by `export-web`.

## Verification and remaining requirements

The follow-up adds 21 synthetic regression cases covering both frost targets,
later-observation perturbations, earlier-fold invariance, approved model
columns, the training holdback, missing/partial years, leap normalization,
station windows, timezone context, issue-date validation, cache bypass,
protocol artifact context, calendar-bound rejection, and CLI train/predict
round trips with climatology/trend and RF. Full verification after the calibration
follow-up: **132 Python
tests passed**, one optional Nginx test skipped; five frontend tests passed.
Targeted Ruff and `git diff --check` passed.

Remaining blockers include real-data/vintage auditing, calibrated intervals,
sufficient calibration samples, threshold/window-specific performance and
baseline comparisons, promotion criteria, forecast-window support, calendar
conversion, and production browser/TLS checks. This protocol supplies a
reproducible research boundary; it is not evidence of predictive accuracy or
calibration. No forecasts are published.

Subsequent calibration follow-up: RF now requires the separately versioned
held-out calibration contract described in `18_RF_CALIBRATION_CONTRACT.md`.
It reserves at least five calibration years by default plus a full year after
the fitting partition, and requires eight fitting years. Earlier RF artifacts
must be retrained. The protocol's own input version is unchanged; its RF round
trip now supplies the larger required
history. Temporal coverage and real-data validation remain unmeasured.
