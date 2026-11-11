# CLI spec

The repo exposes a single CLI:

`idwx`

## Commands

### `idwx data build`
Build canonical + daily parquet cache.

Args:
- `--config configs/idaho.yml`
- `--rebuild`

### `idwx dataset build`
Build X/y datasets for targets.

Args:
- `--target first_fall_frost`
- `--threshold 0.0`
- `--config ...`

### `idwx train`
Train one or more models.

Args:
- `--target ...`
- `--models climatology,trend,rf`
- `--station all|<name>`
- `--threshold 0.0`
- `--config ...`

### `idwx eval`
Run walk-forward backtest.

Args:
- `--target ...`
- `--models ...`
- `--threshold 0.0`
- `--config ...`

### `idwx predict`
Predict upcoming season.

Args:
- `--target ...`
- `--station "Twin Falls"`
- `--model rf`
- `--threshold 0.0`
- `--config ...`

Output:
- JSON to stdout
- optional pretty table

### Research model artifacts and compatibility

Modeling commands remain local research tools. The static exporter publishes
no forecasts. `eval` and `predict` label JSON as `unvalidated_research`.

Training writes version 2 artifacts at:

```text
models/<target>/<station>/<model>/threshold_0p0/<run-id>/
  model.pkl
  feature_schema.json
  metrics.json
  config.yml
```

The threshold directory is `threshold_m2p0` for -2 C. Float thresholds retain
their exact Python value in identity; negative zero shares the zero directory.
No-threshold Python API calls use `threshold_none`. Nonfinite thresholds fail.
Automatic run IDs include UTC time with microseconds and a UUID. Existing
explicit run IDs cannot be overwritten. `config.yml` is installed atomically
after the other files as the completion marker; incomplete runs are ignored.

`predict` looks up both the requested model and climatology at the requested
threshold. Before loading either pickle, it verifies artifact version,
target/station/model/threshold, configuration, dataset and feature-schema
hashes. Configuration identity includes station window overrides and the
station metadata file's bytes, so changes to timezone metadata invalidate
existing models. This is conservative: any hashed configuration change,
including path or station-file formatting changes, requires retraining.
Dataset changes likewise require retraining rather than silently using an
older model. The latest completed incompatible/corrupt run fails lookup;
there is no automatic fallback to an older run.

Legacy artifact directories without threshold identity are intentionally not
loaded. Preserve them for research reproducibility, rebuild corrected caches,
and retrain to produce new compatible artifacts. The artifact version is a
registry format version, not proof of validated feature availability or cache
freshness. Issue-date protocols and version rejection of old derived caches
remain outstanding for this legacy path; see `16_FORECAST_CORRECTNESS_AUDIT.md`.

### Opt-in January 1 frost research

`train`, `eval` and `predict` accept `--protocol frost_jan01_v1`. Training and
prediction additionally require `--season-year YEAR --issue-date YEAR-01-01`.
The protocol excludes weather/WSI, recomputes confirmed frost records from daily
caches, and holds back the entire preceding calendar year. Inputs and targets
use the shared leap-year calendar. Artifact paths add
`frost_jan01_v1/season_YEAR/` after the threshold directory, and lookup verifies
the exact issue context and input provenance. Protocol reports add a protocol
subdirectory. These artifacts cannot substitute for legacy models or another
forecast year. No modeling mode publishes predictions to the website.

See [the protocol contract and examples](17_PRESEASON_RESEARCH_PROTOCOL.md)
for the information cutoff, output fields and remaining validation requirements.

## Config keys

- data_root
- cache_dir
- models_dir
- reports_dir
- stations
- frost_thresholds
- windows
- model_params
- backtest
