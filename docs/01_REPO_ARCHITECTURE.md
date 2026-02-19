# Repo architecture

## High-level pipeline

Raw station CSVs
  -> canonical schema
  -> daily aggregation
  -> per-year seasonal features
  -> target construction
  -> training datasets
  -> model training
  -> walk-forward backtesting
  -> prediction + intervals
  -> serving (later) + Rails UI (later)

## Python package layout

idwx/
  io.py          # read raw CSVs, normalize schema
  schema.py      # canonical columns + validation
  daily.py       # hourly->daily aggregation
  targets.py     # frost dates, WSI, GDD, etc
  features.py    # seasonal summaries + lag features
  datasets.py    # build X/y tables
  models/
    base.py      # Model interface
    climatology.py
    trend.py
    rf.py
    arima.py
    conformal.py
  eval.py        # backtesting + metrics
  registry.py    # save/load models + metadata
  cli.py         # typer entrypoint

## Artifact layout (local)

data_cache/
  canonical/
  daily/
  datasets/

models/
  <target>/<station>/<model_name>/<run_id>/
    model.pkl
    config.yml
    metrics.json
    feature_schema.json

reports/
  <target>/<model_name>/
    summary.csv
    per_station.csv
    plots/

## Determinism

- All randomness must be seeded.
- Every model run writes:
  - git commit hash
  - config hash
  - dataset hash
  - feature schema hash
