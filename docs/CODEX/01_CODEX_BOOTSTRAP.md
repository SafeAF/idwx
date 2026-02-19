# CODEX_BOOTSTRAP — build idwx from scratch

You are building a new repo from scratch.

## Hard constraints
- Local data root: `~/historical_weather_data-1980-2022/idaho/`
- No data committed
- Must support multiple stations and multiple model types
- Must implement walk-forward backtesting
- Must output uncertainty intervals

## Step 1 — scaffolding
- Create python package `idwx/`
- Add `pyproject.toml` (uv friendly)
- Add `idwx/cli.py` using Typer
- Add `idwx/__main__.py` that calls the CLI

## Step 2 — config + parsing
- Implement YAML config loading
- Expand `~` in paths
- Load stations metadata from `configs/stations.yml`

## Step 3 — ingest + daily cache
- Discover CSVs under data_root
- Parse into canonical schema
- Write parquet caches per station

## Step 4 — targets
- Build last spring frost, first fall frost, freeze-free days
- Build WSI

## Step 5 — features
- Build per-year seasonal features
- Include lag features + trend

## Step 6 — models
- climatology baseline
- trend quantile regression
- RF regressor + conformal intervals

## Step 7 — evaluation
- walk-forward backtest
- output reports tables

## Step 8 — predict
- JSON output including P10/P50/P90 and baseline comparison
