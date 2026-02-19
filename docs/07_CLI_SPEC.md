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
- `--config ...`

### `idwx eval`
Run walk-forward backtest.

Args:
- `--target ...`
- `--models ...`
- `--config ...`

### `idwx predict`
Predict upcoming season.

Args:
- `--target ...`
- `--station "Twin Falls"`
- `--model rf`
- `--config ...`

Output:
- JSON to stdout
- optional pretty table

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
