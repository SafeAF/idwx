# Config spec

Main config file:

`configs/idaho.yml`

This file defines:

- local raw data root
- local cache/model/report paths
- timezone handling
- frost thresholds and windows
- winter severity definition
- GDD definition
- backtest scheme
- model parameters

## Example config

~~~yaml
data_root: "~/historical_weather_data-1980-2022/idaho"
cache_dir: "data_cache"
models_dir: "models"
reports_dir: "reports"

timezone: "America/Boise"

stations_file: "configs/stations.yml"

frost_thresholds_c: [0.0, -2.0]

windows:
  last_spring_frost:
    start: "01-01"
    end: "06-30"
  first_fall_frost:
    start: "08-01"
    end: "12-31"

winter:
  season_months: [12, 1, 2]
  hdd_base_c: 18.0
  cold_snap_threshold_c: -10.0
  cold_snap_min_run_days: 3

  weights:
    hdd: 0.40
    tmin_extreme: 0.25
    cold_snap_days: 0.20
    cold_snap_run: 0.15

  include_precip_if_available: true

gdd:
  base_c: 10.0
  season_start: "04-01"
  season_end: "10-31"
  upper_cap_c: null

backtest:
  start_year: 1980
  end_year: 2022
  min_train_years: 20

models:
  climatology: {}

  trend:
    quantiles: [0.1, 0.5, 0.9]

  arima:
    enabled: false
    order: [1, 0, 0]

  rf:
    n_estimators: 500
    min_samples_leaf: 2
    max_features: "sqrt"
    random_state: 1337

  conformal:
    alpha: 0.2
~~~

## Rules

- `data_root` may contain `~` and must be expanded.
- `cache_dir`, `models_dir`, `reports_dir` are repo-relative.
- Date strings in `windows` are `MM-DD` and interpreted in local time.
- Winter season belongs to season_year `Y` and includes Dec(Y-1), Jan(Y), Feb(Y).
- All randomness must be seeded via config.
