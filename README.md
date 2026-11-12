# idwx — Idaho Seasonal Agro-Weather Models (frost, winter, growing season)

`idwx` is a local-first modeling repo for turning long historical station data into:

- **Last spring frost** date predictions (P10/P50/P90)
- **First fall frost** date predictions (P10/P50/P90)
- **Winter severity** forecasts (rank + percentile + components)
- **Growing season** metrics and forecasts (freeze-free window, GDD)

This is not daily weather forecasting.
This is seasonal / event forecasting for agriculture.

## Static frost dashboard delivery

The historical dashboard, exporter, portable package verifier, Nginx template,
and regression suite are implemented. Predictions remain research-only and
are excluded from the website. The available demonstration uses synthetic
data; real-data publication requires a reviewed source audit.

```powershell
uv sync --locked
uv run idwx export-web --demo --output .tmp/demo-web
uv run idwx verify-web .tmp/demo-web --allow-demo
uv run idwx package-web .tmp/demo-web --allow-demo --output .tmp/delivery/idwx-demo-site.zip
uv run python -m http.server 8765 --bind 127.0.0.1 --directory .tmp/demo-web
```

Open `http://127.0.0.1:8765/`. The ZIP contains the complete static site;
production needs only Nginx. See the [delivery report](docs/21_DELIVERY_STATUS.md)
and [deployment instructions](docs/15_DASHBOARD_DEPLOYMENT.md) for verification,
staging, atomic activation, rollback, and remaining launch requirements.

## Data location (local, not committed)

The configured raw CSV location is:

`~/historical_weather_data-1980-2022/idaho/`

That directory is currently absent in this workspace. Station metadata lists
Twin Falls; real source availability and coverage remain unverified.

Run the private [source audit](docs/20_SOURCE_DATA_AUDIT.md) before trusting
historical results:

```powershell
.venv/Scripts/python.exe -m idwx data audit
```

It writes `reports_dir/data-audit.json`, reports missing/invalid inputs and
compares existing daily caches with current aggregation. It does not modify
sources/caches or certify provenance; even an exit code of zero requires
manual review. The current workspace report is blocked by missing sources.

## Core principles

- **Local-first**: data stays on disk, never committed.
- **Reproducible**: configs define everything.
- **Comparable**: multiple models, same targets, same scoring.
- **Honest**: always output uncertainty intervals and baseline comparisons.

## Historical frost implementation

The first implementation slice of `docs/12_FROST_STATIC_WEBSITE_PLAN.md`
adds conservative historical target calculations. Build local caches with:

```powershell
uv sync
uv run idwx data build --rebuild
uv run idwx target build --rebuild
uv run pytest -q
```

Rebuild existing daily and target caches to apply the new coverage rules.
Previously generated features, datasets, and models must also be regenerated
before using these targets for modeling.

Frost events use daily minimum temperature **at or below** each configured
threshold. The defaults are 0 C (32 F) and -2 C (28.4 F). These
temperature-based freeze dates are a proxy for frost risk; surface frost and
crop damage also depend on local conditions.

Only fully covered seasonal windows produce confirmed dates. Partial records
retain provisional candidates and coverage counts. The legacy `freeze_free_days`
value is the difference between confirmed fall and spring dates; its separate
`frost_free_days` field counts strictly intervening days only when all those
days are observed and above the threshold. See
[the calculation contract and verification notes](docs/13_FROST_CALCULATIONS.md).

Export versioned historical JSON and CSV data for the dashboard:

```powershell
uv run idwx export-web --output web
```

The command reads configured stations' daily caches and recomputes frost
results under the current configuration. It excludes incomplete events from
historical quantiles and requires 10 confirmed years by default. Use
`--baseline-start YEAR --baseline-end YEAR` to select a baseline period.

Without local weather data, generate explicitly labeled synthetic examples:

```powershell
uv run idwx export-web --demo --output .tmp/demo-web
```

The export includes the complete buildless dashboard: station/threshold/year
controls, unit switching, confirmed dates and historical ranges, coverage,
an accessible history table, and CSV downloads. Open it through an HTTP server:

```powershell
uv run python -m http.server 8765 --bind 127.0.0.1 --directory .tmp/demo-web
```

Visit `http://127.0.0.1:8765/`. The demo is clearly labeled synthetic.
Selections are saved in the URL for sharing. No external APIs, fonts, or
frontend build tools are required in production.

See [the static data contract](docs/14_WEB_DATA_CONTRACT.md) and
[deployment and verification instructions](docs/15_DASHBOARD_DEPLOYMENT.md).
The Nginx configuration is in `deploy/nginx.conf`. Real-source auditing,
production browser/TLS checks, and forecast validation remain outstanding.

The [forecast correctness audit](docs/16_FORECAST_CORRECTNESS_AUDIT.md) records
input leakage fixes, required cache rebuilding, and the remaining blockers.
Local modeling commands remain research tools; no forecasts are published.

An opt-in [January 1 frost research protocol](docs/17_PRESEASON_RESEARCH_PROTOCOL.md)
now shares explicit information cutoffs across training, backtesting and
prediction. It uses confirmed older frost dates, excludes weather/WSI inputs,
and requires an explicit forecast year and issue date. It remains unvalidated
and does not add forecasts to the website.

The [RF calibration contract](docs/18_RF_CALIBRATION_CONTRACT.md) now uses
held-out calendar years and a finite-sample residual rank. Unsupported intervals
fail explicitly; older RF artifacts require retraining. Coverage on real weather
data still needs validation, and forecasts remain unpublished.

The [forecast date policy](docs/19_FORECAST_DATE_POLICY.md) now enforces station
windows and maps research bounds onto actual forecast-year dates. Backtests
report elapsed-day errors separately from raw calendar-coordinate errors.
Earlier protocol artifacts require rebuilding/retraining under input version 2.

