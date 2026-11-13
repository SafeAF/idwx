# idwx — Idaho Seasonal Agro-Weather Models (frost, winter, growing season)

`idwx` is a local-first modeling repo for turning long historical station data into:

- **Last spring frost** date predictions (P10/P50/P90)
- **First fall frost** date predictions (P10/P50/P90)
- **Winter severity** forecasts (rank + percentile + components)
- **Growing season** metrics and forecasts (freeze-free window, GDD)

This is not daily weather forecasting.
This is seasonal / event forecasting for agriculture.

## Static frost dashboard delivery

The static dashboard, exporter, portable package verifier, Nginx template,
and regression suite are implemented. The homepage fetches live Twin Falls
guidance directly in the browser: NWS point guidance is compared with
Open-Meteo for the first seven days, and Open-Meteo supplies a clearly labeled,
lower-confidence days 8–14 outlook. This operational guidance is separate from
the site's planning-year historical climatology projection: P50 is the projected
date and P10–P90 is the historical range. The site does not publish the repository's
research model predictions. The repository includes a compact daily
SQLite database for the Twin Falls ERA5/Oikolab archive, so a real-data static
site can be built without copying the private 80 MB hourly source to the
deployment host.

```powershell
uv sync --locked
uv run idwx export-web --output .tmp/twin-falls-web
uv run idwx verify-web .tmp/twin-falls-web
uv run idwx package-web .tmp/twin-falls-web --output .tmp/delivery/idwx-twin-falls-site.zip
```

The resulting ZIP is the deployable static application and contains the
generated Twin Falls JSON/CSV. The checked-in database remains the canonical
deployment input and is not served directly by Nginx.

For an explicitly synthetic demonstration instead:

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

## Twin Falls data

[`data/idwx.sqlite3`](data/idwx.sqlite3) is intentionally committed. It contains
17,114 local-calendar daily rows for `twin_falls`, derived from the source
archive plus a validated Oikolab refresh at `42.4634, -114.4975`, through
November 7, 2026. It contains no other station.
The archive is ERA5 reanalysis supplied through Oikolab, not a NOAA observing
station feed. See [`data/README.md`](data/README.md) for provenance, hashes,
coverage, and the exact reproducible build command.

The raw hourly CSV remains outside this repository. In the current sibling
checkout layout it is:

`../idaho-weather-data/archive/42.4634--114.4975-1980-01-01-2022-12-31-full-params-idaho.csv`

Rebuild the committed database after any reviewed source or aggregation change:

```powershell
uv run idwx data build-sqlite ../idaho-weather-data/archive/42.4634--114.4975-1980-01-01-2022-12-31-full-params-idaho.csv
```

The command fingerprints the source, validates the station mapping, aggregates
to coverage-aware daily records, integrity-checks a new SQLite file, and
atomically replaces the configured database. Commit the resulting database
with the corresponding code/configuration changes. Do not commit the raw CSV.

Refresh the database from Oikolab with a hidden credential prompt:

```powershell
uv run idwx data refresh-oikolab
```

The refresh checks published ERA5 availability, downloads only the Twin Falls
gap, validates continuous hourly temperature and precipitation, imports only
complete local days, records a source hash, and atomically replaces SQLite.
The API key is used at runtime and is never written to the repository.

## Core principles

- **Bounded data**: raw hourly files stay local; only the allowlisted Twin Falls
  daily SQLite database is committed.
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

Export versioned Twin Falls historical JSON and CSV data from SQLite:

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

The export includes the complete buildless dashboard: the live two-week weather
section, a planning-year climatology projection, station/threshold/year controls, unit switching,
confirmed dates and historical ranges, coverage, an accessible history table,
and CSV downloads. Open it through an HTTP server:

```powershell
uv run python -m http.server 8765 --bind 127.0.0.1 --directory .tmp/demo-web
```

Visit `http://127.0.0.1:8765/`. The historical demo is clearly labeled synthetic.
Selections are saved in the URL for sharing. The live weather section needs
browser access to `api.weather.gov` and `api.open-meteo.com`; no third-party
scripts, fonts, or frontend build tools are required in production.

See [the static data contract](docs/14_WEB_DATA_CONTRACT.md) and
[deployment and verification instructions](docs/15_DASHBOARD_DEPLOYMENT.md).
The Nginx configuration is in `deploy/nginx.conf` and allowlists those two API
origins in `connect-src`. Research-forecast validation remains separate.

The [forecast correctness audit](docs/16_FORECAST_CORRECTNESS_AUDIT.md) records
input leakage fixes, required cache rebuilding, and the remaining blockers.
Local modeling commands remain research tools; no model forecasts are
published. The website projection is the historical baseline only.

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

