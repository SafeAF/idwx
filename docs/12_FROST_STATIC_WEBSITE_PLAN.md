# Frost dates and Nginx website implementation plan

## Goal and scope

Identify the last spring frost and first fall frost for each Idaho station and year, summarize their historical distribution, and display the results in a responsive HTML/CSS/JavaScript website served by Nginx.

Deliver historical results first. Add seasonal predictions after validating the existing models. Clearly distinguish observed dates, historical expectations, and model predictions throughout the interface.

Use the existing local weather files and Python package. Nginx will serve static files and generated JSON; the initial release does not require Rails, Node.js, a database, or an application server in production. This replaces the Rails direction in `docs/10_RAILS_GUI_PLAN.md` for this website.

## Existing foundation

- `idwx/io.py` and `idwx/daily.py`: ingestion and daily temperature aggregation.
- `idwx/targets.py`: last spring frost, first fall frost, and freeze-free interval calculations.
- `configs/idaho.yml`: thresholds, seasonal windows, timezone, and training years.
- `configs/stations.yml`: station names and coordinates; currently one configured station, Twin Falls.
- `idwx/models/`, `idwx/eval.py`, and the CLI: climatology, trend and random forest models, evaluation, and predictions.

The README describes weather data through 2022. Confirm actual file availability and coverage before making data-dependent estimates. The current prediction command targets the year after the latest dataset year; it must not be presented as a current-year forecast automatically.

## Phase 1: Define and verify frost calculations

1. Inspect available stations, source schemas, units, date ranges, timestamp conventions, duplicates, and missing observations. Verify daily minima represent each station's local reporting day; add station timezone overrides if required.
2. Define an event as daily minimum temperature at or below the selected threshold. Default to 0 degrees C / 32 degrees F. Retain the existing secondary threshold of -2 degrees C / 28.4 degrees F, displaying its exact value. If exact 28 degrees F is needed, configure its Celsius equivalent rather than relabeling -2 degrees C.
3. Explain that temperature-based freeze dates are a proxy for frost risk. Actual surface frost and crop damage also depend on local conditions.
4. Compute the last qualifying date in the spring window and first qualifying date in the fall window. Preserve the event date and observed minimum temperature for traceability.
5. Review the current January-June and August-December windows. July is currently excluded. Inspect summer freezes and provide configurable station windows; flag summer-freezing stations and avoid claiming an uninterrupted freeze-free season when qualifying events occur between the selected dates.
6. Measure valid daily minimum coverage against the full expected calendar window, including entirely absent days and years. Distinguish `observed`, `no_event_in_complete_window`, `incomplete_window`, and `insufficient_data`; record gaps and provisional candidates. An observed freeze in an unfinished spring window is not yet a confirmed last spring freeze.
7. Use complete windows for confirmed historical dates initially. Do not fill missing minima in ways that could invent or erase frost events. Retain partial records with visible status, excluded from definitive summary statistics.
8. Preserve the existing date-difference interval for compatibility, but label it explicitly. Count strictly intervening frost-free days separately as `max(0, fall_date - spring_date - 1)` only when coverage and temperatures support that claim. Return null when dependencies are invalid or unknown.

**Completion:** Hand-checked station-years agree with the program, and missing data cannot silently become a confirmed frost date or an absence of frost.

## Phase 2: Historical summaries and prediction validation

- Produce a per-station, per-year table for every configured threshold, including event dates, temperatures, coverage, status, and interval length.
- Calculate historical P10/P50/P90 date quantiles, earliest/latest dates, valid-year count, and baseline period. Use a consistent calendar representation so leap years do not shift month/day summaries.
- Label the historical median as a typical date. Explain spring and fall quantiles separately: spring P90 is a later last-freeze date, while fall P10 is an earlier first-freeze date. Quantiles are not guarantees for an individual year.
- Audit forecasting features, imputation, and calibration to ensure each backtest uses only information available at its declared issue date. Fit preprocessing within training folds and do not copy unavailable seasonal values into future rows as if observed.
- Backtest both frost targets with chronological walk-forward evaluation. Report error in days, bias, interval coverage, interval width, and comparison with climatology.
- Publish model predictions only after meeting documented accuracy and calibration criteria; otherwise retain the historical baseline. Check whether conformal bounds support quantile labels before calling them P10/P90.
- Require explicit forecast season year and issue date. Display the last observation date and training period. If the available records stop in 2022, show that limitation prominently and defer current-season predictions until data and features support them.

**Completion:** Historical summaries are reproducible; any displayed prediction has its forecast year, uncertainty, baseline, and validation results attached.

## Phase 3: Export a static website data contract

Add an `idwx export-web` command that reads validated outputs and writes a complete publication directory:

```text
web/
  index.html
  css/styles.css
  js/app.js
  js/charts.js
  data/manifest.json
  data/<release-id>/stations.json
  data/<release-id>/stations/<station-id>.json
  data/<release-id>/downloads/frost-history.csv
deploy/
  nginx.conf
```

The data contract should include:

- Schema and release versions, generation timestamp, source coverage dates, baseline period, configured windows and thresholds.
- Station ID, name, coordinates, elevation, and timezone.
- Annual event records, quality status, historical summaries, and optional validated forecast/model metrics.
- ISO calendar dates (`YYYY-MM-DD`), Celsius internally, JSON `null` for missing values, and finite numeric values only.

Keep raw observations, local paths, and trained model files outside the public directory. CSV exports contain the published results. Parse date-only values without browser timezone shifts. Load only the selected station's dataset, and use the manifest to keep every request within one release.

**Completion:** The frontend can render all supported states from documented JSON without invoking Python or fetching an external weather API.

## Phase 4: Build the HTML/CSS/JavaScript dashboard

Use semantic HTML, responsive CSS, and browser JavaScript modules with no required frontend build step.

The dashboard includes:

1. Station selector, threshold selector, year selector, and Celsius/Fahrenheit display toggle.
2. Summary cards for last spring frost, first fall frost, and the verified season interval, with explicit observed/historical/predicted labels.
3. History chart and accessible table showing annual dates, the historical median, and P10-P90 historical range. Add forecast intervals only where validated predictions exist.
4. Detail table with event temperatures, quality status, available years, and CSV download.
5. Methodology and freshness information: event definition, windows, source period, quantile explanation, last update, and model performance when applicable.

Provide keyboard controls, readable mobile layouts, loading and error states, and clear messages for incomplete seasons, missing events, insufficient history, and stale data. Reflect selections in URL parameters for shareable views. Use a small SVG chart or a locally bundled chart library chosen during implementation.

**Completion:** A user can choose a station and threshold, understand the dates and their uncertainty, inspect history, and download the displayed results on desktop or mobile.

## Phase 5: Nginx deployment and updates

- Provide an Nginx server configuration with a dedicated document root, correct MIME types, and real 404 responses for missing assets and JSON files. Enable compression and HTTPS using the server's certificate setup.
- Revalidate HTML, JavaScript, CSS, and the release manifest initially. Cache release-specific data for longer because its paths never change. Introduce hashed frontend asset names if long-lived asset caching is needed later.
- Keep the Python environment on a data-processing machine or in a separate scheduled job. Deploy only the generated website directory to Nginx.
- Build and validate a complete release in staging, publish its versioned data, and switch the active release atomically. Retain previous releases long enough for open browser sessions and rollback.
- Run regeneration after new observations arrive or configuration changes. Add a scheduler once the data update frequency is established; do not imply fresh observations merely because an export ran recently.
- Document installation, export, deployment, `nginx -t`, HTTP smoke checks, update cadence, and rollback.

**Completion:** Nginx serves the website and its data independently of Python, and a failed data refresh leaves the previous valid release available.

## Phase 6: Verification and delivery

Test meaningful calculation cases: threshold equality, multiple events, complete windows without events, missing days, sparse hourly coverage, missing years, leap dates, duplicate records, window boundaries, July freezes, and unfinished seasons. Verify interval counting explicitly.

Validate exported schemas, null handling, consistent release references, and reproducibility. If forecasts are included, verify chronological splits, training-only preprocessing, forecast-year selection, and baseline comparisons.

Check browser station/threshold/year changes, units, URL state, chart/table consistency, download contents, keyboard use, mobile layout, and failed data requests. Test against Nginx, including MIME types, caching, missing-file responses, and rollback.

Deliver the calculation improvements, static exporter, website source, sample generated outputs clearly labeled if real data is unavailable, Nginx configuration, deployment instructions, and a short verification report.

## Recommended delivery order

1. Data audit and verified historical frost calculations.
2. Historical summaries and versioned JSON/CSV export.
3. Dashboard and Nginx deployment package: the first useful release.
4. Validated seasonal predictions as a subsequent release.

Initial assumptions: Idaho stations, local historical CSV inputs, a public read-only dashboard, and updates by batch generation. Additional geography, live alerts, accounts, and interactive server-side analysis can be scoped separately if needed.

## References

- NOAA NCEI Climate Atlas: freeze-date calculations based on daily minimum temperatures at or below 32 degrees F: https://www.ncei.noaa.gov/products/climate-atlas
- NOAA NWS frost/freeze information: local conditions influence actual frost and freeze occurrence: https://www.weather.gov/fgf/frostfreeze
- NGINX static content documentation: https://github.com/nginx/documentation/blob/main/content/nginx/admin-guide/web-server/serving-static-content.md
