# Web data exploration and analytics plan

Status: phases 1–4 implemented locally, November 21, 2026. Temperature/frost
exploration, monthly anomalies, and coverage-aware seasonal metrics are
included. Phase 5 is implemented for the independently audited original hourly
archive only; refresh-period precipitation remains unavailable. A growing-year
requirements assessment is also included. See [methods and audit](24_PRECIPITATION_GROWING_ASSESSMENT.md).
Production deployment and visual browser acceptance remain separate. See the
[implementation and verification record](23_DATA_EXPLORER_VERIFICATION.md).

## Goal

Add an **Explore data** workspace where someone can investigate historical
Twin Falls weather, compare years, understand frost patterns, and download
the exact data behind a chart. Keep the existing weather homepage and its
separation of live guidance, historical reanalysis, and planning climatology.

Start with temperature and frost history. Expand to growing-season and winter
analytics after the shared exploration tools work. Precipitation follows a
review of its source and aggregation semantics.

## Current foundation

The following facts were checked in the local checkout on `main` at
`a17896a9b668255349a114b97e5f4c26f84305ef`:

- `web/` is a buildless HTML/CSS/JavaScript application. `web/js/charts.js`
  renders an SVG annual freeze-date chart; tables, station/threshold/year
  controls, unit switching, URL state, and CSV downloads already exist.
- `idwx/web_export.py` publishes annual frost records and calendar-day
  temperature climatology into immutable, checksummed releases. Daily history
  is not currently published for browser exploration.
- A read-only query of `data/idwx.sqlite3` found 17,114 Twin Falls daily rows,
  covering local dates 1979-12-31 through 2026-11-07. There are 17,113 non-null
  values for each temperature field and precipitation; the first date is
  flagged `insufficient_coverage`. Presence alone does not validate a metric.
- These are ERA5/Oikolab historical reanalysis values at one configured
  coordinate, not physical station observations. KTWF live observations are a
  separate source in `web/js/forecast.js`.
- `idwx/web_verify.py` strictly allowlists publication paths and fields. New
  data files and frontend assets require coordinated exporter, verifier,
  package, and wheel changes.
- `idwx/targets.py` has research GDD and winter calculations, but some sums
  accept partial periods and runs do not explicitly reindex absent dates.
  These need coverage rules before they become public analytics.

The plan extends the static architecture in
[the dashboard plan](12_FROST_STATIC_WEBSITE_PLAN.md) and
[the public data contract](14_WEB_DATA_CONTRACT.md). The older
[Rails GUI proposal](10_RAILS_GUI_PLAN.md) remains separate future work.

## Questions the UI should answer

| Question | View | Useful interaction |
| --- | --- | --- |
| How did temperatures change through a particular year? | Daily minimum/mean/maximum time series | Select dates, inspect values, compare with calendar-day climatology |
| How does one year compare with another? | Up to three years overlaid by month/day | Pick years, toggle series, view a shared table |
| Which months or years were unusually warm or cold? | Monthly temperature anomaly heatmap | Select a cell to open that period's daily view |
| How variable are spring/fall freeze dates and frost-free intervals? | Annual frost plot and distribution | Filter years and threshold; inspect confirmed and incomplete seasons |
| How much data supports the result? | Coverage summary and data table | Inspect valid/expected days and exclusion reasons |
| Can I reproduce this comparison elsewhere? | Filtered CSV and shareable URL | Export rows plus their units, filters, baseline, and release identity |

## Workspace and interactions

Add `web/explore.html`, reached through an **Explore data** link from the
homepage. Give it three views: **Temperature**, **Frost & growing season**,
and **Data table**. Add **Precipitation** only when its validation gate passes.
Keep the existing homepage controls and meanings intact.

The explorer has shared controls for station, date/year range, temperature
units, view, and aggregation. View-specific controls include comparison years,
the configured freeze threshold, and the metric. Initially Twin Falls is the
only available station; do not imply other stations have data.

Use explicit date controls as the primary selection method. Chart selection
can update the same controls, with a visible reset action. Hover, keyboard
focus, and touch selection expose the date, value, units, and coverage. Every
chart has an accessible table and a concise text summary. Series must remain
distinguishable without color; honor reduced motion and mobile layouts.

Filters must update the chart, statistics, table, and filtered download from
one selection model. Distinguish the **analysis period** from the **reference
baseline**. For the first release, use the published baseline as a fixed,
visible reference; filtering years must not silently recalculate it. A custom
baseline selector can follow when its calculation contract is implemented.

Store validated controls in query parameters, including comparison years.
Restore them on reload and browser back/forward. A shared link normally opens
the latest published release with those filters; display the loaded release
ID. Exported files include the exact release ID for reproducibility. A URL
pinned to a retained release can be added later with an explicit unavailable
state when that release has expired.

## First useful release

Deliver these together so users can perform a complete exploration:

1. Temperature time series for a selected period, with minimum, maximum, and
   mean series, explicit gaps, and an optional historical P10/P50/P90 overlay.
2. Comparison of up to three years using a common month/day axis.
3. An explorer version of the frost chart with an inclusive year-range filter,
   verified frost-free-day series, and event-date distribution. Keep confirmed,
   complete-no-event, incomplete, and insufficient-data counts visible.
4. Summary statistics: valid/expected daily count, arithmetic mean temperature,
   minimum daily minimum, maximum daily maximum, and confirmed frost counts.
   Label the statistic precisely rather than calling every center an average.
5. An accessible filtered table, shareable state, and CSV export that agrees
   with the selected chart and its statistics.

The monthly anomaly heatmap is the next increment after this first release.
GDD, HDD, cold spells, precipitation, and fitted trends are later increments.

## Data publication and computation

Keep SQLite on the batch-processing side. Export only allowlisted, derived
public JSON/CSV; serve no raw hourly files, database, private paths, model
artifacts, or general query endpoint. No API server or browser SQL engine is
needed for the initial single-station daily dataset.

Implemented additions inside the existing `data/<release-id>/` tree:

| File | Purpose |
| --- | --- |
| `analytics/<station-id>/index.json` | Available years, coverage, metric definitions, and URLs for daily chunks |
| `analytics/<station-id>/daily/<year>.json` | One local calendar year of public daily temperatures and quality/coverage fields |
| `analytics/<station-id>/summary.json` | Monthly/annual temperature summaries and later supported seasonal metrics |

Daily rows include `date`, `tmin_c`, `tmax_c`, `tmean_c`, an explicit temperature
validity status, and source coverage information such as `n_obs`. Reindex to
the calendar so absent dates are represented as unavailable, with null values
and a distinct missing-record status. Use documented public status codes;
do not expose arbitrary internal flag text. Precipitation fields are added
only after review. Year chunks include partial boundary years and disclose
their coverage; a full calendar row count is not proof of observations.

All documents carry compatible schema/release/data-kind identities, and all
new files participate in content hashing and the manifest file inventory.
Extend the station document with a release-relative analytics index URL.
Fetch only the selected station and needed years, cache by release, cancel
obsolete requests, and reject mixed releases before rendering.

Version the extended public schema explicitly. Because the current verifier
accepts one schema version and checks retained releases, implement versioned
validation for both existing 1.0.0 releases and the new schema before
retaining old and new releases together. Never rewrite old immutable files.
Update `docs/14_WEB_DATA_CONTRACT.md` to cover the already implemented
`temperature_climatology` field as well as the new analytics contract.

Compute canonical summaries and metric validity in Python during export.
The browser filters daily values and computes simple selection statistics
under the same documented rules. Use independent small fixtures to check
Python/JavaScript agreement. Initially extend the existing SVG renderer and
plain JavaScript modules; evaluate another renderer only if measured size or
interaction limits justify it. Do not load chart code from a third-party CDN.

## Statistical and coverage rules

- **Temperature:** use finite, coverage-valid values only. Show valid and
  expected calendar-day counts for each metric. A partial-period mean may
  appear as an explicitly labeled mean of available days; exclude incomplete
  periods from full-period rankings and comparisons by default.
- **Frost:** reuse `idwx/frost.py` and `idwx/history.py`. Events use `Tmin <=`
  the configured threshold. Do not treat missing or complete-no-event years
  as date zero. Keep provisional candidates out of confirmed statistics.
  Keep date differences separate from verified strictly intervening frost-free
  days. Quantiles remain conditional on a confirmed event.
- **Dates:** compare month/day on the existing leap-year-2000 calendar.
  February 29 stays distinct; non-leap years have no invented February 29.
  Calculate elapsed intervals on actual dates and use the station's local
  calendar without browser timezone conversion.
- **Quantiles:** frost uses the existing nearest-rank `inverted_cdf` method
  and minimum confirmed-year policy. Temperature climatology currently uses
  linearly interpolated quantiles with at least five paired valid days for
  each month/day. Document these separately; do not silently change methods.
- **Anomalies:** initially define daily anomaly as the selected temperature
  minus the corresponding published calendar-day P50, labeled **departure
  from historical median**. Monthly heatmap cells aggregate paired daily
  anomalies only when the entire month has valid observations and reference
  values; otherwise show coverage and an unavailable cell. Derive any future
  arithmetic-mean baseline separately and name it explicitly.
- **GDD/HDD:** use canonical Celsius and publish bases, caps, windows, and
  degree-day units. The initial GDD convention is
  `max(0, min(tmean_c, cap_c) - base_c)` with no cap when unset; HDD is
  `max(0, base_c - tmean_c)`. Require full window coverage for a final total.
  A partial year's running subtotal stops at the first missing day and is
  labeled through its last continuous valid date. December–February winter
  metrics belong to the year containing January/February.
- **Cold spells:** operate on a complete daily calendar. Missing days break
  candidate runs; a period with missing data cannot claim a confirmed maximum
  run or complete cold-day total.
- **Precipitation:** the stored sources both record incremental aggregation,
  but the adapter selects its mode heuristically and coverage is currently
  temperature based. Confirm source units, interval/reset semantics, local-day
  boundaries, and precipitation-specific completeness against the source
  before publishing totals, wet-day counts, or cumulative comparisons.
  Never replace missing rain with zero. Audit daily values and at least one
  independently checked monthly total across the archive/refresh boundary
  before enabling refresh-period amounts. The implemented archive-only scope
  uses a separately reviewed daily supplement, with three independent monthly
  reference totals. No refresh-boundary totals are claimed.
- **Trends:** defer fitted slopes, significance, and climate attribution.
  A later descriptive slope must state its period, units, sample count,
  coverage exclusions, and method; it is not a forecast or causal claim.

## Implementation sequence

| Phase | Work and likely files | Completion evidence |
| --- | --- | --- |
| 1. Contract and fixtures | New `idwx/analytics.py`; update data contract; fixtures for gaps, leap days, partial years, and no-event windows | Agreed formulas, statuses, example public documents, and independently calculated expected results |
| 2. Static exports | `idwx/web_export.py`, `idwx/web_verify.py`, export/verifier tests | Real-data export and package verify; deterministic release identity; tampering/mixed versions fail; old releases still verify |
| 3. Explorer MVP | New `web/explore.html`, `web/js/explorer.js`, `web/js/analytics.js`; extend charts/styles and homepage navigation | All five first-release capabilities work together; controls, URL, table, and CSV agree |
| 4. Patterns and seasonal metrics | Monthly heatmap, GDD/HDD and cold-spell summaries; coverage fixes where needed in `idwx/targets.py` | Incomplete periods cannot appear as complete totals or rankings; heatmap drill-down works |
| 5. Precipitation | Source review, metric-specific coverage, optional export/view extension | Reproducible aggregation evidence and independent reference totals before UI enablement |

For every new asset, update `WEB_ASSETS`, the verifier's path rules, and
`pyproject.toml` wheel force-includes. Exercise both checkout and installed
wheel exports. Keep all analytics under the existing immutable data path so
the current static hosting/cache design can be reused.

## Verification and acceptance

Use the existing Python and Node test suites, adding meaningful cases for the
new behavior. Implementation acceptance requires:

- Known fixture results for filtering, statistics, quantiles, unit conversion,
  missingness, threshold equality, leap days, and seasonal boundaries. Absolute
  temperatures and temperature differences use the correct distinct unit
  conversions; changing display units never changes canonical calculations.
- Export/verifier/package tests for all new files, full digest coverage,
  unsupported schema handling, real versus synthetic labels, failure
  preservation of the active manifest, and retained release compatibility.
- Frontend integration tests for restored URLs, back/forward, invalid filters,
  rapid selection changes, failed chunks and retry, and chart/table/CSV
  agreement. A failed year chunk must show an error, not silently omit a year.
- Real-data export and local HTTP preview with live weather requests disabled:
  historical exploration must work independently of those providers. Check
  wheel inclusion and package verification through the existing delivery flow.
- Actual desktop/mobile browser checks for chart readability, keyboard and
  touch controls, focus, downloads, and accessible tables. DOM tests alone do
  not establish browser layout or interaction quality.
- Record payload sizes and browser performance against the full archive.
  Provisional budgets: no daily chunks fetched on the homepage; explorer shell
  plus default selected data under 1 MB compressed; filter-to-render under
  200 ms after data load on a recorded reference device. Aggregate long ranges
  to monthly/yearly marks, then load daily detail on selection if needed.

Success means someone can open the explorer, compare historical years, inspect
temperature and frost patterns, understand exclusions, share their selection,
and download matching values without installing tools or needing a server
analysis session.

Implementation and production release are subsequent tasks. Deployment must
follow the existing verified-package, staging, atomic promotion, and rollback
process in [the deployment guide](15_DASHBOARD_DEPLOYMENT.md), within the
repository's host and vhost boundaries. This plan adds no training, collection,
accounts, API service, or other deployment scope.
