# IdaWeather comparison and recommended additions

Reviewed November 21, 2026. IDWX starting revision:
`51678e43d8cc025d5f2d1548957bf7d93cc59261` on `main` (initially clean).
The sibling projects were inspected read-only:

- `~/source/IdaWeather`, revision `b94609e`: graph form, controller, study
  rendering, graph/study models, and graph interactions.
- `~/source/IdaWeatherAPI`, revision `1d38c05`: coordinate endpoint,
  Ruby/Python statistics and historical helpers, historical tests, and ANN
  prototype structure. No legacy service or training script was run.

This is a local source update, not a production deployment. Existing daily
inputs, provider collection, research models, and public data schema remain
unchanged. The frontend computes the new analysis from pinned daily chunks.

## Feature comparison

| Legacy capability | IDWX disposition |
| --- | --- |
| Range across years; min/max/mean | Already present: Yearly statistics, for daily minimum/mean/maximum temperature, with complete-year coverage requirements. |
| Compare one month across years | Already present: Same month across years, including leap-month completeness. |
| Month-by-month profile of a year | Already present: Monthly profile with mean/min/max. |
| Line, scatter, bar, area | Already present in the explorer. |
| Pie option in the legacy form | Not ported: these time series, signed temperatures, and overlapping statistics are not shares of a whole. |
| Daily historical low, high, average | Added: Calendar-day records & averages, over selectable whole calendar years. |
| Historical average at a particular hour | Deferred: validated daily data cannot recover an hourly average. |
| Hourly series for a month | Deferred: needs reviewed hourly publication, units, timezone, and coverage rules. The daily timeline is already present but is not hourly parity. |
| Studies containing named graphs and descriptions | Added: browser-local collections of chart selections, with notes, open/remove controls, and JSON backup/import. Selections reopen one chart at a time; no simultaneous study gallery or account-backed storage. |
| Location selector with many Idaho locations | Current explorer is station-aware; the public dataset is allowlisted to Twin Falls. Additional locations require reviewed inputs. |
| Soil, wind, humidity, radiation, pressure, snow, cloud variables | Not in the validated public daily temperature/precipitation contract. Retained as recommended follow-up work, not silently mapped to temperature or filled with examples. |
| Precipitation | Already present, with an independently reviewed reporting-day reducer, totals, cumulative comparisons, units, coverage and downloads. |
| ANN temperature prototypes | Not ported. The scripts are not an operational, independently validated prediction pipeline; research models remain behind IDWX's existing evaluation protocol. |

The legacy API returns example coordinates on several invalid-input paths;
IDWX retains explicit unavailable/error states. Its `/historics` Ruby helper
also refers to an undefined average variable. The new analysis implements
specified statistics from validated inputs rather than copying these failures.

## Calendar-day records & averages

Open `explore.html?view=calendar`. The selected start/end years expand to
whole calendar years. Each month/day shows:

- The lowest valid daily minimum and all tied dates.
- The highest valid daily maximum and all tied dates.
- The arithmetic mean of paired valid daily mean temperatures.
- Valid/expected year counts and complete, partial, or no-data status.

February 29 counts leap years only. A missing year contributes to expected
coverage but never contributes a zero or fabricated extreme. A partial sample
can still display its available average and extremes, explicitly labeled;
it may miss more extreme events. An entirely missing sample displays nulls.
Records are within the selected reanalysis period, not official observing
station records or all-time claims. The historical average is a mean of daily
means, not an hourly-weighted average and not the fixed baseline P50.

Chart inspection, table pagination, line/scatter/bar/area styles, Celsius /
Fahrenheit display, URL restoration, and CSV downloads use the same rows.
CSV keeps canonical Celsius, tied dates, coverage and release identity.

## Weather studies

Save the currently applied selection under a study name and chart name,
optionally with notes. Several charts can belong to a study. Unapplied form
edits do not alter the selection being saved. Charts retain their saved
station, source kind, release identity and save time.

Studies use this website's browser local storage (`idwx.weather-studies.v1`).
They do not sync across origins, browsers or devices, and are lost if browser
storage is cleared without a backup. Export/import uses a versioned JSON
selection document, capped at 50 charts and a 1 MB import. Import merges
without removing existing selections or duplicating identical entries.
Names and notes render as text, never markup. Corrupt or unavailable storage
leaves historical exploration usable and preserves existing stored bytes.

Reopening recomputes filters using the page's loaded publication. A different
release is clearly labeled; a missing station is refused. A study backup
contains filters and notes, not archived results. Download each chart's CSV
when a reproducible snapshot of calculated values is needed. Saved charts
open individually; a simultaneous multi-chart study board remains future work.

## Recommended next work, in order

1. **Audit and publish more agricultural variables.** Start with soil
   temperature and volumetric soil moisture by layer, then dewpoint/humidity,
   wind and radiation. These extend the existing growing-year workflow most
   directly. Review source model, coordinates, source hashes, physical units,
   valid ranges and completeness independently for every variable. Define
   daily min/max/mean versus accumulated energy/water before building graphs.
   Wind direction needs circular/vector statistics; a simple arithmetic mean
   is inappropriate. Reanalysis soil moisture is not measured field moisture
   or a soil fertility test. Publish compact allowlisted daily chunks with
   metric-specific statuses before adding matching selectors and charts.
   [ERA5 single-level dataset and variable documentation](https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels)
   provides the model-variable reference; it does not establish the units or
   aggregation of the local Oikolab files, which require their own audit.
2. **Add bounded hourly drill-down.** Port the missing one-month hourly
   chart and same-hour historical average after reviewing the original
   hourly source. Publish only selected, validated hourly fields in monthly
   static chunks; keep raw CSV/SQLite private. Preserve timestamp and reporting
   timezone, distinguish repeated/skipped daylight-saving hours, retain missing
   hours, and require meaningful per-hour historical sample counts. Avoid
   reconstructing hours from daily means or starting a legacy API service.
3. **Compare Idaho locations.** Reuse the current station-aware exporter and
   explorer after reviewing each location's source and coverage. Start with
   a small explicit allowlist and side-by-side growing-season, precipitation,
   and freeze-date comparisons. Show coordinate/elevation and distinguish
   reanalysis grid locations from physical stations. Do not silently expose
   the entire legacy weather database.
4. **Make climate comparisons easier to interpret.** Add selectable fixed
   baselines, complete-period trend plots with uncertainty, and daily
   precipitation departures/cumulative reference bands. Keep analysis years
   separate from the selected reference period and show the sample count.
   Use 1991–2020 as a recognizable reference option, while labeling IDWX
   calculations as reanalysis climatology rather than official NOAA station
   normals. NOAA describes the purpose and construction of its official
   normals in its [Climate Normals documentation](https://www.ncei.noaa.gov/products/land-based-station/us-climate-normals).
5. **Build a simultaneous study board.** Render several saved selections
   together, allow editing/reordering, and add a printable report containing
   methods, coverage and release identity. Keep browser-local storage as the
   first option; shared/private account studies would require a separately
   scoped persistence and authorization design.
6. **Expose the existing evaluation evidence before adding prediction models.**
   Add walk-forward error/coverage comparisons against climatology, interval
   reliability, information cutoffs and skill by season/threshold. A model
   should appear publicly only after the existing validation gates are met.
   Keep that evidence distinct from current provider weather guidance and
   historical distributions. See [the evaluation contract](06_EVALUATION.md)
   and [preseason research protocol](17_PRESEASON_RESEARCH_PROTOCOL.md).

## Verification

Automated checks cover tied records, arithmetic means, invalid/missing rows,
leap-day denominators, calendar normalization, keyboard inspection, canonical
CSV, saved applied filters, safe note rendering, grouped collections, restoring
selections, backup merging and corrupt-file preservation. Additional checks
exercise corrupt local storage, changed release identity and missing stations.
The existing publication exporter/verifier also checks the complete real-data
static site. Native browser visual review is unavailable: the browser access
host reports unavailable. DOM tests do not establish responsive layout, native
file dialogs/downloads, touch or screen-reader behavior.

Verified results for this increment:

- Full frontend suite, Node 24.18.0 / jsdom: **64 passed**.
- Python publication export/verifier suites: **36 passed**.
- Real export and package verification: historical reanalysis, one station,
  94 annual threshold records, unchanged daily-data release
  `2d205c7f80159f444c418a127f6dbd7da274b51d93c3abc9880153e1176f844b`.
- Real-data DOM smoke: 366 calendar rows for 1980–2022, three chart series,
  CSV and 47 static requests, with no provider requests. An independent Python
  calculation matched January 1's 43/43 samples: minimum −19.88 °C
  (2016-01-01), maximum 11.79 °C (1997-01-01), mean daily temperature
  −3.3613275193798446 °C.
- Local package `.tmp/delivery/idaweather-review.zip`, SHA-256
  `86917effbe04f33195bc7904a7da86ecd8a321890f4ea2ec9e6c06442c27532a`.
  This ignored review artifact is not a deployed release.
- `git diff --check` passed. Changes are local and uncommitted on `main`.
