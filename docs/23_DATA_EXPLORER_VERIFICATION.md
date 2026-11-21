# Historical explorer: implementation and verification

Implemented locally November 21, 2026, on `main`, based on
`2d2203e` plus the current uncommitted precipitation and growing-year changes.
This is implementation and local verification evidence, not production
activation or visual browser acceptance.

## Using the explorer

New static exports contain `explore.html`, linked as **Explore data** from
the homepage. To preview the real archive from this checkout:

```sh
.venv/bin/idwx export-web --output .tmp/explorer-real
.venv/bin/idwx verify-web .tmp/explorer-real
.venv/bin/python -m http.server 8765 --bind 127.0.0.1 --directory .tmp/explorer-real
```

Open `http://127.0.0.1:8765/explore.html` and stop the server when finished.
The directory already contains the locally generated real-data export.

- **Temperature timeline:** inspect daily minimum, mean, and maximum with a
  fixed historical reference. Choose a shorter period for daily detail;
  longer periods use complete monthly/yearly means with matching tables.
- **Compare years:** choose up to three years from the dropdowns and apply the
  filters. This mode compares full calendars and disables date-range controls.
- **Yearly statistics:** compare complete calendar years with arithmetic mean,
  minimum, or maximum of the chosen daily temperature field.
- **Same month across years:** select a month and a year range, then graph
  its temperature statistics. A leap February requires 29 valid days.
- **Monthly profile of one year:** choose one year and compare its 12 monthly
  statistics. Missing months remain unavailable.
- **Chart styles:** choose line, scatter, bar, or area for temperature graphs.
  Styling changes preserve the underlying table and CSV values.
- **Frost history:** inspect full seasonal windows in the inclusive selected
  years, status counts, verified frost-free intervals, and spring/fall
  event-date distributions. Missing/no-event seasons never become date zero.
- **Monthly anomalies:** inspect mean daily departures from the published
  calendar-day median. Select a month to open its daily temperatures. Missing
  paired coverage leaves a month unavailable.
- **Growing season & winter:** inspect final GDD, continuous partial GDD,
  winter HDD, cold-day counts, and maximum cold runs. Policy bases/windows and
  units are explicit. Include the preceding December for a complete winter;
  the selected dates bound the calculation inputs.
- **Daily data table:** inspect paginated daily records and download the
  full selected rows, including missing values and canonical Celsius values.

Apply filters to update the view and URL. Back/forward restores selections.
The homepage includes a prominent **Graphs & year comparisons** section with
direct links, plus navigation in the header. Keyboard arrows inspect chart values. Downloads retain release identity,
coverage, baseline, effective selection scope, and metric units/policy.
The historical explorer makes no live weather-provider requests.

## Data and implementation

The batch exporter adds schema 1.2.0 annual daily chunks and verified
monthly/yearly/seasonal summaries. SQLite, hourly data, private source metadata,
and research artifacts remain outside the publication. Existing 1.0.0 and 1.1.0 releases
can remain active or retained alongside new releases without rewriting bytes.
See [the data contract](14_WEB_DATA_CONTRACT.md) for exact fields and hashing.

Public temperature/seasonal calculations live in `idwx/analytics.py`; they
reindex absent days and require complete periods for final totals. They do
not reuse the research pipeline's permissive partial-season sums. Explicit
left-to-right floating-point accumulation keeps Python/browser calculations
consistent across Python versions. No models are trained or published.

Missing or mixed-release chunks prevent a partial view. Obsolete requests
cannot overwrite newer selections, retries clear rejected cached responses,
and the chart/table/CSV share the applied selection. The homepage fetches no
new daily chunks. New assets are included in both checkout and wheel exports.

## Checks completed

| Check | Result |
| --- | --- |
| Full Python suite | 244 passed; 2 optional Nginx tests skipped because Nginx is unavailable |
| Full frontend suite | 58 passed with Node 24.22.1 and the existing jsdom dependency |
| Ruff on modified Python files and whitespace checks | Passed |
| Real Twin Falls export, verification, ZIP extraction/reverification | Passed; `historical_reanalysis`, 1 station, 94 annual threshold records |
| Wheel export from extracted package | Passed |
| Wheel installed in a fresh isolated environment with locked dependencies | Export, verification, and packaging passed |
| Loopback static HTTP | Explorer HTML, JS, CSS, daily JSON, and frost CSV returned 200 with expected MIME types; missing/source/database paths returned 404 |
| Prior 1.1.0 default shell plus selected data payload | Approximately 62 KB gzip (sum of individually compressed files), below the proposed 1 MB budget |
| Prior 1.1.0 full archive cached filter calculation and DOM rendering | 97.9 ms on Node/jsdom, AMD Ryzen 7 5700G, with zero additional data requests; not a browser layout benchmark |

The real export uses the existing committed database, unchanged at SHA-256
`cb7f325643d26648aac229ff6d01077c3eee7246675b34a2b328df135a9e40ce`.
It contains data through local date November 7, 2026. The data release ID is
`2d205c7f80159f444c418a127f6dbd7da274b51d93c3abc9880153e1176f844b`.
Precipitation has its own reviewed cutoff, December 30, 2022.
The ignored delivery directory `.tmp/explorer-delivery/` contains the verified
real-data ZIP and wheel; the ZIP checksum is recorded in `SHA256SUMS.txt`
there. Export timestamps and frontend changes can change ZIP bytes without
changing the historical data release ID.

## Remaining boundaries

The browser tool reported **“The browser access host is unavailable.”**
Actual desktop/mobile visual review, touch behavior, native downloads, and
screen-reader behavior remain unverified. DOM checks do not replace those
checks. Loopback Python HTTP checks do not establish Nginx caching, TLS, or
production behavior; the existing optional Nginx suite was skipped.

Precipitation analytics use only the independently reviewed original hourly
archive; refresh-period precipitation remains unavailable. See the
[source audit and growing-year methods](24_PRECIPITATION_GROWING_ASSESSMENT.md).
Custom baselines, fitted trends,
release-pinned sharing, and research-model comparisons remain later work.
No database refresh, new data collection, production deployment, or host/vhost
changes were performed.

## IdaWeather graph-builder follow-up

The local reference at `~/source/IdaWeather/` was inspected read-only. Its
`graphs/_form.html.erb` and study graph rendering informed the yearly,
same-month-across-years, monthly-profile, mean/min/max, and chart-style
controls. IDWX retains its static publication and validated daily temperature
inputs. No legacy API calls, fallback example coordinates, hourly data,
additional variables, accounts, or study database were introduced.

The additional graph-builder tests independently check known arithmetic
means/extremes, leap-month completeness, year dropdowns, CSV scope, identical
tables across chart styles, and homepage links. That graph-builder increment passed 52 frontend checks; the current
suite passes 58. Browser access remains unavailable; the live local preview
is refreshed for owner review.


## Precipitation and growing-year follow-up

The real schema 1.2.0 publication passed verification, ZIP extraction and
reverification, retaining the prior immutable 1.1.0 release. Both extracted-wheel
and isolated installed-wheel exports, verification and packaging passed. The
new module is present in each package. Ruff and whitespace checks passed.

A real-data Node/jsdom smoke check rendered the default precipitation view
as 12 months of 2021, 365/365 valid reporting days, 331.86 mm total, and 70
wet days. The growing-year view rendered 46 years (1980–2025), both charts,
per-requirement results and the explicit soil-fertility limitation. Both views
requested only local static publication data, with no weather-provider calls.
The HTTP preview returned 200 for the refreshed explorer. These are DOM/HTTP
checks, not native browser visual review. See the local artifact receipt in
`.tmp/explorer-delivery/verification-receipt.json` for current full digests.
