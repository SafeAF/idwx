# Historical explorer: implementation and verification

Implemented locally November 21, 2026, on `main`, based on
`4024f08258631aff742d85b45431efc85ff1f273` plus the current uncommitted changes.
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
- **Compare years:** enter up to three comma-separated years and apply the
  filters. This mode compares full calendars and disables date-range controls.
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
Keyboard arrows inspect timeline values. Downloads retain release identity,
coverage, baseline, effective selection scope, and metric units/policy.
The historical explorer makes no live weather-provider requests.

## Data and implementation

The batch exporter adds schema 1.1.0 annual daily chunks and verified
monthly/yearly/seasonal summaries. SQLite, hourly data, private source metadata,
and research artifacts remain outside the publication. Existing 1.0.0 releases
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
| Full Python suite | 238 passed; 2 optional Nginx tests skipped because Nginx is unavailable |
| Full frontend suite | 47 passed with Node 24.22.1 and the existing jsdom dependency |
| Focused analytics/explorer checks after final adjustments | 5 Python and 13 frontend checks passed |
| Ruff on modified Python files and whitespace checks | Passed |
| Real Twin Falls export, verification, ZIP extraction/reverification | Passed; `historical_reanalysis`, 1 station, 94 annual threshold records |
| Wheel export from extracted package | Passed |
| Wheel installed in a fresh isolated environment with locked dependencies | Export, verification, and packaging passed |
| Loopback static HTTP | Explorer HTML, JS, CSS, daily JSON, and frost CSV returned 200 with expected MIME types; missing/source/database paths returned 404 |
| Default shell plus selected data payload | Approximately 59 KB gzip (sum of individually compressed files), below the proposed 1 MB budget |
| Full archive cached filter calculation and DOM rendering | 67.9 ms on Node/jsdom, AMD Ryzen 7 5700G, with zero additional data requests; not a browser layout benchmark |

The real export uses the existing committed database, unchanged at SHA-256
`cb7f325643d26648aac229ff6d01077c3eee7246675b34a2b328df135a9e40ce`.
It contains data through local date November 7, 2026. The data release ID is
`4896343496264db793a9674c26e004fee1f19f679f2ec06f6a2591221ea7ea3e`.
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

Precipitation analytics remain excluded until source interval/reset semantics
and precipitation-specific coverage pass the review specified in
[the plan](22_DATA_EXPLORER_PLAN.md). Custom baselines, fitted trends,
release-pinned sharing, and research-model comparisons remain later work.
No database refresh, new data collection, production deployment, or host/vhost
changes were performed.
