# Precipitation and growing-year assessment

Implemented locally November 21, 2026, on `main` from base `2d2203e`.
This adds historical climate exploration; no new collection, database change,
model training, production deployment, or soil-fertility prediction occurred.

## Views

The homepage links directly to **Total precipitation** and **Growing-year
assessment** in the existing explorer.

Precipitation supports daily, complete monthly/yearly totals, up to three
calendar years overlaid, continuous year-to-date curves, mm/in display,
coverage, wet-day counts (at least 1 mm), tables, and matching CSV downloads.
Missing values remain missing; cumulative curves stop at the first missing
reporting day. Canonical CSV amounts are mm, with display units recorded.

The growing-year view compares historical years against editable minimum
GDD and verified frost-free-day requirements, plus an optional minimum
precipitation requirement. Initial 1,000 GDD and 100-day values are illustrative
inputs, not crop recommendations. The configured GDD base/cap/window and
selected frost threshold are displayed. Seasonal precipitation uses the GDD
window and is total water equivalent, not effective crop water or irrigation.

Each requirement is met, shortfall, unknown, or not selected. A known shortfall
is reported even if another metric is unknown. With no shortfall, unavailable
required data produces insufficient data. Only all required checks met produces
a passing result. Missing or unconfirmed frost events cannot supply a verified
interval. There is no combined fertility score or forecast of crop success.

Soil fertility remains **Not assessed**. Site-specific nutrient, organic
matter, and pH measurements require soil testing; weather is not a substitute.
See [University of Minnesota Extension soil testing guidance](https://extension.umn.edu/garden-and-home/yard-and-garden/gardening-in-minnesota/soil-testing-for-lawns-and-gardens).

## Source review and reporting day

The SQLite precipitation column was not reused because its original quality
flags depended on temperature and aggregation was selected heuristically.
Instead, a compact derived daily supplement is checked in at
`data/precipitation/twin_falls.json`, bound to the original source fingerprint
in the configured database. See [input provenance and rebuild](../data/README.md).

The original source contains 376,944 consecutive hourly timestamps, ERA5 model,
Twin Falls coordinate `(42.4634, -114.4975)`, fixed offset −7, and finite
nonnegative total-precipitation amounts. Source SHA-256:
`1a439c26f6426236e943b22221a1246da8d9b5fc438d145f00b0971536311a8e`.

[Oikolab's parameter documentation](https://docs.oikolab.com/parameters/)
defines total precipitation in mm, including rain and snow-water equivalent.
[ECMWF's ERA5 accumulation conversion guidance](https://confluence.ecmwf.int/spaces/CKB/pages/197702790/Conversion%2Btable%2Bfor%2Baccumulated%2Bvariables%2Btotal%2Bprecipitation%2Bfluxes)
describes hourly increments ending at their timestamp. These are not the
ERA5-Land cumulative values requiring differencing. The reducer assigns each
amount to `timestamp − 1 hour − 7 hours`, then sums its reporting day.
The fixed UTC−07 convention is explicit; it does not switch for daylight saving.
It requires 24 distinct, valid precipitation hours per day independently of
temperature. Genuine zero is valid, and partial days expose no total.

Complete reporting days run from **1980-01-01 through 2022-12-30**:
15,705 complete days. The source's first boundary day, 1979-12-31, has 8 hours;
2022-12-31 has 16 hours. Both totals are null. Hourly refresh bytes are no
longer present locally, so the refresh boundary and 2023–2026 precipitation
have not been audited. They remain `not_reviewed`, with null amounts/counts.
Temperature/frost history retains its existing later coverage through 2026.
No request was made to a weather provider.

An independent Python `csv`/`datetime` calculation, separate from the pandas
reducer, reproduced these reference months from the original source:

| Reporting month | Hours | Total water equivalent (mm) |
| --- | ---: | ---: |
| 1980-01 | 744 | 71.18 |
| 2020-02 | 696 | 20.70 |
| 2022-11 | 720 | 46.59 |

Tests retain these fixed reference totals, leap-February completeness,
hour-ending midnight assignment, dry zeros, missing hours, duplicate hours,
independent precipitation coverage, cumulative gaps, unit conversion, and
met/shortfall/unknown assessment behavior. The publication verifier recomputes
coverage and summaries, checks release identities, and rejects false complete
days even when their manifest checksum has been updated.

## Local verification

- Full Python suite: 244 passed, 2 optional Nginx tests skipped.
- Full frontend suite: 58 passed with Node 24.22.1 and jsdom.
- Real export: schema 1.2.0, historical reanalysis, release
  `2d205c7f80159f444c418a127f6dbd7da274b51d93c3abc9880153e1176f844b`.
- The existing SQLite bytes remain unchanged, SHA-256
  `cb7f325643d26648aac229ff6d01077c3eee7246675b34a2b328df135a9e40ce`.

Native visual browser review remains unavailable because the browser access
host is unavailable. DOM tests do not establish browser layout, native
CSV downloads, touch, screen-reader behavior, TLS, or production caching.
