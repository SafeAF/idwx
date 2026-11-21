# Static historical data contract, version 1.1.0

This implements historical summaries from Phase 2 and the export contract of
Phase 3 in `12_FROST_STATIC_WEBSITE_PLAN.md`. It does not publish model
prediction records. The browser derives a clearly labeled planning-year
climatology projection from the historical summaries. The exporter now also
installs the dashboard assets; see
`15_DASHBOARD_DEPLOYMENT.md` for preview, deployment, and verification.

## Build

```powershell
uv sync
uv run idwx export-web --output web
```

The default configuration reads the checked-in Twin Falls daily SQLite
database. Configurations without `database_path` retain the earlier Parquet
cache path. The exporter recalculates frost records from daily inputs,
bypassing potentially stale target caches. Only configured stations are
exported; a missing or empty station fails the entire export. No data is
downloaded or fabricated as a fallback.

Options:

- `--config`: main YAML configuration, default `configs/idaho.yml`.
- `--output`: publication root, default `web`.
- `--baseline-start` and `--baseline-end`: inclusive years, supplied together
  and within the exported season-year range. By default the start follows the
  configured backtest start while the end advances through the latest complete
  observed calendar year. The model backtest cutoff does not suppress newer
  historical records from the website.
- `--min-history-years`: positive confirmed-year minimum for displaying
  quantiles, default 10. This is a display policy, not an accuracy guarantee.
- `--demo`: explicitly synthetic 2000-2022 examples, with partial coverage,
  a no-event year, and a July freeze. Ignores private weather caches. Existing
  configuration still controls stations, thresholds, windows, and export years.

```powershell
uv run idwx export-web --demo --output .tmp/demo-web
```

Every JSON document and CSV record identifies demo data as `synthetic_demo`.
The checked-in ERA5/Oikolab database exports as `historical_reanalysis`;
legacy observation-cache exports use `historical_observations`. These values
identify the input class, not an independent certification of provenance. The
generated timestamp describes export time, never observation freshness.

## Release layout

```text
<output>/
  index.html
  css/styles.css
  js/app.js
  js/charts.js
  js/format.js
  data/manifest.json
  data/<release-id>/stations.json
  data/<release-id>/stations/<station-id>.json
  data/<release-id>/downloads/frost-history.csv
```

All URLs in the contract are relative to the publication root, including URLs
inside station-index documents. Resolve them against the site's root/base
directory, not against the directory containing the index JSON. Load the
manifest once, then keep its release references for the lifetime of a view.
Load the index and selected station only. Fetch a newer manifest to start a
new release view; do not combine files across releases.

Station IDs must be unique lowercase snake_case identifiers. Raw hourly observations,
private filesystem paths, feature tables, and model files are not exported.
CSV uses UTF-8, a header row, and blank cells for missing values. JSON uses
finite numbers and `null`; dates are ISO date-only strings (`YYYY-MM-DD`).
Treat dates as calendar values in browsers, avoiding conversion through local
timezones. Temperatures are always Celsius; -2 C is exactly 28.4 F.

## Common fields and manifest

All JSON documents carry `schema_version`, `release_id`, `data_kind`, and
`methodology`. The manifest additionally provides:

| Field | Meaning |
| --- | --- |
| `generated_at_utc` | Timezone-aware ISO export timestamp. |
| `stations_url` | Station-index URL within the pinned release. |
| `history_csv_url` | Combined results CSV URL within the pinned release. |
| `source_coverage` | Earliest/latest source record dates across configured stations. |
| `files` | Mapping from publication-relative URLs to SHA-256 file checksums. |

The station index contains `stations`, an array of station metadata,
`source_coverage`, and a station-document `url` per entry.

## Station document

| Field | Meaning |
| --- | --- |
| `station` | ID, name, optional latitude/longitude/elevation/source, effective timezone. Missing or nonfinite optional numeric metadata is null. |
| `source_coverage` | First/last recorded day, first/last valid-minimum day, recorded-day and valid-minimum-day counts. Covers the whole cache, which can exceed export years. |
| `season_years` | Inclusive exported year bounds. Entirely absent years are retained. |
| `windows` | Effective spring/fall month-day boundaries after station overrides. |
| `thresholds_c` | Configured, unique, finite Celsius thresholds. |
| `annual` | One record per year and threshold. |
| `summaries` | One pair of historical event distributions per threshold. |

An annual record includes `season_year`, `threshold_c`, `last_spring_frost`,
`first_fall_frost`, and `interval`. Each event object has:

- `status`: `observed`, `no_event_in_complete_window`, `incomplete_window`,
  or `insufficient_data`.
- `date` and `tmin_c`: confirmed event values only; otherwise null.
- `candidate_date` and `candidate_tmin_c`: provisional first/last qualifying
  observation, even if coverage prevents confirmation.
- `window_start`, `window_end`, `expected_days`, `valid_days`, `missing_days`,
  and `coverage_fraction`.

Interval fields are `date_difference_days`, `frost_free_days`, `status`, and
`summer_freeze_observed`. The first count preserves the date difference between
confirmed events. The second counts strictly intervening days only when every
day is valid and above the threshold. A false summer-freeze flag means no July
freeze was found; it is not a coverage guarantee. Detailed status meanings
are in `13_FROST_CALCULATIONS.md`.

The CSV flattens these same annual records. Event/interval fields are prefixed
with their object name; rows also carry `schema_version`, `release_id`,
`data_kind`, and `station_id`. It contains published results, not daily inputs.

## Historical summaries

Each threshold's summary has separate spring and fall distributions. Only
confirmed `observed` events inside the baseline period contribute. No-event
and incomplete years are counted in `status_counts`, but excluded from date
statistics. These are distributions conditional on a confirmed event, not
estimates of the probability that an event occurs in a year.

Fields include `kind: historical_distribution`, `baseline_period`,
`valid_year_count`, `valid_years`, `status_counts`, `minimum_valid_years`,
`earliest`, `latest`, `p10`, `p50`, and `p90`. A point is an object with
`calendar_day` and `month_day`; empty statistics are null. Below the minimum
valid-year count, status is `insufficient_history` and all three quantiles are
null. Earliest/latest remain available whenever at least one event exists.

All month/day values map onto leap year **2000**, with February 29 included.
March 1 is day 61 regardless of the original observation's year. This avoids
artificial day shifts in leap/non-leap comparisons. Quantiles use NumPy's
`inverted_cdf` method (nearest rank), so each point corresponds to an actual
observed month/day rather than an interpolated fractional date. `p50` is the
typical historical date. Spring P90 describes a later last freeze; fall P10
describes an earlier first freeze. These are not forecast uncertainty bounds
or guarantees for an individual season.

## Planning-year climatology projection

The dashboard uses the station's IANA timezone and the latest archived season
to select a planning year. It uses the current local year while the archive is
older; once the archive contains the current season, it advances to the next
year. Thus the Twin Falls archive containing partial 2026 records presents a
2027 projection. For each threshold, it presents historical P50 as the
projected spring and fall date and historical P10–P90 as the accompanying
range. When both medians exist, it also shows the strictly intervening
calendar-day count.

This is a browser-derived historical baseline. It does not add forecast fields
to JSON/CSV, use observations after the archive cutoff, ingest current weather,
or run the research model pipeline. If either distribution has insufficient
confirmed history, that event and the derived span display as unavailable.

## Publication and rollback

The release ID is a full SHA-256 digest of canonical public content, including
the schema version, methodology, configuration-derived values, and station
results. Timestamps are excluded from the content identity. Repeated exports
of identical inputs reuse the release without rewriting it. Changed content
creates a new release. Contract or serialization changes must bump the schema
version so older immutable paths retain their meaning.

All inputs are processed before publication. A complete release is written
in a temporary directory on the destination filesystem and renamed into place.
Only then is the manifest replaced atomically. Failed refreshes preserve the
previous manifest; a failed final switch may leave a complete unreferenced
release. Existing release bytes and file inventory must match exactly before
reuse. Previous releases are retained automatically for open views and rollback.
The exporter is intended for one batch writer at a time. Frontend assets are
copied from a fixed allowlist (also bundled into installed Python wheels),
with atomic replacement per file. For application-code updates, export into
a separate staging directory and switch the server document root only after
verification. The data manifest switch does not make a multi-file asset
update atomic.

This is a local filesystem publication guarantee. Deployment to Nginx must
likewise upload complete versioned data before switching the remote manifest.
Do not expose staging directories through the production server. Keep old
release directories during deployment. To roll back, atomically restore the
previous manifest with its matching retained release files. Scheduling,
retention cleanup, server configuration, and HTTP verification remain Phase 5.

## Verification

Run `uv run pytest -q`. Tests verify calendar-normalized quantiles, baseline
filtering, minimum history, null handling, public field selection, CSV/JSON
agreement, release checksums and references, reproducibility, station safety,
synthetic labels, SQLite/legacy-cache inputs, CLI options, and failure
preservation of the active manifest. The checked-in Twin Falls database also
has a bounded-content regression and is exercised by the real export in CI.

## Independent publication verification and portable delivery

Run `uv run idwx verify-web PUBLICATION_DIRECTORY` before activation. This
read-only check accepts schemas 1.0.0 and 1.1.0, verifies the active and every retained
release, canonical content identity, checksums, path/file/field allowlists,
finite JSON values, event and interval consistency, recalculated historical
summaries, and CSV/JSON agreement. Unexpected files or directories, private
fields, model forecast records, symlinks within the publication, and synthetic
releases are rejected. The verifier reports the allowlisted browser-derived
climatology projection separately from `forecasts_included: false`. A
deployment root symlink itself is supported. A retained
release must satisfy the same contract as the active release.

`valid_static_package` means the package is internally consistent. It does
not certify source units, reporting days, provenance, or forecast validity;
complete the private source audit separately. Synthetic examples require
explicit `--allow-demo`, including when a synthetic release is retained under
an otherwise historical publication. Keep demo and real publication roots
separate.

Create an archive with:

```powershell
uv run idwx package-web PUBLICATION_DIRECTORY --output delivery/site.zip
```

The command verifies before packaging, extracts and verifies its own archive,
then atomically replaces the destination ZIP. It prints a SHA-256 digest.
Only public allowlisted files are archived, with stable ZIP timestamps;
identical input bytes produce identical archive bytes. Export timestamps can
change between exports even when the data release ID is unchanged. Choose
an archive destination outside the publication root. For demo delivery only,
add `--allow-demo` to both commands. Never deploy the repository as a site.

## Historical explorer extension (1.1.0)

New exports include `explore.html`, `js/analytics.js`, `js/explorer.js`, and
`js/explorer-charts.js`. The homepage still loads no daily history. The explorer
uses the same manifest and release identity, and fetches only the chosen
station and selected years. It needs no weather-provider requests.

A station document adds `analytics_url`, pointing to
`data/<release-id>/analytics/<station-id>/index.json`. Like existing URLs,
resolve it against the publication root. The index carries the common fields,
`station_id`, ordered `years` entries (`year`, publication-relative `url`),
and `summary_url`. The years cover every source calendar year, including
partial boundary years and completely missing intervening years.

Each `daily/<year>.json` has common fields, `station_id`, `year`, and `days`:

| Daily row field | Contract |
| --- | --- |
| `date` | Local ISO calendar date; full ordered 365/366-day calendar without duplicates. |
| `tmin_c`, `tmax_c`, `tmean_c` | Finite Celsius or null; derived daily values, never raw hourly source records. |
| `status` | `valid` when all three values exist; `invalid_temperature` for a recorded day with any unavailable temperature; `missing_record` for an absent source date. |
| `n_obs` | Nonnegative source-hour count where available, otherwise null. It is not an independent completeness guarantee. |

An insufficient-coverage flag masks all temperatures. Other recorded days
retain individual finite fields even when one field is missing. Statistics
count valid values separately per metric. Missing calendar rows carry only
null values and a `missing_record` status, including outside the source span
in boundary years. Internal quality-flag text and private source metadata are
not published. Precipitation remains excluded pending its separate review.

`summary.json` includes common fields, `station_id`, `months`, `years`,
`policy`, and `seasons`. Monthly/yearly temperature records include the period,
`expected_days`, per-field `valid_days`/`mean`/`min`/`max`, and `complete`.
Means describe available values; incomplete periods cannot provide complete
period means in long-range charts or rankings. No values means null statistics.

The policy records the configured GDD base/cap/start/end, winter HDD base,
and cold-day threshold in Celsius. Public winter metrics use December–February,
assigned to the ending year; a configuration with different winter months is
rejected for this export. Seasonal records contain final `gdd`, a continuous
`gdd_subtotal` and `through` date, GDD valid/expected counts, winter `hdd`,
`cold_days`, `cold_run`, and winter valid/expected counts. Final totals and cold
runs require full metric coverage. The continuous GDD subtotal stops before
its first missing day. These calculations are separate from research targets.

The existing `temperature_climatology` station field contains a fixed
`baseline_period`, explanatory `method`, and calendar-day `days`. Each day has
`month_day`, `sample_count`, and `low_c`/`mean_c`/`high_c` P10/P50/P90 objects.
It uses paired valid temperatures, a minimum of five samples per month/day,
and linearly interpolated quantiles. It differs from the nearest-rank frost
quantiles. Explorer anomalies are departures from this historical median;
whole-month anomaly values require every selected calendar day and reference.

All analytics files are checksummed. The 1.1.0 content identity hashes the
existing common identity and station contents (excluding the derived
`analytics_url`), plus an `analytics` array in station order. Each bundle has
`station_id`, a year-string-to-daily-rows `daily` mapping, and `summary` without
common identity fields. Release-specific URLs and timestamps are derived
after hashing. The verifier rebuilds this identity, validates daily calendars
and source counts, and recomputes monthly/annual/seasonal summaries.

Version 1.0.0 releases retain their original identity and field contract.
Both versions may be active or retained together; old bytes are never
rewritten. The older release supports the existing homepage, while the new
explorer requires 1.1.0 data and shows an explicit error on an older manifest.

Browser CSVs include release/schema/data kind, station, actual applied filters,
reference baseline, selected display unit, and canonical Celsius values.
Daily downloads include display values; frost downloads retain all annual
coverage/candidate fields. Long-range temperature downloads match their
monthly/yearly table; the daily table view exports individual dates. Comparison
mode uses full selected calendar years. Frost mode uses full seasonal windows
for the inclusive selected years. Other views use inclusive local dates.


The graph builder adds browser-only yearly statistics, same-month comparisons
across years, and monthly profiles. These reuse the existing daily chunks;
they add no release files or schema fields. Statistics select arithmetic mean,
minimum, or maximum of the chosen daily temperature field and require all
expected calendar days. Same-month comparisons use each year's actual month
length; missing leap days invalidate February instead of being imputed.
CSV metadata includes `statistic`, `selected_month`, `chart_style`, and an
explicit `selection_scope` (`full_calendar_years`,
`selected_month_across_years`, or `full_profile_year` for these views).
