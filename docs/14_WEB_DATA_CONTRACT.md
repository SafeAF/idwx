# Static historical data contract, version 1.0.0

This implements historical summaries from Phase 2 and the export contract of
Phase 3 in `12_FROST_STATIC_WEBSITE_PLAN.md`. It does not publish model
predictions. The exporter now also installs the dashboard assets; see
`15_DASHBOARD_DEPLOYMENT.md` for preview, deployment, and verification.

## Build

```powershell
uv sync
uv run idwx data build --rebuild
uv run idwx export-web --output web
```

Rebuild daily caches produced before the coverage corrections. The exporter
recalculates frost records from daily caches, bypassing potentially stale
target caches. Only configured stations are exported; a missing or empty
station cache fails the entire export. No data is downloaded or fabricated
as a fallback. Actual source provenance, units, and reporting conventions
still require the real-data audit described in `13_FROST_CALCULATIONS.md`.

Options:

- `--config`: main YAML configuration, default `configs/idaho.yml`.
- `--output`: publication root, default `web`.
- `--baseline-start` and `--baseline-end`: inclusive years, supplied together
  and within the exported season-year range. Defaults to the configured
  backtest years, or observed years if configuration omits those bounds.
- `--min-history-years`: positive confirmed-year minimum for displaying
  quantiles, default 10. This is a display policy, not an accuracy guarantee.
- `--demo`: explicitly synthetic 2000-2022 examples, with partial coverage,
  a no-event year, and a July freeze. Ignores private weather caches. Existing
  configuration still controls stations, thresholds, windows, and export years.

```powershell
uv run idwx export-web --demo --output .tmp/demo-web
```

Every JSON document and CSV record identifies demo data as `synthetic_demo`.
Real-cache exports use `historical_observations`; this identifies the input
mode, not an independent certification of station observations or provenance.
The generated timestamp describes export time, never observation freshness.

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

Station IDs must be unique lowercase snake_case identifiers. Raw observations,
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
synthetic labels, stale-cache bypass, CLI options, and failure preservation
of the active manifest. These checks use synthetic data because the configured
real source directory remains unavailable.

## Independent publication verification and portable delivery

Run `uv run idwx verify-web PUBLICATION_DIRECTORY` before activation. This
read-only check accepts schema 1.0.0, verifies the active and every retained
release, canonical content identity, checksums, path/file/field allowlists,
finite JSON values, event and interval consistency, recalculated historical
summaries, and CSV/JSON agreement. Unexpected files or directories, private
fields, forecasts, symlinks within the publication, and synthetic releases
are rejected. A deployment root symlink itself is supported. A retained
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
