# Private source-data audit and ingestion safeguards

`idwx data audit` inspects the configured raw CSV directory, recomputes daily
aggregates and frost history in memory, and writes a private JSON report.
It does not modify source files, caches, models or the static website.
It never certifies source provenance or sets publication readiness to true.

## Run the audit

In this Windows checkout:

```powershell
.venv/Scripts/python.exe -m idwx data audit
.venv/Scripts/python.exe -m idwx data audit --config configs/idaho.yml --output .tmp/data-audit.json
```

The default output is `reports_dir/data-audit.json` from the selected config.
The CLI prints a small JSON summary containing status, file/issue counts and
the report path. A completed report has one of two statuses:

| Status | Exit code | Meaning |
| --- | --- | --- |
| `blocked` | 1 | Missing sources, ambiguous mappings, parsing failures, invalid values or incompatible daily caches require correction. A report is still written. |
| `review_required` | 0 | Automatic checks found no blocking issue; manual source and calculation review is still required. |

Invalid configuration or an invalid/unwritable report destination returns a
CLI error rather than claiming to have written a report. Serialization,
temporary-write and final-switch failures preserve the previous report.
JSON uses finite values or null, and the final report is replaced atomically.

Reports include private filesystem paths and source metadata. Keep them
outside raw input and publication directories. The writer rejects paths under
the configured source root, the checkout's `web/`, and detected publication
directories with a manifest or dashboard assets. It cannot anticipate a future
directory being made public: never upload reports alongside the static export.
The default `reports/` and `.tmp/` directories are gitignored.

## What is recorded

The version 1 report records generation time separately from source dates,
`publication_ready: false`, a private-report marker and explicit outstanding
manual review items. It inventories:

- Configured stations, declared source metadata, effective timezone, matching
  filenames, and missing/multiple-source availability states.
- Source filenames relative to the raw root, byte sizes, SHA-256 fingerprints
  and coordinates parsed from filenames.
- Recognized raw headers and parsing diagnostics: row count, UTC range,
  duplicate/invalid timestamps, date-only values, timestamps without an
  explicit zone suffix, missing/invalid temperatures, finite temperature
  range, offset/fallback counts, offset range and timestamp-order violations.
- Distinct UTC timestamps and distinct UTC hours with finite temperatures.
  A diagnostic `complete: false` means processing failed; counts cover only
  the inspected prefix and must not be interpreted as whole-file totals.
- Presence of precipitation and the adapter's mode heuristic, when present.
  This heuristic does not validate accumulation/reset conventions.
- The same coverage-aware annual frost records and historical summaries used
  by the exporter, recomputed from current raw-source aggregation. These are
  review material, not published results or forecasts. They include provisional
  candidates, event/interval status, valid/missing days and exact thresholds.
- Event status counts, years with no recorded local days, contiguous missing
  daily-minimum ranges over the configured year span, and July-freeze records
  by year/threshold. Entirely missing years and years containing invalid minima
  remain distinguishable. An empty July-freeze list does not establish complete
  July coverage; review the missing-minimum ranges alongside it.

The audit always reads raw files; it does not substitute existing target,
feature or dataset caches. Source size/mtime changes during a file's audit
block that file's historical review. Use stable source snapshots and one batch
writer; this check is not a filesystem snapshot or a data-vintage certificate.

### Daily cache comparison

If a daily Parquet cache already exists, the audit compares its current
aggregation columns with a fresh raw-source aggregation, after normalizing
column types/order and local date representation. Current numeric values,
coverage counts and quality flags must match exactly; optional extra cached
columns are not part of this comparison. Results are:

- `missing`: no daily cache exists yet; this is informational during source
  auditing, not a reason to invent one or fabricate results.
- `matches_current_aggregation`: current aggregation values match. This does
  not prove source identity, historical availability or correct declared units.
- `mismatch` / `unreadable`: blocked; preserve and investigate old artifacts,
  then rebuild and re-audit before using derived results.

## Supported source adapter and stricter failures

The implemented adapter is the existing ERA5-style **hourly Celsius** adapter.
It uses source coordinates embedded in the filename, such as
`42.5629---114.4609-hourly.csv`, and requires the exact
`temperature (degC)` header. The first column supplies timestamps. Optional
recognized columns include `utc_offset (hrs)`, model elevation, coordinates
and precipitation in the adapter's declared units.

This is not a general daily/NOAA/Fahrenheit/Kelvin adapter. A degC column label
is not proof that its values have that unit. Date-only timestamp rows are
flagged as unsupported daily input by the audit. Other schemas need a reviewed
adapter or explicit normalization; no automatic unit guessing occurs.

Ingestion now rejects empty/duplicate headers, invalid/missing UTC timestamps,
duplicate UTC instants across chunks, and malformed/nonfinite explicit offsets
or offset magnitudes of 24 hours or more. Missing offsets still fall back to
the station IANA timezone, then the global timezone. Zone-free timestamps are
interpreted as UTC by the adapter, an assumption that requires source review.
The offset magnitude bound is an application input rule, not verification of
the correct station reporting convention.

Missing/nonfinite temperatures remain unavailable to aggregation. The audit
distinguishes missing values from nonmissing invalid values and blocks the
latter for review. Daily minima still require the configured number of
distinct UTC hours, not subhourly reading count. This criterion does not prove
that the true daily minimum was captured.

Coordinate matching retains the existing strict tolerance of 0.0001 degrees.
Multiple configured matches now fail instead of choosing the first. Configured
station IDs must be unique lowercase snake_case for auditing. Unmatched source
coordinates are flagged; historical review is not assigned to an invented
configured station.

## Preventing cache overwrite and partial input refreshes

The cache builder now checks every source-to-station mapping before writing.
**Multiple CSV files for one station fail explicitly**, including with
`--no-rebuild`; the builder no longer skips or overwrites a second source
silently. The audit inventories/fingerprints those files but does not pretend
that one file or a join of daily aggregates represents the combined source.

Multiple parts must first be consolidated at the hourly-record level under
a reviewed process with duplicate checks and consistent units/reporting-day
rules. Daily aggregation must happen after consolidation: separate files can
split a station's local reporting day. That consolidation adapter/workflow is
not implemented by this change.

For a valid single-file-per-station build, all requested inputs are parsed and
serialized in a private staging directory before any existing daily cache is
replaced. A parsing or serialization failure preserves every old station cache.
Successful replacements are atomic **per file**. A filesystem replacement
failure or process crash during the final multi-file commit can still leave
a mixed private cache set; this is not an all-station transaction. Stop on
failure, rerun the build and audit, and do not export an incomplete refresh.
The public exporter's separate immutable-release/manifest guarantee is unchanged.

Existing daily files are skipped when rebuilding is not requested. This does
not validate changed raw contents. Run the audit, which reads sources even
when caches exist, and use `--rebuild` deliberately after correcting inputs.
Missing/empty raw roots fail cache builds rather than reporting an empty
successful update.

## Manual review and workflow

Before publishing historical observations, retain evidence for actual provider/
product, station identity, retrieval and revision history, unit conventions,
timestamps and local reporting days, and the chosen coverage/window policy.
Hand-check representative station-years including gaps, boundary events, no
events, July freezes and both thresholds. The configured `source: NOAA` value
is a declaration; this tool does not independently verify it.

A source-review workflow, once real files exist:

```powershell
.venv/Scripts/python.exe -m idwx data audit --output reports/source-before-build.json
# Resolve blockers and manually review source evidence before continuing.
.venv/Scripts/python.exe -m idwx data build --rebuild
.venv/Scripts/python.exe -m idwx data audit --output reports/source-after-build.json
# Review cache agreement and hand-checked historical results before export.
```

Run commands as separate reviewed steps and stop on nonzero exit; an exit code
of zero still requires manual review. No signoff mechanism, scheduler,
source download, public publication gate or forecast publication is added.
Forecast validation additionally requires historical data-vintage evidence,
clean chronological holdouts, calibrated interval assessment and agreed
promotion criteria from the forecast audit.

## Workspace result and verification

The Twin Falls source is now available at the sibling checkout coordinate
`42.4634, -114.4975` and has been reproducibly reduced to the checked-in daily
SQLite database. The one-source builder validates the configured coordinate,
stores the raw filename/size/SHA-256, detects concurrent source changes, runs
SQLite integrity checking, and replaces the database atomically. See
`data/README.md` for the recorded source and database hashes. This bounded
build evidence is not a substitute for the manual provider, retrieval-history,
unit, reporting-day, and representative-year review described above.

Thirty-two new synthetic regression cases cover missing/empty roots, private
report output, raw fingerprints, fresh coverage/frost results, missing-year
ranges, July freezes, stale/corrupt caches, invalid input values and headers,
ambiguous/multiple/unconfigured stations, source changes, timezone/offset and
ordering diagnostics, cache staging preservation, report failure cleanup,
publication-path rejection and CLI exit/default-output behavior.

The current regression suite additionally validates the checked-in database,
configuration drift rejection, and real publication path. Browser rendering,
production TLS, manual source review, and real-data model validation remain
separate outstanding work. No forecasts are published.
