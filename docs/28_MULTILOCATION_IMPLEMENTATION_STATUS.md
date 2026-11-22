# Climate workspace implementation status

November 21, 2026 (America/Denver). Base: `main` at
`086489e7dc990a761ca3b64fb651e97fb7ea17f0`.

This is the first implementation increment specified in
[the seven-phase plan](27_MULTILOCATION_ANALYTICS_IMPLEMENTATION_PLAN.md#8-implementation-sequence-and-acceptance-gates).
It implements the full-file source/metric audit and the independent graph-card
foundation against existing published data. It does **not** complete the entire
plan. No production host, collection job, raw source, SQLite database, research
model or existing release contents were changed.

## Implemented

- `configs/climate_sources.yml` explicitly pins all 21 CSV filenames, full
  SHA-256 fingerprints, coordinates, ERA5 identity and fixed reporting offsets.
  Twin Falls retains its existing identity; other locations use coordinate
  labels rather than unverified place names. The raw root is supplied explicitly
  to the command, not assumed from the research config.
- `idwx/climate_metrics.py` defines all 27 parameters, exact column/unit
  contracts, physical quantities, display conversion, ERA5 soil depths,
  reducer allowlists, date bases and admission reasons. It prohibits arithmetic
  direction means, snow-state sums and unverified radiation-energy integration.
  Definitions reference [Oikolab's parameter documentation](https://docs.oikolab.com/parameters/).
- `idwx data audit-climate` streams every allowlisted file in 50,000-row chunks.
  It checks pinned fingerprints before and after reading; exact headers;
  ordered, unique whole UTC hours; expected endpoints and continuity; coordinate,
  model, elevation and reporting-offset consistency; and every parameter's valid,
  missing and out-of-range counts and extrema. Invalid examples are bounded to
  five per metric/file. Missing/unlisted files, duplicates/conflicts, manifest
  drift and source modifications block the audit. Reports are private and
  atomically replaced. Successful automation leaves `publication_ready: false`.
- The explorer now hosts one to three scoped graph instances. Each owns its
  selection, controller, request generation, chart, statistics, table and CSV.
  Add, duplicate, edit, expand/collapse and remove actions preserve independent
  selections. One graph always remains. A loading/failed graph does not cancel
  another graph's work.
- A shared data client pins one publication, deduplicates in-flight requests
  without exposing a shared fetch to a card's abort signal, and uses an LRU
  payload budget of approximately 64 MiB (JSON length × 2, not a measured heap
  cap). Failed/invalid requests can be retried. New-publication checks offer an
  explicit reload rather than mixing releases.
- Workspace controls include focus/equal layouts, temperature and water presets,
  date/location links and exact-label inspection links. Date links explain
  unsupported overlay/seasonal calendars. Temperatures and precipitation keep
  independent units. Charts use the card's width and redraw on resize; frost SVG
  accessibility IDs are unique across cards. Responsive CSS stacks narrow cards,
  and the three-card focus layout places the primary graph over two supporting
  graphs. Visual acceptance remains outstanding.
- Version 2 workspace URLs retain up to three graph configurations, layout and
  links, with bounded input. Old single-graph URLs still open. Workspace studies
  save applied selections, notes, release identity and all card settings. Backup,
  import, removal and printable current-view reports are available. V1 charts
  migrate as one-card workspaces; original V1 storage is preserved. Corrupt
  storage disables writes rather than replacing old content. Missing saved
  locations fail in their own card rather than silently substituting a location.
- Export/package allowlists and wheel assets include the new modules. Data
  remains schema 1.2.0 for this increment; old 1.0/1.1/1.2 verification remains
  supported. New workspace assets are required when the HTML references them,
  without making them mandatory for an older 1.2 publication.

## Full-file source result

Private report: `.tmp/climate-audit.json`, generated
`2026-11-22T01:29:14.402688+00:00` (November 21 locally).

| Evidence | Result |
| --- | --- |
| Manifest SHA-256 | `1a50743587c467179eb3d01b39fa0959401a1f84d7fccea44027db4060d039b1` |
| Report SHA-256 | `ed94baa11dc1a04cb57e3fb4e12b3c2cadb594c5fe2d11ff52a823a813b6ba7b` |
| Files / coordinates | 21 / 21 |
| Source bytes | 1,695,903,321 |
| Rows per source / total | 376,944 / 7,915,824 |
| UTC span of every file | 1980-01-01 00:00 through 2022-12-31 23:00 |
| Missing hours, duplicate/reversed steps, off-hour timestamps | 0 |
| Metadata mismatches / source or manifest drift | 0 |
| Missing/unlisted CSV files | 0 |
| Missing surface-pressure / sea-level-pressure samples | 184,464 / 191 |
| Out-of-bound relative-humidity samples | 2,645 across all 21 locations |
| Relative-humidity observed range | 0.02–1.72; registry contract is 0–1 |
| Other out-of-bound metric samples | 0 under the registry's conservative screening bounds |
| Admission result | `review_required`; humidity unavailable at every location; no new metric admitted |

A complete timestamp inventory does not establish complete parameter coverage.
The humidity findings require source-specific investigation; they were not
clamped, dropped from the audit denominator or described as corrected data.
Screening bounds do not establish physical validity, licensing or independent
calculation correctness. The existing reviewed Twin Falls temperature and
precipitation publication remains authoritative.

Reproduce without collecting new data:

```sh
.venv/bin/python -m idwx data audit-climate \
  --source-root ~/source/idaho-weather-data/archive \
  --manifest configs/climate_sources.yml \
  --output reports/climate-audit.json
```

Exit 0 means the full audit completed with review still required. Exit 1 means
inventory, identity, continuity or read failures blocked it; inspect the report.
An individual metric can remain unavailable even when the overall audit exits 0.
A rerun changes the generation timestamp and report digest. Do not silently
repin changed source files or upload private reports alongside the site.

## Local verification and artifact

- Full Python regression run: **264 passed, 2 skipped**. The two skips require
  an explicitly configured Nginx executable. Subsequent audit hardening and the
  added elevation/manifest-drift cases passed the final **15-case audit suite**.
- Frontend regression coverage includes legacy views, three independent cards,
  shared fetches, failed-card retry, removal during loading, matching keyboard
  cursors, unique frost SVG IDs, workspace URLs, storage migration and backup
  round trips. The full suite passed 72 cases, followed by the final 28-case
  explorer/workspace run including the added URL/unavailable-location case.
  The final complete frontend run passed **73 cases**.
- Both added verifier checks passed: older 1.2 asset sets remain accepted,
  and new workspace HTML requires its runtime modules.
- Ruff and `git diff --check` pass.
- An extracted wheel generated, verified and packaged its synthetic fixture,
  establishing that the new runtime modules are included in a portable wheel.
  This wheel check is separate from the real-data site.
- The local real-data prototype is `.tmp/workspace-preview`. Its data release is
  `2d205c7f80159f444c418a127f6dbd7da274b51d93c3abc9880153e1176f844b`,
  classified `historical_reanalysis`, with one published location and 94 annual
  records. No additional archive parameters are exposed in this prototype.
  The verified ZIP `.tmp/delivery/idwx-workspace-prototype.zip` has SHA-256
  `7c9ae3c0e4c03ba9b10e0d585734d462517f2aaf4791dcf3a2df213230462caf`.
  Its local receipt is `.tmp/delivery/idwx-workspace-receipt.json`.
- Compressed file-size inspection (not measured browser/network performance):
  the five JSON files needed for the default 2025 graph total 33,642 gzip bytes
  and 390,774 JSON text bytes. Three graphs selecting 2020, 2021 and 2022 share
  four metadata files and need three daily chunks: 49,051 gzip bytes and
  604,089 JSON text bytes across seven unique files. These counts exclude
  HTML/CSS/JS, HTTP headers, browser heap and DOM overhead. Three identical
  selections fetch their shared daily chunk once in the integration test.
- The native browser inventory returned **“The browser access host is
  unavailable”**. No native screenshots, mobile/desktop visual acceptance,
  browser downloads, zoom or touch checks are claimed. DOM regression tests
  establish behavior, not visual acceptance.

Build and serve the local prototype for review:

```sh
.venv/bin/python -m idwx export-web --output .tmp/workspace-preview
.venv/bin/python -m idwx verify-web .tmp/workspace-preview
.venv/bin/python -m http.server 8765 --bind 127.0.0.1 --directory .tmp/workspace-preview
```

Open `http://127.0.0.1:8765/explore.html`. Review one/two/three cards at 390, 768,
1280 and 1920 CSS pixels, keyboard inspection, reduced motion, browser zoom,
downloads, native history navigation and independent error recovery. Stop the
local server after review. This is a temporary static preview, not a new service.

## Remaining planned work and acceptance gates

| Phase | State |
| --- | --- |
| 1. Source and metric audit | Automated full-file audit and registry implemented; source/metric semantic admission remains explicit and open. |
| 2. Workspace prototype | Independent cards and existing real-data prototype implemented; native visual review is unavailable in this session. |
| 3. Climate snapshot | Not implemented: normalized hourly Parquet, metric-specific daily reductions, independent numerical references and Twin Falls overlap comparison. |
| 4. Publication contract | Not implemented: schema 2.0.0 family chunks, availability, content identity, semantic verifier and pinned-snapshot updater composition/rehearsal. |
| 5. Broad explorer integration | Not implemented: 21-location/27-parameter selectors, compatible generic metric controls and plots. |
| 6. Analysis and studies | Workspace persistence/migration implemented; new rolling/threshold/distribution and cross-location analyses remain planned. |
| 7. Advanced views/readiness | Not implemented: hourly views, wind rose, aligned paired analysis, trend methods, coordinate map, image exports and measured browser budgets. |

The new archive must not be published by relabeling this audit as admission.
Resolve humidity/source-method questions and establish independent reducer
references before generating the expanded products. The plan also calls for
review of the real-data workspace prototype before the complete redesign.
All seven phases remain in scope; this first increment does not claim their
completion. Production deployment follows local acceptance and the existing
narrow production procedure, with its own rollback and evidence.
