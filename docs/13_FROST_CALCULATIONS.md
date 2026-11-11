# Historical frost calculation contract and implementation status

This implements the calculation foundation of Phase 1 in
`12_FROST_STATIC_WEBSITE_PLAN.md`. No historical station results or forecasts
have been published by this change.

## Input and coverage

Daily inputs must have unique, nonmissing, timezone-naive local calendar dates.
Duplicate daily dates fail with an actionable error rather than selecting an
arbitrary observation. Nonfinite minima and minima flagged
`insufficient_coverage` are unavailable. Missing temperatures are never filled.

Hourly ingestion rejects duplicate UTC timestamps, including across chunk
boundaries. Coverage counts distinct UTC hours containing finite temperatures,
so subhourly readings cannot inflate daily coverage. A day still needs the
configured `hourly_min_coverage` (18 by default). This is a coverage criterion,
not proof that every hour or the true daily minimum was observed.

Explicit source UTC offsets take precedence; otherwise ingestion uses the
station's optional IANA `timezone`, falling back to the global timezone.
The source adapter currently expects ERA5-style headers with Celsius units and
UTC timestamps. It does not automatically validate arbitrary daily CSV formats
or ambiguous timestamp conventions.

## Event records

Every configured year, including entirely absent years, gets records for each
threshold. The year range comes from `backtest.start_year` / `end_year`; absent
bounds fall back to observed years. Empty data requires explicit bounds.

Spring selects the last minimum at or below the threshold; fall selects the
first. Window boundaries are inclusive and within one calendar year.

| Status | Meaning |
| --- | --- |
| `observed` | Every expected day has a valid minimum and an event was found. |
| `no_event_in_complete_window` | Every expected day is valid and none qualifies. |
| `incomplete_window` | At least one valid day exists, but other expected days are missing or invalid. |
| `insufficient_data` | No valid minimum exists in the window. |

`value_date`, `value_doy`, and `event_tmin_c` describe confirmed events only.
`candidate_date` and `candidate_tmin_c` retain the first/last available event
even when coverage prevents confirmation. An unfinished spring cannot be
confirmed just because a candidate exists. Coverage includes `window_start`,
`window_end`, `expected_days`, `valid_days`, `missing_days`, and
`coverage_fraction`. Leap years use the actual expected calendar days.

The global windows default to January-June and August-December. Override
complete windows per station in the main configuration, for example:

```yaml
station_windows:
  twin_falls:
    last_spring_frost:
      start: "01-01"
      end: "07-31"
```

No station-specific overrides have been selected without source-data review.

## Interval records

For compatibility, `freeze_free_days.value_float` retains
`fall_date - spring_date` when both dates are confirmed and correctly ordered.
`interval_definition` labels that convention explicitly. This date difference
alone does not establish a freeze-free season.

The separate `frost_free_days` field is `max(0, difference - 1)` only if every
strictly intervening day has a valid minimum above the selected threshold.
Otherwise it is missing. `status` / `quality_flags` identify
`missing_dependency`, `invalid_event_order`, `incomplete_interval`, or
`intervening_freeze`; a fully verified interval has status `verified`.
`summer_freeze` flags any observed July freeze at the selected threshold. A
false flag does not imply full July coverage. Intervening freezes in any month
invalidate the frost-free count.

## Data audit and verification

The configured `~/historical_weather_data-1980-2022/idaho/` directory was absent
in the implementation workspace. Station metadata currently lists Twin Falls.
Actual station availability, units, reporting-day conventions, duplicates,
summer freezes, and date ranges still require review against real source files.
This prevents declaring Phase 1 complete against real data.

`idwx data audit` now inventories raw sources, fingerprints files, reports
parsing/offset/coverage findings, recomputes this contract's frost records and
compares current daily caches. It is read-only with respect to sources/caches
and never certifies provenance or publication readiness. See
`20_SOURCE_DATA_AUDIT.md` for commands, manual review requirements and the
blocked workspace result. Ingestion now also rejects invalid timestamps and
explicit offsets, ambiguous station matches and duplicate headers; cache
builds reject multiple source files mapping to one station rather than
silently skipping/overwriting them.

Synthetic regression tests cover threshold equality, multiple events, exact
-2 C behavior, inclusive boundaries, leap-year coverage, absent/null/nonfinite
days, quality flags, missing years, empty stations, unfinished seasons, July
freezes and gaps, interval counting, duplicate daily/hourly records, subhourly
coverage, station timezone overrides, and explicit source offsets.

Verification: all 21 synthetic tests passed, including a Parquet cache round
trip. Targeted Ruff checks and `git diff --check` passed.

Run `uv run pytest -q` and `uv run ruff check idwx/frost.py idwx/config.py
idwx/targets.py idwx/io.py tests`. Rebuild old daily and target caches; remove
or regenerate derived feature/dataset/model artifacts before training again.
Historical calendar-normalized summaries and static export are implemented;
see `14_WEB_DATA_CONTRACT.md` and `15_DASHBOARD_DEPLOYMENT.md`. The initial
forecast source audit and input corrections are documented in
`16_FORECAST_CORRECTNESS_AUDIT.md`. Existing model predictions have not been
validated for publication.
