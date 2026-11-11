# Research forecast date policy, version 1

This implements date interpretation and seasonal-window checks for
`frost_jan01_v1`. It does not validate model accuracy or enable forecast
publication. The website remains historical-only and its export contract is
unchanged.

## Numeric coordinates and configured support

Models continue to return numeric lower/point/upper coordinates on the common
leap-year 2000 calendar. March 1 is coordinate 61 in every forecast year.
The new policy retains these raw numbers without clipping, reordering,
interpolation changes or calibration refitting.

Before producing dates, require exactly three finite ordered values in
[1, 366], with **all bounds inside the station's effective target window**.
Window boundaries are inclusive `MM-DD` values mapped onto the reference
calendar. Reversed/cross-year or malformed windows fail. A spring interval
extending beyond the configured June 30 endpoint fails even if its point is
in May; a station override extending spring through July is respected.
No bound is silently moved onto a window endpoint.

## Exact discrete date representation

For raw numeric bounds L, P, U and explicit forecast year Y:

1. Enumerate integer reference-calendar coordinates from `ceil(L)` through
   `floor(U)`.
2. Map each coordinate's month/day into Y, excluding February 29 when Y is
   not a leap year.
3. Reject the interval if no actual date remains.
4. Choose the earliest/latest remaining dates as date bounds.
5. Choose the remaining coordinate nearest P as the point date. On an exact
   distance tie, choose the later date.

The result represents exactly the actual dates supported by the unchanged
numeric interval. The choice of point date is an explicit research rounding
policy, not evidence that the selected date is more likely. No timezone or
instant conversion is involved; outputs are ISO date-only strings.

Examples using the full calendar window:

| Raw lower/point/upper | Forecast year | Lower / point / upper dates | Elapsed endpoint width |
| --- | --- | --- | --- |
| 59 / 60 / 61 | 2000 | Feb 28 / Feb 29 / Mar 1 | 2 days |
| 59 / 60 / 61 | 2001 | Feb 28 / Mar 1 / Mar 1 | 1 day |
| 59.5 / 60 / 61 | 2001 | Mar 1 / Mar 1 / Mar 1 | 0 days |
| 60 / 60 / 60 | 2001 | Rejected: no real date exists | Unavailable |
| 151.1 / 151.5 / 151.9 | 2024 | Rejected: no integer date coordinate is included | Unavailable |

A zero-width interval on an actual date remains valid. A fractional or
non-leap February 29 interval containing no possible date is withheld rather
than assigned a fabricated date or expanded to include a neighbor.

## Shared prediction and evaluation behavior

CLI prediction applies the policy to both the selected model and its
climatology baseline. If either cannot be interpreted, the command returns
an actionable error and emits no partial prediction JSON.

Successful protocol JSON retains numeric `prediction` and
`baseline_climatology` values and adds separately named `prediction_dates`
and `baseline_climatology_dates` objects:

```json
{
  "lower": "2001-02-28",
  "point": "2001-03-01",
  "upper": "2001-03-01",
  "interval_width_days": 1
}
```

`interval_width_days` is elapsed days between uncertainty endpoints. The
date object does not certify P10/P90 quantiles, crop safety or calibrated
coverage. Existing `unvalidated_research` and `unvalidated_model_interval`
labels remain.

Backtesting uses the same interpretation function and the station's effective
window. Out-of-window/unsupported bounds become `invalid_prediction` folds
with a reason; raw numeric predictions remain in the diagnostic row, while
date fields and scores remain unavailable. Summary `n_invalid_predictions`
counts these explicitly. Invalid intervals are excluded from both coordinate
and date scores, so coverage applies only to successfully evaluated folds.
Read counts and rejection reasons alongside coverage; a high score on a small
accepted subset is not a publication criterion.

Protocol actual dates must match their season year and normalized target
coordinate before scoring. Inconsistent inputs fail rather than contributing
misleading errors.

## Separate calendar-coordinate and elapsed-day metrics

Existing `mae`, `rmse`, `bias`, `mean_interval_width`, and paired numeric
baseline comparisons retain their raw reference-calendar meanings. They are
not silently redefined by rounding. Near February 29, those coordinate
differences can differ from elapsed days in the forecast year.

Yearly rows now also contain policy version, effective window boundaries,
actual date, predicted date bounds/point, `date_error_days`,
`date_interval_covered` and `date_interval_width_days`.

Summary/per-station metrics add:

- `n_date_evaluated`: successful rows with interpreted dates.
- `date_mae_days`, `date_rmse_days`, `date_bias_days`: error of the chosen point
  date against the actual event date using elapsed days in that year.
- `date_coverage`: inclusive coverage by the interpreted date endpoints.
- `mean_date_interval_width_days`: elapsed-day width of those date intervals.
- `n_date_paired`, `paired_date_model_mae_days`, `baseline_date_mae_days`, and
  `delta_date_mae_days`: date error comparisons on identical successful
  station-years after joining the baseline.

For example, a February 28 point against a March 1 actual has raw coordinate
error -2 in both 2011 and 2012, but elapsed-date error -1 in 2011 and -2 in
2012. Date metrics are blank/null for legacy modeling without this policy or
when no valid date forecasts exist. No confidence interval or promotion
threshold for these metrics has been added.

## Versioning and migration

The protocol ID remains `frost_jan01_v1`, with its January 1 issue, history
holdback and model feature allowlist unchanged. **Input version is now 2**,
and **date policy version is 1**. Input rows record effective window boundaries;
provenance records the date policy version. Forecast context also records the
policy and effective target window.

Rebuild/retrain protocol artifacts under the new input version before use.
Old or missing date-policy provenance fails input validation, and registry
context/provenance checks prevent older artifacts from being selected silently.
Protocol registry format remains version 2; RF calibration version remains 2.
The numerical calibration method has not been altered by date interpretation.

## Verification and remaining requirements

Thirty-seven new synthetic cases cover leap/non-leap February behavior,
fractional and empty intervals, ties, year/calendar/window boundaries,
malformed inputs, station overrides, actual-date consistency, elapsed-day
metrics, paired date baselines, input provenance migration, and CLI prediction/
baseline failure handling. Full verification: **169 Python tests passed**,
one optional live Nginx test skipped; five frontend tests passed. Targeted Ruff
and `git diff --check` passed.

Real source/vintage audits, empirical accuracy and calibration under this
date policy, sufficient validation history, predetermined promotion criteria,
browser verification and production TLS remain outstanding. The new metrics
must be reviewed on real paired holdouts before predictions can be published.
