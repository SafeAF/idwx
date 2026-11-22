import assert from 'node:assert/strict';
import { execFileSync } from 'node:child_process';
import { readFileSync, mkdtempSync, mkdirSync, rmSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { test } from 'node:test';
import { aggregate, csv, monthlyAnomalies, normalizeSelection, seasonalMetrics, statistics, toUnit, validateChunk } from '../js/analytics.js';

const root = fileURLToPath(new URL('../../', import.meta.url));
const row = (date, value) => ({ date, tmin_c: value, tmean_c: value, tmax_c: value, status: value === null ? 'missing_record' : 'valid', n_obs: value === null ? null : 24 });

test('units distinguish absolute temperatures from differences; missing is never zero', () => {
  assert.equal(toUnit(0, 'f'), 32); assert.equal(toUnit(10, 'f', true), 18); assert.equal(toUnit(null, 'f'), null);
  const s = statistics([row('2024-01-01', -2), row('2024-01-02', 4), row('2024-01-03', null)]);
  assert.equal(s.expected, 3); assert.deepEqual(s.tmean_c, { valid: 2, mean: 1, min: -2, max: 4 });
});
test('complete monthly anomaly uses paired days and fixed median; gaps invalidate whole-month values', () => {
  const rows = Array.from({ length: 29 }, (_, i) => row(`2024-02-${String(i + 1).padStart(2, '0')}`, 5));
  const climate = { days: rows.map(r => ({ month_day: r.date.slice(5), mean_c: { p50: 2 } })) };
  assert.equal(monthlyAnomalies(rows, climate, 'tmean_c')[0].value, 3);
  assert.equal(monthlyAnomalies(rows.slice(1), climate, 'tmean_c')[0].value, null);
  assert.equal(aggregate(rows.slice(1), 'tmean_c')[0].valid, 28);
});
test('URL values are bounded, comparisons deduplicated, and date order repaired', () => {
  const s = normalizeSelection(new URLSearchParams('start=1900-01-01&end=bad&compare=2020,2020,1999,2021,2022,2023&threshold=99&unit=f'), [2020, 2021, 2022, 2023], [0, -2]);
  assert.equal(s.start, '2020-01-01'); assert.equal(s.end, '2022-12-31'); assert.deepEqual(s.compare, [2020, 2021, 2022]); assert.equal(s.threshold, 0);
  assert.equal(normalizeSelection(new URLSearchParams('start=2023-11-01&end=2021-01-01'), [2021, 2022, 2023], [0]).end, '2023-11-01');
});
test('CSV preserves missing values and selection metadata without altering canonical values', () => {
  const text = csv([row('2024-02-29', null)], { release_id: 'abc', display_unit: 'f', compare: '2020,2024' });
  assert.match(text, /"2020,2024"/); assert.match(text, /"2024-02-29","","","","missing_record"/); assert.ok(text.endsWith('\r\n'));
});
test('Python and JavaScript agree on coverage-aware seasonal metrics, including leap winter and gaps', () => {
  mkdirSync(path.join(root, '.tmp'), { recursive: true });
  const scratch = mkdtempSync(path.join(root, '.tmp', 'analytics-parity-'));
  try {
    const python = process.env.IDWX_PYTHON || path.join(root, '.venv/bin/python');
    const data = JSON.parse(execFileSync(python, ['-c', `
import json,pandas as pd
from idwx.analytics import public_daily,seasonal_summary,metric_policy
from idwx.config import load_config
frame=pd.DataFrame({'date':pd.date_range('2023-01-01','2024-12-31'),'tmin_c':-10.,'tmean_c':12.,'tmax_c':20.})
frame.loc[frame.date.eq('2024-04-03'),'tmean_c']=None
frame.loc[frame.date.eq('2024-01-15'),'tmin_c']=None
daily=public_daily(frame); policy=metric_policy(load_config('configs/idaho.yml'))
print(json.dumps({'daily':daily,'policy':policy,'expected':seasonal_summary(daily,policy)}))
`], { cwd: root, encoding: 'utf8' }));
    assert.deepEqual(seasonalMetrics(Object.values(data.daily).flat(), data.policy), data.expected);
    assert.equal(data.expected[1].gdd_subtotal, 4);
    assert.throws(() => validateChunk({ station_id: 'x', year: 2024, days: data.daily['2024'].slice(1) }, 'x', 2024), /archive/);
  } finally { rmSync(scratch, { recursive: true }); }
});

test('graph-builder period statistics compute known means/extremes and reject incomplete leap months', async () => {
  const { periodStatistics, selectedRows } = await import('../js/analytics.js');
  const january = Array.from({ length: 31 }, (_, i) => row(`2023-01-${String(i + 1).padStart(2, '0')}`, i));
  for (const [statistic, expected] of [['mean', 15], ['min', 0], ['max', 30]]) {
    assert.equal(periodStatistics(january, 'tmean_c', 'month', statistic)[0].value, expected);
  }
  const february = Array.from({ length: 28 }, (_, i) => row(`2024-02-${String(i + 1).padStart(2, '0')}`, 5));
  const missingLeapDay = periodStatistics(february, 'tmean_c', 'year', 'mean', 2)[0];
  assert.equal(missingLeapDay.expected, 29); assert.equal(missingLeapDay.valid, 28); assert.equal(missingLeapDay.value, null);
  const selection = normalizeSelection(new URLSearchParams('view=month-compare&month=2&start=2023-04-01&end=2024-06-01&statistic=max&mark=scatter'), [2023, 2024], [0]);
  assert.equal(selection.start, '2023-01-01'); assert.equal(selection.end, '2024-12-31');
  assert.equal(selection.mark, 'scatter'); assert.equal(selection.statistic, 'max');
  assert.equal(selectedRows([{ days: [...january, ...february] }], selection).length, 28);
  const profile = normalizeSelection(new URLSearchParams('view=monthly&profile_year=2024&month=99&mark=pie&statistic=unknown'), [2023, 2024], [0]);
  assert.equal(profile.start, '2024-01-01'); assert.equal(profile.end, '2024-12-31');
  assert.equal(profile.month, 1); assert.equal(profile.mark, 'bar'); assert.equal(profile.statistic, 'mean');
});

test('calendar records preserve tied dates, arithmetic means, leap samples and missing coverage', async () => {
  const { calendarRecords } = await import('../js/analytics.js');
  const data = [
    { ...row('2020-01-01', 2), tmin_c: -5, tmax_c: 10 },
    { ...row('2021-01-01', 4), tmin_c: -5, tmax_c: 12 },
    { ...row('2022-01-01', 9), status: 'invalid_temperature' },
    row('2020-02-29', 0), row('2024-02-29', null),
  ];
  const records = calendarRecords(data, 2020, 2024), january = records[0];
  assert.equal(records.length, 366);
  assert.equal(january.record_low_c, -5); assert.equal(january.record_high_c, 12);
  assert.equal(january.mean_daily_c, 3); assert.equal(january.valid_years, 2);
  assert.equal(january.expected_years, 5); assert.equal(january.status, 'partial');
  assert.equal(january.record_low_dates, '2020-01-01;2021-01-01');
  assert.equal(january.record_high_dates, '2021-01-01');
  const leap = records.find(r => r.month_day === '02-29');
  assert.equal(leap.expected_years, 2); assert.equal(leap.valid_years, 1); assert.equal(leap.mean_daily_c, 0);
  assert.equal(records[1].status, 'no_data'); assert.equal(records[1].record_low_c, null);
  assert.equal(calendarRecords([], 2021, 2023).length, 365);
  const selection = normalizeSelection(new URLSearchParams('view=calendar&start=2020-05-01&end=2024-06-01'), [2020, 2021, 2022, 2023, 2024], [0]);
  assert.equal(selection.start, '2020-01-01'); assert.equal(selection.end, '2024-12-31');
});

test('study backups require a bounded, versioned selection document', async () => {
  const { parseStudies } = await import('../js/analytics.js');
  assert.deepEqual(parseStudies(null), []);
  assert.throws(() => parseStudies('{}'), /study format/);
  assert.throws(() => parseStudies('{bad'));
  assert.throws(() => parseStudies(JSON.stringify({ version: 1, charts: Array(51).fill({}) })), /study format/);
  assert.throws(() => parseStudies(JSON.stringify({ version: 1, charts: [{ name: 'Missing filters' }] })), /saved chart/);
});
