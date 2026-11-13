import assert from 'node:assert/strict';
import { test } from 'node:test';
import { calendarDay, displayDate, dateParts, temperature, historicalText, historyCSV, releaseURL, validateRelease, baselineDetails, yearInTimeZone, climatologyProjection } from '../web/js/format.js';

test('date-only values retain their calendar day in every local timezone', () => {
  assert.equal(displayDate('2020-03-01'), 'Mar 1');
  assert.equal(calendarDay('2019-03-01'), calendarDay('2020-03-01'));
  assert.equal(calendarDay('2020-02-29'), 60);
  assert.equal(dateParts('2019-02-29'), null);
  assert.equal(displayDate(null), '—');
});

test('baseline coverage separates no-event and missing years and compares month/day across leap years', () => {
  const records = [
    [1999, 'observed', '1999-01-01'],
    [2000, 'observed', '2000-02-29'],
    [2001, 'observed', '2001-03-01'],
    [2002, 'no_event_in_complete_window', null],
    [2003, 'incomplete_window', null],
    [2004, 'insufficient_data', null],
    [2005, 'observed', '2005-12-31'],
  ].map(([season_year, status, date]) => ({ season_year, last_spring_frost: { status, date } }));
  assert.deepEqual(baselineDetails(records, 'last_spring_frost', { baseline_period: { start_year: 2000, end_year: 2004 } }), {
    total: 5, confirmed: 2, noEvent: 1, incomplete: 1, unavailable: 1,
    earliest: '2000-02-29', latest: '2001-03-01',
  });
  assert.deepEqual(baselineDetails(records, 'last_spring_frost', { baseline_period: { start_year: 2002, end_year: 2004 } }), {
    total: 3, confirmed: 0, noEvent: 1, incomplete: 1, unavailable: 1, earliest: null, latest: null,
  });
});

test('threshold conversion preserves exact -2 C Fahrenheit label and missing values', () => {
  assert.equal(temperature(-2, 'f'), '28.4 °F');
  assert.equal(temperature(0, 'f'), '32 °F');
  assert.equal(temperature(null, 'c'), '—');
  assert.equal(temperature(Infinity, 'c'), '—');
});

test('insufficient history never becomes a typical date', () => {
  assert.match(historicalText({ status: 'insufficient_history', valid_year_count: 2, minimum_valid_years: 10 }), /2 confirmed years; 10 required/);
});

test('climatology projection uses station-local year, historical median, and P10-P90 range', () => {
  assert.equal(yearInTimeZone('America/Boise', new Date('2026-01-01T01:00:00Z')), 2025);
  assert.equal(yearInTimeZone('UTC', new Date('2026-01-01T01:00:00Z')), 2026);
  const distribution = (p10, p50, p90, valid_year_count = 40) => ({
    status: 'available', valid_year_count,
    baseline_period: { start_year: 1980, end_year: 2022 },
    p10: { month_day: p10 }, p50: { month_day: p50 }, p90: { month_day: p90 },
  });
  const projection = climatologyProjection({
    last_spring_frost: distribution('04-10', '04-20', '05-01'),
    first_fall_frost: distribution('09-20', '10-01', '10-12', 39),
  }, 2026);
  assert.deepEqual(projection.spring, {
    p10: '04-10', p50: '04-20', p90: '05-01', confirmedYears: 40,
    baseline: { start_year: 1980, end_year: 2022 },
  });
  assert.equal(projection.fall.p50, '10-01');
  assert.equal(projection.frostFreeDays, calendarDay('2000-10-01') - calendarDay('2000-04-20') - 1);
  assert.equal(climatologyProjection({
    last_spring_frost: { status: 'insufficient_history' },
    first_fall_frost: { status: 'insufficient_history' },
  }, 2026).frostFreeDays, null);
});

test('all data references are pinned to the same release and origin', () => {
  const base = 'https://example.com/frost/';
  assert.equal(releaseURL('data/abc/stations/a.json', base, 'abc').href, base + 'data/abc/stations/a.json');
  for (const path of ['https://elsewhere.test/data/abc/a.json', 'data/other/a.json', '../data/abc/a.json', 'data/abc/a.json?new=1']) {
    assert.throws(() => releaseURL(path, base, 'abc'));
  }
  assert.throws(() => validateRelease({ schema_version: '1.0.0', release_id: 'other', data_kind: 'synthetic_demo' }, { release_id: 'abc', data_kind: 'synthetic_demo' }));
});

test('filtered CSV keeps missing values blank and includes source labels', () => {
  const doc = { schema_version: '1.0.0', release_id: 'abc', data_kind: 'synthetic_demo', station: { station_id: 'test' } };
  const record = { season_year: 2020, threshold_c: -2,
    last_spring_frost: { date: null, status: 'incomplete_window', candidate_date: '2020-05-01' },
    first_fall_frost: { date: '2020-09-01', status: 'observed' },
    interval: { frost_free_days: null, status: 'missing_dependency' } };
  const csv = historyCSV(doc, [record]);
  assert.match(csv, /"synthetic_demo"/);
  assert.match(csv, /"incomplete_window"/);
  assert.match(csv, /"2020-05-01"/);
  assert.ok(!csv.includes('null'));
  assert.equal(csv.split('\r\n').length, 3);
});
