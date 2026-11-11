import assert from 'node:assert/strict';
import { test } from 'node:test';
import { calendarDay, displayDate, dateParts, temperature, historicalText, historyCSV, releaseURL, validateRelease } from '../web/js/format.js';

test('date-only values retain their calendar day in every local timezone', () => {
  assert.equal(displayDate('2020-03-01'), 'Mar 1');
  assert.equal(calendarDay('2019-03-01'), calendarDay('2020-03-01'));
  assert.equal(calendarDay('2020-02-29'), 60);
  assert.equal(dateParts('2019-02-29'), null);
  assert.equal(displayDate(null), '—');
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
