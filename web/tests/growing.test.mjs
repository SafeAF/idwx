import assert from 'node:assert/strict';
import { test } from 'node:test';
import { cumulativePrecipitation, growingAssessment, precipitationPeriods, precipToUnit } from '../js/growing.js';
const row = (date, precip = 1) => ({ date, tmean_c: 12, tmin_c: 6, tmax_c: 18, precip_mm: precip, precip_status: precip === null ? 'not_reviewed' : 'complete', precip_valid_hours: precip === null ? null : 24, precip_expected_hours: precip === null ? null : 24 });

test('precipitation totals preserve dry zeroes, inch units, and missing leap days', () => {
  const days = Array.from({ length: 29 }, (_, i) => row(`2020-02-${String(i + 1).padStart(2, '0')}`, 0));
  assert.equal(precipitationPeriods(days)[0].total_mm, 0);
  days[0] = row('2020-02-01', 25.4);
  assert.equal(precipToUnit(precipitationPeriods(days)[0].total_mm, 'in'), 1);
  assert.equal(precipitationPeriods(days)[0].wet_days_ge_1mm, 1);
  assert.equal(precipitationPeriods(days.slice(1))[0].total_mm, null);
  assert.equal(precipitationPeriods(days.slice(1))[0].expected_days, 29);
});
test('cumulative curves stop at missing values or absent dates and restart only with a new year', () => {
  assert.deepEqual(cumulativePrecipitation([row('2020-01-01'), row('2020-01-02', null), row('2020-01-03'), row('2021-01-01', 0)]).map(r => r.cumulative_mm), [1, null, null, 0]);
  assert.deepEqual(cumulativePrecipitation([row('2020-01-01'), row('2020-01-03')]).map(r => r.cumulative_mm), [1, null]);
});
test('growing-year requirements are separate evidence checks, never an inferred fertility score', () => {
  const days = Array.from({ length: 365 }, (_, i) => row(new Date(Date.UTC(2021, 0, i + 1)).toISOString().slice(0, 10)));
  const annual = [{ season_year: 2021, threshold_c: 0, interval: { status: 'verified', frost_free_days: 122 } }];
  const policy = { gdd: { start: '04-01', end: '10-31', base_c: 10, cap_c: null }, hdd_base_c: 18, cold_threshold_c: -10 };
  const requirements = { gdd: 400, days: 120, water: true, precip_mm: 100 };
  const assess = (requirementsOverride = {}) => growingAssessment(days, annual, policy, { ...requirements, ...requirementsOverride }, 0)[0];
  assert.equal(assess().gdd, 428); assert.equal(assess().seasonal_precip_mm, 214); assert.equal(assess().assessment, 'requirements_met');
  days[151] = row('2021-06-01', null);
  assert.equal(assess().assessment, 'insufficient_data'); assert.equal(assess().precipitation_requirement, 'unknown');
  assert.equal(assess({ water: false }).assessment, 'requirements_met');
  assert.equal(assess({ gdd: 500 }).assessment, 'requirements_not_met');
  days[181].tmean_c = null;
  assert.equal(assess({ water: false }).assessment, 'insufficient_data');
  assert.equal(assess().soil_fertility, undefined);
});
