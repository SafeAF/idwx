import { finite, mean, seasonalMetrics } from './analytics.js';

export const precipToUnit = (value, unit) => finite(value) ? value / (unit === 'in' ? 25.4 : 1) : null;
const completeDay = r => r?.precip_status === 'complete' && finite(r.precip_mm) && r.precip_mm >= 0 && r.precip_valid_hours === 24 && r.precip_expected_hours === 24;
export function precipitationPeriods(rows, frequency = 'month') {
  const groups = new Map();
  rows.forEach(row => {
    const period = frequency === 'year' ? row.date.slice(0, 4) : row.date.slice(0, 7);
    if (!groups.has(period)) groups.set(period, []);
    groups.get(period).push(row);
  });
  return [...groups].map(([period, days]) => {
    const year = Number(period.slice(0, 4));
    const expected = frequency === 'year' ? (Date.UTC(year + 1, 0, 1) - Date.UTC(year, 0, 1)) / 86400000 : new Date(Date.UTC(year, Number(period.slice(5, 7)), 0)).getUTCDate();
    const valid = days.filter(completeDay), isComplete = valid.length === expected;
    return { period, total_mm: isComplete ? valid.reduce((sum, r) => sum + r.precip_mm, 0) : null,
      wet_days_ge_1mm: isComplete ? valid.filter(r => r.precip_mm >= 1).length : null,
      valid_days: valid.length, expected_days: expected, status: isComplete ? 'complete' : 'incomplete' };
  });
}
export function cumulativePrecipitation(rows) {
  let year = null, total = 0, available = true, previous = null;
  return rows.map(row => {
    const current = row.date.slice(0, 4);
    if (current !== year) { year = current; total = 0; available = row.date.endsWith('01-01'); }
    if (current === year && previous && row.date.slice(0, 4) === previous.slice(0, 4) && Date.parse(row.date + 'T00:00:00Z') - Date.parse(previous + 'T00:00:00Z') !== 86400000) available = false;
    if (!completeDay(row)) available = false;
    if (available) total += row.precip_mm;
    previous = row.date;
    return { ...row, cumulative_mm: available ? total : null };
  });
}
export function growingAssessment(rows, annual, policy, requirements, threshold) {
  const seasons = seasonalMetrics(rows, policy);
  return seasons.map(season => {
    const frost = annual.find(r => r.season_year === season.year && r.threshold_c === threshold);
    const span = frost?.interval.status === 'verified' ? frost.interval.frost_free_days : null;
    const start = `${season.year}-${policy.gdd.start}`, end = `${season.year}-${policy.gdd.end}`;
    const window = rows.filter(r => r.date >= start && r.date <= end);
    const expected = (Date.parse(end + 'T00:00:00Z') - Date.parse(start + 'T00:00:00Z')) / 86400000 + 1;
    const valid = window.filter(completeDay);
    const total = valid.length === expected ? valid.reduce((sum, r) => sum + r.precip_mm, 0) : null;
    const check = (value, required) => !finite(value) ? 'unknown' : value >= required ? 'met' : 'shortfall';
    const heat = check(season.gdd, requirements.gdd), frostFree = check(span, requirements.days);
    const water = requirements.water ? check(total, requirements.precip_mm) : 'not_selected';
    const checks = [heat, frostFree, ...(requirements.water ? [water] : [])];
    return { year: season.year, gdd: season.gdd, gdd_valid: season.gdd_valid, gdd_expected: season.gdd_expected,
      frost_free_days: span, frost_status: frost?.interval.status ?? 'missing_dependency',
      seasonal_precip_mm: total, precip_valid_days: valid.length, precip_expected_days: expected,
      heat_requirement: heat, frost_free_requirement: frostFree, precipitation_requirement: water,
      assessment: checks.includes('shortfall') ? 'requirements_not_met' : checks.includes('unknown') ? 'insufficient_data' : 'requirements_met' };
  });
}
export function precipitationStats(rows) {
  const valid = rows.filter(completeDay);
  return { valid: valid.length, expected: rows.length, complete: valid.length === rows.length,
    total: valid.length === rows.length && rows.length ? valid.reduce((sum, r) => sum + r.precip_mm, 0) : null,
    dailyMean: mean(valid.map(r => r.precip_mm)), wetDays: valid.filter(r => r.precip_mm >= 1).length };
}
