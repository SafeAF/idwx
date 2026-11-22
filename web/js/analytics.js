import { calendarDay, dateParts } from './format.js';

export const METRICS = { tmin_c: 'Daily minimum', tmean_c: 'Daily mean', tmax_c: 'Daily maximum' };
export const finite = value => typeof value === 'number' && Number.isFinite(value);
export const toUnit = (value, unit, difference = false) => finite(value) ? (unit === 'f' ? value * 1.8 + (difference ? 0 : 32) : value) : null;
export const mean = values => values.length ? values.reduce((a, b) => a + b, 0) / values.length : null;
export const shown = value => finite(value) ? Number(value.toFixed(2)).toLocaleString('en-US') : '—';

export function validatePrecipitation(row) {
  const status = row.precip_status;
  if (status === 'not_reviewed') return row.precip_mm === null && row.precip_valid_hours === null && row.precip_expected_hours === null;
  if (!['complete', 'incomplete'].includes(status) || row.precip_expected_hours !== 24 || !Number.isInteger(row.precip_valid_hours) || row.precip_valid_hours < 0 || row.precip_valid_hours > 24) return false;
  return status === 'complete' ? finite(row.precip_mm) && row.precip_mm >= 0 && row.precip_valid_hours === 24 : row.precip_mm === null && row.precip_valid_hours < 24;
}

export function validateChunk(doc, station, year) {
  const expected = new Date(Date.UTC(year + 1, 0, 1)) - new Date(Date.UTC(year, 0, 1));
  if (doc.station_id !== station || doc.year !== year || !Array.isArray(doc.days) || doc.days.length !== expected / 86400000) throw new Error('Invalid daily archive.');
  doc.days.forEach((row, i) => {
    const date = new Date(Date.UTC(year, 0, 1 + i)).toISOString().slice(0, 10);
    if (doc.schema_version === '1.2.0' && !validatePrecipitation(row)) throw new Error('Invalid precipitation coverage.');
    if (row.date !== date || !['valid', 'invalid_temperature', 'missing_record'].includes(row.status) ||
        !Object.keys(METRICS).every(key => row[key] === null || finite(row[key])) ||
        (row.status === 'valid') !== Object.keys(METRICS).every(key => finite(row[key])) ||
        (row.status === 'missing_record' && (row.n_obs !== null || Object.keys(METRICS).some(key => row[key] !== null))) ||
        !(row.n_obs === null || Number.isInteger(row.n_obs) && row.n_obs >= 0)) throw new Error('Invalid daily archive values.');
  });
  return doc;
}

export function normalizeSelection(params, years, thresholds, sourceLastDate = null) {
  const first = `${years[0]}-01-01`, last = `${years.at(-1)}-12-31`;
  const sourceYear = Number(sourceLastDate?.slice(0, 4));
  const completeYear = sourceLastDate?.endsWith('-12-31') ? sourceYear : sourceYear - 1;
  const defaultYear = years.includes(completeYear) ? completeYear : years.length > 1 ? years.at(-2) : years[0];
  const clamp = (value, fallback) => dateParts(value) ? (value < first ? first : value > last ? last : value) : fallback;
  let start = clamp(params.get('start'), `${defaultYear}-01-01`);
  let end = clamp(params.get('end'), `${defaultYear}-12-31`);
  if (end < start) end = start;
  const compare = [...new Set((params.get('compare') || years.filter(y => y <= defaultYear).slice(-3).join(',')).split(',').map(Number).filter(y => years.includes(y)))].slice(0, 3);
  const views = ['temperature', 'compare', 'annual', 'month-compare', 'monthly', 'calendar', 'frost', 'heatmap', 'seasonal', 'table', 'precipitation', 'assessment'];
  const view = views.includes(params.get('view')) ? params.get('view') : 'temperature';
  if (['annual', 'month-compare', 'calendar'].includes(view) && !params.has('start')) start = first;
  const profileYear = years.includes(Number(params.get('profile_year'))) ? Number(params.get('profile_year')) : Number(start.slice(0, 4));
  if (view === 'monthly') { start = `${profileYear}-01-01`; end = `${profileYear}-12-31`; }
  if (['annual', 'month-compare', 'calendar', 'frost', 'assessment'].includes(view)) { start = start.slice(0, 4) + '-01-01'; end = end.slice(0, 4) + '-12-31'; }
  return {
    view, start, end, profile_year: profileYear,
    rain_mode: ['daily', 'monthly', 'yearly', 'compare', 'cumulative'].includes(params.get('rain_mode')) ? params.get('rain_mode') : 'monthly',
    rain_unit: params.get('rain_unit') === 'in' ? 'in' : 'mm',
    need_gdd: finite(Number(params.get('need_gdd'))) && params.has('need_gdd') ? Math.max(0, Math.min(10000, Number(params.get('need_gdd')))) : 1000,
    need_days: finite(Number(params.get('need_days'))) && params.has('need_days') ? Math.round(Math.max(0, Math.min(366, Number(params.get('need_days'))))) : 100,
    need_precip: finite(Number(params.get('need_precip'))) && params.has('need_precip') ? Math.max(0, Math.min(10000, Number(params.get('need_precip')))) : 0,
    need_water: params.get('need_water') === 'on',
    month: Number.isInteger(Number(params.get('month'))) && Number(params.get('month')) >= 1 && Number(params.get('month')) <= 12 ? Number(params.get('month')) : 1,
    statistic: ['mean', 'min', 'max'].includes(params.get('statistic')) ? params.get('statistic') : 'mean',
    mark: ['line', 'scatter', 'bar', 'area'].includes(params.get('mark')) ? params.get('mark') : ['annual', 'month-compare', 'monthly'].includes(view) ? 'bar' : 'line',
    compare: compare.length ? compare : [defaultYear], unit: params.get('unit') === 'f' ? 'f' : 'c',
    threshold: params.has('threshold') && thresholds.includes(Number(params.get('threshold'))) ? Number(params.get('threshold')) : thresholds[0],
    metric: Object.hasOwn(METRICS, params.get('metric')) ? params.get('metric') : 'tmean_c',
    baseline: params.get('baseline') !== 'off',
  };
}

export function selectionYears(selection, available) {
  return (selection.view === 'compare' || selection.view === 'precipitation' && ['compare', 'cumulative'].includes(selection.rain_mode)) ? selection.compare : available.filter(y => y >= Number(selection.start.slice(0, 4)) && y <= Number(selection.end.slice(0, 4)));
}

export function selectedRows(chunks, selection) {
  return chunks.flatMap(chunk => chunk.days).filter(row => (selection.view === 'compare' || selection.view === 'precipitation' && ['compare', 'cumulative'].includes(selection.rain_mode) || row.date >= selection.start && row.date <= selection.end) && (selection.view !== 'month-compare' || Number(row.date.slice(5, 7)) === selection.month)).sort((a, b) => a.date.localeCompare(b.date));
}

export function statistics(rows) {
  const stats = { expected: rows.length };
  for (const key of Object.keys(METRICS)) {
    const values = rows.map(r => r[key]).filter(finite);
    stats[key] = { valid: values.length, mean: mean(values), min: values.length ? Math.min(...values) : null, max: values.length ? Math.max(...values) : null };
  }
  return stats;
}

export function calendarRecords(rows, startYear, endYear) {
  const byDate = new Map(rows.map(row => [row.date, row]));
  const groups = new Map();
  for (let year = startYear; year <= endYear; year++) {
    for (let time = Date.UTC(year, 0, 1); time < Date.UTC(year + 1, 0, 1); time += 86400000) {
      const date = new Date(time).toISOString().slice(0, 10), day = date.slice(5);
      if (!groups.has(day)) groups.set(day, { expected: 0, days: [] });
      const group = groups.get(day), row = byDate.get(date);
      group.expected++;
      if (row?.status === 'valid' && Object.keys(METRICS).every(key => finite(row[key]))) group.days.push(row);
    }
  }
  return [...groups].sort(([a], [b]) => a.localeCompare(b)).map(([month_day, { expected, days }]) => {
    const low = days.length ? Math.min(...days.map(row => row.tmin_c)) : null;
    const high = days.length ? Math.max(...days.map(row => row.tmax_c)) : null;
    return { month_day, record_low_c: low, record_high_c: high, mean_daily_c: mean(days.map(row => row.tmean_c)),
      record_low_dates: days.filter(row => row.tmin_c === low).map(row => row.date).sort().join(';'),
      record_high_dates: days.filter(row => row.tmax_c === high).map(row => row.date).sort().join(';'),
      valid_years: days.length, expected_years: expected, status: !days.length ? 'no_data' : days.length === expected ? 'complete' : 'partial' };
  });
}

// Stored studies contain selections, never executable markup or remote URLs.
export function parseStudies(text) {
  if (text === null) return [];
  const doc = JSON.parse(text);
  if (doc?.version !== 1 || !Array.isArray(doc.charts) || doc.charts.length > 50) throw new Error('Invalid study format.');
  for (const chart of doc.charts) {
    if (!chart || !['study', 'name', 'notes', 'query', 'station', 'release', 'saved_at', 'data_kind'].every(key => typeof chart[key] === 'string') ||
        !chart.study.trim() || chart.study.length > 120 || !chart.name.trim() || chart.name.length > 120 || chart.notes.length > 500 || chart.query.length > 4000 ||
        !/^[a-z][a-z0-9_]*$/.test(chart.station) || !/^[a-f0-9]{64}$/.test(chart.release) ||
        !['historical_reanalysis', 'synthetic_demo'].includes(chart.data_kind) || !Number.isFinite(Date.parse(chart.saved_at))) throw new Error('Invalid saved chart.');
  }
  return doc.charts;
}

export function aggregate(rows, key, frequency = 'month') {
  const groups = new Map();
  rows.forEach(row => {
    const period = frequency === 'year' ? row.date.slice(0, 4) : row.date.slice(0, 7);
    if (!groups.has(period)) groups.set(period, []);
    groups.get(period).push(row);
  });
  return [...groups].map(([period, days]) => {
    const year = Number(period.slice(0, 4)), month = Number(period.slice(5, 7));
    const expected = frequency === 'year' ? (Date.UTC(year + 1, 0, 1) - Date.UTC(year, 0, 1)) / 86400000 : new Date(Date.UTC(year, month, 0)).getUTCDate();
    const values = days.map(d => d[key]).filter(finite);
    return { period, value: values.length === expected ? mean(values) : null, valid: values.length, expected };
  });
}

export function periodStatistics(rows, key, frequency, statistic, month = null) {
  const periods = aggregate(rows, key, frequency);
  return periods.map(period => {
    const values = rows.filter(row => row.date.startsWith(period.period)).map(row => row[key]).filter(finite);
    const expected = month === null ? period.expected : new Date(Date.UTC(Number(period.period), month, 0)).getUTCDate();
    const complete = values.length === expected;
    return { period: period.period, value: !complete ? null : statistic === 'min' ? Math.min(...values) : statistic === 'max' ? Math.max(...values) : mean(values), valid: values.length, expected, status: complete ? 'complete' : 'incomplete' };
  });
}

export function monthlyAnomalies(rows, climatology, key) {
  const baseline = new Map(climatology.days.map(d => [d.month_day, d]));
  const field = { tmin_c: 'low_c', tmax_c: 'high_c', tmean_c: 'mean_c' }[key];
  const anomalies = rows.map(row => {
    const reference = baseline.get(row.date.slice(5))?.[field]?.p50;
    return { ...row, [key]: finite(row[key]) && finite(reference) ? row[key] - reference : null };
  });
  return aggregate(anomalies, key);
}

export function seasonalMetrics(rows, policy) {
  // Full calendars are essential: missing dates break runs and invalidate totals.
  const byDate = new Map(rows.map(r => [r.date, r]));
  const years = [...new Set(rows.map(r => Number(r.date.slice(0, 4))))];
  const window = (start, end) => {
    const values = [];
    for (let day = Date.parse(start + 'T00:00:00Z'); day <= Date.parse(end + 'T00:00:00Z'); day += 86400000) values.push(byDate.get(new Date(day).toISOString().slice(0, 10)));
    return values;
  };
  return years.map(year => {
    const growing = window(`${year}-${policy.gdd.start}`, `${year}-${policy.gdd.end}`);
    const winter = window(`${year - 1}-12-01`, `${year}-03-01`).slice(0, -1);
    const complete = (days, key) => days.length > 0 && days.every(r => r && finite(r[key]));
    let subtotal = 0, through = null;
    for (const row of growing) {
      if (!row || !finite(row.tmean_c)) break;
      subtotal += Math.max(0, Math.min(row.tmean_c, policy.gdd.cap_c ?? Infinity) - policy.gdd.base_c);
      through = row.date;
    }
    let run = 0, maxRun = 0, coldDays = 0;
    for (const row of winter) {
      if (row && finite(row.tmin_c) && row.tmin_c <= policy.cold_threshold_c) { run++; coldDays++; maxRun = Math.max(maxRun, run); } else run = 0;
    }
    const winterComplete = complete(winter, 'tmean_c') && complete(winter, 'tmin_c');
    return { year, gdd: complete(growing, 'tmean_c') ? subtotal : null, gdd_subtotal: through ? subtotal : null, through,
      gdd_valid: growing.filter(r => r && finite(r.tmean_c)).length, gdd_expected: growing.length,
      hdd: winterComplete ? winter.reduce((sum, r) => sum + Math.max(0, policy.hdd_base_c - r.tmean_c), 0) : null,
      cold_days: winterComplete ? coldDays : null, cold_run: winterComplete ? maxRun : null,
      winter_valid: winter.filter(r => r && finite(r.tmean_c) && finite(r.tmin_c)).length, winter_expected: winter.length };
  });
}

export function csv(rows, metadata) {
  if (!rows.length) return '';
  const result = rows.map(row => ({ ...metadata, ...row }));
  const keys = Object.keys(result[0]);
  const cell = value => `"${String(value ?? '').replaceAll('"', '""')}"`;
  return [keys.map(cell).join(','), ...result.map(row => keys.map(key => cell(row[key])).join(','))].join('\r\n') + '\r\n';
}

export function frostDistribution(records, name) {
  const counts = new Map();
  records.filter(r => r[name].status === 'observed').forEach(r => {
    const day = r[name].date.slice(5); counts.set(day, (counts.get(day) || 0) + 1);
  });
  return [...counts].sort(([a], [b]) => calendarDay(`2000-${a}`) - calendarDay(`2000-${b}`)).map(([day, count]) => ({ day, count }));
}
