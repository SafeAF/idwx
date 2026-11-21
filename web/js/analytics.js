import { calendarDay, dateParts } from './format.js';

export const METRICS = { tmin_c: 'Daily minimum', tmean_c: 'Daily mean', tmax_c: 'Daily maximum' };
export const finite = value => typeof value === 'number' && Number.isFinite(value);
export const toUnit = (value, unit, difference = false) => finite(value) ? (unit === 'f' ? value * 1.8 + (difference ? 0 : 32) : value) : null;
export const mean = values => values.length ? values.reduce((a, b) => a + b, 0) / values.length : null;
export const shown = value => finite(value) ? Number(value.toFixed(2)).toLocaleString('en-US') : '—';

export function validateChunk(doc, station, year) {
  const expected = new Date(Date.UTC(year + 1, 0, 1)) - new Date(Date.UTC(year, 0, 1));
  if (doc.station_id !== station || doc.year !== year || !Array.isArray(doc.days) || doc.days.length !== expected / 86400000) throw new Error('Invalid daily archive.');
  doc.days.forEach((row, i) => {
    const date = new Date(Date.UTC(year, 0, 1 + i)).toISOString().slice(0, 10);
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
  const defaultYear = sourceLastDate?.endsWith('-12-31') ? years.at(-1) : years.length > 1 ? years.at(-2) : years[0];
  const clamp = (value, fallback) => dateParts(value) ? (value < first ? first : value > last ? last : value) : fallback;
  const start = clamp(params.get('start'), `${defaultYear}-01-01`);
  let end = clamp(params.get('end'), `${defaultYear}-12-31`);
  if (end < start) end = start;
  const compare = [...new Set((params.get('compare') || String(defaultYear)).split(',').map(Number).filter(y => years.includes(y)))].slice(0, 3);
  const views = ['temperature', 'compare', 'frost', 'heatmap', 'seasonal', 'table'];
  return {
    view: views.includes(params.get('view')) ? params.get('view') : 'temperature', start, end,
    compare: compare.length ? compare : [defaultYear], unit: params.get('unit') === 'f' ? 'f' : 'c',
    threshold: params.has('threshold') && thresholds.includes(Number(params.get('threshold'))) ? Number(params.get('threshold')) : thresholds[0],
    metric: Object.hasOwn(METRICS, params.get('metric')) ? params.get('metric') : 'tmean_c',
    baseline: params.get('baseline') !== 'off',
  };
}

export function selectionYears(selection, available) {
  return selection.view === 'compare' ? selection.compare : available.filter(y => y >= Number(selection.start.slice(0, 4)) && y <= Number(selection.end.slice(0, 4)));
}

export function selectedRows(chunks, selection) {
  return chunks.flatMap(chunk => chunk.days).filter(row => selection.view === 'compare' || row.date >= selection.start && row.date <= selection.end).sort((a, b) => a.date.localeCompare(b.date));
}

export function statistics(rows) {
  const stats = { expected: rows.length };
  for (const key of Object.keys(METRICS)) {
    const values = rows.map(r => r[key]).filter(finite);
    stats[key] = { valid: values.length, mean: mean(values), min: values.length ? Math.min(...values) : null, max: values.length ? Math.max(...values) : null };
  }
  return stats;
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
