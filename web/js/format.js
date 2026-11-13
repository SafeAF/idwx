export const EVENT_NAMES = ['last_spring_frost', 'first_fall_frost'];
export const STATUSES = {
  observed: 'Confirmed event', no_event_in_complete_window: 'No event in complete window',
  incomplete_window: 'Incomplete window', insufficient_data: 'Insufficient data',
  verified: 'Verified', missing_dependency: 'Unconfirmed event dates',
  invalid_event_order: 'Invalid event order', incomplete_interval: 'Missing days between events',
  intervening_freeze: 'Freeze between selected dates',
};
export const statusLabel = value => STATUSES[value] || 'Unknown status';

export function dateParts(value) {
  if (typeof value !== 'string' || !/^\d{4}-\d{2}-\d{2}$/.test(value)) return null;
  const [year, month, day] = value.split('-').map(Number);
  const date = new Date(Date.UTC(year, month - 1, day));
  return date.getUTCFullYear() === year && date.getUTCMonth() === month - 1 && date.getUTCDate() === day
    ? { year, month, day, date } : null;
}

export function displayDate(value, withYear = false) {
  const parts = dateParts(value);
  if (!parts) return '—';
  return new Intl.DateTimeFormat('en-US', {
    month: 'short', day: 'numeric', ...(withYear ? { year: 'numeric' } : {}), timeZone: 'UTC',
  }).format(parts.date);
}

export function calendarDay(value) {
  const parts = dateParts(value);
  return parts ? Math.round((Date.UTC(2000, parts.month - 1, parts.day) - Date.UTC(2000, 0, 1)) / 86400000) + 1 : null;
}

export function temperature(value, unit) {
  if (typeof value !== 'number' || !Number.isFinite(value)) return '—';
  const number = unit === 'f' ? value * 9 / 5 + 32 : value;
  return `${Number(number.toFixed(1))} °${unit.toUpperCase()}`;
}

export function historicalText(summary) {
  if (!summary || summary.status !== 'available') {
    return `Insufficient history (${summary?.valid_year_count ?? 0} confirmed years; ${summary?.minimum_valid_years ?? '—'} required).`;
  }
  return `Typical ${displayDate(`2000-${summary.p50.month_day}`)} · P10–P90 ${displayDate(`2000-${summary.p10.month_day}`)}–${displayDate(`2000-${summary.p90.month_day}`)} · ${summary.valid_year_count} years`;
}

export function yearInTimeZone(timeZone, now = new Date()) {
  try {
    const year = Number(new Intl.DateTimeFormat('en-US', {
      year: 'numeric', timeZone,
    }).format(now));
    return Number.isInteger(year) ? year : now.getUTCFullYear();
  } catch {
    return now.getUTCFullYear();
  }
}

export function climatologyProjection(summary, year) {
  const project = distribution => {
    if (!distribution || distribution.status !== 'available') return null;
    const points = ['p10', 'p50', 'p90'].map(key => distribution[key]?.month_day);
    const days = points.map(monthDay => calendarDay(`2000-${monthDay}`));
    if (days.some(day => day === null) || days[0] > days[1] || days[1] > days[2]) return null;
    return {
      p10: points[0], p50: points[1], p90: points[2],
      confirmedYears: distribution.valid_year_count,
      baseline: distribution.baseline_period,
    };
  };
  if (!Number.isInteger(year) || year < 1 || year > 9999) return null;
  const spring = project(summary?.last_spring_frost);
  const fall = project(summary?.first_fall_frost);
  const springDay = spring ? calendarDay(`2000-${spring.p50}`) : null;
  const fallDay = fall ? calendarDay(`2000-${fall.p50}`) : null;
  return {
    year, spring, fall,
    frostFreeDays: springDay !== null && fallDay !== null && fallDay > springDay
      ? fallDay - springDay - 1 : null,
  };
}

export function baselineDetails(records, eventName, summary) {
  const period = summary.baseline_period;
  const events = records.filter(row => row.season_year >= period.start_year && row.season_year <= period.end_year)
    .map(row => row[eventName]);
  const dates = events.filter(event => event.status === 'observed').map(event => event.date)
    .sort((a, b) => calendarDay(a) - calendarDay(b));
  return {
    total: events.length, confirmed: dates.length,
    noEvent: events.filter(event => event.status === 'no_event_in_complete_window').length,
    incomplete: events.filter(event => event.status === 'incomplete_window').length,
    unavailable: events.filter(event => event.status === 'insufficient_data').length,
    earliest: dates[0] ?? null, latest: dates.at(-1) ?? null,
  };
}

export function historyCSV(document, records) {
  const rows = records.map(record => {
    const row = { schema_version: document.schema_version, release_id: document.release_id,
      data_kind: document.data_kind, station_id: document.station.station_id,
      season_year: record.season_year, threshold_c: record.threshold_c };
    for (const name of EVENT_NAMES) for (const [key, value] of Object.entries(record[name])) row[`${name}_${key}`] = value;
    for (const [key, value] of Object.entries(record.interval)) row[`interval_${key}`] = value;
    return row;
  });
  if (!rows.length) return '';
  const quote = value => `"${String(value ?? '').replaceAll('"', '""')}"`;
  const keys = Object.keys(rows[0]);
  return [keys.map(quote).join(','), ...rows.map(row => keys.map(key => quote(row[key])).join(','))].join('\r\n') + '\r\n';
}

export function releaseURL(path, base, releaseID) {
  const root = new URL(base);
  const url = new URL(path, root);
  const prefix = new URL(`data/${releaseID}/`, root);
  if (url.origin !== root.origin || !url.pathname.startsWith(prefix.pathname) || url.search || url.hash) {
    throw new Error('Data reference is outside the selected release.');
  }
  return url;
}

export function validateRelease(document, manifest) {
  if (document.schema_version !== '1.0.0' || document.release_id !== manifest.release_id || document.data_kind !== manifest.data_kind) {
    throw new Error('The archive contains incompatible or mixed release data.');
  }
  return document;
}

export function validateStationDocument(doc, stationID) {
  const invalid = () => { throw new Error('Station data is invalid or incomplete.'); };
  const count = value => Number.isInteger(value) && value >= 0;
  const nullableNumber = value => value === null || Number.isFinite(value);
  const nullableDate = value => value === null || !!dateParts(value);
  if (!doc?.station || doc.station.station_id !== stationID || typeof doc.station.name !== 'string' ||
      typeof doc.station.timezone !== 'string' || typeof doc.methodology !== 'string' ||
      !doc.source_coverage || !dateParts(doc.source_coverage.first_date) || !dateParts(doc.source_coverage.last_date) ||
      !nullableDate(doc.source_coverage.last_valid_minimum_date) ||
      !Array.isArray(doc.thresholds_c) || !doc.thresholds_c.length || !doc.thresholds_c.every(Number.isFinite) ||
      new Set(doc.thresholds_c).size !== doc.thresholds_c.length ||
      !Array.isArray(doc.annual) || !doc.annual.length || !Array.isArray(doc.summaries)) invalid();
  const keys = new Set(), years = new Set();
  for (const row of doc.annual) {
    if (!Number.isInteger(row?.season_year) || !doc.thresholds_c.includes(row.threshold_c)) invalid();
    const key = `${row.season_year}/${row.threshold_c}`;
    if (keys.has(key)) invalid();
    keys.add(key); years.add(row.season_year);
    for (const name of EVENT_NAMES) {
      const event = row[name];
      if (!event || !['observed', 'no_event_in_complete_window', 'incomplete_window', 'insufficient_data'].includes(event.status) ||
          !nullableDate(event.date) || !nullableDate(event.candidate_date) || !nullableNumber(event.tmin_c) ||
          !nullableNumber(event.candidate_tmin_c) || !dateParts(event.window_start) || !dateParts(event.window_end) ||
          !count(event.valid_days) || !count(event.missing_days) || !count(event.expected_days) || !event.expected_days ||
          event.valid_days + event.missing_days !== event.expected_days || !Number.isFinite(event.coverage_fraction) ||
          Math.abs(event.coverage_fraction - event.valid_days / event.expected_days) > 1e-12 ||
          (event.status === 'observed' ? !event.date || !Number.isFinite(event.tmin_c) || event.missing_days !== 0
            : event.date !== null || event.tmin_c !== null)) invalid();
    }
    const interval = row.interval;
    if (!interval || !Object.hasOwn(STATUSES, interval.status) || !nullableNumber(interval.date_difference_days) ||
        !(interval.frost_free_days === null || count(interval.frost_free_days)) ||
        typeof interval.summer_freeze_observed !== 'boolean') invalid();
  }
  if (keys.size !== years.size * doc.thresholds_c.length) invalid();
  for (const threshold of doc.thresholds_c) {
    const summaries = doc.summaries.filter(item => item.threshold_c === threshold);
    if (summaries.length !== 1) invalid();
    for (const name of EVENT_NAMES) {
      const summary = summaries[0][name];
      if (!summary || !['available', 'insufficient_history'].includes(summary.status) ||
          !count(summary.valid_year_count) || !count(summary.minimum_valid_years) || !summary.minimum_valid_years ||
          !Number.isInteger(summary.baseline_period?.start_year) || !Number.isInteger(summary.baseline_period?.end_year)) invalid();
      for (const key of ['p10', 'p50', 'p90']) {
        const point = summary[key];
        if (summary.status === 'available' ? !point || !Number.isInteger(point.calendar_day) ||
            calendarDay(`2000-${point.month_day}`) !== point.calendar_day : point !== null) invalid();
      }
    }
  }
  return doc;
}
