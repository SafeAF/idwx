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
