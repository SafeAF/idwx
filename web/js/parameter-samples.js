// Original supplied values. Cadence does not define a variable's time interval.
import { dateParts } from './format.js';
export const SAMPLE_VERSION = 'idwx.parameter-samples.1';
const finite = value => typeof value === 'number' && Number.isFinite(value);
const utc = value => typeof value === 'string' && /^\d{4}-\d{2}-\d{2}T\d{2}:00:00Z$/.test(value) && Number.isFinite(Date.parse(value));
export function validateParameterCatalog(doc, manifest) {
  if (doc?.version !== SAMPLE_VERSION || doc.method !== 'source_samples_unchanged_v1' || doc.timestamp_basis !== 'UTC' || doc.sampling_interval_seconds !== 3600 || doc.data_kind !== manifest.data_kind || !/^[a-f0-9]{64}$/.test(doc.snapshot_id) || !Array.isArray(doc.metrics) || doc.metrics.length !== 27 || !Array.isArray(doc.stations) || !doc.stations.length) throw new Error('Invalid supplied-parameter catalog.');
  const ids = new Set();
  for (const metric of doc.metrics) {
    if (!metric || !/^[a-z0-9_]+$/.test(metric.id) || ids.has(metric.id) || typeof metric.label !== 'string' || typeof metric.unit !== 'string' || metric.unit.length > 80 || metric.value_interval !== null || metric.reducer !== 'none' || !finite(metric.screening_minimum) || !finite(metric.screening_maximum) || metric.screening_minimum > metric.screening_maximum) throw new Error('Invalid supplied-parameter metadata.');
    ids.add(metric.id);
  }
  const stations = new Set();
  for (const station of doc.stations) {
    if (!station || !/^[a-z0-9]+(?:_[a-z0-9]+)*$/.test(station.station_id) || stations.has(station.station_id) || typeof station.name !== 'string' || !utc(station.first_utc) || !utc(station.last_utc) || ![-7, -8].includes(station.reporting_utc_offset_hours) || !Array.isArray(station.months) || !station.months.length) throw new Error('Invalid parameter location.');
    stations.add(station.station_id);
    let next = Date.parse(station.first_utc);
    for (const month of station.months) {
      if (!/^\d{4}-\d{2}$/.test(month.month) || month.path !== `hourly/${station.station_id}/${month.month}.json` || !/^[a-f0-9]{64}$/.test(month.sha256) || !utc(month.start_utc) || Date.parse(month.start_utc) !== next || !Number.isInteger(month.count) || month.count < 1 || month.count > 744) throw new Error('Invalid parameter month index.');
      next += month.count * 3600000;
    }
    if (next !== Date.parse(station.last_utc) + 3600000) throw new Error('Invalid parameter coverage.');
  }
  return doc;
}
export function normalizeParameterSelection(params, station, metrics) {
  const first = station.first_utc.slice(0, 10), last = station.last_utc.slice(0, 10);
  const clamp = (value, fallback) => dateParts(value) ? value < first ? first : value > last ? last : value : fallback;
  const start = clamp(params.get('start'), last.slice(0, 7) + '-01');
  let end = clamp(params.get('end'), last);
  if (end < start) end = start;
  const maxEnd = new Date(Date.parse(start + 'T00:00:00Z') + 30 * 86400000).toISOString().slice(0, 10);
  const shortened = end > maxEnd;
  if (shortened) end = maxEnd;
  const requested = params.get('parameter');
  const matched = metrics.find(m => m.id === requested);
  if (requested && !matched) throw new Error('This saved parameter is unavailable in the loaded publication.');
  const metric = matched || metrics.find(m => m.id === 'soil_temperature_level_1') || metrics[0];
  const mark = metric.quantity === 'direction' ? 'scatter' : ['line', 'scatter'].includes(params.get('mark')) ? params.get('mark') : 'line';
  return { view: 'parameters', parameter: metric.id, start, end, mark, shortened };
}
export function validateParameterMonth(doc, entry, station, catalog) {
  if (doc?.version !== SAMPLE_VERSION || doc.data_kind !== catalog.data_kind || doc.station_id !== station.station_id || doc.month !== entry.month || doc.start_utc !== entry.start_utc || doc.sampling_interval_seconds !== 3600 || JSON.stringify(doc.metric_ids) !== JSON.stringify(catalog.metrics.map(m => m.id)) || !Array.isArray(doc.values) || doc.values.length !== entry.count || doc.values.some(row => !Array.isArray(row) || row.length !== catalog.metrics.length || row.some(v => v !== null && !finite(v)))) throw new Error('Invalid original-unit parameter samples.');
  const end = new Date(Date.parse(doc.start_utc) + (doc.values.length - 1) * 3600000).toISOString();
  if (doc.start_utc.slice(0, 7) !== doc.month || end.slice(0, 7) !== doc.month) throw new Error('Parameter samples cross their UTC month.');
  return doc;
}
export function parameterRows(chunks, selection, station, catalog) {
  const index = catalog.metrics.findIndex(m => m.id === selection.parameter), metric = catalog.metrics[index];
  const begin = Date.parse(selection.start + 'T00:00:00Z'), end = Date.parse(selection.end + 'T23:00:00Z');
  return chunks.flatMap(chunk => chunk.values.map((values, i) => {
    const stamp = Date.parse(chunk.start_utc) + i * 3600000, value = values[index];
    return { stamp, timestamp_utc: new Date(stamp).toISOString().replace('.000', ''),
      reporting_local_time: new Date(stamp + station.reporting_utc_offset_hours * 3600000).toISOString().replace('.000Z', ''),
      provided_value: value, provided_unit: metric.unit || 'not specified',
      quality: value === null ? 'missing' : value < metric.screening_minimum || value > metric.screening_maximum ? 'outside_screening_range_as_supplied' : 'provided' };
  })).filter(row => row.stamp >= begin && row.stamp <= end).sort((a, b) => a.stamp - b.stamp);
}
