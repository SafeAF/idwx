const ID = /^[a-z0-9]+(?:_[a-z0-9]+)*$/;
const RELEASE = /^[a-f0-9]{64}$/;
export const STUDIES_KEY = 'idwx.weather-studies.v2';
export const MAX_URL_LENGTH = 12000;

export function selection(value) {
  if (!value || typeof value !== 'object' || Array.isArray(value) || Object.keys(value).some(k => !['station', 'query'].includes(k)) || typeof value.station !== 'string' || !ID.test(value.station) || typeof value.query !== 'string' || value.query.length > 2500) throw new Error('Invalid graph selection.');
  return { station: value.station, query: new URLSearchParams(value.query).toString() };
}
export function workspace(value) {
  if (!value || value.version !== 2 || !Array.isArray(value.graphs) || value.graphs.length < 1 || value.graphs.length > 3 || Object.keys(value).some(k => !['version', 'graphs', 'links', 'layout'].includes(k))) throw new Error('A workspace needs one to three graphs.');
  const graphs = value.graphs.map(selection);
  const links = value.links ?? { dates: false, location: false, cursor: false };
  if (!links || typeof links !== 'object' || Array.isArray(links) || Object.keys(links).some(k => !['dates', 'location', 'cursor'].includes(k)) || Object.values(links).some(v => typeof v !== 'boolean')) throw new Error('Invalid workspace links.');
  if (!['focus', 'equal'].includes(value.layout ?? 'focus')) throw new Error('Invalid workspace layout.');
  return { version: 2, graphs, links: { dates: false, location: false, cursor: false, ...links }, layout: value.layout ?? 'focus' };
}
export function encodeWorkspace(value) { return JSON.stringify(workspace(value)); }
export function workspaceFromURL(url) {
  const encoded = url.searchParams.get('workspace');
  if (encoded === null) return null;
  if (encoded.length > MAX_URL_LENGTH) throw new Error('This workspace URL is too large. Import a study backup instead.');
  return workspace(JSON.parse(encoded));
}
export function study(value) {
  const keys = ['name', 'notes', 'saved_at', 'release', 'data_kind', 'workspace'];
  if (!value || Object.keys(value).length !== keys.length || keys.some(k => !(k in value)) || typeof value.name !== 'string' || !value.name.trim() || value.name.length > 120 || typeof value.notes !== 'string' || value.notes.length > 500 || !RELEASE.test(value.release) || !['historical_reanalysis', 'historical_observations', 'synthetic_demo'].includes(value.data_kind) || typeof value.saved_at !== 'string' || value.saved_at.length > 40 || !Number.isFinite(Date.parse(value.saved_at))) throw new Error('Invalid saved workspace.');
  return { ...value, workspace: workspace(value.workspace) };
}
export function parseWorkspaceStudies(text) {
  if (text === null) return [];
  if (typeof text !== 'string' || text.length > 1000000) throw new Error('Study backup is too large.');
  const value = JSON.parse(text);
  if (!value || Object.keys(value).length !== 2 || value.version !== 2 || !Array.isArray(value.studies) || value.studies.length > 50) throw new Error('Invalid workspace study backup.');
  return value.studies.map(study);
}
export function migrateCharts(charts) {
  return charts.map(chart => study({ name: `${chart.study}: ${chart.name}`.slice(0, 120), notes: chart.notes,
    release: chart.release, data_kind: chart.data_kind, saved_at: chart.saved_at,
    workspace: { version: 2, graphs: [{ station: chart.station, query: chart.query }] } }));
}
// Only daily-date views can receive dates; overlays and seasonal windows have
// their own calendar semantics and keep their independent configuration.
export function linkedSelection(target, source, links) {
  const to = new URLSearchParams(target.query), from = new URLSearchParams(source.query);
  const daily = params => ['temperature', 'table', 'heatmap'].includes(params.get('view') || 'temperature') || params.get('view') === 'precipitation' && ['daily', 'monthly', 'yearly'].includes(params.get('rain_mode') || 'monthly');
  let reason = '';
  if (links.dates) {
    if (daily(to) && daily(from) || to.get('view') === 'parameters' && from.get('view') === 'parameters') { for (const key of ['start', 'end']) if (from.has(key)) to.set(key, from.get(key)); }
    else reason = 'Date linking requires compatible calendars. UTC parameter samples, local daily timelines, year overlays and seasonal windows remain independent when mixed.';
  }
  return { value: { station: links.location ? source.station : target.station, query: to.toString() }, reason };
}
