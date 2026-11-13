import { EVENT_NAMES, statusLabel, displayDate, temperature, historicalText, historyCSV, releaseURL, validateRelease, validateStationDocument, baselineDetails, planningProjectionYear, climatologyProjection } from './format.js';
import { renderHistory } from './charts.js';

const $ = id => document.getElementById(id);
const base = new URL('./', location.href);
const controls = ['station', 'threshold', 'year', 'units'];
const state = { manifest: null, index: null, document: null, threshold: 0, year: 0, unit: 'c', request: 0 };
let controller;

function node(tag, text, className) {
  const item = document.createElement(tag);
  if (text !== undefined) item.textContent = text;
  if (className) item.className = className;
  return item;
}
function options(id, items, selected) {
  $(id).replaceChildren(...items.map(([value, label]) => new Option(label, value)));
  $(id).value = String(selected);
}
function busy(message) {
  $('status').textContent = message;
  $('status').className = 'status';
  $('retry').hidden = true;
  $('dashboard').hidden = true;
  $('notice').hidden = true;
  for (const id of controls) $(id).disabled = true;
}
function failed(error) {
  $('status').className = 'status error';
  $('status').textContent = `The archive could not be loaded. ${error.message} Check that the website data has been exported, then try again.`;
  $('retry').hidden = false;
  // Keep station selection available after a station-specific failure.
  if (state.index) $('station').disabled = false;
}
async function fetchJSON(url, signal) {
  const response = await fetch(url, { signal, cache: 'no-cache' });
  if (!response.ok) throw new Error(`Data request returned HTTP ${response.status}.`);
  return response.json();
}
function updateURL() {
  const url = new URL(location.href);
  url.searchParams.set('station', state.document.station.station_id);
  url.searchParams.set('threshold', state.threshold);
  url.searchParams.set('year', state.year);
  url.searchParams.set('unit', state.unit);
  history.replaceState(null, '', url);
}
function card(title, event, summary, className) {
  const article = node('article', undefined, `card ${className}`);
  article.append(node('h3', title), node('p', 'Observed season · ' + state.year, 'kind'));
  const confirmed = event.status === 'observed';
  article.append(node('p', confirmed ? displayDate(event.date) : event.status === 'no_event_in_complete_window' ? 'No event found' : 'Not confirmed', `value ${confirmed ? '' : 'text-value'}`));
  article.append(node('p', confirmed ? `Daily minimum ${temperature(event.tmin_c, state.unit)} · ${statusLabel(event.status).toLowerCase()}` : `${statusLabel(event.status)}. ${event.candidate_date ? `Candidate ${displayDate(event.candidate_date)}; not a confirmed date.` : 'No confirmed date is available.'}`, 'context'));
  article.append(node('p', 'Historical · ' + historicalText(summary), 'historical'));
  return article;
}
function projectionCard(title, projection, className, projectionYear) {
  const article = node('article', undefined, `projection-card ${className}`);
  article.append(node('h3', title));
  if (!projection) {
    article.append(node('p', 'Not available', 'projection-value text-value'));
    article.append(node('p', 'The historical baseline does not contain enough confirmed events.', 'projection-context'));
    return article;
  }
  article.append(node('p', displayDate(`${projectionYear}-${projection.p50}`, true), 'projection-value'));
  article.append(node('p', `Historical P10–P90 · ${displayDate(`${projectionYear}-${projection.p10}`, true)}–${displayDate(`${projectionYear}-${projection.p90}`, true)}`, 'projection-range'));
  article.append(node('p', `${projection.confirmedYears} confirmed years · baseline ${projection.baseline.start_year}–${projection.baseline.end_year}`, 'projection-context'));
  return article;
}
function eventCell(event) {
  const cell = node('td', event.date ? displayDate(event.date) : '—');
  cell.append(node('span', statusLabel(event.status), 'cell-status'));
  return cell;
}
function render() {
  const doc = state.document;
  const records = doc.annual.filter(r => r.threshold_c === state.threshold).sort((a, b) => a.season_year - b.season_year);
  const selected = records.find(r => r.season_year === state.year);
  if (!selected) throw new Error('The selected season is missing from this release.');
  const summary = doc.summaries.find(s => s.threshold_c === state.threshold);
  if (!summary) throw new Error('Historical summaries are missing from this release.');
  const projectionYear = planningProjectionYear(doc);
  const projection = climatologyProjection(summary, projectionYear);
  if (!projection) throw new Error('The historical projection could not be calculated.');
  options('threshold', doc.thresholds_c.map(t => [t, `${temperature(t, state.unit)} or colder`]), state.threshold);
  options('year', [...records].reverse().map(r => [r.season_year, r.season_year]), state.year);
  document.querySelector(`input[name="unit"][value="${state.unit}"]`).checked = true;
  $('station-title').textContent = doc.station.name;
  const coordinates = Number.isFinite(doc.station.lat) && Number.isFinite(doc.station.lon) ? `${doc.station.lat.toFixed(3)}°, ${doc.station.lon.toFixed(3)}° · ` : '';
  $('station-meta').textContent = `${coordinates}${doc.station.timezone} · Source records ${doc.source_coverage.first_date} to ${doc.source_coverage.last_date}`;
  $('year-badge').textContent = `Archive view · ${state.year}`;
  $('projection-title').textContent = `${projectionYear} frost projection`;
  $('projection-copy').textContent = `Historical climatology for ${doc.station.name} at ${temperature(state.threshold, state.unit)} or colder. The projected date is the historical median; the range shows P10–P90 of confirmed events.`;
  const springProjection = projectionCard('Last spring freeze', projection.spring, 'spring-projection', projectionYear);
  const fallProjection = projectionCard('First fall freeze', projection.fall, 'fall-projection', projectionYear);
  const spanProjection = node('article', undefined, 'projection-card span-projection');
  spanProjection.append(node('h3', 'Projected frost-free span'));
  spanProjection.append(node('p', projection.frostFreeDays === null ? 'Not available' : `About ${projection.frostFreeDays} days`, `projection-value ${projection.frostFreeDays === null ? 'text-value' : ''}`));
  spanProjection.append(node('p', 'Median spring date to median fall date, excluding both endpoints.', 'projection-range'));
  spanProjection.append(node('p', 'Calendar estimate only · daily conditions can shorten the actual season', 'projection-context'));
  $('projection-cards').replaceChildren(springProjection, fallProjection, spanProjection);
  const spring = card('Last spring freeze', selected.last_spring_frost, summary.last_spring_frost, 'spring-card');
  const fall = card('First fall freeze', selected.first_fall_frost, summary.first_fall_frost, 'fall-card');
  const interval = node('article', undefined, 'card interval-card');
  interval.append(node('h3', 'Frost-free days'), node('p', 'Verified interval · ' + state.year, 'kind'));
  const days = selected.interval.frost_free_days;
  interval.append(node('p', days === null ? 'Not verified' : `${days} days`, `value ${days === null ? 'text-value' : ''}`));
  interval.append(node('p', days === null ? statusLabel(selected.interval.status) + '.' : 'Strictly between the two freeze dates, with valid daily minima above the threshold.', 'context'));
  interval.append(node('p', selected.interval.summer_freeze_observed ? 'July freeze observed at this threshold.' : 'A season interval requires complete daily coverage.', 'historical'));
  $('cards').replaceChildren(spring, fall, interval);
  $('baseline-caption').textContent = `${doc.station.name}: historical baseline at ${temperature(state.threshold, state.unit)}. Counts apply separately to each seasonal window.`;
  $('baseline-rows').replaceChildren(...EVENT_NAMES.map((name, index) => {
    const distribution = summary[name];
    const details = baselineDetails(records, name, distribution);
    const row = node('tr');
    const window = node('th', index === 0 ? 'Spring' : 'Fall'); window.scope = 'row';
    window.append(node('span', `${distribution.baseline_period.start_year}–${distribution.baseline_period.end_year}`, 'cell-status'));
    const dates = node('td', distribution.status === 'available' ? `Typical ${displayDate(`2000-${distribution.p50.month_day}`)}` : 'Insufficient history');
    dates.append(node('span', distribution.status === 'available'
      ? `P10–P90 ${displayDate(`2000-${distribution.p10.month_day}`)}–${displayDate(`2000-${distribution.p90.month_day}`)}`
      : `${distribution.minimum_valid_years} confirmed years required`, 'cell-status'));
    const extremes = node('td', details.earliest ? `${displayDate(details.earliest)}–${displayDate(details.latest)}` : 'No confirmed dates');
    extremes.append(node('span', 'Earliest to latest month/day', 'cell-status'));
    const included = node('td', `${details.confirmed}/${details.total} years`);
    included.append(node('span', 'Confirmed events used for date statistics', 'cell-status'));
    const excluded = node('td', `${details.noEvent} complete without an event`);
    excluded.append(node('span', `${details.incomplete} incomplete · ${details.unavailable} without valid minima`, 'cell-status'));
    row.append(window, dates, extremes, included, excluded);
    return row;
  }));
  renderHistory($('chart'), records, summary, state.year);
  $('history-caption').textContent = `${records.length} configured years · ${temperature(state.threshold, state.unit)} threshold · Missing and provisional events are omitted from the chart and retained in the table.`;
  $('table-caption').textContent = `${doc.station.name}: all configured years at ${temperature(state.threshold, state.unit)}. Selected year ${state.year}.`;
  $('history-rows').replaceChildren(...[...records].reverse().map(record => {
    const row = node('tr', undefined, record.season_year === state.year ? 'current' : '');
    const year = node('th', record.season_year); year.scope = 'row';
    row.append(year, eventCell(record.last_spring_frost), eventCell(record.first_fall_frost));
    const days = node('td', record.interval.frost_free_days === null ? '—' : record.interval.frost_free_days);
    days.append(node('span', statusLabel(record.interval.status), 'cell-status'));
    row.append(days);
    return row;
  }));
  $('detail-caption').textContent = `${state.year} coverage and provisional observations; candidates are excluded from confirmed dates and historical summaries.`;
  $('detail-rows').replaceChildren(...EVENT_NAMES.map((name, index) => {
    const event = selected[name];
    const row = node('tr');
    const window = node('th', index === 0 ? 'Spring' : 'Fall'); window.scope = 'row';
    window.append(node('span', `${displayDate(event.window_start)}–${displayDate(event.window_end)}`, 'cell-status'));
    row.append(window, eventCell(event), node('td', temperature(event.tmin_c, state.unit)),
      node('td', `${event.valid_days}/${event.expected_days} valid days · ${event.missing_days} missing`),
      node('td', event.status !== 'observed' && event.candidate_date ? `${displayDate(event.candidate_date)} · ${temperature(event.candidate_tmin_c, state.unit)}` : '—'));
    return row;
  }));
  $('interval-note').textContent = `Date difference between confirmed fall and spring events: ${selected.interval.date_difference_days === null ? 'unavailable' : selected.interval.date_difference_days + ' days'}. This includes an endpoint and does not establish an uninterrupted frost-free season. ${selected.interval.summer_freeze_observed ? 'A July freeze was observed.' : 'No July freeze was found; this alone does not establish July coverage.'}`;
  $('method-text').textContent = doc.methodology;
  const s = summary.last_spring_frost, f = summary.first_fall_frost;
  $('summary-note').textContent = `Baseline ${s.baseline_period.start_year}–${s.baseline_period.end_year}. Confirmed years: spring ${s.valid_year_count}, fall ${f.valid_year_count}. Quantiles require ${s.minimum_valid_years} confirmed years and use nearest rank on a shared leap-year calendar. Complete windows without an event and incomplete years are excluded from date statistics.`;
  const facts = [
    ['Last valid minimum', doc.source_coverage.last_valid_minimum_date ?? 'Unavailable'],
    ['Archive generated', new Date(state.manifest.generated_at_utc).toLocaleString()],
    ['Data type', doc.data_kind === 'synthetic_demo' ? 'Synthetic demonstration'
      : doc.data_kind === 'historical_reanalysis' ? 'Historical reanalysis' : 'Historical source records'],
  ];
  $('freshness').replaceChildren(...facts.flatMap(([key, value]) => [node('dt', key), node('dd', value)]));
  $('full-download').href = releaseURL(state.manifest.history_csv_url, base, state.manifest.release_id);
  const lastValid = doc.source_coverage.last_valid_minimum_date;
  const archiveAge = lastValid ? (Date.now() - Date.parse(lastValid + 'T00:00:00Z')) / 86400000 : Infinity;
  $('notice').hidden = false;
  $('notice').textContent = doc.data_kind === 'synthetic_demo'
    ? 'Synthetic demonstration · These dates and the planning-year projection are invented examples. Do not use them for planting decisions.'
    : archiveAge > 730 ? `${doc.data_kind === 'historical_reanalysis' ? 'Historical reanalysis archive' : 'Historical archive'} · The last valid minimum is ${lastValid ?? 'unavailable'}. The ${projectionYear} projection is historical climatology only and does not use current conditions.`
    : `Historical records · The ${projectionYear} projection is historical climatology only, not a live weather forecast.`;
  $('dashboard').hidden = false;
  for (const id of controls) $(id).disabled = false;
  $('status').textContent = `${doc.station.name}, ${state.year}, ${temperature(state.threshold, state.unit)} threshold loaded.`;
  updateURL();
}

async function selectStation(id, params = new URLSearchParams(location.search)) {
  controller?.abort();
  controller = new AbortController();
  const request = ++state.request;
  busy('Loading station history…');
  try {
    const entry = state.index.stations.find(s => s.station.station_id === id);
    if (!entry) throw new Error('Station not found in this release.');
    const doc = validateRelease(await fetchJSON(releaseURL(entry.url, base, state.manifest.release_id), controller.signal), state.manifest);
    if (request !== state.request) return;
    validateStationDocument(doc, id);
    state.document = doc;
    const threshold = params.has('threshold') ? Number(params.get('threshold')) : state.threshold;
    state.threshold = doc.thresholds_c.includes(threshold) ? threshold : doc.thresholds_c[0];
    const years = doc.annual.filter(r => r.threshold_c === state.threshold).map(r => r.season_year);
    const year = Number(params.get('year'));
    state.year = years.includes(year) ? year : Math.max(...years);
    state.unit = params.get('unit') === 'f' ? 'f' : 'c';
    $('station').value = id;
    render();
  } catch (error) {
    if (error.name !== 'AbortError' && request === state.request) failed(error);
  }
}

async function load() {
  busy('Loading the historical archive…');
  try {
    const manifest = await fetchJSON(new URL('data/manifest.json', base));
    if (manifest.schema_version !== '1.0.0' || !/^[a-f0-9]{64}$/.test(manifest.release_id)) throw new Error('Unsupported archive schema or release.');
    const index = validateRelease(await fetchJSON(releaseURL(manifest.stations_url, base, manifest.release_id)), manifest);
    if (!index.stations?.length) throw new Error('No stations are available.');
    state.manifest = manifest;
    state.index = index;
    const params = new URLSearchParams(location.search);
    const id = index.stations.some(s => s.station.station_id === params.get('station')) ? params.get('station') : index.stations[0].station.station_id;
    options('station', index.stations.map(s => [s.station.station_id, s.station.name]), id);
    await selectStation(id, params);
  } catch (error) { failed(error); }
}

$('station').addEventListener('change', () => selectStation($('station').value));
$('threshold').addEventListener('change', () => { state.threshold = Number($('threshold').value); render(); });
$('year').addEventListener('change', () => { state.year = Number($('year').value); render(); });
$('units').addEventListener('change', event => { state.unit = event.target.value; render(); });
$('retry').addEventListener('click', load);
window.addEventListener('popstate', () => {
  if (!state.index) return;
  const id = new URLSearchParams(location.search).get('station');
  selectStation(state.index.stations.some(entry => entry.station.station_id === id) ? id : state.index.stations[0].station.station_id);
});
$('download').addEventListener('click', () => {
  const csv = historyCSV(state.document, state.document.annual.filter(r => r.threshold_c === state.threshold));
  const url = URL.createObjectURL(new Blob([csv], { type: 'text/csv;charset=utf-8' }));
  const link = node('a');
  link.href = url;
  link.download = `${state.document.station.station_id}-${state.threshold}C-frost-history.csv`;
  link.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
});
load();
