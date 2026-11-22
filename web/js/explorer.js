import { EVENT_NAMES, calendarDay, releaseURL, statusLabel, temperature, validateRelease, validateStationDocument } from './format.js';
import { METRICS, aggregate, calendarRecords, csv, finite, frostDistribution, monthlyAnomalies, normalizeSelection, parseStudies, periodStatistics, seasonalMetrics, selectedRows, selectionYears, shown, statistics, toUnit, validateChunk } from './analytics.js';
import { renderHistory } from './charts.js';
import { lineChart, temperatureSeries } from './explorer-charts.js';
import { cumulativePrecipitation, growingAssessment, precipitationPeriods, precipitationStats, precipToUnit } from './growing.js';

export function createExplorer(root, options = {}) {
const elements = new Map([...root.querySelectorAll('[id]')].map(el => [el.id, el]));
const $ = id => elements.get(id);
// Prefix cloned DOM IDs while preserving local lookup and label relationships.
if (options.idPrefix) {
  for (const [id, el] of elements) el.id = options.idPrefix + id;
  for (const el of root.querySelectorAll('[for], [aria-labelledby], [aria-describedby], a[href^="#"]')) {
    for (const attr of ['for', 'aria-labelledby', 'aria-describedby']) if (el.hasAttribute(attr)) el.setAttribute(attr, el.getAttribute(attr).split(' ').map(id => options.idPrefix + id).join(' '));
    if (el.getAttribute('href')?.startsWith('#')) el.setAttribute('href', '#' + options.idPrefix + el.getAttribute('href').slice(1));
  }
}
const base = new URL('.', location.href);
const state = { manifest: null, stations: null, doc: null, index: null, summary: null, selection: null, rows: [], table: [], columns: [], page: 0, request: 0, cache: new Map() };
let controller;
const node = (tag, text, className) => { const n = document.createElement(tag); if (text !== undefined) n.textContent = text; if (className) n.className = className; return n; };
const option = (value, label) => { const n = node('option', label); n.value = value; return n; };
const inspect = text => { $('ex-inspection').textContent = text; };
const baseline = () => state.doc.temperature_climatology.baseline_period;
const years = () => state.index.years.map(y => y.year);
const studiesKey = 'idwx.weather-studies.v1';
let studies = [], studiesAvailable = true;

function saveStudies(charts) {
  try {
    const text = JSON.stringify({ version: 1, charts });
    parseStudies(text);
    localStorage.setItem(studiesKey, text);
    studies = charts; renderStudies();
    return true;
  } catch {
    $('studies-status').textContent = 'Could not save studies. Browser storage may be unavailable or full; at most 50 charts are supported. Your previous studies are preserved.';
    return false;
  }
}
function renderStudies() {
  const list = $('studies-list'); list.replaceChildren();
  $('studies-download').disabled = !studies.length;
  if (!studies.length) { list.append(node('p', 'No saved charts yet. Display a chart, give it a name, then add it to a study.', 'muted')); return; }
  const groups = new Map();
  studies.forEach((chart, index) => {
    if (!groups.has(chart.study)) {
      const group = node('section', undefined, 'saved-study'); group.append(node('h3', chart.study)); groups.set(chart.study, group); list.append(group);
    }
    const card = node('article', undefined, 'saved-chart'), open = node('a', 'Open chart'), remove = node('button', 'Remove chart', 'button');
    const url = new URL('explore.html', base); url.search = chart.query; url.searchParams.set('station', chart.station); open.href = url.href;
    card.append(node('h4', chart.name), node('p', chart.notes), node('p', `${chart.station} · ${chart.data_kind} · saved ${chart.saved_at.slice(0, 10)} · release ${chart.release.slice(0, 12)}${state.manifest ? chart.release === state.manifest.release_id ? ' · matches loaded release' : ' · data changed; opening uses loaded release' : ''}`, 'small muted'));
    open.onclick = event => {
      event.preventDefault();
      if (!state.stations?.stations.some(s => s.station.station_id === chart.station)) { $('studies-status').textContent = 'This saved station is unavailable in the loaded release.'; return; }
      $('study-name').value = chart.study; $('study-chart-name').value = chart.name; $('study-notes').value = chart.notes;
      $('studies-status').textContent = chart.release === state.manifest.release_id ? 'Opening the saved selection with the same data release.' : 'Data release changed. Recalculating the saved filters with the loaded release; saved results may differ.';
      openSelection(chart.station, new URLSearchParams(chart.query));
      $('explorer-filters').scrollIntoView?.({ block: 'start' });
    };
    remove.type = 'button'; remove.disabled = !studiesAvailable;
    remove.setAttribute('aria-label', `Remove ${chart.name} from ${chart.study}`);
    remove.onclick = () => { if (saveStudies(studies.filter((_, i) => i !== index))) $('studies-status').textContent = 'Chart removed from this browser.'; };
    const actions = node('div', undefined, 'study-actions'); actions.append(open, remove); card.append(actions); groups.get(chart.study).append(card);
  });
}

async function fetchJSON(url, signal, cache = true) {
  if (options.client) {
    const value = await options.client.json(url, { cache });
    if (signal.aborted) throw new DOMException('Selection changed', 'AbortError');
    return value;
  }
  const key = url.href;
  if (cache && state.cache.has(key)) return state.cache.get(key);
  const response = await fetch(url, { signal, cache: cache ? 'default' : 'no-cache' });
  if (!response.ok) throw new Error(`Archive request failed (${response.status}). Retry this selection.`);
  const doc = await response.json();
  if (signal.aborted) throw new DOMException('Selection changed', 'AbortError');
  if (cache) state.cache.set(key, doc);
  return doc;
}
const releaseDocument = async (path, signal) => validateRelease(await fetchJSON(releaseURL(path, base, state.manifest.release_id), signal), state.manifest);

function visibility() {
  const view = $('ex-view').value;
  const custom = ['annual', 'month-compare', 'monthly'].includes(view);
  const rain = view === 'precipitation';
  const rainCompare = rain && ['compare', 'cumulative'].includes($('ex-rain_mode').value);
  $('rain-mode-control').hidden = $('rain-unit-control').hidden = !rain;
  $('requirements-control').hidden = view !== 'assessment';
  $('compare-control').hidden = view !== 'compare' && !rainCompare; $('threshold-control').hidden = !['frost', 'assessment'].includes(view);
  $('profile-year-control').hidden = view !== 'monthly'; $('month-control').hidden = view !== 'month-compare';
  $('statistic-control').hidden = !custom;
  $('mark-control').hidden = !['temperature', 'compare', 'annual', 'month-compare', 'monthly', 'calendar', 'precipitation', 'assessment'].includes(view);
  $('ex-start').disabled = $('ex-end').disabled = (['compare', 'monthly'].includes(view) || rainCompare);
  $('ex-metric').disabled = ['calendar', 'frost', 'seasonal', 'precipitation', 'assessment'].includes(view);
  $('ex-unit').disabled = rain;
  $('ex-baseline').disabled = !['temperature', 'compare'].includes(view);
}
function syncControls() {
  const s = state.selection;
  for (const key of ['view', 'start', 'end', 'unit', 'metric', 'threshold', 'month', 'statistic', 'mark', 'profile_year', 'rain_mode', 'rain_unit', 'need_gdd', 'need_days', 'need_precip']) $('ex-' + key).value = s[key];
  $('ex-compare').value = s.compare.join(','); $('ex-baseline').checked = s.baseline; $('ex-need_water').checked = s.need_water;
  for (let i = 1; i <= 3; i++) $('ex-year-' + i).value = s.compare[i - 1] ?? '';
  root.querySelectorAll('.explorer-shortcuts a').forEach(a => { if (new URL(a.href).searchParams.get('view') === s.view) a.setAttribute('aria-current', 'page'); else a.removeAttribute('aria-current'); });
  visibility();
}
function writeURL() {
  if (options.onChange) { options.onChange(snapshot()); return; }
  const url = new URL(location.href), s = state.selection;
  url.searchParams.set('station', state.doc.station.station_id);
  for (const key of ['view', 'start', 'end', 'unit', 'metric', 'threshold', 'month', 'statistic', 'mark', 'profile_year', 'rain_mode', 'rain_unit', 'need_gdd', 'need_days', 'need_precip']) url.searchParams.set(key, s[key]);
  url.searchParams.set('compare', s.compare.join(',')); url.searchParams.set('baseline', s.baseline ? 'on' : 'off'); url.searchParams.set('need_water', s.need_water ? 'on' : 'off');
  if (url.href !== location.href) history.pushState(null, '', url);
}
function paramsFromForm() {
  const params = new URLSearchParams();
  for (const key of ['view', 'start', 'end', 'unit', 'metric', 'threshold', 'compare', 'month', 'statistic', 'mark', 'profile_year', 'rain_mode', 'rain_unit', 'need_gdd', 'need_days', 'need_precip']) params.set(key, $('ex-' + key).value);
  params.set('baseline', $('ex-baseline').checked ? 'on' : 'off');
  params.set('need_water', $('ex-need_water').checked ? 'on' : 'off');
  return params;
}
function showError(error, request) {
  if (error.name === 'AbortError' || request !== state.request) return;
  state.cache.clear();
  options.client?.clear();
  $('explorer-status').textContent = error.message;
  $('explorer-results').hidden = true; $('explorer-retry').hidden = false;
}
async function openSelection(stationID, params, push = true) {
  state.pending = { stationID, params: new URLSearchParams(params), push };
  controller?.abort(); controller = new AbortController();
  const signal = controller.signal, request = ++state.request;
  $('explorer-status').textContent = 'Loading selected historical data…'; $('explorer-results').hidden = true; $('explorer-retry').hidden = true;
  $('ex-download').disabled = true; $('study-save').disabled = true;
  try {
    const found = state.stations.stations.find(s => s.station.station_id === stationID);
    if (options.strictStation && stationID && !found) throw new Error('This saved location is unavailable in the loaded release. Edit the graph to select an available location.');
    const entry = found || state.stations.stations[0];
    const doc = validateStationDocument(await releaseDocument(entry.url, signal), entry.station.station_id);
    if (!doc.analytics_url) throw new Error('This release has no daily explorer data.');
    const index = await releaseDocument(doc.analytics_url, signal);
    if (index.station_id !== entry.station.station_id || !Array.isArray(index.years) || !index.years.length || index.years.some((r, i) => !Number.isInteger(r.year) || r.year !== Number(doc.source_coverage.first_date.slice(0, 4)) + i)) throw new Error('Invalid analytics index.');
    const normalized = new URLSearchParams(params);
    if (params.get('view') === 'precipitation' && !params.has('mark')) normalized.set('mark', ['compare', 'cumulative'].includes(params.get('rain_mode')) ? 'line' : 'bar');
    if (params.get('view') === 'assessment' && !params.has('start')) normalized.set('start', `${doc.season_years.start_year}-01-01`);
    const coverageDate = params.get('view') === 'precipitation' ? index.precipitation_method?.last_complete_date : doc.source_coverage.last_date;
    const selection = normalizeSelection(normalized, index.years.map(y => y.year), doc.thresholds_c, coverageDate);
    if (selection.view === 'precipitation' && !index.precipitation_method) throw new Error('This release has no reviewed precipitation data.');
    // Do not overwrite the active view until all files for its selection arrive.
    const needed = selectionYears(selection, index.years.map(y => y.year));
    const chunks = await Promise.all(needed.map(async year => validateChunk(await releaseDocument(index.years.find(y => y.year === year).url, signal), entry.station.station_id, year)));
    const summary = ['seasonal', 'assessment'].includes(selection.view) ? await releaseDocument(index.summary_url, signal) : null;
    if (summary && (summary.station_id !== entry.station.station_id || !summary.policy)) throw new Error('Invalid seasonal analytics.');
    if (request !== state.request || signal.aborted) return;
    Object.assign(state, { doc, index, selection, summary, rows: selectedRows(chunks, selection), page: 0 });
    $('ex-station').value = entry.station.station_id;
    for (let i = 1; i <= 3; i++) $('ex-year-' + i).replaceChildren(...(i > 1 ? [option('', 'None')] : []), ...[...years()].reverse().map(y => option(y, y)));
    $('ex-profile_year').replaceChildren(...[...years()].reverse().map(y => option(y, y)));

    $('ex-threshold').replaceChildren(...doc.thresholds_c.map(t => option(t, temperature(t, selection.unit) + ' or colder')));
    for (const id of ['ex-start', 'ex-end']) { $(id).min = `${years()[0]}-01-01`; $(id).max = `${years().at(-1)}-12-31`; }
    syncControls();
    render();
    if (push) writeURL();
    $('explorer-results').hidden = false; $('ex-download').disabled = !state.table.length; $('study-save').disabled = !studiesAvailable; renderStudies();
    $('explorer-status').textContent = `${doc.station.name}: ${(selection.view === 'compare' || selection.view === 'precipitation' && ['compare', 'cumulative'].includes(selection.rain_mode)) ? selection.compare.join(', ') : selection.start + ' through ' + selection.end} loaded.`;
  } catch (error) { showError(error, request); }
}

function card(title, value, detail) {
  const n = node('article', undefined, 'explorer-stat'); n.append(node('h3', title), node('strong', value), node('p', detail)); return n;
}
function renderStatistics() {
  const stats = statistics(state.rows), s = state.selection;
  $('ex-statistics').replaceChildren(
    card('Daily mean coverage', `${stats.tmean_c.valid}/${stats.expected}`, 'Valid / selected calendar days'),
    card('Mean of available daily means', temperature(stats.tmean_c.mean, s.unit), 'Incomplete periods remain partial'),
    card('Lowest daily minimum', temperature(stats.tmin_c.min, s.unit), `${stats.tmin_c.valid} valid minimum values`),
    card('Highest daily maximum', temperature(stats.tmax_c.max, s.unit), `${stats.tmax_c.valid} valid maximum values`),
  );
}
function table(columns, rows, note) {
  state.columns = columns; state.table = rows; state.page = 0;
  $('ex-table-note').textContent = note;
  $('ex-table-caption').textContent = `${state.doc.station.name} · ${$('ex-title').textContent} · release ${state.manifest.release_id.slice(0, 12)}`;
  const tr = node('tr');
  columns.forEach(([, label]) => { const th = node('th', label); th.scope = 'col'; tr.append(th); });
  $('ex-table-head').replaceChildren(tr); renderTable();
}
function renderTable() {
  const count = 100, start = state.page * count;
  $('ex-table-body').replaceChildren(...state.table.slice(start, start + count).map(row => {
    const tr = node('tr'); state.columns.forEach(([key, , format]) => tr.append(node('td', format ? format(row[key], row) : row[key] ?? '—'))); return tr;
  }));
  $('ex-page').textContent = state.table.length ? `${start + 1}–${Math.min(start + count, state.table.length)} of ${state.table.length} rows` : 'No matching rows';
  $('ex-previous').disabled = state.page === 0; $('ex-next').disabled = start + count >= state.table.length;
}
function renderDaily() {
  const s = state.selection, climate = new Map(state.doc.temperature_climatology.days.map(d => [d.month_day, d]));
  const field = { tmin_c: 'low_c', tmean_c: 'mean_c', tmax_c: 'high_c' }[s.metric];
  const rows = state.rows.map(r => {
    const reference = climate.get(r.date.slice(5))?.[field]?.p50;
    return { ...r, ...Object.fromEntries(Object.keys(METRICS).map(key => [key + '_display', toUnit(r[key], s.unit)])),
      historical_p10_c: climate.get(r.date.slice(5))?.[field]?.p10 ?? null, historical_median_c: reference ?? null, historical_p90_c: climate.get(r.date.slice(5))?.[field]?.p90 ?? null, departure_c: finite(r[s.metric]) && finite(reference) ? r[s.metric] - reference : null };
  });
  table([['date', 'Local date'], ...Object.keys(METRICS).map(key => [key + '_display', `${METRICS[key]} (°${s.unit.toUpperCase()})`, shown]),
    ...[['historical_p10_c', 'Reference P10'], ['historical_median_c', 'Reference P50'], ['historical_p90_c', 'Reference P90']].map(([key, label]) => [key, `${label} (°${s.unit.toUpperCase()})`, v => shown(toUnit(v, s.unit))]),
    ['departure_c', `${METRICS[s.metric]} departure (Δ°${s.unit.toUpperCase()})`, v => shown(toUnit(v, s.unit, true))], ['status', 'Temperature status'], ['n_obs', 'Source hours']], rows,
  'All selected daily rows, including missing values. CSV includes canonical Celsius, display values, filters, and baseline identity. Source hours may be unavailable for legacy inputs.');
}
function renderTemperature() {
  const s = state.selection;
  $('ex-title').textContent = s.view === 'compare' ? 'Years, side by side' : s.view === 'table' ? 'The daily record' : 'Temperature through time';
  const { series, long, frequency } = temperatureSeries(state.rows, s, state.doc.temperature_climatology);
  $('ex-description').textContent = s.view === 'compare' ? 'Full calendar years aligned by month/day. February 29 remains distinct. Partial years retain their gaps.' : long ? `Long selections show ${frequency} means only for complete periods. Select a shorter date range for daily detail. Historical overlays are available in daily views.` : 'Daily minimum, mean, and maximum; gaps remain open. The selected temperature series is emphasized. Historical reference lines describe the fixed baseline.';
  $('ex-legend').textContent = s.view === 'compare' ? s.compare.map((y, i) => `${y} (${(s.mark === 'scatter' ? ['circles', 'squares', 'diamonds'] : s.mark === 'bar' ? s.compare.length === 2 ? ['left bar', 'right bar'] : ['left bar', 'middle bar', 'right bar'] : ['solid', 'dashed', 'dotted'])[i]})`).join(' · ') : s.mark === 'scatter' ? 'Minimum: green circles · Mean: rust squares · Maximum: blue diamonds · Reference quantiles: gray' : s.mark === 'bar' ? 'Minimum: left green bars · Mean: middle rust bars · Maximum: right blue bars · Reference quantiles: gray' : 'Minimum: solid green · Mean: dashed rust · Maximum: dotted blue · Reference quantiles: gray';
  if (s.view !== 'table') lineChart($('ex-chart'), series, { title: $('ex-title').textContent, unit: `°${s.unit.toUpperCase()}`, inspect, mark: s.mark });
  else $('ex-chart').textContent = 'The selected daily record is available in the table below.';
  if (long && s.view !== 'table') {
    const grouped = Object.fromEntries(Object.keys(METRICS).map(key => [key, aggregate(state.rows, key, frequency)]));
    const rows = grouped.tmean_c.map((p, i) => ({ period: p.period, expected_days: p.expected,
      ...Object.fromEntries(Object.keys(METRICS).flatMap(key => [[key, grouped[key][i].value], [key + '_valid', grouped[key][i].valid]])) }));
    table([['period', frequency === 'year' ? 'Year' : 'Month'], ...Object.keys(METRICS).flatMap(key => [[key, `${METRICS[key]} mean (°${s.unit.toUpperCase()})`, v => shown(toUnit(v, s.unit))], [key + '_valid', 'Valid days']]), ['expected_days', 'Expected days']], rows,
      'Chart and table show complete-period means; partial periods remain unavailable. CSV uses canonical Celsius. Choose Daily data table for individual dates.');
  } else renderDaily();
}
function renderCalendarRecords() {
  const s = state.selection;
  const rows = calendarRecords(state.rows, Number(s.start.slice(0, 4)), Number(s.end.slice(0, 4)));
  $('ex-title').textContent = 'Calendar-day records & averages';
  $('ex-description').textContent = 'Each month/day combines the selected calendar years: lowest daily minimum, highest daily maximum, and arithmetic mean of valid daily means. These are records within this reanalysis selection, not official station records. All tied record dates are retained. February 29 includes leap years only.';
  $('ex-legend').textContent = 'Record low: green · Mean daily temperature: rust · Record high: blue. Counts show available / expected years; partial samples can miss more extreme values. This mean is separate from the fixed baseline median.';
  const fields = [['record_low_c', 'Record low'], ['mean_daily_c', 'Mean daily temperature'], ['record_high_c', 'Record high']];
  lineChart($('ex-chart'), fields.map(([key, name], i) => ({ name, dash: ['', '7 3', '2 3'][i], points: rows.map(r => ({ x: calendarDay(`2000-${r.month_day}`), label: r.month_day, y: toUnit(r[key], s.unit), detail: `${shown(toUnit(r[key], s.unit))} °${s.unit.toUpperCase()} · ${r.valid_years}/${r.expected_years} years · ${r.status}${key === 'record_low_c' ? ' · ' + r.record_low_dates : key === 'record_high_c' ? ' · ' + r.record_high_dates : ''}` })) })), { title: $('ex-title').textContent, unit: `°${s.unit.toUpperCase()}`, inspect, mark: s.mark });
  table([['month_day', 'Month/day'], ...fields.map(([key, label]) => [key, `${label} (°${s.unit.toUpperCase()})`, v => shown(toUnit(v, s.unit))]), ['record_low_dates', 'Record low dates'], ['record_high_dates', 'Record high dates'], ['valid_years', 'Valid years'], ['expected_years', 'Expected years'], ['status', 'Sample coverage']], rows,
    'Available paired valid daily temperatures only; missing years are counted, never filled with zero. Partial records and averages remain labeled. CSV temperatures stay in Celsius, with all tied dates and selected-year coverage.');
}

function renderPeriodStatistics() {
  const s = state.selection, monthly = s.view === 'monthly', sameMonth = s.view === 'month-compare';
  const monthName = new Intl.DateTimeFormat('en', { month: 'long', timeZone: 'UTC' }).format(new Date(Date.UTC(2000, s.month - 1, 1)));
  const statisticLabel = { mean: 'Arithmetic mean', min: 'Minimum', max: 'Maximum' }[s.statistic];
  const rows = periodStatistics(state.rows, s.metric, monthly ? 'month' : 'year', s.statistic, sameMonth ? s.month : null);
  $('ex-title').textContent = monthly ? `${s.profile_year} month by month` : sameMonth ? `${monthName} across the years` : 'Year-by-year temperature statistics';
  $('ex-description').textContent = `${statisticLabel} of ${METRICS[s.metric].toLowerCase()} values for each ${monthly || sameMonth ? 'complete calendar month' : 'complete calendar year'}. Missing coverage makes a period unavailable, including partial boundary years. These are historical temperatures; the reference baseline is unchanged.`;
  $('ex-legend').textContent = `${statisticLabel} · ${METRICS[s.metric]} · ${s.mark} chart · °${s.unit.toUpperCase()}`;
  lineChart($('ex-chart'), [{ name: `${statisticLabel} ${METRICS[s.metric].toLowerCase()}`, points: rows.map((r, i) => ({ x: i, label: r.period, y: toUnit(r.value, s.unit), detail: `${finite(r.value) ? shown(toUnit(r.value, s.unit)) + ' °' + s.unit.toUpperCase() : 'unavailable'} · ${r.valid}/${r.expected} days · ${r.status}` })) }], { title: $('ex-title').textContent, unit: `°${s.unit.toUpperCase()}`, inspect, mark: s.mark });
  table([['period', monthly ? 'Month' : 'Year'], ['value_c', `${statisticLabel} (°${s.unit.toUpperCase()})`, v => shown(toUnit(v, s.unit))], ['valid', 'Valid days'], ['expected', 'Expected days'], ['status', 'Coverage status']], rows.map(r => ({ period: r.period, value_c: r.value, valid: r.valid, expected: r.expected, status: r.status })), 'The chart and CSV use these same complete-period statistics. Downloaded values stay in canonical Celsius and include the statistic, month, filters, and selected display unit.');
}

function renderFrost() {
  const s = state.selection, start = Number(s.start.slice(0, 4)), end = Number(s.end.slice(0, 4));
  const records = state.doc.annual.filter(r => r.threshold_c === s.threshold && r.season_year >= start && r.season_year <= end);
  const summary = state.doc.summaries.find(r => r.threshold_c === s.threshold);
  $('ex-title').textContent = 'Freeze dates & verified frost-free days';
  $('ex-description').textContent = `Full seasonal windows for inclusive years ${start}–${end}; month/day filters apply to daily views only. Freeze threshold: ${temperature(s.threshold, s.unit)} or colder. Historical bands retain the published baseline.`;
  $('ex-legend').textContent = 'Spring: green circles · Fall: rust diamonds · Distributions below include confirmed events only.';
  if (records.length) renderHistory($('ex-chart'), records, summary, records.at(-1).season_year, { symbols: true, inspect });
  else $('ex-chart').textContent = 'No annual frost records in this range.';
  const statuses = ['observed', 'no_event_in_complete_window', 'incomplete_window', 'insufficient_data'];
  $('ex-statistics').replaceChildren(...EVENT_NAMES.flatMap(name => [card(name === EVENT_NAMES[0] ? 'Confirmed spring events' : 'Confirmed fall events', String(records.filter(r => r[name].status === 'observed').length), statuses.slice(1).map(status => `${statusLabel(status)}: ${records.filter(r => r[name].status === status).length}`).join(' · '))]),
    card('Verified intervals', String(records.filter(r => r.interval.status === 'verified').length), 'Strictly intervening days; requires full daily coverage'));
  const intervals = node('div', undefined, 'chart explorer-chart'); intervals.tabIndex = 0; intervals.setAttribute('role', 'region'); intervals.setAttribute('aria-label', 'Verified frost-free days chart');
  $('ex-extra').append(node('h3', 'Verified frost-free intervals'), intervals);
  lineChart(intervals, [{ name: 'Verified frost-free days', points: records.map(r => ({ x: r.season_year, label: String(r.season_year), y: r.interval.frost_free_days })) }], { title: 'Verified frost-free days', unit: 'days', inspect });
  EVENT_NAMES.forEach(name => {
    const distribution = frostDistribution(records, name), details = node('details'), list = node('div', undefined, 'distribution-list');
    const label = name === EVENT_NAMES[0] ? 'Spring' : 'Fall';
    details.append(node('summary', `${label} event-date distribution (${distribution.reduce((sum, r) => sum + r.count, 0)} confirmed years)`));
    const chart = node('div', undefined, 'chart explorer-chart'); chart.tabIndex = 0;
    lineChart(chart, [{ name: `${label} event count`, points: distribution.map(r => ({ x: calendarDay(`2000-${r.day}`), label: r.day, y: r.count })) }], { title: `${label} confirmed event distribution`, unit: 'years', inspect });
    distribution.forEach(r => list.append(node('p', `${r.day}: ${r.count} confirmed ${r.count === 1 ? 'year' : 'years'}`)));
    details.append(chart, list); $('ex-extra').append(details);
  });
  const rows = records.map(r => ({ season_year: r.season_year, threshold_c: r.threshold_c,
    spring_date: r.last_spring_frost.date, spring_status: r.last_spring_frost.status,
    fall_date: r.first_fall_frost.date, fall_status: r.first_fall_frost.status,
    frost_free_days: r.interval.frost_free_days, interval_status: r.interval.status,
    date_difference_days: r.interval.date_difference_days }));
  table([['season_year', 'Season'], ['spring_date', 'Last spring freeze'], ['spring_status', 'Spring status', statusLabel], ['fall_date', 'First fall freeze'], ['fall_status', 'Fall status', statusLabel], ['frost_free_days', 'Verified frost-free days'], ['date_difference_days', 'Date difference'], ['interval_status', 'Interval status', statusLabel]], rows,
    'Full configured seasonal windows for selected years. CSV retains all event coverage fields and provisional candidates. No-event and incomplete years are excluded from date distributions.');
}
function renderHeatmap() {
  const s = state.selection, rows = monthlyAnomalies(state.rows, state.doc.temperature_climatology, s.metric);
  $('ex-title').textContent = 'Monthly departures from historical median';
  $('ex-description').textContent = 'Each cell is the arithmetic mean of daily departures from the corresponding calendar-day historical P50. A month needs complete paired daily coverage. Select a cell to inspect its daily temperatures.';
  $('ex-legend').textContent = `Warm: positive · Cool: negative · Hatched: unavailable or incomplete · Δ°${s.unit.toUpperCase()}`;
  const scroll = node('div', undefined, 'table-scroll'), grid = node('table', undefined, 'heatmap'), head = node('tr');
  head.append(node('th', 'Year'));
  for (let month = 1; month <= 12; month++) head.append(node('th', new Intl.DateTimeFormat('en', { month: 'short', timeZone: 'UTC' }).format(new Date(Date.UTC(2000, month - 1, 1)))));
  const thead = node('thead'); thead.append(head); grid.append(thead);
  const tbody = node('tbody');
  [...new Set(rows.map(r => r.period.slice(0, 4)))].forEach(year => {
    const tr = node('tr'), th = node('th', year); th.scope = 'row'; tr.append(th);
    for (let month = 1; month <= 12; month++) {
      const period = `${year}-${String(month).padStart(2, '0')}`, r = rows.find(r => r.period === period), td = node('td');
      const value = toUnit(r?.value, s.unit, true), button = node('button', shown(value), !finite(value) ? 'missing' : value > 0 ? 'warm' : 'cold');
      button.type = 'button'; button.title = button.ariaLabel = `${period}: ${finite(value) ? shown(value) + ' degrees departure' : 'unavailable'}; ${r?.valid ?? 0}/${r?.expected ?? new Date(Date.UTC(Number(year), month, 0)).getUTCDate()} paired days. Open daily view.`;
      button.onclick = () => {
        const params = new URLSearchParams(); params.set('view', 'temperature'); params.set('start', period + '-01'); params.set('end', `${period}-${new Date(Date.UTC(Number(year), month, 0)).getUTCDate()}`);
        openSelection(state.doc.station.station_id, params);
      };
      td.append(button); tr.append(td);
    }
    tbody.append(tr);
  }); grid.append(tbody); scroll.append(grid); $('ex-chart').append(scroll);
  table([['period', 'Month'], ['departure_c', `Mean daily departure (Δ°${s.unit.toUpperCase()})`, v => shown(toUnit(v, s.unit, true))], ['valid', 'Valid paired days'], ['expected', 'Expected days']], rows.map(r => ({ period: r.period, departure_c: r.value, valid: r.valid, expected: r.expected })), 'Incomplete months have null departures. CSV stores departures in Celsius with the selected display unit identified.');
}
function renderSeasonal() {
  const s = state.selection, policy = state.summary.policy;
  const rows = seasonalMetrics(state.rows, policy);
  $('ex-title').textContent = 'Growing-season & winter metrics';
  $('ex-description').textContent = `GDD: ${policy.gdd.start}–${policy.gdd.end}, base ${policy.gdd.base_c} °C, ${policy.gdd.cap_c === null ? 'no upper cap' : 'upper cap ' + policy.gdd.cap_c + ' °C'}. Winter: December–February ending in the labeled year; HDD base ${policy.hdd_base_c} °C; cold threshold ${policy.cold_threshold_c} °C or colder. Values use only dates inside the selection.`;
  $('ex-legend').textContent = 'Degree-days stay in °C·days. Missing coverage prevents final totals. A running GDD subtotal stops at the first missing day.';
  lineChart($('ex-chart'), [{ name: 'Complete growing-season GDD', points: rows.map(r => ({ x: r.year, label: String(r.year), y: r.gdd })) }], { title: 'Complete growing-season GDD', unit: '°C·days', inspect });
  table([['year', 'Year'], ['gdd', 'Final GDD (°C·days)', shown], ['gdd_subtotal', 'Continuous GDD subtotal', shown], ['through', 'Subtotal through'], ['gdd_valid', 'Valid GDD days'], ['gdd_expected', 'Expected GDD days'], ['hdd', 'Winter HDD (°C·days)', shown], ['cold_days', 'Winter cold days'], ['cold_run', 'Longest cold run (days)'], ['winter_valid', 'Valid winter days'], ['winter_expected', 'Expected winter days']], rows, 'December belongs to the following winter year. Include the preceding December in your date range for a complete winter. Partial periods cannot produce final totals or confirmed maximum cold runs.');
}
function renderPrecipitation() {
  const s = state.selection, method = state.index.precipitation_method, unit = s.rain_unit;
  const compare = ['compare', 'cumulative'].includes(s.rain_mode);
  $('ex-title').textContent = { daily: 'Daily total precipitation', monthly: 'Monthly precipitation totals', yearly: 'Yearly precipitation totals', compare: 'Daily precipitation, year by year', cumulative: 'Cumulative precipitation by year' }[s.rain_mode];
  $('ex-description').textContent = `Rain and snow-water equivalent, reported in ${unit}. Reviewed complete reporting days: ${method.first_complete_date ?? 'none'} through ${method.last_complete_date ?? 'none'}. The source uses fixed UTC−07:00 days. Later precipitation is unreviewed and stays unavailable. Cumulative totals stop at the first missing day; final period totals require every day.`;
  $('ex-legend').textContent = compare ? s.compare.map((year, i) => `${year} · ${['green', 'rust', 'blue'][i]}`).join(' / ') : 'Total precipitation · wet day = at least 1 mm · gaps remain unavailable';
  const stats = precipitationStats(state.rows);
  $('ex-statistics').replaceChildren(card('Precipitation coverage', `${stats.valid}/${stats.expected}`, 'Reviewed complete / selected reporting days'),
    card('Complete selection total', `${shown(precipToUnit(stats.total, unit))} ${unit}`, stats.complete ? 'Every selected reporting day is complete' : 'Unavailable for an incomplete selection'),
    card('Mean of available daily amounts', `${shown(precipToUnit(stats.dailyMean, unit))} ${unit}`, 'Uses reviewed days only'),
    card('Wet days among valid days', stats.valid ? String(stats.wetDays) : '—', 'At least 1 mm of total precipitation'));
  let rows, series;
  if (['monthly', 'yearly'].includes(s.rain_mode)) {
    rows = precipitationPeriods(state.rows, s.rain_mode === 'yearly' ? 'year' : 'month');
    series = [{ name: 'Total precipitation', points: rows.map((r, i) => ({ x: i, label: r.period, y: precipToUnit(r.total_mm, unit), detail: `${finite(r.total_mm) ? shown(precipToUnit(r.total_mm, unit)) + ' ' + unit : 'unavailable'} · ${r.valid_days}/${r.expected_days} days` })) }];
    table([['period', 'Period'], ['total_mm', `Total (${unit})`, v => shown(precipToUnit(v, unit))], ['wet_days_ge_1mm', 'Wet days (≥1 mm)'], ['valid_days', 'Valid days'], ['expected_days', 'Expected days'], ['status', 'Coverage']], rows, 'Totals and wet-day counts require complete reporting days. Downloaded amounts use millimeters; the selected display unit and source method are recorded.');
  } else {
    rows = s.rain_mode === 'cumulative' ? cumulativePrecipitation(state.rows) : state.rows;
    const key = s.rain_mode === 'cumulative' ? 'cumulative_mm' : 'precip_mm';
    const groupYears = compare ? s.compare : [null];
    series = groupYears.map((year, i) => ({ name: year === null ? 'Daily precipitation' : String(year), color: ['#36644b', '#aa5735', '#385f91'][i], dash: ['', '7 3', '2 3'][i], points: rows.filter(r => year === null || Number(r.date.slice(0, 4)) === year).map((r, index) => ({ x: compare ? calendarDay(r.date) : index, label: compare ? r.date.slice(5) : r.date, y: precipToUnit(r[key], unit), detail: `${r.date}: ${finite(r[key]) ? shown(precipToUnit(r[key], unit)) + ' ' + unit : 'unavailable'} · ${r.precip_status}` })) }));
    table([['date', 'Reporting date (UTC−07)'], ['precip_mm', `Daily total (${unit})`, v => shown(precipToUnit(v, unit))], ...(s.rain_mode === 'cumulative' ? [['cumulative_mm', `Continuous year total (${unit})`, v => shown(precipToUnit(v, unit))]] : []), ['precip_valid_hours', 'Valid hours'], ['precip_expected_hours', 'Expected hours'], ['precip_status', 'Precipitation coverage']], rows.map(r => ({ date: r.date, precip_mm: r.precip_mm, ...(s.rain_mode === 'cumulative' ? { cumulative_mm: r.cumulative_mm } : {}), precip_valid_hours: r.precip_valid_hours, precip_expected_hours: r.precip_expected_hours, precip_status: r.precip_status })), 'Each daily amount requires all 24 precipitation hours independently of temperature coverage. A cumulative curve stops at its first gap. CSV values remain in millimeters.');
  }
  lineChart($('ex-chart'), series, { title: $('ex-title').textContent, unit, inspect, mark: s.mark });
}
function renderAssessment() {
  const s = state.selection, policy = state.summary.policy;
  const requirements = { gdd: s.need_gdd, days: s.need_days, water: s.need_water, precip_mm: s.need_precip };
  const rows = growingAssessment(state.rows, state.doc.annual, policy, requirements, s.threshold);
  const labels = { requirements_met: 'Selected climate requirements met', requirements_not_met: 'Selected requirements not met', insufficient_data: 'Insufficient data', met: 'Met', shortfall: 'Shortfall', unknown: 'Unknown', not_selected: 'Not selected' };
  $('ex-title').textContent = 'Growing-year climate assessment';
  $('ex-description').textContent = `Historical comparison with your requirements: GDD ≥ ${s.need_gdd} °C·days (base ${policy.gdd.base_c} °C, ${policy.gdd.start}–${policy.gdd.end}, ${policy.gdd.cap_c === null ? 'no upper cap' : 'cap ' + policy.gdd.cap_c + ' °C'}), verified frost-free interval ≥ ${s.need_days} days at ${temperature(s.threshold, s.unit)}. ${s.need_water ? 'Season precipitation ≥ ' + s.need_precip + ' mm.' : 'Precipitation is shown as context and is not a pass/fail requirement.'} These example requirements are yours to set; no future season or yield is predicted.`;
  $('ex-legend').textContent = 'Historical heat and frost-free windows are separate checks. A missing required metric cannot pass. Precipitation is total water equivalent, not effective crop water or an irrigation estimate.';
  const count = status => rows.filter(r => r.assessment === status).length;
  $('ex-statistics').replaceChildren(card('Requirements met', String(count('requirements_met')), 'Historical years meeting every selected requirement'), card('Requirements not met', String(count('requirements_not_met')), 'At least one known shortfall'), card('Insufficient data', String(count('insufficient_data')), 'A required metric is unavailable'), card('Soil fertility', 'Not assessed', 'No soil nutrient, organic matter, or pH measurements'));
  lineChart($('ex-chart'), [{ name: 'Complete-season GDD', points: rows.map(r => ({ x: r.year, label: String(r.year), y: r.gdd })) }, { name: 'Your minimum GDD', reference: true, dash: '6 4', points: rows.map(r => ({ x: r.year, label: String(r.year), y: s.need_gdd })) }], { title: 'Growing degree-days against your requirement', unit: '°C·days', inspect, mark: s.mark });
  const interval = node('div', undefined, 'chart explorer-chart'); interval.tabIndex = 0; interval.setAttribute('role', 'region'); interval.setAttribute('aria-label', 'Frost-free days against selected requirement');
  $('ex-extra').append(node('h3', 'Verified frost-free days'), interval);
  lineChart(interval, [{ name: 'Verified interval', points: rows.map(r => ({ x: r.year, label: String(r.year), y: r.frost_free_days })) }, { name: 'Your minimum days', reference: true, dash: '6 4', points: rows.map(r => ({ x: r.year, label: String(r.year), y: s.need_days })) }], { title: 'Frost-free days against your requirement', unit: 'days', inspect, mark: s.mark });
  const note = node('p', 'Soil fertility needs site-specific soil measurements. Weather records cannot establish nutrient availability, pH, soil moisture, or crop yield. ');
  const link = node('a', 'About soil testing'); link.href = 'https://extension.umn.edu/garden-and-home/yard-and-garden/gardening-in-minnesota/soil-testing-for-lawns-and-gardens'; note.append(link); $('ex-extra').append(note);
  table([['year', 'Year'], ['assessment', 'Climate requirement result', v => labels[v]], ['gdd', 'GDD (°C·days)', shown], ['heat_requirement', 'Heat requirement', v => labels[v]], ['frost_free_days', 'Verified frost-free days'], ['frost_free_requirement', 'Frost-free requirement', v => labels[v]], ['seasonal_precip_mm', 'Season precipitation (mm)', shown], ['precipitation_requirement', 'Precipitation requirement', v => labels[v]], ['gdd_valid', 'Valid GDD days'], ['gdd_expected', 'Expected GDD days'], ['precip_valid_days', 'Valid precipitation days'], ['precip_expected_days', 'Expected precipitation days'], ['frost_status', 'Interval coverage', statusLabel]], rows, 'All years use full configured seasonal windows. Missing required data cannot produce a passing result. A known shortfall is reported even when a different metric is unknown. These are historical climate checks, not soil-fertility or crop-success scores.');
}

function render() {
  const s = state.selection, period = baseline();
  $('ex-chart').removeEventListener('idwx-inspect-linked', $('ex-chart')._linkedInspection);
  $('ex-extra').replaceChildren(); $('ex-chart').replaceChildren(); $('ex-chart').onkeydown = null;
  inspect('Select a chart position or use the arrow keys to inspect values.');
  $('ex-period').textContent = (s.view === 'compare' || s.view === 'precipitation' && ['compare', 'cumulative'].includes(s.rain_mode)) ? `Full years ${s.compare.join(' / ')}` : `${s.start} through ${s.end}`;
  $('ex-provenance').textContent = `${state.doc.station.name} · ${state.manifest.data_kind} · ${state.doc.station.timezone} · Source through ${state.doc.source_coverage.last_date} · Reference baseline ${period.start_year}–${period.end_year} · Exported ${state.manifest.generated_at_utc} · Release ${state.manifest.release_id}`;
  $('explorer-notice').hidden = false;
  $('explorer-notice').textContent = state.manifest.data_kind === 'synthetic_demo' ? 'Synthetic demonstration — these are not actual weather measurements.' : 'Historical ERA5/Oikolab reanalysis. Archive freshness and export time are shown separately; this explorer does not fetch live weather.';
  $('ex-method').textContent = 'Temperature reference quantiles use linear interpolation and at least five paired valid years per calendar day. Frost quantiles use nearest rank and the published minimum confirmed-year requirement. Filtering the analysis period does not change the reference baseline.';
  renderStatistics();
  if (s.view === 'calendar') renderCalendarRecords(); else if (s.view === 'precipitation') renderPrecipitation(); else if (s.view === 'assessment') renderAssessment(); else if (['annual', 'month-compare', 'monthly'].includes(s.view)) renderPeriodStatistics(); else if (s.view === 'frost') renderFrost(); else if (s.view === 'heatmap') renderHeatmap(); else if (s.view === 'seasonal') renderSeasonal(); else renderTemperature();
}

$('explorer-filters').addEventListener('submit', event => { event.preventDefault(); if (state.stations) openSelection($('ex-station').value, paramsFromForm()); });
$('ex-view').addEventListener('change', () => {
  if (state.index && ['annual', 'month-compare', 'calendar'].includes($('ex-view').value) && !['annual', 'month-compare', 'calendar'].includes(state.selection.view)) {
    $('ex-start').value = `${years()[0]}-01-01`;
    const last = state.doc.source_coverage.last_date;
    $('ex-end').value = `${last.endsWith('-12-31') ? years().at(-1) : years().length > 1 ? years().at(-2) : years()[0]}-12-31`;
    $('ex-mark').value = $('ex-view').value === 'calendar' ? 'line' : 'bar';
  }
  if ($('ex-view').value === 'monthly') $('ex-mark').value = 'bar';
  if ($('ex-view').value === 'precipitation' && state.index?.precipitation_method?.last_complete_date) {
    const date = state.index.precipitation_method.last_complete_date;
    const year = Number(date.slice(0, 4)) - (date.endsWith('12-31') ? 0 : 1);
    $('ex-start').value = `${year}-01-01`; $('ex-end').value = `${year}-12-31`;
    $('ex-compare').value = years().filter(y => y <= year).slice(-3).join(',');
    for (let i = 1; i <= 3; i++) $('ex-year-' + i).value = $('ex-compare').value.split(',')[i - 1] ?? '';
    $('ex-mark').value = 'bar';
  }
  visibility();
});
$('ex-rain_mode').addEventListener('change', visibility);
for (let i = 1; i <= 3; i++) $('ex-year-' + i).addEventListener('change', () => {
  $('ex-compare').value = [...new Set([1, 2, 3].map(i => $('ex-year-' + i).value).filter(Boolean))].join(',');
});
$('ex-station').addEventListener('change', () => openSelection($('ex-station').value, paramsFromForm()));
$('ex-reset').addEventListener('click', () => openSelection(state.doc.station.station_id, new URLSearchParams()));
$('ex-previous').onclick = () => { state.page--; renderTable(); };
$('ex-next').onclick = () => { state.page++; renderTable(); };
$('ex-download').onclick = () => {
  const period = baseline(), s = state.selection;
  const comparison = s.view === 'compare' || s.view === 'precipitation' && ['compare', 'cumulative'].includes(s.rain_mode);
  const metadata = { schema_version: state.manifest.schema_version, release_id: state.manifest.release_id, data_kind: state.manifest.data_kind, station_id: state.doc.station.station_id,
    view: s.view, start: comparison ? '' : s.view === 'frost' ? s.start.slice(0, 4) + '-01-01' : s.start, end: comparison ? '' : s.view === 'frost' ? s.end.slice(0, 4) + '-12-31' : s.end, comparison_years: comparison ? s.compare.join(';') : '', display_unit: s.view === 'precipitation' ? s.rain_unit : s.unit, metric: s.view === 'precipitation' ? 'total_precipitation' : s.view === 'assessment' ? 'growing_year_requirements' : s.view === 'calendar' ? 'calendar_day_temperatures' : s.metric,
    statistic: ['annual', 'month-compare', 'monthly'].includes(s.view) ? s.statistic : '', selected_month: s.view === 'month-compare' ? s.month : '', chart_style: ['temperature', 'compare', 'annual', 'month-compare', 'monthly', 'calendar', 'precipitation', 'assessment'].includes(s.view) ? s.mark : '',
    baseline_start: period.start_year, baseline_end: period.end_year, threshold_c: ['frost', 'assessment'].includes(s.view) ? s.threshold : '',
    selection_scope: comparison ? 'full_comparison_years' : ['frost', 'assessment'].includes(s.view) ? 'full_seasonal_windows' : s.view === 'month-compare' ? 'selected_month_across_years' : s.view === 'calendar' ? 'calendar_days_across_selected_years' : s.view === 'annual' ? 'full_calendar_years' : s.view === 'monthly' ? 'full_profile_year' : 'inclusive_local_dates' };
  if (s.view === 'precipitation') Object.assign(metadata, { precipitation_view: s.rain_mode, precipitation_unit: s.rain_unit, reporting_utc_offset_hours: -7, precipitation_last_reviewed: state.index.precipitation_method.last_complete_date });
  if (s.view === 'assessment') Object.assign(metadata, { minimum_gdd: s.need_gdd, minimum_frost_free_days: s.need_days, precipitation_requirement_enabled: s.need_water, minimum_precip_mm: s.need_precip });
  if (['seasonal', 'assessment'].includes(s.view)) Object.assign(metadata, { gdd_base_c: state.summary.policy.gdd.base_c, gdd_cap_c: state.summary.policy.gdd.cap_c, gdd_start: state.summary.policy.gdd.start, gdd_end: state.summary.policy.gdd.end, hdd_base_c: state.summary.policy.hdd_base_c, cold_threshold_c: state.summary.policy.cold_threshold_c, winter_months: '12;1;2', degree_day_unit: 'C_days' });
  // Frost CSV retains the full existing event contract, augmented with the selection metadata.
  let rows = state.table;
  if (s.view === 'frost') {
    const records = state.doc.annual.filter(r => r.threshold_c === s.threshold && r.season_year >= Number(s.start.slice(0, 4)) && r.season_year <= Number(s.end.slice(0, 4)));
    rows = records.map(r => ({ season_year: r.season_year, ...Object.fromEntries(EVENT_NAMES.flatMap(name => Object.entries(r[name]).map(([key, value]) => [`${name}_${key}`, value]))), ...Object.fromEntries(Object.entries(r.interval).map(([key, value]) => [`interval_${key}`, value])) }));
  }
  const blob = new Blob([csv(rows, metadata)], { type: 'text/csv;charset=utf-8' });
  const url = URL.createObjectURL(blob), a = node('a'); a.href = url; a.download = `${state.doc.station.station_id}-${s.view}.csv`; a.click(); setTimeout(() => URL.revokeObjectURL(url), 1000);
};
async function load() {
  controller?.abort(); controller = new AbortController();
  try {
    const manifest = options.client ? await options.client.pin() : await fetchJSON(new URL('data/manifest.json', base), controller.signal, false);
    if (!['1.1.0', '1.2.0'].includes(manifest.schema_version) || !/^[a-f0-9]{64}$/.test(manifest.release_id)) throw new Error('This archive does not yet include the data explorer.');
    state.manifest = manifest;
    state.stations = await releaseDocument(manifest.stations_url, controller.signal);
    if (!Array.isArray(state.stations.stations) || !state.stations.stations.length) throw new Error('No stations are available.');
    $('ex-station').replaceChildren(...state.stations.stations.map(s => option(s.station.station_id, s.station.name))); $('ex-station').disabled = false;
    const params = new URLSearchParams(options.query ?? location.search);
    await openSelection(options.station ?? params.get('station'), params, false);
    options.onReady?.(snapshot());
  } catch (error) { showError(error, state.request); }
}
$('explorer-retry').onclick = () => state.stations && state.pending ? openSelection(state.pending.stationID, state.pending.params, state.pending.push) : load();
const popstate = () => { if (state.stations) { const params = new URLSearchParams(location.search); openSelection(params.get('station'), params, false); } };
if (!options.onChange) window.addEventListener('popstate', popstate);
try { studies = parseStudies(localStorage.getItem(studiesKey)); }
catch { studiesAvailable = false; $('studies-import').disabled = true; $('studies-status').textContent = 'Saved studies could not be read. Existing storage is preserved; the explorer is still available.'; }
renderStudies();
$('study-form').onsubmit = event => {
  event.preventDefault();
  if (!studiesAvailable || !state.selection || $('explorer-results').hidden) return;
  const study = $('study-name').value.trim(), name = $('study-chart-name').value.trim();
  if (!study || !name) { $('studies-status').textContent = 'Enter a study name and chart name.'; return; }
  // Save applied filters, even if there are unsubmitted edits in the form.
  const params = new URLSearchParams();
  for (const [key, value] of Object.entries(state.selection)) params.set(key, Array.isArray(value) ? value.join(',') : typeof value === 'boolean' ? value ? 'on' : 'off' : value);
  if (saveStudies([...studies, { study, name, notes: $('study-notes').value, query: params.toString(), station: state.doc.station.station_id, release: state.manifest.release_id, saved_at: new Date().toISOString(), data_kind: state.manifest.data_kind }])) $('studies-status').textContent = 'Displayed chart selection saved in this browser.';
};
$('studies-download').onclick = () => {
  const blob = new Blob([JSON.stringify({ version: 1, charts: studies }, null, 2) + '\n'], { type: 'application/json' });
  const url = URL.createObjectURL(blob), a = node('a'); a.href = url; a.download = 'idwx-weather-studies.json'; a.click(); setTimeout(() => URL.revokeObjectURL(url), 1000);
};
$('studies-import').onchange = async event => {
  const file = event.target.files?.[0];
  if (!file || !studiesAvailable) return;
  try {
    if (file.size > 1000000) throw new Error('Study file is too large.');
    // Merge backups without replacing existing selections or duplicating identical entries.
    const incoming = parseStudies(await file.text());
    const merged = [...studies], known = new Set(studies.map(chart => JSON.stringify(chart)));
    for (const chart of incoming) if (!known.has(JSON.stringify(chart))) { merged.push(chart); known.add(JSON.stringify(chart)); }
    if (saveStudies(merged)) $('studies-status').textContent = `Backup imported; ${studies.length} saved charts now available.`;
  } catch { $('studies-status').textContent = 'Could not import this study file. Choose an IDWX studies backup of at most 50 charts. Existing studies are preserved.'; }
  event.target.value = '';
};
function snapshot() {
  if (!state.selection || !state.doc) return null;
  const params = new URLSearchParams();
  for (const [key, value] of Object.entries(state.selection)) params.set(key, Array.isArray(value) ? value.join(',') : typeof value === 'boolean' ? value ? 'on' : 'off' : value);
  return { station: state.doc.station.station_id, query: params.toString() };
}
const ready = load();
let lastWidth = 0, resizeFrame;
const resize = typeof ResizeObserver === 'function' ? new ResizeObserver(entries => {
  const width = entries[0]?.contentRect.width;
  if (!(width > 0) || Math.abs(width - lastWidth) < 2) return;
  lastWidth = width;
  if (!state.selection || $('explorer-results').hidden) return;
  cancelAnimationFrame(resizeFrame);
  resizeFrame = requestAnimationFrame(() => {
    const page = state.page; render(); state.page = Math.min(page, Math.max(0, Math.ceil(state.table.length / 100) - 1)); renderTable();
  });
}) : null;
resize?.observe($('ex-chart'));
return { ready, snapshot, elements,
  setSelection: (selection, push = true) => openSelection(selection.station, new URLSearchParams(selection.query), push),
  download: () => $('ex-download').click(),
  edit: () => { $('explorer-filters').hidden = !$('explorer-filters').hidden; if (!$('explorer-filters').hidden) $('ex-view').focus(); },
  destroy: () => { controller?.abort(); ++state.request; resize?.disconnect(); if (resizeFrame) cancelAnimationFrame(resizeFrame); state.cache.clear(); window.removeEventListener('popstate', popstate); },
};
}
