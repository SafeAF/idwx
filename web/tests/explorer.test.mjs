import assert from 'node:assert/strict';
import { execFileSync } from 'node:child_process';
import { readFile, readdir } from 'node:fs/promises';
import { mkdirSync, mkdtempSync, rmSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import vm from 'node:vm';
import { after, test } from 'node:test';
import { JSDOM, VirtualConsole } from 'jsdom';

const root = fileURLToPath(new URL('../../', import.meta.url));
mkdirSync(path.join(root, '.tmp'), { recursive: true });
const fixture = mkdtempSync(path.join(root, '.tmp', 'explorer-tests-'));
const python = process.env.IDWX_PYTHON || path.join(root, '.venv/bin/python');
execFileSync(python, ['-c', `
from pathlib import Path
from dataclasses import replace
import sys
from idwx.config import load_config,StationMeta
from idwx.web_export import export_web_data
export_web_data(replace(load_config('configs/idaho.yml'),backtest={'start_year':2000,'end_year':2022}),[StationMeta('demo','Demo')],Path(sys.argv[1]),demo=True)
`, fixture], { cwd: root });
after(() => rmSync(fixture, { recursive: true }));
const files = new Map();
async function inventory(directory) {
  for (const entry of await readdir(directory, { withFileTypes: true })) {
    const target = path.join(directory, entry.name);
    if (entry.isDirectory()) await inventory(target);
    else files.set('/almanac/' + path.relative(fixture, target).split(path.sep).join('/'), await readFile(target, 'utf8'));
  }
}
await inventory(fixture);
const html = await readFile(path.join(root, 'web/explore.html'), 'utf8');
const manifest = JSON.parse(files.get('/almanac/data/manifest.json'));
const settle = async predicate => {
  const limit = Date.now() + 10000;
  while (!predicate()) { if (Date.now() > limit) throw new Error('Explorer did not settle'); await new Promise(resolve => setImmediate(resolve)); }
};
async function explorer(t, { query = '?start=2020-01-01&end=2020-12-31', route, savedStudies } = {}) {
  const errors = [], requests = [], downloads = [], blobs = new Map(), console = new VirtualConsole();
  console.on('jsdomError', error => errors.push(error));
  const dom = new JSDOM(html, { url: 'https://idwx.test/almanac/explore.html' + query, runScripts: 'outside-only', virtualConsole: console });
  t.after(() => { dom.window.close(); assert.deepEqual(errors, []); });
  const { window } = dom;
  if (savedStudies !== undefined) window.localStorage.setItem('idwx.weather-studies.v1', savedStudies);
  window.Blob = Blob;
  window.URL.createObjectURL = blob => { const key = 'blob:test-' + blobs.size; blobs.set(key, blob); return key; };
  window.URL.revokeObjectURL = () => {};
  window.HTMLAnchorElement.prototype.click = function () { downloads.push(this.href); };
  window.fetch = async (url, options) => {
    const parsed = new URL(url); requests.push(parsed.pathname);
    const response = route && await route(parsed, options); if (response) return response;
    return { ok: files.has(parsed.pathname), status: files.has(parsed.pathname) ? 200 : 404, json: async () => JSON.parse(files.get(parsed.pathname)) };
  };
  const context = dom.getInternalVMContext(), modules = new Map();
  function moduleFor(url) {
    if (!modules.has(url.href)) modules.set(url.href, readFile(url, 'utf8').then(source => new vm.SourceTextModule(source, { context, identifier: url.href })));
    return modules.get(url.href);
  }
  const app = await moduleFor(pathToFileURL(path.join(root, 'web/js/explorer.js')));
  await app.link((specifier, reference) => moduleFor(new URL(specifier, reference.identifier)));
  await app.evaluate();
  const $ = id => window.document.getElementById(id);
  await settle(() => !$('explorer-results').hidden || !$('explorer-retry').hidden);
  const apply = async values => {
    for (const [key, value] of Object.entries(values)) $('ex-' + key).value = value;
    $('explorer-filters').dispatchEvent(new window.Event('submit', { bubbles: true, cancelable: true }));
    await settle(() => !$('explorer-results').hidden || !$('explorer-retry').hidden);
  };
  return { window, $, apply, requests, downloads, blobs };
}

test('daily explorer restores URL, only loads selected chunks and exports identical daily selection', async t => {
  const { $, requests, blobs } = await explorer(t, { query: '?start=2020-02-28&end=2020-03-01&unit=f' });
  assert.equal($('explorer-results').hidden, false, $('explorer-status').textContent);
  assert.equal($('ex-unit').value, 'f'); assert.equal($('ex-table-body').children.length, 3);
  assert.match($('ex-table-body').textContent, /2020-02-29/);
  assert.match($('explorer-notice').textContent, /Synthetic/);
  assert.deepEqual(requests.filter(r => r.includes('/daily/')).map(r => r.split('/').at(-1)), ['2020.json']);
  assert.ok(requests.every(r => r.startsWith('/almanac/data/')));
  $('ex-download').click(); const text = await [...blobs.values()][0].text();
  assert.equal(text.split('\r\n').filter(Boolean).length, 4);
  assert.match(text, new RegExp(manifest.release_id)); assert.match(text, /"2020-02-29"/);
  assert.match(text, /"display_unit"/);
});

test('comparison uses full selected years, shared month/day axis, URL history, and units', async t => {
  const { $, apply, window, blobs } = await explorer(t);
  await apply({ view: 'compare', compare: '2020,2021,2020,9999', unit: 'f' });
  assert.equal($('ex-compare').value, '2020,2021'); assert.equal($('ex-start').disabled, true);
  assert.match($('ex-page').textContent, /731 rows/); assert.match($('ex-legend').textContent, /2021 \(dashed\)/);
  assert.match(window.location.search, /view=compare/);
  $('ex-download').click();
  const exported = await [...blobs.values()][0].text();
  assert.match(exported, /"compare","","","2020;2021"/);
  assert.match(exported, /full_comparison_years/);
  window.history.back();
  await settle(() => $('ex-view').value === 'temperature' && !$('explorer-results').hidden);
  assert.equal($('ex-unit').value, 'c');
});

test('frost status counts, verified intervals, and event distributions respect selected years', async t => {
  const { $, apply, blobs } = await explorer(t);
  await apply({ view: 'frost', start: '2019-01-01', end: '2021-12-31', threshold: '0' });
  assert.equal($('ex-table-body').children.length, 3);
  assert.match($('ex-statistics').textContent, /No event in complete window: 1/);
  assert.match($('ex-table-body').textContent, /Incomplete window/);
  assert.match($('ex-extra').textContent, /Spring event-date distribution/);
  $('ex-download').click(); const text = await [...blobs.values()][0].text();
  assert.match(text, /candidate_date/); assert.match(text, /coverage_fraction/);
  assert.equal(text.split('\r\n').filter(Boolean).length, 4);
});

test('monthly anomalies expose complete paired counts and drill down to the selected month', async t => {
  const { $, apply, window } = await explorer(t);
  await apply({ view: 'heatmap', start: '2020-01-01', end: '2020-12-31' });
  assert.equal($('ex-chart').querySelectorAll('button').length, 12);
  assert.equal($('ex-table-body').children.length, 12);
  const feb = $('ex-chart').querySelectorAll('button')[1]; assert.match(feb.title, /29\/29/); feb.click();
  await settle(() => $('ex-view').value === 'temperature' && !$('explorer-results').hidden);
  assert.equal($('ex-start').value, '2020-02-01'); assert.equal($('ex-end').value, '2020-02-29');
  $('ex-chart').dispatchEvent(new window.KeyboardEvent('keydown', { key: 'ArrowRight', bubbles: true }));
  assert.match($('ex-inspection').textContent, /2020-02-02/);
});

test('failed year chunks show an error and retry preserves the requested selection', async t => {
  let fail = true;
  const { $, requests } = await explorer(t, { query: '?start=2020-02-01&end=2020-02-29', route: parsed => {
    if (parsed.pathname.endsWith('/daily/2020.json') && fail) { fail = false; return { ok: false, status: 503 }; }
  } });
  assert.equal($('explorer-results').hidden, true); assert.match($('explorer-status').textContent, /503/);
  $('explorer-retry').click(); await settle(() => !$('explorer-results').hidden);
  assert.equal($('ex-start').value, '2020-02-01'); assert.equal($('ex-table-body').children.length, 29);
  assert.equal(requests.filter(r => r.endsWith('/daily/2020.json')).length, 2);
});

test('seasonal totals cannot present a partially selected growing window as complete', async t => {
  const { $, apply, blobs } = await explorer(t);
  await apply({ view: 'seasonal', start: '2020-04-01', end: '2020-04-02' });
  const cells = [...$('ex-table-body').querySelectorAll('td')].map(n => n.textContent);
  assert.equal(cells[1], '—'); assert.equal(cells[3], '2020-04-02'); assert.equal(cells[5], '214');
  assert.match($('ex-description').textContent, /December–February/);
  $('ex-download').click();
  const exported = await [...blobs.values()][0].text();
  assert.match(exported, /gdd_base_c/); assert.match(exported, /C_days/);
});

test('a mixed-release response is rejected and not retained in the retry cache', async t => {
  let corrupt = true;
  const { $, requests } = await explorer(t, { route: parsed => {
    if (corrupt && parsed.pathname.endsWith('/daily/2020.json')) {
      corrupt = false;
      const doc = JSON.parse(files.get(parsed.pathname)); doc.release_id = 'f'.repeat(64);
      return { ok: true, json: async () => doc };
    }
  } });
  assert.equal($('explorer-results').hidden, true); assert.match($('explorer-status').textContent, /mixed release/);
  $('explorer-retry').click(); await settle(() => !$('explorer-results').hidden);
  assert.equal(requests.filter(p => p.endsWith('/daily/2020.json')).length, 2);
});

test('late data cannot overwrite a newer selection and incomplete long periods stay unavailable', async t => {
  let releaseSlow;
  const { $, apply, window } = await explorer(t, { route: async parsed => {
    if (parsed.pathname.endsWith('/daily/2018.json')) await new Promise(resolve => { releaseSlow = resolve; });
  } });
  $('ex-start').value = '2018-01-01'; $('ex-end').value = '2018-12-31';
  $('explorer-filters').dispatchEvent(new window.Event('submit', { bubbles: true, cancelable: true }));
  await settle(() => !!releaseSlow);
  await apply({ start: '2020-01-01', end: '2020-12-31' });
  releaseSlow(); await new Promise(resolve => setImmediate(resolve));
  assert.equal($('ex-start').value, '2020-01-01'); assert.match($('explorer-status').textContent, /2020-12-31/);
  await apply({ start: '2019-01-01', end: '2022-12-31' });
  assert.equal($('ex-table-body').children.length, 48);
  assert.match($('ex-table-note').textContent, /complete-period means/);
  const june2021 = [...$('ex-table-body').children].find(tr => tr.firstChild.textContent === '2021-06');
  assert.equal(june2021.children[1].textContent, '—');
});

test('year selectors build a comparison without typing and preserve shareable years', async t => {
  const { $, apply, window } = await explorer(t);
  await apply({ view: 'compare' });
  $('ex-year-1').value = '2019'; $('ex-year-1').dispatchEvent(new window.Event('change'));
  $('ex-year-2').value = '2020'; $('ex-year-2').dispatchEvent(new window.Event('change'));
  $('ex-year-3').value = ''; $('ex-year-3').dispatchEvent(new window.Event('change'));
  await apply({ mark: 'scatter' });
  assert.equal($('ex-compare').value, '2019,2020');
  assert.equal($('ex-year-3').value, ''); assert.match($('ex-legend').textContent, /2019 \(circles\).*2020 \(squares\)/);
  assert.ok($('ex-chart').querySelectorAll('[data-mark="scatter"]').length > 700);
  assert.equal(new URL(window.location.href).searchParams.get('compare'), '2019,2020');
});

test('same-month comparisons show one statistic per year with coverage and matching CSV metadata', async t => {
  const { $, apply, blobs, window } = await explorer(t);
  await apply({ view: 'month-compare', start: '2019-05-10', end: '2021-08-09', month: '6', metric: 'tmin_c', statistic: 'max', mark: 'bar' });
  assert.match($('ex-title').textContent, /June across the years/);
  assert.equal($('ex-start').value, '2019-01-01'); assert.equal($('ex-end').value, '2021-12-31');
  const rows = [...$('ex-table-body').children]; assert.equal(rows.length, 3);
  assert.equal(rows[0].children[1].textContent, '8');
  assert.equal(rows[2].children[1].textContent, '—');
  assert.equal(rows[2].children[2].textContent, '29'); assert.equal(rows[2].children[3].textContent, '30');
  assert.equal($('ex-chart').querySelectorAll('[data-mark="bar"]').length, 2);
  $('ex-download').click(); const exported = await [...blobs.values()][0].text();
  assert.match(exported, /selected_month/); assert.match(exported, /"max","6","bar"/);
  assert.match(exported, /selected_month_across_years/);
  $('ex-chart').dispatchEvent(new window.KeyboardEvent('keydown', { key: 'End', bubbles: true }));
  assert.match($('ex-inspection').textContent, /2021.*unavailable/);
});

test('monthly profile loads one selected year and every chart style keeps the same table values', async t => {
  const { $, apply, requests, window } = await explorer(t, { query: '?view=monthly&profile_year=2020&statistic=min&mark=bar' });
  assert.equal($('profile-year-control').hidden, false);
  assert.equal($('ex-profile_year').value, '2020'); assert.equal($('ex-start').disabled, true);
  assert.equal($('ex-table-body').children.length, 12);
  assert.deepEqual(requests.filter(r => r.includes('/daily/')).map(r => r.split('/').at(-1)), ['2020.json']);
  const table = $('ex-table-body').textContent;
  for (const mark of ['line', 'scatter', 'area', 'bar']) {
    await apply({ mark });
    assert.ok($('ex-chart').querySelectorAll(`[data-mark="${mark}"]`).length > 0);
    assert.equal($('ex-table-body').textContent, table);
    assert.equal(new URL(window.location.href).searchParams.get('mark'), mark);
  }
});

test('homepage offers prominent working links for each visualization family', () => {
  const home = new JSDOM(files.get('/almanac/index.html'), { url: 'https://idwx.test/almanac/' });
  try {
    assert.match(home.window.document.querySelector('#visualizations-title').textContent, /Graphs & year comparisons/);
    const links = [...home.window.document.querySelectorAll('.visualization-hub a')];
    for (const view of ['compare', 'month-compare', 'monthly', 'annual', 'heatmap', 'frost']) {
      const link = links.find(a => new URL(a.href).searchParams.get('view') === view);
      assert.ok(link, view); assert.equal(new URL(link.href).pathname, '/almanac/explore.html');
    }
  } finally { home.window.close(); }
});

test('precipitation totals and CSV keep independent coverage, water-equivalent units, and missing periods', async t => {
  const { $, apply, blobs } = await explorer(t, { query: '?view=precipitation&start=2021-01-01&end=2021-12-31&rain_mode=monthly&rain_unit=in' });
  assert.equal($('explorer-results').hidden, false, $('explorer-status').textContent);
  assert.match($('ex-description').textContent, /Rain and snow-water equivalent/);
  assert.equal($('rain-unit-control').hidden, false); assert.equal($('requirements-control').hidden, true);
  assert.equal($('ex-table-body').children.length, 12);
  const june = [...$('ex-table-body').children].find(tr => tr.firstChild.textContent === '2021-06');
  assert.equal(june.children[1].textContent, '—'); assert.equal(june.children[3].textContent, '29'); assert.equal(june.children[4].textContent, '30');
  $('ex-download').click(); const exported = await [...blobs.values()][0].text();
  assert.match(exported, /total_mm/); assert.match(exported, /precipitation_unit/); assert.match(exported, /"monthly","in","-7"/);
  await apply({ rain_mode: 'cumulative', compare: '2020,2021', mark: 'line' });
  assert.equal($('ex-start').disabled, true); assert.equal($('compare-control').hidden, false);
  assert.match($('ex-table-head').textContent, /Continuous year total/);
});

test('growing-year requirements persist in URL and CSV and missing required evidence cannot pass', async t => {
  const { $, apply, blobs, window } = await explorer(t, { query: '?view=assessment&start=2022-01-01&end=2022-12-31&need_gdd=0&need_days=0&need_water=on&need_precip=0' });
  assert.equal($('requirements-control').hidden, false); assert.equal($('ex-need_water').checked, true);
  assert.equal($('threshold-control').hidden, false);
  assert.match($('ex-table-body').textContent, /Selected climate requirements met/);
  assert.match($('ex-statistics').textContent, /Soil fertilityNot assessed/);
  assert.match($('ex-extra').textContent, /Soil fertility needs site-specific soil measurements/);
  $('ex-download').click(); const exported = await [...blobs.values()][0].text();
  assert.match(exported, /minimum_frost_free_days/); assert.match(exported, /gdd_base_c/);
  assert.match(exported, /requirements_met/); assert.match(exported, /full_seasonal_windows/);
  await apply({ start: '2021-01-01', end: '2021-12-31' });
  assert.match($('ex-table-body').textContent, /Insufficient data/);
  assert.equal(new URL(window.location.href).searchParams.get('need_water'), 'on');
});

test('malformed precipitation cannot be rendered as complete even when temperature is valid', async t => {
  const { $ } = await explorer(t, { query: '?view=precipitation&start=2020-01-01&end=2020-12-31', route: parsed => {
    if (parsed.pathname.endsWith('/daily/2020.json')) {
      const data = JSON.parse(files.get(parsed.pathname)); data.days[0].precip_valid_hours = 23;
      return { ok: true, json: async () => data };
    }
  } });
  assert.equal($('explorer-results').hidden, true); assert.match($('explorer-status').textContent, /precipitation coverage/);
});

test('calendar-day records have matching canonical CSV, leap counts and keyboard inspection', async t => {
  const { $, window, blobs } = await explorer(t, { query: '?view=calendar&start=2020-05-01&end=2021-06-01&unit=f' });
  assert.match($('ex-title').textContent, /Calendar-day/);
  assert.equal($('ex-start').value, '2020-01-01'); assert.equal($('ex-end').value, '2021-12-31');
  assert.equal($('ex-metric').disabled, true); assert.equal($('ex-baseline').disabled, true);
  assert.equal($('ex-chart').querySelectorAll('path[data-mark=line]').length, 3);
  $('ex-chart').dispatchEvent(new window.KeyboardEvent('keydown', { key: 'Home', bubbles: true }));
  assert.match($('ex-inspection').textContent, /01-01.*2\/2 years/);
  $('ex-download').click(); const exported = await [...blobs.values()][0].text();
  assert.match(exported, /calendar_days_across_selected_years/);
  assert.match(exported, /record_low_dates/);
  assert.equal(exported.trim().split('\r\n').length, 367);
  const lines = exported.trim().split('\r\n'), keys = lines[0].slice(1, -1).split('","');
  const leap = lines.find(line => line.includes('"02-29"')).slice(1, -1).split('","');
  assert.equal(leap[keys.indexOf('expected_years')], '1'); assert.equal(leap[keys.indexOf('valid_years')], '1');
});

test('studies group named charts, save applied filters, reopen, export, import and remove', async t => {
  const { $, window, apply, blobs } = await explorer(t);
  const save = () => $('study-form').dispatchEvent(new window.Event('submit', { bubbles: true, cancelable: true }));
  $('study-name').value = 'Cold seasons'; $('study-chart-name').value = '<img src=x onerror=alert(1)>'; $('study-notes').value = '<script>untrusted note</script>';
  // Unapplied edits must not change the saved chart selection.
  $('ex-start').value = '2021-01-01'; save();
  assert.equal($('studies-list').querySelector('img'), null); assert.equal($('studies-list').querySelector('script'), null);
  let stored = JSON.parse(window.localStorage.getItem('idwx.weather-studies.v1'));
  assert.equal(new URLSearchParams(stored.charts[0].query).get('start'), '2020-01-01');
  assert.equal(stored.charts[0].release, manifest.release_id);
  await apply({ view: 'monthly', profile_year: '2021' });
  $('study-chart-name').value = 'Monthly profile'; save();
  assert.equal($('studies-list').querySelectorAll('.saved-study').length, 1);
  assert.equal($('studies-list').querySelectorAll('.saved-chart').length, 2);
  $('studies-list').querySelector('a').dispatchEvent(new window.MouseEvent('click', { bubbles: true, cancelable: true }));
  await settle(() => $('ex-view').value === 'temperature' && !$('explorer-results').hidden);
  assert.equal($('ex-start').value, '2020-01-01');
  $('studies-download').click(); const backup = await [...blobs.values()][0].text();
  assert.equal(JSON.parse(backup).charts.length, 2);
  $('studies-list').querySelector('button').click();
  assert.equal($('studies-list').querySelectorAll('.saved-chart').length, 1);
  Object.defineProperty($('studies-import'), 'files', { configurable: true, value: [{ size: backup.length, text: async () => backup }] });
  $('studies-import').dispatchEvent(new window.Event('change'));
  await settle(() => $('studies-list').querySelectorAll('.saved-chart').length === 2);
  assert.equal(JSON.parse(window.localStorage.getItem('idwx.weather-studies.v1')).charts.length, 2);
  const previous = window.localStorage.getItem('idwx.weather-studies.v1');
  Object.defineProperty($('studies-import'), 'files', { configurable: true, value: [{ size: 2, text: async () => '{}' }] });
  $('studies-import').dispatchEvent(new window.Event('change'));
  await settle(() => $('studies-status').textContent.includes('Could not import'));
  assert.equal(window.localStorage.getItem('idwx.weather-studies.v1'), previous);
});

test('corrupt study storage is preserved and does not prevent historical exploration', async t => {
  const { $, window } = await explorer(t, { savedStudies: '{unreadable' });
  assert.equal($('explorer-results').hidden, false);
  assert.equal($('study-save').disabled, true); assert.equal($('studies-import').disabled, true);
  assert.match($('studies-status').textContent, /Existing storage is preserved/);
  assert.equal(window.localStorage.getItem('idwx.weather-studies.v1'), '{unreadable');
});

test('saved charts identify data changes and reject stations missing from the loaded release', async t => {
  const chart = { study: 'Old study', name: 'Old chart', notes: '', query: 'view=calendar&start=2020-01-01&end=2021-12-31', station: 'demo', release: 'a'.repeat(64), saved_at: '2026-11-21T00:00:00Z', data_kind: 'historical_reanalysis' };
  const { $, window } = await explorer(t, { savedStudies: JSON.stringify({ version: 1, charts: [chart, { ...chart, station: 'removed' }] }) });
  assert.match($('studies-list').textContent, /data changed/);
  const links = $('studies-list').querySelectorAll('a');
  links[0].dispatchEvent(new window.MouseEvent('click', { bubbles: true, cancelable: true }));
  await settle(() => $('ex-view').value === 'calendar' && !$('explorer-results').hidden);
  assert.match($('studies-status').textContent, /Data release changed/);
  links[1].dispatchEvent(new window.MouseEvent('click', { bubbles: true, cancelable: true }));
  assert.match($('studies-status').textContent, /station is unavailable/);
  assert.equal($('ex-view').value, 'calendar');
});
