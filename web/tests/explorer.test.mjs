import assert from 'node:assert/strict';
import { webcrypto } from 'node:crypto';
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
import tempfile, csv, hashlib, yaml
import pandas as pd
from idwx.climate_metrics import METRICS
from idwx.parameter_samples import build_snapshot
with tempfile.TemporaryDirectory(dir='.tmp') as private:
    private=Path(private); raw=private/'raw'; raw.mkdir(); sources=[]
    for sid,lat in [('demo',42.4634),('archive_point',43.0)]:
        source=raw/(str(lat)+'--114.4975-test.csv')
        values={m.column:(m.minimum+m.maximum)/2 for m in METRICS.values()}
        values.update({'surface_pressure (Pa)':91000.5,'surface_solar_radiation (W/m^2)':123.45,'relative_humidity (0-1)':1.72})
        rows=[{'timestamp':str(stamp),'coordinates (lat,lon)':str((lat,-114.4975)),'model (name)':'era5','model elevation (surface)':1300,'utc_offset (hrs)':-7,**values} for stamp in pd.date_range('2020-02-28','2020-03-01T23:00',freq='h')]
        rows[1]['surface_pressure (Pa)']=''
        with source.open('w',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
        sources.append({'station_id':sid,'file':source.name,'latitude':lat,'longitude':-114.4975,'name':sid,'label_status':'coordinate_label','reporting_utc_offset_hours':-7,'model':'era5','sha256':hashlib.sha256(source.read_bytes()).hexdigest()})
    manifest=private/'sources.yml'
    manifest.write_text(yaml.safe_dump({'version':1,'adapter':'oikolab_era5_archive_v1','timestamp_convention':'naive_utc','start_utc':'2020-02-28T00:00:00Z','end_utc':'2020-03-01T23:00:00Z','sources':sources}))
    snapshot=build_snapshot(manifest,raw,private/'snapshots',data_kind='synthetic_demo')
    export_web_data(replace(load_config('configs/idaho.yml'),backtest={'start_year':2000,'end_year':2022}),[StationMeta('demo','Demo')],Path(sys.argv[1]),demo=True,parameter_snapshot=Path(snapshot['path']))
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
async function explorer(t, { query = '?start=2020-01-01&end=2020-12-31', route, savedStudies, workspaceMode = false, savedWorkspaces } = {}) {
  const errors = [], requests = [], downloads = [], blobs = new Map(), console = new VirtualConsole();
  console.on('jsdomError', error => errors.push(error));
  const dom = new JSDOM(html, { url: 'https://idwx.test/almanac/explore.html' + query, runScripts: 'outside-only', virtualConsole: console });
  t.after(() => { dom.window.close(); assert.deepEqual(errors, []); });
  const { window } = dom;
  Object.defineProperty(window.crypto, 'subtle', { value: webcrypto.subtle });
  window.TextEncoder = TextEncoder;
  if (savedStudies !== undefined) window.localStorage.setItem('idwx.weather-studies.v1', savedStudies);
  if (savedWorkspaces !== undefined) window.localStorage.setItem('idwx.weather-studies.v2', savedWorkspaces);
  window.Blob = Blob;
  window.URL.createObjectURL = blob => { const key = 'blob:test-' + blobs.size; blobs.set(key, blob); return key; };
  window.URL.revokeObjectURL = () => {};
  window.HTMLAnchorElement.prototype.click = function () { downloads.push(this.href); };
  window.fetch = async (url, options) => {
    const parsed = new URL(url); requests.push(parsed.pathname);
    const response = route && await route(parsed, options); if (response) return response;
    return { ok: files.has(parsed.pathname), status: files.has(parsed.pathname) ? 200 : 404, json: async () => JSON.parse(files.get(parsed.pathname)), text: async () => files.get(parsed.pathname) };
  };
  const context = dom.getInternalVMContext(), modules = new Map();
  function moduleFor(url) {
    if (!modules.has(url.href)) modules.set(url.href, readFile(url, 'utf8').then(source => new vm.SourceTextModule(source, { context, identifier: url.href })));
    return modules.get(url.href);
  }
  const app = await moduleFor(pathToFileURL(path.join(root, workspaceMode ? 'web/js/workspace.js' : 'web/js/explorer.js')));
  await app.link((specifier, reference) => moduleFor(new URL(specifier, reference.identifier)));
  await app.evaluate();
  if (!workspaceMode) await app.namespace.createExplorer(window.document).ready;
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


test('workspace adds three independent scoped graphs and deduplicates shared requests', async t => {
  const { $, window, requests, apply } = await explorer(t, { workspaceMode: true });
  const cards = () => [...window.document.querySelectorAll('.graph-card')];
  const graphReady = card => !card.querySelector('[id$="explorer-results"]').hidden;
  assert.equal(cards().length, 1);
  $('workspace-add').click(); await settle(() => cards().length === 2 && cards().every(graphReady));
  $('workspace-add').click(); await settle(() => cards().length === 3 && cards().every(graphReady));
  assert.equal($('workspace-add').disabled, true);
  const ids = [...window.document.querySelectorAll('[id]')].map(n => n.id);
  assert.equal(new Set(ids).size, ids.length);
  assert.equal(requests.filter(path => path.endsWith('/daily/2020.json')).length, 1);
  const second = cards()[1], secondStart = second.querySelector('[id$="ex-start"]');
  assert.equal(secondStart.value, '2020-01-01');
  await apply({ start: '2021-01-01', end: '2021-01-31', unit: 'f' });
  assert.equal(secondStart.value, '2020-01-01');
  $('workspace-link-dates').click();
  await apply({ start: '2022-02-01', end: '2022-02-28' });
  await settle(() => secondStart.value === '2022-02-01' && cards().every(graphReady));
  assert.equal(second.querySelector('[id$="ex-unit"]').value, 'c');
  const encoded = new URL(window.location.href).searchParams.get('workspace');
  assert.equal(JSON.parse(encoded).graphs.length, 3);
  second.querySelector('.graph-card-actions button[aria-label^="Expand"]').click();
  assert.ok(second.classList.contains('graph-expanded'));
  second.querySelector('.graph-card-actions button[aria-label^="Remove"]').click();
  assert.equal(cards().length, 2); assert.equal($('workspace-add').disabled, false);
});

test('workspace studies preserve all cards, applied filters, links and safe notes in backup', async t => {
  const { $, window, blobs } = await explorer(t, { workspaceMode: true });
  $('workspace-add').click();
  await settle(() => window.document.querySelectorAll('.graph-card').length === 2 && !window.document.querySelectorAll('[id$="explorer-results"]')[1].hidden);
  $('workspace-study-name').value = '<img src=x> Study'; $('workspace-study-notes').value = '<script>plain notes</script>';
  $('workspace-link-dates').click();
  $('ex-start').value = '2002-01-01'; // Unapplied edits do not enter saved state.
  $('workspace-save').click();
  let stored = JSON.parse(window.localStorage.getItem('idwx.weather-studies.v2'));
  assert.equal(stored.studies[0].workspace.graphs.length, 2);
  assert.equal(new URLSearchParams(stored.studies[0].workspace.graphs[0].query).get('start'), '2020-01-01');
  assert.equal(stored.studies[0].workspace.links.dates, true);
  assert.equal($('workspace-studies-list').querySelector('img,script'), null);
  $('workspace-studies-download').click();
  const backup = await [...blobs.values()].at(-1).text();
  assert.deepEqual(JSON.parse(backup), stored);
  $('workspace-studies-list').querySelector('button').click();
  await settle(() => window.document.querySelectorAll('.graph-card').length === 2 && !$('explorer-results').hidden && $('ex-start').value === '2020-01-01');
  assert.match($('workspace-studies-status').textContent, /same release/);
});

test('workspace migration retains v1 bytes and corrupt v2 storage remains untouched', async t => {
  const chart = { study: 'Old', name: 'Cold', notes: '', query: 'view=temperature', station: 'demo', release: manifest.release_id, data_kind: 'synthetic_demo', saved_at: '2026-11-21T00:00:00Z' };
  const original = JSON.stringify({ version: 1, charts: [chart] });
  const { $, window } = await explorer(t, { workspaceMode: true, savedStudies: original });
  assert.equal(window.localStorage.getItem('idwx.weather-studies.v1'), original);
  const migrated = JSON.parse(window.localStorage.getItem('idwx.weather-studies.v2'));
  assert.equal(migrated.studies[0].workspace.graphs.length, 1);
  assert.match($('workspace-studies-list').textContent, /Old: Cold/);
  const corrupt = await explorer(t, { workspaceMode: true, savedWorkspaces: '{bad' });
  assert.equal(corrupt.window.localStorage.getItem('idwx.weather-studies.v2'), '{bad');
  assert.equal(corrupt.$('workspace-save').disabled, true);
  assert.equal(corrupt.$('explorer-results').hidden, false);
});

test('one card failure retries independently and removing a loading graph does not cancel peers', async t => {
  let failed = true, releaseBlocked, blocked;
  const { $, window, apply } = await explorer(t, { workspaceMode: true, route: async parsed => {
    if (parsed.pathname.endsWith('/daily/2021.json') && failed) return { ok: false, status: 503 };
    if (parsed.pathname.endsWith('/daily/2018.json')) {
      blocked = true; await new Promise(resolve => { releaseBlocked = resolve; });
      return { ok: true, json: async () => JSON.parse(files.get(parsed.pathname)) };
    }
  } });
  $('workspace-add').click();
  const cards = () => [...window.document.querySelectorAll('.graph-card')];
  await settle(() => cards().length === 2 && !cards()[1].querySelector('[id$="explorer-results"]').hidden);
  const secondary = cards()[1];
  secondary.querySelector('[id$="ex-start"]').value = '2021-01-01'; secondary.querySelector('[id$="ex-end"]').value = '2021-01-31';
  secondary.querySelector('form').dispatchEvent(new window.Event('submit', { bubbles: true, cancelable: true }));
  await settle(() => !secondary.querySelector('[id$="explorer-retry"]').hidden);
  assert.equal($('explorer-results').hidden, false);
  failed = false; secondary.querySelector('[id$="explorer-retry"]').click();
  await settle(() => !secondary.querySelector('[id$="explorer-results"]').hidden);
  secondary.querySelector('[id$="ex-start"]').value = '2018-01-01'; secondary.querySelector('[id$="ex-end"]').value = '2018-01-31';
  secondary.querySelector('form').dispatchEvent(new window.Event('submit', { bubbles: true, cancelable: true }));
  await settle(() => blocked);
  secondary.querySelector('[aria-label^="Remove graph"]').click(); releaseBlocked();
  await apply({ start: '2022-01-01', end: '2022-01-31' });
  assert.equal(cards().length, 1); assert.match($('ex-table-body').textContent, /2022-01-01/);
});

test('linked keyboard cursors use matching dates and frost charts keep unique accessible IDs', async t => {
  const { $, window, apply } = await explorer(t, { workspaceMode: true });
  $('workspace-add').click();
  const cards = () => [...window.document.querySelectorAll('.graph-card')];
  await settle(() => cards().length === 2 && !cards()[1].querySelector('[id$="explorer-results"]').hidden);
  $('workspace-link-cursor').click();
  $('ex-chart').dispatchEvent(new window.KeyboardEvent('keydown', { key: 'Home' }));
  assert.match(cards()[1].querySelector('[id$="ex-inspection"]').textContent, /2020-01-01/);
  await apply({ view: 'frost' });
  const second = cards()[1]; second.querySelector('[id$="ex-view"]').value = 'frost';
  second.querySelector('form').dispatchEvent(new window.Event('submit', { bubbles: true, cancelable: true }));
  await settle(() => second.querySelector('[id$="ex-title"]').textContent.includes('Freeze') && !second.querySelector('[id$="explorer-results"]').hidden);
  const ids = [...window.document.querySelectorAll('[id]')].map(n => n.id);
  assert.equal(ids.length, new Set(ids).size);
  for (const svg of window.document.querySelectorAll('.graph-card svg[aria-labelledby]')) for (const id of svg.getAttribute('aria-labelledby').split(' ')) assert.ok(window.document.getElementById(id));
});

test('workspace URLs restore all cards and missing saved locations fail within their own card', async t => {
  const config = { version: 2, graphs: [
    { station: 'demo', query: 'view=temperature&start=2020-01-01&end=2020-12-31&unit=f' },
    { station: 'demo', query: 'view=calendar&start=2021-01-01&end=2021-12-31&unit=c' },
  ], links: { dates: false, location: false, cursor: true }, layout: 'equal' };
  const query = '?workspace=' + encodeURIComponent(JSON.stringify(config));
  const { $, window } = await explorer(t, { workspaceMode: true, query });
  const cards = [...window.document.querySelectorAll('.graph-card')];
  assert.equal(cards.length, 2); assert.equal($('ex-unit').value, 'f');
  assert.equal(cards[1].querySelector('[id$="ex-view"]').value, 'calendar');
  assert.equal($('workspace-link-cursor').checked, true); assert.equal($('workspace-layout').value, 'equal');
  config.graphs[1].station = 'unavailable_location';
  const failed = await explorer(t, { workspaceMode: true, query: '?workspace=' + encodeURIComponent(JSON.stringify(config)) });
  const failedCards = [...failed.window.document.querySelectorAll('.graph-card')];
  assert.equal(failed.$('explorer-results').hidden, false);
  assert.equal(failedCards[1].querySelector('[id$="explorer-results"]').hidden, true);
  assert.match(failedCards[1].querySelector('[id$="explorer-status"]').textContent, /unavailable/);
});

test('all 27 supplied parameters are selectable with original units and no inferred interval', async t => {
  const { $, apply, requests, blobs } = await explorer(t, { query: '?view=parameters&station=demo&parameter=surface_pressure&start=2020-02-28&end=2020-03-01' });
  assert.equal($('ex-parameter').options.length, 27);
  assert.match($('ex-description').textContent, /Pa.*interval is unspecified/);
  assert.match($('ex-table-body').textContent, /91000.5/);
  assert.equal($('ex-unit').closest('label').hidden, true);
  assert.equal($('ex-table-body').children.length, 72);
  assert.equal(requests.filter(path => path.includes('/parameters/hourly/')).length, 2);
  $('ex-download').click(); const pressure = await [...blobs.values()].at(-1).text();
  assert.match(pressure, /"91000.5","Pa"/); assert.match(pressure, /"unspecified","none","3600"/);
  assert.equal(pressure.trim().split('\r\n').length, 73);
  await apply({ view: 'parameters', parameter: 'surface_solar_radiation' });
  assert.match($('ex-table-body').textContent, /123.45W\/m\^2/);
  assert.doesNotMatch($('ex-table-body').textContent, /MJ/);
  await apply({ parameter: 'relative_humidity' });
  assert.match($('ex-table-body').textContent, /1.720-1outside_screening_range_as_supplied/);
  $('ex-download').click(); assert.match(await [...blobs.values()].at(-1).text(), /"1.72","0-1","outside_screening_range_as_supplied"/);
  for (const option of [...$('ex-parameter').options]) {
    await apply({ view: 'parameters', parameter: option.value });
    assert.equal($('explorer-results').hidden, false, option.value);
    assert.equal($('ex-table-body').children.length, 72);
  }
  assert.equal($('ex-mark').value, 'scatter'); // Direction cannot imply a wraparound line average.
});

test('parameter graphs are independent across archive locations and workspace backups retain them', async t => {
  const { $, window, apply } = await explorer(t, { workspaceMode: true, query: '?view=parameters&station=demo&parameter=surface_solar_radiation&start=2020-02-28&end=2020-03-01' });
  $('workspace-add').click();
  const cards = () => [...window.document.querySelectorAll('.graph-card')];
  await settle(() => cards().length === 2 && !cards()[1].querySelector('[id$="explorer-results"]').hidden);
  const other = cards()[1];
  other.querySelector('[id$="ex-station"]').value = 'archive_point';
  other.querySelector('[id$="ex-parameter"]').value = 'snow_depth';
  other.querySelector('form').dispatchEvent(new window.Event('submit', { bubbles: true, cancelable: true }));
  await settle(() => other.querySelector('[id$="ex-title"]').textContent.includes('Snow water') && !other.querySelector('[id$="explorer-results"]').hidden);
  assert.equal($('ex-parameter').value, 'surface_solar_radiation');
  assert.match(other.querySelector('[id$="ex-description"]').textContent, /mm of water equivalent/);
  $('workspace-save').click();
  const saved = JSON.parse(window.localStorage.getItem('idwx.weather-studies.v2')).studies[0].workspace;
  assert.equal(new URLSearchParams(saved.graphs[1].query).get('parameter'), 'snow_depth');
  assert.equal(saved.graphs[1].station, 'archive_point');
  await apply({ view: 'temperature' });
  assert.match($('ex-title').textContent, /Temperature/);
  assert.equal($('ex-unit').closest('label').hidden, false);
  assert.equal(other.querySelector('[id$="ex-view"]').value, 'parameters');
});

test('modified parameter samples fail checksum validation and retry can recover original values', async t => {
  let corrupt = true;
  const { $, window } = await explorer(t, { workspaceMode: true, query: '?view=parameters&station=demo&parameter=surface_pressure&start=2020-02-28&end=2020-02-29', route: parsed => {
    if (corrupt && parsed.pathname.endsWith('/parameters/hourly/demo/2020-02.json')) {
      const doc = JSON.parse(files.get(parsed.pathname)); doc.values[0][doc.metric_ids.indexOf('surface_pressure')] = 1;
      return { ok: true, text: async () => JSON.stringify(doc), json: async () => doc };
    }
  } });
  assert.equal($('explorer-results').hidden, true); assert.match($('explorer-status').textContent, /checksum mismatch/);
  corrupt = false; $('explorer-retry').click();
  await settle(() => !$('explorer-results').hidden);
  assert.match($('ex-table-body').textContent, /91000.5/);
  assert.equal(window.document.querySelectorAll('.graph-card').length, 1);
});

test('parameter presets keep archive-only locations and supplied-value semantics', async t => {
  const { $, window } = await explorer(t, { workspaceMode: true, query: '?view=parameters&station=archive_point&parameter=surface_pressure&start=2020-02-28&end=2020-03-01' });
  $('workspace-preset').value = 'temperature';
  $('workspace-preset').dispatchEvent(new window.Event('change'));
  const cards = () => [...window.document.querySelectorAll('.graph-card')];
  await settle(() => cards().length === 3 && cards().every(card => !card.querySelector('[id$="explorer-results"]').hidden));
  assert.deepEqual(cards().map(card => card.querySelector('[id$="ex-parameter"]').value), ['temperature', 'soil_temperature_level_1', 'dewpoint_temperature']);
  assert.ok(cards().every(card => card.querySelector('[id$="ex-station"]').value === 'archive_point'));
  assert.ok(cards().every(card => card.querySelector('[id$="ex-view"]').value === 'parameters'));
  assert.match($('workspace-status').textContent, /original units/);
});
