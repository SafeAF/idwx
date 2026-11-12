// Execute the real modules in a DOM. This does not verify browser layout.
import assert from 'node:assert/strict';
import { execFileSync } from 'node:child_process';
import { readFile, readdir } from 'node:fs/promises';
import { existsSync, mkdirSync, mkdtempSync, rmSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import vm from 'node:vm';
import { after, test } from 'node:test';
import { JSDOM, VirtualConsole } from 'jsdom';

const root = fileURLToPath(new URL('../../', import.meta.url));
const temporaryRoot = path.join(root, '.tmp');
mkdirSync(temporaryRoot, { recursive: true });
const fixture = mkdtempSync(path.join(temporaryRoot, 'dashboard-tests-'));
const python = process.env.IDWX_PYTHON || path.join(root, '.venv', process.platform === 'win32' ? 'Scripts/python.exe' : 'bin/python');
assert.ok(existsSync(python), 'Set IDWX_PYTHON to an environment with idwx installed');
execFileSync(python, ['-c', `
from dataclasses import replace
from pathlib import Path
import sys
from idwx.config import load_config, StationMeta
from idwx.web_export import export_web_data
cfg = replace(load_config('configs/idaho.yml'), backtest={'start_year':2000,'end_year':2022})
export_web_data(cfg,[StationMeta('alpha','Alpha'),StationMeta('beta','Beta')],Path(sys.argv[1]),demo=True)
`, fixture], { cwd: root });
after(() => {
  const relative = path.relative(temporaryRoot, fixture);
  assert.ok(relative && !relative.startsWith('..') && !path.isAbsolute(relative));
  rmSync(fixture, { recursive: true, force: true });
});
const files = new Map();
async function inventory(directory) {
  for (const entry of await readdir(directory, { withFileTypes: true })) {
    const target = path.join(directory, entry.name);
    if (entry.isDirectory()) await inventory(target);
    else files.set('/almanac/' + path.relative(fixture, target).split(path.sep).join('/'), await readFile(target, 'utf8'));
  }
}
await inventory(fixture);
const manifest = JSON.parse(files.get('/almanac/data/manifest.json'));
const stationDoc = id => JSON.parse(files.get(`/almanac/data/${manifest.release_id}/stations/${id}.json`));
const html = await readFile(path.join(root, 'web/index.html'), 'utf8');
async function settle(predicate) {
  const until = Date.now() + 3000;
  while (!predicate()) {
    if (Date.now() > until) throw new Error('Dashboard did not reach its expected state');
    await new Promise(resolve => setImmediate(resolve));
  }
}
async function dashboard(t, { query = '', route } = {}) {
  const errors = [], requests = [], downloads = [], blobs = new Map();
  const console = new VirtualConsole();
  console.on('jsdomError', error => errors.push(error));
  const dom = new JSDOM(html, { url: `https://idwx.test/almanac/${query}`, runScripts: 'outside-only', virtualConsole: console });
  t.after(() => { dom.window.close(); assert.deepEqual(errors, []); });
  const { window } = dom;
  window.Blob = Blob;
  window.URL.createObjectURL = blob => { const url = `blob:test-${blobs.size}`; blobs.set(url, blob); return url; };
  window.URL.revokeObjectURL = url => blobs.delete(url);
  window.HTMLAnchorElement.prototype.click = function () { downloads.push({ href: this.href, name: this.download }); };
  window.fetch = async (url, options) => {
    const parsed = new URL(url);
    requests.push(parsed.pathname);
    const response = route && await route(parsed, options);
    if (response) return response;
    return { ok: files.has(parsed.pathname), status: files.has(parsed.pathname) ? 200 : 404,
      json: async () => JSON.parse(files.get(parsed.pathname)) };
  };
  const context = dom.getInternalVMContext(), modules = new Map();
  function moduleFor(url) {
    if (!modules.has(url.href)) modules.set(url.href, readFile(url, 'utf8').then(source => new vm.SourceTextModule(source, { context, identifier: url.href })));
    return modules.get(url.href);
  }
  const app = await moduleFor(pathToFileURL(path.join(root, 'web/js/app.js')));
  await app.link((specifier, reference) => moduleFor(new URL(specifier, reference.identifier)));
  await app.evaluate();
  const $ = id => window.document.getElementById(id);
  await settle(() => !$('dashboard').hidden || !$('retry').hidden);
  const change = (id, value) => { $(id).value = value; $(id).dispatchEvent(new window.Event('change', { bubbles: true })); };
  return { window, $, change, requests, downloads, blobs };
}

test('shared URL restores station, threshold, season, units and synthetic notice', async t => {
  const { $, window, requests } = await dashboard(t, { query: '?station=beta&threshold=-2&year=2020&unit=f' });
  assert.equal($('station-title').textContent, 'Beta');
  assert.equal($('threshold').value, '-2');
  assert.match($('threshold').selectedOptions[0].textContent, /28\.4 °F/);
  assert.equal($('year').value, '2020');
  assert.match($('notice').textContent, /Synthetic demonstration/);
  assert.match($('cards').textContent, /Not verified/);
  assert.match($('interval-note').textContent, /July freeze was observed/);
  assert.match($('freshness').textContent, /2022-12-31/);
  assert.equal(new URL(window.location.href).searchParams.get('unit'), 'f');
  assert.equal(requests.length, 3);
  assert.ok(!requests.some(url => url.endsWith('/alpha.json')));
  assert.equal($('full-download').pathname, '/almanac/' + manifest.history_csv_url);
});

test('baseline table shows exported extremes and included/excluded years at the selected threshold', async t => {
  const { $, change } = await dashboard(t);
  const verify = threshold => {
    const doc = stationDoc('alpha');
    const summary = doc.summaries.find(row => row.threshold_c === threshold);
    const rows = [...$('baseline-rows').children];
    for (const [index, name] of ['last_spring_frost', 'first_fall_frost'].entries()) {
      const distribution = summary[name], cells = rows[index].children;
      assert.match(cells[0].textContent, /2000–2022/);
      assert.ok(cells[2].textContent.includes(new Intl.DateTimeFormat('en-US', {
        month: 'short', day: 'numeric', timeZone: 'UTC',
      }).format(new Date(`2000-${distribution.earliest.month_day}T00:00:00Z`))));
      assert.ok(cells[3].textContent.startsWith(`${distribution.valid_year_count}/23 years`));
      assert.ok(cells[4].textContent.startsWith(`${distribution.status_counts.no_event_in_complete_window ?? 0} complete without an event`));
      assert.ok(cells[4].textContent.includes(`${distribution.status_counts.incomplete_window ?? 0} incomplete`));
    }
  };
  verify(0);
  change('threshold', '-2'); verify(-2);
  assert.match($('baseline-caption').textContent, /-2 °C/);
  const before = $('baseline-rows').textContent;
  change('year', '2021');
  assert.equal($('baseline-rows').textContent, before);
});

test('a short baseline shows observed extremes without inventing typical dates', async t => {
  const doc = stationDoc('alpha');
  for (const summary of doc.summaries) for (const name of ['last_spring_frost', 'first_fall_frost']) {
    Object.assign(summary[name], {
      baseline_period: { start_year: 2000, end_year: 2001 }, status: 'insufficient_history',
      valid_year_count: 2, p10: null, p50: null, p90: null,
    });
  }
  const { $ } = await dashboard(t, { route: async url => url.pathname.endsWith('/alpha.json') ? { ok: true, json: async () => doc } : null });
  for (const row of $('baseline-rows').children) {
    assert.match(row.children[0].textContent, /2000–2001/);
    assert.match(row.children[1].textContent, /Insufficient history/);
    assert.match(row.children[2].textContent, /Earliest to latest/);
    assert.match(row.children[3].textContent, /^2\/2 years/);
  }
});

test('controls render no-event, incomplete, verified and exact threshold states', async t => {
  const { $, change, window } = await dashboard(t);
  change('year', '2019');
  assert.match($('cards').textContent, /No event found/);
  change('year', '2021');
  assert.match($('cards').textContent, /Not confirmed/);
  assert.match($('detail-rows').textContent, /May 2/);
  assert.match($('detail-rows').textContent, /1 missing/);
  change('year', '2000');
  const record = stationDoc('alpha').annual.find(row => row.season_year === 2000 && row.threshold_c === 0);
  assert.match($('cards').textContent, new RegExp(`${record.interval.frost_free_days} days`));
  change('threshold', '-2');
  const radio = window.document.querySelector('input[value="f"]');
  radio.checked = true; radio.dispatchEvent(new window.Event('change', { bubbles: true }));
  assert.match($('threshold').selectedOptions[0].textContent, /28\.4 °F/);
  assert.equal(new URL(window.location.href).searchParams.get('threshold'), '-2');
});

test('SVG, accessible table and downloaded CSV agree on the selected history', async t => {
  const { $, downloads, blobs } = await dashboard(t);
  const records = stationDoc('alpha').annual.filter(row => row.threshold_c === 0);
  assert.equal($('history-rows').children.length, records.length);
  const count = records.reduce((sum, row) => sum + ['last_spring_frost', 'first_fall_frost'].filter(name => row[name].status === 'observed').length, 0);
  assert.equal($('chart').querySelectorAll('circle').length, count);
  for (const record of records) {
    const row = [...$('history-rows').children].find(tr => tr.firstChild.textContent === String(record.season_year));
    assert.ok(row);
    assert.equal(row.querySelector('th').scope, 'row');
    for (const [index, name] of ['last_spring_frost', 'first_fall_frost'].entries()) {
      assert.match(row.children[index + 1].textContent, record[name].status === 'observed' ? /Confirmed event/ : /Incomplete|No event|Insufficient/);
    }
  }
  $('download').click();
  assert.equal(downloads[0].name, 'alpha-0C-frost-history.csv');
  const csv = await blobs.get(downloads[0].href).text();
  const parsed = JSON.parse(execFileSync(python, ['-c', 'import csv,json,sys;print(json.dumps(list(csv.DictReader(sys.stdin))))'], { input: csv, encoding: 'utf8' }));
  assert.equal(parsed.length, records.length);
  assert.ok(parsed.every(row => row.station_id === 'alpha' && Number(row.threshold_c) === 0 && row.data_kind === 'synthetic_demo'));
  assert.equal(parsed.find(row => row.season_year === '2021').last_spring_frost_date, '');
  assert.equal(parsed.find(row => row.season_year === '2021').last_spring_frost_candidate_date, '2021-05-02');
});

test('station failure keeps recovery available and selecting a station restores the view', async t => {
  let fail = true;
  const { $, change } = await dashboard(t, { route: async url => fail && url.pathname.endsWith('/beta.json') ? { ok: false, status: 503 } : null });
  change('station', 'beta');
  await settle(() => !$('retry').hidden);
  assert.match($('status').textContent, /HTTP 503/);
  assert.equal($('dashboard').hidden, true);
  assert.equal($('station').disabled, false);
  fail = false; change('station', 'beta');
  await settle(() => !$('dashboard').hidden);
  assert.equal($('station-title').textContent, 'Beta');
  assert.equal($('retry').hidden, true);
});

test('failed manifest request is recoverable through the retry button', async t => {
  let fail = true;
  const { $ } = await dashboard(t, { route: async url => fail && url.pathname.endsWith('/manifest.json') ? { ok: false, status: 404 } : null });
  assert.match($('status').textContent, /HTTP 404/);
  fail = false; $('retry').click();
  await settle(() => !$('dashboard').hidden);
  assert.equal($('station-title').textContent, 'Alpha');
});

test('mixed-release station data never becomes a visible dashboard', async t => {
  const { $ } = await dashboard(t, { route: async url => url.pathname.endsWith('/alpha.json') ? {
    ok: true, json: async () => ({ ...stationDoc('alpha'), release_id: '0'.repeat(64) }),
  } : null });
  assert.equal($('dashboard').hidden, true);
  assert.match($('status').textContent, /mixed release/);
});

test('semantic controls, focus targets and table alternatives remain available', async t => {
  const { $, window } = await dashboard(t);
  for (const id of ['station', 'threshold', 'year']) {
    assert.ok($(id).closest('label'));
    $(id).focus(); assert.equal(window.document.activeElement, $(id));
  }
  assert.equal($('status').getAttribute('role'), 'status');
  assert.equal($('main').tabIndex, -1);
  assert.equal($('chart').tabIndex, 0);
  assert.ok(window.document.querySelector('[aria-label="Historical baseline dates and coverage"] thead').textContent.includes('P10–P90'));
  assert.equal($('chart').querySelector('svg').getAttribute('role'), 'img');
  assert.ok($('chart').querySelector('#chart-desc').textContent.includes('history table'));
  assert.ok(window.document.querySelector('a.skip-link[href="#main"]'));
  for (const region of window.document.querySelectorAll('.table-scroll')) assert.equal(region.tabIndex, 0);
  assert.ok($('table-caption').textContent.includes('Alpha'));
});

test('missing early years stay on the chart axis and in the table', async t => {
  const doc = stationDoc('alpha');
  for (const threshold of doc.thresholds_c) {
    const row = structuredClone(doc.annual.find(record => record.threshold_c === threshold));
    row.season_year = 1999;
    for (const name of ['last_spring_frost', 'first_fall_frost']) Object.assign(row[name], {
      status: 'insufficient_data', date: null, tmin_c: null, candidate_date: null, candidate_tmin_c: null,
      valid_days: 0, missing_days: row[name].expected_days, coverage_fraction: 0,
    });
    Object.assign(row.interval, { status: 'missing_dependency', date_difference_days: null, frost_free_days: null, summer_freeze_observed: false });
    doc.annual.push(row);
  }
  const { $ } = await dashboard(t, { route: async url => url.pathname.endsWith('/alpha.json') ? { ok: true, json: async () => doc } : null });
  assert.equal($('history-rows').children.length, 24);
  assert.ok([...$('chart').querySelectorAll('text')].some(node => node.textContent === '1999'));
});

test('entirely unavailable history has an empty chart and accessible status table', async t => {
  const doc = stationDoc('alpha');
  doc.annual = doc.annual.filter(row => row.season_year === 2019);
  for (const summary of doc.summaries) for (const name of ['last_spring_frost', 'first_fall_frost']) Object.assign(summary[name], {
    status: 'insufficient_history', valid_year_count: 0, p10: null, p50: null, p90: null,
  });
  const { $ } = await dashboard(t, { route: async url => url.pathname.endsWith('/alpha.json') ? { ok: true, json: async () => doc } : null });
  assert.match($('chart').textContent, /No confirmed dates/);
  assert.match($('cards').textContent, /Insufficient history/);
  assert.match($('baseline-rows').textContent, /No confirmed dates/);
  assert.match($('baseline-rows').textContent, /0\/1 years/);
  assert.equal($('history-rows').children.length, 1);
});

test('invalid URL selections and popstate normalize to a valid station', async t => {
  const { $, window } = await dashboard(t, { query: '?station=unknown&threshold=999&year=12&unit=invalid' });
  assert.equal($('station').value, 'alpha');
  assert.equal($('threshold').value, '0');
  assert.equal($('year').value, '2022');
  window.history.pushState(null, '', '?station=unknown');
  window.dispatchEvent(new window.PopStateEvent('popstate'));
  await settle(() => !$('dashboard').hidden);
  assert.equal($('station-title').textContent, 'Alpha');
  assert.equal(new URL(window.location.href).searchParams.get('station'), 'alpha');
});

test('malformed station content fails with an actionable error rather than a partial view', async t => {
  const doc = stationDoc('alpha');
  delete doc.annual[0].last_spring_frost;
  const { $ } = await dashboard(t, { route: async url => url.pathname.endsWith('/alpha.json') ? { ok: true, json: async () => doc } : null });
  assert.equal($('dashboard').hidden, true);
  assert.match($('status').textContent, /Station data is invalid or incomplete/);
  assert.equal($('retry').hidden, false);
});

test('a late station response cannot overwrite a more recent station selection', async t => {
  let release;
  const { $, change } = await dashboard(t, { route: async url => {
    if (!url.pathname.endsWith('/beta.json')) return null;
    await new Promise(resolve => { release = resolve; });
    return { ok: true, json: async () => stationDoc('beta') };
  } });
  change('station', 'beta');
  await settle(() => !!release);
  change('station', 'alpha');
  await settle(() => !$('dashboard').hidden);
  release();
  await new Promise(resolve => setImmediate(resolve));
  assert.equal($('station-title').textContent, 'Alpha');
});
