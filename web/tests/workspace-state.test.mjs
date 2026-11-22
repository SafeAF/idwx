import assert from 'node:assert/strict';
import { test } from 'node:test';
import { DataClient } from '../js/data-client.js';
import { encodeWorkspace, linkedSelection, migrateCharts, parseWorkspaceStudies, workspace, workspaceFromURL } from '../js/workspace-state.js';

const graph = { station: 'twin_falls', query: 'view=temperature&start=2000-01-01&end=2000-12-31&unit=c' };
test('workspace state bounds, URL roundtrip, stable catalog identifiers and legacy migration', () => {
  const value = workspace({ version: 2, graphs: [graph, { ...graph, query: 'view=precipitation' }, graph], links: { dates: true } });
  const url = new URL('https://example.test/explore.html'); url.searchParams.set('workspace', encodeWorkspace(value));
  assert.deepEqual(workspaceFromURL(url), value);
  assert.equal(workspaceFromURL(new URL('https://example.test/?view=calendar')), null);
  for (const graphs of [[], [graph, graph, graph, graph], [{ ...graph, station: '../secret' }], [{ ...graph, query: 'x'.repeat(2501) }]]) assert.throws(() => workspace({ version: 2, graphs }));
  assert.throws(() => workspace({ version: 2, graphs: [graph], links: { dates: 'yes' } }));
  assert.throws(() => workspace({ version: 2, graphs: [graph], links: { script: true } }));
  const charts = [{ study: 'Cold', name: 'January', notes: '<script>plain text</script>', station: graph.station, query: graph.query, release: 'a'.repeat(64), data_kind: 'historical_reanalysis', saved_at: '2026-11-21T00:00:00Z' }];
  const studies = migrateCharts(charts);
  assert.deepEqual(parseWorkspaceStudies(JSON.stringify({ version: 2, studies })), studies);
  assert.deepEqual(studies[0].workspace.graphs, [graph]);
  assert.throws(() => parseWorkspaceStudies(JSON.stringify({ version: 2, studies: Array(51).fill(studies[0]) })));
  assert.throws(() => parseWorkspaceStudies('{corrupt'));
});

test('date links keep independent units and unsupported calendars', () => {
  const other = { station: 'point_42', query: 'view=temperature&unit=f&start=1990-01-01' };
  const linked = linkedSelection(other, graph, { dates: true, location: false });
  assert.equal(new URLSearchParams(linked.value.query).get('start'), '2000-01-01');
  assert.equal(new URLSearchParams(linked.value.query).get('unit'), 'f');
  assert.equal(linked.value.station, other.station);
  const overlay = { ...other, query: 'view=compare&compare=1980,1981' };
  const independent = linkedSelection(overlay, graph, { dates: true }).value;
  assert.equal(independent.station, overlay.station);
  assert.equal(new URLSearchParams(independent.query).get('compare'), '1980,1981');
  assert.equal(new URLSearchParams(independent.query).has('start'), false);
  assert.match(linkedSelection(overlay, graph, { dates: true }).reason, /independent/);
});

test('shared fetches deduplicate, rejected fetches retry, decoded LRU cache is bounded', async () => {
  let calls = 0, resolve;
  const client = new DataClient(new URL('https://example.test/'), { budget: 100, fetcher: () => { calls++; return new Promise(r => { resolve = r; }); } });
  const a = client.json('https://example.test/a'), b = client.json('https://example.test/a');
  resolve({ ok: true, json: async () => ({ data: 'a'.repeat(20) }) });
  assert.deepEqual(await a, await b); assert.equal(calls, 1);
  const c = client.json('https://example.test/b'); resolve({ ok: true, json: async () => ({ data: 'b'.repeat(20) }) }); await c;
  assert.ok(client.bytes <= client.budget); assert.equal(client.cache.size, 1);
  const d = client.json('https://example.test/fail'); resolve({ ok: false, status: 503 }); await assert.rejects(d, /503/);
  const retry = client.json('https://example.test/fail'); resolve({ ok: true, json: async () => ({ ok: true }) }); assert.deepEqual(await retry, { ok: true });
  assert.equal(client.pending.size, 0);
});
