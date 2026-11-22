import { createExplorer } from './explorer.js';
import { DataClient } from './data-client.js';
import { parseStudies } from './analytics.js';
import { STUDIES_KEY, MAX_URL_LENGTH, encodeWorkspace, linkedSelection, migrateCharts, parseWorkspaceStudies, workspace, workspaceFromURL } from './workspace-state.js';

const $ = id => document.getElementById(id);
const el = (tag, text, cls) => { const n = document.createElement(tag); if (text !== undefined) n.textContent = text; if (cls) n.className = cls; return n; };
const button = (text, action) => { const n = el('button', text, 'button'); n.type = 'button'; n.onclick = action; return n; };
const main = $('explorer'), template = document.createElement('div');
let first = $('explorer-status');
while (first && first.tagName !== 'NOSCRIPT') { const next = first.nextElementSibling; template.append(first); first = next; }
const shell = el('section', undefined, 'workspace-shell');
shell.setAttribute('aria-label', 'Climate analysis workspace');
const toolbar = el('div', undefined, 'workspace-toolbar panel'), graphs = el('div', undefined, 'workspace-graphs');
graphs.id = 'workspace-graphs';
const message = el('p', '', 'small muted'); message.id = 'workspace-status'; message.setAttribute('role', 'status');
const client = new DataClient(new URL('.', location.href));
let cards = [], nextID = 1, restoring = false, restoration = 0, links = { dates: false, location: false, cursor: false }, layout = 'focus';
const current = () => workspace({ version: 2, graphs: cards.map(card => card.api.snapshot() || card.initial).filter(Boolean), links, layout });
const say = text => { message.textContent = text; };
function writeURL() {
  if (restoring || cards.some(card => !card.api.snapshot())) return;
  const url = new URL(location.href);
  // Keep single-graph links compatible with the existing share format.
  if (cards.length === 1) {
    url.search = cards[0].api.snapshot().query;
    url.searchParams.set('station', cards[0].api.snapshot().station);
    if (Object.values(links).some(Boolean) || layout !== 'focus') url.searchParams.set('workspace', encodeWorkspace(current()));
  } else { url.search = ''; url.searchParams.set('workspace', encodeWorkspace(current())); }
  if (url.href.length > MAX_URL_LENGTH) { say('This workspace is too large for a share URL. Download a study backup to keep it.'); return; }
  if (url.href !== location.href) history.pushState(null, '', url);
}
function updateLayout() {
  graphs.dataset.count = cards.length; graphs.dataset.layout = layout;
  add.disabled = cards.length >= 3;
  for (const card of cards) { card.remove.disabled = cards.length === 1; card.duplicate.disabled = cards.length >= 3; }
}
async function changed(card, value) {
  if (restoring) return;
  card.initial = value; card.title.textContent = card.api?.elements.get('ex-title').textContent || 'Historical graph';
  card.subtitle.textContent = `${value.station} · ${new URLSearchParams(value.query).get('parameter') || new URLSearchParams(value.query).get('metric')} · independently configurable`;
  const updates = [];
  for (const other of cards) if (other !== card && other.api.snapshot()) {
    const result = linkedSelection(other.api.snapshot(), value, links);
    if (result.reason) say(result.reason);
    if (links.location && ![...other.api.elements.get('ex-station').options].some(option => option.value === value.station)) {
      result.value.station = other.api.snapshot().station;
      say('This location is available for supplied parameters but not for the other graph’s daily products. Its location stays independent.');
    }
    if (JSON.stringify(result.value) !== JSON.stringify(other.api.snapshot())) updates.push(other.api.setSelection(result.value, false));
  }
  await Promise.allSettled(updates);
  writeURL();
}
function addCard(initial, { primary = false } = {}) {
  if (cards.length >= 3) return null;
  const id = `graph-${nextID++}`, article = el('article', undefined, 'graph-card'); article.dataset.graphId = id;
  const heading = el('header', undefined, 'graph-card-heading'), title = el('h2', 'Historical graph'), subtitle = el('p', 'Loading selection…', 'small muted');
  title.id = `${id}-title`; article.setAttribute('aria-labelledby', title.id);
  const text = el('div'); text.append(title, subtitle);
  const actions = el('div', undefined, 'graph-card-actions');
  const card = { id, article, title, subtitle, initial, api: null };
  const edit = button('Edit', () => { card.api.edit(); edit.setAttribute('aria-expanded', String(!card.api.elements.get('explorer-filters').hidden)); }); edit.setAttribute('aria-expanded', 'true');
  const duplicate = button('Duplicate', () => { const copy = addCard(card.api.snapshot()); copy?.api.ready.then(writeURL); });
  const expand = button('Expand', () => { const expanded = article.classList.toggle('graph-expanded'); expand.textContent = expanded ? 'Collapse' : 'Expand'; expand.setAttribute('aria-label', `${expand.textContent} ${id.replace('-', ' ')}`); expand.setAttribute('aria-expanded', String(expanded)); });
  const download = button('Download CSV', () => card.api.download());
  const remove = button('Remove', () => {
    if (cards.length <= 1) return;
    card.api.destroy(); cards = cards.filter(c => c !== card); article.remove(); updateLayout(); writeURL();
    cards[0].title.tabIndex = -1; cards[0].title.focus();
  });
  card.remove = remove; card.duplicate = duplicate;
  for (const action of [edit, duplicate, expand, download, remove]) action.setAttribute('aria-label', `${action.textContent} ${id.replace('-', ' ')}`);
  actions.append(edit, duplicate, expand, download, remove); heading.append(text, actions);
  const body = template.cloneNode(true); body.className = 'graph-card-body';
  // Legacy charts migrate to the workspace library. Preserve their original
  // storage and the standalone card API without showing two study editors.
  body.querySelector('#studies').hidden = true;
  article.append(heading, body); graphs.append(article); cards.push(card);
  card.api = createExplorer(body, { client, query: initial?.query ?? location.search, station: initial?.station,
    strictStation: Boolean(initial),
    idPrefix: primary ? '' : `${id}-`, onChange: value => changed(card, value),
    onReady: value => { if (value) changed(card, value); } });
  updateLayout(); return card;
}
const add = button('Add graph', () => { const card = addCard(cards.at(-1)?.api.snapshot()); card?.api.ready.then(writeURL); }); add.id = 'workspace-add';
const layoutLabel = el('label', 'Layout '), layoutSelect = el('select'); layoutSelect.id = 'workspace-layout';
for (const [value, text] of [['focus', 'Primary graph + supporting graphs'], ['equal', 'Equal graph columns']]) { const option = el('option', text); option.value = value; layoutSelect.append(option); }
layoutSelect.onchange = () => { layout = layoutSelect.value; updateLayout(); writeURL(); }; layoutLabel.append(layoutSelect);
const linkGroup = el('fieldset', undefined, 'workspace-links'); linkGroup.append(el('legend', 'Link graphs'));
for (const [key, text] of [['dates', 'Date range'], ['location', 'Location'], ['cursor', 'Inspection cursor']]) {
  const label = el('label'), input = el('input'); input.type = 'checkbox'; input.id = `workspace-link-${key}`;
  input.onchange = () => { links[key] = input.checked; writeURL(); say(key === 'cursor' ? 'Inspection links matching UTC sample timestamps, daily dates, or calendar month/day positions. Unsupported combinations stay independent.' : 'Links apply to subsequent edits. Each graph retains its own settings.'); };
  label.append(input, document.createTextNode(' ' + text)); linkGroup.append(label);
}
const check = button('Check for new data', async () => {
  try { say(await client.newerRelease() ? 'A newer release is available. Reload the workspace to use it in every graph.' : 'All graphs use the current publication.'); }
  catch (error) { say(error.message); }
});
const presetLabel = el('label', 'Start with '), preset = el('select'); preset.id = 'workspace-preset';
for (const [value, text] of [['', 'Choose a preset'], ['temperature', 'Three temperature series'], ['water', 'Temperature + precipitation']]) {
  const option = el('option', text); option.value = value; preset.append(option);
}
preset.onchange = async () => {
  const source = cards[0]?.api.snapshot(); if (!source || !preset.value) return;
  const settings = new URLSearchParams(source.query), supplied = settings.get('view') === 'parameters';
  if (!supplied) settings.set('view', 'temperature');
  const selections = supplied ? (preset.value === 'temperature' ? ['temperature', 'soil_temperature_level_1', 'dewpoint_temperature'] : ['temperature', 'total_precipitation']).map(parameter => {
    const params = new URLSearchParams(settings); params.set('parameter', parameter); return { station: source.station, query: params.toString() };
  }) : preset.value === 'temperature' ? ['tmin_c', 'tmean_c', 'tmax_c'].map(metric => {
    const params = new URLSearchParams(settings); params.set('metric', metric); return { station: source.station, query: params.toString() };
  }) : [{ station: source.station, query: settings.toString() }, { station: source.station, query: 'view=precipitation' }];
  await restore(workspace({ version: 2, graphs: selections }), true);
  say(supplied ? 'Independent supplied-value graphs use the same UTC selection and their original units. No aggregation is inferred.' : preset.value === 'temperature' ? 'Three independent temperature selections. Each card emphasizes a different series.' : 'Temperature and total water equivalent use independently reviewed coverage dates.');
  preset.value = '';
};
presetLabel.append(preset);
toolbar.append(add, presetLabel, layoutLabel, linkGroup, check); shell.append(toolbar, message, graphs); main.append(shell);

// Cursor events contain labels, never pixel offsets. Other graphs decide if
// they can represent that date exactly and never substitute a daily value for a month.
graphs.addEventListener('idwx-inspect', event => {
  if (!links.cursor || !event.detail || event.detail.linked) return;
  const source = cards.find(c => c.article.contains(event.target));
  for (const card of cards) if (card !== source) card.api.elements.get('ex-chart').dispatchEvent(new CustomEvent('idwx-inspect-linked', { detail: event.detail }));
});
async function restore(value, push = false) {
  const generation = ++restoration;
  restoring = true;
  try {
    for (const card of cards) card.api.destroy(); cards = []; graphs.replaceChildren();
    links = value.links; layout = value.layout;
    for (const key of Object.keys(links)) $(`workspace-link-${key}`).checked = links[key];
    layoutSelect.value = layout;
    const created = value.graphs.map((graph, i) => addCard(graph, { primary: i === 0 }));
    await Promise.all(created.map(card => card.api.ready));
  } finally { if (generation === restoration) { restoring = false; updateLayout(); } }
  if (push && generation === restoration) writeURL();
}
window.addEventListener('popstate', async () => {
  try {
    const url = new URL(location.href), stored = workspaceFromURL(url);
    await restore(stored || workspace({ version: 2, graphs: [{ station: url.searchParams.get('station') || client.manifest && cards[0]?.api.snapshot()?.station || 'twin_falls', query: url.searchParams.toString() }] }));
  } catch (error) { say(error.message); }
});

const library = el('section', undefined, 'panel workspace-studies'); library.id = 'workspace-studies';
library.append(el('p', 'A complete workspace', 'eyebrow'), el('h2', 'Save your study'), el('p', 'Keep all graph settings, layout, links and notes in this browser. Download a backup to move a study to another device. Saved selections recalculate against the loaded release.'));
const form = el('form', undefined, 'study-form'), name = el('input'), notes = el('textarea'); name.maxLength = 120; name.required = true; name.value = 'My climate study'; name.id = 'workspace-study-name'; notes.maxLength = 500; notes.rows = 2; notes.id = 'workspace-study-notes';
for (const [label, input] of [['Study name', name], ['Notes', notes]]) { const n = el('label', label); n.append(input); form.append(n); }
const save = el('button', 'Save workspace', 'button'); save.id = 'workspace-save'; save.type = 'submit'; form.append(save);
const status = el('p'); status.id = 'workspace-studies-status'; status.setAttribute('role', 'status');
const list = el('div'); list.id = 'workspace-studies-list'; let studies = [], storageOK = true;
const backup = value => { const blob = new Blob([JSON.stringify(value, null, 2) + '\n'], { type: 'application/json' }); const url = URL.createObjectURL(blob), anchor = el('a'); anchor.href = url; anchor.download = 'idwx-workspace-studies.json'; anchor.click(); setTimeout(() => URL.revokeObjectURL(url), 1000); };
const exportButton = button('Download study backup', () => backup({ version: 2, studies })); exportButton.id = 'workspace-studies-download';
const importLabel = el('label', 'Import study backup', 'button'), file = el('input'); file.type = 'file'; file.accept = 'application/json,.json'; file.id = 'workspace-studies-import'; importLabel.append(file);
const print = button('Print study report', () => window.print()); print.id = 'workspace-print';
const writeStudies = values => {
  try {
    const text = JSON.stringify({ version: 2, studies: values }); const validated = parseWorkspaceStudies(text);
    localStorage.setItem(STUDIES_KEY, text);
    if (localStorage.getItem(STUDIES_KEY) !== text) throw new Error('Storage verification failed');
    studies = validated; renderStudies(); return true;
  } catch { status.textContent = 'Could not save the backup. Existing studies are preserved; storage may be full or unavailable.'; return false; }
};
function renderStudies() {
  list.replaceChildren(); exportButton.disabled = studies.length === 0;
  for (const [index, saved] of studies.entries()) {
    const row = el('article', undefined, 'saved-chart');
    row.append(el('h3', saved.name), el('p', saved.notes), el('p', `${saved.workspace.graphs.length} graphs · ${saved.data_kind} · saved ${saved.saved_at.slice(0, 10)} · release ${saved.release.slice(0, 12)}${client.manifest && saved.release !== client.manifest.release_id ? ' · data changed since saving' : ''}`, 'small muted'));
    row.append(button('Open study', async () => {
      await restore(saved.workspace, true); name.value = saved.name; notes.value = saved.notes;
      status.textContent = saved.release === client.manifest?.release_id ? 'Opened with the same release.' : 'Release changed. These are saved filters recalculated with the loaded publication.';
    }), button('Remove study', () => { if (storageOK) writeStudies(studies.filter((_, i) => i !== index)); }));
    list.append(row);
  }
  if (!studies.length) list.append(el('p', 'No saved workspaces yet.', 'muted'));
}
form.onsubmit = event => {
  event.preventDefault();
  if (!storageOK || !client.manifest || cards.some(card => !card.api.snapshot() || card.api.elements.get('explorer-results').hidden)) { status.textContent = 'Wait until each graph has a valid displayed selection.'; return; }
  if (writeStudies([...studies, { name: name.value.trim(), notes: notes.value, saved_at: new Date().toISOString(), release: client.manifest.release_id, data_kind: client.manifest.data_kind, workspace: current() }])) status.textContent = 'Complete workspace saved in this browser.';
};
file.onchange = async () => {
  if (!storageOK || !file.files?.[0]) return;
  try {
    if (file.files[0].size > 1000000) throw new Error('Backup is too large.');
    const text = await file.files[0].text(); let incoming;
    if (JSON.parse(text).version === 1) incoming = migrateCharts(parseStudies(text)); else incoming = parseWorkspaceStudies(text);
    const merged = [...studies], known = new Set(studies.map(s => JSON.stringify(s)));
    for (const saved of incoming) if (!known.has(JSON.stringify(saved))) { merged.push(saved); known.add(JSON.stringify(saved)); }
    if (writeStudies(merged)) status.textContent = 'Backup imported. Existing studies retained.';
  } catch { status.textContent = 'Could not import this IDWX backup. Existing studies are preserved.'; }
  file.value = '';
};
library.append(form, status, exportButton, importLabel, print, list); main.append(library);
try {
  const stored = localStorage.getItem(STUDIES_KEY);
  if (stored !== null) studies = parseWorkspaceStudies(stored);
  else {
    const old = localStorage.getItem('idwx.weather-studies.v1');
    if (old !== null) { const migrated = migrateCharts(parseStudies(old)); if (!writeStudies(migrated)) storageOK = false; }
  }
} catch { storageOK = false; status.textContent = 'Saved studies could not be read. Existing storage is preserved; graphs remain available.'; }
save.disabled = file.disabled = !storageOK; renderStudies();
try {
  const stored = workspaceFromURL(new URL(location.href));
  if (stored) await restore(stored);
  else { const card = addCard(null, { primary: true }); await card.api.ready; }
  $('parameters-shortcut').hidden = !client.manifest?.parameters_url;
  renderStudies();
} catch (error) { say(error.message); if (!cards.length) addCard(null, { primary: true }); }
