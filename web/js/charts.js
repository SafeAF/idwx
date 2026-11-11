import { calendarDay, displayDate } from './format.js';

const NS = 'http://www.w3.org/2000/svg';
const COLORS = { last_spring_frost: '#36644b', first_fall_frost: '#aa5735' };

function element(tag, attributes = {}, text) {
  const node = document.createElementNS(NS, tag);
  for (const [key, value] of Object.entries(attributes)) node.setAttribute(key, value);
  if (text !== undefined) node.textContent = text;
  return node;
}

export function renderHistory(container, records, summaries, selectedYear) {
  container.replaceChildren();
  const observed = records.filter(r => r.last_spring_frost.status === 'observed' || r.first_fall_frost.status === 'observed');
  if (!observed.length) {
    const message = document.createElement('p');
    message.className = 'chart-empty';
    message.textContent = 'No confirmed dates at this threshold. See the history table for coverage and event status.';
    container.append(message);
    return;
  }
  const width = 1000, height = 350, left = 52, right = 26, top = 22, bottom = 38;
  const plotWidth = width - left - right, plotHeight = height - top - bottom;
  const first = Math.min(...records.map(r => r.season_year), selectedYear);
  const last = Math.max(...records.map(r => r.season_year), selectedYear);
  const x = year => left + (first === last ? .5 : (year - first) / (last - first)) * plotWidth;
  const y = day => top + ((day - 1) / 365) * plotHeight;
  const svg = element('svg', { viewBox: `0 0 ${width} ${height}`, role: 'img', 'aria-labelledby': 'chart-title chart-desc' });
  svg.append(element('title', { id: 'chart-title' }, 'Annual confirmed spring and fall freeze dates'));
  svg.append(element('desc', { id: 'chart-desc' }, 'Years run from left to right. Calendar dates run from January at the top to December at the bottom. Green dots are last spring freezes; rust dots are first fall freezes. Shaded historical P10 to P90 bands and dashed medians appear where sufficient history exists. Missing or provisional events have no dot. All values are available in the history table.'));
  for (const [date, label] of [['2000-01-01', 'Jan'], ['2000-04-01', 'Apr'], ['2000-07-01', 'Jul'], ['2000-10-01', 'Oct'], ['2000-12-31', 'Dec']]) {
    const yy = y(calendarDay(date));
    svg.append(element('line', { x1: left, x2: width - right, y1: yy, y2: yy, class: 'grid' }));
    svg.append(element('text', { x: left - 12, y: yy + 4, 'text-anchor': 'end' }, label));
  }
  for (const [name, color] of Object.entries(COLORS)) {
    const summary = summaries[name];
    if (summary?.status !== 'available') continue;
    const start = y(summary.p10.calendar_day), end = y(summary.p90.calendar_day);
    svg.append(element('rect', { x: left, y: start, width: plotWidth, height: Math.max(2, end - start), fill: color, opacity: '.09' }));
    const median = y(summary.p50.calendar_day);
    svg.append(element('line', { x1: left, x2: width - right, y1: median, y2: median, stroke: color, class: 'median' }));
  }
  svg.append(element('line', { x1: x(selectedYear), x2: x(selectedYear), y1: top, y2: height - bottom, class: 'selected' }));
  const years = new Set([first, last]);
  const step = Math.max(1, Math.ceil((last - first) / 6));
  for (let year = first + step; year < last - step / 2; year += step) years.add(year);
  for (const year of years) svg.append(element('text', { x: x(year), y: height - 10, 'text-anchor': 'middle' }, year));
  for (const record of observed) for (const [name, color] of Object.entries(COLORS)) {
    const event = record[name];
    if (event.status !== 'observed' || !event.date) continue;
    const dot = element('circle', { cx: x(record.season_year), cy: y(calendarDay(event.date)), r: record.season_year === selectedYear ? 5.5 : 3.8, fill: color, stroke: '#fffef9', 'stroke-width': 1.5 });
    dot.append(element('title', {}, `${record.season_year} ${name === 'last_spring_frost' ? 'spring' : 'fall'}: ${displayDate(event.date, true)}`));
    svg.append(dot);
  }
  container.append(svg);
}
