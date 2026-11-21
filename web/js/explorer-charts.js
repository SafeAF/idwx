import { calendarDay } from './format.js';
import { METRICS, aggregate, finite, shown, toUnit } from './analytics.js';
const NS = 'http://www.w3.org/2000/svg';
const colors = ['#36644b', '#aa5735', '#385f91', '#74509a', '#56645d'];
const make = (tag, attrs = {}, text) => {
  const el = document.createElementNS(NS, tag);
  Object.entries(attrs).forEach(([key, value]) => el.setAttribute(key, value));
  if (text !== undefined) el.textContent = text;
  return el;
};

export function lineChart(container, series, { title, unit = '', inspect }) {
  container.replaceChildren();
  const points = series.flatMap(s => s.points), valid = points.filter(p => finite(p.y));
  if (!valid.length) { container.textContent = 'No complete values to plot in this selection. See the table and coverage.'; return; }
  const width = 1000, height = 360, left = 65, right = 28, top = 25, bottom = 48;
  const xs = points.map(p => p.x), ys = valid.map(p => p.y);
  const minX = Math.min(...xs), maxX = Math.max(...xs);
  const minY = Math.min(...ys), maxY = Math.max(...ys), padding = Math.max(1, (maxY - minY) * .12);
  const x = value => left + (value - minX) / (maxX - minX || 1) * (width - left - right);
  const y = value => top + (maxY + padding - value) / (maxY - minY + 2 * padding) * (height - top - bottom);
  const svg = make('svg', { viewBox: `0 0 ${width} ${height}`, role: 'img', 'aria-label': title });
  svg.append(make('title', {}, title), make('desc', {}, 'Use left/right arrow keys on the chart or select a position to read values. The table contains the selected data.'));
  for (let i = 0; i < 5; i++) {
    const value = minY + (maxY - minY) * i / 4;
    svg.append(make('line', { x1: left, x2: width - right, y1: y(value), y2: y(value), class: 'grid' }), make('text', { x: left - 10, y: y(value) + 4, 'text-anchor': 'end' }, shown(value)));
  }
  svg.append(make('text', { x: left, y: 15 }, unit));
  series.forEach((s, index) => {
    let path = '', open = false;
    for (const point of s.points) {
      if (!finite(point.y)) { open = false; continue; }
      path += `${open ? 'L' : 'M'}${x(point.x).toFixed(2)},${y(point.y).toFixed(2)} `;
      open = true;
    }
    svg.append(make('path', { d: path, fill: 'none', stroke: s.color || colors[index % colors.length], 'stroke-width': s.light ? 1.3 : 2.5, 'stroke-dasharray': s.dash || '', opacity: s.light ? .55 : 1 }));
    if (s.points.filter(p => finite(p.y)).length === 1) {
      const p = s.points.find(p => finite(p.y)); svg.append(make('circle', { cx: x(p.x), cy: y(p.y), r: 4, fill: colors[index % colors.length] }));
    }
  });
  const positions = [...new Map(points.map(p => [p.x, p.label])).entries()].sort(([a], [b]) => a - b);
  for (const i of [...new Set([0, Math.floor((positions.length - 1) / 2), positions.length - 1])]) {
    svg.append(make('text', { x: x(positions[i][0]), y: height - 14, 'text-anchor': i === 0 ? 'start' : i === positions.length - 1 ? 'end' : 'middle' }, positions[i][1]));
  }
  const cursor = make('line', { y1: top, y2: height - bottom, stroke: '#243e33', 'stroke-dasharray': '3 3', visibility: 'hidden' });
  svg.append(cursor); container.append(svg);
  let current = 0;
  const show = index => {
    current = Math.max(0, Math.min(positions.length - 1, index));
    const [position, label] = positions[current];
    cursor.setAttribute('x1', x(position)); cursor.setAttribute('x2', x(position)); cursor.setAttribute('visibility', 'visible');
    inspect(`${label} · ` + series.map(s => {
      const p = s.points.find(p => p.x === position);
      return `${s.name}: ${p?.detail || (finite(p?.y) ? `${shown(p.y)} ${unit}` : 'unavailable')}`;
    }).join(' · '));
  };
  container.onkeydown = event => {
    if (['ArrowLeft', 'ArrowRight', 'Home', 'End'].includes(event.key)) {
      event.preventDefault(); show(event.key === 'Home' ? 0 : event.key === 'End' ? positions.length - 1 : current + (event.key === 'ArrowRight' ? 1 : -1));
    }
  };
  svg.onpointermove = svg.onclick = event => {
    const bounds = svg.getBoundingClientRect();
    const target = minX + ((event.clientX - bounds.left) / bounds.width * width - left) / (width - left - right) * (maxX - minX || 1);
    let nearest = 0;
    positions.forEach(([position], i) => { if (Math.abs(position - target) < Math.abs(positions[nearest][0] - target)) nearest = i; });
    show(nearest);
  };
}

export function temperatureSeries(rows, selection, climatology) {
  const key = selection.metric, compare = selection.view === 'compare';
  const long = !compare && rows.length > 800;
  const frequency = rows.length > 4000 ? 'year' : 'month';
  let series;
  if (compare) {
    series = selection.compare.map((year, i) => ({ name: String(year), color: colors[i], dash: ['', '7 3', '2 3'][i],
      points: rows.filter(r => Number(r.date.slice(0, 4)) === year).map(r => ({ x: calendarDay(r.date), label: r.date.slice(5), y: toUnit(r[key], selection.unit), detail: `${r.date}: ${shown(toUnit(r[key], selection.unit))} °${selection.unit.toUpperCase()} (${r.status})` })) }));
  } else {
    series = Object.keys(METRICS).map((metric, i) => ({ name: long ? `${METRICS[metric]} (${frequency} mean)` : METRICS[metric], color: colors[i], light: metric !== key, dash: ['', '7 3', '2 3'][i],
      points: long ? aggregate(rows, metric, frequency).map((p, i) => ({ x: i, label: p.period, y: toUnit(p.value, selection.unit), detail: `${shown(toUnit(p.value, selection.unit))} °${selection.unit.toUpperCase()} · ${p.valid}/${p.expected} days` })) : rows.map((r, i) => ({ x: i, label: r.date, y: toUnit(r[metric], selection.unit) })) }));
  }
  if (selection.baseline && !long) {
    const field = { tmin_c: 'low_c', tmean_c: 'mean_c', tmax_c: 'high_c' }[key];
    const days = new Map(climatology.days.map(d => [d.month_day, d]));
    const referenceRows = compare ? Array.from({ length: 366 }, (_, i) => ({ date: new Date(Date.UTC(2000, 0, i + 1)).toISOString().slice(0, 10) })) : rows;
    for (const q of ['p10', 'p50', 'p90']) series.push({ name: `History ${q.toUpperCase()}`, color: '#56645d', dash: q === 'p50' ? '6 4' : '2 4', light: true,
      points: referenceRows.map((r, i) => ({ x: compare ? calendarDay(r.date) : i, label: compare ? r.date.slice(5) : r.date, y: toUnit(days.get(r.date.slice(5))?.[field]?.[q], selection.unit) })) });
  }
  return { series, long, frequency };
}
