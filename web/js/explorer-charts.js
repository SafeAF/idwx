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

export function lineChart(container, series, { title, unit = '', inspect, mark = 'line' }) {
  container.replaceChildren();
  const points = series.flatMap(s => s.points), valid = points.filter(p => finite(p.y));
  if (!valid.length) { container.textContent = 'No complete values to plot in this selection. See the table and coverage.'; return; }
  const width = Math.max(320, Math.min(1000, container.getBoundingClientRect().width || 1000));
  const height = width < 520 ? 300 : 360, left = 65, right = 28, top = 25, bottom = 48;
  const xs = points.map(p => p.x), ys = valid.map(p => p.y);
  const minX = Math.min(...xs), maxX = Math.max(...xs);
  const minY = Math.min(...ys, ...(['bar', 'area'].includes(mark) ? [0] : [])), maxY = Math.max(...ys, ...(['bar', 'area'].includes(mark) ? [0] : [])), padding = Math.max(1, (maxY - minY) * .12);
  const x = value => left + (value - minX) / (maxX - minX || 1) * (width - left - right);
  const y = value => top + (maxY + padding - value) / (maxY - minY + 2 * padding) * (height - top - bottom);
  const svg = make('svg', { viewBox: `0 0 ${width} ${height}`, role: 'img', 'aria-label': title });
  svg.append(make('title', {}, title), make('desc', {}, 'Use left/right arrow keys on the chart or select a position to read values. The table contains the selected data.'));
  for (let i = 0; i < 5; i++) {
    const value = minY + (maxY - minY) * i / 4;
    svg.append(make('line', { x1: left, x2: width - right, y1: y(value), y2: y(value), class: 'grid' }), make('text', { x: left - 10, y: y(value) + 4, 'text-anchor': 'end' }, shown(value)));
  }
  svg.append(make('text', { x: left, y: 15 }, unit));
  const plotted = series.filter(s => !s.reference);
  const uniqueX = [...new Set(points.map(p => p.x))].sort((a, b) => a - b);
  const spacing = uniqueX.length > 1 ? Math.min(...uniqueX.slice(1).map((value, i) => x(value) - x(uniqueX[i]))) : 40;
  const barWidth = Math.max(1, Math.min(35, spacing * .75 / Math.max(1, plotted.length)));
  if (['bar', 'area'].includes(mark)) svg.append(make('line', { x1: left, x2: width - right, y1: y(0), y2: y(0), stroke: '#65716a', 'stroke-width': 1 }));
  series.forEach((s, index) => {
    const style = s.reference ? 'line' : mark, color = s.color || colors[index % colors.length];
    let path = '', open = false, segment = [];
    const area = () => {
      if (segment.length) {
        const d = `M${x(segment[0].x)},${y(0)} ` + segment.map(p => `L${x(p.x)},${y(p.y)}`).join(' ') + ` L${x(segment.at(-1).x)},${y(0)} Z`;
        svg.append(make('path', { d, fill: color, opacity: .16, 'data-mark': 'area' }));
      }
      segment = [];
    };
    for (const point of s.points) {
      if (!finite(point.y)) { open = false; if (style === 'area') area(); continue; }
      const detail = `${s.name} · ${point.label}: ${point.detail || shown(point.y) + ' ' + unit}`;
      if (style === 'scatter' || style === 'bar') {
        const shape = plotted.indexOf(s) % 3;
        const scatter = shape === 0 ? make('circle', { cx: x(point.x), cy: y(point.y), r: 3.5, fill: color })
          : shape === 1 ? make('rect', { x: x(point.x) - 3, y: y(point.y) - 3, width: 6, height: 6, fill: color })
          : make('path', { d: `M${x(point.x)},${y(point.y) - 4} l4,4 l-4,4 l-4,-4 Z`, fill: color });
        const glyph = style === 'scatter'
          ? scatter
          : make('rect', { x: x(point.x) + (plotted.indexOf(s) - (plotted.length - 1) / 2) * barWidth - barWidth / 2, y: Math.min(y(0), y(point.y)), width: barWidth, height: Math.max(.7, Math.abs(y(point.y) - y(0))), fill: color, opacity: .85, 'data-mark': 'bar' });
        glyph.setAttribute('data-mark', style);
        glyph.setAttribute('opacity', s.light ? .45 : .9);
        glyph.append(make('title', {}, detail));
        svg.append(glyph);
      } else {
        path += `${open ? 'L' : 'M'}${x(point.x).toFixed(2)},${y(point.y).toFixed(2)} `;
        open = true; segment.push(point);
      }
    }
    if (style === 'area') area();
    if (style === 'line' || style === 'area') svg.append(make('path', { d: path, fill: 'none', stroke: color, 'stroke-width': s.light ? 1.3 : 2.5, 'stroke-dasharray': s.dash || '', opacity: s.light ? .55 : 1, 'data-mark': 'line' }));
    if ((style === 'line' || style === 'area') && s.points.filter(p => finite(p.y)).length === 1) {
      const p = s.points.find(p => finite(p.y)); svg.append(make('circle', { cx: x(p.x), cy: y(p.y), r: 4, fill: color }));
    }
  });
  const positions = [...new Map(points.map(p => [p.x, p.label])).entries()].sort(([a], [b]) => a - b);
  for (const i of [...new Set([0, Math.floor((positions.length - 1) / 2), positions.length - 1])]) {
    svg.append(make('text', { x: x(positions[i][0]), y: height - 14, 'text-anchor': i === 0 ? 'start' : i === positions.length - 1 ? 'end' : 'middle' }, positions[i][1]));
  }
  const cursor = make('line', { y1: top, y2: height - bottom, stroke: '#243e33', 'stroke-dasharray': '3 3', visibility: 'hidden' });
  svg.append(cursor); container.append(svg);
  let current = 0, linked = false;
  const show = index => {
    current = Math.max(0, Math.min(positions.length - 1, index));
    const [position, label] = positions[current];
    cursor.setAttribute('x1', x(position)); cursor.setAttribute('x2', x(position)); cursor.setAttribute('visibility', 'visible');
    if (!linked) container.dispatchEvent(new CustomEvent('idwx-inspect', { bubbles: true, detail: { label } }));
    inspect(`${label} · ` + series.map(s => {
      const p = s.points.find(p => p.x === position);
      return `${s.name}: ${p?.detail || (finite(p?.y) ? `${shown(p.y)} ${unit}` : 'unavailable')}`;
    }).join(' · '));
  };
  container.onidwxinspect = event => {
    const index = positions.findIndex(([, label]) => label === event.detail.label);
    if (index >= 0) { linked = true; show(index); linked = false; }
  };
  container.removeEventListener('idwx-inspect-linked', container._linkedInspection);
  container._linkedInspection = container.onidwxinspect;
  container.addEventListener('idwx-inspect-linked', container._linkedInspection);
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
    for (const q of ['p10', 'p50', 'p90']) series.push({ name: `History ${q.toUpperCase()}`, reference: true, color: '#56645d', dash: q === 'p50' ? '6 4' : '2 4', light: true,
      points: referenceRows.map((r, i) => ({ x: compare ? calendarDay(r.date) : i, label: compare ? r.date.slice(5) : r.date, y: toUnit(days.get(r.date.slice(5))?.[field]?.[q], selection.unit) })) });
  }
  return { series, long, frequency };
}
