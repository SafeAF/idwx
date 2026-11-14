const LOCATION = { latitude: 42.4634, longitude: -114.4975, name: 'Twin Falls' };
const NWS_POINT = `https://api.weather.gov/points/${LOCATION.latitude},${LOCATION.longitude}`;
const NWS_ALERTS = `https://api.weather.gov/alerts/active?point=${LOCATION.latitude},${LOCATION.longitude}`;
const DAILY_CORE = 'weather_code,temperature_2m_max,temperature_2m_min,precipitation_sum,precipitation_probability_max,snowfall_sum,wind_speed_10m_max,wind_gusts_10m_max,wind_direction_10m_dominant';
const DAILY_DETAIL = `${DAILY_CORE},apparent_temperature_max,apparent_temperature_min,relative_humidity_2m_mean,dew_point_2m_mean,visibility_mean,uv_index_max,sunrise,sunset`;

function modelUrl(model, daily, current = null) {
  const url = new URL('https://api.open-meteo.com/v1/forecast');
  url.search = new URLSearchParams({
    latitude: LOCATION.latitude, longitude: LOCATION.longitude, timezone: 'America/Boise',
    temperature_unit: 'fahrenheit', wind_speed_unit: 'mph', precipitation_unit: 'inch',
    forecast_days: '14', models: model, daily, ...(current ? { current } : {}),
  });
  return url;
}

const GFS = modelUrl('gfs_seamless', DAILY_DETAIL,
  'temperature_2m,apparent_temperature,relative_humidity_2m,weather_code,wind_speed_10m,wind_direction_10m,wind_gusts_10m');
const ECMWF = modelUrl('ecmwf_ifs025', DAILY_CORE);
const CACHE_KEY = 'idwx-weather-analysis-v2';
const FRESH_MS = 15 * 60 * 1000;
const STALE_MS = 6 * 60 * 60 * 1000;

const WEATHER = {
  0: ['Clear', '☀'], 1: ['Mostly clear', '🌤'], 2: ['Partly cloudy', '⛅'], 3: ['Cloudy', '☁'],
  45: ['Fog', '≋'], 48: ['Freezing fog', '≋'], 51: ['Light drizzle', '🌦'], 53: ['Drizzle', '🌦'],
  55: ['Heavy drizzle', '🌧'], 56: ['Freezing drizzle', '🌧'], 57: ['Freezing drizzle', '🌧'],
  61: ['Light rain', '🌦'], 63: ['Rain', '🌧'], 65: ['Heavy rain', '🌧'], 66: ['Freezing rain', '🌨'],
  67: ['Freezing rain', '🌨'], 71: ['Light snow', '🌨'], 73: ['Snow', '❄'], 75: ['Heavy snow', '❄'],
  77: ['Snow grains', '❄'], 80: ['Rain showers', '🌦'], 81: ['Rain showers', '🌧'], 82: ['Heavy showers', '🌧'],
  85: ['Snow showers', '🌨'], 86: ['Heavy snow showers', '❄'], 95: ['Thunderstorms', '⛈'],
  96: ['Thunderstorms with hail', '⛈'], 99: ['Thunderstorms with hail', '⛈'],
};

const round = value => Number.isFinite(value) ? Math.round(value) : null;
const mean = values => values.length ? values.reduce((sum, value) => sum + value, 0) / values.length : null;
const spread = values => values.length > 1 ? Math.max(...values) - Math.min(...values) : null;
const median = values => {
  if (!values.length) return null;
  const sorted = [...values].sort((a, b) => a - b), middle = Math.floor(sorted.length / 2);
  return sorted.length % 2 ? sorted[middle] : (sorted[middle - 1] + sorted[middle]) / 2;
};
const valueAt = (daily, key, index) => Array.isArray(daily?.[key]) && Number.isFinite(daily[key][index]) ? daily[key][index] : null;
const text = value => value == null ? '—' : String(value);
const temperature = value => Number.isFinite(value) ? `${Math.round(value)}°` : '—';
const signedTemperature = value => Number.isFinite(value) ? `${value >= 0 ? '+' : ''}${Math.round(value)}°` : '—';
const amount = value => Number.isFinite(value) ? (value < 0.01 ? '<0.01 in' : `${value.toFixed(2)} in`) : '—';
const compass = degrees => Number.isFinite(degrees) ? ['N','NE','E','SE','S','SW','W','NW'][Math.round(degrees / 45) % 8] : '—';
const weather = code => WEATHER[code] || ['Variable', '◌'];
const cToF = value => Number.isFinite(value) ? value * 9 / 5 + 32 : null;
const dateLabel = (date, options) => new Intl.DateTimeFormat('en-US', { timeZone: 'America/Boise', ...options }).format(new Date(`${date}T12:00:00-07:00`));
const timeLabel = value => value ? new Intl.DateTimeFormat('en-US', { hour: 'numeric', minute: '2-digit', timeZone: 'America/Boise' }).format(new Date(`${value}:00-07:00`)) : '—';
const mph = value => {
  const numbers = String(value || '').match(/\d+(?:\.\d+)?/g)?.map(Number) || [];
  return numbers.length ? Math.max(...numbers) : null;
};

export function normalizeModel(payload, label, { requireCurrent = false } = {}) {
  if (!payload || !Array.isArray(payload.daily?.time) || payload.daily.time.length < 14 || (requireCurrent && !payload.current)) {
    throw new Error(`${label} returned an incomplete forecast`);
  }
  const daily = payload.daily;
  return {
    label,
    current: payload.current ? {
      time: payload.current.time, temperature: payload.current.temperature_2m,
      apparent: payload.current.apparent_temperature, humidity: payload.current.relative_humidity_2m,
      code: payload.current.weather_code, wind: payload.current.wind_speed_10m,
      windDirection: payload.current.wind_direction_10m, gust: payload.current.wind_gusts_10m,
    } : null,
    days: daily.time.slice(0, 14).map((date, index) => ({
      date, code: valueAt(daily, 'weather_code', index), high: valueAt(daily, 'temperature_2m_max', index),
      low: valueAt(daily, 'temperature_2m_min', index), feelsHigh: valueAt(daily, 'apparent_temperature_max', index),
      feelsLow: valueAt(daily, 'apparent_temperature_min', index), precipitation: valueAt(daily, 'precipitation_sum', index),
      precipitationChance: valueAt(daily, 'precipitation_probability_max', index), snow: valueAt(daily, 'snowfall_sum', index),
      wind: valueAt(daily, 'wind_speed_10m_max', index), gust: valueAt(daily, 'wind_gusts_10m_max', index),
      windDirection: valueAt(daily, 'wind_direction_10m_dominant', index), humidity: valueAt(daily, 'relative_humidity_2m_mean', index),
      dewPoint: valueAt(daily, 'dew_point_2m_mean', index), visibilityFeet: valueAt(daily, 'visibility_mean', index),
      uv: valueAt(daily, 'uv_index_max', index), sunrise: daily.sunrise?.[index] || null, sunset: daily.sunset?.[index] || null,
    })),
  };
}

export const normalizeOpenMeteo = payload => normalizeModel(payload, 'NOAA GFS/RRFS', { requireCurrent: true });

export function normalizeNws(payload) {
  if (!Array.isArray(payload?.properties?.periods)) throw new Error('NWS returned an incomplete forecast');
  const byDate = new Map();
  for (const period of payload.properties.periods) {
    const date = period.startTime?.slice(0, 10);
    if (!date) continue;
    const day = byDate.get(date) || { date, day: null, night: null };
    day[period.isDaytime ? 'day' : 'night'] = {
      temperature: period.temperature, precipitationChance: period.probabilityOfPrecipitation?.value,
      wind: period.windSpeed, windDirection: period.windDirection, summary: period.shortForecast,
      detail: period.detailedForecast,
    };
    byDate.set(date, day);
  }
  return { generatedAt: payload.properties.generatedAt, updatedAt: payload.properties.updateTime, byDate };
}

function nwsSource(day) {
  if (!day) return null;
  const precipitation = [day.day?.precipitationChance, day.night?.precipitationChance].filter(Number.isFinite);
  return {
    label: 'NWS', high: day.day?.temperature ?? null, low: day.night?.temperature ?? null,
    precipitationChance: precipitation.length ? Math.max(...precipitation) : null,
    wind: mph(day.day?.wind || day.night?.wind), windText: `${day.day?.windDirection || day.night?.windDirection || '—'} ${day.day?.wind || day.night?.wind || '—'}`,
    summary: day.day?.summary || day.night?.summary || 'Forecast available', detail: `${day.day?.detail || ''} ${day.night?.detail || ''}`.trim(),
  };
}

function historicalSource(climatology, date) {
  const row = climatology?.days?.find(day => day.month_day === date.slice(5));
  if (!row) return null;
  const convert = values => Object.fromEntries(Object.entries(values).map(([key, value]) => [key, cToF(value)]));
  return { label: 'IDWX history', sampleCount: row.sample_count, high: convert(row.high_c), low: convert(row.low_c), mean: convert(row.mean_c), baseline: climatology.baseline_period };
}

export function mergeForecast(gfsPayload, ecmwfPayload, nwsPayload = null, climatology = null) {
  const gfs = gfsPayload ? normalizeModel(gfsPayload, 'NOAA GFS/RRFS', { requireCurrent: true }) : null;
  const ecmwf = ecmwfPayload ? normalizeModel(ecmwfPayload, 'ECMWF IFS') : null;
  if (!gfs && !ecmwf) throw new Error('Both 14-day model sources are unavailable');
  const nws = nwsPayload ? normalizeNws(nwsPayload) : null;
  const base = gfs || ecmwf;
  const byDate = model => new Map((model?.days || []).map(day => [day.date, day]));
  const gfsDates = byDate(gfs), ecmwfDates = byDate(ecmwf);
  const days = base.days.map((day, index) => {
    const sources = [nwsSource(nws?.byDate.get(day.date)), gfsDates.get(day.date) && { label: 'NOAA GFS/RRFS', ...gfsDates.get(day.date) },
      ecmwfDates.get(day.date) && { label: 'ECMWF IFS', ...ecmwfDates.get(day.date) }].filter(Boolean);
    const highs = sources.map(source => source.high).filter(Number.isFinite), lows = sources.map(source => source.low).filter(Number.isFinite);
    const history = historicalSource(climatology, day.date);
    const highConsensus = mean(highs), lowConsensus = mean(lows), maxSpread = Math.max(spread(highs) || 0, spread(lows) || 0);
    return {
      ...day, sources, history, extended: index >= 7, consensus: {
        high: highConsensus, low: lowConsensus, highSpread: spread(highs), lowSpread: spread(lows), sourceCount: Math.max(highs.length, lows.length),
        agreement: maxSpread <= 3 ? 'Tight' : maxSpread <= 7 ? 'Moderate' : 'Wide',
        highAnomaly: history && Number.isFinite(highConsensus) ? highConsensus - history.high.p50 : null,
        lowAnomaly: history && Number.isFinite(lowConsensus) ? lowConsensus - history.low.p50 : null,
      },
    };
  });
  return { current: gfs?.current, days, nws, climatology, modelCount: [gfs, ecmwf].filter(Boolean).length };
}

export function forecastAnalytics(days) {
  const spreads = days.map(day => Math.max(day.consensus.highSpread || 0, day.consensus.lowSpread || 0));
  const widest = spreads.length ? spreads.indexOf(Math.max(...spreads)) : -1;
  const anomalies = days.map(day => day.consensus.highAnomaly).filter(Number.isFinite);
  return {
    threeSourceDays: days.filter(day => day.sources.length === 3).length,
    medianSpread: median(spreads), widestDay: widest >= 0 ? days[widest].date : null, widestSpread: widest >= 0 ? spreads[widest] : null,
    tightDays: days.filter(day => day.consensus.agreement === 'Tight').length,
    averageHighAnomaly: mean(anomalies), aboveHistoryDays: anomalies.filter(value => value > 0).length,
  };
}

async function responseJson(url) {
  const response = await fetch(url, {
    headers: { Accept: 'application/geo+json, application/json' },
    signal: AbortSignal.timeout(20_000),
  });
  if (!response.ok) throw new Error(`${new URL(url).hostname} returned HTTP ${response.status}`);
  return response.json();
}

async function fetchClimatology() {
  const manifest = await responseJson(new URL('data/manifest.json', document.baseURI));
  const stations = await responseJson(new URL(manifest.stations_url, document.baseURI));
  const entry = stations.stations.find(item => item.station.station_id === 'twin_falls') || stations.stations[0];
  const station = await responseJson(new URL(entry.url, document.baseURI));
  return station.temperature_climatology;
}

async function fetchLive() {
  const [gfsResult, ecmwfResult, pointResult, alertResult, climateResult] = await Promise.allSettled([
    responseJson(GFS), responseJson(ECMWF), responseJson(NWS_POINT), responseJson(NWS_ALERTS), fetchClimatology(),
  ]);
  if (gfsResult.status === 'rejected' && ecmwfResult.status === 'rejected') throw new Error('NOAA GFS/RRFS and ECMWF IFS are both unavailable');
  let nws = null, nwsError = null;
  if (pointResult.status === 'fulfilled') {
    try { nws = await responseJson(pointResult.value?.properties?.forecast); } catch (error) { nwsError = error.message; }
  } else nwsError = pointResult.reason.message;
  return {
    fetchedAt: Date.now(), gfs: gfsResult.status === 'fulfilled' ? gfsResult.value : null,
    ecmwf: ecmwfResult.status === 'fulfilled' ? ecmwfResult.value : null, nws,
    alerts: alertResult.status === 'fulfilled' ? alertResult.value.features || [] : [],
    climatology: climateResult.status === 'fulfilled' ? climateResult.value : null,
    errors: [gfsResult, ecmwfResult, climateResult].filter(result => result.status === 'rejected').map(result => result.reason.message), nwsError,
  };
}

function cached(maxAge) {
  try { const value = JSON.parse(localStorage.getItem(CACHE_KEY)); return value && Date.now() - value.fetchedAt <= maxAge ? value : null; }
  catch { return null; }
}

function element(tag, className, content) {
  const node = document.createElement(tag); if (className) node.className = className; if (content != null) node.textContent = content; return node;
}

function metric(label, value) {
  const node = element('div', 'weather-metric'); node.append(element('span', 'metric-label', label), element('strong', '', value)); return node;
}

function renderAlerts(features) {
  const target = document.getElementById('weather-alerts'); target.replaceChildren();
  for (const feature of features.slice(0, 3)) {
    const properties = feature.properties || {}, alert = element('a', 'weather-alert');
    alert.href = properties['@id'] || feature.id || 'https://www.weather.gov/boi/'; alert.target = '_blank'; alert.rel = 'noopener noreferrer';
    alert.append(element('strong', '', properties.event || 'NWS weather alert'), element('span', '', properties.headline || 'Open the NWS alert for details.')); target.append(alert);
  }
}

function renderCurrent(current, fetchedAt, sourceCount) {
  const target = document.getElementById('current-weather');
  if (!current) { target.replaceChildren(element('p', '', 'Current conditions are temporarily unavailable; daily model comparisons remain below.')); return; }
  const [description, symbol] = weather(current.code), primary = element('div', 'current-primary');
  primary.append(element('span', 'weather-symbol', symbol));
  const copy = element('div'); copy.append(element('p', 'current-temp', temperature(current.temperature)), element('p', 'current-summary', description)); primary.append(copy);
  const details = element('div', 'current-details');
  details.append(metric('Feels like', temperature(current.apparent)), metric('Humidity', `${text(round(current.humidity))}%`),
    metric('Wind', `${compass(current.windDirection)} ${text(round(current.wind))} mph`), metric('Gusts', `${text(round(current.gust))} mph`));
  const meta = element('div', 'current-meta');
  meta.append(element('p', 'eyebrow', 'Conditions now · GFS/RRFS'), element('p', '', `Model estimate at ${timeLabel(current.time)} · Updated ${new Intl.DateTimeFormat('en-US', { hour: 'numeric', minute: '2-digit', timeZone: 'America/Boise' }).format(new Date(fetchedAt))}`),
    element('p', 'small', `${sourceCount} live forecast sources are compared below.`)); target.replaceChildren(meta, primary, details);
}

function renderAnalytics(days) {
  const target = document.getElementById('forecast-analytics'), analysis = forecastAnalytics(days); target.replaceChildren();
  const items = [
    ['3-source coverage', `${analysis.threeSourceDays} of 14 days`, 'NWS coverage ends near day 7; two global models continue.'],
    ['Typical model spread', `${temperature(analysis.medianSpread)}`, `${analysis.tightDays} days have a spread of 3°F or less.`],
    ['Widest disagreement', `${temperature(analysis.widestSpread)}`, analysis.widestDay ? dateLabel(analysis.widestDay, { weekday: 'short', month: 'short', day: 'numeric' }) : '—'],
    ['Vs. IDWX history', signedTemperature(analysis.averageHighAnomaly), `${analysis.aboveHistoryDays} days have consensus highs above the historical median.`],
  ];
  for (const [label, value, detail] of items) { const card = element('div', 'analysis-card'); card.append(element('p', 'analysis-label', label), element('p', 'analysis-value', value), element('p', 'analysis-detail', detail)); target.append(card); }
}

function sourceRow(source) {
  const row = element('div', 'source-row');
  row.append(element('strong', 'source-label', source.label), element('span', '', `${temperature(source.high)} / ${temperature(source.low)}`),
    element('span', '', Number.isFinite(source.precipitationChance) ? `${round(source.precipitationChance)}% precip.` : '—'),
    element('span', '', source.windText || (Number.isFinite(source.wind) ? `${compass(source.windDirection)} ${round(source.wind)} mph` : '—')));
  return row;
}

function historicalRow(history) {
  const row = element('div', 'source-row history-row');
  row.append(element('strong', 'source-label', 'IDWX history'), element('span', '', `${temperature(history.high.p50)} / ${temperature(history.low.p50)}`),
    element('span', '', `P10–P90 high ${temperature(history.high.p10)}–${temperature(history.high.p90)}`), element('span', '', `${history.sampleCount} years`)); return row;
}

function renderDays(days) {
  const target = document.getElementById('forecast-days'); target.replaceChildren();
  days.forEach((day, index) => {
    const primary = day.sources.find(source => source.label === 'NOAA GFS/RRFS') || day.sources[0];
    const [description, symbol] = weather(primary?.code ?? day.code), card = element('article', `forecast-day${day.extended ? ' extended' : ''}`);
    const heading = element('div', 'day-heading'), title = element('div');
    title.append(element('p', 'day-name', index === 0 ? 'Today' : dateLabel(day.date, { weekday: 'short' })), element('p', 'day-date', dateLabel(day.date, { month: 'short', day: 'numeric' })));
    heading.append(title, element('span', 'agreement-badge', `${day.consensus.agreement} agreement`), element('span', 'day-symbol', symbol));
    const temps = element('p', 'day-temps'); temps.append(element('strong', '', temperature(day.consensus.high)), document.createTextNode(` / ${temperature(day.consensus.low)}`));
    card.append(heading, element('p', 'day-summary', `${description} · ${day.consensus.sourceCount}-source temperature consensus`), temps);
    const metrics = element('div', 'day-metrics');
    metrics.append(metric('High spread', temperature(day.consensus.highSpread)), metric('Low spread', temperature(day.consensus.lowSpread)),
      metric('Vs history high', signedTemperature(day.consensus.highAnomaly)), metric('Vs history low', signedTemperature(day.consensus.lowAnomaly)),
      metric('GFS precip.', `${text(round(primary?.precipitationChance))}% · ${amount(primary?.precipitation)}`), metric('GFS gusts', Number.isFinite(primary?.gust) ? `${round(primary.gust)} mph` : '—'));
    card.append(metrics);
    const comparison = element('div', 'source-comparison'); comparison.append(element('p', 'source-name', 'Source matrix · high / low'));
    for (const source of day.sources) comparison.append(sourceRow(source)); if (day.history) comparison.append(historicalRow(day.history)); card.append(comparison);
    const detail = element('details', 'day-details'); detail.append(element('summary', '', 'Method and details'));
    const detailCopy = element('p', '', 'Consensus is the arithmetic mean of available source temperatures. Spread is warmest minus coolest source. IDWX is calendar-day climatology, not a live model.');
    const nws = day.sources.find(source => source.label === 'NWS'); if (nws?.detail) detailCopy.textContent += ` NWS: ${nws.detail}`;
    detail.append(detailCopy); card.append(detail); target.append(card);
  });
}

function render(data, stale = false) {
  const merged = mergeForecast(data.gfs, data.ecmwf, data.nws, data.climatology);
  renderAlerts(data.alerts || []); renderCurrent(merged.current, data.fetchedAt, Math.max(...merged.days.map(day => day.sources.length))); renderAnalytics(merged.days); renderDays(merged.days);
  document.getElementById('forecast-content').hidden = false;
  const notes = []; if (stale) notes.push('Live refresh failed; showing recently cached guidance.');
  if (data.nwsError) notes.push(`NWS unavailable: ${data.nwsError}.`); if (data.errors?.length) notes.push(`Partial data: ${data.errors.join(' ')}`);
  const status = document.getElementById('forecast-status'); status.textContent = notes.join(' '); status.classList.toggle('error', Boolean(notes.length));
}

export async function loadForecast({ force = false } = {}) {
  const button = document.getElementById('forecast-refresh'), status = document.getElementById('forecast-status'); if (!button || !status) return;
  button.disabled = true; status.classList.remove('error'); status.textContent = 'Loading three forecast sources and IDWX history…';
  if (!force) { const fresh = cached(FRESH_MS); if (fresh) { render(fresh); button.disabled = false; return; } }
  try { const data = await fetchLive(); try { localStorage.setItem(CACHE_KEY, JSON.stringify(data)); } catch { /* optional */ } render(data); }
  catch (error) { const stale = cached(STALE_MS); if (stale) render(stale, true); else { status.classList.add('error'); status.textContent = `Forecast unavailable right now: ${error.message}. The historical archive remains available below.`; } }
  finally { button.disabled = false; }
}

if (typeof document !== 'undefined') {
  document.getElementById('forecast-refresh')?.addEventListener('click', () => loadForecast({ force: true })); loadForecast();
}
