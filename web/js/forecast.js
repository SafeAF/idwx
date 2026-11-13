const LOCATION = { latitude: 42.4634, longitude: -114.4975, name: 'Twin Falls' };
const NWS_POINT = `https://api.weather.gov/points/${LOCATION.latitude},${LOCATION.longitude}`;
const NWS_ALERTS = `https://api.weather.gov/alerts/active?point=${LOCATION.latitude},${LOCATION.longitude}`;
const OPEN_METEO = new URL('https://api.open-meteo.com/v1/forecast');
OPEN_METEO.search = new URLSearchParams({
  latitude: LOCATION.latitude, longitude: LOCATION.longitude, timezone: 'America/Boise',
  temperature_unit: 'fahrenheit', wind_speed_unit: 'mph', precipitation_unit: 'inch', forecast_days: '14',
  current: 'temperature_2m,apparent_temperature,relative_humidity_2m,precipitation,weather_code,wind_speed_10m,wind_direction_10m,wind_gusts_10m',
  daily: 'weather_code,temperature_2m_max,temperature_2m_min,apparent_temperature_max,apparent_temperature_min,precipitation_sum,precipitation_probability_max,snowfall_sum,wind_speed_10m_max,wind_gusts_10m_max,wind_direction_10m_dominant,relative_humidity_2m_mean,dew_point_2m_mean,visibility_mean,uv_index_max,sunrise,sunset',
});
const CACHE_KEY = 'idwx-live-weather-v1';
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
const valueAt = (daily, key, index) => Array.isArray(daily?.[key]) ? daily[key][index] : null;
const escapeText = value => value == null ? '—' : String(value);
const temperature = value => Number.isFinite(value) ? `${Math.round(value)}°` : '—';
const amount = value => Number.isFinite(value) ? (value < 0.01 ? '<0.01 in' : `${value.toFixed(2)} in`) : '—';
const compass = degrees => Number.isFinite(degrees) ? ['N','NE','E','SE','S','SW','W','NW'][Math.round(degrees / 45) % 8] : '—';
const weather = code => WEATHER[code] || ['Variable', '◌'];
const dateLabel = (date, options) => new Intl.DateTimeFormat('en-US', { timeZone: 'America/Boise', ...options }).format(new Date(`${date}T12:00:00-07:00`));
const timeLabel = value => new Intl.DateTimeFormat('en-US', { hour: 'numeric', minute: '2-digit', timeZone: 'America/Boise' }).format(new Date(`${value}:00-07:00`));

export function normalizeOpenMeteo(payload) {
  if (!payload || !payload.current || !Array.isArray(payload.daily?.time) || payload.daily.time.length < 14) throw new Error('Open-Meteo returned an incomplete forecast');
  const daily = payload.daily;
  return {
    current: {
      time: payload.current.time, temperature: payload.current.temperature_2m,
      apparent: payload.current.apparent_temperature, humidity: payload.current.relative_humidity_2m,
      precipitation: payload.current.precipitation, code: payload.current.weather_code,
      wind: payload.current.wind_speed_10m, windDirection: payload.current.wind_direction_10m,
      gust: payload.current.wind_gusts_10m,
    },
    days: daily.time.slice(0, 14).map((date, index) => ({
      date, code: valueAt(daily, 'weather_code', index), high: valueAt(daily, 'temperature_2m_max', index),
      low: valueAt(daily, 'temperature_2m_min', index), feelsHigh: valueAt(daily, 'apparent_temperature_max', index),
      feelsLow: valueAt(daily, 'apparent_temperature_min', index), precipitation: valueAt(daily, 'precipitation_sum', index),
      precipitationChance: valueAt(daily, 'precipitation_probability_max', index), snow: valueAt(daily, 'snowfall_sum', index),
      wind: valueAt(daily, 'wind_speed_10m_max', index), gust: valueAt(daily, 'wind_gusts_10m_max', index),
      windDirection: valueAt(daily, 'wind_direction_10m_dominant', index), humidity: valueAt(daily, 'relative_humidity_2m_mean', index),
      dewPoint: valueAt(daily, 'dew_point_2m_mean', index), visibilityFeet: valueAt(daily, 'visibility_mean', index),
      uv: valueAt(daily, 'uv_index_max', index), sunrise: valueAt(daily, 'sunrise', index), sunset: valueAt(daily, 'sunset', index),
    })),
  };
}

export function normalizeNws(payload) {
  if (!Array.isArray(payload?.properties?.periods)) throw new Error('NWS returned an incomplete forecast');
  const byDate = new Map();
  for (const period of payload.properties.periods) {
    const date = period.startTime?.slice(0, 10);
    if (!date) continue;
    const day = byDate.get(date) || { date, day: null, night: null };
    day[period.isDaytime ? 'day' : 'night'] = {
      name: period.name, temperature: period.temperature, precipitationChance: period.probabilityOfPrecipitation?.value,
      wind: period.windSpeed, windDirection: period.windDirection, summary: period.shortForecast,
      detail: period.detailedForecast,
    };
    byDate.set(date, day);
  }
  return { generatedAt: payload.properties.generatedAt, updatedAt: payload.properties.updateTime, byDate };
}

export function mergeForecast(openMeteo, nws) {
  const model = normalizeOpenMeteo(openMeteo);
  const national = nws ? normalizeNws(nws) : null;
  return { ...model, days: model.days.map((day, index) => ({ ...day, nws: national?.byDate.get(day.date) || null, extended: index >= 7 })), nws: national };
}

async function responseJson(url) {
  const response = await fetch(url, { headers: { Accept: 'application/geo+json, application/json' } });
  if (!response.ok) throw new Error(`${new URL(url).hostname} returned HTTP ${response.status}`);
  return response.json();
}

async function fetchLive() {
  const [modelResult, pointResult, alertResult] = await Promise.allSettled([
    responseJson(OPEN_METEO), responseJson(NWS_POINT), responseJson(NWS_ALERTS),
  ]);
  if (modelResult.status === 'rejected') throw modelResult.reason;
  let nws = null, nwsError = null;
  if (pointResult.status === 'fulfilled') {
    try { nws = await responseJson(pointResult.value?.properties?.forecast); } catch (error) { nwsError = error.message; }
  } else nwsError = pointResult.reason.message;
  const alerts = alertResult.status === 'fulfilled' ? alertResult.value.features || [] : [];
  return { fetchedAt: Date.now(), model: modelResult.value, nws, alerts, nwsError };
}

function cached(maxAge) {
  try {
    const value = JSON.parse(localStorage.getItem(CACHE_KEY));
    return value && Date.now() - value.fetchedAt <= maxAge ? value : null;
  } catch { return null; }
}

function element(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text != null) node.textContent = text;
  return node;
}

function metric(label, value) {
  const node = element('div', 'weather-metric');
  node.append(element('span', 'metric-label', label), element('strong', '', value));
  return node;
}

function renderAlerts(features) {
  const target = document.getElementById('weather-alerts');
  target.replaceChildren();
  for (const feature of features.slice(0, 3)) {
    const properties = feature.properties || {};
    const alert = element('a', 'weather-alert');
    alert.href = properties['@id'] || feature.id || 'https://www.weather.gov/boi/';
    alert.target = '_blank'; alert.rel = 'noopener noreferrer';
    alert.append(element('strong', '', properties.event || 'NWS weather alert'), element('span', '', properties.headline || 'Open the NWS alert for details.'));
    target.append(alert);
  }
}

function renderCurrent(current, fetchedAt, hasNws) {
  const target = document.getElementById('current-weather');
  const [description, symbol] = weather(current.code);
  const primary = element('div', 'current-primary');
  primary.append(element('span', 'weather-symbol', symbol));
  const copy = element('div');
  copy.append(element('p', 'current-temp', temperature(current.temperature)), element('p', 'current-summary', description));
  primary.append(copy);
  const details = element('div', 'current-details');
  details.append(metric('Feels like', temperature(current.apparent)), metric('Humidity', `${escapeText(round(current.humidity))}%`),
    metric('Wind', `${compass(current.windDirection)} ${escapeText(round(current.wind))} mph`), metric('Gusts', `${escapeText(round(current.gust))} mph`));
  const meta = element('div', 'current-meta');
  meta.append(element('p', 'eyebrow', 'Conditions now'), element('p', '', `Model estimate at ${timeLabel(current.time)} · Updated ${new Intl.DateTimeFormat('en-US', { hour: 'numeric', minute: '2-digit', timeZone: 'America/Boise' }).format(new Date(fetchedAt))}`),
    element('p', 'small', hasNws ? 'Seven-day NWS guidance is available below.' : 'NWS guidance is temporarily unavailable; model data remains visible.'));
  target.replaceChildren(meta, primary, details);
}

function sourceComparison(day) {
  const box = element('div', 'source-comparison');
  if (!day.nws) {
    box.append(element('p', 'source-name', day.extended ? 'Extended model outlook' : 'Open-Meteo only'),
      element('p', '', day.extended ? 'Beyond the NWS point-forecast window' : 'NWS guidance is not available for this date'));
    return box;
  }
  const nwsHigh = day.nws.day?.temperature, nwsLow = day.nws.night?.temperature;
  box.append(element('p', 'source-name', 'NWS comparison'),
    element('p', '', `${day.nws.day?.summary || day.nws.night?.summary || 'Forecast available'} · ${temperature(nwsHigh)} / ${temperature(nwsLow)}`),
    element('p', 'source-wind', `${day.nws.day?.windDirection || day.nws.night?.windDirection || '—'} wind ${day.nws.day?.wind || day.nws.night?.wind || '—'}`));
  return box;
}

function renderDays(days) {
  const target = document.getElementById('forecast-days');
  target.replaceChildren();
  days.forEach((day, index) => {
    const [description, symbol] = weather(day.code);
    const card = element('article', `forecast-day${day.extended ? ' extended' : ''}`);
    const heading = element('div', 'day-heading');
    const title = element('div');
    title.append(element('p', 'day-name', index === 0 ? 'Today' : dateLabel(day.date, { weekday: 'short' })), element('p', 'day-date', dateLabel(day.date, { month: 'short', day: 'numeric' })));
    heading.append(title, element('span', 'day-symbol', symbol));
    const temps = element('p', 'day-temps');
    temps.append(element('strong', '', temperature(day.high)), document.createTextNode(` / ${temperature(day.low)}`));
    card.append(heading, element('p', 'day-summary', description), temps);
    const metrics = element('div', 'day-metrics');
    metrics.append(metric('Precip.', `${escapeText(round(day.precipitationChance))}% · ${amount(day.precipitation)}`),
      metric('Wind', `${compass(day.windDirection)} ${escapeText(round(day.wind))} mph`),
      metric('Gusts', `${escapeText(round(day.gust))} mph`), metric('Humidity', `${escapeText(round(day.humidity))}%`),
      metric('Feels', `${temperature(day.feelsHigh)} / ${temperature(day.feelsLow)}`), metric('UV max', escapeText(Number.isFinite(day.uv) ? day.uv.toFixed(1) : null)));
    if (day.snow > 0) metrics.append(metric('Snow', amount(day.snow)));
    card.append(metrics, sourceComparison(day));
    const detail = element('details', 'day-details');
    detail.append(element('summary', '', 'More details'));
    const detailCopy = element('p', '');
    detailCopy.textContent = `Dew point ${temperature(day.dewPoint)} · Visibility ${Number.isFinite(day.visibilityFeet) ? Math.round(day.visibilityFeet / 5280) + ' mi' : '—'} · Sunrise ${timeLabel(day.sunrise)} · Sunset ${timeLabel(day.sunset)}.`;
    if (day.nws?.day?.detail || day.nws?.night?.detail) detailCopy.textContent += ` NWS: ${day.nws.day?.detail || ''} ${day.nws.night?.detail || ''}`;
    detail.append(detailCopy); card.append(detail); target.append(card);
  });
}

function render(data, stale = false) {
  const merged = mergeForecast(data.model, data.nws);
  renderAlerts(data.alerts || []); renderCurrent(merged.current, data.fetchedAt, Boolean(data.nws)); renderDays(merged.days);
  document.getElementById('forecast-content').hidden = false;
  const notes = [];
  if (stale) notes.push('Live refresh failed; showing recently cached guidance.');
  if (data.nwsError) notes.push(`NWS comparison unavailable: ${data.nwsError}.`);
  document.getElementById('forecast-status').textContent = notes.join(' ');
  document.getElementById('forecast-status').classList.toggle('error', stale || Boolean(data.nwsError));
}

export async function loadForecast({ force = false } = {}) {
  const button = document.getElementById('forecast-refresh');
  const status = document.getElementById('forecast-status');
  if (!button || !status) return;
  button.disabled = true;
  status.classList.remove('error'); status.textContent = 'Loading current weather guidance…';
  if (!force) {
    const fresh = cached(FRESH_MS);
    if (fresh) { render(fresh); button.disabled = false; return; }
  }
  try {
    const data = await fetchLive();
    try { localStorage.setItem(CACHE_KEY, JSON.stringify(data)); } catch { /* cache is optional */ }
    render(data);
  } catch (error) {
    const stale = cached(STALE_MS);
    if (stale) render(stale, true);
    else { status.classList.add('error'); status.textContent = `Forecast unavailable right now: ${error.message}. The historical archive remains available below.`; }
  } finally { button.disabled = false; }
}

if (typeof document !== 'undefined') {
  document.getElementById('forecast-refresh')?.addEventListener('click', () => loadForecast({ force: true }));
  loadForecast();
}
