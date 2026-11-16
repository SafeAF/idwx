import assert from 'node:assert/strict';
import { test } from 'node:test';
import { JSDOM } from 'jsdom';
import { forecastAnalytics, mergeForecast, normalizeModel, normalizeNws, normalizeNwsObservation, normalizeOpenMeteo, renderForecast } from '../js/forecast.js';

function openMeteo() {
  const keys = [
    'weather_code', 'temperature_2m_max', 'temperature_2m_min', 'apparent_temperature_max',
    'apparent_temperature_min', 'precipitation_sum', 'precipitation_probability_max', 'snowfall_sum',
    'wind_speed_10m_max', 'wind_gusts_10m_max', 'wind_direction_10m_dominant',
    'relative_humidity_2m_mean', 'dew_point_2m_mean', 'visibility_mean', 'uv_index_max', 'sunrise', 'sunset',
  ];
  const daily = { time: Array.from({ length: 14 }, (_, i) => `2026-11-${String(i + 13).padStart(2, '0')}`) };
  for (const key of keys) daily[key] = Array.from({ length: 14 }, (_, i) => key === 'sunrise' || key === 'sunset' ? `${daily.time[i]}T07:00` : i + 1);
  return { current: { time: '2026-11-13T12:00', temperature_2m: 52, apparent_temperature: 48, relative_humidity_2m: 40, precipitation: 0,
    weather_code: 2, wind_speed_10m: 12, wind_direction_10m: 180, wind_gusts_10m: 21 }, daily };
}

function nws() {
  return { properties: { generatedAt: '2026-11-13T18:00:00Z', updateTime: '2026-11-13T17:00:00Z', periods: [
    { name: 'Friday', startTime: '2026-11-13T06:00:00-07:00', isDaytime: true, temperature: 60,
      probabilityOfPrecipitation: { value: 20 }, windSpeed: '10 mph', windDirection: 'S', shortForecast: 'Partly Sunny', detailedForecast: 'A mild day.' },
    { name: 'Friday Night', startTime: '2026-11-13T18:00:00-07:00', isDaytime: false, temperature: 38,
      probabilityOfPrecipitation: { value: 40 }, windSpeed: '5 mph', windDirection: 'SW', shortForecast: 'Showers', detailedForecast: 'A chance of rain.' },
  ] } };
}

function observation(description = 'Rain') {
  return { properties: {
    timestamp: '2026-11-16T00:53:00Z', textDescription: description,
    temperature: { unitCode: 'wmoUnit:degC', value: 9 },
    windChill: { unitCode: 'wmoUnit:degC', value: 6 },
    heatIndex: { unitCode: 'wmoUnit:degC', value: null },
    relativeHumidity: { unitCode: 'wmoUnit:percent', value: 45 },
    windDirection: { unitCode: 'wmoUnit:degree_(angle)', value: 160 },
    windSpeed: { unitCode: 'wmoUnit:km_h-1', value: 27.78 },
    windGust: { unitCode: 'wmoUnit:km_h-1', value: null },
  } };
}

function ecmwf() {
  const value = openMeteo();
  delete value.current;
  value.daily.temperature_2m_max = value.daily.temperature_2m_max.map(number => number + 4);
  value.daily.temperature_2m_min = value.daily.temperature_2m_min.map(number => number + 2);
  return value;
}

function climatology() {
  return { baseline_period: { start_year: 1980, end_year: 2025 }, days: Array.from({ length: 14 }, (_, index) => ({
    month_day: `11-${String(index + 13).padStart(2, '0')}`, sample_count: 46,
    high_c: { p10: -3, p50: 0, p90: 3 }, low_c: { p10: -8, p50: -5, p90: -2 }, mean_c: { p10: -5, p50: -2.5, p90: 0 },
  })) };
}

test('Open-Meteo normalization keeps fourteen daily outlooks and current wind', () => {
  const result = normalizeOpenMeteo(openMeteo());
  assert.equal(result.days.length, 14);
  assert.equal(result.current.wind, 12);
  assert.equal(result.days[0].high, 1);
  assert.equal(result.days[13].date, '2026-11-26');
});

test('NWS day and night periods are paired by local date', () => {
  const result = normalizeNws(nws());
  assert.equal(result.byDate.get('2026-11-13').day.temperature, 60);
  assert.equal(result.byDate.get('2026-11-13').night.temperature, 38);
});

test('latest NWS station observation reports rain and converts SI values for current conditions', () => {
  const result = normalizeNwsObservation(observation(), Date.parse('2026-11-16T01:00:00Z'));
  assert.equal(result.source, 'NWS observation · KTWF');
  assert.equal(result.description, 'Rain');
  assert.equal(result.symbol, '🌧');
  assert.equal(result.temperature, 48.2);
  assert.ok(Math.abs(result.wind - 17.26) < 0.01);
});

test('stale or incomplete NWS observations fail closed so the model fallback can be used', () => {
  assert.throws(() => normalizeNwsObservation(observation(), Date.parse('2026-11-16T04:00:00Z')), /stale/);
  assert.throws(() => normalizeNwsObservation({ properties: { timestamp: '2026-11-16T00:53:00Z' } }, Date.parse('2026-11-16T01:00:00Z')), /no temperature/);
});

test('a current NWS observation remains visible when both model feeds are unavailable', () => {
  const dom = new JSDOM(`
    <div id="forecast-status"></div><div id="forecast-content" hidden>
      <div id="weather-alerts"></div><section id="current-weather"></section>
      <section id="forecast-analytics"></section><section id="forecast-days"></section>
    </div>`);
  global.document = dom.window.document;
  const currentObservation = normalizeNwsObservation(observation(), Date.parse('2026-11-16T01:00:00Z'));
  renderForecast({ gfs: null, ecmwf: null, currentObservation, fetchedAt: Date.parse('2026-11-16T01:00:00Z'), alerts: [] });
  assert.equal(document.querySelector('.current-summary').textContent, 'Rain');
  assert.equal(document.querySelector('.current-meta .eyebrow').textContent, 'NWS observation · KTWF');
  assert.equal(document.querySelector('#forecast-content').hidden, false);
  assert.match(document.querySelector('#forecast-status').textContent, /model feeds are temporarily unavailable/);
  delete global.document;
});

test('merged outlook compares three live sources, history and marks days eight through fourteen extended', () => {
  const result = mergeForecast(openMeteo(), ecmwf(), nws(), climatology());
  assert.equal(result.days[0].sources.length, 3);
  assert.equal(result.days[0].sources[0].summary, 'Partly Sunny');
  assert.equal(result.days[0].history.sampleCount, 46);
  assert.equal(result.days[0].consensus.sourceCount, 3);
  assert.ok(Number.isFinite(result.days[0].consensus.highAnomaly));
  assert.equal(result.days[6].extended, false);
  assert.equal(result.days[7].extended, true);
  assert.equal(result.days[7].sources.length, 2);
  assert.equal(result.days[13].extended, true);
});

test('analytics identify source coverage, median spread and historical anomaly', () => {
  const result = mergeForecast(openMeteo(), ecmwf(), nws(), climatology());
  const analytics = forecastAnalytics(result.days);
  assert.equal(analytics.threeSourceDays, 1);
  assert.ok(analytics.medianSpread > 0);
  assert.equal(analytics.widestDay, '2026-11-13');
  assert.equal(analytics.aboveHistoryDays, 0);
});

test('a model without current conditions remains a valid comparison source', () => {
  assert.equal(normalizeModel(ecmwf(), 'ECMWF IFS').days.length, 14);
  assert.equal(normalizeModel(ecmwf(), 'ECMWF IFS').current, null);
});

test('incomplete model data fails closed', () => {
  assert.throws(() => normalizeOpenMeteo({ current: {}, daily: { time: ['2026-11-13'] } }), /incomplete forecast/);
  assert.throws(() => mergeForecast(null, null), /both 14-day model sources/i);
});
