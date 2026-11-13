import assert from 'node:assert/strict';
import { test } from 'node:test';
import { mergeForecast, normalizeNws, normalizeOpenMeteo } from '../js/forecast.js';

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

test('merged outlook compares NWS only where available and marks days eight through fourteen extended', () => {
  const result = mergeForecast(openMeteo(), nws());
  assert.equal(result.days[0].nws.day.summary, 'Partly Sunny');
  assert.equal(result.days[6].extended, false);
  assert.equal(result.days[7].extended, true);
  assert.equal(result.days[13].extended, true);
});

test('incomplete model data fails closed', () => {
  assert.throws(() => normalizeOpenMeteo({ current: {}, daily: { time: ['2026-11-13'] } }), /incomplete forecast/);
});
