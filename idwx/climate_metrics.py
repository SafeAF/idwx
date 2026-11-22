"""Versioned archive catalog. Availability is distinct from semantic admission.

Source definitions: https://docs.oikolab.com/parameters/ (reviewed 2026-11-21).
Raw columns are exact contracts; arbitrary reducers and guessed units are forbidden.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

REGISTRY_VERSION = '1.0.0'
METHOD_VERSION = 'climate_daily_v1'
DEPTHS = ('0–7 cm', '7–28 cm', '28–100 cm', '100–289 cm')


@dataclass(frozen=True)
class Metric:
    id: str
    label: str
    family: str
    source_unit: str
    unit: str
    quantity: str
    minimum: float
    maximum: float
    reducers: tuple[str, ...] = ('min', 'mean', 'max')
    scale: float = 1.0
    date_basis: str = 'instantaneous_fixed_offset'
    admission: str = 'review_required'
    reason: str = 'Full-file quality and independent reduction references required before publication.'
    layer: str | None = None

    @property
    def column(self) -> str:
        return f'{self.id} ({self.source_unit})'

    def public(self) -> dict:
        return {**asdict(self), 'column': self.column, 'coverage': '24 valid distinct hours of 24 expected',
                'source_kind': 'historical_reanalysis', 'method_version': METHOD_VERSION}


_METRICS = [
    Metric('temperature', 'Air temperature · 2 m', 'temperature', 'degC', '°C', 'temperature', -100, 70),
    Metric('dewpoint_temperature', 'Dewpoint · 2 m', 'temperature', 'degC', '°C', 'temperature', -100, 70),
    *[Metric(f'soil_temperature_level_{i}', f'Soil temperature · {depth}', 'soil_temperature', 'degC', '°C', 'temperature', -100, 80, layer=depth) for i, depth in enumerate(DEPTHS, 1)],
    *[Metric(name, label, 'radiation', 'W/m^2', 'W/m²', 'radiation_flux', 0, 1800,
             reason='Hourly flux statistics only after source review; interval duration/end convention is unverified, so integrated energy is unavailable.')
      for name, label in [('surface_thermal_radiation', 'Downward thermal radiation'), ('surface_solar_radiation', 'Surface solar radiation'), ('direct_normal_solar_radiation', 'Direct-normal solar radiation'), ('surface_diffuse_solar_radiation', 'Diffuse solar radiation')]],
    Metric('relative_humidity', 'Relative humidity', 'humidity', '0-1', 'fraction', 'fraction', 0, 1),
    Metric('surface_pressure', 'Surface pressure', 'pressure', 'Pa', 'hPa', 'pressure', 10000, 120000, scale=.01),
    Metric('mean_sea_level_pressure', 'Mean sea-level pressure', 'pressure', 'Pa', 'hPa', 'pressure', 80000, 120000, scale=.01),
    Metric('total_cloud_cover', 'Cloud cover', 'humidity', '0-1', 'fraction', 'fraction', 0, 1),
    *[Metric(name, label, 'water', 'mm of water equivalent', 'mm water equivalent', 'water_amount', 0, 1000,
             reducers=('sum',), date_basis='hour_ending_fixed_offset',
             reason='Extend the reviewed Twin Falls accumulation method with independent references at both reporting offsets before admission.')
      for name, label in [('total_precipitation', 'Total precipitation'), ('snowfall', 'Snowfall water equivalent')]],
    Metric('snow_depth', 'Snow water equivalent on the ground', 'snow', 'mm of water equivalent', 'mm water equivalent', 'water_state', 0, 10000,
           reason='Header declares water equivalent; retain as a state variable. Confirm archive-specific source definition before admission.'),
    Metric('snow_density', 'Snow density', 'snow', 'kg/m^3', 'kg/m³', 'density', 0, 1000),
    *[Metric(f'volumetric_soil_water_layer_{i}', f'Soil moisture · {depth}', 'soil_moisture', '0-1', 'fraction', 'fraction', 0, 1, layer=depth) for i, depth in enumerate(DEPTHS, 1)],
    Metric('wind_speed', 'Wind speed · 10 m', 'wind', 'm/s', 'm/s', 'speed', 0, 150),
    Metric('wind_direction', 'Wind direction · 10 m', 'wind', 'deg', 'degrees', 'direction', 0, 360, reducers=('vector_mean', 'frequency'),
           reason='Requires matched hourly 10 m speed samples, calm counts and a resultant threshold; arithmetic means are prohibited.'),
    Metric('10m_wind_gust', 'Maximum sampled gust · 10 m', 'wind', 'm/s', 'm/s', 'speed', 0, 150, reducers=('max',)),
    Metric('100m_wind_speed', 'Wind speed · 100 m', 'wind', 'm/s', 'm/s', 'speed', 0, 150),
    Metric('100m_wind_direction', 'Wind direction · 100 m', 'wind', 'deg', 'degrees', 'direction', 0, 360, reducers=('vector_mean', 'frequency'),
           reason='Requires matched hourly 100 m speed samples, calm counts and a resultant threshold; arithmetic means are prohibited.'),
]
METRICS = {metric.id: metric for metric in _METRICS}


def metric_catalog() -> dict:
    return {'version': REGISTRY_VERSION, 'method_version': METHOD_VERSION,
            'reference': 'https://docs.oikolab.com/parameters/',
            'metrics': [metric.public() for metric in _METRICS]}
