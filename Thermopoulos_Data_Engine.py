# -*- coding: utf-8 -*-
"""
Thermopoulos.Meteodata v3.1 - Extended Historical Data Engine
==============================================================

Standalone console-executable meteorological data acquisition and processing
engine for the Thermopoulos heat-stress modeling suite.

NEW in v3.1:
- wind_speed_unit="ms" added explicitly to all Open-Meteo API requests.
  Open-Meteo default is km/h; without this parameter wind_10m would be
  stored in km/h in the Excel output while all downstream calculations
  (wind_speed_at_height, globe temperature, MRT, JOS-3, UTCI) assume m/s.
  This fix affects fetch_hourly_forecast() and fetch_historical_data().
  Verified against Open-Meteo API documentation (Settings section,
  default wind speed unit = km/h). March 2026.

NEW in v3.0:
- Automatic 14-day hindcast (historical data before forecast)
- Optional custom historical period selection
- Multi-sheet Excel export (Forecast / Hindcast / Custom Historical)
- Unified processing pipeline for all data types
- Enhanced data validation and error handling

Features from v2.2:
- Interactive city selection
- Hourly meteorological data from Open-Meteo
- Urban Heat Island (UHI) correction (Pre-radiation calculation)
- Mean Radiant Temperature (MRT) via ISO 7726 (HESTIA-consistent)
- UTCI & WBGT calculations
- Vectorized performance for high-speed processing

UTCI CALCULATION NOTES:
- UTCI is calculated using the pythermalcomfort library
- Valid input ranges per UTCI standard:
  * Temperature: -50 to +50°C
  * Wind speed: 0.5 to 17 m/s (1.8 to 61.2 km/h)
  * Relative humidity: 5 to 100%
- Wind values outside this range are automatically clipped
- Column 'UTCI_valid' flags where original wind was within standard range
- UTCI values for clipped wind should be interpreted with caution
- For high wind conditions (>17 m/s), UTCI represents extrapolated estimates

Author: Koos de Boer 25-01-2026
v3.1:  wind_speed_unit fix — March 2026
"""

import os
import sys
import requests
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
import logging

import numpy as np
import pandas as pd
import pvlib
from timezonefinder import TimezoneFinder

# Pythermalcomfort imports
from pythermalcomfort.models import wbgt, utci

# Import wet bulb temperature handling different library versions
try:
    from pythermalcomfort.models import wet_bulb_temperature
    WET_BULB_FUNC = 'models'
except ImportError:
    try:
        from pythermalcomfort.utilities import wet_bulb_tmp as wet_bulb_temperature
        WET_BULB_FUNC = 'utilities'
    except ImportError:
        raise ImportError("Could not import wet bulb temperature function.")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# =============================================================================
# Coastal correction: settings and helper function
# =============================================================================
# [fix] See INTEGRATION_CHANGELOG for the full rationale (Falmouth hindcast,
# KFMH station comparison, literature review of ERA5 coastal bias). Thresholds
# are tuned to the actual resolution of the model in use -- not a single fixed
# assumption that would be correct for one model but wrong for another.
GEO_THRESHOLD_KM = {
    "era5":          12.5,  # 25 km grid, half cell width
    "era5_land":      5.0,  # 9-11 km grid, half cell width
    "era5_seamless":  5.0,  # temperature/humidity come from ERA5-Land
}

# [PROVISIONAL, VALIDATED AGAINST ONE STATION/DAY] Peak value (around solar
# elevation maximum) of the coastal correction, in °C. Originally based on the
# 95th-percentile coastal bias figure from the literature (~4K, "Multivariate
# bias correction of ERA5", IOP Science 2026) and our own KFMH comparison at
# Falmouth (3-4°C around midday). Re-validated against real KFMH station
# observations for Falmouth, 2024-08-10: peak=4.0 gave a mean absolute error
# of 0.92°C across race-window hours; peak=3.0 reduced that to 0.47°C, with
# the error now split roughly symmetrically (sometimes slightly low, sometimes
# slightly high) rather than consistently biased in one direction -- a good
# sign for the shape of the correction, but this is still based on a single
# day/location and should not be treated as a definitively calibrated value.
# Override this per location once you have your own station comparison (see
# test_era5_seamless.py / diagnose_era5_land.py for how to do that).
COASTAL_CORRECTION_PEAK_C = 3.0

# =============================================================================
# Terrain roughness: settings and helper function
# =============================================================================
# [fix v3.1] Previously inferred purely from `city.get("population") > 0`
# (0.8 "urban" vs. 0.1 "rural") -- which for a named place is almost always
# true regardless of what the actual course/site looks like (e.g. an open
# beachfront road in a town with population data still got 0.8). Replaced
# by an explicit choice from a small set of common terrain categories,
# following the standard Davenport/Wieringa roughness-length classification.
# z0 values in metres.
ROUGHNESS_Z0_TERRAIN = {
    "1": ("Open water / zee",                              0.0002),
    "2": ("Open kust, strand, kort gras (weinig obstakels)", 0.03),
    "3": ("Open agrarisch terrein, verspreide obstakels",    0.1),
    "4": ("Parkland / verspreide bebouwing, bomen",          0.3),
    "5": ("Voorstedelijk (lage bebouwing, tuinen, bomen)",   0.8),
    "6": ("Stedelijk / binnenstad (hoge, dichte bebouwing)", 1.5),
}

# [fix v3.1] Previously a module-level global set as a side effect inside
# fetch_historical_data() and read back inside process_weather_data(). That
# meant whichever dataset was fetched *last* (hindcast, then optionally
# custom-historical) silently determined the coastal-correction status used
# when *every* dataset -- including the forecast, processed first in
# Step 5 -- was processed. Replaced by an explicit (df, coastal_active)
# return value from each fetch_* function, threaded through main() and
# passed directly into process_weather_data(). No shared state anymore.

def _haversine_km(lat1, lon1, lat2, lon2):
    """Distance in km between two coordinates (spherical-earth approximation)."""
    r = 6371.0
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(a))

# =============================================================================
# Geocoding & API
# =============================================================================

def geocode_city_candidates(city: str, max_results: int = 7) -> List[Dict]:
    """Geocode city name to coordinates."""
    url = "https://geocoding-api.open-meteo.com/v1/search"
    params = {"name": city, "count": max_results, "language": "en", "format": "json"}
    
    try:
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()
    except requests.RequestException as e:
        logger.error(f"Geocoding failed: {e}")
        raise

    if "results" not in data:
        raise ValueError(f"No locations found for '{city}'.")
    return data["results"]

def select_city(candidates: List[Dict]) -> Dict:
    """Interactive city selection from multiple candidates."""
    print("\nMultiple locations found:\n")
    for i, c in enumerate(candidates, start=1):
        print(f"[{i}] {c['name']}, {c.get('country', 'Unknown')} "
              f"({c.get('admin1', '')}) — Pop: {c.get('population', 'unknown')}")

    while True:
        try:
            choice = int(input(f"\nSelect location [1–{len(candidates)}]: "))
            if 1 <= choice <= len(candidates):
                return candidates[choice - 1]
        except ValueError:
            pass
        print("Invalid selection.")

def fetch_hourly_forecast(lat: float, lon: float, tz: str, days: int = 7) -> pd.DataFrame:
    """
    Fetch hourly weather forecast from Open-Meteo API.
    
    Parameters
    ----------
    lat : float
        Latitude
    lon : float
        Longitude
    tz : str
        Timezone string (e.g., 'Europe/Amsterdam')
    days : int
        Number of forecast days (default: 7, max: 16)
    
    Returns
    -------
    pd.DataFrame
        Hourly forecast data with timezone-aware index
    """
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat, 
        "longitude": lon, 
        "forecast_days": days,
        "hourly": ["temperature_2m", "relative_humidity_2m", "surface_pressure",
                   "wind_speed_10m", "shortwave_radiation", "cloud_cover"],
        "timezone": tz,
        "wind_speed_unit": "ms"   # explicit: Open-Meteo default is km/h; all
                                  # downstream code (wind_speed_at_height, globe,
                                  # MRT, JOS-3, UTCI) expects m/s. (v3.1 fix)
    }

    try:
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()["hourly"]
    except requests.RequestException as e:
        logger.error(f"Weather forecast API failed: {e}")
        raise

    df = pd.DataFrame(data)
    df["time"] = pd.to_datetime(df["time"])
    df.set_index("time", inplace=True)
    
    if df.index.tz is None:
        df.index = df.index.tz_localize(tz, nonexistent='shift_forward', ambiguous='infer')

    df.rename(columns={
        "temperature_2m": "T_air_rural",
        "relative_humidity_2m": "RH",
        "surface_pressure": "pressure",
        "wind_speed_10m": "wind_10m",
        "shortwave_radiation": "solar_radiation",
        "cloud_cover": "cloud_cover"
    }, inplace=True)

    # [fix v3.1] Forecast uses Open-Meteo's best-match high-resolution model
    # (not era5_seamless), which doesn't have ERA5's land/sea grid-dilution
    # problem, so coastal_active is always False here. Returned explicitly
    # per dataset instead of being inherited from whatever a *different*
    # fetch call (hindcast/custom-historical) last set on a shared global.
    coastal_active = False

    return df, coastal_active

def fetch_historical_data(lat: float, lon: float, tz: str, 
                         start_date: str, end_date: str) -> pd.DataFrame:
    """
    Fetch hourly historical weather data from Open-Meteo Archive API.
    
    Parameters
    ----------
    lat : float
        Latitude
    lon : float
        Longitude
    tz : str
        Timezone string
    start_date : str
        Start date in 'YYYY-MM-DD' format
    end_date : str
        End date in 'YYYY-MM-DD' format
    
    Returns
    -------
    tuple of (pd.DataFrame, bool)
        Hourly historical data with timezone-aware index, and a
        coastal_active flag (True if the ERA5 grid cell used is far enough
        from the requested location to warrant the coastal correction --
        see GEO_THRESHOLD_KM). Specific to *this* dataset only.
    """
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": ["temperature_2m", "relative_humidity_2m", "surface_pressure",
                   "wind_speed_10m", "shortwave_radiation", "cloud_cover"],
        "timezone": tz,
        "wind_speed_unit": "ms",  # explicit: Open-Meteo default is km/h; all
                                  # downstream code (wind_speed_at_height, globe,
                                  # MRT, JOS-3, UTCI) expects m/s. (v3.1 fix)
        # [fix] Without an explicit 'models' parameter, this endpoint used
        # plain ERA5 (0.25° / ~25 km) -- too coarse for narrow coastal strips
        # (e.g. Woods Hole/Falmouth), where a grid cell falls largely over
        # open water and temperature/WBGT come out systematically too low/
        # damped as a result (the sea warms up much more slowly than land).
        # era5_seamless combines ERA5-Land (0.1° / ~9-11 km) for temperature/
        # humidity with plain ERA5 for wind/solar radiation (which aren't
        # available at finer resolution) -- best resolution where possible,
        # with no gaps in the required variables. Available from 1950
        # onward, so also suitable for the Falmouth hindcasts (1997, 2003).
        "models": "era5_seamless",
        # [fix] Was already the default, now set explicitly so it's visible
        # in the code and can't silently change with a future Open-Meteo
        # update.
        "cell_selection": "land",
    }

    try:
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        response_json = response.json()
        data = response_json["hourly"]
    except requests.RequestException as e:
        logger.error(f"Historical weather API failed: {e}")
        raise

    # [fix] Automatic geo-distance check: compare the grid-cell coordinates
    # Open-Meteo actually used against the requested location. Threshold is
    # tuned to the real resolution of the model in use (~9-11 km for
    # era5_land/era5_seamless, ~25 km for plain era5) -- not a single fixed
    # assumption that would be right for one model and wrong for another.
    # [fix v3.1] Returns coastal_active as a local value tied to *this*
    # dataset's own distance check, instead of setting a module-level global
    # that a later fetch call (or a dataset processed afterward) would
    # silently inherit.
    returned_lat = response_json.get("latitude", lat)
    returned_lon = response_json.get("longitude", lon)
    distance_km = _haversine_km(lat, lon, returned_lat, returned_lon)
    threshold_km = GEO_THRESHOLD_KM.get(params.get("models", "era5"), 12.5)
    if distance_km > threshold_km:
        logger.warning(
            f"Grid cell is {distance_km:.1f} km from the requested location "
            f"(threshold {threshold_km} km for model '{params.get('models')}') "
            f"-- possible coastal bias. Coastal correction activated for this dataset."
        )
        coastal_active = True
    else:
        coastal_active = False

    df = pd.DataFrame(data)
    df["time"] = pd.to_datetime(df["time"])
    df.set_index("time", inplace=True)
    
    if df.index.tz is None:
        df.index = df.index.tz_localize(tz, nonexistent='shift_forward', ambiguous='infer')

    df.rename(columns={
        "temperature_2m": "T_air_rural",
        "relative_humidity_2m": "RH",
        "surface_pressure": "pressure",
        "wind_speed_10m": "wind_10m",
        "shortwave_radiation": "solar_radiation",
        "cloud_cover": "cloud_cover"
    }, inplace=True)

    return df, coastal_active

def validate_weather_data(df: pd.DataFrame, data_type: str = "weather") -> pd.DataFrame:
    """
    Validate weather data for physical plausibility and missing values.
    
    Parameters
    ----------
    df : pd.DataFrame
        Weather data to validate
    data_type : str
        Description for logging (e.g., 'forecast', 'historical')
    
    Returns
    -------
    pd.DataFrame
        Validated data (may have warnings logged)
    """
    # Check for missing values
    missing_pct = df.isnull().sum() / len(df) * 100
    if (missing_pct > 0).any():
        logger.warning(f"{data_type.capitalize()} data has missing values:")
        for col in missing_pct[missing_pct > 0].index:
            logger.warning(f"  {col}: {missing_pct[col]:.1f}% missing")
    
    # Temperature bounds
    if 'T_air_rural' in df.columns:
        temp_out_of_bounds = (df['T_air_rural'] < -90) | (df['T_air_rural'] > 60)
        if temp_out_of_bounds.any():
            logger.error(f"{data_type.capitalize()}: {temp_out_of_bounds.sum()} temperature values out of physical bounds (-90 to 60°C)")
    
    # Humidity bounds
    if 'RH' in df.columns:
        rh_out_of_bounds = (df['RH'] < 0) | (df['RH'] > 100)
        if rh_out_of_bounds.any():
            logger.warning(f"{data_type.capitalize()}: {rh_out_of_bounds.sum()} RH values out of bounds (0-100%)")
            df.loc[df['RH'] < 0, 'RH'] = 0
            df.loc[df['RH'] > 100, 'RH'] = 100
    
    # Wind bounds
    if 'wind_10m' in df.columns:
        wind_extreme = df['wind_10m'] > 50  # > 180 km/h
        if wind_extreme.any():
            logger.warning(f"{data_type.capitalize()}: {wind_extreme.sum()} extreme wind values (>50 m/s)")
    
    return df

# =============================================================================
# Solar & Wind Physics
# =============================================================================

def calculate_solar_parameters(df: pd.DataFrame, lat: float, lon: float, tz: str) -> pd.DataFrame:
    """Calculate solar elevation angle using pvlib."""
    location = pvlib.location.Location(lat, lon, tz=tz)
    try:
        solar_pos = location.get_solarposition(df.index)
        df['solar_elevation'] = solar_pos['elevation'].clip(lower=0)
    except Exception as e:
        logger.error(f"Solar calculation failed: {e}")
        logger.warning("Setting solar_elevation to 0 - MRT accuracy will be reduced")
        df['solar_elevation'] = 0
    return df

def wind_speed_at_height(v_ref, z_ref, z_target, z0=0.1):
    """
    Logarithmic wind profile correction.
    
    Parameters
    ----------
    v_ref : array-like
        Reference wind speed [m/s]
    z_ref : float
        Reference height [m]
    z_target : float
        Target height [m]
    z0 : float
        Roughness length [m] (0.1 = rural, 0.8 = urban)
    
    Returns
    -------
    array-like
        Wind speed at target height [m/s]
    
    Notes
    -----
    Simplified logarithmic wind profile. Does not include:
    - Displacement height (d) for urban canopy
    - Atmospheric stability corrections (Monin-Obukhov)
    Valid only in surface layer (<100m) and neutral conditions.
    """
    if z_ref <= 0 or z_target <= 0 or z0 <= 0:
        logger.warning(f"Invalid wind profile parameters: z_ref={z_ref}, z_target={z_target}, z0={z0}. Using reference wind.")
        return v_ref
    return v_ref * (np.log(z_target / z0) / np.log(z_ref / z0))

def dew_point_from_rh(t_air, rh):
    """
    Dew point via the standard Magnus formula.

    Parameters
    ----------
    t_air : array-like
        Air temperature [°C]
    rh : array-like
        Relative humidity [%]

    Returns
    -------
    array-like
        Dew point [°C]

    Notes
    -----
    Used by the coastal correction (see coastal_active in process_weather_data): the dew
    point represents the absolute moisture content and stays unchanged when
    T_air is corrected -- only the RH derived from it changes. Magnus
    constants (a=17.62, b=243.12) are the common, widely used set for
    temperatures between 0-60°C; not validated for extreme cold, which is
    not an issue here since this correction is only relevant for heat cases.
    """
    a, b = 17.62, 243.12
    gamma = (a * t_air) / (b + t_air) + np.log(np.clip(rh, 1e-6, 100.0) / 100.0)
    return (b * gamma) / (a - gamma)

def rh_from_dew_point(t_air, dew_point):
    """
    Recompute relative humidity from dew point and (new) T_air, inverse of
    dew_point_from_rh, same Magnus constants.
    """
    a, b = 17.62, 243.12
    numerator = np.exp((a * dew_point) / (b + dew_point))
    denominator = np.exp((a * t_air) / (b + t_air))
    return np.clip(100.0 * numerator / denominator, 0.0, 100.0)

def calculate_uhi_oke(population: int, index: pd.DatetimeIndex) -> pd.Series:
    """
    Oke (1973) population-based UHI model with modern corrections.
    
    Parameters
    ----------
    population : int
        City population
    index : pd.DatetimeIndex
        Datetime index for output series
    
    Returns
    -------
    pd.Series
        UHI temperature increment [°C] with diurnal variation
    
    References
    ----------
    Oke, T. R. (1973). City size and the urban heat island.
    Stewart & Oke (2012). Local Climate Zones for Urban Temperature Studies.
    MDPI (2017). Western Sydney UHI measurements (99th percentile: 5-6°C)
    
    Notes
    -----
    Original Oke formula overestimates UHI for modern large cities.
    A 40% reduction is applied for cities > 1M population to account for:
    - Modern urban planning and green spaces
    - Better building standards
    - Empirical validation (Sydney 99th percentile: 5.7°C vs formula 9.4°C)
    
    Limitations:
    - No seasonal variation
    - No climate zone dependency
    - No green space factor
    - Diurnal factors are simplified (step functions)
    """
    if not population or population <= 0:
        return pd.Series(0.0, index=index)

    # Calculate base UHI using Oke formula
    base_uhi = 2.01 * np.log10(population) - 4.06
    
    # Apply modern city correction for large populations
    if population > 1_000_000:
        # Large cities: reduce by 40% for modern planning
        uhi_max = base_uhi * 0.6
        uhi_max = np.clip(uhi_max, 0.0, 6.0)  # Cap at 6°C (realistic maximum)
    else:
        # Smaller cities: original formula with conservative cap
        uhi_max = np.clip(base_uhi, 0.0, 5.0)

    # Vectorized diurnal factors
    hours = index.hour
    factors = np.zeros(len(index))
    
    # Apply factors based on hour ranges
    # Night: maximum UHI (city retains heat)
    factors[(hours >= 0) & (hours <= 5)] = 1.0
    # Morning transition
    factors[(hours >= 6) & (hours <= 9)] = 0.6
    # Day: minimum UHI (ventilation, solar heating of rural areas)
    factors[(hours >= 10) & (hours <= 16)] = 0.2
    # Evening transition
    factors[(hours >= 17) & (hours <= 20)] = 0.5
    # Late night buildup
    factors[(hours >= 21) & (hours <= 23)] = 0.9

    return pd.Series(uhi_max * factors, index=index)

# =============================================================================
# Thermal Physics (Vectorized Wrappers)
# =============================================================================

def _calculate_globe_scalar(dry_bulb, ghi, wind, solar_elev, pressure, cloud, aqi=1):
    """
    Scalar implementation of ISO 7726 globe temperature.
    
    Parameters
    ----------
    dry_bulb : float
        Dry bulb temperature [°C]
    ghi : float
        Global horizontal irradiance [W/m²]
    wind : float
        Wind speed at measurement height [m/s]
    solar_elev : float
        Solar elevation angle [degrees]
    pressure : float
        Surface pressure [hPa]
    cloud : float
        Cloud cover [%]
    aqi : int
        Air quality index (1-5, affects sky temperature)
    
    Returns
    -------
    float
        Globe temperature [°C]
    
    Notes
    -----
    Uses Newton-Raphson iteration to solve thermal equilibrium.
    Convergence tolerance: 0.1°C energy imbalance.
    Maximum iterations: 10 (may not converge in extreme conditions).
    """
    GLOBE_DIA = 0.15
    EMISSIVITY = 0.95
    ABSORPTIVITY = 0.95
    SIGMA = 5.67e-8
    
    t_air_k = dry_bulb + 273.15
    eff_wind = max(0.1, wind)
    h_c = 6.3 * (eff_wind**0.6) / (GLOBE_DIA**0.4)

    # NIGHT / NO SUN
    if ghi <= 0 or solar_elev <= 0:
        sky_depression = 10 - (cloud / 100) * 6
        if aqi > 3: sky_depression -= 2
        sky_temp_k = t_air_k - sky_depression
        net_rad = EMISSIVITY * SIGMA * (sky_temp_k**4 - t_air_k**4)
        t_globe = dry_bulb + (net_rad / h_c)
        return max(dry_bulb - 3, min(dry_bulb + 1, t_globe))

    # DAY
    solar_input = ABSORPTIVITY * ghi / 4.0
    amb_rad = EMISSIVITY * SIGMA * (t_air_k**4)
    total_in = solar_input + amb_rad
    
    # Newton-Raphson iteration
    t_globe_k = t_air_k + (solar_input / (h_c + 4 * EMISSIVITY * SIGMA * t_air_k**3))
    
    converged = False
    for iteration in range(10):
        rad_out = EMISSIVITY * SIGMA * (t_globe_k**4)
        conv_out = h_c * (t_globe_k - t_air_k)
        imbalance = total_in - (rad_out + conv_out)
        
        if abs(imbalance) < 0.1:
            converged = True
            break
            
        deriv = 4 * EMISSIVITY * SIGMA * (t_globe_k**3) + h_c
        t_globe_k += 0.7 * (imbalance / deriv)
    
    # Log convergence issues (only for extreme cases)
    if not converged and abs(imbalance) > 1.0:
        logger.debug(f"Globe temp convergence issue: imbalance={imbalance:.2f}°C at T={dry_bulb:.1f}°C, GHI={ghi:.0f} W/m²")

    t_globe_c = t_globe_k - 273.15
    
    # Sanity checks
    limit = 15 if eff_wind > 2 else 20
    return max(dry_bulb, min(dry_bulb + limit, t_globe_c))

def _calculate_mrt_scalar(tg, ta, v, ghi, elev):
    """
    Scalar implementation of ISO 7726 Mean Radiant Temperature.
    
    Parameters
    ----------
    tg : float
        Globe temperature [°C]
    ta : float
        Air temperature [°C]
    v : float
        Wind speed [m/s]
    ghi : float
        Global horizontal irradiance [W/m²]
    elev : float
        Solar elevation [degrees]
    
    Returns
    -------
    float
        Mean Radiant Temperature [°C]
    """
    GLOBE_DIA = 0.15
    EMISSIVITY = 0.95
    SIGMA = 5.67e-8
    
    tg_k = tg + 273.15
    ta_k = ta + 273.15
    eff_wind = max(0.1, v)
    h_c = 6.3 * (eff_wind**0.6) / (GLOBE_DIA**0.4)
    
    conv_term = (h_c / (EMISSIVITY * SIGMA)) * (tg_k - ta_k)
    mrt_k4 = (tg_k**4) + conv_term
    
    if mrt_k4 < 0: 
        return tg
    
    mrt_c = (mrt_k4**0.25) - 273.15
    
    # Validation logic
    if ghi > 0 and elev > 0:
        return np.clip(mrt_c, tg - 2, tg + 15)
    else:
        return np.clip(mrt_c, ta - 15, ta + 3)

# Create vectorized versions for DataFrame application
calculate_globe_vectorized = np.vectorize(_calculate_globe_scalar)
calculate_mrt_vectorized = np.vectorize(_calculate_mrt_scalar)

# =============================================================================
# Unified Processing Pipeline
# =============================================================================

def process_weather_data(df: pd.DataFrame, city: Dict, lat: float, lon: float, tz: str,
                          coastal_active: bool, roughness_z0: float) -> pd.DataFrame:
    """
    Complete processing pipeline for weather data (forecast or historical).
    
    Parameters
    ----------
    df : pd.DataFrame
        Raw weather data from API
    city : dict
        City metadata (must contain 'population' key)
    lat : float
        Latitude
    lon : float
        Longitude
    tz : str
        Timezone string
    coastal_active : bool
        Whether the coastal correction applies to *this* dataset. Comes
        directly from the fetch_* call that produced `df` -- never shared
        across datasets. [fix v3.1]
    roughness_z0 : float
        Terrain roughness length (m) used for the 10m->1.5m wind-speed
        profile. User-selected per run from a fixed set of terrain
        categories (see ROUGHNESS_Z0_TERRAIN) -- no longer inferred from
        population. [fix v3.1]
    
    Returns
    -------
    pd.DataFrame
        Fully processed data with all thermal indices
    """
    # 1. Solar calculations
    df = calculate_solar_parameters(df, lat, lon, tz)

    # [fix] Temporary coastal correction, only applied if the geo-distance
    # check (grid-cell coordinates vs. requested location) raised a warning
    # elsewhere in the call chain. Applied before the UHI step, so UHI is
    # still added normally on top of the corrected T_air_rural afterward,
    # rather than double-counting or cutting across it.
    # Dew point stays fixed (the physically correct assumption for a T
    # correction without a new moisture measurement), RH is recomputed from
    # it -- not simply left unchanged next to a higher T_air.
    # [PROVISIONAL, PARTIALLY VALIDATED] COASTAL_CORRECTION_PEAK_C is based on
    # comparison against a single real weather station (KFMH, Falmouth,
    # 10 Aug 2003) -- explicitly flagged as a stopgap, see the
    # coastal_correction_applied column below.
    if coastal_active:
        # [fix] Diurnal scaling instead of a flat correction: ERA5's sea
        # component has almost no diurnal cycle (confirmed in the
        # literature), while land does warm up during the day -- the
        # land-sea contrast, and therefore the dilution error, is largest
        # around midday and near zero at night. Solar elevation (already
        # computed elsewhere in this function) is a convenient, already
        # available proxy for this.
        scale = np.sin(np.radians(np.clip(df["solar_elevation"].values, 0, 90)))
        correction_c = COASTAL_CORRECTION_PEAK_C * scale
        dew_point = dew_point_from_rh(df["T_air_rural"].values, df["RH"].values)
        df["T_air_rural"] = df["T_air_rural"] + correction_c
        df["RH"] = rh_from_dew_point(df["T_air_rural"].values, dew_point)
        df["coastal_correction_applied"] = True
        df["coastal_correction_degrees"] = correction_c
    else:
        df["coastal_correction_applied"] = False
        df["coastal_correction_degrees"] = 0.0

    # 2. UHI & Urban Temperature
    # [fix v3.1] UHI still follows population (that part was fine -- Oke's
    # UHI relationship is genuinely population-driven). Roughness no longer
    # is: it now comes from the explicit roughness_z0 parameter (terrain
    # chosen by the user, see ROUGHNESS_Z0_TERRAIN/select_terrain_roughness),
    # since "does this place have a population field > 0" said nothing
    # about whether the actual course/site is open coast, farmland, or a
    # built-up area.
    pop = city.get("population", 0)
    if pop and pop > 0:
        logger.info(f"Applying UHI for population: {pop:,}")
        df["UHI_delta"] = calculate_uhi_oke(pop, df.index)
        df["T_air_urban"] = df["T_air_rural"] + df["UHI_delta"]
    else:
        logger.info("No population data - assuming rural conditions")
        df["UHI_delta"] = 0.0
        df["T_air_urban"] = df["T_air_rural"]
    
    # 3. Wind at height (1.5m) using correct roughness
    df["wind_1.5m"] = wind_speed_at_height(
        df["wind_10m"].values, 10.0, 1.5, z0=roughness_z0
    )
    
    # 4. Globe & MRT (Vectorized)
    logger.info("Calculating radiation balance (Globe/MRT)...")
    df["T_globe"] = calculate_globe_vectorized(
        df["T_air_urban"].values,
        df["solar_radiation"].values,
        df["wind_1.5m"].values,
        df["solar_elevation"].values,
        df["pressure"].values,
        df["cloud_cover"].values
    )
    
    df["MRT"] = calculate_mrt_vectorized(
        df["T_globe"].values,
        df["T_air_urban"].values,
        df["wind_1.5m"].values,
        df["solar_radiation"].values,
        df["solar_elevation"].values
    )
    
    # 5. Wet Bulb
    logger.info("Calculating Wet Bulb temperature...")
    if WET_BULB_FUNC == 'models':
        df["T_wetbulb"] = wet_bulb_temperature(
            tdb=df["T_air_urban"].values,
            rh=df["RH"].values,
            pressure=df["pressure"].values
        )
    else:
        logger.warning("Using pressure-independent wet bulb - reduced accuracy")
        df["T_wetbulb"] = wet_bulb_temperature(
            tdb=df["T_air_urban"].values,
            rh=df["RH"].values
        )
    
    # 6. WBGT
    logger.info("Calculating WBGT...")
    wbgt_res = wbgt(
        twb=df["T_wetbulb"].values,
        tg=df["T_globe"].values,
        tdb=df["T_air_urban"].values,
        with_solar_load=True,
    )
    df["WBGT"] = wbgt_res.wbgt if hasattr(wbgt_res, 'wbgt') else wbgt_res
    
    # 7. UTCI (with input validation for pythermalcomfort limits)
    logger.info("Calculating UTCI...")
    
    # UTCI valid ranges (per pythermalcomfort library):
    # - Temperature: -50 to +50°C
    # - Wind speed: 0.5 to 17 m/s
    # - Relative humidity: 5 to 100%
    
    # Store original wind values
    wind_original = df["wind_10m"].values.copy()
    
    # Clip wind to UTCI valid range
    wind_clipped = np.clip(wind_original, 0.5, 17.0)
    
    # Track which values were clipped
    wind_clipped_low = (wind_original < 0.5).sum()
    wind_clipped_high = (wind_original > 17.0).sum()
    
    if wind_clipped_low > 0 or wind_clipped_high > 0:
        logger.warning(f"UTCI wind limits applied: {wind_clipped_low} values clipped to 0.5 m/s, "
                      f"{wind_clipped_high} values clipped to 17 m/s")
        logger.warning(f"Original wind range: {wind_original.min():.1f} - {wind_original.max():.1f} m/s")
    
    # Calculate UTCI with clipped wind values
    utci_res = utci(
        tdb=df["T_air_urban"].values,
        tr=df["MRT"].values,
        v=wind_clipped,  # Use clipped wind values
        rh=df["RH"].values
    )
    df["UTCI"] = utci_res.utci if hasattr(utci_res, 'utci') else utci_res
    
    # Add flag column to indicate where UTCI is extrapolated beyond standard limits
    df["UTCI_valid"] = (wind_original >= 0.5) & (wind_original <= 17.0)
    valid_pct = df["UTCI_valid"].sum() / len(df) * 100
    logger.info(f"UTCI validity: {valid_pct:.1f}% of values within standard range (0.5-17 m/s wind)")
    
    return df

# =============================================================================
# User Input & Date Management
# =============================================================================

def select_terrain_roughness() -> Tuple[float, str]:
    """
    Interactive prompt for terrain roughness selection.

    Replaces the old population>0 heuristic: the actual course/site terrain
    (e.g. an open beachfront road) often has nothing to do with whether the
    place as a whole has population data. [fix v3.1]

    Returns
    -------
    tuple of (roughness_z0, label)
        z0 in metres, and the terrain description for the Excel metadata.
    """
    print("\n" + "="*60)
    print("Terrain roughness (for 10m -> 1.5m wind-speed profile)")
    print("="*60)
    print("Choose the terrain that best matches the actual course/site --")
    print("not just the town it's located in.\n")

    for key, (label, z0) in ROUGHNESS_Z0_TERRAIN.items():
        print(f"  {key}. {label} (z0 = {z0} m)")

    default_key = "5"
    while True:
        choice = input(f"\nEnter choice [{default_key}]: ").strip() or default_key
        if choice in ROUGHNESS_Z0_TERRAIN:
            label, z0 = ROUGHNESS_Z0_TERRAIN[choice]
            print(f"  Using: {label} (z0 = {z0} m)")
            return z0, label
        print("Invalid choice, please enter one of: " + ", ".join(ROUGHNESS_Z0_TERRAIN.keys()))

def get_custom_historical_period() -> Optional[Tuple[str, str]]:
    """
    Interactive prompt for custom historical period selection.
    
    Returns
    -------
    tuple of (start_date, end_date) as strings, or None if skipped
    """
    print("\n" + "="*60)
    print("OPTIONAL: Custom Historical Period")
    print("="*60)
    print("You can retrieve additional historical data for any past period.")
    print("Note: Open-Meteo Archive has data from 1940 onwards.")
    print("\nPress ENTER to skip, or enter dates to proceed.")
    
    choice = input("\nDo you want to fetch custom historical data? (y/N): ").strip().lower()
    
    if choice not in ['y', 'yes']:
        return None
    
    print("\nEnter date range (format: YYYY-MM-DD)")
    
    while True:
        try:
            start_str = input("Start date: ").strip()
            end_str = input("End date: ").strip()
            
            # Validate dates
            start = datetime.strptime(start_str, "%Y-%m-%d")
            end = datetime.strptime(end_str, "%Y-%m-%d")
            
            if start >= end:
                print("Error: Start date must be before end date.")
                continue
            
            if end >= datetime.now():
                print("Error: End date must be in the past.")
                continue
            
            # Check maximum span (API limit typically 1 year for free tier)
            days_span = (end - start).days
            if days_span > 365:
                print(f"Warning: {days_span} days requested. This may take longer to process.")
                confirm = input("Continue? (y/N): ").strip().lower()
                if confirm not in ['y', 'yes']:
                    continue
            
            return (start_str, end_str)
            
        except ValueError as e:
            print(f"Invalid date format. Please use YYYY-MM-DD. Error: {e}")
        except KeyboardInterrupt:
            print("\nSkipping custom historical period.")
            return None

# =============================================================================
# Excel Export (Multi-Sheet)
# =============================================================================

def save_to_excel_multisheet(
    forecast_df: pd.DataFrame,
    hindcast_df: pd.DataFrame,
    historical_df: Optional[pd.DataFrame],
    meta: Dict
) -> str:
    """
    Save multiple datasets to Excel with separate sheets.
    
    Parameters
    ----------
    forecast_df : pd.DataFrame
        7-day forecast data
    hindcast_df : pd.DataFrame
        14-day hindcast data
    historical_df : pd.DataFrame or None
        Optional custom historical data
    meta : dict
        Metadata dictionary
    
    Returns
    -------
    str
        Filepath of saved Excel file
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    filename = f"Thermopoulos_DataEngine_{meta['city']}_{timestamp}.xlsx"
    
    try:
        # Remove timezone for Excel compatibility
        forecast_save = forecast_df.copy()
        hindcast_save = hindcast_df.copy()
        
        if forecast_save.index.tz is not None:
            forecast_save.index = forecast_save.index.tz_localize(None)
        if hindcast_save.index.tz is not None:
            hindcast_save.index = hindcast_save.index.tz_localize(None)
        
        with pd.ExcelWriter(filename, engine="xlsxwriter") as writer:
            # Sheet 1: Forecast
            forecast_save.to_excel(writer, sheet_name="Forecast_7d")
            
            # Sheet 2: Hindcast
            hindcast_save.to_excel(writer, sheet_name="Hindcast_14d")
            
            # Sheet 3: Custom Historical (if provided)
            if historical_df is not None:
                historical_save = historical_df.copy()
                if historical_save.index.tz is not None:
                    historical_save.index = historical_save.index.tz_localize(None)
                historical_save.to_excel(writer, sheet_name="Historical_Custom")
            
            # Sheet 4: Metadata
            meta_df = pd.DataFrame([meta])
            meta_df.to_excel(writer, sheet_name="Metadata", index=False)
            
            # Format sheets
            workbook = writer.book
            header_format = workbook.add_format({
                'bold': True,
                'bg_color': '#4472C4',
                'font_color': 'white',
                'border': 1
            })
            
            for sheet_name in writer.sheets:
                worksheet = writer.sheets[sheet_name]
                if sheet_name != "Metadata":
                    # Freeze first row and column
                    worksheet.freeze_panes(1, 1)
        
        filepath = os.path.abspath(filename)
        logger.info(f"Excel file saved: {filepath}")
        print(f"\n{'='*60}")
        print(f"✓ Data exported successfully!")
        print(f"{'='*60}")
        print(f"File: {filepath}")
        print(f"\nSheets created:")
        print(f"  1. Forecast_7d      - {len(forecast_df)} hours")
        print(f"  2. Hindcast_14d     - {len(hindcast_df)} hours")
        if historical_df is not None:
            print(f"  3. Historical_Custom - {len(historical_df)} hours")
        print(f"  {'4' if historical_df is not None else '3'}. Metadata")
        
        return filepath
        
    except Exception as e:
        logger.error(f"Excel save failed: {e}")
        raise

# =============================================================================
# Main Execution
# =============================================================================

def main():
    """Main execution function."""
    print("\n" + "="*60)
    print("  Thermopoulos.Meteodata v3.0 - Data Engine")
    print("="*60)
    print("  Automated: 7-day Forecast + 14-day Hindcast")
    print("  Optional: Custom Historical Period")
    print("="*60 + "\n")

    # =========================================================================
    # STEP 1: City Selection
    # =========================================================================
    city_name = input("Enter city name: ").strip()
    if not city_name:
        print("No city entered. Exiting.")
        return

    try:
        candidates = geocode_city_candidates(city_name)
        city = candidates[0] if len(candidates) == 1 else select_city(candidates)
    except Exception as e:
        print(f"Error during geocoding: {e}")
        return

    lat = city["latitude"]
    lon = city["longitude"]
    tz = city["timezone"]
    
    print(f"\n{'='*60}")
    print(f"Selected: {city['name']}, {city.get('country', 'Unknown')}")
    print(f"Coordinates: {lat:.4f}°, {lon:.4f}°")
    print(f"Timezone: {tz}")
    print(f"Population: {city.get('population', 'Unknown'):,}" if city.get('population') else "Population: Unknown")
    print(f"{'='*60}\n")

    # [fix v3.1] Explicit terrain choice, replacing the population>0
    # heuristic. Applies to all three datasets fetched below.
    roughness_z0, terrain_label = select_terrain_roughness()

    # =========================================================================
    # STEP 2: Fetch 7-Day Forecast
    # =========================================================================
    print("Fetching 7-day forecast...")
    try:
        forecast_df, forecast_coastal_active = fetch_hourly_forecast(lat, lon, tz, days=16)
        forecast_df = validate_weather_data(forecast_df, "forecast")
        logger.info(f"Forecast data: {len(forecast_df)} hours ({forecast_df.index[0]} to {forecast_df.index[-1]})")
    except Exception as e:
        print(f"Error fetching forecast: {e}")
        return

    # =========================================================================
    # STEP 3: Fetch 14-Day Hindcast
    # =========================================================================
    print("Fetching 14-day hindcast (historical)...")
    
    # Calculate date range: 14 days before first forecast date
    forecast_start = forecast_df.index[0].date()
    hindcast_end = forecast_start - timedelta(days=1)
    hindcast_start = hindcast_end - timedelta(days=13)  # 14 days total
    
    try:
        hindcast_df, hindcast_coastal_active = fetch_historical_data(
            lat, lon, tz,
            start_date=hindcast_start.strftime("%Y-%m-%d"),
            end_date=hindcast_end.strftime("%Y-%m-%d")
        )
        hindcast_df = validate_weather_data(hindcast_df, "hindcast")
        logger.info(f"Hindcast data: {len(hindcast_df)} hours ({hindcast_df.index[0]} to {hindcast_df.index[-1]})")
    except Exception as e:
        print(f"Error fetching hindcast: {e}")
        return

    # =========================================================================
    # STEP 4: Optional Custom Historical Period
    # =========================================================================
    historical_df = None
    custom_period = get_custom_historical_period()
    
    if custom_period:
        start_date, end_date = custom_period
        print(f"\nFetching custom historical data: {start_date} to {end_date}...")
        
        try:
            historical_df, historical_coastal_active = fetch_historical_data(lat, lon, tz, start_date, end_date)
            historical_df = validate_weather_data(historical_df, "custom historical")
            logger.info(f"Custom historical data: {len(historical_df)} hours")
        except Exception as e:
            logger.error(f"Error fetching custom historical data: {e}")
            print("Continuing without custom historical data...")
            historical_df = None
            historical_coastal_active = False

    # =========================================================================
    # STEP 5: Process All Datasets
    # =========================================================================
    print("\n" + "="*60)
    print("Processing thermal calculations...")
    print("="*60 + "\n")
    
    # Process forecast
    # [fix v3.1] Each call gets its *own* dataset's coastal_active value
    # (forecast_coastal_active, hindcast_coastal_active, ...) -- no longer a
    # shared global that the last fetch call happened to set. roughness_z0
    # is the single value chosen once in select_terrain_roughness(), used
    # consistently across all three datasets since they describe the same
    # physical site.
    print("[1/3] Processing 7-day forecast...")
    forecast_df = process_weather_data(forecast_df, city, lat, lon, tz,
                                        coastal_active=forecast_coastal_active,
                                        roughness_z0=roughness_z0)
    
    # Process hindcast
    print("\n[2/3] Processing 14-day hindcast...")
    hindcast_df = process_weather_data(hindcast_df, city, lat, lon, tz,
                                        coastal_active=hindcast_coastal_active,
                                        roughness_z0=roughness_z0)
    
    # Process custom historical (if exists)
    if historical_df is not None:
        print("\n[3/3] Processing custom historical period...")
        historical_df = process_weather_data(historical_df, city, lat, lon, tz,
                                              coastal_active=historical_coastal_active,
                                              roughness_z0=roughness_z0)
    else:
        print("\n[3/3] No custom historical period - skipping.")

    # =========================================================================
    # STEP 6: Export to Excel
    # =========================================================================
    print("\n" + "="*60)
    print("Saving to Excel...")
    print("="*60 + "\n")
    
    meta = {
        "city": city["name"],
        "country": city.get("country", "Unknown"),
        "latitude": lat,
        "longitude": lon,
        "timezone": tz,
        "population": city.get("population", 0),
        "roughness_z0": roughness_z0,
        "terrain_type": terrain_label,
        "model_version": "Thermopoulos v3.0",
        "processing_timestamp": datetime.now().isoformat(),
        "forecast_period": f"{forecast_df.index[0]} to {forecast_df.index[-1]}",
        "hindcast_period": f"{hindcast_df.index[0]} to {hindcast_df.index[-1]}",
        "custom_historical_period": f"{historical_df.index[0]} to {historical_df.index[-1]}" if historical_df is not None else "None"
    }
    
    try:
        filepath = save_to_excel_multisheet(forecast_df, hindcast_df, historical_df, meta)
    except Exception as e:
        print(f"Error saving Excel file: {e}")
        return

    # =========================================================================
    # STEP 7: Summary Statistics
    # =========================================================================
    print("\n" + "="*60)
    print("SUMMARY STATISTICS (Urban Conditions)")
    print("="*60 + "\n")
    
    print("7-DAY FORECAST:")
    print(forecast_df[["T_air_urban", "MRT", "WBGT", "UTCI"]].describe().loc[['mean', 'min', 'max']].round(1))
    
    print("\n14-DAY HINDCAST:")
    print(hindcast_df[["T_air_urban", "MRT", "WBGT", "UTCI"]].describe().loc[['mean', 'min', 'max']].round(1))
    
    if historical_df is not None:
        print("\nCUSTOM HISTORICAL PERIOD:")
        print(historical_df[["T_air_urban", "MRT", "WBGT", "UTCI"]].describe().loc[['mean', 'min', 'max']].round(1))
    
    print("\n" + "="*60)
    print("Processing complete!")
    print("="*60)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nProcess interrupted by user. Exiting...")
    except Exception as e:
        logger.exception("Fatal error occurred")
        print(f"\nFatal error: {e}")
        sys.exit(1)
