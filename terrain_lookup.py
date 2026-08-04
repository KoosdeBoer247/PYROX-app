"""
Real terrain classification along a GPX route, via ESA WorldCover
=====================================================================
Free, no API key: ESA WorldCover 10 m global land-cover (2021, v200),
public AWS Open Data COGs. https://esa-worldcover.org, CC BY 4.0
(Zanaga et al. 2022, doi:10.5281/zenodo.5571936). Tiles are 3x3 degree,
named by their SW corner per ESA's documented convention (e.g. N51E003
covers 51-54N, 3-6E) -- confirmed from ESA's own product user manual and
AWS Open Data Registry docs, not guessed.

WHAT THIS CHANGES, AND WHAT IT DELIBERATELY DOESN'T
-------------------------------------------------------
Per-segment terrain roughness (z0) feeds into the SAME wind-profile and
WBGT/globe-temperature functions Thermopoulos_Data_Engine already uses
(wind_speed_at_height, calculate_globe_vectorized, wbgt) -- no new
physics is introduced here, only a spatially-varying input to existing,
already-tested functions.

- WBGT and MRT DO vary by terrain segment: both depend on the 1.5m wind,
  which depends on roughness.
- UTCI does NOT vary by terrain segment. pythermalcomfort's UTCI is
  defined at a fixed 10m reference wind by convention, independent of
  local roughness -- showing a terrain-varying UTCI would misrepresent
  what the index actually measures, so it is deliberately left as the
  single city-level value along the whole route.
- T_air, humidity, radiation, cloud cover are NOT varied by segment.
  A 16 km race course is far smaller than one weather grid cell, so only
  the wind-dependent terms have a physical basis for varying here; the
  rest would be fabricated spatial detail the input data doesn't support.

LAND-COVER DATA CAVEAT: ESA WorldCover's "Built-up" class does not
distinguish suburban from dense urban core (both are one class), so
built-up segments default to the "Voorstedelijk" (suburban) roughness
category as the more conservative middle option -- override manually if
a segment is known to be a dense city centre.

NOT TESTED WITH A LIVE FETCH IN DEVELOPMENT. This sandbox's network
allowlist does not include AWS S3, so the actual remote-COG read could
not be exercised end-to-end while building this. The tile-naming and
bucket structure follow ESA's own documented, publicly verified
conventions, but the first real run on Streamlit Cloud (full internet
access) is the actual test. Any fetch failure falls back to the
sidebar's manual terrain selector, with a visible message rather than a
silent failure or a fabricated result.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from Thermopoulos_Data_Engine import (
    ROUGHNESS_Z0_TERRAIN,
    wind_speed_at_height,
    calculate_globe_vectorized,
    calculate_mrt_vectorized,
    wet_bulb_temperature,
    WET_BULB_FUNC,
)
from pythermalcomfort.models import wbgt as _wbgt

WORLDCOVER_BASE_URL = "https://esa-worldcover.s3.amazonaws.com/v200/2021/map"

# ESA WorldCover 11-class legend -> nearest ROUGHNESS_Z0_TERRAIN key.
# NOTE: ROUGHNESS_Z0_TERRAIN is keyed by STRINGS ('1'..'6'), not ints --
# it backs a CLI input() prompt in Thermopoulos_Data_Engine. Keys here
# must match that exactly or the lookup raises KeyError.
# See ROUGHNESS_Z0_TERRAIN for the z0 values and Dutch labels themselves.
_WORLDCOVER_TO_ROUGHNESS_KEY = {
    10: "4",   # Tree cover            -> Parkland/verspreide bebouwing, bomen
    20: "3",   # Shrubland             -> Open agrarisch terrein
    30: "3",   # Grassland             -> Open agrarisch terrein
    40: "3",   # Cropland              -> Open agrarisch terrein
    50: "5",   # Built-up              -> Voorstedelijk (conservative default; see caveat above)
    60: "2",   # Bare / sparse veg.    -> Open kust/strand/kort gras
    70: "1",   # Snow and ice          -> Open water/zee (roughness proxy)
    80: "1",   # Permanent water       -> Open water/zee
    90: "1",   # Herbaceous wetland    -> Open water/zee
    95: "1",   # Mangroves             -> Open water/zee
    100: "2",  # Moss and lichen       -> Open kust/strand/kort gras
}

# Fallback for any class not in the table above (e.g. 0 = no data):
# open farmland, the middle of the roughness range.
_DEFAULT_ROUGHNESS_KEY = "3"

_WORLDCOVER_LABELS = {
    10: "Tree cover", 20: "Shrubland", 30: "Grassland", 40: "Cropland",
    50: "Built-up", 60: "Bare / sparse vegetation", 70: "Snow and ice",
    80: "Permanent water", 90: "Herbaceous wetland", 95: "Mangroves",
    100: "Moss and lichen",
}


def worldcover_tile_id(lat: float, lon: float) -> str:
    """3x3-degree tile name for ESA WorldCover, keyed by the tile's SW
    corner (e.g. N51E003 for 51-54N, 3-6E)."""
    lat_floor = int(math.floor(lat / 3.0) * 3)
    lon_floor = int(math.floor(lon / 3.0) * 3)
    ns = "N" if lat_floor >= 0 else "S"
    ew = "E" if lon_floor >= 0 else "W"
    return f"{ns}{abs(lat_floor):02d}{ew}{abs(lon_floor):03d}"


def worldcover_tile_url(lat: float, lon: float) -> str:
    tile = worldcover_tile_id(lat, lon)
    return f"{WORLDCOVER_BASE_URL}/ESA_WorldCover_10m_2021_v200_{tile}_Map.tif"


def fetch_landcover_along_route(route_df: pd.DataFrame, sample_every_km: float = 0.2) -> dict:
    """Sample ESA WorldCover class at points along the route (subsampled
    every `sample_every_km` to limit remote reads -- land cover doesn't
    change fast enough to need every GPX trackpoint), then forward/back-
    fill the classification onto every route point.

    Returns {"route": route_df with worldcover_class/roughness_z0/
    terrain_label columns added, "tiles_used": [...], "error": str|None}.
    Never raises -- callers should check "error" and fall back to the
    manual terrain selector on failure.
    """
    import rasterio

    out = route_df.copy()
    if out.empty:
        return {"route": out, "tiles_used": [], "error": "Empty route."}

    # Subsample: at least start/end, plus points every sample_every_km.
    total_km = float(out["distance_km"].iloc[-1])
    n_samples = max(2, int(total_km / max(sample_every_km, 0.05)) + 1)
    sample_dists = np.linspace(0, total_km, n_samples)
    sample_idx = out["distance_km"].searchsorted(sample_dists).clip(0, len(out) - 1)
    sample_idx = sorted(set(sample_idx.tolist()))

    tiles_used = {}
    classes = {}
    try:
        for i in sample_idx:
            lat, lon = float(out.loc[i, "lat"]), float(out.loc[i, "lon"])
            tile_id = worldcover_tile_id(lat, lon)
            if tile_id not in tiles_used:
                url = worldcover_tile_url(lat, lon)
                tiles_used[tile_id] = rasterio.open(url)
            ds = tiles_used[tile_id]
            value = list(ds.sample([(lon, lat)]))[0][0]
            classes[i] = int(value)
    except Exception as e:
        for ds in tiles_used.values():
            try:
                ds.close()
            except Exception:
                pass
        return {
            "route": route_df, "tiles_used": [], "error":
            f"Could not fetch ESA WorldCover data ({e}). "
            "Falling back to the manually selected terrain type for the whole route."
        }
    finally:
        for ds in tiles_used.values():
            try:
                ds.close()
            except Exception:
                pass

    wc_series = pd.Series(classes).reindex(range(len(out)))
    wc_series = wc_series.interpolate(method="nearest").ffill().bfill()
    out["worldcover_class"] = wc_series.astype(int)
    out["roughness_key"] = out["worldcover_class"].map(
        lambda c: _WORLDCOVER_TO_ROUGHNESS_KEY.get(int(c), _DEFAULT_ROUGHNESS_KEY)
    )
    out["roughness_z0"] = out["roughness_key"].map(lambda k: ROUGHNESS_Z0_TERRAIN[k][1])
    out["terrain_label"] = out["roughness_key"].map(lambda k: ROUGHNESS_Z0_TERRAIN[k][0])
    out["worldcover_label"] = out["worldcover_class"].map(
        lambda c: _WORLDCOVER_LABELS.get(int(c), f"Unclassified ({int(c)})")
    )

    return {"route": out, "tiles_used": list(tiles_used.keys()), "error": None}


def recompute_wbgt_mrt_for_terrain(route_df_timed: pd.DataFrame, weather_df: pd.DataFrame) -> pd.DataFrame:
    """Recompute WBGT and MRT per route point using that point's own
    roughness_z0 (from fetch_landcover_along_route), reusing the exact
    globe/WBGT functions Thermopoulos_Data_Engine uses for the city-level
    calculation -- only the 1.5m wind input varies by segment.

    `route_df_timed` must already have clock_time and roughness_z0
    columns (pace_schedule + fetch_landcover_along_route). Other inputs
    (T_air_urban, RH, pressure, cloud_cover, solar_radiation,
    solar_elevation, wind_10m) are interpolated from weather_df, which
    must carry those raw columns (the app's forecast_df does).
    """
    out = route_df_timed.copy()
    # Same resolution hazard as in gpx_route.interpolate_weather_along_route:
    # astype("int64") is ns or us depending on pandas' chosen resolution, so
    # use the shared helper instead (see its docstring).
    from gpx_route import _to_epoch_seconds
    reference = weather_df.index[0]
    idx_numeric = _to_epoch_seconds(weather_df.index, reference)
    query_numeric = _to_epoch_seconds(out["clock_time"], reference)

    needed = ["T_air_urban", "RH", "pressure", "cloud_cover",
              "solar_radiation", "solar_elevation", "wind_10m"]
    missing = [c for c in needed if c not in weather_df.columns]
    if missing:
        raise ValueError(f"weather_df is missing required columns: {missing}")

    for col in needed:
        out[col] = np.interp(query_numeric, idx_numeric, weather_df[col].to_numpy())

    _wind_at_height_vec = np.vectorize(wind_speed_at_height, otypes=[float])
    out["wind_1.5m_terrain"] = _wind_at_height_vec(
        out["wind_10m"].to_numpy(), 10.0, 1.5, out["roughness_z0"].to_numpy()
    )
    out["T_globe_terrain"] = calculate_globe_vectorized(
        out["T_air_urban"].to_numpy(), out["solar_radiation"].to_numpy(),
        out["wind_1.5m_terrain"].to_numpy(), out["solar_elevation"].to_numpy(),
        out["pressure"].to_numpy(), out["cloud_cover"].to_numpy(),
    )
    out["MRT"] = calculate_mrt_vectorized(
        out["T_globe_terrain"].to_numpy(), out["T_air_urban"].to_numpy(),
        out["wind_1.5m_terrain"].to_numpy(), out["solar_radiation"].to_numpy(),
        out["solar_elevation"].to_numpy(),
    )

    if WET_BULB_FUNC == "models":
        out["T_wetbulb_terrain"] = wet_bulb_temperature(
            tdb=out["T_air_urban"].to_numpy(), rh=out["RH"].to_numpy(),
            pressure=out["pressure"].to_numpy(),
        )
    else:
        out["T_wetbulb_terrain"] = wet_bulb_temperature(
            tdb=out["T_air_urban"].to_numpy(), rh=out["RH"].to_numpy(),
        )

    wbgt_res = _wbgt(
        twb=out["T_wetbulb_terrain"].to_numpy(), tg=out["T_globe_terrain"].to_numpy(),
        tdb=out["T_air_urban"].to_numpy(), with_solar_load=True,
    )
    out["WBGT"] = wbgt_res.wbgt if hasattr(wbgt_res, "wbgt") else wbgt_res

    # UTCI is intentionally NOT recomputed here -- see module docstring.
    # It stays whatever was already interpolated onto route_df_timed
    # (the single city-level value), unchanged by terrain.
    return out
