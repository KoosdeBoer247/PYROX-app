# -*- coding: utf-8 -*-
"""
Thermopoulos data loader
================================================================================
One data source for the whole suite. Reads the Excel file produced by the
Thermopoulos Data Engine and serves it in the two shapes the models need:

    get_daily_heat_loads(...)  -> for PYROX  (one baseline_heat_load per day)
    get_hourly_weather(...)    -> for HESTIA (hourly weather dicts)

This is the cleaned-up successor to the ThermopoulosDataLoader in the old
Integrated_Thermopoulos_System.py. The data-reading logic is unchanged in
spirit; what is new and explicit here is the WEATHER -> BASELINE_HEAT_LOAD
bridge that PYROX needs (see daily_weather_to_heat_load).

No exec(), no subprocess, no global namespace tricks. Plain importable module.
"""

from __future__ import annotations

from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional

import numpy as np
import pandas as pd


# ============================================================================
# WEATHER -> BASELINE HEAT LOAD  (the Tier-1 -> Tier-2 bridge for PYROX)
# ============================================================================
#
# PYROX's dynamics run on an abstract daily "baseline_heat_load" (the old R1),
# NOT on raw temperature. Something has to map weather onto that scale. The old
# code did it with a full JOS-3 physiological simulation per day, aggregated and
# scaled to 0-5. That is the most faithful route and is provided separately by
# run_pyrox.py when JOS-3 is available.
#
# For a fast, transparent, dependency-free default we provide a simple
# WBGT-style index below. It is deliberately explicit about its scale because we
# learned the hard way that baseline_heat_load scale is decisive for model
# behaviour: a load near 1.0 is "mild", a sustained load near 1.3-1.5 is a
# stressful heatwave for vulnerable groups, and loads >= 2 push every group
# toward runaway. The mapping is tuned to put a hot summer day around 1.0-1.5.
#
# This is a MODELLING CHOICE, not a measured relationship. It is isolated in one
# documented function so it can be swapped for the JOS-3 route without touching
# anything else.

HEAT_LOAD_REFERENCE_TEMP = 22.0
"""Apparent temperature (deg C) mapped to a baseline_heat_load of ~0 (benign)."""

HEAT_LOAD_PER_DEGREE = 0.10
"""Baseline_heat_load gained per deg C of apparent temperature above reference.

Calibrated against Fouillet et al. (2006), Paris August 2003: at this scaling the
healthy-older group reproduces the observed mortality timing (deviation ~day 4-6,
critical/peak around day 9-12) and the age gradient (no excess <35 y, heaviest
load on 75+). NOTE: this calibrates the weather->load BRIDGE, not the dynamics.
Still to be tested against a second, cooler-climate heatwave to confirm the
bridge is climate-robust (it may need nightly Tmin, since Paris nights >23 C were
themselves lethal). Previous engineering value was 0.11; Fouillet refines, not
overturns, it."""


def apparent_temperature(t_air: float, relative_humidity: float,
                         wind_speed: float) -> float:
    """A simple humidity- and wind-adjusted "feels like" temperature (deg C).

    Humidity adds heat (impaired evaporative cooling), wind removes a little
    (convective cooling). This is a lightweight stand-in for a full WBGT/UTCI
    computation, adequate for the daily aggregate PYROX consumes.
    """
    humidity_load = 0.05 * max(0.0, relative_humidity - 40.0)   # +deg per %RH over 40
    wind_relief = 0.3 * max(0.0, wind_speed - 1.0)              # -deg per m/s over 1
    return t_air + humidity_load - wind_relief


def daily_weather_to_heat_load(t_air_max: float, relative_humidity: float,
                               wind_speed: float) -> float:
    """Map one day's weather aggregates to PYROX's baseline_heat_load.

    Uses the day's apparent maximum temperature, referenced and scaled by the
    module constants above. Clamped at 0 (cooler-than-reference days carry no
    heat load). See the block comment above for the scale rationale.
    """
    feels_like = apparent_temperature(t_air_max, relative_humidity, wind_speed)
    load = (feels_like - HEAT_LOAD_REFERENCE_TEMP) * HEAT_LOAD_PER_DEGREE
    return max(0.0, load)


# ----------------------------------------------------------------------------
# Optional refinement: nocturnal recovery via overnight minimum temperature
# ----------------------------------------------------------------------------
# PYROX's recovery term is scaled by a per-day `sleep_quality` factor (default
# 1.0 = full recovery). Physiologically, a warm night disrupts sleep and blunts
# overnight recovery — and the absence of nocturnal cooling is one of the
# strongest predictors of heatwave harm. This helper translates the night's
# minimum temperature into a sleep-quality factor, so the recovery term reflects
# how warm the night actually was rather than assuming full recovery.
#
# This is OPTIONAL and OFF by default: the baseline behaviour (sleep_quality=1.0)
# is unchanged, so the Paris-2003 calibration and validation are untouched. Pass
# the resulting series to PyroxModel.simulate(sleep_quality_series=...).

NIGHT_COMFORT_TEMP = 18.0   # °C — at/below this, overnight recovery is full
NIGHT_SEVERE_TEMP = 26.0    # °C — at/above this (tropical night), recovery floored
NIGHT_RECOVERY_FLOOR = 0.4  # even a hot night allows some recovery


def night_sleep_quality(t_air_min: float,
                        comfort: float = NIGHT_COMFORT_TEMP,
                        severe: float = NIGHT_SEVERE_TEMP,
                        floor: float = NIGHT_RECOVERY_FLOOR) -> float:
    """Translate a night's minimum temperature into a sleep-quality factor in
    [floor, 1.0], for use as PYROX's per-day `sleep_quality`.

    - t_min <= comfort (18 °C): undisturbed sleep, factor 1.0
    - t_min >= severe  (26 °C): tropical night, factor floored (0.4)
    - linear in between.

    This is a plausibility mapping (like the weather→load bridge), not a measured
    relationship; the thresholds reflect the common definition of a tropical
    night (>=20 °C) widened to a comfort/severe band. It refines the recovery
    term, not the heat-load bridge, so it leaves the PYROX dynamics calibration
    intact.
    """
    if t_air_min <= comfort:
        return 1.0
    if t_air_min >= severe:
        return floor
    frac = (t_air_min - comfort) / (severe - comfort)
    return 1.0 - (1.0 - floor) * frac


# ============================================================================
# LOADER
# ============================================================================

class ThermopoulosData:
    """Reads a Thermopoulos Excel file and serves model-ready data.

    Parameters
    ----------
    excel_path : str | Path
        Path to a Thermopoulos_*.xlsx file produced by the data engine.
    """

    def __init__(self, excel_path: str | Path):
        self.excel_path = Path(excel_path)
        if not self.excel_path.exists():
            raise FileNotFoundError(
                f"Thermopoulos Excel not found: {self.excel_path}\n"
                "Run the Thermopoulos Data Engine first to generate it."
            )

        meta = pd.read_excel(self.excel_path, sheet_name="Metadata")
        self.city = meta["city"].iloc[0]
        self.latitude = float(meta["latitude"].iloc[0])
        self.longitude = float(meta["longitude"].iloc[0])
        self.timezone = meta["timezone"].iloc[0]
        self.population = int(meta["population"].iloc[0]) if "population" in meta else 0

        with pd.ExcelFile(self.excel_path) as xls:
            self.available_sheets = [s for s in xls.sheet_names if s != "Metadata"]

    # ---- internal -----------------------------------------------------------
    def _load_sheet(self, sheet: str) -> pd.DataFrame:
        if sheet not in self.available_sheets:
            raise ValueError(
                f"Sheet '{sheet}' not found. Available: {self.available_sheets}"
            )
        df = pd.read_excel(self.excel_path, sheet_name=sheet, index_col=0)
        df.index = pd.to_datetime(df.index)
        return df

    # ---- metadata -----------------------------------------------------------
    def metadata(self) -> Dict:
        return {
            "city": self.city,
            "latitude": self.latitude,
            "longitude": self.longitude,
            "timezone": self.timezone,
            "population": self.population,
        }

    # ---- PYROX: daily baseline heat loads -----------------------------------
    def get_daily_heat_loads(self, sheet: str = "Forecast_7d") -> pd.DataFrame:
        """Return one row per day with weather aggregates AND the derived
        baseline_heat_load that PYROX consumes.

        Columns: date, t_air_mean, t_air_max, t_air_min, rh_mean,
                 wind_mean, solar_mean, baseline_heat_load
        """
        hourly = self._load_sheet(sheet)
        grouped = hourly.groupby(hourly.index.date).agg(
            t_air_mean=("T_air_urban", "mean"),
            t_air_max=("T_air_urban", "max"),
            t_air_min=("T_air_urban", "min"),
            rh_mean=("RH", "mean"),
            wind_mean=("wind_10m", "mean"),
            solar_mean=("solar_radiation", "mean"),
        )
        daily = grouped.reset_index(names="date")
        daily["baseline_heat_load"] = daily.apply(
            lambda r: daily_weather_to_heat_load(
                r["t_air_max"], r["rh_mean"], r["wind_mean"]
            ),
            axis=1,
        )
        return daily

    def get_heat_load_series(self, sheet: str = "Forecast_7d") -> List[float]:
        """Just the baseline_heat_load list, ready to feed PyroxModel.simulate."""
        return self.get_daily_heat_loads(sheet)["baseline_heat_load"].tolist()

    def get_combined_window(self,
                            hindcast_days: int = 14,
                            forecast_days: int = 7,
                            hindcast_sheet: str = "Hindcast_14d",
                            forecast_sheet: str = "Forecast_7d") -> pd.DataFrame:
        """Concatenate the tail of the hindcast with the head of the forecast
        into one continuous daily series, for a forecast-style PYROX run that
        uses REAL recent history instead of an assumed pre-heatwave load.

        Returns a DataFrame with the daily columns plus:
            - 'date'                : the calendar date of each day
            - 'baseline_heat_load'  : the load that day
            - 'period'              : 'hindcast' or 'forecast'
        ordered chronologically (oldest hindcast day first). The number of
        forecast days is capped (default 7) because Open-Meteo's high-resolution
        local models only retain skill for the first ~week; later forecast days
        fall back to coarser global models.

        The boundary between the last hindcast row and the first forecast row is
        the 'forecast start', which the plots mark with a vertical line. The
        hindcast portion is what warms up the acclimatization (FIR) memory, so
        the forecast strain begins from a physiologically grounded state.
        """
        hind = self.get_daily_heat_loads(hindcast_sheet).copy()
        fore = self.get_daily_heat_loads(forecast_sheet).copy()

        # keep the most recent `hindcast_days` of history and the first
        # `forecast_days` of the forecast
        hind = hind.tail(hindcast_days)
        fore = fore.head(forecast_days)

        hind["period"] = "hindcast"
        fore["period"] = "forecast"

        combined = pd.concat([hind, fore], ignore_index=True)
        return combined

    def forecast_start_index(self, combined: pd.DataFrame) -> int:
        """Index of the first forecast day in a get_combined_window() frame
        (= number of hindcast days), for placing the transition marker."""
        return int((combined["period"] == "hindcast").sum())

    @staticmethod
    def sleep_quality_series(daily: pd.DataFrame) -> List[float]:
        """Build a per-day sleep-quality series from a daily frame's
        't_air_min', using night_sleep_quality(). Use the result as
        PyroxModel.simulate(sleep_quality_series=...) to let warm nights blunt
        overnight recovery.

        If 't_air_min' is absent, returns all-1.0 (the unchanged baseline).
        """
        if "t_air_min" not in daily.columns:
            return [1.0] * len(daily)
        return [night_sleep_quality(float(tm)) for tm in daily["t_air_min"]]

    # ---- HESTIA: hourly weather --------------------------------------------
    def get_hourly_weather(self, sheet: str = "Forecast_7d",
                           start_time: Optional[str] = None,
                           duration_hours: Optional[float] = None) -> List[Dict]:
        """Return hourly weather as a list of dicts in HESTIA's expected shape:
            {dt, main:{temp, humidity, pressure}, wind:{speed}, clouds:{all},
             solar:{ghi, elevation}, radiant:{globe_temp, mrt, twb},
             indices:{wbgt, utci}}
        Optionally restricted to a [start_time, start_time+duration] window.

        [fix] solar/radiant/indices fields added so HESTIA can use
        Thermopoulos/Klimatos's own precomputed solar radiation, solar
        elevation, globe temperature, MRT, wet-bulb temperature, WBGT and
        UTCI directly, instead of recomputing them internally. The solar/
        globe/MRT recalculation was found (Amsterdam DtD hindcast,
        2024-09-22) to underestimate GHI by roughly 2.5-3x at high cloud
        cover (132 vs ~350-400 W/m^2 at 97% cloud_cover) versus this
        already-computed, more realistic reference. `twb` closes the same
        gap for wet-bulb temperature (Klimatos can use a fuller, pressure-
        aware psychrometric model vs. HESTIA's Stull-only recalculation).
        `wbgt`/`utci` close the last step of the same chain: even with
        correct twb/tg/mrt inputs, HESTIA was still recomputing the final
        WBGT/UTCI values itself via the pythermalcomfort wbgt()/utci() calls
        rather than reading Thermopoulos's own already-computed columns --
        verified to leave a small (~0.2-1.6C) residual gap after the twb/tg
        fixes alone. Using the precomputed columns directly closes that.
        These keys are optional/None if the sheet predates this column set,
        so old files still load (callers fall back to the internal model).
        """
        hourly = self._load_sheet(sheet)
        if start_time is not None:
            start = pd.to_datetime(start_time)
            if duration_hours is not None:
                hourly = hourly.loc[start:start + pd.Timedelta(hours=duration_hours)]
            else:
                hourly = hourly.loc[start:]

        out = []
        for timestamp, row in hourly.iterrows():
            # [fix] `timestamp` here is naive (no tz attached, straight from the
            # Excel index). Calling .timestamp() on a naive Timestamp uses
            # Python's SYSTEM timezone to interpret it -- on this sandboxed
            # environment that's UTC, silently shifting every "11:00 local"
            # reading to the epoch value for "11:00 UTC" (2 hours off from the
            # true CEST instant). interpolate_weather()'s times_ts, in
            # contrast, is built from tz-aware Amsterdam-local timestamps --
            # so the two epoch scales didn't line up, and np.interp silently
            # clamped every precomputed field (ghi/mrt/twb/wbgt/utci) to
            # whichever single hourly value happened to fall in range,
            # instead of genuinely interpolating between hours. Caught via a
            # real hindcast run (Amsterdam, 2024-09-22 10:30) where WBGT/UTCI
            # stayed completely flat across a 1.5h window despite the raw
            # hourly data clearly changing (WBGT 19.5 -> 21.0). Explicitly
            # localizing to this sheet's own known timezone makes the epoch
            # value correct regardless of what timezone the machine running
            # this code happens to be set to.
            dt_local = timestamp.tz_localize(self.timezone) if timestamp.tzinfo is None else timestamp
            out.append({
                "dt": int(dt_local.timestamp()),
                "main": {
                    "temp": float(row["T_air_urban"]),
                    "humidity": float(row["RH"]),
                    "pressure": float(row.get("pressure", 1013.25)),
                },
                "wind": {"speed": float(row["wind_10m"])},
                "clouds": {"all": float(row.get("cloud_cover", 50))},
                "solar": {
                    "ghi": float(row["solar_radiation"]) if "solar_radiation" in row and pd.notna(row["solar_radiation"]) else None,
                    "elevation": float(row["solar_elevation"]) if "solar_elevation" in row and pd.notna(row["solar_elevation"]) else None,
                },
                "radiant": {
                    "globe_temp": float(row["T_globe"]) if "T_globe" in row and pd.notna(row["T_globe"]) else None,
                    "mrt": float(row["MRT"]) if "MRT" in row and pd.notna(row["MRT"]) else None,
                    "twb": float(row["T_wetbulb"]) if "T_wetbulb" in row and pd.notna(row["T_wetbulb"]) else None,
                },
                "indices": {
                    "wbgt": float(row["WBGT"]) if "WBGT" in row and pd.notna(row["WBGT"]) else None,
                    "utci": float(row["UTCI"]) if "UTCI" in row and pd.notna(row["UTCI"]) else None,
                },
            })
        return out


# ============================================================================
# convenience: find the newest Thermopoulos file in a directory
# ============================================================================

def find_latest_thermopoulos_file(directory: str | Path = ".") -> Optional[Path]:
    """Return the most recently modified Thermopoulos_*.xlsx, or None."""
    files = sorted(Path(directory).glob("Thermopoulos_*.xlsx"),
                   key=lambda p: p.stat().st_mtime, reverse=True)
    return files[0] if files else None


def find_all_thermopoulos_files(directory: str | Path = ".") -> List[Path]:
    """Return every Thermopoulos_*.xlsx in the directory, newest first."""
    return sorted(Path(directory).glob("Thermopoulos_*.xlsx"),
                  key=lambda p: p.stat().st_mtime, reverse=True)


def _city_from_filename(path: Path) -> str:
    """Best-effort city name from a Thermopoulos_<City>_<timestamp>.xlsx name."""
    stem = path.stem  # e.g. Thermopoulos_DataEngine_Schagerbrug_20260623_1906
    parts = stem.split("_")
    # drop leading 'Thermopoulos'/'DataEngine' tokens and trailing date/time tokens
    parts = [p for p in parts if p not in ("Thermopoulos", "DataEngine")]
    # trailing tokens that are all digits are date/time -> drop them
    while parts and parts[-1].isdigit():
        parts.pop()
    return " ".join(parts) if parts else stem


def choose_thermopoulos_file(directory: str | Path = ".") -> Optional[Path]:
    """Interactive menu: list all Thermopoulos files and let the user pick one.

    Returns the chosen Path, or None if there are no files / the user quits.
    Reads the city name from each file's Metadata sheet when possible, falling
    back to parsing the filename.
    """
    files = find_all_thermopoulos_files(directory)
    if not files:
        print("No Thermopoulos_*.xlsx files found in this directory.")
        return None
    if len(files) == 1:
        print(f"One data file found: {files[0].name}")
        return files[0]

    print("\nAvailable Thermopoulos data files:")
    print("-" * 60)
    for i, f in enumerate(files, 1):
        try:
            meta = pd.read_excel(f, sheet_name="Metadata")
            city = str(meta["city"].iloc[0])
        except Exception:
            city = _city_from_filename(f)
        mtime = datetime.fromtimestamp(f.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
        print(f"  {i:>2}. {city:<22} ({mtime})  {f.name}")
    print("-" * 60)

    while True:
        choice = input(f"Choose a file (1-{len(files)}, or Q to quit): ").strip()
        if choice.lower() in ("q", "quit", ""):
            return None
        if choice.isdigit() and 1 <= int(choice) <= len(files):
            return files[int(choice) - 1]
        print("  Invalid choice, try again.")


if __name__ == "__main__":
    # Smoke test against the newest file in the working directory, if any.
    latest = find_latest_thermopoulos_file()
    if latest is None:
        print("No Thermopoulos_*.xlsx found in current directory.")
    else:
        data = ThermopoulosData(latest)
        print(f"Loaded {data.city}, sheets: {data.available_sheets}")
        daily = data.get_daily_heat_loads(data.available_sheets[0])
        print(daily[["date", "t_air_max", "rh_mean", "baseline_heat_load"]].to_string(index=False))
