# -*- coding: utf-8 -*-
"""
GPX route parsing and race-day exposure mapping
=================================================
Turns a race-route GPX file (trackpoints + optional water-post waypoints)
into a distance profile, and combines it with a pace assumption and start
time to show what T_air/WBGT/UTCI a runner will actually experience at
each point on the course, and during which stretch of the day.

SCOPE, DELIBERATELY NARROW: a race lasts hours, not days, so this reuses
the SAME-DAY WBGT/UTCI screening layer (decision_support.py -- the
Activity/rest guide), not PYROX's multi-day cumulative-strain model,
which answers a different question (see decision_support.py's docstring).
Concretely: slice the hourly weather dataframe to the runner's actual
start-to-finish window and hand it to the existing
`render_hourly_safety_panel` unchanged -- no separate risk model.

TERRAIN NOTE: gradient is computed from the GPX elevation profile where
present (Minetti et al. 2002 cost-of-running-on-gradient equation), but
this module does NOT infer land-cover roughness (urban canyon vs. open
field) from GPX alone -- that needs external land-use data this module
does not fetch. The app's existing terrain-type selector (sidebar) is
applied uniformly to the whole route, a documented simplification.

TIMESTAMPS IN THE GPX ARE IGNORED. Route-planning tools (e.g.
afstandmeten.nl, the source of Dam tot Damloop's own published routes)
often embed arbitrary walking-speed timestamps used only to draw the
track, not real race pace. Distance is computed purely from consecutive
lat/lon via the haversine formula; clock time comes from the pace and
start time the user enters here.
"""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd

_GPX_NS = {"g": "http://www.topografix.com/GPX/1/1"}


def _haversine_km(lat1, lon1, lat2, lon2) -> float:
    R = 6371.0088
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlambda / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


def parse_gpx(path_or_file) -> dict:
    """Parse a GPX file into a route dataframe (lat, lon, ele,
    distance_km) and a list of waypoints (e.g. water posts), each snapped
    to the nearest distance-along-course."""
    tree = ET.parse(path_or_file)
    root = tree.getroot()
    namespaced = root.tag.startswith("{http://www.topografix.com/GPX/1/1}")
    ns = _GPX_NS if namespaced else {}
    tag = (lambda t: f"g:{t}") if namespaced else (lambda t: t)

    trkpts = root.findall(f".//{tag('trkpt')}", ns)
    rows = []
    dist = 0.0
    prev = None
    for pt in trkpts:
        lat, lon = float(pt.attrib["lat"]), float(pt.attrib["lon"])
        ele_el = pt.find(tag("ele"), ns)
        ele = float(ele_el.text) if ele_el is not None else None
        if prev is not None:
            dist += _haversine_km(prev[0], prev[1], lat, lon)
        rows.append({"lat": lat, "lon": lon, "ele": ele, "distance_km": dist})
        prev = (lat, lon)
    route_df = pd.DataFrame(rows)

    waypoints = []
    if len(route_df):
        for wpt in root.findall(f"{tag('wpt')}", ns):
            lat, lon = float(wpt.attrib["lat"]), float(wpt.attrib["lon"])
            name_el = wpt.find(tag("name"), ns)
            name = name_el.text if name_el is not None and name_el.text else "Waypoint"
            d2 = (route_df["lat"] - lat) ** 2 + (route_df["lon"] - lon) ** 2
            nearest_idx = d2.idxmin()
            waypoints.append({
                "name": name, "lat": lat, "lon": lon,
                "distance_km": float(route_df.loc[nearest_idx, "distance_km"]),
            })
        waypoints.sort(key=lambda w: w["distance_km"])

    return {"route": route_df, "waypoints": waypoints}


def route_summary(route_df: pd.DataFrame) -> dict:
    if route_df.empty:
        return {"total_km": 0.0, "has_elevation": False,
                "elevation_gain_m": 0.0, "elevation_loss_m": 0.0}
    total_km = float(route_df["distance_km"].iloc[-1])
    # Treat sub-1m ele range as "no usable elevation data" (GPS/DEM noise
    # floor, and this is exactly the flat-route case: DTD26 reports
    # ele=0.0 throughout).
    has_elevation = (
        route_df["ele"].notna().any()
        and (route_df["ele"].max() - route_df["ele"].min()) > 1.0
    )
    if has_elevation:
        d_ele = route_df["ele"].diff()
        gain = float(d_ele.clip(lower=0).sum())
        loss = float(-d_ele.clip(upper=0).sum())
    else:
        gain = loss = 0.0
    return {
        "total_km": total_km, "has_elevation": has_elevation,
        "elevation_gain_m": gain, "elevation_loss_m": loss,
    }


def _minetti_cost(i: float) -> float:
    """Energy cost of running on a gradient (J/kg/m), relative to level
    ground. Minetti et al. (2002), J Appl Physiol 93:1039-1046. i =
    gradient as rise/run (decimal), fitted over roughly [-0.45, 0.45]."""
    return 155.4 * i**5 - 30.4 * i**4 - 43.3 * i**3 + 46.3 * i**2 + 19.5 * i + 3.6


def gradient_met_profile(route_df: pd.DataFrame) -> np.ndarray:
    """Per-point MET multiplier from the Minetti cost-of-running curve,
    relative to level ground (1.0 = no adjustment). Returns an array of
    1.0s if the route carries no usable elevation data."""
    summary = route_summary(route_df)
    if not summary["has_elevation"] or len(route_df) < 2:
        return np.ones(len(route_df))
    d_ele = route_df["ele"].diff().fillna(0).to_numpy()
    d_dist_m = (route_df["distance_km"].diff().fillna(0) * 1000).to_numpy()
    with np.errstate(divide="ignore", invalid="ignore"):
        grad = np.where(d_dist_m > 1, d_ele / d_dist_m, 0.0)
    grad = np.clip(grad, -0.30, 0.30)
    cost = np.array([_minetti_cost(g) for g in grad])
    return cost / _minetti_cost(0.0)


def pace_schedule(route_df: pd.DataFrame, start_time: pd.Timestamp, pace_min_per_km: float) -> pd.DataFrame:
    """Add elapsed_min and clock_time columns from a constant pace
    assumption. Ignores any <time> values embedded in the GPX itself --
    see module docstring."""
    out = route_df.copy()
    out["elapsed_min"] = out["distance_km"] * pace_min_per_km
    out["clock_time"] = start_time + pd.to_timedelta(out["elapsed_min"], unit="m")
    return out


def _to_epoch_seconds(dt_index_or_series, reference: pd.Timestamp) -> np.ndarray:
    """Seconds relative to a shared reference timestamp.

    Deliberately NOT `.astype("int64")`: pandas may store datetimes at
    nanosecond OR microsecond resolution depending on version and how the
    object was constructed, and the two differ by a factor of 1000. Mixing
    them silently pushes every query outside the interpolation range, so
    np.interp clamps to an endpoint and every point gets the same value --
    a flat line that looks like real (constant) weather rather than a bug.
    Subtracting a common reference and taking total_seconds() is
    resolution-independent.
    """
    delta = dt_index_or_series - reference
    if isinstance(delta, pd.Series):
        return delta.dt.total_seconds().to_numpy()
    return delta.total_seconds().to_numpy()


def interpolate_weather_along_route(route_df_timed: pd.DataFrame, weather_df: pd.DataFrame,
                                     columns=("T_air_urban", "WBGT", "UTCI", "MRT")) -> pd.DataFrame:
    """Linearly interpolate the hourly weather series onto each point's
    clock_time."""
    out = route_df_timed.copy()
    reference = weather_df.index[0]
    idx_numeric = _to_epoch_seconds(weather_df.index, reference)
    query_numeric = _to_epoch_seconds(out["clock_time"], reference)
    for col in columns:
        if col in weather_df.columns:
            out[col] = np.interp(query_numeric, idx_numeric, weather_df[col].to_numpy())
    return out


def route_exposure_chart(route_df_timed: pd.DataFrame, waypoints: list, title: str):
    """Weather-vs-distance chart along the course, with water-post
    markers -- the along-route equivalent of the app's existing
    time-based thermal_chart."""
    import plotly.graph_objects as go

    clock_labels = route_df_timed["clock_time"].dt.strftime("%H:%M")
    fig = go.Figure()
    series = [
        ("T_air_urban", "T_air (urban)", "#f97316", None),
        ("WBGT", "WBGT", "#dc2626", None),
        ("UTCI", "UTCI", "#7c3aed", None),
        ("MRT", "MRT", "#0ea5e9", "dot"),
    ]
    for col, name, colour, dash in series:
        if col not in route_df_timed.columns:
            continue
        line = dict(color=colour)
        if dash:
            line["dash"] = dash
        fig.add_trace(go.Scatter(
            x=route_df_timed["distance_km"], y=route_df_timed[col],
            name=name, line=line, customdata=clock_labels,
            hovertemplate=f"%{{x:.1f}} km (%{{customdata}})<br>{name}: %{{y:.1f}}\u00b0C<extra></extra>",
        ))

    for w in waypoints:
        fig.add_vline(
            x=w["distance_km"], line_dash="dot", line_color="#0284c7",
            annotation_text=w["name"], annotation_position="top",
            annotation_font=dict(size=9, color="#0284c7"),
        )

    fig.update_layout(
        title=dict(text=title, x=0, xanchor="left", y=0.97, yanchor="top"),
        xaxis_title="Distance (km)", yaxis_title="\u00b0C",
        height=420, margin=dict(l=10, r=20, t=70, b=10),
        legend=dict(orientation="h", yanchor="top", y=-0.18, xanchor="left", x=0,
                    font=dict(size=10)),
        hovermode="x unified",
    )
    return fig


def route_map(route_df: pd.DataFrame, waypoints: list, title: str,
              terrain_route_df: pd.DataFrame = None):
    """Map of the course on free OpenStreetMap tiles (no API key needed).

    If `terrain_route_df` is supplied (from
    terrain_lookup.fetch_landcover_along_route), the track is split into
    one trace per land-cover class so the map itself shows where the
    course runs through built-up area, tree cover, open water, etc. --
    the same classification that drives the per-segment WBGT/MRT.
    """
    import plotly.graph_objects as go

    fig = go.Figure()

    # Per-class colours, chosen to read as the land cover they represent.
    class_colours = {
        "Tree cover": "#166534", "Shrubland": "#65a30d", "Grassland": "#84cc16",
        "Cropland": "#ca8a04", "Built-up": "#b91c1c",
        "Bare / sparse vegetation": "#a8a29e", "Snow and ice": "#e0f2fe",
        "Permanent water": "#0284c7", "Herbaceous wetland": "#0891b2",
        "Mangroves": "#047857", "Moss and lichen": "#a3a3a3",
    }

    has_terrain = (
        terrain_route_df is not None
        and "worldcover_label" in terrain_route_df.columns
        and len(terrain_route_df) == len(route_df)
    )

    if has_terrain:
        plot_df = route_df.copy()
        plot_df["worldcover_label"] = terrain_route_df["worldcover_label"].to_numpy()
        # One trace per contiguous run of the same class, so the line stays
        # geographically continuous instead of jumping between segments.
        run_id = (plot_df["worldcover_label"] != plot_df["worldcover_label"].shift()).cumsum()
        seen_labels = set()
        for _, seg in plot_df.groupby(run_id):
            label = seg["worldcover_label"].iloc[0]
            fig.add_trace(go.Scattermap(
                lat=seg["lat"], lon=seg["lon"], mode="lines",
                line=dict(width=5, color=class_colours.get(label, "#6b7280")),
                name=label, legendgroup=label,
                showlegend=label not in seen_labels,
                hovertemplate=f"{label}<extra></extra>",
            ))
            seen_labels.add(label)
    else:
        fig.add_trace(go.Scattermap(
            lat=route_df["lat"], lon=route_df["lon"], mode="lines",
            line=dict(width=5, color="#dc2626"), name="Course",
            hoverinfo="skip",
        ))

    # Start / finish
    fig.add_trace(go.Scattermap(
        lat=[route_df["lat"].iloc[0]], lon=[route_df["lon"].iloc[0]],
        mode="markers", marker=dict(size=14, color="#16a34a"),
        name="Start", hovertemplate="Start<extra></extra>",
    ))
    fig.add_trace(go.Scattermap(
        lat=[route_df["lat"].iloc[-1]], lon=[route_df["lon"].iloc[-1]],
        mode="markers", marker=dict(size=14, color="#111827"),
        name="Finish", hovertemplate="Finish<extra></extra>",
    ))

    if waypoints:
        fig.add_trace(go.Scattermap(
            lat=[w["lat"] for w in waypoints], lon=[w["lon"] for w in waypoints],
            mode="markers", marker=dict(size=11, color="#0284c7"),
            name="Water posts",
            hovertext=[f"{w['name']} \u2014 {w['distance_km']:.1f} km" for w in waypoints],
            hoverinfo="text",
        ))

    center_lat = float(route_df["lat"].mean())
    center_lon = float(route_df["lon"].mean())
    lat_span = float(route_df["lat"].max() - route_df["lat"].min())
    lon_span = float(route_df["lon"].max() - route_df["lon"].min())
    span = max(lat_span, lon_span, 1e-4)
    # Rough zoom fit: each zoom level halves the visible span.
    zoom = float(np.clip(math.log2(360.0 / span) - 1.2, 3, 15))

    fig.update_layout(
        map=dict(style="open-street-map", center=dict(lat=center_lat, lon=center_lon), zoom=zoom),
        title=dict(text=title, x=0, xanchor="left", y=0.97, yanchor="top"),
        height=520, margin=dict(l=0, r=0, t=50, b=0),
        legend=dict(orientation="h", yanchor="top", y=-0.02, xanchor="left", x=0,
                    font=dict(size=10), bgcolor="rgba(255,255,255,0.75)"),
    )
    return fig


def render_race_profile(st, route_df: pd.DataFrame, waypoints: list, weather_df: pd.DataFrame,
                         profile_label: str, met_base: float,
                         start_time: pd.Timestamp, pace_min_per_km: float,
                         render_hourly_safety_panel,
                         terrain_route_df: pd.DataFrame = None) -> None:
    """Render one runner profile's full exposure picture: distance-based
    weather chart, then the existing hour-by-hour guide restricted to
    this profile's actual start-to-finish window.

    `render_hourly_safety_panel` is passed in (from decision_support.py)
    rather than imported, to avoid a circular import and keep this module
    independently testable.

    `terrain_route_df`, if supplied (from fetch_landcover_along_route,
    same row order/length as route_df), makes WBGT and MRT vary by
    real per-segment terrain roughness instead of using one flat
    city-level value for the whole course. UTCI is never terrain-adjusted
    -- see the module docstring in terrain_lookup.py for why.
    """
    summary = route_summary(route_df)
    grad_mult = gradient_met_profile(route_df)
    avg_grad_mult = float(np.mean(grad_mult))
    effective_met = met_base * avg_grad_mult

    timed = pace_schedule(route_df, start_time, pace_min_per_km)
    timed = interpolate_weather_along_route(timed, weather_df)

    terrain_note = ""
    if terrain_route_df is not None and "roughness_z0" in terrain_route_df.columns:
        from terrain_lookup import recompute_wbgt_mrt_for_terrain
        timed["roughness_z0"] = terrain_route_df["roughness_z0"].to_numpy()
        timed = recompute_wbgt_mrt_for_terrain(timed, weather_df)
        terrain_note = (
            " WBGT and MRT below vary by real terrain along the course "
            "(ESA WorldCover); UTCI stays at the single city-level value "
            "(defined at fixed 10m wind by convention, not terrain-adjustable)."
        )

    finish_time = timed["clock_time"].iloc[-1]

    st.markdown(f"### {profile_label}")
    pace_km_h = 60 / pace_min_per_km if pace_min_per_km > 0 else 0
    st.caption(
        f"Pace {pace_min_per_km:.1f} min/km ({pace_km_h:.1f} km/h) \u2192 "
        f"start {start_time.strftime('%H:%M')}, finish "
        f"{finish_time.strftime('%H:%M')} "
        f"({timed['elapsed_min'].iloc[-1] / 60:.1f} h on course). "
        + (
            f"Course has real elevation change ({summary['elevation_gain_m']:.0f} m gain) "
            f"\u2014 metabolic load adjusted by {avg_grad_mult:.2f}\u00d7 on average "
            "(Minetti et al. 2002) to an effective "
            f"{effective_met:.1f} MET."
            if summary["has_elevation"] else
            "Course is flat (no usable elevation data) \u2014 no gradient "
            f"adjustment applied ({met_base:.1f} MET)."
        )
        + terrain_note
    )

    st.plotly_chart(
        route_exposure_chart(timed, waypoints, f"{profile_label} \u2014 conditions along the course"),
        use_container_width=True,
        key=f"gpx_route_chart_{profile_label}",
    )

    # A perfectly constant series over a race lasting an hour or more is
    # almost certainly an interpolation failure (e.g. a datetime-resolution
    # mismatch putting every query outside the weather series' range), not
    # real weather. Say so rather than drawing a confident flat line.
    varying = {c: timed[c].nunique() for c in ("T_air_urban", "WBGT", "UTCI", "MRT")
               if c in timed.columns}
    if varying and all(n <= 1 for n in varying.values()) and timed["elapsed_min"].iloc[-1] > 30:
        st.warning(
            "\u26a0\ufe0f Every value is identical along the whole course, which "
            "is implausible for a race of this length \u2014 the weather series "
            "and the race window may not overlap, or the interpolation "
            "failed. Treat this chart as unreliable and check that the race "
            "date/time falls inside the forecast period."
        )

    st.caption(
        "Distance-based, not time-based: shows what this runner will "
        "actually pass through, given their own pace, at each km \u2014 "
        "dotted markers are water posts from the GPX. Hover for clock time."
    )

    race_weather = weather_df[(weather_df.index >= start_time) & (weather_df.index <= finish_time)]
    if race_weather.empty:
        # Race window shorter than one hourly step: pad by one step each
        # side so the safety panel has at least two points to interpolate.
        race_weather = weather_df[
            (weather_df.index >= start_time - pd.Timedelta(hours=1))
            & (weather_df.index <= finish_time + pd.Timedelta(hours=1))
        ]
    render_hourly_safety_panel(st, race_weather, f"{profile_label} (race window only)", effective_met)
