# -*- coding: utf-8 -*-
"""
Decision-support layer for the PYROX/Thermopoulos app
======================================================
Adds two things the app did not previously answer directly for someone who
has to make an operational call:

1. `render_key_concepts_explainer()` — a short, plain-language explanation of
   `final_risk` vs. cumulative strain, so a reader who is not a modeller
   knows which number answers which question.

2. An hour-by-hour work/rest safety guide for a chosen metabolic rate,
   answering "which hours are safe, which are dangerous, and what should
   I do about it". PYROX's own model CANNOT answer this: it runs at DAILY
   resolution (paper Sec 2.2) and answers a different question (is the
   regulatory loop opening over days). This module instead classifies each
   hour of the already-computed hourly WBGT series (Thermopoulos_Data_Engine)
   against standard occupational WBGT action limits (ISO 7243 / ACGIH
   lineage), by metabolic rate.

RELATIONSHIP TO THE FNV BOUW CORRESPONDENCE
--------------------------------------------
This panel uses WBGT as its sole input, which is exactly the metric whose
limitations (narrow validation population, no multi-day cumulative load,
coarse MET banding) have already been raised with Khalid Azougagh. That
critique still applies here and is not undone by this panel — which is why
it is presented as a same-shift screening layer only, explicitly alongside
(not instead of) PYROX's cumulative-strain view, which is the one that
captures multi-day load. The in-app caption says this directly.

The WBGT limit table below is the standard four-regimen occupational
reference (100/75/50/25% work per hour), interpolated across metabolic
rate. It is a widely used screening reference, not a PYROX output, and
carries the same evidentiary status as WBGT itself: a coarse, broadly
validated screening tool, not a substitute for individual clinical
judgement or for PYROX/HESTIA's own calibrated outputs.
"""

from __future__ import annotations

from itertools import groupby
from functools import lru_cache

import numpy as np
import pandas as pd

from pyrox_model import PyroxModel
from pyrox_groups import TARGET_GROUPS as _ORIGINAL_GROUPS
from pyrox_revised_calibration import (
    apply_revised_calibration, met_adjusted_apparent_temperature, MET_REFERENCE,
)
from thermopoulos_loader import HEAT_LOAD_REFERENCE_TEMP, HEAT_LOAD_PER_DEGREE


# =============================================================================
# 0. Human-interpretable framing for the dimensionless `final_risk` number
# =============================================================================
# `final_risk` (paper Sec 2.2 Step 5) is deliberately dimensionless -- it is
# an acute load amplified by accumulated strain, not a probability or a
# physical unit. Read as a bare number ("1.69"), it means nothing to anyone
# who isn't the model's author. Rather than inventing new arbitrary bands,
# this reuses the two reference apparent-temperature scenarios ALREADY
# defined and tested in test_revised_calibration.py -- a mild, unremarkable
# summer week and the Paris August 2003 heatwave (the suite's own severity
# benchmark) -- and expresses any computed peak_risk as a multiple of what
# the SAME group would show under those same two references. This keeps the
# comparison honest: it says "how does this compare to a known scenario for
# this specific group", not "this is a calibrated probability of harm",
# which PYROX's population tier cannot yet claim (no event-level validation
# -- see the calibration warning in app.py).
_GROUPS = apply_revised_calibration(_ORIGINAL_GROUPS)

# Apparent-temperature sequences (°C), identical to test_revised_calibration.py
_MILD_SUMMER = (24, 25, 26, 25, 24, 26, 27, 26, 25, 24, 25, 26)
_PARIS_2003 = (30, 32, 35, 37, 39, 40, 41, 40, 38, 35, 32, 30)


def _loads(temps, met: float):
    return [
        max(0.0, (met_adjusted_apparent_temperature(t, met) - HEAT_LOAD_REFERENCE_TEMP)
            * HEAT_LOAD_PER_DEGREE)
        for t in temps
    ]


def _peak_risk_for(group_key: str, temps: tuple, met: float) -> float:
    model = PyroxModel(_GROUPS[group_key])
    res = model.simulate(_loads(temps, met))
    return float(max(res["final_risk"]))


@lru_cache(maxsize=None)
def reference_anchors(group_key: str, met: float = MET_REFERENCE) -> dict:
    """This group's own peak_risk under a mild summer week and under Paris
    2003 -- the two fixed comparison points."""
    return {
        "mild_summer": _peak_risk_for(group_key, _MILD_SUMMER, met),
        "paris_2003": _peak_risk_for(group_key, _PARIS_2003, met),
    }


def relative_risk_text(peak_risk: float, group_key: str, met: float = MET_REFERENCE) -> tuple:
    """Return (vs_mild_summer_multiple, vs_paris_2003_percent) for display."""
    a = reference_anchors(group_key, round(met, 1))
    mild_mult = peak_risk / a["mild_summer"] if a["mild_summer"] > 1e-6 else None
    paris_pct = 100 * peak_risk / a["paris_2003"] if a["paris_2003"] > 1e-6 else None
    return mild_mult, paris_pct


# =============================================================================
# 1. Plain-language explainer: risk vs. strain
# =============================================================================

def render_key_concepts_explainer(st) -> None:
    """Render a short explainer distinguishing final_risk from cumulative
    strain. Call once, near the top of the PYROX section, before the charts.
    `st` is passed in rather than imported to avoid a hard Streamlit
    dependency in this module.
    """
    with st.expander(
        "\U0001f4d8 What do 'risk level' and 'strain' mean? (read this first)",
        expanded=True,
    ):
        st.markdown(
            "This page shows **two different numbers** that answer two "
            "different questions. Mixing them up is the most common way to "
            "misread this page.\n"
        )
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(
                "**\U0001f4c8 Risk level** (`final_risk`)\n\n"
                "*How hard is the body working right now, given today's "
                "heat and workload?*\n\n"
                "- Goes up and down smoothly, hour to hour and day to day.\n"
                "- Good for **ranking**: is Tuesday worse than Wednesday? "
                "Is this group more exposed than that one?\n"
                "- Has no fixed 'safe' ceiling — a higher number is always "
                "worse, but there is no single cutoff where it 'becomes "
                "dangerous'."
            )
        with col2:
            st.markdown(
                "**\u26a0\ufe0f Cumulative strain**\n\n"
                "*Has the body's cooling system started to lose the fight?*\n\n"
                "- Shown as **% of a critical threshold** — a number with "
                "real meaning: 0% = fully coping, 100% = the regulatory "
                "system has lost control (runaway decompensation).\n"
                "- Builds up over **multiple days** if heat and recovery "
                "don't balance; a bad night's sleep or no cool-down period "
                "makes it worse.\n"
                "- This is the number to alarm on: **50% = caution, "
                "75% = danger, 90% = emergency.**"
            )
        st.caption(
            "Rule of thumb: use **risk level** to decide *which day or "
            "group* needs attention first. Use **cumulative strain** to "
            "decide *whether today is actually dangerous*, and the hourly "
            "guide below to decide *which hours*. "
            "Neither number is a validated probability of a medical event "
            "for this population tier (see the calibration warning above) "
            "— treat both as a graded warning signal, not a percentage "
            "chance of harm."
        )


# =============================================================================
# 2. Hourly WBGT-based work/rest safety guide
# =============================================================================

# MET anchors for the standard ISO 7243 / ACGIH workload categories:
# resting, light, moderate, heavy, very heavy.
_MET_ANCHORS = np.array([1.2, 2.5, 4.0, 6.0, 8.0])

# WBGT action limits (°C), acclimatized workers, by work:rest regimen.
# Columns follow _MET_ANCHORS. This is the standard four-regimen
# occupational reference table (ISO 7243 / ACGIH TLV lineage).
_LIMITS = {
    100: np.array([33.0, 30.0, 28.0, 25.5, 23.0]),
    75: np.array([33.5, 30.6, 29.0, 25.9, 24.0]),
    50: np.array([34.0, 31.4, 30.4, 27.9, 26.0]),
    25: np.array([34.5, 32.2, 31.8, 30.0, 28.0]),
}

_STATUS_STYLE = {
    "safe": dict(colour="#16a34a", label="Safe — continuous work"),
    "caution_75": dict(colour="#eab308", label="Caution — work 45 min / rest 15 min per hour"),
    "caution_50": dict(colour="#f97316", label="Caution — work 30 min / rest 30 min per hour"),
    "danger": dict(colour="#dc2626", label="Danger — work 15 min / rest 45 min per hour"),
    "emergency": dict(colour="#7f1d1d", label="Emergency — avoid outdoor exertion"),
    "unknown": dict(colour="#9ca3af", label="No data"),
}


def wbgt_limit(met: float, work_pct: int) -> float:
    """Interpolated WBGT action limit (°C) for a given metabolic rate and
    work:rest regimen (100/75/50/25% work per hour)."""
    return float(np.interp(met, _MET_ANCHORS, _LIMITS[work_pct]))


def classify_hour(wbgt: float, met: float) -> dict:
    """Classify a single hour's WBGT reading for a worker at `met`."""
    if pd.isna(wbgt):
        return {"status": "unknown", **_STATUS_STYLE["unknown"]}
    if wbgt <= wbgt_limit(met, 100):
        return {"status": "safe", **_STATUS_STYLE["safe"]}
    if wbgt <= wbgt_limit(met, 75):
        return {"status": "caution_75", **_STATUS_STYLE["caution_75"]}
    if wbgt <= wbgt_limit(met, 50):
        return {"status": "caution_50", **_STATUS_STYLE["caution_50"]}
    if wbgt <= wbgt_limit(met, 25):
        return {"status": "danger", **_STATUS_STYLE["danger"]}
    return {"status": "emergency", **_STATUS_STYLE["emergency"]}


def hourly_schedule(wbgt_series: pd.Series, met: float, utci_series: pd.Series = None) -> pd.DataFrame:
    """Build an hour-by-hour safety classification from an hourly WBGT
    series (index = timestamps, as produced by Thermopoulos_Data_Engine).

    If `utci_series` is supplied (same index), each hour is cross-checked
    against UTCI. WBGT's own formula -- 0.7x wet-bulb (humidity) + 0.2x
    globe (radiant) + 0.1x dry-bulb -- gives radiant/solar load only a 20%
    weight. On a moderate-humidity day with a large radiant excess (full
    sun, high MRT), WBGT can stay well under its action limit while UTCI --
    which weights radiation more fully -- already shows strong heat stress.
    That is not a bug in either calculation; it is WBGT's known,
    documented insensitivity to radiant load relative to humidity, the
    same limitation already raised with FNV Bouw. Flagging the divergence
    here is what makes it visible instead of silently passing as 'safe'.
    """
    rows = []
    for ts, wbgt in wbgt_series.items():
        c = classify_hour(wbgt, met)
        utci_val = utci_series.get(ts) if utci_series is not None else None
        c["UTCI"] = utci_val
        c["utci_diverges"] = bool(
            utci_val is not None and not pd.isna(utci_val)
            and c["status"] in ("safe", "caution_75")
            and utci_val >= 32.0  # UTCI "strong heat stress" threshold
        )
        rows.append({"time": ts, "WBGT": wbgt, **c})
    return pd.DataFrame(rows)


def _consecutive_windows(schedule: pd.DataFrame, statuses: set[str]) -> list[tuple]:
    """Collapse hours whose status is in `statuses` into contiguous
    (start, end) windows."""
    windows = []
    in_window = schedule["status"].isin(statuses).tolist()
    times = schedule["time"].tolist()
    for is_in, group in groupby(zip(times, in_window), key=lambda x: x[1]):
        group = list(group)
        if is_in:
            start = group[0][0]
            end = group[-1][0] + pd.Timedelta(hours=1)
            windows.append((start, end))
    return windows


def summarize_day(schedule: pd.DataFrame, day) -> str:
    """Plain-language, one-paragraph summary of a single calendar day."""
    day_df = schedule[schedule["time"].dt.date == day].reset_index(drop=True)
    if day_df.empty:
        return "No data for this day."

    worst = day_df["status"].map(
        lambda s: {"safe": 0, "caution_75": 1, "caution_50": 2,
                   "danger": 3, "emergency": 4, "unknown": -1}[s]
    ).max()

    def fmt_windows(statuses, verb):
        wins = _consecutive_windows(day_df, statuses)
        if not wins:
            return None
        parts = []
        for s, e in wins:
            if (e - s) >= pd.Timedelta(hours=23):
                parts.append("all day")
            else:
                parts.append(f"{s.strftime('%H:%M')}\u2013{e.strftime('%H:%M')}")
        return f"{verb} {', '.join(parts)}"

    pieces = []
    safe_txt = fmt_windows({"safe"}, "Safe")
    caution_txt = fmt_windows({"caution_75", "caution_50"}, "Caution")
    danger_txt = fmt_windows({"danger", "emergency"}, "Danger")
    for txt in (safe_txt, caution_txt, danger_txt):
        if txt:
            pieces.append(txt)

    if worst <= 0:
        verdict = "Safe to work normal hours."
    elif worst in (1, 2):
        verdict = "Plan extra breaks during the caution window(s)."
    else:
        verdict = "Reschedule heavy or outdoor tasks out of the danger window(s), if possible."

    summary = " \u00b7 ".join(pieces) + f". {verdict}"

    if "utci_diverges" in day_df.columns and day_df["utci_diverges"].any():
        n = int(day_df["utci_diverges"].sum())
        diverging_hours = day_df.loc[day_df["utci_diverges"], "time"].dt.strftime("%H:%M").tolist()
        summary += (
            f" \u26a0\ufe0f **{n} of these hour(s) ({', '.join(diverging_hours)}) are "
            "WBGT-'safe' but UTCI \u226532\u00b0C (strong heat stress)** \u2014 likely "
            "high direct-sun/radiant load that WBGT's formula under-weights. "
            "Treat as caution, especially for unshaded work."
        )

    return summary


def render_hourly_safety_panel(st, df: pd.DataFrame, group_label: str, met: float) -> None:
    """Render the full hour-by-hour safety guide for one group/MET.

    `df` must be an hourly weather dataframe with WBGT and (ideally) UTCI
    columns and a datetime index (forecast_df or hindcast_df from the
    Thermopoulos engine — the same objects already used for
    `thermal_chart`). When UTCI is present, hours where WBGT reads "safe"
    but UTCI already shows strong heat stress are flagged — see
    `hourly_schedule` for why that divergence happens and matters.
    """
    import plotly.graph_objects as go

    if "WBGT" not in df.columns:
        st.info("No WBGT data available for the hourly safety guide.")
        return

    utci_series = df["UTCI"] if "UTCI" in df.columns else None
    schedule = hourly_schedule(df["WBGT"], met, utci_series)
    has_divergence = "utci_diverges" in schedule.columns and schedule["utci_diverges"].any()

    st.markdown(
        f"**\u23f0 Which hours carry elevated heat exposure for {group_label}** "
        f"(at {met:.1f} MET, WBGT-based screening)"
    )
    st.caption(
        "This uses the standard ISO 7243 / ACGIH WBGT action limits, "
        "applied hour by hour — a same-day screening layer, shown "
        "alongside (not instead of) PYROX's cumulative-strain view above. "
        "For groups with a physical workload, treat the labels below as "
        "work/rest guidance; for others, as guidance on how much of the "
        "hour is safe to spend active and exposed versus resting/sheltered. "
        "WBGT has known limitations as a sole metric (validated on a "
        "narrow population, no multi-day load, coarse workload bands) — "
        "treat this as a quick screening check, not a precise prediction. "
        "Hours outlined in black are cross-checked against UTCI: WBGT "
        "weights radiant/solar load at only 20% (vs. 70% for humidity), so "
        "it can call an hour 'safe' during high direct-sun exposure that "
        "UTCI already flags as strong heat stress — outlined hours are "
        "where that happens."
    )

    # Colour-coded hourly bar. Diverging hours get a dark outline so the
    # WBGT-only "safe" colour doesn't silently hide the UTCI disagreement.
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=schedule["time"], y=[1] * len(schedule),
        marker=dict(
            color=schedule["colour"],
            line=dict(
                color=["#111827" if d else "rgba(0,0,0,0)" for d in schedule.get("utci_diverges", [False] * len(schedule))],
                width=[2.5 if d else 0 for d in schedule.get("utci_diverges", [False] * len(schedule))],
            ),
        ),
        hovertext=[
            f"{t.strftime('%a %d %b, %H:%M')}<br>WBGT {w:.1f}\u00b0C<br>{lbl}"
            + (f"<br>UTCI {u:.1f}\u00b0C \u2014 \u26a0\ufe0f strong heat stress not reflected in WBGT" if d else "")
            for t, w, lbl, d, u in zip(
                schedule["time"], schedule["WBGT"], schedule["label"],
                schedule.get("utci_diverges", [False] * len(schedule)),
                schedule.get("UTCI", [None] * len(schedule)),
            )
        ],
        hoverinfo="text",
        showlegend=False,
    ))
    fig.update_yaxes(visible=False, range=[0, 1])
    fig.update_layout(
        height=110,
        margin=dict(l=10, r=10, t=10, b=30),
        bargap=0,
    )
    st.plotly_chart(fig, use_container_width=True, key=f"hourly_safety_{group_label}")

    legend_html = "  ".join(
        "<span style='color:{colour}'>\u25cf</span> {label}".format(**v)
        for k, v in _STATUS_STYLE.items() if k != "unknown"
    )
    if has_divergence:
        legend_html += "  \u25a1 outlined = WBGT/UTCI disagree (see caption)"
    st.markdown(legend_html, unsafe_allow_html=True)

    # Per-day plain-language summary
    for day in sorted(schedule["time"].dt.date.unique()):
        st.markdown(f"**{pd.Timestamp(day).strftime('%A %d %b')}:** " + summarize_day(schedule, day))
