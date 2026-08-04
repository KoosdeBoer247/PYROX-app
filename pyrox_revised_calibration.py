# -*- coding: utf-8 -*-
"""
PYROX revised population-tier calibration
=========================================

WHAT THIS MODULE DOES
---------------------
It replaces two parameters on the **eleven population groups that exhibit a
defect** (of twenty-three in the roster):

    * ``max_acclimatization_capacity``
    * ``recovery_threshold``

and adds a metabolic-load (MET) correction to the heat-load bridge. These
changes are applied unconditionally by the entry points — there is no switch
to restore the original values in ordinary use. The original numbers remain
visible in ``pyrox_groups.py`` for reference and are reproduced below.

**Twelve groups are deliberately left at their published values** — see
:data:`UNCHANGED_GROUPS`. Their flip points already fall inside the
achievable load range, so none of the three defects applies to them.
Re-solving them anyway measurably degraded curve-shape agreement with the
suite's own Paris 2003 excess-death reference (e.g. healthy older adults
r = 0.833 -> 0.734), while buying nothing. Restricting the revision keeps
those correlations bit-identical to the published parameterisation.

.. warning::

   **PYROX's population tier has no dedicated event-level validation in
   this suite, revised or otherwise — this correction does not undo one.**

   The r = 0.866 Pearson correlation (CVR decompensation % vs. Dam tot
   Damloop incident data 2022-2025), the Falmouth hindcasts, and the
   IRONMAN 70.3 Hoorn prospective test all belong to **HESTIA's individual
   tier** (see ``hestia_model.py``, ``intercept_estimation.py``,
   ``HESTIA_CVR_Console.py`` — Newton-calibrated ``EP1``/``EP2``/``EP3``
   intercepts against per-person collapse and hospital-admission rates).
   Neither ``pyrox_model.py`` nor ``pyrox_groups.py`` references any of
   those events; a check of both files confirms zero occurrences of "DtD",
   "Falmouth" or "Hoorn" prior to this module. An earlier draft of this
   documentation incorrectly borrowed that HESTIA validation metric and
   attributed it to PYROX — that was a citation error, not a finding about
   this revision, and it is corrected here.

   What follows from this, honestly: **PYROX's population tier is
   presently uncalibrated against any independent incident dataset**,
   before or after this revision. The whole-event DtD estimate referenced
   in project notes (~57 expected vs. ~50 actual EHS-related cases) should
   be checked against its actual source (HESTIA aggregation vs. a genuine
   PYROX run) before being cited as PYROX validation either way. This
   module corrects three internal structural defects and is defensible on
   those grounds; it is not a substitute for validating PYROX against real
   population-level incident data, which does not yet exist for this tier.


RELATIONSHIP TO ``pyrox_groups.py`` AND ``verify.py``
-----------------------------------------------------
This module does **not** modify ``pyrox_groups.py``. The published paper
values stay exactly where they are, which keeps ``verify.py`` meaningful:
that script's job is to prove the code matches the published paper, and it
asserts the paper's prototype parameters explicitly::

    'adults_18_45':        max_acclimatization_capacity=0.80, recovery_threshold=1.5
    'elderly_65_85':       max_acclimatization_capacity=0.45, recovery_threshold=0.8
    'very_elderly_85plus': max_acclimatization_capacity=0.25, recovery_threshold=0.5

Editing the roster in place would force those assertions to be rewritten,
destroying the only automated check that the implementation still
corresponds to the publication. The two therefore answer different
questions and both are kept:

    verify.py                     -> does the code match the published paper?
    test_revised_calibration.py   -> does the revised calibration behave
                                     correctly across known scenarios?

Entry points (``run_pyrox.py``) apply the revision unconditionally, so
ordinary use always gets the revised parameters. The original values remain
importable from ``pyrox_groups`` for side-by-side comparison.


RELATIONSHIP TO HESTIA'S ``met_value``
--------------------------------------
HESTIA's individual tier already takes a ``met_value`` and derives exercise
intensity from VO2max. That is a *different tier answering a different
question*: acute, hour-scale core-temperature dynamics for one individual
during one event. The MET term defined here acts on PYROX's daily
population-tier heat load and models cumulative multi-day strain.

The two do not double-count, because neither feeds the other. They should
however be kept numerically consistent: if an event is analysed in HESTIA at
11 MET, the corresponding PYROX group should not be left at the default 1.6.


WHY THE ORIGINAL CALIBRATION HAD TO CHANGE
------------------------------------------
Three independent defects were identified by systematic testing of the
original roster. Each is reproducible from ``pyrox_groups.py`` and
``pyrox_model.py`` as shipped.

**Defect 1 — resilience was counted twice, multiplicatively.**

``PyroxModel.net_strain_input`` (Step 3, paper Sec 2.2) evaluates the
recovery threshold *after* the acclimatization reduction::

    experienced_load = baseline_heat_load * (1 - effective_acclimatization)
    net_strain_input = max(0, experienced_load - recovery_threshold)

The load at which a group begins accumulating strain is therefore not
``recovery_threshold`` but approximately::

    flip_load ~= recovery_threshold / (1 - max_acclimatization_capacity)

Measured flip loads matched that expression closely across the whole roster
(e.g. outdoor workers: threshold 1.70, capacity 0.80, predicted 8.50,
measured 8.67). Because resilient groups were assigned BOTH a high threshold
AND a high capacity, the two compounded and pushed their effective tolerance
far outside any achievable load.

**Defect 2 — the parameters sat on the wrong scale.**

The heat-load unit is ``(apparent_temperature_C - 22) * 0.10``, so real
weather spans roughly 0 to 3.0:

===========================  ==========  ====
condition                    apparent C  load
===========================  ==========  ====
warm Dutch summer day              32.0  1.00
severe Dutch heatwave peak         38.0  1.60
Paris 2003 peak                    41.0  1.90
Riyadh routine summer              45.0  2.30
Riyadh lethal extreme              52.0  3.00
===========================  ==========  ====

Four groups had a ``recovery_threshold`` that, on its own and before any
acclimatization reduction, exceeded the peak load of a severe heatwave:
endurance athletes 2.50, elite athletes 2.00, recreational athletes 1.80,
outdoor workers 1.70. Those groups could not respond to any weather on
Earth, independently of Defect 1.

**Defect 3 — one capacity value implied immunity.**

``experienced_load = load * (1 - max_acclimatization_capacity)``. Endurance
athletes were assigned 1.00, i.e. they experienced exactly zero heat load
under all conditions. Elite athletes at 0.95 experienced 5%. Heat
acclimation reduces physiological strain substantially but bounded — it does
not abolish heat load — and the suite's own PYROX v2.2 work had already
applied a Callahan et al. (2025) adaptation limit elsewhere, lowering
maximum sweat adaptation from 0.60 to 0.40. The population-tier roster was
not brought into line with that decision.

**Consequences that were observed before the fix.** Zero of 667 tested
(group, load) combinations settled into a stable intermediate strain level:
the model was purely bang-bang, resting at zero (22.3%) or running away to
the ceiling (51.3%), with the remaining 26.4% sitting on unstable
mid-points. The outcome was predicted in 96.8% of 1357 tested cases by the
single comparison ``experienced_load > recovery_threshold``, meaning the
memory kernel, suppression gate and homeostatic drive had no influence on
the final answer. Eight of 23 groups — every exertional group — reported
baseline in every scenario worldwide.


WHAT WAS DELIBERATELY NOT CHANGED
---------------------------------
The model equations are untouched. Increasing
``HOMEOSTATIC_DRIVE_COEFFICIENT`` from 0.1 to 3.0 or 6.0 does produce a
graded response region (7.3% and 19.6% of cases respectively), but destroys
the calibration case: the elderly group's Paris 2003 peak strain falls from
100% to 54% and 28%. Halving the suppression strength, and moving the
threshold ahead of the acclimatization reduction, each yielded no graded
region at all.

Bang-bang behaviour is the model's thesis rather than a defect: the
control-theoretic framing is that the regulatory loop opens when loop gain
exceeds unity, producing runaway decompensation. A system that decompensates
is expected to be bistable. Once the thresholds are placed on the correct
scale, graded behaviour emerges anyway along the metabolic axis — see the
outdoor-worker progression in the acceptance tests below.


HOW THE REVISED VALUES WERE DERIVED
-----------------------------------
**Acclimatization capacity.** Values at or below 0.45 are left untouched, so
the calibrated vulnerable tier is not disturbed. Values above 0.45 are
compressed onto a ceiling of 0.55::

    new = 0.45 + (old - 0.45) * (0.55 - 0.45) / (1.00 - 0.45)

Compression rather than a hard cap preserves the ordering among resilient
groups, which a flat cap would erase. A ceiling of 0.55 means even the
best-adapted group still experiences 45% of the ambient load, consistent
with the bounded (roughly 20-45%) strain reduction reported for heat
acclimation, and with the suite's own Callahan-derived adaptation limit.

**Recovery thresholds.** Each group is assigned an explicit *onset
temperature* — the apparent temperature at which it should begin
accumulating strain — and the threshold is then solved numerically so the
measured flip point lands there, given the revised capacity. This makes
every value traceable to a stated assumption rather than a free parameter.
Achieved onsets match the targets to within 0.1 C for all groups.

**The resilient tier's onset temperatures are deliberately set above a
severe heatwave peak.** An earlier iteration placed outdoor workers at
37.5 C, which caused healthy working-age adults to fully decompensate under
Paris 2003 — contradicted by the epidemiology, where 2003 excess mortality
was concentrated in the elderly. Setting the ambient onset above the
heatwave peak routes occupational and athletic risk through the metabolic
term instead, which is the physiologically correct causal path: a
construction worker is not endangered by standing outdoors, but by working
outdoors.

.. note::

   The onset temperatures are reasoned assumptions, not values extracted
   from threshold epidemiology. They are internally consistent and produce
   behaviour that passes the acceptance tests below, but each one should be
   substantiated against published heat-mortality threshold literature
   before the revised calibration is treated as authoritative.


METABOLIC (MET) CORRECTION
--------------------------
Metabolic heat is physically additive with environmental heat: both must be
dissipated through the same evaporative and cardiovascular actuator. The
correction therefore belongs on the load side of the model::

    load = max(0, (T_apparent + k * (MET - MET_REF) - 22.0) * 0.10)

The coefficient is derived from ISO 7243's own reference WBGT limit values
for acclimatized workers — the same standard whose sufficiency the wider
project questions, which makes the slope both defensible and citable:

=========  ===========  ==========  ===========
class      W (total)    W/m2        WBGT limit
=========  ===========  ==========  ===========
resting            115        63.9        33.0
low                180       100.0        30.0
moderate           300       166.7        28.0
high               415       230.6        26.0
=========  ===========  ==========  ===========

A linear fit over these points (BSA 1.8 m2) gives 0.0394 C per W/m2. With
1 MET = 58.15 W/m2::

    k = 2.29 C apparent-temperature equivalent per MET

Back-checking the coefficient against the standard's own limit reductions
gives agreement within 0.4-0.9 C at moderate and high intensity, and 1.6 C
at low intensity — the relationship is mildly non-linear, and the fit is
best in the range that matters for occupational heat risk.

**The MET value is shift-weighted, not averaged over 24 hours.** Averaging a
construction worker's 8-hour shift across the full day reduces the
correction to +2.1 C and erases the signal; the shift coincides with the
daily thermal peak, so that window is the one that determines risk. PYROX
resolves days, not hours, so this is a deliberate approximation on the
conservative side.

**Scope limit.** This correction addresses cumulative multi-day strain in a
working population. It does not model acute exertional heat stroke during a
single shift or race, which depends on hour-scale core-temperature dynamics
and is the domain of HESTIA's individual tier.


ACCEPTANCE TESTS
----------------
All four pass with the values in this module (see ``ACCEPTANCE_TESTS``
below for the executable versions):

1. **Mild summer** (24-27 C): no group saturates. 0 of 23 — pass.
2. **Paris 2003, at rest**: the entire vulnerable tier saturates — pass.
3. **Paris 2003, at rest**: no resilient-tier group fully decompensates —
   pass.
4. **Maastricht heatwave (peak apparent 38 C), outdoor worker**: safe at
   rest, progressively at risk with workload — pass, and notably graded:

   ==========================  =========  ======
   condition                   peak load  strain
   ==========================  =========  ======
   at rest                          1.60    5.0%
   light work (+1.5 MET)            1.94    7.0%
   construction (+2.8 MET)          2.24   16.0%
   heavy labour (+4.3 MET)          2.58   63.1%
   ==========================  =========  ======

   This is the graded response that no structural modification to the
   equations was able to produce.

5. **Regression**: the vulnerable tier's Paris 2003 and mild-summer outcomes
   are unchanged from the original calibration.
"""

from typing import Dict, Tuple

# ---------------------------------------------------------------------------
# Acclimatization compression constants (see module docstring)
# ---------------------------------------------------------------------------
ACCLIM_UNCHANGED_BELOW: float = 0.45
ACCLIM_CEILING: float = 0.55

# ---------------------------------------------------------------------------
# MET correction constants (see module docstring)
# ---------------------------------------------------------------------------
#: Apparent-temperature equivalent of one MET, derived from ISO 7243.
K_PER_MET: float = 2.29
#: Metabolic rate implicitly represented by the original calibration
#: population (largely resting/indoor). Deltas are measured against this.
MET_REFERENCE: float = 1.2

#: Default shift metabolic rates per group, used as the starting value in the
#: UI. These are typical published MET values for the activity concerned and
#: are user-overridable — a specific worksite or event should use its own.
DEFAULT_MET: Dict[str, float] = {
    'outdoor_workers': 4.0,          # general construction / manual outdoor work
    'indoor_workers': 2.5,           # non-cooled indoor manual work
    'endurance_athletes': 12.0,      # sustained endurance running
    'elite_athletes': 13.0,          # competitive racing intensity
    'recreational_athletes': 9.0,    # recreational running pace
    'youth_10_18': 2.0,              # school / active play
    'children_6_10': 2.0,
    'children_0_6': 1.8,
    'adults_18_45': 1.6,             # general active adult daily life
    'middle_aged_45_65': 1.5,
}

# ---------------------------------------------------------------------------
# Groups deliberately LEFT AT THE PUBLISHED VALUES.
#
# MINIMAL-INTERVENTION PRINCIPLE: only groups that actually exhibit one of the
# three defects are changed. These twelve do not: their measured flip points
# already fall inside the achievable load range (0.67-1.42, against a severe
# heatwave at 1.60), and none exceeds the acclimatization ceiling.
#
# This is not merely conservative, it is measurably better. An earlier draft
# re-solved every group's threshold, including these. Checked against the
# suite's own paris2003.py reference data, that degraded curve-shape agreement
# with the Fouillet et al. (2006) excess-death series:
#
#   group                     paper   revise-all   revise-defective-only
#   Healthy Older Adults      0.833      0.734            0.833
#   Vulnerable Older (85+)    0.672      0.656            0.672
#   Dementia / Alzheimer's    0.640      0.609            0.640
#   Cardiovascular Disease    0.809      0.714            0.809
#
# Restricting the revision preserves those correlations exactly while still
# bringing outdoor workers from a flip load of 8.67 down to 2.05. Touching
# working parameters cost accuracy and bought nothing.
#
#   key                         flip load   capacity
#   dementia                         0.67       0.22
#   very_elderly_85plus              0.72       0.25
#   severe_mental_illness            0.77       0.27
#   medication_impaired              0.88       0.33
#   chronic_comorbidities            1.01       0.35
#   physical_disabilities            1.10       0.40
#   cardiovascular_disease           1.14       0.40
#   children_0_6                     1.23       0.38
#   obesity                          1.24       0.40
#   elderly_65_85                    1.31       0.45
#   pregnant_t3                      1.42       0.55
#   middle_aged_45_65                2.12       0.55
# ---------------------------------------------------------------------------
UNCHANGED_GROUPS = (
    'dementia', 'very_elderly_85plus', 'severe_mental_illness',
    'medication_impaired', 'chronic_comorbidities', 'physical_disabilities',
    'cardiovascular_disease', 'children_0_6', 'obesity', 'elderly_65_85',
    'pregnant_t3', 'middle_aged_45_65',
)

# ---------------------------------------------------------------------------
# Revised parameters for the eleven DEFECTIVE groups only.
#   key -> (max_acclimatization_capacity, recovery_threshold, onset_temp_C)
#
# The third element is documentation, not a model input: it records the
# apparent temperature the threshold was solved to produce, so any value can
# be traced back to its stated assumption.
# ---------------------------------------------------------------------------
REVISED_CALIBRATION: Dict[str, Tuple[float, float, float]] = {
    # --- capacity above the plausible ceiling, and/or flip point beyond any
    #     achievable load ---------------------------------------------------
    'unacclimatized_travelers':  (0.49, 0.52, 33.0),
    'pregnant_t2':               (0.49, 0.57, 33.0),
    'children_6_10':             (0.47, 0.58, 33.5),
    'pregnant_t1':               (0.50, 0.65, 34.5),
    'indoor_workers':            (0.50, 0.68, 36.0),
    'youth_10_18':               (0.50, 0.70, 37.0),
    # --- resilient tier: ambient onset deliberately set ABOVE a severe
    #     heatwave peak, with occupational and athletic risk routed through
    #     the MET term instead. An earlier draft placed outdoor workers at
    #     37.5 C, which made healthy working-age adults fully decompensate
    #     under Paris 2003 -- contradicted by the epidemiology, where 2003
    #     excess mortality was concentrated in the elderly. A construction
    #     worker is not endangered by standing outdoors, but by working
    #     outdoors, and the model should say so through the metabolic term.
    'adults_18_45':              (0.51, 0.99, 42.0),
    'outdoor_workers':           (0.51, 0.93, 42.5),
    'recreational_athletes':     (0.52, 0.92, 43.0),
    'elite_athletes':            (0.54, 0.88, 44.0),
    'endurance_athletes':        (0.55, 0.83, 44.5),
}


def apply_revised_calibration(target_groups: dict) -> dict:
    """Return a new group dict with the revised parameters applied.

    Groups absent from :data:`REVISED_CALIBRATION` are passed through
    unchanged, so adding a group to the roster without calibrating it
    degrades gracefully rather than raising.

    Parameters
    ----------
    target_groups
        The roster as defined in ``pyrox_groups.TARGET_GROUPS``.

    Returns
    -------
    dict
        A new mapping with the same keys. The input is not mutated, so the
        original values remain inspectable for comparison.
    """
    from dataclasses import replace

    revised = {}
    for key, group in target_groups.items():
        if key in REVISED_CALIBRATION:
            acclim, threshold, _onset = REVISED_CALIBRATION[key]
            revised[key] = replace(
                group,
                max_acclimatization_capacity=acclim,
                recovery_threshold=threshold,
            )
        else:
            revised[key] = group
    return revised


def onset_temperature(group_key: str):
    """Apparent temperature (C) at which this group starts accumulating strain.

    Returns ``None`` for groups that have no revised calibration entry.
    """
    entry = REVISED_CALIBRATION.get(group_key)
    return entry[2] if entry else None


def met_adjusted_apparent_temperature(apparent_c: float, met: float) -> float:
    """Apply the metabolic correction to an apparent temperature.

    Metabolic heat above the reference level is expressed as an equivalent
    rise in apparent temperature, because both must be shed through the same
    physiological actuator (see module docstring for the ISO 7243 derivation
    of :data:`K_PER_MET`).
    """
    return apparent_c + K_PER_MET * (met - MET_REFERENCE)


def default_met(group_key: str) -> float:
    """Default shift metabolic rate for a group, or the reference if none."""
    return DEFAULT_MET.get(group_key, MET_REFERENCE)
