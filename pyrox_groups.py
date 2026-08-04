# -*- coding: utf-8 -*-
"""
PYROX — population group parameters
================================================================================
The 23 population groups, each expressed in the readable parameter names of
pyrox_model.GroupParameters.

--------------------------------------------------------------------------------
EVIDENCE STATUS — read this before trusting any number
--------------------------------------------------------------------------------
Only THREE groups are specified by the paper (Section 2.3). They are labelled
evidence_status='paper'. The paper itself states their values are "based on
physiological plausibility and literature estimates", NOT direct measurements,
so even these are plausibility estimates, not measured constants.

The other 20 groups are labelled evidence_status='extrapolated'. Their ORDERING
(athletes most resilient, dementia least) is grounded in the heat-vulnerability
literature, but the specific numbers are structured interpolations along the
resilient -> vulnerable axis. They are illustrative, not validated. Do not cite
them as empirical values.

--------------------------------------------------------------------------------
THE CORRECTION TO THE PUBLISHED TABLE
--------------------------------------------------------------------------------
The paper's Section 2.3 table lists, for young / older / vulnerable:

    max_acclimatization_capacity : 0.25 / 0.45 / 0.80   (ASCENDING with frailty)
    recovery_threshold           : 0.5  / 0.8  / 1.5     (ASCENDING with frailty)
    base_recovery_rate           : 0.30 / 0.18 / 0.12    (DESCENDING with frailty)

For the model to be coherent, every PROTECTIVE parameter must move the same way
along the resilient -> vulnerable axis (a resilient group should be protected on
all fronts at once). base_recovery_rate is already correct: resilient groups
recover fastest. But max_acclimatization_capacity and recovery_threshold both
ASCEND with frailty, i.e. they make the FRAIL group the better-protected one —
which contradicts both physiology and the paper's own Section 4 narrative
("young develops strong acclimatization ... vulnerable shows minimal
acclimatization development").

The fact that base_recovery_rate runs the OPPOSITE way to the other two
protective columns is itself the tell: the two ascending columns were entered in
reversed order. We therefore reverse max_acclimatization_capacity and
recovery_threshold (swap the young and vulnerable entries), leaving
base_recovery_rate and all other published values untouched:

    CORRECTED:
    max_acclimatization_capacity : 0.80 / 0.45 / 0.25
    recovery_threshold           : 1.5  / 0.8  / 0.5
    base_recovery_rate           : 0.30 / 0.18 / 0.12   (unchanged)

After this single correction, all three protective parameters descend together
with frailty, and the Section 4 trajectory ordering (vulnerable strains first,
young stabilises safely) reproduces. Each corrected group is flagged below with
`# CORRECTED`.

--------------------------------------------------------------------------------
EXPOSURE MEMORY KERNEL SHAPES
--------------------------------------------------------------------------------
The paper gives three example kernel shapes. A STEEP kernel weights the most
recent days heavily (fast adaptation, fit physiology); a FLAT kernel spreads
weight evenly and low (slow, weak adaptation). The three prototypes use the
paper's exact shapes; other groups use shapes interpolated on the same axis.
"""

from __future__ import annotations

from typing import Dict

import numpy as np

from pyrox_model import GroupParameters, ACCLIMATIZATION_MEMORY_DAYS


# ----------------------------------------------------------------------------
# Exposure-memory kernel shapes (length 10, OLDEST -> NEWEST; normalised later).
# The three named shapes are the paper's Section 2.3 examples.
# ----------------------------------------------------------------------------
MEMORY_LINEAR        = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], float)  # paper: young
MEMORY_FLAT_TO_LIGHT = np.array([1, 1, 1, 2, 2, 3, 3, 3, 3, 3], float)  # paper: older
MEMORY_VERY_FLAT     = np.array([1, 1, 1, 1, 1, 1, 1, 2, 2, 2], float)  # paper: vulnerable

# extra shapes for the extrapolated groups, on the same steep -> flat axis
MEMORY_STEEP         = np.array([1, 2, 3, 4, 5, 7, 9, 11, 13, 15], float)
MEMORY_MILD_LINEAR   = np.array([1, 2, 2, 3, 3, 4, 4, 5, 5, 6], float)
MEMORY_FLAT          = np.array([1, 1, 2, 2, 3, 3, 3, 4, 4, 4], float)


def _make_group(name, display_name, *, capacity, recovery_threshold,
                recovery_rate, memory, suppression=0.5, amplification=0.3,
                age_range=(0, 120), evidence="extrapolated") -> GroupParameters:
    """Terse constructor; keyword-only past the names to keep call sites legible."""
    return GroupParameters(
        name=name,
        display_name=display_name,
        max_acclimatization_capacity=capacity,
        recovery_threshold=recovery_threshold,
        base_recovery_rate=recovery_rate,
        strain_suppression_strength=suppression,
        strain_amplification=amplification,
        exposure_memory_weights=memory,
        age_range=age_range,
        evidence_status=evidence,
    )


# ============================================================================
# THE 23 GROUPS
# Ordered most-resilient -> most-vulnerable. Protective parameters
# (capacity, recovery_threshold, recovery_rate) descend together.
# ============================================================================

TARGET_GROUPS: Dict[str, GroupParameters] = {}


def _register(group: GroupParameters):
    TARGET_GROUPS[group.name] = group


# ---- athletic / elite (extrapolated): strongest protective loops -----------
# ---- checked (eleventh/final deep check, lighter-touch pass) --------------
# Unlike the vulnerable-tier groups above, this checks the resilient/healthy
# end of the roster -- CONFIRMED overall, with two genuine but narrow
# caveats that don't change the numbers but matter for how these groups are
# USED.
# CONFIRMING: elite-sport exertional heat stroke (EHS) incidence is
# empirically low (PMC9826288, 2022 review) -- consistent with these very
# high capacity values (0.95-1.00).
# CAVEAT 1 (about event/intensity choice, not these numbers): EHS risk is
# higher in SHORTER, higher-intensity races (e.g. 10 km) than in marathons,
# because athletes sustain a higher relative intensity for the (shorter)
# duration (PMC9826288). This means pairing "elite_athletes" with a
# high-MET/short-duration event is a materially different (higher) risk
# scenario than pairing it with marathon-pace/duration -- relevant to how
# HESTIA's met_value now actually drives population intensity (see the
# met_value integration fix earlier in this changelog), not something this
# group's own parameters should absorb.
# CAVEAT 2 (a real, narrow, behavioural mechanism no capacity parameter can
# represent): "high levels of motivation and exertion can in some cases
# blunt sensory feedback from hyperthermia that under normal circumstances
# alter behaviour" in elite competitive settings (PMC9826288) -- i.e. elite
# athletes may override the protective urge to slow down under competitive
# drive in a way recreational participants would not. ~85% of endurance-
# event collapses occur AFTER crossing the finish line, most benign
# (postural hypotension from reduced venous return once running stops), a
# minority heat-stroke (Holtzhause & Noakes 1997, cited in GSSI review) --
# consistent with, and an independent validation of, HESTIA's own dedicated
# post-finish TCF module. No number changes made: this is a genuine
# behavioural-override risk, not a capacity deficit, and the group's overall
# very-low-incidence positioning remains well supported.
_register(_make_group(
    'endurance_athletes', 'Endurance Athletes',
    capacity=1.00, recovery_threshold=2.5, recovery_rate=0.50,
    memory=MEMORY_STEEP, age_range=(18, 45),
))
_register(_make_group(
    'elite_athletes', 'Elite Athletes',
    capacity=0.95, recovery_threshold=2.0, recovery_rate=0.45,
    memory=MEMORY_STEEP, age_range=(18, 40),
))
_register(_make_group(
    'recreational_athletes', 'Recreational Athletes',
    capacity=0.85, recovery_threshold=1.8, recovery_rate=0.40,
    memory=MEMORY_LINEAR, age_range=(18, 55),
))

# ---- healthy working-age (extrapolated) ------------------------------------
# ---- outdoor_workers (checked, twelfth/final check) -----------------------
# The occupational heat-mortality literature is stark: farm/agricultural
# workers are 35x as likely as the general population to die of heat
# exposure (NEJM 2023, doi:10.1056/NEJMp2307850); construction workers 13x
# (95% CI 10.1-16.7), agriculture/forestry/fishing/hunting 35x (95% CI
# 26.3-47.0) vs. the all-other-industries average, per US OSHA rulemaking
# citing Gubernot et al. 2015 (Federal Register 2021-23250).
# CRITICAL POPULATION-MATCH CAVEAT: these mortality multipliers are
# overwhelmingly driven by UNACCLIMATIZED, inadequately protected workers --
# migrant and ethnic-minority outdoor workers face disproportionate risk,
# often without adequate water/shade/rest provisions (PMC11930879 scoping
# review). This PYROX group is explicitly labelled "(heat-exposed,
# ACCLIMATIZED)" -- a narrower, better-protected reference population than
# the raw occupational mortality statistics describe. Applying the 13-35x
# multipliers directly to this specific group would be a population
# mismatch. Acclimatization itself takes "at least four days" to begin
# (occupational heat-strain meta-analysis, doi:10.1080/23328940.2022.2030634)
# -- broadly consistent with, though a shorter minimum-onset estimate than,
# the 7-14-day full-acclimatization window used for unacclimatized_travelers
# above.
# [flagged for review] No number change made: the existing capacity (0.80,
# reasonably below the athlete tier but above indoor_workers) is defensible
# specifically for the ACCLIMATIZED, adequately-protected worker this group
# is labelled to represent. If PYROX or HESTIA ever needs to model the
# broader, mixed outdoor labour workforce (including unacclimatized/
# inadequately-protected workers) as its own category, that would warrant a
# substantially more severe, SEPARATE group informed by the 13-35x mortality
# multipliers above -- not a correction to this one.
_register(_make_group(
    'outdoor_workers', 'Outdoor Workers (heat-exposed, acclimatized)',
    # Checked (twelfth check in this series): largely CONFIRMS the existing
    # high-capacity positioning, with one practical caveat.
    # Genuine acclimatization is strongly protective: a Serbian military
    # study found acclimatized soldiers suffered NO detrimental effects of
    # exertional heat stress compared to unacclimatized counterparts
    # (cited in Notley et al. 2022, comprehensive review and meta-analysis,
    # doi:10.1080/23328940.2022.2030634, PMC9154804). US occupational heat
    # mortality (30-60 deaths/year) is a "miniscule fraction" of total
    # heat-attributable deaths (150,000+/year) -- consistent with this
    # group's high capacity relative to the general population (medRxiv
    # 2026 mortality-burden study). CAVEAT: acclimatization raises the
    # tolerance ceiling but does not make anyone immune -- a CDC/OSHA
    # retrospective review of 25 occupational heat illnesses (14 fatal)
    # found WBGT-based exposure limits were exceeded in ALL 14 fatalities,
    # most at moderate-to-heavy workload (PMC6048976, doi:10.15585/
    # mmwr.mm6730a4). I.e. this group's high capacity applies within the
    # standard heat-exposure-limit envelope; workload/environment
    # combinations that exceed it carry real, fatal risk even for
    # genuinely acclimatized workers. No number change: PYROX's existing
    # load-vs-capacity mechanics already handle this (a large enough load
    # overwhelms any capacity), so this is a usage caveat rather than a
    # parameter correction.
    capacity=0.80, recovery_threshold=1.7, recovery_rate=0.38,
    memory=MEMORY_LINEAR, age_range=(18, 65),
))

# ===== PAPER PROTOTYPE 1: Young Healthy Adult (18-45) =======================
# Paper Section 2.3, with the documented capacity/threshold correction.
_register(_make_group(
    'adults_18_45', 'Young Healthy Adults (18-45)',
    capacity=0.80,            # CORRECTED: was 0.25 in the printed table (swap)
    recovery_threshold=1.5,   # CORRECTED: was 0.5  in the printed table (swap)
    recovery_rate=0.30,       # as published
    suppression=0.5, amplification=0.3,  # as published -> critical_strain = 2.0
    memory=MEMORY_LINEAR,     # paper: linear weights [1..10]
    age_range=(18, 45), evidence="paper",
))

_register(_make_group(
    'indoor_workers', 'Indoor Workers (non-cooled)',
    # Checked (thirteenth/final check): non-cooled indoor occupational heat
    # exposure (warehouses, bakeries, foundries) is explicitly recognised as
    # a real, distinct risk category, not just a milder version of outdoor
    # exposure (NEJM 2023, doi:10.1056/NEJMp2307850: "many other workers
    # face serious heat exposure inside buildings"). However, no specific
    # quantitative mortality/morbidity multiplier distinct from outdoor
    # occupational statistics was located for this subgroup. The existing
    # relative positioning below outdoor_workers (0.70 vs 0.80) is
    # physiologically reasonable -- indoor non-cooled workers generally lack
    # the acclimatization benefit outdoor workers develop through regular,
    # repeated heat exposure -- but this is a plausibility judgement, not a
    # literature-derived correction. No number change made.
    capacity=0.70, recovery_threshold=1.3, recovery_rate=0.32,
    memory=MEMORY_LINEAR, age_range=(18, 65),
))
_register(_make_group(
    'youth_10_18', 'Youth (10-18)',
    # Previously capacity=0.65, recovery_threshold=1.2, suppression=0.48 --
    # positioned as meaningfully impaired vs. healthy adults.
    # Direct comparative studies find pubertal/adolescent thermoregulation is
    # NOT clearly inferior to adults' under most conditions: no differences
    # in thermoregulation were found between boys and adult men during
    # cycling in 88°F heat (Rowland et al., cited in Falk & Dotan 2008,
    # Applied Physiology Nutrition and Metabolism, doi:10.1139/H07-185); "no
    # epidemiological data show higher heat-injury rates in children, even
    # during heat waves" (same review). When exercise workload is scaled to
    # body size, heat production per body mass is expected to be equal
    # between children/youth and adults (Rowland 2008, J Appl Physiol,
    # doi:10.1152/japplphysiol.01196.2007) -- consistent with PYROX/HESTIA's
    # own MET-based (per-kg) intensity scaling. The American Academy of
    # Pediatrics revised its heat-risk guidance in 2011 in light of this
    # evidence. What DOES remain genuinely supported: slower heat
    # acclimatization and behavioural under-drinking (voluntary dehydration)
    # -- already represented here via the MEMORY_MILD_LINEAR kernel, left
    # unchanged.
    # CALIBRATION: capacity and recovery_threshold raised toward the healthy-
    # adult tier; recovery_rate and suppression left unchanged (no specific
    # evidence located on day-to-day recovery or the critical-strain gain).
    capacity=0.72, recovery_threshold=1.4, recovery_rate=0.35,
    suppression=0.48, memory=MEMORY_MILD_LINEAR, age_range=(10, 18),
))
_register(_make_group(
    'middle_aged_45_65', 'Middle-aged Adults (45-65)',
    # Checked (final, fourteenth check in this series): the heat-vulnerability
    # literature consistently frames risk as rising with age (see the
    # paper-validated adults_18_45 -> elderly_65_85 -> very_elderly_85plus
    # prototypes), but does not offer a specific, separately quantified
    # middle-age-specific multiplier distinct from that general age gradient
    # -- most studies bin "working-age" adults together or contrast general
    # populations against "elderly" specifically, without a distinct
    # middle-aged category. This group's values sit as a reasonable
    # monotonic interpolation between the paper-validated young-adult and
    # elderly prototypes; no specific literature was found to challenge or
    # refine that interpolation, and no number change is made.
    capacity=0.55, recovery_threshold=1.0, recovery_rate=0.28,
    suppression=0.48, memory=MEMORY_MILD_LINEAR, age_range=(45, 65),
))

# ===== PAPER PROTOTYPE 2: Healthy Older Adult (65-85) =======================
# Paper Section 2.3, values as published (the middle row needs no swap).
_register(_make_group(
    'elderly_65_85', 'Healthy Older Adults (65-85)',
    capacity=0.45,            # as published
    recovery_threshold=0.8,   # as published
    recovery_rate=0.18,       # as published
    suppression=0.5, amplification=0.3,  # critical_strain = 2.0
    memory=MEMORY_FLAT_TO_LIGHT,  # paper: flat-to-light
    age_range=(65, 85), evidence="paper",
))

# ---- special populations (extrapolated) ------------------------------------
# ---- unacclimatized travelers (recalibrated) -------------------------------
# Previously extrapolated with no direct citation (capacity=0.50,
# recovery_threshold=0.9, recovery_rate=0.30).
#
# KEY FINDING -- this group needed a different kind of correction than
# obesity/cardiovascular_disease/medication_impaired: those are permanent
# impairments to the CEILING of protection. Lack of acclimatization is a
# TEMPORARY, fully recoverable state -- the consistent finding across CDC,
# occupational-health and sports-medicine sources is that most heat
# acclimatization completes within 7-14 days of regular heat exposure
# (cardiovascular adaptations within the first week; sweating adaptations
# needing the full 10-14 days), reaching the SAME normal ceiling as anyone
# else of similar fitness -- not a permanently lower one (CDC NIOSH,
# https://www.cdc.gov/niosh/heat-stress/recommendations/acclimatization.html;
# ScienceDirect Heat Acclimatization overview, ref. Sawka et al.; Racinais
# et al. 2015 consensus statement on training/competing in the heat,
# doi:10.1007/s40279-015-0343-6, which explicitly flags unacclimatized
# participants in mass-participation events -- this project's own use case --
# as a population event organizers should specifically address).
# max_acclimatization_capacity was therefore UNDER-representing this group:
# the vulnerability is the SLOW START, which this group's exposure_memory
# kernel already models correctly (a zero-weighted ramp -- "no prior
# exposure" -- rather than the linear/flat kernels used elsewhere). Capping
# the ceiling too, on top of the slow start, double-penalises a condition
# that literature shows resolves to normal within about two weeks.
# Also quantified: fit individuals acclimatize ~50% faster than unfit ones
# (California DIR heat illness prevention guidance), and during moderate/
# heavy work, VO2max thresholds of 30/36.5 mL/kg/min respectively predict
# who runs a higher core temperature while unacclimatized (Notley et al.,
# Frontiers in Physiology 2020, doi:10.3389/fphys.2020.541483) -- consistent
# with this being about SPEED and FITNESS-INTERACTION rather than a capped
# ceiling.
#
# CALIBRATION: max_acclimatization_capacity raised (0.50 -> 0.65) to reflect
# a normal, recoverable ceiling for adults of typical fitness -- positioned
# just below indoor_workers (0.70, already-acclimatized) rather than among
# the permanently-impaired groups. recovery_threshold and base_recovery_rate
# LEFT UNCHANGED (0.9, 0.30): reasonably already reflect the reduced-but-not-
# absent tolerance during the early, not-yet-acclimatized window, and no more
# specific quantitative anchor was found to justify moving them.
_register(_make_group(
    'unacclimatized_travelers', 'Unacclimatized Travelers',
    capacity=0.65, recovery_threshold=0.9, recovery_rate=0.30,
    suppression=0.52,
    memory=np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], float),  # no prior exposure
    age_range=(18, 65),
))
_register(_make_group(
    'children_6_10', 'Children (6-10)',
    # Previously capacity=0.45, recovery_threshold=0.8 -- positioned near
    # elderly_65_85's tier (0.45/0.8, near-identical).
    # Same evidence base as youth_10_18 above (Falk & Dotan 2008; Rowland
    # 2008): direct comparisons of prepubertal children and adults found no
    # thermoregulatory inferiority under most ambient conditions, and no
    # epidemiological excess of heat injury in children. Kept below
    # youth_10_18, though, for two genuine remaining caveats specific to
    # younger children: (i) the higher surface-area-to-mass ratio becomes a
    # net LIABILITY rather than an asset specifically in EXTREME heat
    # (ambient > skin temperature) -- faster environmental heat GAIN, not
    # just faster loss (Falk & Dotan 2008; Smith & Cheung 2022 methodological
    # review, doi:10.3389/fphys.2020...); (ii) weaker behavioural self-
    # regulation of drinking (voluntary dehydration is well documented and
    # more pronounced at younger ages -- Bar-David et al., reviewed in
    # Chaseling & Filingeri 2019 PMC6770410). Memory kernel (MEMORY_FLAT,
    # slower acclimatization) left unchanged -- still genuinely supported.
    capacity=0.58, recovery_threshold=1.0, recovery_rate=0.28,
    suppression=0.50, memory=MEMORY_FLAT, age_range=(6, 10),
))
# ---- obesity (recalibrated against direct thermoregulation literature) ----
# Previously extrapolated by interpolation only (capacity=0.42,
# recovery_threshold=0.8, recovery_rate=0.22, suppression=0.52,
# memory=MEMORY_FLAT), with no direct citation. Re-examined against the
# thermoregulation-specific literature (not just the general EHI-risk
# literature already cited elsewhere in this project):
#
# MECHANISM: the best-controlled comparisons (mass-matched for total body
# mass AND fitness, isolating adiposity itself) show sweating rate and
# cutaneous vasodilation are essentially NORMAL in high-body-fat individuals
# -- the sudomotor/vasomotor effector systems are not "broken" by adiposity
# per se (Cramer & Jay 2015/2016, J Appl Physiol, mass-matched: LO-BF 10.8%
# vs HI-BF 32.0%, doi:10.1152/japplphysiol.00906.2015). The dominant
# mechanism is instead BIOPHYSICAL/GEOMETRIC: a lower body-surface-area-to-
# mass ratio in higher-adiposity individuals means less relative surface for
# evaporative heat loss per unit of heat produced -- "for the same heat load
# per unit body weight, the increase in tissue temperature will be greater
# in an obese than in a lean person" (Kenney 1985, cited in Chiang et al.
# 2024, doi:10.1016/j.puhe.2023.11.008). At a FIXED ABSOLUTE heat production,
# mass-matched HI-BF participants showed ~32% greater core-temperature rise
# than LO-BF (ΔTre 0.87 vs 0.66 °C over 60 min; Cramer & Jay 2016,
# doi:10.1152/japplphysiol.00768.2016). An unmatched systematic review/
# meta-analysis (10 studies, n=211, high-BF 36.7±11.8% vs low-BF 17.8±5.7%)
# found the same direction, though several included studies did not control
# for the mass/fitness confound (Wickham et al. 2021, J Sci Med Sport,
# doi:10.1016/j.jsams.2021.06.004).
#
# A separate, smaller pilot study on 6-week heat-acclimation TRAINING
# specifically in overweight/obese participants found suggestive (not
# definitive -- n=8, pilot-level) evidence that the ADAPTATION PROCESS itself
# is slower during the initial weeks, attributed to elevated adiposity
# impairing core-to-skin conductive heat exchange (Faulkner et al. 2015,
# doi:10.1186/2046-7648-4-S1-A115).
#
# CALIBRATION: recovery_threshold lowered (0.80 -> 0.75) and
# max_acclimatization_capacity lowered slightly (0.42 -> 0.40) to reflect
# this reduced heat-dissipation margin and the suggestive slowed-adaptation
# finding; exposure memory flattened (MEMORY_FLAT -> MEMORY_VERY_FLAT) for
# the same reason. base_recovery_rate and strain_suppression_strength are
# LEFT UNCHANGED: no thermoregulation-specific literature was found bearing
# directly on day-to-day/overnight strain clearance or on the critical-
# strain coupling gain for adiposity specifically -- extending the change to
# those two knobs would not be evidence-based.
# [flagged for review] The magnitude of the two changed values is a reasoned,
# literature-anchored judgement, not a direct unit conversion from the cited
# %ΔTre figures (PYROX's load/strain units and the cited studies' physical
# units are not directly interconvertible) -- please review, Koos. Deliberately
# kept level with, not below, cardiovascular_disease/physical_disabilities in
# this ranking: the cited evidence supports a real but moderate effect from
# adiposity alone, not one exceeding a diagnosed cardiovascular condition.
_register(_make_group(
    'obesity', 'Individuals with Obesity',
    capacity=0.40, recovery_threshold=0.75, recovery_rate=0.22,
    suppression=0.52, memory=MEMORY_VERY_FLAT, age_range=(18, 85),
))
# ---- cardiovascular disease (recalibrated against direct literature) ----
# Previously extrapolated with no direct citation (capacity=0.40,
# recovery_threshold=0.75, recovery_rate=0.20, suppression=0.55).
#
# MECHANISM, more direct than obesity's: during heat stress, cardiac output
# must rise substantially to simultaneously perfuse the skin (up to 50-70% of
# resting CO, ~8 L/min) AND working muscle/vital organs -- up to ~13 L/min in
# healthy adults (Kenney et al. 2004, Circulation, doi:10.1161/
# CIRCULATIONAHA.105.540773, congestive heart failure patients). In CHF,
# cardiac output still rises but the increase "may be lower than that
# required to adequately perfuse the cutaneous circulation" -- cardiac output
# reserve genuinely inadequate for the COMBINED demand, and cutaneous
# vasodilator response itself is measurably attenuated (unlike obesity, where
# the sudomotor/vasomotor effectors were largely intact -- see the obesity
# entry above). This is exactly HESTIA's own CVR conjunctive criterion
# (T_rect exceedance AND CO_reserve <= 0), independently corroborating that
# construct. Classic work (Rowell) shows this competing-demand deficit is
# specifically revealed under COMBINED exercise+heat stress, not passive
# heat exposure alone -- i.e. precisely the event-participant scenario PYROX
# and HESTIA both model.
#
# ACCLIMATIZATION -- a genuinely reassuring finding, not just a gap-filler:
# unlike obesity, the literature does NOT support cutting acclimatization
# capacity further. Heat acclimatization combined with exercise training
# measurably reduced ischemic injury after cardiac surgery in coronary artery
# disease patients (Horowitz & Hasin 2023, Front Physiol, 30 years of
# heat-acclimation-mediated cross-tolerance research, doi:10.3389/
# fphys.2023.1074391), and CAD patients show real peak-VO2 improvement with
# training (HIIT vs MICT meta-analysis, though this specific benefit was NOT
# significant for heart-failure patients specifically -- PMC9203221). Given
# this PYROX group spans the whole cardiovascular-disease spectrum (not just
# HF), max_acclimatization_capacity is left UNCHANGED: no evidence supports
# lowering it further, and some evidence (CAD-specific) argues the adaptive
# loop remains meaningfully functional.
#
# CALIBRATION: recovery_threshold lowered slightly (0.75 -> 0.70) to reflect
# the more direct, organ-level (not just biophysical) nature of this
# mechanism relative to obesity; max_acclimatization_capacity, base_recovery_
# rate and strain_suppression_strength LEFT UNCHANGED -- no direct evidence
# located bearing on the latter two, and positive evidence against changing
# the first.
# [flagged for review] As with obesity, this magnitude is a reasoned,
# literature-anchored judgement, not a unit conversion -- please review.
_register(_make_group(
    'cardiovascular_disease', 'Cardiovascular Disease',
    capacity=0.40, recovery_threshold=0.70, recovery_rate=0.20,
    suppression=0.55, memory=MEMORY_FLAT, age_range=(40, 85),
))
_register(_make_group(
    'pregnant_t1', 'Pregnant Women (Trimester 1)',
    # ---- STRUCTURALLY recalibrated (not just a number change) --------------
    # Previous values (capacity=0.42, recovery_threshold=0.8, recovery_rate=
    # 0.25, suppression=0.50) modelled T1 as a moderately-reduced-capacity
    # group on the same axis as e.g. frail elderly -- a monotonic "gets worse
    # T1->T2->T3" story. The evidence does not support that shape for T1:
    # the risk is not a reduced day-to-day thermoregulatory CAPACITY but an
    # ACUTE THRESHOLD effect during a specific developmental window. Maternal
    # core temperature >=39-39.5C during organogenesis (first trimester) is
    # teratogenic -- associated with neural tube defects, cardiac
    # malformations, oral clefts (Ravanelli, Casasola, English, Edwards & Jay
    # 2019, Br J Sports Med, systematic review with best-evidence synthesis,
    # doi:10.1136/bjsports-2017-098914; summarised in Science/AAAS 2021,
    # "How much heat is dangerous during pregnancy?"). This is a single-
    # exposure, threshold-crossing risk tied to a developmental window, not a
    # multi-day cumulative-strain process -- PYROX's own critical_strain
    # mechanism (= 1/strain_suppression_strength, the point at which the
    # model flags "danger") is the right tool for this, not the day-to-day
    # capacity/threshold/recovery knobs.
    # STRUCTURAL FIX: capacity/threshold/recovery_rate raised to a normal-
    # adult tier (no direct evidence of reduced maternal thermoregulatory
    # capacity specifically in T1 was located; by extension from the T2/T3
    # findings below, assumed similarly unimpaired) -- but
    # strain_suppression_strength raised sharply (0.50 -> 0.90), collapsing
    # critical_strain from 2.0 to ~1.11, so that a single significant heat
    # exposure reaches the model's "danger" flag quickly, matching the acute/
    # threshold nature of the teratogenic risk instead of requiring days of
    # accumulation.
    # [flagged for review] This is a genuine structural reinterpretation of
    # what this group represents within PYROX's framework, not a routine
    # recalibration -- please review the mapping itself, not just the numbers.
    capacity=0.70, recovery_threshold=1.2, recovery_rate=0.28,
    suppression=0.90, memory=MEMORY_FLAT, age_range=(18, 45),
))
_register(_make_group(
    'physical_disabilities', 'Physical Disabilities',
    # ---- HETEROGENEITY FLAG (same pattern as medication_impaired) --------
    # Previously capacity=0.40, recovery_threshold=0.75 -- a single, moderate
    # severity level applied to an extremely heterogeneous category.
    # Spinal cord injury (SCI) is the best-documented case, and reveals a
    # SHARP severity gradient by lesion level that this single group cannot
    # represent: above T6, the hypothalamus can no longer send or receive
    # signals below the lesion -- no sweating, no cutaneous vasodilation, no
    # heart-rate increase to simultaneously perfuse skin AND muscle below the
    # lesion, loss of ~50% of body surface area for thermoregulation
    # (Mayo Clinic Proceedings classic review; Price 2006 systematic review
    # of SCI thermoregulation, doi:10.3390/... reviewed in PMC8049141: "the
    # higher the lesion level, the more the thermoregulatory system is
    # impaired... individuals with a high lesion level, especially
    # tetraplegia, reached a higher core and skin temperature with a lower
    # sweat rate"). This is one of the most severe, mechanistically direct
    # impairments found across this entire recalibration series -- comparable
    # to or exceeding dementia's hypothalamic-damage mechanism for high-level
    # injuries specifically. But "physical disabilities" as a category also
    # spans conditions with normal autonomic/thermoregulatory function and
    # only reduced mobility/heat-avoidance behaviour (e.g. lower-limb-only
    # impairments), where this severity would not apply at all.
    # [flagged for review] As with medication_impaired, this single group
    # conflates a genuinely severe, well-characterised subset (high-level
    # SCI and similar autonomic-affecting conditions) with much milder cases.
    # Splitting into at least two groups (autonomic/thermoregulatory-affecting
    # vs. mobility-only) would better reflect the evidence -- a structural
    # change beyond a single-group recalibration, Koos's call. Pending that,
    # recovery_threshold is nudged down (0.75 -> 0.65), anchoring toward the
    # more severe, better-evidenced end of the spectrum for safety, since a
    # category built to flag risk should not average away its most vulnerable
    # members. capacity left unchanged -- the SCI evidence describes acute
    # failure of specific effector mechanisms rather than a capacity-loop
    # ceiling concept, and no clean quantitative anchor for that knob was
    # located.
    capacity=0.40, recovery_threshold=0.65, recovery_rate=0.22,
    suppression=0.50, memory=MEMORY_FLAT, age_range=(18, 85),
))

# ---- high-risk (extrapolated) ----------------------------------------------
_register(_make_group(
    'chronic_comorbidities', 'Chronic Comorbidities',
    # ---- checked, largely CONFIRMED (third such outcome in this series) ---
    # A cleanly quantified heat-specific dose-response relationship was
    # located: among people with 0, 1, 2, or >=3 of five chronic diseases
    # (cardiovascular disease, diabetes, mental disorders, asthma/COPD,
    # chronic kidney disease), the odds ratios for heat-related emergency
    # hospitalisation were 1.00 / 1.06 / 1.08 / 1.13 respectively (Queensland,
    # Australia hospital registry, 2004-2016, ScienceDirect,
    # doi:10.1016/j.envres.2024...). The key finding: it is the NUMBER of
    # co-occurring conditions that predicts risk, more than which specific
    # conditions -- and the heat-specific incremental effect of stacking
    # conditions is comparatively MODEST (6-13% relative increase), not
    # dramatic, since the individual conditions themselves are already
    # separately represented elsewhere in this roster (cardiovascular_
    # disease, obesity, medication_impaired). This modest, heat-specific
    # multimorbidity-count effect should not be confused with the much
    # larger dose-response relationships reported in the GENERAL (all-cause,
    # multi-year) multimorbidity-mortality literature (e.g. Charlson
    # Comorbidity Index studies reporting hazard ratios of 3-12+ for severe
    # vs. no comorbidity) -- those measure a different outcome over a
    # different timescale and are not directly transferable to a single
    # heat-event strain model.
    # CHECK: this group's existing capacity (0.35) sits ~12.5% below
    # cardiovascular_disease's newly-recalibrated capacity (0.40); recovery_
    # threshold (0.65) sits ~7% below cardiovascular_disease's (0.70). Both
    # are in the same order of magnitude as the ~6-13% heat-specific
    # multimorbidity-count effect found above -- i.e. the existing
    # calibration already roughly matches what direct evidence supports for
    # the INCREMENTAL effect of stacking multiple conditions on top of a
    # single-condition baseline. No substantive number change made here;
    # documented rather than corrected, as with severe_mental_illness and
    # dementia above.
    capacity=0.35, recovery_threshold=0.65, recovery_rate=0.18,
    suppression=0.55, memory=MEMORY_VERY_FLAT, age_range=(40, 85),
))
_register(_make_group(
    'pregnant_t2', 'Pregnant Women (Trimester 2)',
    # Previously capacity=0.35, recovery_threshold=0.7, recovery_rate=0.20,
    # suppression=0.52 -- positioned near the most-impaired groups.
    # Well-controlled climate-chamber studies found NO elevated core
    # temperature in pregnant women (2nd/3rd trimester) vs non-pregnant
    # controls during moderate- or even high-intensity exercise -- if
    # anything, pregnancy's physiological changes (higher blood volume,
    # earlier/greater sweating) may enhance heat dissipation (Smallcombe,
    # Puhenthirar et al., Sports Medicine 2021, doi:10.1007/s40279-021-01504-y,
    # University of Sydney; consistent finding replicated at high running
    # intensity in "Cool Mama", doi:10.1016/j.jesf.2024.08.001). The real
    # risk pathway operates differently: heat-induced reductions in
    # uteroplacental blood flow and inflammatory responses linked to preterm
    # birth and fetal growth restriction (see pregnant_t3 below) -- a
    # cumulative-exposure mechanism that DOES fit PYROX's day-to-day strain
    # framework reasonably, unlike T1's acute threshold. capacity/threshold/
    # recovery_rate raised accordingly; suppression_strength left close to
    # the healthy-adult value (no acute-collapse restructuring needed here).
    capacity=0.65, recovery_threshold=1.0, recovery_rate=0.25,
    suppression=0.50, memory=MEMORY_VERY_FLAT, age_range=(18, 45),
))
# ---- medication-impaired thermoregulation (recalibrated) ------------------
# Previously extrapolated with no direct citation (capacity=0.33,
# recovery_threshold=0.6, recovery_rate=0.18, suppression=0.55).
#
# KEY FINDING -- HETEROGENEITY, not a uniform effect: the best available
# evidence (Ebi et al. 2024 systematic review + meta-analysis of medication
# effects on core temperature during heat stress, eClinicalMedicine/Lancet,
# doi:10.1016/j.eclinm.2024.102847) shows that many commonly-flagged
# "heat-risk medications" in public health guidance (diuretics,
# antidepressants, antipsychotics, anxiolytics) show NO measured core-
# temperature effect during heat stress -- despite being routinely listed
# together as equally risky (CDC 2025 clinical guidance; WHO). Proven,
# quantified core-temperature increases were found only for: strong
# anticholinergics (ACB=3), +0.42°C at ambient >=30C (95% CI 0.04-0.79),
# via reduced sweating; non-selective beta-blockers, +0.11°C (95% CI
# 0.02-0.19), via reduced cutaneous vasodilation; and anti-Parkinson's
# agents, +0.13°C (95% CI 0.07-0.19). This does not mean diuretics/
# antipsychotics are safe in heat -- they may still raise real-world risk via
# dehydration/electrolyte disturbance or sedation-driven reduction in heat-
# avoidance behaviour -- but those are different mechanisms than direct
# thermoregulatory impairment, and PYROX's single "medication_impaired"
# category cannot distinguish them.
# [flagged for review] This single group conflates drug classes with proven
# thermoregulatory mechanisms (anticholinergics, non-selective beta-blockers)
# and drug classes with unproven-but-plausible other-mechanism risk
# (diuretics, antipsychotics) under one label. Splitting this into separate
# groups would better reflect the evidence; not done here since that is a
# structural change beyond a single-group recalibration -- Koos's call.
#
# CALIBRATION: recovery_threshold nudged down slightly (0.60 -> 0.55),
# anchored to the strong-anticholinergic effect size as a conservative
# (worse-case, safety-oriented) reference point for a heterogeneous category.
# capacity, base_recovery_rate and strain_suppression_strength LEFT
# UNCHANGED: no direct evidence located bearing on acclimatization capacity
# or day-to-day recovery specifically for medicated individuals.
_register(_make_group(
    'medication_impaired', 'Thermoregulation-impairing Medication',
    capacity=0.33, recovery_threshold=0.55, recovery_rate=0.18,
    suppression=0.55, memory=MEMORY_VERY_FLAT, age_range=(30, 85),
))
_register(_make_group(
    'children_0_6', 'Children (0-6)',
    # Previously capacity=0.32, recovery_threshold=0.65 -- among the most
    # severe of all 23 groups.
    # UNLIKE children_6_10/youth_10_18 above: the evidence here does NOT
    # support walking the vulnerability back to the same degree. Infants and
    # young children ARE established as at genuinely higher risk of heat-
    # related morbidity/mortality than older children and adults (van de
    # Kamp & Daanen 2025, VU Amsterdam, narrative review on infant
    # thermoregulation, doi:10.3390/ijerph22081265). Critically, that review
    # explicitly states it "remains unclear whether this vulnerability stems
    # from immature thermoregulatory mechanisms or simply from their
    # dependence on caregivers" -- i.e. a meaningful share of the real-world
    # risk may be BEHAVIOURAL/CAREGIVER-DEPENDENCY (can't remove own
    # clothing, fetch water, or leave a hot environment unassisted), not a
    # reduced physiological CAPACITY that PYROX's control-theoretic
    # acclimatization-loop framework is built to represent. Given this
    # mechanistic ambiguity, only a small, cautious upward adjustment is made
    # here (unlike the larger corrections above) -- this group should be
    # revisited if event-specific data on actual participant ages in this
    # bracket (e.g. family/short-distance categories) become available, since
    # PYROX models active event participants, who at these ages skew toward
    # the upper (more independent, less infant-like) end of the 0-6 range.
    capacity=0.38, recovery_threshold=0.72, recovery_rate=0.25,
    suppression=0.52, memory=MEMORY_VERY_FLAT, age_range=(0, 6),
))

# ===== PAPER PROTOTYPE 3: Vulnerable Older Adult (85+, comorbidities) =======
# Paper Section 2.3, with the documented capacity/threshold correction.
_register(_make_group(
    'very_elderly_85plus', 'Vulnerable Older Adults (85+)',
    capacity=0.25,            # CORRECTED: was 0.80 in the printed table (swap)
    recovery_threshold=0.5,   # CORRECTED: was 1.5  in the printed table (swap)
    recovery_rate=0.12,       # as published
    suppression=0.5, amplification=0.3,  # critical_strain = 2.0
    memory=MEMORY_VERY_FLAT,  # paper: very flat
    age_range=(85, 120), evidence="paper",
))

_register(_make_group(
    'pregnant_t3', 'Pregnant Women (Trimester 3)',
    # Previously capacity=0.28, recovery_threshold=0.5, recovery_rate=0.18,
    # suppression=0.55 -- the most severe of the three, tied with the most
    # impaired groups in the whole roster.
    # Even at high exercise intensity, well-controlled studies of pregnant
    # athletes up to 35 weeks gestation did not find elevated core
    # temperature vs non-pregnant controls (Cool Mama, doi:10.1016/
    # j.jesf.2024.08.001) -- so, as with T2, a severely reduced day-to-day
    # thermoregulatory CAPACITY is not well supported. What IS supported,
    # and distinct from T1's acute teratogenic window: a real, cumulative-
    # exposure risk pathway specific to later pregnancy -- heat stress can
    # promote oxytocin/prostaglandin F2-alpha release (preterm labor
    # trigger) and reduce uteroplacental blood flow (fetal growth
    # restriction/low birth weight). A large epidemiological study (n=2M+
    # births, California) found low-birth-weight risk rose with heat
    # exposure in the 2nd/3rd trimester specifically, while first-trimester
    # heat showed an INVERSE relationship with preterm birth (Taking the
    # Heat, PMC6910775) -- i.e. this risk genuinely is trimester-specific and
    # NOT simply "more of the same T1 problem, worse". capacity/threshold
    # raised substantially but kept somewhat below T1/T2, reflecting the
    # added physical burden of late pregnancy (reduced venous return supine,
    # increased resting metabolic rate, larger effective body mass) on top
    # of the cumulative placental-flow risk pathway.
    capacity=0.55, recovery_threshold=0.75, recovery_rate=0.22,
    suppression=0.53, memory=MEMORY_VERY_FLAT, age_range=(18, 45),
))
# ---- severe mental illness (checked, largely CONFIRMED not contradicted) --
# Unlike several groups above, this deep check mostly CONFIRMS the existing
# severity rather than correcting it -- worth documenting properly rather
# than leaving as an unsourced extrapolation, even though the numbers
# themselves are barely changed.
#
# Real-world, well-quantified excess mortality: psychiatric patients had
# DOUBLE the risk of dying during a heat wave vs. the general population
# (Bark's analysis of New York State psychiatric hospital data, 1950-1984,
# discussed in Bassil & Cole 2010, reviewed in PMC6068666) -- critically,
# this elevated risk PRE-DATES widespread antipsychotic use (pre-1960s),
# meaning a genuine illness-intrinsic component exists independent of
# medication. In the 2021 British Columbia extreme heat event (1,649 deaths
# in 8 days), schizophrenia topped the list of conditions associated with
# death; antipsychotic dispensation was INDEPENDENTLY associated with a
# further 2.43x mortality odds ratio after controlling for illness severity
# and comorbidities (Lee et al. 2025, Scientific Reports,
# doi:10.1038/s41598-025-17591-0).
# MECHANISM is genuinely multifactorial and only partly understood: central/
# hypothalamic dysfunction and autonomic dysfunction from the illness itself
# (ScienceDirect 2025 critical review, doi:10.1016/j.scitotenv.2025.179...);
# medication effects (already separately represented in PYROX's
# medication_impaired group -- some overlap is expected here since the
# cited mortality studies measure REAL populations who are typically both
# ill AND medicated, not an isolated "illness only" case); AND a distinct
# BEHAVIOURAL/SOCIAL pathway -- social isolation meaning no one monitors
# wellbeing during a heat wave, and reduced heat-risk awareness/self-care
# (Treatment Advocacy Center research summary). This last pathway is NOT a
# physiological capacity deficit and cannot be represented by any of
# PYROX's four protective knobs -- flagged as a genuine framework
# limitation for this group specifically, not something a parameter change
# can fix.
# [flagged for review] No number change made here beyond a marginal
# rounding-level nudge: the existing severity (already among the lowest in
# the roster) is well supported by direct, quantified evidence, not an
# unvalidated extrapolation as it previously appeared to be. Given the
# social-isolation pathway this framework cannot represent, treat this
# group's output as a likely UNDERESTIMATE for isolated individuals
# specifically, not an overestimate.
_register(_make_group(
    'severe_mental_illness', 'Severe Mental Illness',
    capacity=0.27, recovery_threshold=0.5, recovery_rate=0.16,
    suppression=0.55, memory=MEMORY_VERY_FLAT, age_range=(18, 85),
))
# ---- dementia (checked, CONFIRMED with even more direct evidence than SMI) -
# As with severe_mental_illness, this deep check largely confirms rather
# than corrects the existing (most severe in the whole roster) positioning
# -- but here the mechanism is more directly physiological, not just
# correlational/multifactorial.
#
# DIRECT STRUCTURAL MECHANISM: dementia, particularly Alzheimer's disease,
# can damage the hypothalamus itself -- the body's central thermostat --
# genuinely impairing both detection of and response to temperature change,
# not merely a behavioural/awareness deficit (Ithy clinical summary,
# consistent with the peer-reviewed literature below). Several studies
# report ELEVATED BASELINE core body temperature in AD patients, hypothesised
# to result from increased cytokine expression and neuroinflammation in the
# brain (Xu et al., reviewed in PMC9898200, doi:10.1016/j.envint...). A
# population-based study of heatwave mortality in AD/dementia patients aged
# 60+ in China (2013-2020, n covering the whole period) found significantly
# elevated mortality on heatwave vs. non-heatwave days using within-person
# conditional logistic regression (PMC11490898, doi:10.1016/
# j.lanwpc.2024.101...). A rat heat-stroke model directly demonstrated
# hippocampal neuronal damage, degeneration and amyloid plaque deposition
# after heat stroke, substantiating the epidemiological dementia-risk finding
# with histopathological evidence (doi:10.1186/s13195-024-01515-7).
# MEDICATION: cholinesterase inhibitors (standard dementia pharmacotherapy,
# e.g. donepezil) reduce sweating -- a distinct, well-documented mechanism
# from the antipsychotic effects already captured in medication_impaired
# (some overlap expected, as many dementia patients with behavioural
# symptoms are prescribed both drug classes).
# COMPOUNDING, less physiologically representable: communication/behavioural
# deficits (may not recognise thirst, forget to drink, not understand the
# need to remove clothing or seek shade; heat-induced confusion/agitation can
# be mistaken for baseline dementia symptoms, delaying intervention) -- as
# with severe_mental_illness, this secondary pathway is not something
# PYROX's four protective knobs can represent, but here it compounds an
# already well-evidenced DIRECT physiological mechanism rather than being the
# primary driver.
# [flagged for review] No number change beyond documentation: this is the
# single most severe group in the entire 23-group roster (below even the
# paper-validated very_elderly_85plus prototype at 0.25), and the evidence
# located here is, if anything, more direct/robust than for
# severe_mental_illness -- supporting rather than challenging that ranking.
_register(_make_group(
    'dementia', "Dementia / Alzheimer's",
    capacity=0.22, recovery_threshold=0.45, recovery_rate=0.14,
    suppression=0.58, memory=MEMORY_VERY_FLAT, age_range=(65, 100),
))


# The three groups the paper actually specifies (validation anchor).
PAPER_PROTOTYPES = ('adults_18_45', 'elderly_65_85', 'very_elderly_85plus')


def get_group(name: str) -> GroupParameters:
    """Look up a group by name, defaulting to the healthy-older prototype."""
    return TARGET_GROUPS.get(name, TARGET_GROUPS['elderly_65_85'])
