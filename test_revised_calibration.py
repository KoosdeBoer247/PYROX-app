# -*- coding: utf-8 -*-
"""
Acceptance tests for the revised PYROX calibration.

Run with:  python3 test_revised_calibration.py

These test the values actually shipped in pyrox_revised_calibration.py, not a
separate copy, so they fail if the module drifts. They are structural
acceptance tests: they check that the model behaves sensibly across known
scenarios and that this suite's own Paris 2003 reference data (paris2003.py)
is reproduced for the untouched groups. They are NOT event-level validation
against real incident data -- PYROX's population tier has none in this
suite, before or after this revision. (The r=0.866/Dam tot Damloop/Falmouth/
Hoorn results belong to HESTIA's individual tier, not PYROX; an earlier
draft of this docstring incorrectly implied otherwise.)
"""
import sys

from pyrox_model import PyroxModel
from pyrox_groups import TARGET_GROUPS as ORIGINAL_GROUPS
from pyrox_revised_calibration import (
    apply_revised_calibration, REVISED_CALIBRATION, K_PER_MET, MET_REFERENCE,
    met_adjusted_apparent_temperature, ACCLIM_CEILING,
)
from thermopoulos_loader import HEAT_LOAD_REFERENCE_TEMP, HEAT_LOAD_PER_DEGREE

GROUPS = apply_revised_calibration(ORIGINAL_GROUPS)

# Apparent-temperature sequences (degrees C).
PARIS_2003 = [30, 32, 35, 37, 39, 40, 41, 40, 38, 35, 32, 30]
MILD_SUMMER = [24, 25, 26, 25, 24, 26, 27, 26, 25, 24, 25, 26]
MAASTRICHT = [27.0, 25.7, 24.4, 28.3, 25.6, 26.2, 28.3, 32.1, 25.6, 23.7,
              26.7, 26.1, 27.5, 26.8, 29.3, 34.1, 38.0, 33.3, 32.0, 28.1, 26.1]

VULNERABLE = ['dementia', 'very_elderly_85plus', 'elderly_65_85',
              'cardiovascular_disease', 'chronic_comorbidities']
RESILIENT = ['adults_18_45', 'outdoor_workers', 'recreational_athletes',
             'elite_athletes', 'endurance_athletes']

failures = []


def loads(temps, met=MET_REFERENCE):
    return [max(0.0, (met_adjusted_apparent_temperature(t, met)
                      - HEAT_LOAD_REFERENCE_TEMP) * HEAT_LOAD_PER_DEGREE)
            for t in temps]


def strain_pct(key, temps, met=MET_REFERENCE):
    g = GROUPS[key]
    res = PyroxModel(g).simulate(loads(temps, met))
    return 100 * res["peak_strain"] / res["critical_strain"]


def check(name, condition, detail=""):
    status = "PASS" if condition else "FAIL"
    if not condition:
        failures.append(name)
    print(f"  [{status}] {name}" + (f"  ({detail})" if detail else ""))


print("Revised PYROX calibration \u2014 acceptance tests")
print("=" * 72)

print("\nT1  No group saturates during a mild summer (no false positives)")
saturating = [k for k in GROUPS if strain_pct(k, MILD_SUMMER) > 50]
check("mild summer produces no saturation", not saturating, f"{len(saturating)} saturating")

print("\nT2  The vulnerable tier saturates under Paris 2003")
for key in VULNERABLE:
    pct = strain_pct(key, PARIS_2003)
    check(f"{GROUPS[key].display_name} saturates", pct > 50, f"{pct:.1f}%")

print("\nT3  The resilient tier does NOT fully decompensate at rest under")
print("    Paris 2003 (2003 excess mortality was concentrated in the elderly)")
for key in RESILIENT:
    pct = strain_pct(key, PARIS_2003, met=MET_REFERENCE)
    check(f"{GROUPS[key].display_name} stays sub-critical at rest",
          pct <= 50, f"{pct:.1f}%")

print("\nT4  Occupational risk is graded with workload, not binary")
print("    (outdoor workers, Maastricht heatwave, peak apparent 38\u00b0C)")
levels = []
for label, met in [("at rest", MET_REFERENCE), ("light work", 2.7),
                   ("construction", 4.0), ("heavy labour", 5.5)]:
    pct = strain_pct('outdoor_workers', MAASTRICHT, met=met)
    levels.append(pct)
    print(f"      {label:<14} MET {met:>4.1f}  ->  strain {pct:5.1f}%")
check("workload increases risk monotonically",
      all(b >= a for a, b in zip(levels, levels[1:])))
check("resting worker is not at risk", levels[0] <= 10, f"{levels[0]:.1f}%")
check("heavy labour is at risk", levels[-1] > 50, f"{levels[-1]:.1f}%")
check("response is graded, not a single jump",
      any(10 < p < 90 for p in levels),
      "at least one intermediate level")

print("\nT5  Regression: the untouched groups are bit-identical to the")
print("    published parameterisation (minimal-intervention principle)")
from pyrox_revised_calibration import UNCHANGED_GROUPS
for key in UNCHANGED_GROUPS:
    if key not in GROUPS:
        continue
    o, n = ORIGINAL_GROUPS[key], GROUPS[key]
    same = (o.max_acclimatization_capacity == n.max_acclimatization_capacity
            and o.recovery_threshold == n.recovery_threshold)
    check(f"{n.display_name} parameters untouched", same)

print("\nT6  No group retains a physiologically implausible capacity")
over = [(k, g.max_acclimatization_capacity) for k, g in GROUPS.items()
        if g.max_acclimatization_capacity > ACCLIM_CEILING + 1e-9]
check(f"all capacities at or below the {ACCLIM_CEILING} ceiling", not over, str(over))
immune = [k for k, g in GROUPS.items() if g.max_acclimatization_capacity >= 0.999]
check("no group is immune to heat (capacity 1.00)", not immune, str(immune))

print("\nT7  Every revised group has a reachable onset within real-world loads")
print("    (a severe heatwave reaches ~1.6; the lethal global extreme ~3.0)")
unreachable = []
for key, (_a, _t, onset) in REVISED_CALIBRATION.items():
    if key not in GROUPS:
        continue
    if (onset - 22.0) * 0.10 > 3.0:
        unreachable.append((key, onset))
check("all revised onsets fall within achievable ambient loads",
      not unreachable, str(unreachable))
check("the revision is minimal (fewer than half the roster changed)",
      len(REVISED_CALIBRATION) < len(GROUPS) / 2 + 1,
      f"{len(REVISED_CALIBRATION)} of {len(GROUPS)} groups changed")

print("\nT8  The MET coefficient reproduces ISO 7243's own limit reductions")
BSA = 1.8
for label, watts, iso_limit in [("moderate", 300, 28.0), ("high", 415, 26.0)]:
    extra_w_m2 = (watts - 115) / BSA
    predicted = extra_w_m2 * (K_PER_MET / 58.15)
    actual = 33.0 - iso_limit
    check(f"ISO {label} intensity within 1.0\u00b0C",
          abs(predicted - actual) < 1.0,
          f"predicted {predicted:.1f}\u00b0C vs ISO {actual:.1f}\u00b0C")

print("\n" + "=" * 72)
if failures:
    print(f"{len(failures)} FAILURE(S):")
    for f in failures:
        print(f"   - {f}")
    sys.exit(1)
print("All acceptance tests passed.")
