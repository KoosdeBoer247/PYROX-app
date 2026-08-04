# PYROX app â€” file inventory and deployment

Complete, verified set for the PYROX Streamlit app, as of 4 August 2026.
Everything in this folder goes in the **root** of the GitHub repo
(`KoosdeBoer247/PYROX`) â€” no subfolders. Verified from a clean directory:
all modules compile, all acceptance tests pass, and the app boots without
errors.

## What each file is

| File | Role | Change this? |
|---|---|---|
| `app.py` | The Streamlit page itself: layout, sidebar, all UI | Yes, main entry point |
| `requirements.txt` | Python dependencies for Streamlit Cloud | Only when adding a library |
| `decision_support.py` | Risk/strain explainer, hourly activity/rest guide, WBGTâ†”UTCI divergence check | App-layer, not model |
| `gpx_route.py` | GPX parsing, race pace/exposure profiles, course map | App-layer |
| `terrain_lookup.py` | ESA WorldCover land cover â†’ per-segment roughness â†’ terrain-varying WBGT/MRT | App-layer |
| `pyrox_model.py` | **PYROX model core** (paper Sec 2.2) | No â€” keep in sync with the suite |
| `pyrox_groups.py` | **Published 23-group roster** | No â€” keep in sync with the suite |
| `pyrox_revised_calibration.py` | Revised calibration + MET term, with derivation | No â€” keep in sync with the suite |
| `Thermopoulos_Data_Engine.py` | **Weather acquisition + thermal indices** (WBGT, UTCI, MRT, UHI, coastal) | No â€” keep in sync with the suite |
| `thermopoulos_loader.py` | Reads the engine's Excel output into PYROX inputs | No â€” keep in sync with the suite |
| `test_revised_calibration.py` | Acceptance tests (T1â€“T8) for the revised calibration | Run after any calibration change |
| `README.md` | Scientific scope and validation status | Update when scope changes |

The five "keep in sync" files are byte-identical to the HESTIA-PYROX
suite. Do not edit them here â€” edit them in the suite and copy across,
so the app can never silently drift from the research code.

## Deploying

1. Upload every file in this folder to the repo root (GitHub â†’ Add file â†’
   **Upload files**; drag the files in rather than pasting contents, which
   avoids the line-break corruption that caused an earlier SyntaxError).
2. Streamlit Cloud auto-redeploys on commit. If not: Manage app â†’ Reboot.
3. Main file path must be `app.py`.

## Running the tests

```
python test_revised_calibration.py
```
Expect: `All acceptance tests passed.`

## API quota note

Open-Meteo's free tier allows 10,000 calls/day, 5,000/hour, 600/minute,
counted **per IP** â€” so several Streamlit apps on the same host share one
quota. Cache lifetimes in `app.py` are tuned to this (see
`CACHE_TTL_GEOCODE` / `CACHE_TTL_FORECAST` / `CACHE_TTL_HISTORICAL`):
geocoding 30 days, forecasts 2 hours, historical ERA5 7 days, climatology
30 days. Re-running the same location within those windows costs no quota.

The 30-year climatology option is by far the most expensive feature (one
request per year of history) â€” leave it off while testing.

## Known limitations, deliberately kept visible in the UI

- PYROX's **population tier has no event-level validation** against real
  incident data. The r=0.866 correlation, Falmouth hindcasts and IRONMAN
  Hoorn results belong to HESTIA's individual tier, not PYROX.
- `final_risk` is **dimensionless** â€” meaningful only relative to the
  mild-summer and Paris 2003 reference scenarios shown beside it, not as a
  probability.
- WBGT under-weights radiant load (0.7 wet-bulb / 0.2 globe / 0.1 dry-bulb).
  The activity/rest guide flags hours where WBGT reads "safe" but UTCI is
  â‰¥32Â°C, rather than letting them pass silently.
- **ESA WorldCover fetching has not been exercised against the live AWS
  bucket** â€” the development sandbox had no S3 access. Tile naming follows
  ESA's documented convention and the downstream pipeline is tested, but
  the first real Streamlit Cloud run is the actual test. Any failure falls
  back to the sidebar terrain selector with a visible message.
- UTCI is **not** terrain-adjusted (it is defined at a fixed 10m reference
  wind); only WBGT and MRT vary by land cover.
- GPX timestamps are ignored; pace comes from the values entered in the UI.

