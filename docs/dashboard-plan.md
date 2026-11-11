> **Superseded 2026-11-11.** This was the plan for the first Fire spread page. The dashboard now has Active fires (built from satellite detections) and Past fires (this model, checked against what burned). Current plan and status: `../../docs/23-active-fires-redesign.md`; the model results: `results.md`.

# Fire spread in the dashboard: plan

Status: plan only. Nothing in `website/` has been changed. Written after the second look at the frozen test blocks (see docs/results.md).

## What ships

Nav label **Fire spread** (Hindi आग का फैलाव, Odia ନିଆଁ ବିସ୍ତାର). Pick an active fire, pick 24 h or 48 h (default 48 h), see a map of the cells within 8 km ranked by how likely they are to burn next.

- It is a **ranking**, shown as three bands (highest, high, moderate) by rank inside the window. No percentages: the scores are not calibrated.
- It is **not wind-driven**. Wind added nothing in every test, so the page says so and shows no wind arrows or animation.
- A forecast is not a claim. Nothing here enters the claims ladder.
- Skill is on the page, from docs/results.md: direction AUC 0.70 against 0.59 for plain proximity, and IoU at the true area 0.14 (24 h) and 0.17 (48 h) against 0.11 and 0.12, on blocks the model never saw. A short note says the truth is MODIS 500 m and the numbers are low in absolute terms.
- "What drove this ranking": the few inputs that mattered for this fire (distance, greenness, land cover, fuel dryness).
- Validation view: for past fires, the ranking drawn over what MODIS later showed burned.

Not shipped until it has its own test: a **footprint** (the burned area). The model ranks cells but does not say how many will burn. A separate area model (burned area from fire size, FRP, season, greenness) must beat "the median area" on held-out blocks first. Until then the map shows bands only, with no outline.

## Work packages

| # | Work | Exit |
|---|---|---|
| F1 | One feature builder: `build_features(lat, lon, t0, detections) -> grid + features` in `fire-spread/src/firespread/live.py`, reusing `make_grid`, `rasters`, `weather`, `activity`, and the extra features from script 21 | Reproduces the saved features of 50 test cases (max difference under 1e-4 outside the unavoidable source updates) |
| F2 | Static layer store: WorldCover tiles for the served states, roads and rivers rasterised from the OSM extract, WorldPop 1 km, one hashed manifest | Row counts and hashes match; layers cover every served state |
| F3 | Greenness cache: nightly latest MODIS NDVI composite and the same one a year earlier, per tile, stored; no Planetary Computer call at request time | A staged failure serves the cached composite and marks it stale |
| F4 | Weather: Open-Meteo forecast for t0 to +48 h (existing feed), ERA5 rain history and soil moisture (existing archive calls, cached daily) | Inputs freshness is returned with every run |
| F5 | Active-fire source: FIRMS NRT through the existing `feeds/firms.py`, same filters as research (type 0, nominal/high confidence, event linking at 1 km and 36 h, forest share at least 0.5) | Event table for a past week equals the research table |
| F6 | Scoring job: for each active fire, build features, apply `rank2_24h` and `rank2_48h`, store cells and bands with `issued_at`, `model_version`, input freshness and a `degraded` flag; runs after each new VIIRS pass | Idempotent; a run for a past fire reproduces the research scores |
| F7 | Model bundle: the two joblib files with feature list, sha256, training case list and code hash; model fitted only on research data | Loaded by hash, refused on mismatch |
| F8 | API: `GET /api/fire-spread/events`, `GET /api/fire-spread/{event}?horizon=48`; bands only, no raw scores; server-side access scope by state | Contract tests; no raw score in any response |
| F9 | Frontend page under `features/`, three languages, mobile first, skill panel and "not a forecast of wind" note | Visual check at phone width; accessibility pass |
| F10 | Rolling validation: when MODIS burn dates arrive (about a month later) score each past ranking (AUC, direction AUC, IoU) and show the running numbers; a drop below proximity raises an alert | Backfill on 2025 reproduces the research numbers within the bootstrap interval |
| F11 | Footprint (only if its test passes): area model, then a drawn outline | Beats the median-area baseline on held-out blocks; otherwise not shipped |

Rough order and size: F1 to F4 about a week, F5 to F8 about a week, F9 and F10 about a week, F11 optional and separate.

## Decisions for the user

1. Ship bands only (recommended) or wait for the footprint test?
2. Where the static layers and the nightly greenness cache live (the backend host, or a private bucket under one organisation-owned project, as docs/08 recommends)?
3. Which states to serve first. Training covered the cases in docs/results.md; elsewhere the page should carry `validated: false`.

## Known limits to state on the page

- Wind plays no role in this ranking. In every test it added nothing.
- Roads and population files are from 2019 to 2021.
- The day-before-forecast check covered only 7 cases, so the effect of forecast error on skill is untested.
- The horizon is 24 to 48 h. Beyond that is untested, because the stored weather stopped at about 62 h.
- The ranking is limited to 8 km around the fire.

## Status (2026-11-04)

Built, uncommitted, all additive:
- Scoring job `scripts/34_score_active.py` (`--at` replay, `--live` for FIRMS near-real-time with the weather forecast), model `data/models/rank3_{24,48}h.joblib` (one boosted-tree model, no wind, no ensemble), skill numbers `data/live/skill.json`. Uses `src/firespread/{live,rank,ancillary}.py`; a replay of a past fire reproduces the research bundle's inputs.
- Backend `website/backend/app/routers/fire_spread.py` (`/api/fire-spread/runs/latest`, `/{run}/{event}/{24|48}`, `/{run}/{event}/pixels`), registered in `main.py`, 6 tests. Reads `website/backend/runtime_data/fire_spread/<run>/` and `skill.json` there.
- Frontend `features/firespread/` page at `/ops/fire-spread`, linked from the Satellite page, English/Hindi/Odia text, 2 unit tests; the production build succeeds.
- Run now: a live run (6 fires, 2026-11-04 08:17 UTC) and a replay (8 fires, 2025-03-12) are in the backend data folder; the page shows the newer one.

Not done: a schedule for the live job (every 3 hours after VIIRS passes; it needs the fire-spread environment, the OSM line file, the forest mask and the WorldCover tiles on whatever host runs it), a navigation entry (the control-room navigation is capped at five fixed items), the burned-area footprint (F11), and a browser check of the page.

## Status (2026-11-07): spread outlook with the gated U-Net

- The what-if simulator (wind and rate-of-spread inputs, ellipse engine, assets, packs; `features/scenarios/WhatIfPage.tsx`, `scenarios/whatif/`, its tests) is removed. `/app/whatif` and `/ops/fire-spread` redirect to the new page.
- New page **Spread outlook** (`/ops/outlook`, `features/outlook/OutlookPage.tsx`, nav item for the control room): pick a fire, read one sentence (direction, ground, slope), switch 24 h / 48 h, see three bands on the map, "show what actually burned" for archived fires, the fire's facts, and a trust box (measured overlap against nearest-first and three plain limits). No sliders, no wind arrows, no distance or speed claims.
- Model: the gated U-Net of the paper (seed 0; 24 h and 48 h). `fire-spread/scripts/62_outlook_run.py` writes a run folder; the served run is `replay-gated-20250425T2103` (7 archived 2025 test fires). The old tree-model runs and their skill file are in `website/backend/runtime_data/fire_spread_archive_tree_model/`.
- Backend: a run may carry its own `skill.json`; `GET /api/fire-spread/{run}/{event}/burned/{24|48}` serves the burned overlay of an archived fire.
- **Not done:** live scoring with the image model. It needs the 46 picture layers and 31 numbers built from live feeds (Sentinel-2, MODIS burn and vegetation, ERA5, weather forecast, OSM, WorldCover, forest mask) for each new fire; the tree-model job (`34_score_active.py`) builds a smaller feature set. Until then the page says it shows archived fires.
- Hindi and Odia strings are machine-written and need a native-speaker check.

- Update (same day): the run is scored by the mean of three seeds; the page has a **What is in the way** card (people and area per band from WorldPop; named places and roads from OpenStreetMap through Overpass, cached, with the January 2021 extract as fallback) and controls to hide bands, show places and roads, and zoom to a place. Live scoring: `fire-spread/docs/live-image-scoring-plan.md`.
- Bands (changed 2026-11-07): the cut-offs follow the usual burned share of the cells asked about at each horizon (2.18 % at 24 h, 2.93 % at 48 h): top band 1x, up to 3x, up to 7x that share. Before, they were fixed at 2, 6 and 15 % at both horizons, so the 24 h and 48 h maps looked the same.
