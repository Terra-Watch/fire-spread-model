# Live scoring with the gated U-Net: plan

Status 2026-11-07: not built. The dashboard shows archived fires scored by the model (`scripts/62_outlook_run.py`). This file says what a live run needs.

## What the model needs for one fire
46 picture layers of 192 x 192 cells (100 m) and 31 numbers, built exactly as for the data set, using only what is known at the first look (t0). The model file stores the order of the layers (`sp_names`, `sc_names`) and the normalisation.

## Where each input comes from, and how it is built today

| Group | Layers | Built today by | Live source | Weight |
|---|---|---|---|---|
| Start footprint and detections | seed, det_frp, det_age_h, dist, fire history (4), n_start, frp_sum_log, extent_km, act5/20/ev20 | `dataset.py` (history_layers, scalar_row), detections filtered by the quality score | FIRMS NRT through `website/forecast/live` (already filtered, p_scar) | light |
| Land cover and forest | WorldCover shares, forest_share/5/15, tree_5/15, non-burnable distances, built/crop/water distances | `04_environment.py`, `dataset.py` | WorldCover tiles and the forest mask, cached once per state | light after caching |
| Terrain | elev, rel_elev, tpi, slope, slope_along | `04_environment.py` | DEM tiles cached once | light after caching |
| Vegetation index | ndvi, ndvi_prev, ndvi_anom | `22_ndvi.py` | MODIS MOD13Q1 composite, refreshed every 16 days, cached per tile | medium |
| Sentinel-2 | nir, swir1, swir2, ndmi, nbr, age, valid | `52_sentinel2.py`, `57_fill_gaps.py` | Planetary Computer, newest clear scene in 45 days (needs SAS signing, retries, baseline offset) | heavy, minutes per fire |
| Burn history | burn_days_since, burn_n | `53_burn_history.py`, `17_burn_long.py` | MODIS MCD64A1 monthly, cached per tile | medium |
| People and access | road_dist, river_dist, pop, path_t | `24_osm.py`, `25_pop.py`, engine | OSM extract, WorldPop, cached | light after caching |
| Weather numbers | 19 weather numbers, rain, soil moisture | `13_bank.py`, `20_antecedent.py`, `23_soil.py` | Open-Meteo forecast and ERA5 (`website/forecast/live` already fetches both) | light |

## Order of work
1. **Parity test first.** Take 20 released cases, rebuild them with the live builder from their t0 and detections, and compare every layer with the released arrays (max difference per layer). The live builder is accepted only when the model's scores agree (rank correlation above 0.99 and the same top band). This is the same golden check the forecast job uses.
2. **One function per group** with the same names as the model's layers; static groups read from a cache folder (WorldCover, DEM, OSM, WorldPop, forest mask) built once per served state.
3. **Slow groups on a schedule, not per request:** NDVI and burn history refreshed on their own cadence; Sentinel-2 fetched per fire with a timeout, and the layer flagged `s2_valid = 0` when no clear scene arrives (the model was trained with that flag; the ablation shows the cost is small: direction AUC about 0.78 without Sentinel-2 against 0.79).
4. **Job:** after each FIRMS refresh, for each fire with at least 3 quality-passing detections: build the 46 + 31, score 24 h and 48 h with the three seeds, write the run folder as `62_outlook_run.py` does (mode `live`), and set `degraded` when a group is missing.
5. **Host:** a CPU machine (no GPU needed; one fire is a few seconds of model time). Static caches need about 20 GB; Sentinel-2 and MODIS reads need outbound internet.

## Size and risk
- Estimate: 1 to 2 weeks for the builder and parity test, because Sentinel-2 and MODIS reads need the retry and signing code from the dataset repair, and because the order and units of all 77 inputs must match.
- Risk: a silent mismatch in one layer makes scores look plausible and wrong. The parity test is the control; do not skip it.
- Fallback if the builder is not finished: keep the page on archived fires (as now) and keep the tree-model job out of the dashboard, since it is a weaker model.

## Decisions for the owner
1. Host for the job (which machine, which account) and a budget for outbound reads.
2. Which states to serve first (static caches are per state).
3. Whether a fire without a Sentinel-2 scene may still be scored (recommended: yes, flagged).
