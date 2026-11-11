# Results so far (MODIS burn-date truth, 12-24 h horizon)

Source: `scripts/14_model_comparison.py cases bank`, log `data/14_final.log`. 834 usable cases (582 dev, 252 test), 132 blocks of 1 x 1 degree. Test blocks were scored once.
"Direction AUC" is AUC inside distance bands, so closeness to the fire cannot help. Intervals are block bootstrap.

| Model | Dev direction AUC | vs proximity | Test direction AUC | vs proximity |
|---|---|---|---|---|
| proximity | 0.593 | | 0.595 | |
| physics engine (US priors) | 0.558 | -0.035 [-0.060, -0.012] | 0.573 | -0.022 [-0.057, +0.012] |
| learned | 0.623 | +0.031 [+0.004, +0.056] | 0.616 | +0.021 [-0.028, +0.067] |
| learned + physics | 0.609 | +0.016 [-0.007, +0.039] | 0.629 | +0.034 [-0.013, +0.079] |

Dev ablations (direction AUC): without wind 0.639 (better than with it), without slope 0.616, without fuel/land cover 0.550 (worse than proximity), without weather 0.611, without season 0.625.

## Reading
- Fuel and land-cover continuity carry the signal. Removing them drops the model below proximity.
- Wind adds nothing detectable; the model is better without it. Slope and weather add a little at best.
- The physics engine with US-style priors is below proximity on dev (significant) and not better on test.
- Learned models beat proximity on dev for the learned model alone (interval excludes zero); on the frozen test the intervals include zero for all of them. The gain is small (about +0.02 to +0.03) and not yet shown to hold on new blocks.
- Overall AUC (0.94 to 0.95) is dominated by distance for every model.

## Not done
- Horizons beyond the epoch gap (24 to 72 h) with MODIS day-resolution truth.
- Forecast-weather error check (previous-day forecasts, 2024-25 only).
- Antecedent dryness features. 3 of 1,995 bank cases failed to download and are excluded.

# Common horizons (24 h and 48 h), MODIS truth

Source: `scripts/17_burn_long.py`, `scripts/18_horizons.py`, log `data/18_horizons.log`. Same 834 cases and split. Truth = burned within H hours of the start. Candidates within 8 km of the start. Weather only reaches about 62 h, so 72 h is not tested. Test blocks were looked at once per horizon.

| Horizon | Model | Test direction AUC | vs proximity |
|---|---|---|---|
| 24 h | proximity | 0.592 | |
| 24 h | learned | 0.637 | +0.045 [+0.004, +0.086] |
| 24 h | learned without wind | 0.660 | +0.068 [+0.036, +0.101] |
| 48 h | proximity | 0.591 | |
| 48 h | learned | 0.659 | +0.068 [+0.031, +0.105] |
| 48 h | learned without wind | 0.667 | +0.076 [+0.047, +0.106] |

Dev agrees (+0.04 to +0.06). With one common horizon the learned model beats proximity on the frozen test blocks at both horizons, and the gain is larger at 48 h. Wind still adds nothing: dropping it helps at both horizons.

# Weather source check (day-before forecast), 2024-25 test cases

`scripts/19_ranking_model.py`, log `data/19_final_model.log`. Model without wind, fitted on dev. 56 test cases from 2024-25, but only 7 of them had the day-before forecast covering the needed hours; the other 49 fell back to the archived stream. So this check is inconclusive: the 7 cases cannot show a difference. Direction AUC with "day-before" weather was 0.636 (24 h) and 0.645 (48 h) against 0.632 and 0.640 with archived weather, which only says nothing broke.
Final models saved: `data/models/rank_24h.joblib`, `rank_48h.joblib` (fitted on all 834 cases; features in the file).

# More features, and IoU (scripts 20-26)

New inputs, all downloaded or built for the 834 cases: antecedent rain and dryness (ERA5 daily, 60 days), soil moisture and vapour pressure deficit (ERA5 hourly), greenness (MODIS NDVI 250 m and its anomaly against a year earlier), roads/tracks/rivers (local OpenStreetMap, January 2021), population (WorldPop 1 km), wider landscape context, and a no-wind fuel-aware path distance. IoU is the overlap of the top-k cells with the burned cells, k = the number of burned cells (equal areas, so the cut-off is not the model's choice).

Dev, direction AUC gain over the no-wind model (script 21, `data/21_new_features.log`): at 24 h only greenness helped (+0.026 [+0.009, +0.046]); at 48 h greenness +0.015 [+0.001, +0.030] and all groups together +0.031 [+0.011, +0.049]; dryness, soil, landscape, path and roads/rivers/population did not help on their own. Kept: all groups.

Frozen test blocks, second look (script 26, `data/26_final_v2.log`; the first look was script 19, so these blocks have now been scored twice):

| Horizon | Model | Direction AUC | vs proximity | IoU |
|---|---|---|---|---|
| 24 h | proximity | 0.592 | | 0.106 |
| 24 h | no-wind model | 0.660 | +0.068 [+0.039, +0.100] | 0.128 |
| 24 h | all new features | 0.697 | +0.105 [+0.072, +0.137] | 0.143 (+0.036 [+0.019, +0.055]) |
| 48 h | proximity | 0.591 | | 0.124 |
| 48 h | no-wind model | 0.670 | +0.079 [+0.050, +0.108] | 0.159 |
| 48 h | all new features | 0.697 | +0.106 [+0.070, +0.141] | 0.171 (+0.047 [+0.022, +0.075]) |

IoU is low in absolute terms: the MODIS truth is 500 m and patchy, and this is an oracle-area IoU (the predicted area is set to the true area). A deployed map also needs to predict how large the burned area will be; that is not built.
Models saved: `data/models/rank2_24h.joblib`, `rank2_48h.joblib` (fitted on all 834 cases).
Limits: wind still adds nothing; roads/population are from 2019-2021 files; the day-before-forecast check covered only 7 cases.

# Held-out years and states (script 30, `data/30_holdouts.log`)

All 834 cases, all-features model without wind, nothing tuned. Gain over plain proximity, block bootstrap.
- Train 2019-2023, test 2024-2025 (173 cases): direction AUC +0.094 [+0.060, +0.127] (24 h), +0.104 [+0.070, +0.137] (48 h); IoU +0.020 [+0.002, +0.038] (24 h), +0.033 [+0.013, +0.051] (48 h).
- Leave one state out (12 states with at least 25 cases): direction gain above zero in all 12 at both horizons (interval above zero in 10); IoU gain above zero in 11 (interval above zero in 7 at 24 h, 8 at 48 h). Weak: Telangana (direction interval includes zero), Chhattisgarh (no IoU gain, -0.004 and -0.003).
- Terrain filled in for all cases (script 28) before these runs; dev data-size check (script 29): 24 h still improving with more cases, 48 h flat from 50% to 100%.
- Scale of IoU: random 0.013, truth shifted 1 km 0.12, shifted 300 m 0.43 (24 h, 120 cases). Published next-day work reports F1 about 0.29-0.42 on other benchmarks; not comparable, see the discussion in the session.

# Third look, terrain filled in for every case (script 26 with TAG=rank3, `data/26_final_v3.log`)

The frozen test blocks have now been scored three times (scripts 19, 26, 26 again after the terrain fix). Final models: `data/models/rank3_24h.joblib`, `rank3_48h.joblib`, fitted on all 834 cases.

| Horizon | Model | Direction AUC | vs proximity | IoU |
|---|---|---|---|---|
| 24 h | proximity | 0.592 | | 0.106 |
| 24 h | all features | 0.704 | +0.112 [+0.077, +0.148] | 0.137 (+0.030 [+0.013, +0.048]) |
| 48 h | proximity | 0.591 | | 0.124 |
| 48 h | all features | 0.705 | +0.114 [+0.078, +0.152] | 0.168 (+0.044 [+0.020, +0.070]) |

Dashboard files for the Fire spread page come from `scripts/34_score_active.py` (replay mode) and `data/live/skill.json`.

# U-Net against boosted trees (script 32 on a GPU VM, `data/log_24.txt`, `data/log_48.txt`)

A small U-Net (4 levels, 25 epochs, random turns and flips) on the same 47-channel maps, same dev folds (582 cases, 5-fold by block), scored like everything else. The "gbm" row is one HistGradientBoosting model on the same cells.

| Horizon | Model | Direction AUC | IoU at the true area |
|---|---|---|---|
| 24 h | proximity | 0.590 | 0.107 |
| 24 h | boosted trees | 0.680 [0.658, 0.701] | 0.128 |
| 24 h | U-Net | 0.696 [0.669, 0.722] | 0.147 |
| 24 h | average of both | 0.696 | 0.139 |
| 48 h | proximity | 0.586 | 0.116 |
| 48 h | boosted trees | 0.685 [0.661, 0.707] | 0.149 |
| 48 h | U-Net | 0.693 [0.665, 0.719] | 0.147 |
| 48 h | average of both | 0.695 | 0.151 |

Reading: the U-Net is slightly ahead at 24 h (+0.016 direction AUC, +0.019 IoU) and tied at 48 h; the intervals overlap, and per-case paired differences were not saved. Both clearly beat proximity. Dev only; the frozen test blocks were not used.

# One outlook, not two horizons (scripts 63 and 64, 2026-11-11)

The 24 h and 48 h models and labels are nearly the same, so the dashboard shows one outlook ("next 1 to 2 days").

- Truth. Over all 2,930 cases the cells that burned within 24 h are always a subset of those within 48 h; the two sets are identical in 45% of the cases, and the 48 h set adds under 10% more cells in 59%. Median Jaccard overlap 0.96. Burned cells: 1.57 million at 24 h, 2.10 million at 48 h.
- Models (three seeds each, scored on the 250 test and 205 blind-state fires, truth = burned within 48 h). Median rank correlation of the two score maps 0.92 (test) and 0.89 (blind). Direction AUC on test: 24 h model 0.805, 48 h model 0.808, average of both 0.812; on blind states 0.833, 0.806, 0.826. IoU on test 0.249, 0.240, 0.248; blind 0.256, 0.224, 0.247. On blind states the 24 h model is better than the 48 h model at the 48 h task (direction AUC +0.027, interval [0.009, 0.049]).
- Timing. Among cells that burned within 48 h, the 48 h minus 24 h ranking cannot tell the ones that burned after 24 h from those before (AUC 0.47 test, 0.48 blind); plain distance from the fire does better (0.66 and 0.61). The models rank where, not when.
- The MODIS burned-area product dates burning to the day, which probably limits any separation of the two horizons; its date error was not measured here.
- Served outlook: mean of the two ranks (six models), scored against the 48 h truth: direction AUC 0.812 against 0.599 for nearest-first, IoU 0.248 against 0.123 (test, 250 fires). Test and blind fires had been scored before, so this is a further look at them.
