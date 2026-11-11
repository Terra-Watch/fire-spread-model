# Training sample and evaluation protocol for the image models (fixed 2026-11-05, before any image model is trained)

Code: `scripts/42_sample_train.py` (draw), `scripts/60_train.py --sample data/sample_train.parquet [--train-frac f]` (use). Files: `data/sample_plan.json` (rule, counts, hash), `data/sample_train.parquet` (every case with its pool and keep flag).

## Population
The 8,911 selected cases of `data/big_cases.parquet` (28 states, 2019-01-02 to 2025-06-21). Selection rules (VIIRS events linked at 1 km and 36 h, at least 3 start detections, start in forest-boundary land, consecutive epoch pairs) do not depend on any model. A case is *usable* for training or scoring when MODIS shows at least 5 burned cells touching the fire and at least 50 candidate cells remain (`usable_24`, `usable_48`). About 35% of built cases pass (692 of 1,992 measured on 2026-11-05).

## Pools (by label year, no mixing across time)
| Pool | Cases | Use |
|---|---|---|
| train: 17 v4 states, label year <= 2023 | 5,473 selected (about 1,900 usable) | fitting; sampled by the rule below |
| validation: same states, 2024 | 821 | early stopping, learning rate; never thinned |
| test: same states, 2025 (to 21 June) | 986 | scored once per frozen model; never thinned |
| blind: Uttar Pradesh, West Bengal, Kerala | 617 | scored once; never thinned |
| outside: the other 8 states | 1,014 | separate check; never thinned |

## Sampling rule (training pool only)
A fire event is kept when `hash(seed|event) < p_state`, with `p_state = min(1, 200 / (0.35 x selected training cases of the state))`, seed 20261105. All cases of a kept event stay together. The cap of 200 usable cases per state is a balance rule: it trims only Odisha, Madhya Pradesh and Chhattisgarh slightly, so the training set stays about 1,870 usable cases (5,334 of 5,473 selected). It is not a large reduction, because the data is not large (about 7 GB of training images). Sikkim has 2 training cases and is evaluated only.
Hash of the kept case list: see `data/sample_plan.json` (`kept_case_list_sha256`). Changing the rule, seed or list after a score is seen is an amendment, written and dated here.

## Why not smaller
The development learning curve (script 29) was still rising with more cases at 24 h and flat from 50% to 100% at 48 h. So 24 h trains on the full sample; 48 h may train on half (`--train-frac 0.5`, nested subsets by a second hash). The learning curve for the final models is run at 25%, 50% and 100% and reported.

## Test discipline
- Validation and test are never sampled. Every usable case in them is scored, so results describe the whole population.
- The 2025 test blocks and the blind states are scored once per frozen model (settings and epoch count fixed from validation 2024).
- Reported against nearest-first: direction AUC, IoU at the true area, hits in the top 200, with block (1 deg) bootstrap and paired differences. Per state, with the North-East states and Himachal Pradesh shown separately (the forest boundary is leaky there).
- Limits stated up front: 2020 has few cases (503 selected); the 2025 test ends 21 June; MODIS-defined truth misses small burns.
