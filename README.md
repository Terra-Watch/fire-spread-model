# fire-spread

Research code for fire spread: the observed fires and their surroundings, the spread dataset and models, and the check of the spread outlook against what burned. Its results feed the Past fires page of the dashboard (`website/`). The dashboard's Active fires board does not use this code: it is built from the satellite detections in the dashboard's own database (`docs/23-active-fires-redesign.md`).

| Path | Content |
|---|---|
| `src/firespread/` | the library: config, viirs, forest, events, grid, rasters, weather, moisture, engine, cases, scoring, rank, imgmodels, live and more |
| `scripts/` | numbered stages `02` to `65`, run in order (events, pilot, environment, evaluation, learned models, dataset export and audit, training, outlook); `scripts/cloud/` holds the cloud jobs |
| `tests/` | unit tests: events, grid, weather, scoring, engine, dataset, live |
| `notebooks/` | the two explanatory notebooks |
| `docs/` | `dataset.md`, `results.md`, `sampling-protocol.md`, `dashboard-plan.md` and the other plans |
| `data/`, `release/` | everything the scripts make (not tracked) |

Own environment in `.venv/` (numpy, pandas, scipy, rasterio, pyproj, shapely, numba, httpx, pytest). `make check` compiles every file and `make test` runs the tests.

## Data (what exists and where it comes from)

| Layer | Source | Where |
|---|---|---|
| Fire detections | NASA FIRMS VIIRS 375 m, NOAA-20 and Suomi-NPP, 2019 to 2025, vegetation fires, nominal/high confidence | read in place from `terrawatch-datasets/inputs_raw/firms_*` |
| Forest land | boundary-defined forest mask, 0.001 degree | `terrawatch-datasets/dataset/forest_union_0p001.npy` |
| Land cover | ESA WorldCover 10 m, 2021 | `terrawatch-datasets/_tiles/` |
| Terrain | Copernicus DEM 30 m (public AWS bucket, read per case over HTTP) | not stored whole |
| Weather | Open-Meteo hourly, three views: archived forecast stream, day-before forecast, ERA5 | `data/weather_cache/` |

`data/` after the scripts: `detections/{year}.parquet` (forest detections with event and epoch), `events.parquet`, `cases.parquet`, `pilot_cases.parquet`, `cases/<id>/{env.npz, weather.csv, meta.json}`, `QA.md`.

## Order of work

1. `02_events.py` builds fire events, observation epochs and forecast cases from the VIIRS files (about 7 minutes for 2019 to 2025).
2. `03_pilot.py` picks the pilot cases and freezes the dev/test split by 1 x 1 degree block.
3. `04_environment.py` writes one bundle per pilot case: terrain, land cover, forest share, the start footprint and the truth in grid cells, hourly weather. Idempotent.
4. `05_qa.py` checks the bundles and writes `data/QA.md`.
5. `06_evaluate.py` runs the engine and the baselines and compares them. Dev cases may be used to tune; test cases only to report.
6. `07_signal.py` asks the plain question first: do the satellite-seen growth directions follow the wind or the slope at all?

## Rules kept on purpose

- A case is two consecutive satellite observations of one fire. The forecast starts from the first, the truth is what the second shows as new growth (farther than 0.75 km from everything seen before).
- Growth is scored as a ranking question. A cell not seen burning is not known to be unburnt (small fires are missed, cloud hides), so calibrated probabilities are not asked of this truth.
- Two ways of looking: overall AUC (dominated by "close to the fire grows") and direction AUC inside distance bands (what wind, slope and fuel add beyond distance). Baselines are proximity and a wind ellipse.
- Uncertainty is by 1 x 1 degree block, never by single fires, because neighbouring fires share weather and terrain.
- The engine's fuel numbers are priors, not measurements; no Indian fuel model exists in this project.

## The outlook for the dashboard (Past fires)

The served outlook is the mean of the 24 h and 48 h gated U-Net models (three seeds each), ranked into three bands within 8 km of a fire. The 24 h and 48 h models rank cells almost identically, so there is one outlook. It does not use wind. Live scoring with it is not built (`docs/live-image-scoring-plan.md`), so the dashboard shows archived 2025 fires.

```
.venv/bin/python scripts/65_outlook_check.py --release release/fire_spread_dataset --models ../paper_fire_spread/results/models        # every test fire against what burned -> data/65_outlook_check.csv
.venv/bin/python scripts/62_outlook_run.py --release release/fire_spread_dataset --models ../paper_fire_spread/results/models \
    --cases release/fire_spread_dataset/cases.csv --out ../website/backend/runtime_data/fire_spread --only CASE_ID,CASE_ID,...     # the chosen example fires
```
`62` writes a run folder (`events.json`, bands, burned area, places and roads, `skill.json`); copy it to `website/backend/app/fire_spread_data/` to ship it. Results: `docs/results.md` (last section: the outlook fire by fire). Older scripts (`34_score_active.py` and the tree models) are the earlier approach and are not used by the dashboard.
