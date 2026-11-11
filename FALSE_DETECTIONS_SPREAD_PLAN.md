# False detections in the fire-spread work: design and plan

Status: design (2026-11-05). Shared design: `docs/16-detection-quality-design.md` (one detection-quality service with a score `p_real` and reason flags). Forecasting side: `wildfire-dataset/FALSE_DETECTIONS_PLAN.md`. Dashboard layer: `docs/15-known-heat-sources-plan.md`.

## 1. Where false detections enter the spread work

Reading the code, in order of how much harm each does:

1. **A case is a cluster of detections** (`scripts/02_events.py`: linked at 1 km and 36 h, consecutive passes). A cluster made of kiln or mine pixels is a fire that never existed. It also happens that one real fire has an industrial pixel inside its cluster, which stretches the starting footprint.
2. **The starting footprint** (`seed`, `dist`, the cells asked about) is built from those detections when the case bundle is made (`scripts/04_environment.py`).
3. **Detection-derived inputs**, computed at export time (`src/firespread/dataset.py`): `fire_hist_90`, `fire_hist_365`, `fire_days_since`, `other_fires_72h`, `det_frp`, `det_age_h`, and the numbers `n_start`, `frp_sum_log`, `extent_km`, `act5`, `act20`, `act_ev20`. They count every detection, false or not.
4. **The live trigger**: the dashboard job starts from the latest near-real-time detections, which carry no vegetation or static-source class.

What already protects us: the truth is independent of the detections (MODIS burned-area dates), a case is used only when at least 5 burned cells touch it (about 40% of selected cases pass), detections must sit in forest land, and a case needs at least 3 start detections. That removes most false fires from training and scoring. It does not clean the inputs, it favours fires big enough to leave a scar (a selection effect), and it does nothing for live.

## 1b. Forest-boundary check of the spread cases (done 2026-11-05, `scripts/58_boundary_audit.py`, 834 usable cases)

The spread data uses the same legal-forest union as the forecasting labels (see `wildfire-dataset/FALSE_DETECTIONS_PLAN.md`, section 1a). Results:
- **Starts are on forest land.** Mean forest share under the start footprint 0.98; one case in 834 is below 0.5, 0.8% are below 0.8. Under the start: 75% tree, 92% tree, shrub or grass, 2.4% crop, 0.02% built; 6 cases (0.7%) start mostly on crop.
- **Industry is almost absent from the usable cases.** 1 case within 1 km of an OpenStreetMap industrial, quarry, mining or power feature, 5 within 2 km (0.6%), against 1.6% and 2.4% for the forecasting label detections. Industrial heat rarely leaves a burned patch, and the burned-patch test removes it.
- **The spread leaves the boundary.** 94.5% of the burned truth cells lie inside the forest boundary, 5.5% outside (2.4% in crop-dominant cells); only 70% of the cells asked about (within 8 km of the start) are inside it.
- **The boundary is leaky in some states.** Burned cells inside the boundary: Mizoram 55%, Himachal Pradesh 76%, against 88% to 95% in Uttarakhand, Jharkhand, Karnataka, West Bengal. This matches the audit's warning that North-East layers are thin, and it fits shifting cultivation (burning of fields outside forest land) in the North-East.

What this changes: (1) report the North-East states (Mizoram, Manipur, Tripura, Arunachal Pradesh, Assam, Sikkim) and Himachal Pradesh separately in the results, with a note that fires there include agricultural burning; (2) treat the 6 crop-start cases and any case with start forest share below 0.8 as flagged (kept, listed in the metadata); (3) no new rule is needed for industry in the spread cases, only for the live path, where no burned patch exists yet (D1 and D7 above).

## 2. Design decisions, with the reason for each

**D1. Admission by `p_real`, on top of the burned-patch test.** A case is admitted if at least 3 of its start detections have `p_real` at or above the `hide` threshold. The burned-patch test stays. Reason: the patch test removes false fires only when MODIS sees them; the score removes the rest and also protects the live path, where no patch exists yet.

**D2. Mixed starts are dropped, not repaired.** If more than 30% of a case's start detections are flagged as sources, the case is left out. Reason: repairing the footprint means rebuilding the bundle (`seed`, `dist`, cells asked about, connected truth all depend on it) for a small number of cases. We count the cases affected first; if they are more than 5%, we rebuild the start from clean detections at export instead.

**D3. History layers and numbers are rebuilt from clean detections, past only.** The seven layers and the numbers in section 1.3 use detections with `p_real` at or above `hide` and a time before the start, using the service version as of that time. Reason: a model must not be fed false history, and must not see the future. The old layers stay in the dataset under their old version for comparison.

**D4. One new static layer, `dist_heat_source`**: distance to the nearest known heat source (industrial, mining, power, flare, seasonal persistent pixel). Reason: a model can then learn that a fire next to a plant looks different, and the layer carries no future information.

**D5. Hard admission is primary; `p_real` as a case weight is a sensitivity run.** Reason: with two models and a limited budget, one clean definition is easier to defend; the weighted run is done only for the winner.

**D6. Selection effect is said, measured, and bounded.** The set favours fires that leave a scar. The data card says so. After training, skill is reported by `p_real` bin, by fire size at the start (3 to 5, 6 to 10, 11 to 30, over 30 detections), and for the secondary truth (later detections of the same fire) where no burned patch exists. If the gain over nearest-first disappears in the weak bins, the live page gates on a higher threshold.

**D7. Live gate and wording.** The live job scores a fire only if its start passes D1 and it has at least 3 detections; the page shows "signal confidence" (high, medium, low) from `p_real`. Fires near a known heat source stay dimmed until later passes spread outside the buffer (see `docs/15`).

## 3. How it reaches the data without redoing the download

The six builders on the VMs are not stopped. The slow part (burn dates, greenness, Sentinel-2, burn history, terrain) does not depend on detections. Only the detection-derived layers do, and they are computed at export time from the detection tables. So:
- `scripts/54b_reexport_detection_layers.py` (new): reads the finished parts, replaces the layers and numbers in section 1.3, adds `dist_heat_source` and `p_real_start`, and writes a new version of each part (`dataset/part_<tag>_v2/`), leaving v1 intact.
- Cases dropped by D1 or D2 get `admitted_v2 = False` in the metadata; nothing is deleted.
- The dataset audit (`scripts/55_audit_part.py`) is extended with the new layers, the share of cases dropped per state, and the same invariants (no detection after the start is used; a case admitted in v2 is also usable in v1).

## 4. Order and dependencies

| Step | Needs | Time |
|---|---|---|
| a. measurements and the score (steps 1 to 6 of `docs/16`) | forecasting side's work | about 1 week |
| b. count cases affected by D1 and D2 on the current parts | finished parts | 1 day |
| c. re-export v2 | step a and the finished parts | 1 day, CPU |
| d. screening runs of the two image models | v1 is enough (the changed layers are 14 of 48) | GPU, as planned |
| e. final runs | v2 | GPU, as planned |
| f. skill by `p_real` bin and size; live gate | step e | 1 day |

## 5. Acceptance
- The v2 export passes every invariant test and the audit.
- Fewer than 5% of admitted v1 cases change status because of D2 (otherwise rebuild the start).
- Skill by bin is reported, and the live gate is set from it.
- The data card lists what was removed and why.

## 6. Not decided here
The score itself, its thresholds and the field-verified labels belong to the shared design. If the forest departments' verified lists arrive, the thresholds are reset on them and this plan is re-run from step c.
