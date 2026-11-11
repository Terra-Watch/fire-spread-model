"""Forest-boundary and industrial-source check of the fire-spread cases (the same questions as c01b and c02 in wildfire-dataset, asked of the cases): is the start of the fire really on forest land, what is under the start footprint, how close is industry, and how much of the burned truth lies outside the forest boundary?
Uses data/maps_48h.npz (usable cases) and the industrial source list from wildfire-dataset/audit/false_detections.   Run: .venv/bin/python scripts/58_boundary_audit.py"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import shapely
from shapely.geometry import shape

ROOT = Path(__file__).resolve().parents[1]; V4 = ROOT.parent / "wildfire-dataset"


def main():
    z = np.load(ROOT / "data" / "maps_48h.npz"); names = list(z["names"]); n = len(z["case"])
    L = lambda k: np.asarray(z["X"][:, names.index(k)], np.float32)
    seed = L("seed") > 0; fs, tree, shrub, grass, crop, built, water = (L(k) for k in ("forest_share", "tree", "shrub", "grass", "crop", "built", "water"))
    y = z["y"] > 0; cand = z["cand"] > 0
    rows = []
    for i in range(n):
        s = seed[i]
        veg = (tree[i][s] + shrub[i][s] + grass[i][s]).mean(); nonveg = (crop[i][s] + built[i][s] + water[i][s]).mean()
        t = y[i]
        rows.append({"case": z["case"][i], "state": z["state"][i], "year": int(z["year"][i]), "seed_forest_share": float(fs[i][s].mean()), "seed_tree": float(tree[i][s].mean()), "seed_veg": float(veg), "seed_crop": float(crop[i][s].mean()), "seed_built": float(built[i][s].mean()),
                     "seed_nonveg": float(nonveg), "truth_cells": int(t.sum()), "truth_in_forest_share": float((fs[i][t] >= 0.5).mean()) if t.any() else np.nan, "truth_in_crop_share": float((crop[i][t] >= 0.5).mean()) if t.any() else np.nan,
                     "cand_forest_share": float((fs[i][cand[i]] >= 0.5).mean())})
    df = pd.DataFrame(rows)
    # industrial proximity of the case centre
    geoms = []
    with open(V4 / "audit" / "false_detections" / "industrial_sources.geojsonseq") as f:
        for row in f:
            ft = json.loads(row.lstrip("\x1e"))
            if ft["geometry"]["type"] in ("Polygon", "MultiPolygon", "Point") and ft["properties"]:
                geoms.append(shape(ft["geometry"]))
    tree_ = shapely.STRtree(geoms); pts = []
    for c in df.case:
        d = next(p for p in (ROOT / "data" / "bank" / c, ROOT / "data" / "cases" / c) if p.exists()); m = json.load(open(d / "meta.json")); pts.append((m["lon"], m["lat"]))
    P = shapely.points(np.array(pts))
    for km in (1, 2):
        i, _ = tree_.query(P, predicate="dwithin", distance=km / 111.0); hit = np.zeros(len(df), bool); hit[np.unique(i)] = True; df[f"industrial_{km}km"] = hit
    out = {"cases": int(n), "start_footprint_mean_forest_share": {"mean": float(df.seed_forest_share.mean()), "share_of_cases_below_0.5": float((df.seed_forest_share < 0.5).mean()), "below_0.8": float((df.seed_forest_share < 0.8).mean())},
           "start_footprint_land_cover": {"tree": float(df.seed_tree.mean()), "vegetation_tree_shrub_grass": float(df.seed_veg.mean()), "crop": float(df.seed_crop.mean()), "built": float(df.seed_built.mean())},
           "cases_with_start_mostly_non_vegetated_over_0.5": float((df.seed_nonveg > 0.5).mean()), "cases_with_start_mostly_crop_over_0.5": float((df.seed_crop > 0.5).mean()), "cases_with_start_mostly_built_over_0.2": float((df.seed_built > 0.2).mean()),
           "industrial_within_1km": float(df.industrial_1km.mean()), "industrial_within_2km": float(df.industrial_2km.mean()),
           "burned_truth_inside_forest_boundary_share": float(df.truth_in_forest_share.mean()), "burned_truth_in_crop_dominant_cells": float(df.truth_in_crop_share.mean()), "asked_about_cells_inside_forest_boundary": float(df.cand_forest_share.mean())}
    by = df.groupby("state").agg(n=("case", "size"), seed_forest=("seed_forest_share", "mean"), crop=("seed_crop", "mean"), nonveg=("seed_nonveg", "mean"), industrial_1km=("industrial_1km", "mean"), truth_in_forest=("truth_in_forest_share", "mean")).round(3)
    out["by_state_lowest_start_forest_share"] = by.sort_values("seed_forest").head(8).reset_index().to_dict("records")
    df.to_parquet(ROOT / "data" / "boundary_audit_cases.parquet"); json.dump(out, open(ROOT / "data" / "boundary_audit.json", "w"), indent=1, default=str)
    print(json.dumps({k: v for k, v in out.items() if k != "by_state_lowest_start_forest_share"}, indent=1)); print(by.sort_values("seed_forest").head(8).to_string())


if __name__ == "__main__":
    main()
