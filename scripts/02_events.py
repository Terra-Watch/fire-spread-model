"""Build the fire-event bank from the VIIRS files already on disk: events, observation epochs and forecast cases, forest detections only.
Output (data/): detections/{year}.parquet, events.parquet, cases.parquet, events_summary.json.   Run: .venv/bin/python scripts/02_events.py [first_year last_year]"""
import json
import sys
import time
from pathlib import Path


import numpy as np
import pandas as pd
import shapely
from shapely.geometry import shape

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from firespread import config, events, forest, viirs  # noqa: E402


def load_states():
    gj = json.load(open(config.ADM1))
    geoms = [shape(f["geometry"]) for f in gj["features"]]
    names = [f["properties"].get("shapeName") for f in gj["features"]]
    return shapely.STRtree(geoms), names


def state_of(lat, lon, tree, names):
    pts = shapely.points(np.asarray(lon, float), np.asarray(lat, float))
    out = np.full(len(pts), "", dtype=object)
    idx = tree.query(pts, predicate="within")
    for i, g in zip(idx[0], idx[1]):
        out[i] = names[g]
    return out


def main(first, last):
    out = config.DATA
    (out / "detections").mkdir(parents=True, exist_ok=True)
    mask = forest.ForestMask()
    tree, names = load_states()
    all_events, all_cases, summary = [], [], {}
    for year in range(first, last + 1):
        t = time.time()
        d = viirs.load_year(year)
        share = mask.share(d["latitude"].values, d["longitude"].values)
        f = d[share >= config.MIN_FOREST_SHARE].reset_index(drop=True)
        f["forest_share"] = share[share >= config.MIN_FOREST_SHARE]
        f["event"] = [f"{year}-{k:06d}" for k in events.link_events(f)]
        f["epoch"] = events.assign_epochs(f)
        ev = events.event_table(f)
        ca = events.case_table(f)
        ev["state"] = state_of(ev["lat"], ev["lon"], tree, names)
        ca["state"] = ca["event"].map(dict(zip(ev["event"], ev["state"])))
        f.to_parquet(out / "detections" / f"{year}.parquet", index=False)
        all_events.append(ev)
        all_cases.append(ca)
        summary[year] = {"detections_all": int(len(d)), "detections_forest": int(len(f)), "events": int(len(ev)), "events_2plus_epochs": int((ev.n_epochs >= 2).sum()),
                         "cases": int(len(ca)), "cases_with_growth": int((ca.n_new >= 2).sum()), "seconds": round(time.time() - t, 1)}
        print(year, summary[year], flush=True)
    pd.concat(all_events, ignore_index=True).to_parquet(out / "events.parquet", index=False)
    pd.concat(all_cases, ignore_index=True).to_parquet(out / "cases.parquet", index=False)
    json.dump(summary, open(out / "events_summary.json", "w"), indent=1)


if __name__ == "__main__":
    a = sys.argv[1:]
    main(int(a[0]) if a else 2019, int(a[1]) if len(a) > 1 else 2025)
