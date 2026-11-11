"""Writes the dataset of one share of the cases (or of everything on this machine): data/dataset/part_<tag>/
  spatial.npy        float16 [N, 48 layers, 192, 192]   (names in names.json)
  scalars_24.npy, scalars_48.npy   float32 [N, 38]       per-fire numbers for the 24 h and 48 h horizons
  labels_24.npz, labels_48.npz     y_modis, y_modis_all, y_viirs_new, y_viirs_any (uint8), burn_rel_day (int8)  [N, 192, 192]
  shared.npz         cand, modis_valid (uint8), dist (float16) [N, 192, 192]
  meta.parquet       one row per case: ids, place, time, split, fold, georeference of the crop, quality flags
  qa.json            missing share, range and mean of every layer; label rates
Run: .venv/bin/python scripts/54_export_dataset.py TAG [limit]"""
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from firespread import cases, config, dataset  # noqa: E402

CROP = dataset.CROP
HORIZONS = (24, 48)


def window(seed_mask, shape):
    r, c = np.argwhere(seed_mask).mean(axis=0)
    r0, c0 = int(np.clip(round(r) - CROP // 2, 0, shape[0] - CROP)), int(np.clip(round(c) - CROP // 2, 0, shape[1] - CROP))
    return r0, c0


def main(tag, limit=None):
    out = config.DATA / "dataset" / f"part_{tag}"
    out.mkdir(parents=True, exist_ok=True)
    ant = pd.read_parquet(config.DATA / "antecedent.parquet").set_index("case").join(pd.read_parquet(config.DATA / "soil.parquet").set_index("case"), how="inner")
    dirs = [p.parent for root in ("cases", "bank") for p in sorted((config.DATA / root).glob("*/burn_long.npz"))]
    if limit:
        dirs = dirs[:limit]
    M = len(dirs); Cs = len(dataset.SPATIAL)
    spatial = np.lib.format.open_memmap(out / "spatial_tmp.npy", mode="w+", dtype=np.float16, shape=(M, Cs, CROP, CROP))
    scal = {H: np.zeros((M, len(dataset.SCALARS)), np.float32) for H in HORIZONS}
    lab = {H: {k: np.zeros((M, CROP, CROP), np.int8 if k == "burn_rel_day" else np.uint8) for k in dataset.LABELS} for H in HORIZONS}
    shared = {"cand": np.zeros((M, CROP, CROP), np.uint8), "modis_valid": np.zeros((M, CROP, CROP), np.uint8), "dist": np.zeros((M, CROP, CROP), np.float16)}
    meta, n, skipped, t_start = [], 0, 0, time.time()
    dets = {}
    for d in dirs:
        try:
            m = json.load(open(d / "meta.json"))
            if m["case_id"] not in ant.index:
                skipped += 1; continue
            c = {H: cases.load_case(d, gap_override=H) for H in HORIZONS}
        except cases.NoWeather:
            skipped += 1; continue
        year = int(m["year"])
        if year not in dets:
            dets = {year: dataset.load_detections(m)}
        det = dets[year]
        event, ef = m["case_id"].rsplit("_", 1)
        row = pd.Series({"epoch_from": int(ef)})
        fire = det[(det.event == m["event"]) & (det.epoch <= int(ef))]
        if fire.empty:
            skipped += 1; continue
        a = ant.loc[m["case_id"]]
        c[48].extra["recent"] = c[24].extra["recent"] = det[(det.t >= pd.Timestamp(m["t0"]) - pd.Timedelta(hours=72)) & (det.t < pd.Timestamp(m["t0"]))][["latitude", "longitude", "t", "event"]]
        L = dataset.spatial_layers(c[48], d, a, det, m, fire)
        r0, c0 = window(c[48].seed_mask, c[48].fuel.shape); sl = (slice(r0, r0 + CROP), slice(c0, c0 + CROP))
        spatial[n] = np.stack([L[k][sl] for k in dataset.SPATIAL]).astype(np.float16)
        wx = pd.read_csv(d / "weather.csv"); src = c[48].weather_source
        gust = lambda H: float(wx[(wx.source == src) & wx.hours_from_t0.between(0, H)].wind_gusts_10m.max())
        flags = {}
        for H in HORIZONS:
            scal[H][n] = dataset.scalar_row(c[H], a, fire, m, H, gust(H))
            lb = dataset.labels(c[H], d, det, m, row, H)
            for k in dataset.LABELS:
                lab[H][k][n] = lb[k][sl]
            flags[f"truth_n_{H}"] = int(lb["y_modis"].sum()); flags[f"usable_{H}"] = bool(lb["y_modis"].sum() >= 5 and (lb["cand"] & ~lb["y_modis_all"]).sum() >= 50)
        shared["cand"][n], shared["modis_valid"][n], shared["dist"][n] = lb["cand"][sl], lb["modis_valid"][sl], c[48].dist_cells[sl]
        blk = f"{int(np.floor(m['lat']))}_{int(np.floor(m['lon']))}"
        sd = L["seed"] > 0                                               # what lies under the start footprint (boundary and land-cover check, see FALSE_DETECTIONS_SPREAD_PLAN.md)
        st = (m.get("state") or "?").lower()
        start = {"start_forest_share": float(L["forest_share"][sd].mean()), "start_crop": float(L["crop"][sd].mean()), "start_built": float(L["built"][sd].mean()), "start_veg": float((L["tree"] + L["shrub"] + L["grass"])[sd].mean()),
                 "region_group": "north_east" if any(k in st for k in ("arunachal", "assam", "manipur", "mizoram", "meghalaya", "nagaland", "sikkim", "tripura")) else ("himachal" if "himachal" in st else "other")}
        start["flag_crop_start"] = start["start_crop"] > 0.5; start["flag_low_forest_start"] = start["start_forest_share"] < 0.8
        env = np.load(d / "env.npz")
        meta.append({"case": m["case_id"], "event": m["event"], "state": m.get("state") or "?", "lat": m["lat"], "lon": m["lon"], "t0": m["t0"], "year": year, "split": m["split"], "block": blk,
                     "fold": int(hashlib.sha1(blk.encode()).hexdigest(), 16) % 5, "n_start": int(m["n_start"]), "weather_source": src, "has_dem": bool(env["has_dem"]) if "has_dem" in env.files else True,
                     "has_ndvi": (d / "ndvi.npz").exists(), "has_s2": (d / "s2.npz").exists(), "has_burnhist": (d / "burnhist.npz").exists(), "has_osm": (d / "osm.npz").exists(),
                     **start, "epsg": int(env["grid"][0]), "grid_x0": float(env["grid"][1]), "grid_y0": float(env["grid"][2]), "res": float(env["grid"][3]), "crop_r0": r0, "crop_c0": c0, **flags})
        n += 1
        if n % 50 == 0:
            print(n, "of", M, round(time.time() - t_start), "s", flush=True)
    spatial.flush()
    np.save(out / "spatial.npy", np.asarray(spatial[:n])); (out / "spatial_tmp.npy").unlink()
    for H in HORIZONS:
        np.save(out / f"scalars_{H}.npy", scal[H][:n]); np.savez_compressed(out / f"labels_{H}.npz", **{k: v[:n] for k, v in lab[H].items()})
    np.savez_compressed(out / "shared.npz", **{k: v[:n] for k, v in shared.items()})
    pd.DataFrame(meta).to_parquet(out / "meta.parquet")
    json.dump({"spatial": dataset.SPATIAL, "scalars": dataset.SCALARS, "labels": dataset.LABELS, "shared": list(shared), "crop": CROP, "horizons": list(HORIZONS)}, open(out / "names.json", "w"))
    X = np.load(out / "spatial.npy", mmap_mode="r").astype(np.float32)
    qa = {"cases": n, "skipped": skipped, "layers": {k: {"nan_share": float(np.isnan(X[:, i]).mean()), "min": float(np.nanmin(X[:, i])) if np.isfinite(X[:, i]).any() else None, "max": float(np.nanmax(X[:, i])) if np.isfinite(X[:, i]).any() else None,
                                                    "mean": float(np.nanmean(X[:, i])) if np.isfinite(X[:, i]).any() else None} for i, k in enumerate(dataset.SPATIAL)},
          "scalars": {H: {k: {"nan": int(np.isnan(scal[H][:n, i]).sum()), "mean": float(np.nanmean(scal[H][:n, i])), "min": float(np.nanmin(scal[H][:n, i])), "max": float(np.nanmax(scal[H][:n, i]))} for i, k in enumerate(dataset.SCALARS)} for H in HORIZONS},
          "labels": {H: {k: float(lab[H][k][:n].astype(bool).mean()) if k != "burn_rel_day" else float((lab[H][k][:n] >= 0).mean()) for k in dataset.LABELS} for H in HORIZONS}}
    json.dump(qa, open(out / "qa.json", "w"), indent=1)
    print("done", n, "cases,", skipped, "skipped ->", out, flush=True)


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else None)
