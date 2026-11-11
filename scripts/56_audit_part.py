"""Audit of one exported part: layer completeness and ranges by split, label statistics, agreement of the two truths (MODIS burned area against later VIIRS detections), file hashes. Writes audit.json next to the part.
Run: .venv/bin/python scripts/56_audit_part.py data/dataset/part_X"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def sha(p, n=1 << 20):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while chunk := f.read(n):
            h.update(chunk)
    return h.hexdigest()


def main(part):
    part = Path(part)
    names = json.load(open(part / "names.json")); meta = pd.read_parquet(part / "meta.parquet")
    X = np.load(part / "spatial.npy", mmap_mode="r"); shared = np.load(part / "shared.npz")
    audit = {"part": part.name, "cases": len(meta), "by_split": meta.split.value_counts().to_dict(), "by_state": meta.state.value_counts().to_dict(), "by_year": {int(k): int(v) for k, v in meta.year.value_counts().sort_index().items()},
             "weather_source": meta.weather_source.value_counts().to_dict(), "flags": {k: float(meta[k].mean()) for k in ("has_dem", "has_ndvi", "has_s2", "has_burnhist", "has_osm")},
             "usable": {H: int(meta[f"usable_{H}"].sum()) for H in (24, 48)}, "t0_range": [str(meta.t0.min()), str(meta.t0.max())]}
    # layer statistics by split (cases of the same split, whole crop)
    layers = {}
    for k, name in enumerate(names["spatial"]):
        row = {}
        for sp in ("dev", "test"):
            idx = np.flatnonzero(meta.split.values == sp)
            if len(idx) == 0:
                continue
            v = np.asarray(X[idx[:200], k], np.float32)
            row[sp] = {"nan_share": float(np.isnan(v).mean()), "mean": float(np.nanmean(v)) if np.isfinite(v).any() else None, "p05": float(np.nanpercentile(v, 5)) if np.isfinite(v).any() else None, "p95": float(np.nanpercentile(v, 95)) if np.isfinite(v).any() else None}
        layers[name] = row
    audit["layers"] = layers
    # labels
    cand = shared["cand"] > 0; labs = {}
    for H in (24, 48):
        L = np.load(part / f"labels_{H}.npz")
        ym, ya, yv, yn = (L[k].astype(bool) for k in ("y_modis", "y_modis_all", "y_viirs_new", "y_viirs_any"))
        pos = ym.reshape(len(ym), -1).sum(1); asked = cand.reshape(len(cand), -1).sum(1)
        inter = (yv & ya).reshape(len(ym), -1).sum(1); uni = (yv | ya).reshape(len(ym), -1).sum(1)
        labs[H] = {"modis_positives_per_case": {q: float(np.percentile(pos, q)) for q in (5, 25, 50, 75, 95)}, "modis_share_of_asked": float(pos.sum() / asked.sum()),
                   "viirs_new_share_of_asked": float(yv.sum() / asked.sum()), "cases_with_viirs_new": float((yv.reshape(len(yv), -1).sum(1) > 0).mean()),
                   "modis_viirs_overlap_iou_median": float(np.median(inter / np.maximum(uni, 1))), "viirs_new_inside_modis_burned_share": float(inter.sum() / max(1, yv.sum())),
                   "persistence_any_detection_share": float(yn.sum() / asked.sum()), "usable_cases": int(meta[f"usable_{H}"].sum())}
    audit["labels"] = labs
    audit["files"] = {p.name: {"bytes": p.stat().st_size, "sha256": sha(p)} for p in sorted(part.iterdir()) if p.is_file() and p.name != "audit.json"}
    json.dump(audit, open(part / "audit.json", "w"), indent=1, default=str)
    print("audit written for", part.name, "|", audit["cases"], "cases | label rates", {H: round(labs[H]["modis_share_of_asked"], 4) for H in labs})


if __name__ == "__main__":
    main(sys.argv[1])
