"""Write the cases as map tensors for a neural network: all feature channels (wind left out, as in the ranking model) cropped to 192 x 192 cells (19.2 km) around the start, the truth, the
candidate mask (cells asked about: 5 to 80 cells from the start, mapped, not burned earlier) and the distance map. One file per horizon: data/maps_<H>h.npz.   Run: .venv/bin/python scripts/31_export_maps.py"""
import dataclasses
import json
import sys
from importlib import import_module
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from firespread import burn_eval, cases, config, rank  # noqa: E402

CROP = 192


def window(seed_mask):
    r, c = np.argwhere(seed_mask).mean(axis=0)
    r0, c0 = int(np.clip(round(r) - CROP // 2, 0, seed_mask.shape[0] - CROP)), int(np.clip(round(c) - CROP // 2, 0, seed_mask.shape[1] - CROP))
    return slice(r0, r0 + CROP), slice(c0, c0 + CROP)


def main():
    ant = pd.read_parquet(config.DATA / "antecedent.parquet").set_index("case").join(pd.read_parquet(config.DATA / "soil.parquet").set_index("case"), how="inner")
    dirs = [p.parent for root in ("cases", "bank") for p in sorted((config.DATA / root).glob("*/burn_long.npz"))]
    for H in (24, 48):
        X = y = cand_ = dist_ = None
        meta, names, n = [], None, 0
        for d in dirs:
            try:
                c = cases.load_case(d, gap_override=H)
            except cases.NoWeather:
                continue
            c = dataclasses.replace(c, meta={**c.meta, "t1": str(pd.Timestamp(c.meta["t0"]) + pd.Timedelta(hours=H))})
            b = np.load(d / "burn_long.npz")
            truth, cand, ok = burn_eval.burn_masks(c, b["doy"], int(b["year"]))
            if not ok or c.meta["case_id"] not in ant.index:
                continue
            truth, cand = burn_eval.connected_truth(c, truth, cand)
            if truth.sum() < 5:
                continue
            full = rank.build(c, d, ant.loc[c.meta["case_id"]])
            full = {k: v for k, v in full.items() if k not in rank.WIND}
            full["seed"] = c.seed_mask.astype(np.float32)
            if names is None:
                names = list(full)
                X = np.zeros((len(dirs), len(names), CROP, CROP), np.float16)
                y, cand_ = np.zeros((len(dirs), CROP, CROP), np.uint8), np.zeros((len(dirs), CROP, CROP), np.uint8)
                dist_ = np.zeros((len(dirs), CROP, CROP), np.float16)
            w = window(c.seed_mask)
            X[n] = np.stack([np.broadcast_to(full[k], c.fuel.shape)[w] for k in names]).astype(np.float16)
            y[n], cand_[n], dist_[n] = truth[w], cand[w], c.dist_cells[w]
            m = json.load(open(d / "meta.json"))
            meta.append((m["case_id"], f"{int(np.floor(m['lat']))}_{int(np.floor(m['lon']))}", m["split"], int(m["year"]), m.get("state") or "?"))
            n += 1
        out = config.DATA / f"maps_{H}h.npz"
        np.savez_compressed(out, X=X[:n], y=y[:n], cand=cand_[:n], dist=dist_[:n], names=np.array(names), case=np.array([m[0] for m in meta]), block=np.array([m[1] for m in meta]),
                            split=np.array([m[2] for m in meta]), year=np.array([m[3] for m in meta]), state=np.array([m[4] for m in meta]))
        print(f"{H} h: {n} cases, {len(names)} channels -> {out} ({out.stat().st_size / 1e6:.0f} MB)", flush=True)


if __name__ == "__main__":
    main()
