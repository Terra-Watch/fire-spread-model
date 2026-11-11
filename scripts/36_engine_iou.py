"""IoU of the formula engine (US-style fuel numbers, with wind) against plain proximity, on the frozen-test cases, same truth and candidates as the learned model.
Run: .venv/bin/python scripts/36_engine_iou.py"""
import dataclasses
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from firespread import burn_eval, cases, config, scoring  # noqa: E402

EDGES = [5, 8, 12, 16, 22, 30, 45, 80]


def main():
    dirs = [p.parent for root in ("cases", "bank") for p in sorted((config.DATA / root).glob("*/burn_long.npz"))]
    rows = []
    for H in (24, 48):
        for d in dirs:
            if json.load(open(d / "meta.json"))["split"] != "test":
                continue
            try:
                c = cases.load_case(d, gap_override=H)
            except cases.NoWeather:
                continue
            c = dataclasses.replace(c, meta={**c.meta, "t1": str(pd.Timestamp(c.meta["t0"]) + pd.Timedelta(hours=H))})
            b = np.load(d / "burn_long.npz")
            truth, cand, ok = burn_eval.burn_masks(c, b["doy"], int(b["year"]))
            if not ok:
                continue
            truth, cand = burn_eval.connected_truth(c, truth, cand)
            if truth.sum() < 5:
                continue
            y, dist = truth[cand], c.dist_cells[cand]
            prox = -dist
            eng = cases.scores_from_arrival(c, cases.forecast_arrival(c))[cand]
            blk = f"{int(np.floor(c.meta['lat']))}_{int(np.floor(c.meta['lon']))}"
            for tag, s in (("prox", prox), ("engine", eng)):
                rows.append({"H": H, "set": tag, "block": blk, "case": c.id, "iou": scoring.iou_at_true_area(s, y), "dir_auc": scoring.auc_within_bins(s, y, dist, EDGES)})
        print("horizon", H, "done", flush=True)
    df = pd.DataFrame(rows)
    for H in (24, 48):
        g = df[df.H == H]
        p = g[g.set == "prox"].set_index("case")
        for tag in ("prox", "engine"):
            x = g[g.set == tag].set_index("case")
            m = scoring.bootstrap_mean(x.groupby("block").iou.mean())
            dm = scoring.bootstrap_mean((x.iou - p.iou.reindex(x.index)).groupby(x.block).mean())
            da = scoring.bootstrap_mean(x.groupby("block").dir_auc.mean())
            print(f"{H} h {tag:6s} n={len(x)} IoU {m[0]:.3f} [{m[1]:.3f}, {m[2]:.3f}]  direction AUC {da[0]:.3f}   IoU vs prox {dm[0]:+.3f} [{dm[1]:+.3f}, {dm[2]:+.3f}]", flush=True)


if __name__ == "__main__":
    main()
