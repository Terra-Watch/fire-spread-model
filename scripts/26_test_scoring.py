"""Final look at the frozen test blocks with the feature groups kept after the dev comparison (script 21): base features plus every new group ("+all"; greenness carried the 24 h gain, all groups together the 48 h gain).
Fits on dev, scores test once per horizon (proximity, base, +all; direction AUC and IoU), then fits on all cases and saves data/models/rank2_<H>h.joblib. This is the second look at these test blocks (the first was script 19).
Run: .venv/bin/python scripts/26_test_scoring.py"""
import os
import sys
from importlib import import_module
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from firespread import config, scoring  # noqa: E402

N = import_module("21_new_features")
F = N.F


def main():
    ant = pd.read_parquet(config.DATA / "antecedent.parquet").set_index("case").join(pd.read_parquet(config.DATA / "soil.parquet").set_index("case"), how="inner")
    dirs = [p.parent for root in ("cases", "bank") for p in sorted((config.DATA / root).glob("*/burn_long.npz"))]
    rng = np.random.default_rng(0)
    for H in (24, 48):
        items = [x for x in (N.prepare(d, H, ant) for d in dirs) if x]
        dev, test = [x for x in items if x["split"] == "dev"], [x for x in items if x["split"] == "test"]
        base = [k for k in dev[0]["f"] if k not in F.GROUPS["wind"] and k not in N.DRY + N.LAND + N.PATH + N.GREEN + N.SOIL + N.HUMAN]
        full = base + N.DRY + N.LAND + N.PATH + N.GREEN + N.SOIL + N.HUMAN
        print(f"\n=== {H} h: {len(dev)} dev, {len(test)} test cases", flush=True)
        rows = N.rows(test, lambda it: -it["d"], "prox")
        models = {}
        for tag, names in (("base", base), ("+all", full)):
            models[tag] = N.fit(dev, names, rng)
            rows += N.rows(test, lambda it, m=models[tag], names=names: m.predict_proba(np.column_stack([it["f"][k] for k in names]))[:, 1], tag)
        df = pd.DataFrame(rows)
        F.summarize(df, f"TEST {H} h (frozen blocks, look {os.environ.get('LOOK', 'second')})")
        for tag in ("prox", "base", "+all"):
            g = df[df.set == tag]
            m, lo, hi = scoring.bootstrap_mean(g.groupby("block").iou.mean())
            print(f"  IoU {tag:5s} {m:.3f} [{lo:.3f}, {hi:.3f}]", flush=True)
        b, a = df[df.set == "prox"].set_index("case"), df[df.set == "+all"].set_index("case")
        m, lo, hi = scoring.bootstrap_mean((a.iou - b.iou.reindex(a.index)).groupby(a.block).mean())
        print(f"  IoU +all vs prox {m:+.3f} [{lo:+.3f}, {hi:+.3f}]", flush=True)
        (config.DATA / "models").mkdir(exist_ok=True)
        joblib.dump({"model": N.fit(items, full, rng), "features": full, "horizon_h": H}, config.DATA / "models" / f"{os.environ.get('TAG', 'rank2')}_{H}h.joblib")
        print(f"saved {os.environ.get('TAG', 'rank2')}_{H}h ({len(items)} cases)", flush=True)


if __name__ == "__main__":
    main()
