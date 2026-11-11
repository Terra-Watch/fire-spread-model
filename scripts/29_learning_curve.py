"""Does more training data help? Dev 5-fold by block with the all-features model, training on 25%, 50% and 100% of each training fold's cases; scored on the same held-out folds.
If the score is still climbing at 100%, more cases are the cheapest gain.   Run: .venv/bin/python scripts/29_learning_curve.py"""
import sys
from importlib import import_module
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

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
        dev = [x for x in (N.prepare(d, H, ant) for d in dirs) if x and x["split"] == "dev"]
        names = list(dev[0]["f"].keys())
        names = [k for k in names if k not in F.GROUPS["wind"]]
        blocks = np.array([x["block"] for x in dev])
        rows = []
        for tr, te in GroupKFold(5).split(np.arange(len(dev)), groups=blocks):
            tes = [dev[i] for i in te]
            for frac in (0.25, 0.5, 1.0):
                sub = rng.choice(tr, int(len(tr) * frac), replace=False)
                m = N.fit([dev[i] for i in sub], names, rng)
                rows += N.rows(tes, lambda it, m=m: m.predict_proba(np.column_stack([it["f"][k] for k in names]))[:, 1], f"{int(frac * 100)}%")
            rows += N.rows(tes, lambda it: -it["d"], "prox")
        df = pd.DataFrame(rows)
        print(f"\n=== {H} h, {len(dev)} dev cases (training folds hold about {int(len(dev) * 0.8)})", flush=True)
        for tag in ("prox", "25%", "50%", "100%"):
            g = df[df.set == tag]
            d, i = (scoring.bootstrap_mean(g.groupby("block")[k].mean()) for k in ("dir_auc", "iou"))
            print(f"  {tag:5s} direction AUC {d[0]:.3f} [{d[1]:.3f}, {d[2]:.3f}]   IoU {i[0]:.3f} [{i[1]:.3f}, {i[2]:.3f}]", flush=True)


if __name__ == "__main__":
    main()
