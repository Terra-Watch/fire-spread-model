"""Harder validation than block CV on dev: (a) train on cases from 2019-2023, test on 2024-2025 (new years); (b) leave one state out. All cases (dev and test splits) are used; nothing is tuned here.
All-features model against plain proximity; direction AUC and IoU with block bootstrap.   Run: .venv/bin/python scripts/30_holdouts.py"""
import sys
from importlib import import_module
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from firespread import config, scoring  # noqa: E402

N = import_module("21_new_features")
F = N.F


def report(df, title):
    print(title, flush=True)
    b = df[df.set == "prox"].set_index("case")
    for tag in ("prox", "model"):
        g = df[df.set == tag]
        d, i = (scoring.bootstrap_mean(g.groupby("block")[k].mean()) for k in ("dir_auc", "iou"))
        line = f"  {tag:5s} direction AUC {d[0]:.3f} [{d[1]:.3f}, {d[2]:.3f}]   IoU {i[0]:.3f} [{i[1]:.3f}, {i[2]:.3f}]"
        if tag == "model":
            gi = g.set_index("case")
            m = scoring.bootstrap_mean((gi.dir_auc - b.dir_auc.reindex(gi.index)).groupby(gi.block).mean())
            mi = scoring.bootstrap_mean((gi.iou - b.iou.reindex(gi.index)).groupby(gi.block).mean())
            line += f"   vs prox: direction {m[0]:+.3f} [{m[1]:+.3f}, {m[2]:+.3f}], IoU {mi[0]:+.3f} [{mi[1]:+.3f}, {mi[2]:+.3f}]"
        print(line, flush=True)


def main():
    import json
    ant = pd.read_parquet(config.DATA / "antecedent.parquet").set_index("case").join(pd.read_parquet(config.DATA / "soil.parquet").set_index("case"), how="inner")
    dirs = [p.parent for root in ("cases", "bank") for p in sorted((config.DATA / root).glob("*/burn_long.npz"))]
    rng = np.random.default_rng(0)
    for H in (24, 48):
        items = []
        for d in dirs:
            x = N.prepare(d, H, ant)
            if x:
                m = json.load(open(d / "meta.json"))
                x["year"], x["state"] = int(m["year"]), m.get("state") or "?"
                items.append(x)
        names = [k for k in items[0]["f"] if k not in F.GROUPS["wind"]]
        score = lambda it, m: m.predict_proba(np.column_stack([it["f"][k] for k in names]))[:, 1]
        print(f"\n=== {H} h: {len(items)} cases", flush=True)
        tr, te = [x for x in items if x["year"] <= 2023], [x for x in items if x["year"] >= 2024]
        if len(te) >= 20:
            m = N.fit(tr, names, rng)
            report(pd.DataFrame(N.rows(te, lambda it: -it["d"], "prox") + N.rows(te, lambda it: score(it, m), "model")), f"TIME HOLDOUT: train {len(tr)} cases 2019-2023, test {len(te)} cases 2024-2025")
        else:
            print(f"TIME HOLDOUT skipped: only {len(te)} cases in 2024-2025", flush=True)
        states = pd.Series([x["state"] for x in items]).value_counts()
        for st in states[states >= 25].index:
            tr, te = [x for x in items if x["state"] != st], [x for x in items if x["state"] == st]
            m = N.fit(tr, names, rng)
            report(pd.DataFrame(N.rows(te, lambda it: -it["d"], "prox") + N.rows(te, lambda it: score(it, m), "model")), f"LEAVE OUT {st}: {len(te)} test cases")


if __name__ == "__main__":
    main()
