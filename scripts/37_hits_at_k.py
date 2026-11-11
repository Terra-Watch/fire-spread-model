"""How many of the top-ranked cells are really burned? For each frozen-test fire: of the 200 highest-ranked cells, how many burned (model fitted on dev only, all features, no wind) against the 200 cells nearest the fire,
and what share of the burned cells sit inside the top 200 / top 1000. Also the plain count at the fire's own burned size. Averages over fires with a block bootstrap.   Run: .venv/bin/python scripts/37_hits_at_k.py"""
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


def top_hits(score, y, k):
    k = min(k, len(y))
    return int(y[np.argpartition(-score, k - 1)[:k]].sum())


def main():
    ant = pd.read_parquet(config.DATA / "antecedent.parquet").set_index("case").join(pd.read_parquet(config.DATA / "soil.parquet").set_index("case"), how="inner")
    dirs = [p.parent for root in ("cases", "bank") for p in sorted((config.DATA / root).glob("*/burn_long.npz"))]
    rng = np.random.default_rng(0)
    for H in (24, 48):
        items = [x for x in (N.prepare(d, H, ant) for d in dirs) if x]
        dev, test = [x for x in items if x["split"] == "dev"], [x for x in items if x["split"] == "test"]
        names = [k for k in dev[0]["f"] if k not in F.GROUPS["wind"]]
        m = N.fit(dev, names, rng)
        rows = []
        for it in test:
            y = it["y"]
            model = m.predict_proba(np.column_stack([it["f"][k] for k in names]))[:, 1]
            for tag, s in (("prox", -it["d"]), ("model", model)):
                rows.append({"set": tag, "block": it["block"], "burned": int(y.sum()), "hits200": top_hits(s, y, 200), "hits1000": top_hits(s, y, 1000),
                             "at_true_size": top_hits(s, y, int(y.sum())) / max(1, int(y.sum()))})
        df = pd.DataFrame(rows)
        k = df[df.set == "model"]
        print(f"\\n=== {H} h: {len(test)} test fires; median burned footprint {int(k.burned.median())} cells (of about {int(np.median([len(x['y']) for x in test]))} asked about)", flush=True)
        for tag in ("prox", "model"):
            g = df[df.set == tag]
            h200 = scoring.bootstrap_mean(g.groupby("block").hits200.mean())
            r200 = scoring.bootstrap_mean(g.assign(r=g.hits200 / g.burned.clip(lower=1)).groupby("block").r.mean())
            r1000 = scoring.bootstrap_mean(g.assign(r=g.hits1000 / g.burned.clip(lower=1)).groupby("block").r.mean())
            ts = scoring.bootstrap_mean(g.groupby("block").at_true_size.mean())
            print(f"  {tag:5s} of the top 200 cells: {h200[0]:.0f} burned [{h200[1]:.0f}, {h200[2]:.0f}]   share of the burned cells inside top 200: {r200[0]:.2f}, top 1000: {r1000[0]:.2f}   correct at the fire's own size: {ts[0]:.2f}", flush=True)


if __name__ == "__main__":
    main()
