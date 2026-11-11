"""Do dryness before the fire, wider landscape context and a fuel-aware path distance add to the ranking? Dev only (5-fold by block); the frozen test blocks are not touched here.
Features are kept for the candidate cells only (8 GB of memory is not enough for whole grids).
Run: .venv/bin/python scripts/21_new_features.py"""
import dataclasses
import os
import sys
from importlib import import_module
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from firespread import burn_eval, cases, config, scoring  # noqa: E402

F = import_module("14_final")
feat = import_module("11_learned")
from firespread.rank import DRY, GREEN, HUMAN, LAND, PATH, SOIL, build  # noqa: E402
EDGES = [5, 8, 12, 16, 22, 30, 45, 80]
NEG = F.NEG


def prepare(d, H, ant):
    try:
        c = cases.load_case(d, gap_override=H)
    except cases.NoWeather:
        return None
    c = dataclasses.replace(c, meta={**c.meta, "t1": str(pd.Timestamp(c.meta["t0"]) + pd.Timedelta(hours=H))})
    b = np.load(d / "burn_long.npz")
    truth, cand, ok = burn_eval.burn_masks(c, b["doy"], int(b["year"]))
    if not ok:
        return None
    truth, cand = burn_eval.connected_truth(c, truth, cand)
    if truth.sum() < 5 or c.meta["case_id"] not in ant.index:
        return None
    full = build(c, d, ant.loc[c.meta["case_id"]])
    f = {k: (v[cand] if np.ndim(v) else np.full(int(cand.sum()), v)).astype(np.float32) for k, v in full.items()}
    return {"case": c.meta["case_id"], "block": f"{int(np.floor(c.meta['lat']))}_{int(np.floor(c.meta['lon']))}", "split": c.meta["split"], "y": truth[cand], "d": c.dist_cells[cand], "f": f}


def fit(items, names, rng):
    X, y = [], []
    for it in items:
        pos, neg = np.flatnonzero(it["y"]), np.flatnonzero(~it["y"])
        neg = rng.choice(neg, min(len(neg), NEG * len(pos)), replace=False)
        idx = np.r_[pos, neg]
        X.append(np.column_stack([it["f"][k][idx] for k in names])); y.append(np.r_[np.ones(len(pos)), np.zeros(len(neg))])
    y = np.concatenate(y)
    return HistGradientBoostingClassifier(max_depth=5, learning_rate=0.06, max_iter=300, l2_regularization=1.0, random_state=0).fit(np.vstack(X), y, sample_weight=np.where(y == 1, NEG, 1.0))


def rows(items, scorer, tag):
    out = []
    for it in items:
        s = scorer(it)
        out.append({"case": it["case"], "block": it["block"], "set": tag, "auc": scoring.auc(s, it["y"]), "dir_auc": scoring.auc_within_bins(s, it["y"], it["d"], EDGES), "iou": scoring.iou_at_true_area(s, it["y"])})
    return out


def main():
    ant = pd.read_parquet(config.DATA / "antecedent.parquet").set_index("case").join(pd.read_parquet(config.DATA / "soil.parquet").set_index("case"), how="inner")
    dirs = [p.parent for root in ("cases", "bank") for p in sorted((config.DATA / root).glob("*/burn_long.npz"))]
    rng = np.random.default_rng(0)
    for H in (24, 48):
        items = [x for x in (prepare(d, H, ant) for d in dirs) if x]
        dev = [x for x in items if x["split"] == "dev"]
        base = [k for k in dev[0]["f"] if k not in F.GROUPS["wind"] and k not in DRY + LAND + PATH + GREEN + SOIL + HUMAN]
        variants = {"base": base, "+dry": base + DRY, "+landscape": base + LAND, "+path": base + PATH, "+green": base + GREEN, "+soil": base + SOIL, "+human": base + HUMAN, "+all": base + DRY + LAND + PATH + GREEN + SOIL + HUMAN}
        print(f"\n=== {H} h: {len(dev)} dev cases (of {len(items)}); base features: {len(base)}", flush=True)
        if os.environ.get("VARIANTS"):
            variants = {k: v for k, v in variants.items() if k in os.environ["VARIANTS"].split(",")}
        blocks = np.array([x["block"] for x in dev])
        res = []
        for tr, te in GroupKFold(5).split(np.arange(len(dev)), groups=blocks):
            trs, tes = [dev[i] for i in tr], [dev[i] for i in te]
            res += rows(tes, lambda it: -it["d"], "prox")
            for tag, names in variants.items():
                m = fit(trs, names, rng)
                res += rows(tes, lambda it, m=m, names=names: m.predict_proba(np.column_stack([it["f"][k] for k in names]))[:, 1], tag)
        df = pd.DataFrame(res)
        F.summarize(df, f"DEV {H} h")
        for tag in ["prox", *variants]:
            g = df[df.set == tag]
            m, lo, hi = scoring.bootstrap_mean(g.groupby("block").iou.mean())
            print(f"  IoU {tag:11s} {m:.3f} [{lo:.3f}, {hi:.3f}]", flush=True)
        b = df[df.set == "base"].set_index("case")
        for tag in variants:
            if tag == "base":
                continue
            if tag != "base":
                g = df[df.set == tag].set_index("case")
                m, lo, hi = scoring.bootstrap_mean((g.dir_auc - b.dir_auc.reindex(g.index)).groupby(g.block).mean())
                print(f"  {tag:11s} vs base: direction {m:+.3f} [{lo:+.3f}, {hi:+.3f}]", flush=True)


if __name__ == "__main__":
    main()
