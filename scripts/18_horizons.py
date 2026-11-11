"""Does anything beyond closeness show up at 48 h that it did not at 24 h? Same cases, same models, truth = burned within H hours of the start (MODIS day resolution, burn_long.npz).
Models: proximity, learned, learned without wind. Dev 5-fold by block, then the frozen test blocks once per horizon. Candidates stay within 8 km of the start (80 cells).
Run: .venv/bin/python scripts/18_horizons.py"""
import dataclasses
import sys
from importlib import import_module
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from firespread import burn_eval, cases, config  # noqa: E402

F = import_module("14_final")
feat = import_module("11_learned")
HORIZONS = (24, 48)


def prepare(d, H):
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
    if truth.sum() < 5:
        return None
    f = {k: v.astype(np.float32) for k, v in feat.features(c).items()}
    return c, truth, cand, f


def main():
    dirs = [p.parent for root in ("cases", "bank") for p in sorted((config.DATA / root).glob("*/burn_long.npz"))]
    rng = np.random.default_rng(0)
    for H in HORIZONS:
        items = [x for x in (prepare(d, H) for d in dirs) if x]
        dev = [x for x in items if x[0].meta["split"] == "dev"]
        test = [x for x in items if x[0].meta["split"] == "test"]
        print(f"\n=== horizon {H} h: {len(dev)} dev, {len(test)} test cases, positives per case median {int(np.median([x[1].sum() for x in items]))}", flush=True)
        names = list(dev[0][3].keys())
        variants = {"learned": names, "learned -wind": [k for k in names if k not in F.GROUPS["wind"]]}
        blocks = np.array([f"{int(np.floor(x[0].meta['lat']))}_{int(np.floor(x[0].meta['lon']))}" for x in dev])
        prox = lambda it: cases.baseline_proximity(it[0])
        rows = []
        for tr, te in GroupKFold(5).split(np.arange(len(dev)), groups=blocks):
            trs, tes = [dev[i] for i in tr], [dev[i] for i in te]
            rows += F.rows_for(tes, prox, "prox")
            for tag, ns in variants.items():
                m = F.fit(trs, ns, rng)
                rows += F.rows_for(tes, lambda it, m=m, ns=ns: m.predict_proba(np.column_stack([it[3][k].ravel() for k in ns]))[:, 1].reshape(it[0].fuel.shape), tag)
        F.summarize(pd.DataFrame(rows), f"DEV {H} h")
        rows = F.rows_for(test, prox, "prox")
        for tag, ns in variants.items():
            m = F.fit(dev, ns, rng)
            rows += F.rows_for(test, lambda it, m=m, ns=ns: m.predict_proba(np.column_stack([it[3][k].ravel() for k in ns]))[:, 1].reshape(it[0].fuel.shape), tag)
        F.summarize(pd.DataFrame(rows), f"TEST {H} h (frozen blocks)")


if __name__ == "__main__":
    main()
