"""Does giving the learned model the physics help? Three feature sets, grouped cross-validation on the dev blocks, then ONE look at the frozen test blocks.
A: plain features (distance, wind along the bearing, slope along the bearing, fuel shares, moisture, fire size)
B: A + the engine's arrival time and the wind-ellipse arrival time as features (physics-informed)
Run: .venv/bin/python scripts/12_hybrid.py"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from firespread import burn_eval, cases, config, scoring  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parent))
from importlib import import_module  # noqa: E402

feat = import_module("11_learned")
NEG = 15


def physics_features(c):
    arr = cases.forecast_arrival(c)
    wind = cases.baseline_wind_ellipse(c)
    cap = c.gap_h + 6.0
    return {"engine_t": np.where(np.isfinite(arr), arr, cap + 6.0), "ellipse_t": np.where(wind > -1e5, -wind, cap + 6.0)}


def build(split_names):
    data = []
    for d in sorted((config.DATA / "cases").glob("*/burn.npz")):
        try:
            c = cases.load_case(d.parent)
        except cases.NoWeather:
            continue
        if c.meta["split"] not in split_names:
            continue
        b = np.load(d)
        truth, cand, ok = burn_eval.burn_masks(c, b["doy"], int(b["year"]))
        if not ok:
            continue
        truth, cand = burn_eval.connected_truth(c, truth, cand)
        if truth.sum() < 5:
            continue
        f = feat.features(c)
        f.update(physics_features(c))
        data.append((c, truth, cand, f))
    return data


def fit_predict(train, test, names, rng):
    X, y = [], []
    for c, truth, cand, f in train:
        pos = np.flatnonzero((truth & cand).ravel())
        neg = rng.choice(np.flatnonzero((cand & ~truth).ravel()), min(int((cand & ~truth).sum()), NEG * max(len(pos), 1)), replace=False)
        idx = np.concatenate([pos, neg])
        X.append(np.column_stack([f[k].ravel()[idx] for k in names])); y.append(np.r_[np.ones(len(pos)), np.zeros(len(neg))])
    X, y = np.vstack(X), np.concatenate(y)
    m = HistGradientBoostingClassifier(max_depth=4, learning_rate=0.06, max_iter=250, l2_regularization=1.0, random_state=0).fit(X, y, sample_weight=np.where(y == 1, NEG, 1.0))
    out = []
    for c, truth, cand, f in test:
        out.append(m.predict_proba(np.column_stack([f[k].ravel() for k in names]))[:, 1].reshape(c.fuel.shape))
    return out


def score_rows(items, preds, tag):
    rows = []
    for (c, truth, cand, f), p in zip(items, preds):
        r = {"case": c.id, "block": f"{int(np.floor(c.meta['lat']))}_{int(np.floor(c.meta['lon']))}", "set": tag}
        for k, v in burn_eval.evaluate_burn(c, p, truth, cand).items():
            r[k] = v
        rows.append(r)
    return rows


def report(df, title):
    print(title)
    p = df[df.set == "prox"].set_index("case")
    for tag in ("prox", "engine", "A", "B"):
        g = df[df.set == tag]
        a, b = (scoring.bootstrap_mean(g.groupby("block")[k].mean()) for k in ("auc", "dir_auc"))
        line = f"  {tag:7s} AUC {a[0]:.3f} [{a[1]:.3f}, {a[2]:.3f}]  direction AUC {b[0]:.3f} [{b[1]:.3f}, {b[2]:.3f}]"
        if tag != "prox":
            d = (g.set_index("case").dir_auc - p.dir_auc).groupby(g.set_index("case").block).mean()
            m, lo, hi = scoring.bootstrap_mean(d)
            line += f"   direction vs prox {m:+.3f} [{lo:+.3f}, {hi:+.3f}]"
        print(line)


def main():
    rng = np.random.default_rng(0)
    dev, test = build({"dev"}), build({"test"})
    print(len(dev), "dev cases,", len(test), "test cases")
    namesA = list(feat.features(dev[0][0]).keys())
    namesB = namesA + ["engine_t", "ellipse_t"]
    blocks = np.array([f"{int(np.floor(c.meta['lat']))}_{int(np.floor(c.meta['lon']))}" for c, *_ in dev])
    rows = []
    for tr, te in GroupKFold(5).split(np.arange(len(dev)), groups=blocks):
        trs, tes = [dev[i] for i in tr], [dev[i] for i in te]
        rows += score_rows(tes, fit_predict(trs, tes, namesA, rng), "A") + score_rows(tes, fit_predict(trs, tes, namesB, rng), "B")
        rows += score_rows(tes, [cases.baseline_proximity(c) for c, *_ in tes], "prox") + score_rows(tes, [cases.scores_from_arrival(c, cases.forecast_arrival(c)) for c, *_ in tes], "engine")
    report(pd.DataFrame(rows), "DEV (5-fold, grouped by block)")
    rows = score_rows(test, fit_predict(dev, test, namesA, rng), "A") + score_rows(test, fit_predict(dev, test, namesB, rng), "B")
    rows += score_rows(test, [cases.baseline_proximity(c) for c, *_ in test], "prox") + score_rows(test, [cases.scores_from_arrival(c, cases.forecast_arrival(c)) for c, *_ in test], "engine")
    report(pd.DataFrame(rows), "TEST (trained on all dev, scored once)")


if __name__ == "__main__":
    main()
