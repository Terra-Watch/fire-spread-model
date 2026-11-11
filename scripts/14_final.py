"""The model comparison on everything built: the pilot (with terrain) and the bank (without). Dev blocks for choosing, the frozen test blocks scored once at the end.
Candidates: prox (baseline), engine (physics only), learned on plain features, learned with the engine's arrival time as a feature, and variants that leave one feature group out.
Run: .venv/bin/python scripts/14_final.py"""
import sys
from importlib import import_module
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from firespread import burn_eval, cases, config, scoring  # noqa: E402

feat = import_module("11_learned")
NEG = 12
GROUPS = {"wind": ["wind", "wind_along"], "slope": ["slope", "slope_along"], "fuel": ["tree", "shrub", "grass", "crop", "built", "water", "forest_share"],
          "weather": ["moist", "temp", "rh"], "fire": ["n_start", "frp", "gap"], "activity": ["act5", "act20", "act_ev20"], "season": ["month", "t0_hour"]}


def usable(dirs):
    """Case directories whose burn truth gives at least 5 growth cells and 50 other candidates."""
    out = []
    for root in dirs:
        for d in sorted((config.DATA / root).glob("*/burn.npz")):
            out.append(d.parent)
    return out


def prepare(d):
    try:
        c = cases.load_case(d)
    except cases.NoWeather:
        return None
    b = np.load(d / "burn.npz")
    truth, cand, ok = burn_eval.burn_masks(c, b["doy"], int(b["year"]))
    if not ok:
        return None
    truth, cand = burn_eval.connected_truth(c, truth, cand)
    if truth.sum() < 5:
        return None
    f = feat.features(c)
    arr = cases.forecast_arrival(c)
    cap = c.gap_h + 12.0
    f["engine_t"] = np.where(np.isfinite(arr), arr, cap)
    f["ellipse_t"] = np.where(cases.baseline_wind_ellipse(c) > -1e5, -cases.baseline_wind_ellipse(c), cap)
    return c, truth, cand, f


def sample(item, rng):
    c, truth, cand, f = item
    pos = np.flatnonzero((truth & cand).ravel())
    neg_all = np.flatnonzero((cand & ~truth).ravel())
    neg = rng.choice(neg_all, min(len(neg_all), NEG * max(len(pos), 1)), replace=False)
    return np.concatenate([pos, neg]), np.r_[np.ones(len(pos)), np.zeros(len(neg))]


def table(items, names, rng):
    X, y = [], []
    for it in items:
        idx, lab = sample(it, rng)
        X.append(np.column_stack([it[3][k].ravel()[idx] for k in names])); y.append(lab)
    return np.vstack(X), np.concatenate(y)


def fit(items, names, rng):
    X, y = table(items, names, rng)
    return HistGradientBoostingClassifier(max_depth=5, learning_rate=0.06, max_iter=300, l2_regularization=1.0, random_state=0).fit(X, y, sample_weight=np.where(y == 1, NEG, 1.0))


def rows_for(items, scorer, tag):
    out = []
    for it in items:
        c, truth, cand, f = it
        r = {"case": c.id, "block": f"{int(np.floor(c.meta['lat']))}_{int(np.floor(c.meta['lon']))}", "set": tag, "year": c.meta["year"], "src": c.extra["dir"].parent.name}
        for k, v in burn_eval.evaluate_burn(c, scorer(it), truth, cand).items():
            r[k] = v
        out.append(r)
    return out


def summarize(df, title, base="prox"):
    print(title)
    b = df[df.set == base].set_index("case")
    for tag in df.set.unique():
        g = df[df.set == tag]
        a, d = (scoring.bootstrap_mean(g.groupby("block")[k].mean()) for k in ("auc", "dir_auc"))
        line = f"  {tag:14s} AUC {a[0]:.3f}  direction AUC {d[0]:.3f} [{d[1]:.3f}, {d[2]:.3f}]"
        if tag != base:
            gi = g.set_index("case")
            diff = (gi.dir_auc - b.dir_auc.reindex(gi.index)).groupby(gi.block).mean()
            m, lo, hi = scoring.bootstrap_mean(diff)
            diffa = (gi.auc - b.auc.reindex(gi.index)).groupby(gi.block).mean()
            ma = scoring.bootstrap_mean(diffa)
            line += f"   vs prox: direction {m:+.3f} [{lo:+.3f}, {hi:+.3f}], overall {ma[0]:+.4f} [{ma[1]:+.4f}, {ma[2]:+.4f}]"
        print(line)


def main(dirs=("cases", "bank")):
    rng = np.random.default_rng(0)
    items = [x for x in (prepare(d) for d in usable(dirs)) if x]
    dev = [x for x in items if x[0].meta["split"] == "dev"]
    test = [x for x in items if x[0].meta["split"] == "test"]
    print(len(items), "usable cases:", len(dev), "dev,", len(test), "test; blocks:", len({(int(np.floor(x[0].meta['lat'])), int(np.floor(x[0].meta['lon']))) for x in items}))
    allnames = list(feat.features.__code__.co_varnames) and [k for k in dev[0][3].keys()]
    base_names = [k for k in allnames if k not in ("engine_t", "ellipse_t")]
    variants = {"learned": base_names, "learned+physics": allnames}
    for g, ks in GROUPS.items():
        variants[f"learned -{g}"] = [k for k in base_names if k not in ks]
    blocks = np.array([f"{int(np.floor(x[0].meta['lat']))}_{int(np.floor(x[0].meta['lon']))}" for x in dev])
    rows = []
    for tr, te in GroupKFold(5).split(np.arange(len(dev)), groups=blocks):
        trs, tes = [dev[i] for i in tr], [dev[i] for i in te]
        rows += rows_for(tes, lambda it: cases.baseline_proximity(it[0]), "prox")
        rows += rows_for(tes, lambda it: cases.scores_from_arrival(it[0], cases.forecast_arrival(it[0])), "engine")
        for tag, names in variants.items():
            m = fit(trs, names, rng)
            rows += rows_for(tes, lambda it, m=m, names=names: m.predict_proba(np.column_stack([it[3][k].ravel() for k in names]))[:, 1].reshape(it[0].fuel.shape), tag)
    summarize(pd.DataFrame(rows), "DEV, 5-fold grouped by block")
    # one look at the frozen test blocks, with the two full models trained on all dev
    rows = rows_for(test, lambda it: cases.baseline_proximity(it[0]), "prox") + rows_for(test, lambda it: cases.scores_from_arrival(it[0], cases.forecast_arrival(it[0])), "engine")
    for tag in ("learned", "learned+physics"):
        m = fit(dev, variants[tag], rng)
        rows += rows_for(test, lambda it, m=m, names=variants[tag]: m.predict_proba(np.column_stack([it[3][k].ravel() for k in names]))[:, 1].reshape(it[0].fuel.shape), tag)
    df = pd.DataFrame(rows)
    summarize(df, "TEST, frozen blocks, scored once")
    df.to_csv(config.DATA / "final_test_rows.csv", index=False)


if __name__ == "__main__":
    main(tuple(sys.argv[1:]) or ("cases", "bank"))
