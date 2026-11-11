"""A learned model of where fires grow, to ask what the physics engine is missing. Features per cell: distance, wind along the bearing, slope along the bearing, fuel shares, moisture,
fire size. Truth: the MODIS burn date, patch touching the start. Grouped cross-validation by 1 x 1 degree block, so no block is in both training and scoring.
Run: .venv/bin/python scripts/11_learned.py"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from firespread import activity, burn_eval, cases, config, scoring  # noqa: E402

NEG_PER_POS = 15


from firespread.rank import base_features as features  # noqa: E402,F401  (moved to the package so a live job uses the same code)


def main():
    data = []
    for d in sorted((config.DATA / "cases").glob("*/burn.npz")):
        try:
            c = cases.load_case(d.parent)
        except cases.NoWeather:
            continue
        b = np.load(d)
        truth, cand, ok = burn_eval.burn_masks(c, b["doy"], int(b["year"]))
        if not ok:
            continue
        truth, cand = burn_eval.connected_truth(c, truth, cand)
        if truth.sum() < 5:
            continue
        data.append((c, truth, cand, features(c)))
    names = list(data[0][3].keys())
    blocks = np.array([f"{int(np.floor(c.meta['lat']))}_{int(np.floor(c.meta['lon']))}" for c, *_ in data])
    rng = np.random.default_rng(0)

    def table(items):
        X, y = [], []
        for c, truth, cand, f in items:
            pos = np.flatnonzero((truth & cand).ravel())
            neg = np.flatnonzero((cand & ~truth).ravel())
            neg = rng.choice(neg, min(len(neg), NEG_PER_POS * max(len(pos), 1)), replace=False)
            idx = np.concatenate([pos, neg])
            X.append(np.column_stack([f[k].ravel()[idx] for k in names]))
            y.append(np.r_[np.ones(len(pos)), np.zeros(len(neg))])
        return np.vstack(X), np.concatenate(y)

    rows = []
    for tr, te in GroupKFold(5).split(np.arange(len(data)), groups=blocks):
        X, y = table([data[i] for i in tr])
        w = np.where(y == 1, NEG_PER_POS, 1.0)                           # undo the negative subsampling
        model = HistGradientBoostingClassifier(max_depth=4, learning_rate=0.06, max_iter=250, l2_regularization=1.0, random_state=0).fit(X, y, sample_weight=w)
        for i in te:
            c, truth, cand, f = data[i]
            Xc = np.column_stack([f[k].ravel() for k in names])
            s = model.predict_proba(Xc)[:, 1].reshape(c.fuel.shape)
            r = {"case": c.id, "block": blocks[i]}
            for nm, sc in (("learned", s), ("prox", cases.baseline_proximity(c)), ("model", cases.scores_from_arrival(c, cases.forecast_arrival(c)))):
                for k, v in burn_eval.evaluate_burn(c, sc, truth, cand).items():
                    r[f"{nm}_{k}"] = v
            rows.append(r)
    df = pd.DataFrame(rows)
    print(len(df), "cases,", df.block.nunique(), "blocks; features:", ", ".join(names))
    for m in ("learned", "prox", "model"):
        a, b = (scoring.bootstrap_mean(df.groupby("block")[f"{m}_{k}"].mean()) for k in ("auc", "dir_auc"))
        print(f"{m:8s} AUC {a[0]:.3f} [{a[1]:.3f}, {a[2]:.3f}]   direction AUC {b[0]:.3f} [{b[1]:.3f}, {b[2]:.3f}]")
    for base in ("prox", "model"):
        for key in ("auc", "dir_auc"):
            m, lo, hi = scoring.bootstrap_mean((df[f"learned_{key}"] - df[f"{base}_{key}"]).groupby(df.block).mean())
            print(f"learned - {base:6s} {key:8s} {m:+.3f} [{lo:+.3f}, {hi:+.3f}]  (learned higher in {(df[f'learned_{key}'] > df[f'{base}_{key}']).mean():.2f} of cases)")
    # what the learned model relies on
    X, y = table(data)
    from sklearn.inspection import permutation_importance
    model = HistGradientBoostingClassifier(max_depth=4, learning_rate=0.06, max_iter=250, l2_regularization=1.0, random_state=0).fit(X, y, sample_weight=np.where(y == 1, NEG_PER_POS, 1.0))
    imp = permutation_importance(model, X[::7], y[::7], n_repeats=3, random_state=0, scoring="roc_auc")
    print("importance (drop in AUC when shuffled):", {n: round(float(v), 3) for n, v in sorted(zip(names, imp.importances_mean), key=lambda kv: -kv[1])[:8]})


if __name__ == "__main__":
    main()
