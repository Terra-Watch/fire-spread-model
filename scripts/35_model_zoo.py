"""Other model families on the same maps, same dev folds (5-fold by block), same scoring as the boosted trees and the U-Net: tuned boosting (XGBoost, LightGBM), random forest, extra trees, logistic regression, a small MLP,
and a rank-average ensemble of the best few. Chosen on dev only; nothing here touches the frozen test blocks.   Run: .venv/bin/python scripts/35_model_zoo.py data/maps_48h.npz [out.json]"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from lightgbm import LGBMClassifier
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from firespread import scoring  # noqa: E402

EDGES = [5, 8, 12, 16, 22, 30, 45, 80]
NEG = 12
imp = lambda: SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True)
MODELS = {
    "hgb": lambda: HistGradientBoostingClassifier(max_depth=5, learning_rate=0.06, max_iter=300, l2_regularization=1.0, random_state=0),
    "hgb_deep": lambda: HistGradientBoostingClassifier(max_depth=8, learning_rate=0.03, max_iter=700, l2_regularization=3.0, min_samples_leaf=60, random_state=0),
    "xgb": lambda: XGBClassifier(n_estimators=600, max_depth=6, learning_rate=0.04, subsample=0.8, colsample_bytree=0.7, min_child_weight=5, reg_lambda=3.0, tree_method="hist", n_jobs=8, random_state=0),
    "lgbm": lambda: LGBMClassifier(n_estimators=600, num_leaves=48, learning_rate=0.04, subsample=0.8, subsample_freq=1, colsample_bytree=0.7, min_child_samples=40, reg_lambda=3.0, n_jobs=8, random_state=0, verbose=-1),
    "rf": lambda: make_pipeline(imp(), RandomForestClassifier(n_estimators=120, min_samples_leaf=30, max_features=0.4, n_jobs=8, random_state=0)),
    "extra": lambda: make_pipeline(imp(), ExtraTreesClassifier(n_estimators=120, min_samples_leaf=30, max_features=0.5, n_jobs=8, random_state=0)),
    "logreg": lambda: make_pipeline(imp(), StandardScaler(), LogisticRegression(C=0.3, max_iter=400)),
    "mlp": lambda: make_pipeline(imp(), StandardScaler(), MLPClassifier(hidden_layer_sizes=(128, 64), alpha=1e-3, early_stopping=True, max_iter=30, random_state=0)),
}
SAMPLE_WEIGHT = {"hgb", "hgb_deep", "xgb", "lgbm"}


def table(X, y, cand, idx, rng):
    xs, ys = [], []
    for i in idx:
        m = cand[i] > 0
        yy = y[i][m] > 0
        pos, neg = np.flatnonzero(yy), np.flatnonzero(~yy)
        neg = rng.choice(neg, min(len(neg), NEG * len(pos)), replace=False)
        sel = np.r_[pos, neg]
        xs.append(X[i][:, m][:, sel].T.astype(np.float32)); ys.append(yy[sel].astype(int))
    return np.vstack(xs), np.concatenate(ys)


def main(path, out=None):
    z = np.load(path)
    X, y, cand, dist = z["X"], z["y"], z["cand"], z["dist"].astype(np.float32)
    wind = [k for k, n in enumerate(z["names"]) if n in ("wind", "wind_along")]
    keep = [k for k in range(X.shape[1]) if k not in wind]
    X = X[:, keep]
    dev_i = np.flatnonzero(z["split"] == "dev")[:int(os.environ.get("LIMIT", 10 ** 9))]
    rng = np.random.default_rng(0)
    rows = []
    print(f"{len(dev_i)} dev cases, {X.shape[1]} channels", flush=True)
    for fold, (tr, te) in enumerate(GroupKFold(5).split(dev_i, groups=z["block"][dev_i])):
        tr, te = dev_i[tr], dev_i[te]
        Xt, yt = table(X, y, cand, tr, rng)
        w = np.where(yt == 1, NEG, 1.0)
        fitted = {}
        for name, make in MODELS.items():
            t_fit = time.time()
            m = make()
            if name in SAMPLE_WEIGHT:
                m.fit(Xt, yt, sample_weight=w)
            elif name == "mlp" or name == "logreg":
                m.fit(Xt, yt)
            else:
                m.fit(Xt, yt, **{f"{m.steps[-1][0]}__sample_weight": w})
            fitted[name] = m
            print(f"  fold {fold} {name} fitted in {time.time() - t_fit:.0f}s", flush=True)
        for i in te:
            m = cand[i] > 0
            yy, d = y[i][m] > 0, dist[i][m]
            feats = X[i][:, m].T.astype(np.float32)
            sc = {"prox": -d, **{n: f.predict_proba(feats)[:, 1] for n, f in fitted.items()}}
            rk = lambda s: s.argsort().argsort()
            sc["ens3"] = rk(sc["xgb"]) + rk(sc["lgbm"]) + rk(sc["rf"])
            sc["ens_all"] = sum(rk(sc[n]) for n in MODELS if n not in ("logreg", "mlp"))
            for tag, s in sc.items():
                rows.append({"case": str(z["case"][i]), "block": str(z["block"][i]), "set": tag, "dir_auc": scoring.auc_within_bins(s, yy, d, EDGES), "iou": scoring.iou_at_true_area(s, yy)})
        print("fold", fold, "done", flush=True)
    base = {r["case"]: r for r in rows if r["set"] == "prox"}
    res = {}
    for tag in dict.fromkeys(r["set"] for r in rows):
        g = [r for r in rows if r["set"] == tag]
        blk = sorted({r["block"] for r in g})

        def per(f):
            return np.array([np.mean([f(r, base[r["case"]]) for r in g if r["block"] == bk]) for bk in blk])
        d, i = scoring.bootstrap_mean(per(lambda r, b: r["dir_auc"])), scoring.bootstrap_mean(per(lambda r, b: r["iou"]))
        dd, di = scoring.bootstrap_mean(per(lambda r, b: r["dir_auc"] - b["dir_auc"])), scoring.bootstrap_mean(per(lambda r, b: r["iou"] - b["iou"]))
        print(f"  {tag:9s} direction AUC {d[0]:.3f} [{d[1]:.3f}, {d[2]:.3f}]  IoU {i[0]:.3f} [{i[1]:.3f}, {i[2]:.3f}]   vs prox: direction {dd[0]:+.3f} [{dd[1]:+.3f}, {dd[2]:+.3f}], IoU {di[0]:+.3f} [{di[1]:+.3f}, {di[2]:+.3f}]", flush=True)
        res[tag] = {"dir_auc": d, "iou": i, "dir_auc_gain": dd, "iou_gain": di}
    if out:
        json.dump(res, open(out, "w"))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
