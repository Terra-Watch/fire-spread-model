"""Trains one image model on the exported dataset parts and scores it the same way as everything else (direction AUC, IoU at the true area, hits in the top 200, all against nearest-first), per case, with a block bootstrap.
Scope v4: train = the 17 v4 states, label years up to 2023; validation = 2024; test = 2025; blind = Uttar Pradesh, West Bengal, Kerala; outside = the other states. Scope smoke is for a quick test.
Run: python scripts/60_train.py --model unet|firesensenet --horizon 24|48 --parts PART_DIR [PART_DIR ...] --out RUN_DIR [--epochs 30] [--scope v4] [--train-fires N] [--pretrained 1]"""
import argparse
import json
import sys
import time
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from firespread import imgmodels, scoring  # noqa: E402

V4_TRAIN = {"chhattisgarh", "jharkhand", "maharashtra", "madhya pradesh", "odisha", "telangana", "andhra pradesh", "uttarakhand", "karnataka", "tamil nadu", "himachal pradesh", "arunachal pradesh", "assam", "manipur", "mizoram", "sikkim", "tripura"}
V4_BLIND = {"kerala", "uttar pradesh", "west bengal"}
DROP_SPATIAL = {"aspect_sin", "aspect_cos"}                                   # absolute directions: turning the picture would make them wrong
DROP_SCALAR = {"wind", "wind_u", "wind_v", "wind_max", "gust_max", "lat", "lon"}   # wind added nothing in every test; lat/lon would let a model memorise places
# layer groups for the ablation (--drop): spatial layers and per-fire numbers
GROUPS = {
    "burn_history": ({"burn_days_since", "burn_n"}, set()),
    "sentinel2": ({"s2_nir", "s2_swir1", "s2_swir2", "s2_ndmi", "s2_nbr", "s2_age", "s2_valid"}, set()),
    "ndvi": ({"ndvi", "ndvi_prev", "ndvi_anom"}, set()),
    "fire_history": ({"fire_hist_90", "fire_hist_365", "fire_days_since", "other_fires_72h"}, {"act5", "act20", "act_ev20"}),
    "weather": (set(), {"temp_mean", "temp_max", "rh_mean", "rh_min", "vpd_max", "rain_sum", "moist_mean", "moist_min", "rain7", "rain30", "rain60", "days_dry", "balance30", "tmax14", "rh14", "sm7", "sm28", "sm7_change", "vpd3"}),
    "terrain_cover": ({"tree", "shrub", "grass", "crop", "built", "water", "bare", "forest_share", "forest_5", "forest_15", "tree_5", "tree_15", "nonburn_dist", "nonburn_frac15", "dist_built", "dist_crop", "dist_water", "elev", "rel_elev", "tpi", "slope", "slope_along", "road_dist", "river_dist", "pop", "path_t"}, set()),
}
EDGES = [5, 8, 12, 16, 22, 30, 45, 80]
norm = lambda s: "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c)).lower()


class Data:
    def __init__(self, dirs, H, drop_spatial=DROP_SPATIAL, drop_scalar=DROP_SCALAR):
        self.X, self.S, self.Y, self.C, self.D, metas = [], [], [], [], [], []
        for i, p in enumerate(map(Path, dirs)):
            names = json.load(open(p / "names.json")); m = pd.read_parquet(p / "meta.parquet")
            self.X.append(np.load(p / "spatial.npy", mmap_mode="r")); self.S.append(np.load(p / f"scalars_{H}.npy"))
            self.Y.append(np.load(p / f"labels_{H}.npz")["y_modis"]); sh = np.load(p / "shared.npz"); self.C.append(sh["cand"]); self.D.append(sh["dist"])
            metas.append(m.assign(part=i, row=np.arange(len(m))))
        self.sp_idx = [k for k, n in enumerate(names["spatial"]) if n not in drop_spatial]; self.sp_names = [names["spatial"][k] for k in self.sp_idx]
        self.sc_idx = [k for k, n in enumerate(names["scalars"]) if n not in drop_scalar]; self.sc_names = [names["scalars"][k] for k in self.sc_idx]
        self.meta = pd.concat(metas, ignore_index=True)
        self.meta["st"] = self.meta.state.map(norm)
        self.H = H
        self.mu = self.sd = self.smu = self.ssd = None

    def raw(self, i):
        r = self.meta.iloc[i]; p, k = int(r.part), int(r.row)
        return np.asarray(self.X[p][k][self.sp_idx], np.float32), self.S[p][k][self.sc_idx], self.Y[p][k], self.C[p][k], self.D[p][k].astype(np.float32)

    def fit_norm(self, idx, n=300, seed=0):
        pick = np.random.default_rng(seed).choice(idx, min(n, len(idx)), replace=False)
        xs = np.stack([self.raw(i)[0] for i in pick]); ss = np.stack([self.raw(i)[1] for i in pick])
        self.mu, self.sd = np.nanmean(xs, axis=(0, 2, 3)), np.nanstd(xs, axis=(0, 2, 3)); self.sd = np.where(self.sd > 1e-6, self.sd, 1.0)
        self.smu, self.ssd = ss.mean(0), np.where(ss.std(0) > 1e-6, ss.std(0), 1.0)

    def get(self, i):
        x, s, y, c, d = self.raw(i)
        x = np.nan_to_num((x - self.mu[:, None, None]) / self.sd[:, None, None], nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        return x, ((s - self.smu) / self.ssd).astype(np.float32), y, c, d

    def split(self, scope, seed=0):
        m, ok = self.meta, self.meta[f"usable_{self.H}"].values
        if scope == "smoke":
            n = len(m); return {"train": np.arange(0, max(1, n - 2)), "val": np.arange(max(1, n - 2), n), "test": np.arange(max(1, n - 2), n)}
        tr = m.st.isin(V4_TRAIN).values
        return {"train": np.flatnonzero(tr & (m.year.values <= 2023) & ok), "val": np.flatnonzero(tr & (m.year.values == 2024) & ok), "test": np.flatnonzero(tr & (m.year.values == 2025) & ok),
                "blind": np.flatnonzero(m.st.isin(V4_BLIND).values & ok), "outside": np.flatnonzero(~tr & ~m.st.isin(V4_BLIND).values & ok)}


def augment(x, y, c, d, rng):
    k, f = int(rng.integers(4)), bool(rng.integers(2))
    if f:
        x, y, c, d = (a.flip(-1) for a in (x, y, c, d))
    return tuple(torch.rot90(a, k, (-2, -1)) for a in (x, y, c, d))


def score_set(model, data, idx, dev, tag, bs=8, pool=None):
    model.eval(); rows = []
    with torch.no_grad():
        for s in range(0, len(idx), bs):
            batch = [data.get(i) for i in idx[s:s + bs]]
            x = torch.from_numpy(np.stack([b[0] for b in batch])).to(dev); sc = torch.from_numpy(np.stack([b[1] for b in batch])).to(dev)
            p = torch.sigmoid(model(x, sc)).float().cpu().numpy()
            for j, b in enumerate(batch):
                m = b[3] > 0; y, d = b[2][m] > 0, b[4][m]
                if y.sum() < 5 or (~y).sum() < 50:
                    continue
                r = data.meta.iloc[idx[s + j]]
                for name, sco in (("model", p[j][m]), ("nearest", -d)):
                    if pool is not None:
                        z = pool.setdefault((tag, name), {"y": [], "s": [], "blk": []}); z["y"].append(y); z["s"].append(sco.astype(np.float32)); z["blk"].append(r.block)
                    top = np.argpartition(-sco, min(199, len(sco) - 1))[:200]
                    rows.append({"set": tag, "who": name, "case": r.case, "block": r.block, "state": r.state, "year": int(r.year), "dir_auc": scoring.auc_within_bins(sco, y, d, EDGES), "iou": scoring.iou_at_true_area(sco, y), "hits200": int(y[top].sum())})
    return rows


def summarize(rows):
    df = pd.DataFrame(rows); out = {}
    for tag, g in df.groupby("set"):
        near = g[g.who == "nearest"].set_index("case"); mod = g[g.who == "model"].set_index("case")
        res = {"n": len(mod)}
        for k in ("dir_auc", "iou", "hits200"):
            res[k] = scoring.bootstrap_mean(mod.groupby("block")[k].mean()); res[k + "_nearest"] = scoring.bootstrap_mean(near.groupby("block")[k].mean())
            res[k + "_gain"] = scoring.bootstrap_mean((mod[k] - near[k].reindex(mod.index)).groupby(mod.block).mean())
        out[tag] = res
    return out


def best_f1(y, s):
    """Threshold on the score that maximises F1 over the pooled cells."""
    o = np.argsort(-s, kind="stable"); ys = y[o]; tp = np.cumsum(ys); k = np.arange(1, len(ys) + 1); prec = tp / k; rec = tp / max(ys.sum(), 1)
    f1 = 2 * prec * rec / np.maximum(prec + rec, 1e-12); i = int(np.argmax(f1))
    return float(s[o][i]), float(f1[i]), float(prec[i]), float(rec[i])


def at_threshold(y, s, t):
    pred = s >= t; tp = float((pred & y).sum()); n_pred = float(pred.sum()); n_pos = float(y.sum())
    prec = tp / max(n_pred, 1); rec = tp / max(n_pos, 1)
    return {"precision": prec, "recall": rec, "f1": 2 * prec * rec / max(prec + rec, 1e-12), "iou": tp / max(n_pred + n_pos - tp, 1), "flagged_over_burned": n_pred / max(n_pos, 1)}


def pooled_metrics(pool, draws=200, seed=0):
    """Pooled (cell-level) metrics in the convention of the published next-day benchmarks, on the cells asked about (new burn only, nothing relabelled). AP = average precision over all pooled cells. F1, precision, recall, IoU at the threshold that maximises F1 on the 2024 validation cases ('val_threshold': nothing is tuned on the evaluated set) and, as the literature does, at the threshold that is best on the evaluated set itself ('oracle_threshold'). Interval for AP: 1-degree block bootstrap of the cases."""
    from sklearn.metrics import average_precision_score
    cat = lambda z, k: np.concatenate(z[k])
    thr = {w: best_f1(cat(pool[("val", w)], "y"), cat(pool[("val", w)], "s"))[0] for w in ("model", "nearest") if ("val", w) in pool}
    out, rng = {}, np.random.default_rng(seed)
    for (tag, who), z in pool.items():
        y, sc = cat(z, "y"), cat(z, "s"); r = {"n_cells": int(len(y)), "prevalence": float(y.mean()), "ap": float(average_precision_score(y, sc))}
        if who in thr:
            r["val_threshold"] = at_threshold(y, sc, thr[who])
        t, f1, pr, rc = best_f1(y, sc); r["oracle_threshold"] = {"f1": f1, "precision": pr, "recall": rc, **at_threshold(y, sc, t)}
        if tag in ("test", "blind"):
            blk = np.array(z["blk"]); ub = np.unique(blk); idx_by = {b: np.flatnonzero(blk == b) for b in ub}; ap = []
            for _ in range(draws):
                pick = np.concatenate([idx_by[b] for b in rng.choice(ub, len(ub))])
                yy = np.concatenate([z["y"][i] for i in pick]); ss = np.concatenate([z["s"][i] for i in pick])
                ap.append(average_precision_score(yy, ss))
            r["ap_ci95"] = [float(np.percentile(ap, 2.5)), float(np.percentile(ap, 97.5))]
        out[f"{tag}/{who}"] = r
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=""); ap.add_argument("--horizon", type=int, default=0); ap.add_argument("--parts", nargs="+", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--rescore", default="", help="RUN_DIR with a trained model.pt: no training, score it again (adds the pooled metrics) into --out")
    ap.add_argument("--epochs", type=int, default=30); ap.add_argument("--scope", default="v4"); ap.add_argument("--train-fires", type=int, default=0); ap.add_argument("--pretrained", type=int, default=1)
    ap.add_argument("--bs", type=int, default=8); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--drop", default="", help="comma-separated layer groups to leave out (ablation): "+", ".join(GROUPS))
    ap.add_argument("--sample", default="", help="data/sample_train.parquet: keep only the pre-registered training sample (evaluation sets are never thinned)"); ap.add_argument("--train-frac", type=float, default=1.0, help="nested subset of the sample for the learning curve")
    a = ap.parse_args()
    ck = torch.load(Path(a.rescore) / "model.pt", weights_only=False) if a.rescore else None
    if ck:
        a.model, a.horizon, a.pretrained, a.epochs, a.seed = ck["args"]["model"], ck["args"]["horizon"], ck["args"].get("pretrained", 1), 0, ck["args"].get("seed", 0)
    assert a.model and a.horizon, "--model and --horizon are needed unless --rescore is given"
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu"); rng = np.random.default_rng(a.seed); torch.manual_seed(a.seed)
    dsp, dsc = set(DROP_SPATIAL), set(DROP_SCALAR)
    for g in [x for x in a.drop.split(",") if x]:
        assert g in GROUPS, f"unknown group {g}"; dsp |= GROUPS[g][0]; dsc |= GROUPS[g][1]
    data = Data(a.parts, a.horizon, dsp, dsc); sp = data.split(a.scope)
    if a.sample:
        smp = pd.read_parquet(a.sample).drop_duplicates("event").set_index("event"); ev = data.meta.event.values[sp["train"]]
        ok = np.array([e in smp.index and bool(smp.keep[e]) and float(smp.u_frac[e]) < a.train_frac for e in ev]); sp["train"] = sp["train"][ok]
        miss = int(sum(e not in smp.index for e in ev)); assert miss == 0, f"{miss} training cases are not in the sample file"
    if a.train_fires and len(sp["train"]) > a.train_fires:
        sp["train"] = np.sort(rng.choice(sp["train"], a.train_fires, replace=False))
    print({k: len(v) for k, v in sp.items()}, "| layers", len(data.sp_names), "numbers", len(data.sc_names), "| device", dev, flush=True)
    if ck:
        data.mu, data.sd, data.smu, data.ssd = ck["mu"], ck["sd"], ck["smu"], ck["ssd"]; assert data.sp_names == ck["sp_names"] and data.sc_names == ck["sc_names"], "layers differ from the trained model"
    else:
        data.fit_norm(sp["train"])
    model = imgmodels.make(a.model, len(data.sp_names), len(data.sc_names), bool(a.pretrained)).to(dev)
    steps = max(1, a.epochs * int(np.ceil(len(sp["train"]) / a.bs)))
    opt = torch.optim.AdamW(model.parameters(), 1e-3, weight_decay=1e-4); sched = torch.optim.lr_scheduler.OneCycleLR(opt, 2e-3, total_steps=steps)
    scaler = torch.amp.GradScaler(enabled=dev.type == "cuda"); posw = torch.tensor(20.0, device=dev)
    best, hist, t0 = -1.0, [], time.time()
    for ep in range(0 if ck else a.epochs):
        model.train(); order = rng.permutation(sp["train"]); tot = 0.0
        for s in range(0, len(order), a.bs):
            batch = [data.get(i) for i in order[s:s + a.bs]]
            x = torch.from_numpy(np.stack([b[0] for b in batch])).to(dev); sc = torch.from_numpy(np.stack([b[1] for b in batch])).to(dev)
            y = torch.from_numpy(np.stack([b[2] for b in batch]).astype(np.float32)).to(dev); c = torch.from_numpy(np.stack([b[3] for b in batch]).astype(np.float32)).to(dev); d = torch.from_numpy(np.stack([b[4] for b in batch])).to(dev)
            x, y, c, d = augment(x, y, c, d, rng)
            with torch.autocast(device_type=dev.type, dtype=torch.float16, enabled=dev.type == "cuda"):
                logit = model(x, sc)
            loss = (F.binary_cross_entropy_with_logits(logit.float(), y, pos_weight=posw, reduction="none") * c).sum() / c.sum().clamp(min=1)
            opt.zero_grad(); scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step(); tot += float(loss)
        v = summarize(score_set(model, data, sp["val"], dev, "val"))["val"] if len(sp["val"]) else None
        hist.append({"epoch": ep, "loss": tot / max(1, len(order) // a.bs), "val_dir_auc": v["dir_auc"][0] if v else None, "val_iou": v["iou"][0] if v else None, "minutes": (time.time() - t0) / 60})
        print(hist[-1], flush=True)
        if v and v["dir_auc"][0] > best:
            best = v["dir_auc"][0]; torch.save({"model": model.state_dict(), "sp_names": data.sp_names, "sc_names": data.sc_names, "mu": data.mu, "sd": data.sd, "smu": data.smu, "ssd": data.ssd, "args": vars(a)}, out / "model.pt")
    if ck:
        model.load_state_dict(ck["model"])
    elif (out / "model.pt").exists():
        model.load_state_dict(torch.load(out / "model.pt", weights_only=False)["model"])
    rows, pool = [], {}
    for tag in ("val", "test", "blind", "outside"):
        if tag in sp and len(sp[tag]):
            rows += score_set(model, data, sp[tag], dev, tag, pool=pool)
    pooled = pooled_metrics(pool)
    pd.DataFrame(rows).to_csv(out / "per_case.csv", index=False)
    res = summarize(rows); json.dump({"args": vars(a), "sizes": {k: len(v) for k, v in sp.items()}, "history": hist, "summary": res, "pooled": pooled, "layers": data.sp_names, "numbers": data.sc_names}, open(out / "results.json", "w"), indent=1, default=str)
    for tag, r in res.items():
        print(f"{tag:8s} n={r['n']:4d} direction AUC {r['dir_auc'][0]:.3f} (nearest {r['dir_auc_nearest'][0]:.3f}, gain {r['dir_auc_gain'][0]:+.3f} [{r['dir_auc_gain'][1]:+.3f}, {r['dir_auc_gain'][2]:+.3f}])  IoU {r['iou'][0]:.3f} (nearest {r['iou_nearest'][0]:.3f})  hits200 {r['hits200'][0]:.0f} (nearest {r['hits200_nearest'][0]:.0f})", flush=True)

    for key, r in pooled.items():
        if key.startswith(("test/", "blind/")):
            vt = r.get("val_threshold", {})
            print(f"pooled {key:15s} cells {r['n_cells']:8d} prevalence {r['prevalence']:.4f} AP {r['ap']:.3f} {r.get('ap_ci95', '')} | val-threshold F1 {vt.get('f1', float('nan')):.3f} P {vt.get('precision', float('nan')):.3f} R {vt.get('recall', float('nan')):.3f} IoU {vt.get('iou', float('nan')):.3f} | oracle F1 {r['oracle_threshold']['f1']:.3f}", flush=True)


if __name__ == "__main__":
    main()
