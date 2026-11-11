"""A published-style spatial network (small U-Net) on the same maps, against the boosted-tree ranking and plain proximity, same dev folds (5-fold by block), same scoring (direction AUC, IoU at the true area).
Needs torch, scikit-learn, numpy and firespread/scoring.py only.   Run: python scripts/32_unet.py MAPS.npz [epochs] [out.json]"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as Fn
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from firespread import scoring  # noqa: E402

EDGES = [5, 8, 12, 16, 22, 30, 45, 80]
NEG = 12


def block(i, o):
    return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(inplace=True), nn.Conv2d(o, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(inplace=True))


class UNet(nn.Module):
    def __init__(self, cin, w=24):
        super().__init__()
        ch = [w, 2 * w, 4 * w, 8 * w]
        self.enc = nn.ModuleList([block(cin, ch[0]), block(ch[0], ch[1]), block(ch[1], ch[2]), block(ch[2], ch[3])])
        self.dec = nn.ModuleList([block(ch[3] + ch[2], ch[2]), block(ch[2] + ch[1], ch[1]), block(ch[1] + ch[0], ch[0])])
        self.out = nn.Conv2d(ch[0], 1, 1)

    def forward(self, x):
        skips = []
        for i, e in enumerate(self.enc):
            x = e(x if i == 0 else Fn.max_pool2d(x, 2))
            skips.append(x)
        for d, s in zip(self.dec, skips[-2::-1]):
            x = d(torch.cat([Fn.interpolate(x, size=s.shape[-2:], mode="bilinear", align_corners=False), s], 1))
        return self.out(x)[:, 0]


def standardise(X, idx):
    sub = X[idx].astype(np.float32)
    mu, sd = np.nanmean(sub, axis=(0, 2, 3)), np.nanstd(sub, axis=(0, 2, 3))
    return mu.astype(np.float32), np.where(sd > 1e-6, sd, 1.0).astype(np.float32)


def batch(X, i, mu, sd, dev):
    x = (X[i].astype(np.float32) - mu[None, :, None, None]) / sd[None, :, None, None]
    return torch.from_numpy(np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)).to(dev)


def train_unet(X, y, cand, tr, epochs, dev, seed=0):
    torch.manual_seed(seed)
    mu, sd = standardise(X, tr)
    net = UNet(X.shape[1]).to(dev)
    opt = torch.optim.AdamW(net.parameters(), 1e-3, weight_decay=1e-4)
    steps = epochs * int(np.ceil(len(tr) / 8))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 2e-3, total_steps=steps)
    rng = np.random.default_rng(seed)
    for ep in range(epochs):
        net.train()
        order = rng.permutation(tr)
        for s in range(0, len(order), 8):
            i = np.sort(order[s:s + 8])
            x, t, m = batch(X, i, mu, sd, dev), torch.from_numpy(y[i].astype(np.float32)).to(dev), torch.from_numpy(cand[i].astype(np.float32)).to(dev)
            k, f = int(rng.integers(4)), bool(rng.integers(2))                # the channels have no absolute direction in them, so turns and flips are valid
            if f:
                x, t, m = x.flip(-1), t.flip(-1), m.flip(-1)
            x, t, m = torch.rot90(x, k, (-2, -1)), torch.rot90(t, k, (-2, -1)), torch.rot90(m, k, (-2, -1))
            loss = (Fn.binary_cross_entropy_with_logits(net(x), t, pos_weight=torch.tensor(20.0, device=dev), reduction="none") * m).sum() / m.sum().clamp(min=1)
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    return net, mu, sd


def predict_unet(net, X, i, mu, sd, dev):
    net.eval()
    with torch.no_grad():
        return torch.sigmoid(net(batch(X, np.array([i]), mu, sd, dev)))[0].cpu().numpy()


def fit_gbm(X, y, cand, tr, rng):
    xs, ys = [], []
    for i in tr:
        pos, neg = np.flatnonzero(y[i][cand[i] > 0]), np.flatnonzero(y[i][cand[i] > 0] == 0)
        neg = rng.choice(neg, min(len(neg), NEG * len(pos)), replace=False)
        idx = np.r_[pos, neg]
        xs.append(X[i][:, cand[i] > 0][:, idx].T.astype(np.float32)); ys.append(np.r_[np.ones(len(pos)), np.zeros(len(neg))])
    yy = np.concatenate(ys)
    return HistGradientBoostingClassifier(max_depth=5, learning_rate=0.06, max_iter=300, l2_regularization=1.0, random_state=0).fit(np.vstack(xs), yy, sample_weight=np.where(yy == 1, NEG, 1.0))


def main(path, epochs=25, out=None):
    z = np.load(path)
    X, y, cand, dist = z["X"], z["y"], z["cand"], z["dist"].astype(np.float32)
    dev_i = np.flatnonzero(z["split"] == "dev")[:int(os.environ.get("LIMIT", 10 ** 9))]       # LIMIT is for a quick test
    blocks = z["block"][dev_i]
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = np.random.default_rng(0)
    rows = []
    print(f"{len(dev_i)} dev cases, {X.shape[1]} channels, device {dev}", flush=True)
    for fold, (tr, te) in enumerate(GroupKFold(5).split(dev_i, groups=blocks)):
        tr, te = dev_i[tr], dev_i[te]
        net, mu, sd = train_unet(X, y, cand, tr, epochs, dev)
        gbm = fit_gbm(X, y, cand, tr, rng)
        for i in te:
            m = cand[i] > 0
            yt, d = y[i][m] > 0, dist[i][m]
            scores = {"prox": -d, "unet": predict_unet(net, X, i, mu, sd, dev)[m], "gbm": gbm.predict_proba(X[i][:, m].T.astype(np.float32))[:, 1]}
            scores["avg"] = scores["unet"].argsort().argsort() + scores["gbm"].argsort().argsort()
            for tag, s in scores.items():
                rows.append({"case": str(z["case"][i]), "block": str(z["block"][i]), "set": tag, "dir_auc": scoring.auc_within_bins(s, yt, d, EDGES), "iou": scoring.iou_at_true_area(s, yt)})
        print("fold", fold, "done", flush=True)
    res = {}
    base = {r["case"]: r for r in rows if r["set"] == "prox"}
    for tag in ("prox", "gbm", "unet", "avg"):
        g = [r for r in rows if r["set"] == tag]
        blk = sorted({r["block"] for r in g})
        def per(f):
            return np.array([np.mean([f(r, base[r["case"]]) for r in g if r["block"] == bk]) for bk in blk])

        d, i = scoring.bootstrap_mean(per(lambda r, b: r["dir_auc"])), scoring.bootstrap_mean(per(lambda r, b: r["iou"]))
        dd, di = scoring.bootstrap_mean(per(lambda r, b: r["dir_auc"] - b["dir_auc"])), scoring.bootstrap_mean(per(lambda r, b: r["iou"] - b["iou"]))
        print(f"  {tag:5s} direction AUC {d[0]:.3f} [{d[1]:.3f}, {d[2]:.3f}]  IoU {i[0]:.3f} [{i[1]:.3f}, {i[2]:.3f}]   vs prox: direction {dd[0]:+.3f} [{dd[1]:+.3f}, {dd[2]:+.3f}], IoU {di[0]:+.3f} [{di[1]:+.3f}, {di[2]:+.3f}]", flush=True)
        res[tag] = {"dir_auc": d, "iou": i, "dir_auc_gain": dd, "iou_gain": di}
    if out:
        json.dump(res, open(out, "w"))


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 25, sys.argv[3] if len(sys.argv) > 3 else None)
