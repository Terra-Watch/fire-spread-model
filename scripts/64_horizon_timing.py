"""Read-only: (1) does averaging the two horizon models help, (2) do the models carry any timing information (can they tell cells that burn after 24 h from those that burn before)?
Run: .venv/bin/python scripts/64_horizon_timing.py"""
import glob, importlib.util, sys
from pathlib import Path
import numpy as np, pandas as pd, torch
from sklearn.metrics import roc_auc_score
from scipy.stats import rankdata
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "src")); sys.argv = sys.argv[:1]
spec = importlib.util.spec_from_file_location("tr", ROOT / "scripts" / "61_train.py"); tr = importlib.util.module_from_spec(spec); spec.loader.exec_module(tr)
from firespread import imgmodels, scoring
REL = ROOT / "release/fire_spread_dataset"; MOD = ROOT.parent / "paper_fire_spread/results/models"; dirs = [str(REL / f"part_p{i}") for i in range(6)]
D = {h: tr.Data(dirs, h) for h in (24, 48)}; M = {}
for h in (24, 48):
    cks = [torch.load(p, weights_only=False, map_location="cpu") for p in sorted(glob.glob(f"{MOD}/fsn_{h}_s*.pt"))]; ms = []
    for ck in cks:
        m = imgmodels.make("firesensenet", len(ck["sp_names"]), len(ck["sc_names"]), pretrained=False); m.load_state_dict(ck["model"]); m.eval(); ms.append(m)
    M[h] = (cks, ms)
def score(h, i):
    cks, ms = M[h]; d = D[h]; out = []
    for ck, m in zip(cks, ms):
        d.mu, d.sd, d.smu, d.ssd = ck["mu"], ck["sd"], ck["smu"], ck["ssd"]; x, s, y, c, dist = d.get(i)
        with torch.no_grad(): out.append(torch.sigmoid(m(torch.from_numpy(x)[None], torch.from_numpy(s)[None]))[0].numpy())
    return np.mean(out, 0)
sp = D[24].split("v4"); rows = []; late = []
for tag in ("test", "blind"):
    for i in sp[tag]:
        _, _, y24, c, dist = D[24].raw(i); y48 = D[48].raw(i)[2]; m = c > 0; d = dist[m]; a, b = y24[m] > 0, y48[m] > 0
        if a.sum() < 5 or b.sum() < 5 or (~b).sum() < 50: continue
        s24, s48 = score(24, i)[m], score(48, i)[m]; avg = rankdata(s24) + rankdata(s48); blk = D[24].meta.iloc[i].block
        rows.append(dict(set=tag, block=blk, auc24=scoring.auc_within_bins(s24, b, d, tr.EDGES), auc48=scoring.auc_within_bins(s48, b, d, tr.EDGES), aucavg=scoring.auc_within_bins(avg, b, d, tr.EDGES),
                         iou24=scoring.iou_at_true_area(s24, b), iou48=scoring.iou_at_true_area(s48, b), iouavg=scoring.iou_at_true_area(avg, b)))
        lt = b & ~a
        if lt.sum() >= 20 and a.sum() >= 20:          # among cells that burned within 48 h: can anything tell the late ones (after 24 h) from the early ones?
            yb = lt[b]; r24, r48 = rankdata(s24)[b], rankdata(s48)[b]
            late.append(dict(set=tag, block=blk, n_late=int(lt.sum()), n_early=int(a.sum()), by_rank_diff=roc_auc_score(yb, r48 - r24), by_distance=roc_auc_score(yb, d[b]), by_low_24h_score=roc_auc_score(yb, -s24[b])))
R = pd.DataFrame(rows); L = pd.DataFrame(late)
for tag, g in R.groupby("set"):
    bm = lambda c: g.groupby("block")[c].mean().mean()
    print(f"{tag} ({len(g)} fires), scored on what burned within 48 h: direction AUC 24h-model {bm('auc24'):.3f}, 48h-model {bm('auc48'):.3f}, both averaged {bm('aucavg'):.3f} | IoU {bm('iou24'):.3f}, {bm('iou48'):.3f}, {bm('iouavg'):.3f}")
for tag, g in L.groupby("set"):
    print(f"{tag}: {len(g)} fires with at least 20 early and 20 late burned cells. AUC for telling late from early burned cells: 48h-minus-24h rank {g.by_rank_diff.mean():.3f}, distance from the fire {g.by_distance.mean():.3f}, low 24h score {g.by_low_24h_score.mean():.3f}")
