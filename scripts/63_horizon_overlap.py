"""Read-only check: how different are the 24 h and 48 h models and labels? Scores the 2025 test and blind-state fires with both models against both truths (docs/results.md, "One outlook, not two horizons").
Run: .venv/bin/python scripts/63_horizon_overlap.py   (needs release/fire_spread_dataset and ../paper_fire_spread/results/models)"""
import glob, importlib.util, json, sys
from pathlib import Path
import numpy as np, pandas as pd, torch
from scipy.stats import spearmanr
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
sp = D[24].split("v4"); rows = []
for tag in ("test", "blind"):
    for i in sp[tag]:
        _, _, y24, c, dist = D[24].raw(i); y48 = D[48].raw(i)[2]; m = c > 0; d = dist[m]; a, b = y24[m] > 0, y48[m] > 0
        if a.sum() < 5 or b.sum() < 5 or (~b).sum() < 50: continue
        s24, s48 = score(24, i)[m], score(48, i)[m]; n = len(d); r = dict(set=tag, case=D[24].meta.iloc[i].case, block=D[24].meta.iloc[i].block, n24=int(a.sum()), n48=int(b.sum()))
        r["rho"] = spearmanr(s24, s48).statistic
        for name, k in (("top_own", None), ("top3pct", int(0.03 * n)), ("top10pct", int(0.10 * n))):
            ka = k or int(0.0218 * n); kb = k or int(0.0293 * n); A = set(np.argpartition(-s24, ka)[:ka]); B = set(np.argpartition(-s48, kb)[:kb]); r[name + "_iou"] = len(A & B) / max(len(A | B), 1); r[name + "_24in48"] = len(A & B) / max(len(A), 1)
        for mn, s in (("m24", s24), ("m48", s48), ("near", -d)):
            for tn, y in (("y24", a), ("y48", b)):
                r[f"auc_{mn}_{tn}"] = scoring.auc_within_bins(s, y, d, tr.EDGES); r[f"iou_{mn}_{tn}"] = scoring.iou_at_true_area(s, y)
        rows.append(r)
R = pd.DataFrame(rows); R.to_csv(ROOT / "data" / "horizon_overlap_cases.csv", index=False)
for tag, g in R.groupby("set"):
    print(f"\n== {tag}: {len(g)} fires; truth identical at 24 and 48 h in {(g.n24 == g.n48).mean():.0%}")
    print(f"rank correlation of the two maps: median {g.rho.median():.3f}, 10th percentile {g.rho.quantile(.1):.3f}")
    print(f"top band (own size) overlap IoU {g.top_own_iou.mean():.2f}; share of 24 h top band inside 48 h top band {g.top_own_24in48.mean():.2f}; same-size top 3% IoU {g.top3pct_iou.mean():.2f}; top 10% IoU {g.top10pct_iou.mean():.2f}")
    bm = lambda c: g.groupby("block")[c].mean().mean()
    for tn in ("y24", "y48"):
        print(f" truth {tn}: direction AUC  24h-model {bm('auc_m24_'+tn):.3f}  48h-model {bm('auc_m48_'+tn):.3f}  nearest {bm('auc_near_'+tn):.3f} | IoU  24h-model {bm('iou_m24_'+tn):.3f}  48h-model {bm('iou_m48_'+tn):.3f}  nearest {bm('iou_near_'+tn):.3f}")
    for tn in ("y24", "y48"):
        dd = (g[f"auc_m48_{tn}"] - g[f"auc_m24_{tn}"]).groupby(g.block).mean(); bs = [dd.sample(len(dd), replace=True, random_state=k).mean() for k in range(1000)]
        di = (g[f"iou_m48_{tn}"] - g[f"iou_m24_{tn}"]).groupby(g.block).mean(); bi = [di.sample(len(di), replace=True, random_state=k).mean() for k in range(1000)]
        print(f" 48h-model minus 24h-model on {tn}: direction AUC {dd.mean():+.3f} [{np.percentile(bs,2.5):+.3f}, {np.percentile(bs,97.5):+.3f}]  IoU {di.mean():+.3f} [{np.percentile(bi,2.5):+.3f}, {np.percentile(bi,97.5):+.3f}]")
