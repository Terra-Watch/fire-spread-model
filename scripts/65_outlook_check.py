"""How the served outlook compares with what actually burned, fire by fire (docs/23). Scores every test fire (2025, 250) with the same six models and the same rank rule as 62_outlook_run.py
and writes data/65_outlook_check.csv: for each fire the area of the top band, the share of what burned that the top band (and the top three bands) catch, the share of the top band that burned,
and the same for "the nearest cells first" at the same size. The truth is the MODIS burned area within 48 h.
Run: .venv/bin/python scripts/65_outlook_check.py --release release/fire_spread_dataset --models ../paper_fire_spread/results/models [--limit N]"""
import argparse, glob, importlib.util, sys, time
from pathlib import Path
import numpy as np, pandas as pd, torch
from scipy.stats import rankdata
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "src")); sys.argv, argv = sys.argv[:1], sys.argv[1:]
spec = importlib.util.spec_from_file_location("tr", ROOT / "scripts" / "61_train.py"); tr = importlib.util.module_from_spec(spec); spec.loader.exec_module(tr)
from firespread import imgmodels
ap = argparse.ArgumentParser(); ap.add_argument("--release", required=True); ap.add_argument("--models", required=True); ap.add_argument("--limit", type=int, default=0); ap.add_argument("--out", default=str(ROOT / "data/65_outlook_check.csv"))
a = ap.parse_args(argv); torch.set_num_threads(4)
P = {48: 0.0293}; BANDS = (P[48], 3 * P[48], 7 * P[48])
dirs = [f"{a.release}/part_p{i}" for i in range(6)]; D = {h: tr.Data(dirs, h) for h in (24, 48)}; MODELS = {}
for h in (24, 48):
    ckpts = [torch.load(p, weights_only=False, map_location="cpu") for p in sorted(glob.glob(f"{a.models}/fsn_{h}_s*.pt"))]; ms = []
    for ck in ckpts:
        m_ = imgmodels.make("firesensenet", len(ck["sp_names"]), len(ck["sc_names"]), pretrained=False); m_.load_state_dict(ck["model"]); m_.eval(); ms.append(m_)
    D[h].sp_names, D[h].sc_names = ckpts[0]["sp_names"], ckpts[0]["sc_names"]; MODELS[h] = (ckpts, ms)
def score_of(h, i):
    cks, ms = MODELS[h]; d = D[h]; out = []
    for ck, mod in zip(cks, ms):
        d.mu, d.sd, d.smu, d.ssd = ck["mu"], ck["sd"], ck["smu"], ck["ssd"]; x, s, _, _, _ = d.get(i)
        with torch.no_grad(): out.append(torch.sigmoid(mod(torch.from_numpy(x)[None], torch.from_numpy(s)[None]))[0].numpy())
    return np.mean(out, axis=0)
meta = D[48].meta.merge(pd.read_csv(f"{a.release}/cases.csv", usecols=["case", "study_split"]), on="case", how="left"); ids = np.flatnonzero((meta.study_split == "test").values & meta.usable_48.values.astype(bool))
if a.limit: ids = ids[:a.limit]
rows = []; t0 = time.time()
for n, i in enumerate(ids):
    _, _, y, cand, dist = D[48].raw(i); m = cand > 0; idx = np.flatnonzero(m.ravel()); truth = (y.ravel()[idx] > 0)
    if truth.sum() < 10: continue
    both = rankdata(-score_of(24, i).ravel()[idx]) + rankdata(-score_of(48, i).ravel()[idx]); order = np.argsort(both, kind="stable"); near = np.argsort(dist.ravel()[idx], kind="stable")
    r = meta.iloc[i]; row = {"case": r.case, "state": r.state, "lat": r.lat, "lon": r.lon, "t0": r.t0, "n_start": int(r.n_start), "cells": len(idx), "burned_cells": int(truth.sum()), "res_m": float(r.res)}
    for tag, o in (("model", order), ("nearest", near)):
        for b, share in zip((1, 3), (BANDS[0], BANDS[1])):
            k = int(round(share * len(idx))); hit = int(truth[o[:k]].sum()); row[f"{tag}_b{b}_cells"] = k; row[f"{tag}_b{b}_capture"] = hit / truth.sum(); row[f"{tag}_b{b}_precision"] = hit / max(k, 1)
    rows.append(row)
    if n % 10 == 0: print(n, len(ids), round(time.time() - t0), "s", flush=True)
df = pd.DataFrame(rows); df.to_csv(a.out, index=False); print(len(df), "fires ->", a.out)
for tag in ("model", "nearest"): print(tag, "top band: median capture", round(df[f"{tag}_b1_capture"].median(), 3), "median precision", round(df[f"{tag}_b1_precision"].median(), 3), "| three bands: capture", round(df[f"{tag}_b3_capture"].median(), 3))
