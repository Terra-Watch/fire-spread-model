"""Replay check for the live path: rebuild a research case from its detections only (no truth, no saved arrays) with live.make_bundle / live.score_bundle and compare with scoring the saved research bundle.
Run: .venv/bin/python scripts/27_replay_parity.py LINES.geojsonseq [n_cases]"""
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from firespread import ancillary, cases, config, grid as G, live, rank  # noqa: E402


def research_scores(d, H, model_path):
    import joblib
    b = joblib.load(model_path)
    c = cases.load_case(d, gap_override=H)
    ant = pd.read_parquet(config.DATA / "antecedent.parquet").set_index("case").join(pd.read_parquet(config.DATA / "soil.parquet").set_index("case"), how="inner").loc[c.meta["case_id"]]
    full = rank.build(c, d, ant)
    cand = c.candidates & (c.dist_cells >= live.START_ZONE_CELLS)
    X = np.column_stack([np.broadcast_to(full[k], c.fuel.shape)[cand] for k in b["features"]])
    s = np.full(c.fuel.shape, np.nan, np.float32)
    s[cand] = b["model"].predict_proba(X)[:, 1]
    return s


def main(lines_path, n=3):
    H = 24
    model = config.DATA / "models" / f"rank2_{H}h.joblib"
    dirs = sorted(p.parent for p in (config.DATA / "bank").glob("*/burn_long.npz"))[::97][:n]
    work = config.DATA / "replay"
    for d in dirs:
        meta = json.load(open(d / "meta.json"))
        t0 = pd.Timestamp(meta["t0"])
        det = pd.read_parquet(config.DATA / "detections" / f"{t0.year}.parquet")
        cp = pd.read_parquet(config.DATA / "cases.parquet")
        row = cp[(cp.event == meta["event"]) & (cp.t0 == t0)].iloc[0]
        fire = det[(det["event"] == meta["event"]) & (det["epoch"] <= int(row.epoch_from))].reset_index(drop=True)
        recent = det[(det["t"] >= t0 - pd.Timedelta(hours=72)) & (det["t"] < t0)][["latitude", "longitude", "t", "event"]]
        out = work / f"replay-{d.name}"
        if out.exists():
            shutil.rmtree(out)
        g = G.make_grid(float(fire.latitude.mean()), float(fire.longitude.mean()))
        lines = ancillary.load_lines(lines_path, ancillary.tiles_of(g))
        live.make_bundle(fire, t0, H, out, lines, weather_kind="hist")
        res = live.score_bundle(out, model, recent=recent, weather_kind="hist")
        ref = research_scores(d, H, model)
        both = np.isfinite(res["score"]) & np.isfinite(ref)
        a, b = res["score"][both], ref[both]
        ra, rb = a.argsort().argsort(), b.argsort().argsort()
        print(d.name, f"cells {both.sum()} (live {np.isfinite(res['score']).sum()}, research {np.isfinite(ref).sum()}), max |diff| {np.abs(a - b).max():.4f}, rank corr {np.corrcoef(ra, rb)[0, 1]:.4f}",
              f"top-2% overlap {len(set(np.flatnonzero(a >= np.quantile(a, .98))) & set(np.flatnonzero(b >= np.quantile(b, .98)))) / max(1, int((a >= np.quantile(a, .98)).sum())):.3f}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 3)
