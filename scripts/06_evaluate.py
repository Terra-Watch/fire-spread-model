"""Run the engine and the baselines on the case bundles and compare them. dev cases may be used to tune; test cases only to report.
Run: .venv/bin/python scripts/06_evaluate.py [dev|test|all]"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from firespread import cases, config, scoring  # noqa: E402


def run(split="dev", **kw):
    rows = []
    for d in sorted((config.DATA / "cases").glob("*/meta.json")):
        try:
            c = cases.load_case(d.parent)
        except cases.NoWeather:
            continue
        if split != "all" and c.meta["split"] != split:
            continue
        if c.truth.sum() == 0:
            continue
        r = {"case": c.id, "block": f"{int(np.floor(c.meta['lat']))}_{int(np.floor(c.meta['lon']))}", "state": c.meta["state"], "wx": c.weather_source}
        arr = cases.forecast_arrival(c, **kw)
        r["reach_pred_km"] = cases.predicted_reach_km(c, arr)
        r["reach_obs_km"], r["n_start"] = c.extra["adv_p95_km"], c.extra["n_start"]
        for name, sc in (("model", cases.scores_from_arrival(c, arr)), ("prox", cases.baseline_proximity(c)), ("wind", cases.baseline_wind_ellipse(c))):
            for k, v in cases.evaluate(c, sc).items():
                r[f"{name}_{k}"] = v
        rows.append(r)
    return pd.DataFrame(rows)


def summarize(df):
    out = []
    for m in ("model", "prox", "wind"):
        a = scoring.bootstrap_mean(df.groupby("block")[f"{m}_auc"].mean())
        b = scoring.bootstrap_mean(df.groupby("block")[f"{m}_area50"].mean())
        c = scoring.bootstrap_mean(df.groupby("block")[f"{m}_dir_auc"].mean())
        out.append(f"{m:6s} AUC {a[0]:.3f} [{a[1]:.3f}, {a[2]:.3f}]   direction AUC {c[0]:.3f} [{c[1]:.3f}, {c[2]:.3f}]   area for 50 % of growth {b[0]:.3f} [{b[1]:.3f}, {b[2]:.3f}]")
    for base in ("prox", "wind"):
        for key, label in (("auc", "AUC"), ("dir_auc", "direction AUC")):
            d = (df[f"model_{key}"] - df[f"{base}_{key}"]).groupby(df.block).mean()
            m, lo, hi = scoring.bootstrap_mean(d)
            out.append(f"model - {base:5s} {label:14s} difference {m:+.3f} [{lo:+.3f}, {hi:+.3f}]  (cases where the model is higher: {(df[f'model_{key}'] > df[f'{base}_{key}']).mean():.2f})")
    from scipy.stats import spearmanr
    ok = df.dropna(subset=["reach_pred_km", "reach_obs_km"])
    if len(ok) > 5:
        out.append(f"how far (95th percentile advance, km): Spearman model reach vs observed {spearmanr(ok.reach_pred_km, ok.reach_obs_km)[0]:+.2f}; start size vs observed {spearmanr(ok.n_start, ok.reach_obs_km)[0]:+.2f}")
    return "\n".join(out)


if __name__ == "__main__":
    split = sys.argv[1] if len(sys.argv) > 1 else "dev"
    df = run(split)
    print(len(df), "cases,", df.block.nunique(), "blocks, split =", split)
    print(summarize(df))
