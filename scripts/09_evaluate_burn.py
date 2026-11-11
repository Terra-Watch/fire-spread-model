"""Like 06, but the truth is the MODIS burn date. Run: .venv/bin/python scripts/09_evaluate_burn.py [dev|test|all] [strict]"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from firespread import burn_eval, cases, config, scoring  # noqa: E402


def run(split="dev", strict=False, connected=True, **kw):
    rows = []
    for d in sorted((config.DATA / "cases").glob("*/burn.npz")):
        try:
            c = cases.load_case(d.parent)
        except cases.NoWeather:
            continue
        if split != "all" and c.meta["split"] != split:
            continue
        b = np.load(d)
        truth, cand, ok = burn_eval.burn_masks(c, b["doy"], int(b["year"]), strict)
        if not ok:
            continue
        if connected:
            truth, cand = burn_eval.connected_truth(c, truth, cand)
            if truth.sum() < 5:
                continue
        r = {"case": c.id, "block": f"{int(np.floor(c.meta['lat']))}_{int(np.floor(c.meta['lon']))}", "year": c.meta["year"]}
        for name, sc in (("model", cases.scores_from_arrival(c, cases.forecast_arrival(c, **kw))), ("prox", cases.baseline_proximity(c)), ("wind", cases.baseline_wind_ellipse(c))):
            for k, v in burn_eval.evaluate_burn(c, sc, truth, cand).items():
                r[f"{name}_{k}"] = v
        rows.append(r)
    return pd.DataFrame(rows)


def summarize(df):
    out = []
    for m in ("model", "prox", "wind"):
        a, b, c = (scoring.bootstrap_mean(df.groupby("block")[f"{m}_{k}"].mean()) for k in ("auc", "dir_auc", "area50"))
        out.append(f"{m:6s} AUC {a[0]:.3f} [{a[1]:.3f}, {a[2]:.3f}]   direction AUC {b[0]:.3f} [{b[1]:.3f}, {b[2]:.3f}]   area for 50 % {c[0]:.3f} [{c[1]:.3f}, {c[2]:.3f}]")
    for base in ("prox", "wind"):
        for key, label in (("auc", "AUC"), ("dir_auc", "direction AUC")):
            d = (df[f"model_{key}"] - df[f"{base}_{key}"]).groupby(df.block).mean()
            m, lo, hi = scoring.bootstrap_mean(d)
            out.append(f"model - {base:5s} {label:14s} {m:+.3f} [{lo:+.3f}, {hi:+.3f}]  (model higher in {(df[f'model_{key}'] > df[f'{base}_{key}']).mean():.2f} of cases)")
    return "\n".join(out)


if __name__ == "__main__":
    split = sys.argv[1] if len(sys.argv) > 1 else "dev"
    strict = "strict" in sys.argv
    df = run(split, strict, connected="loose" not in sys.argv)
    print(len(df), "usable cases,", df.block.nunique() if len(df) else 0, "blocks, split =", split, "strict" if strict else "")
    if len(df) > 5:
        print(df[["model_n_pos", "model_n_cand"]].describe().loc[["25%", "50%", "75%"]].round(0).to_dict())
        print(summarize(df))
