"""Which parts of the physics engine help? Switch wind, slope, moisture and the fuel values on and off and score each against the burn dates (pilot cases, dev and test together because nothing is fitted).
Run: .venv/bin/python scripts/15_engine_variants.py"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from firespread import burn_eval, cases, config, engine, scoring  # noqa: E402

P = engine.FuelTable.priors


def flat_fuel(rate):
    t = P(); t.r0 = np.array([0, rate, rate, rate, rate], float); t.cw = np.zeros(5); return t


def crop_like_forest():
    t = P(); t.r0[engine.CROP] = t.r0[engine.FOREST]; return t


VARIANTS = {
    "proximity": None,
    "engine (priors)": dict(),
    "no wind": dict(wind_scale=0.0),
    "no slope": dict(beta_s=0.0),
    "no wind, no slope": dict(wind_scale=0.0, beta_s=0.0),
    "no wind, no slope, crop = forest": dict(wind_scale=0.0, beta_s=0.0, fuels=crop_like_forest()),
    "uniform speed, fuel gates only": dict(wind_scale=0.0, beta_s=0.0, fuels=flat_fuel(60.0)),
    "fast spread (x3)": dict(rate_scale=3.0),
}


def main():
    items = []
    for d in sorted((config.DATA / "cases").glob("*/burn.npz")):
        try:
            c = cases.load_case(d.parent)
        except cases.NoWeather:
            continue
        b = np.load(d)
        truth, cand, ok = burn_eval.burn_masks(c, b["doy"], int(b["year"]))
        if ok:
            truth, cand = burn_eval.connected_truth(c, truth, cand)
            if truth.sum() >= 5:
                items.append((c, truth, cand))
    print(len(items), "cases")
    base = None
    for name, kw in VARIANTS.items():
        rows = []
        for c, truth, cand in items:
            sc = cases.baseline_proximity(c) if kw is None else cases.scores_from_arrival(c, cases.forecast_arrival(c, **kw))
            r = burn_eval.evaluate_burn(c, sc, truth, cand)
            r["block"] = f"{int(np.floor(c.meta['lat']))}_{int(np.floor(c.meta['lon']))}"
            r["case"] = c.id
            rows.append(r)
        df = pd.DataFrame(rows).set_index("case")
        base = df if base is None else base
        d = (df.dir_auc - base.dir_auc).groupby(df.block).mean()
        m, lo, hi = scoring.bootstrap_mean(d)
        a = scoring.bootstrap_mean(df.groupby("block").auc.mean())
        print(f"{name:34s} AUC {a[0]:.3f}  direction AUC {scoring.bootstrap_mean(df.groupby('block').dir_auc.mean())[0]:.3f}  vs proximity {m:+.3f} [{lo:+.3f}, {hi:+.3f}]")


if __name__ == "__main__":
    main()
