"""Fit the engine's relative parameters on the dev cases (random search, scored on the burn dates), then look once at the test cases.
Ranking scores ignore a uniform change of speed, so this fits the RELATIVE effects (fuel classes, wind, slope, moisture); the overall speed is set afterwards from how far fires really advanced.
Run: .venv/bin/python scripts/16_calibrate.py [n_trials] [dirs...]"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from firespread import burn_eval, cases, config, engine, scoring  # noqa: E402


def load(dirs):
    items = []
    for root in dirs:
        for d in sorted((config.DATA / root).glob("*/burn.npz")):
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
    return items


def table(theta):
    t = engine.FuelTable.priors()
    rel = np.array([0, 1.0, theta["shrub"], theta["grass"], theta["crop"]])
    t.r0 = t.r0[engine.FOREST] * rel
    t.mx = t.mx * theta["mx"]
    return t


def score(items, theta):
    rows = []
    for c, truth, cand in items:
        arr = cases.forecast_arrival(c, fuels=table(theta), beta_s=theta["beta"], wind_scale=theta["wind"])
        r = burn_eval.evaluate_burn(c, cases.scores_from_arrival(c, arr), truth, cand)
        r["block"] = f"{int(np.floor(c.meta['lat']))}_{int(np.floor(c.meta['lon']))}"
        r["case"] = c.id
        r["reach"] = cases.predicted_reach_km(c, arr)
        rows.append(r)
    return pd.DataFrame(rows)


def main(n, dirs):
    items = load(dirs)
    dev = [x for x in items if x[0].meta["split"] == "dev"]
    test = [x for x in items if x[0].meta["split"] == "test"]
    print(len(dev), "dev,", len(test), "test cases")
    prox = pd.DataFrame([{**burn_eval.evaluate_burn(c, cases.baseline_proximity(c), t, k), "case": c.id, "block": f"{int(np.floor(c.meta['lat']))}_{int(np.floor(c.meta['lon']))}"} for c, t, k in dev]).set_index("case")
    rng = np.random.default_rng(1)
    best, best_val = None, -9
    for i in range(n):
        theta = {"shrub": float(np.exp(rng.uniform(np.log(0.3), np.log(4)))), "grass": float(np.exp(rng.uniform(np.log(0.3), np.log(4)))), "crop": float(np.exp(rng.uniform(np.log(0.1), np.log(3)))),
                 "wind": float(rng.uniform(0, 1.2)), "beta": float(rng.uniform(0, 6)), "mx": float(rng.uniform(0.6, 1.5))}
        if i == 0:
            theta = {"shrub": 1.5, "grass": 2.5, "crop": 0.8, "wind": 1.0, "beta": 3.0, "mx": 1.0}           # the priors
        df = score(dev, theta).set_index("case")
        val = float((df.dir_auc - prox.dir_auc).groupby(df.block).mean().mean())                              # direction skill over proximity
        if val > best_val:
            best, best_val = theta, val
            print(f"trial {i:3d}  dev direction AUC over proximity {val:+.4f}  {json.dumps({k: round(v, 2) for k, v in theta.items()})}", flush=True)
    # overall speed from reach: scale so that the median predicted reach matches the median observed advance (dev)
    df = score(dev, best)
    obs = np.median([c.extra["adv_p95_km"] for c, *_ in dev])
    pred = float(np.median(df.reach[df.reach > 0]))
    print("median reach: predicted", round(pred, 2), "km, observed", round(float(obs), 2), "km")
    json.dump(best, open(config.DATA / "engine_calibrated.json", "w"))
    for name, th in (("priors", {"shrub": 1.5, "grass": 2.5, "crop": 0.8, "wind": 1.0, "beta": 3.0, "mx": 1.0}), ("calibrated", best)):
        for tag, group in (("dev", dev), ("TEST", test)):
            d = score(group, th).set_index("case")
            p = pd.DataFrame([{**burn_eval.evaluate_burn(c, cases.baseline_proximity(c), t, k), "case": c.id} for c, t, k in group]).set_index("case")
            m, lo, hi = scoring.bootstrap_mean((d.dir_auc - p.dir_auc.reindex(d.index)).groupby(d.block).mean())
            ma, loa, hia = scoring.bootstrap_mean((d.auc - p.auc.reindex(d.index)).groupby(d.block).mean())
            print(f"{name:10s} {tag:4s} direction vs proximity {m:+.3f} [{lo:+.3f}, {hi:+.3f}]   overall {ma:+.4f} [{loa:+.4f}, {hia:+.4f}]")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 120, tuple(sys.argv[2:]) or ("cases",))
