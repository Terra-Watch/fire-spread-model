"""Is there any wind or slope signal in where fires grew? For every new-growth detection: its bearing from the start footprint's nearest detection, compared with the
wind direction (where it blows to) and with uphill. If growth ignored both, the mean cosine would sit at zero.   Run: .venv/bin/python scripts/07_signal.py"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from firespread import cases, config, scoring  # noqa: E402


def main():
    rows = []
    for d in sorted((config.DATA / "cases").glob("*/meta.json")):
        try:
            c = cases.load_case(d.parent)
        except cases.NoWeather:
            continue
        env = np.load(d.parent / "env.npz")
        sr, tr = env["start_rc"], env["truth_rc"][env["truth_new"]]
        if len(tr) == 0:
            continue
        tree = cKDTree(sr)
        dist, idx = tree.query(tr)
        dr, dc = tr[:, 0] - sr[idx, 0], tr[:, 1] - sr[idx, 1]
        bearing = np.degrees(np.arctan2(dc, -dr)) % 360.0                                        # compass bearing of the step from the nearest start detection
        hours = int(round(c.gap_h))
        w_to = (np.degrees(np.arctan2(np.mean(-c.wind_ms[:hours] * np.sin(np.radians(c.wind_from[:hours]))), np.mean(-c.wind_ms[:hours] * np.cos(np.radians(c.wind_from[:hours]))))) + 360) % 360
        r0, c0 = np.clip(np.floor(sr.mean(axis=0)).astype(int), 0, np.array(c.fuel.shape) - 1)
        up = c.upslope[r0, c0]
        slope = np.degrees(np.arctan(c.tan_slope[r0, c0]))
        rows.append({"case": c.id, "split": c.meta["split"], "n": len(tr), "wind_ms": float(np.mean(c.wind_ms[:hours])), "slope": slope,
                     "cos_wind": float(np.mean(np.cos(np.radians(bearing - w_to)))), "cos_up": float(np.mean(np.cos(np.radians(bearing - up)))),
                     "dist_km": float(np.mean(dist) * c.res / 1000)})
    df = pd.DataFrame(rows)
    print(len(df), "cases; mean advance step", round(df.dist_km.mean(), 2), "km")
    for name, sub in (("all", df), ("wind >= 3 m/s", df[df.wind_ms >= 3]), ("wind < 3 m/s", df[df.wind_ms < 3]), ("slope >= 8 deg", df[df.slope >= 8])):
        m1 = scoring.bootstrap_mean(sub.cos_wind); m2 = scoring.bootstrap_mean(sub.cos_up)
        print(f"{name:14s} n={len(sub):3d}  mean cos(wind) {m1[0]:+.2f} [{m1[1]:+.2f}, {m1[2]:+.2f}]   mean cos(uphill) {m2[0]:+.2f} [{m2[1]:+.2f}, {m2[2]:+.2f}]")


if __name__ == "__main__":
    main()
