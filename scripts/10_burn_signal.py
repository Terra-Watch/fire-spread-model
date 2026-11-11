"""Does the burned patch of a fire line up with the wind or the slope? From the MODIS burn date: the patch touching the start of each fire, its principal axis against the mean wind line,
and the direction from the start to the patch centre against the wind (and uphill). Random shapes would give |cos| of axis alignment near 0.64 and a mean directional cosine near 0.
Run: .venv/bin/python scripts/10_burn_signal.py"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from firespread import burn_eval, cases, config, scoring  # noqa: E402


def main():
    rows = []
    for d in sorted((config.DATA / "cases").glob("*/burn.npz")):
        try:
            c = cases.load_case(d.parent)
        except cases.NoWeather:
            continue
        b = np.load(d)
        truth, cand, ok = burn_eval.burn_masks(c, b["doy"], int(b["year"]))
        if not ok:
            continue
        truth, _ = burn_eval.connected_truth(c, truth, cand)
        if truth.sum() < 30:
            continue
        hours = max(1, int(round(c.gap_h)))
        u = np.mean(-c.wind_ms[:hours] * np.sin(np.radians(c.wind_from[:hours])))
        v = np.mean(-c.wind_ms[:hours] * np.cos(np.radians(c.wind_from[:hours])))
        w_to = (np.degrees(np.arctan2(u, v)) + 360) % 360
        rr, cc = np.nonzero(truth)
        sc = c.seeds.mean(axis=0)
        east, north = cc + 0.5 - (sc[1] + 0.5), -(rr + 0.5 - (sc[0] + 0.5))
        bearing = (np.degrees(np.arctan2(east.mean(), north.mean())) + 360) % 360
        cov = np.cov(np.vstack([east, north]))
        ev, evec = np.linalg.eigh(cov)
        axis = (np.degrees(np.arctan2(evec[0, 1], evec[1, 1])) + 360) % 360      # principal axis, compass degrees (mod 180)
        r0, c0 = np.clip(np.floor(sc).astype(int), 0, np.array(c.fuel.shape) - 1)
        rows.append({"case": c.id, "n": int(truth.sum()), "wind": float(np.hypot(u, v)), "elong": float(np.sqrt(ev[1] / max(ev[0], 1e-9))),
                     "cos_wind": float(np.cos(np.radians(bearing - w_to))), "cos_up": float(np.cos(np.radians(bearing - c.upslope[r0, c0]))),
                     "axis_wind": float(abs(np.cos(np.radians(axis - w_to)))), "slope": float(np.degrees(np.arctan(c.tan_slope[r0, c0])))})
    df = pd.DataFrame(rows)
    print(len(df), "fires with a burned patch of at least 30 cells; random orientation gives axis alignment 0.64 and direction cosine 0")
    for name, sub in (("all", df), ("wind >= 2 m/s", df[df.wind >= 2]), ("wind >= 3 m/s", df[df.wind >= 3]), ("elongated (>1.5)", df[df.elong > 1.5]), ("slope >= 10 deg", df[df.slope >= 10])):
        if len(sub) < 5:
            continue
        a, b, c = (scoring.bootstrap_mean(sub[k]) for k in ("cos_wind", "cos_up", "axis_wind"))
        print(f"{name:17s} n={len(sub):3d}  cos(wind) {a[0]:+.2f} [{a[1]:+.2f}, {a[2]:+.2f}]   cos(uphill) {b[0]:+.2f} [{b[1]:+.2f}, {b[2]:+.2f}]   axis-wind alignment {c[0]:.2f} [{c[1]:.2f}, {c[2]:.2f}]")


if __name__ == "__main__":
    main()
