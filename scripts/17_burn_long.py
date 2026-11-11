"""Burn dates over the 3 days after the start (for 24 h and 48 h horizons) for every case that is usable at its own horizon: data/<root>/<id>/burn_long.npz. Safe to re-run.
Run: WORKERS=5 .venv/bin/python scripts/17_burn_long.py"""
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from firespread import burn_eval, burnt, cases, config, grid as G  # noqa: E402


def usable(d: Path) -> bool:
    try:
        c = cases.load_case(d)
    except cases.NoWeather:
        return False
    b = np.load(d / "burn.npz")
    truth, cand, ok = burn_eval.burn_masks(c, b["doy"], int(b["year"]))
    return ok and burn_eval.connected_truth(c, truth, cand)[0].sum() >= 5


def one(d: Path):
    out = d / "burn_long.npz"
    if out.exists():
        return 1
    env = np.load(d / "env.npz")
    g = G.Grid(int(env["grid"][0]), *map(float, env["grid"][1:4]), int(env["grid"][4]), int(env["grid"][5]))
    t0 = pd.Timestamp(pd.read_json(d / "meta.json", typ="series")["t0"])
    try:
        with httpx.Client(timeout=60) as c:
            doy = burnt.burn_doy(g, t0.date(), (t0 + pd.Timedelta(hours=72)).date(), c)
    except Exception as e:
        print("fail", d.name, repr(e)[:120], flush=True)
        return 0
    np.savez_compressed(out, doy=doy, year=t0.year)
    return 1


def main():
    dirs = [p.parent for root in ("cases", "bank") for p in sorted((config.DATA / root).glob("*/burn.npz"))]
    dirs = [d for d in dirs if usable(d)]
    print("usable", len(dirs), flush=True)
    ok = 0
    with ThreadPoolExecutor(int(os.environ.get("WORKERS", 5))) as ex:
        for i, r in enumerate(ex.map(one, dirs), 1):
            ok += r
            if i % 50 == 0:
                print(i, "of", len(dirs), flush=True)
    print("done", ok, "of", len(dirs))


if __name__ == "__main__":
    main()
