"""Add MODIS burn dates to every case bundle: data/cases/<id>/burn.npz (burn day of year on the case grid, plus the year). Safe to re-run.   Run: .venv/bin/python scripts/08_burn_truth.py"""
import datetime as dt
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from firespread import burnt, config, grid as G  # noqa: E402


def one(d: Path):
    out = d / "burn.npz"
    if out.exists():
        return 1
    meta = pd.read_json(d / "meta.json", typ="series")
    env = np.load(d / "env.npz")
    g = G.Grid(int(env["grid"][0]), *map(float, env["grid"][1:4]), int(env["grid"][4]), int(env["grid"][5]))
    t0, t1 = pd.Timestamp(meta["t0"]), pd.Timestamp(meta["t1"])
    try:
        with httpx.Client(timeout=60) as c:
            doy = burnt.burn_doy(g, t0.date(), (t1 + pd.Timedelta(days=1)).date(), c)
    except Exception as e:
        print("fail", d.name, repr(e)[:120], flush=True)
        return 0
    np.savez_compressed(out, doy=doy, year=t0.year)
    return 1


def main():
    dirs = sorted(p.parent for p in (config.DATA / "cases").glob("*/meta.json"))
    ok = 0
    with ThreadPoolExecutor(3) as ex:
        for i, r in enumerate(ex.map(one, dirs), 1):
            ok += r
            if i % 50 == 0:
                print(i, "of", len(dirs), flush=True)
    print("done", ok, "of", len(dirs))


if __name__ == "__main__":
    main()
