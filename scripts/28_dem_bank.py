"""The bank cases were built without terrain (has_dem False). Read the Copernicus DEM for the ones used in the ranking study and write dem, slope and upslope into their env.npz. Safe to re-run.
Run: WORKERS=6 .venv/bin/python scripts/28_dem_bank.py"""
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from firespread import config, grid as G, rasters  # noqa: E402


def one(d: Path):
    env = dict(np.load(d / "env.npz"))
    if bool(env["has_dem"]):
        return 1
    g = G.Grid(int(env["grid"][0]), *map(float, env["grid"][1:4]), int(env["grid"][4]), int(env["grid"][5]))
    try:
        dem = rasters.read_dem(g)
    except Exception as e:
        print("fail", d.name, repr(e)[:100], flush=True)
        return 0
    slope, up = rasters.slope_upslope(np.where(np.isnan(dem), np.nanmean(dem) if np.isfinite(dem).any() else 0, dem), g.res)
    env.update(dem=dem, slope=slope, upslope=up, has_dem=np.array(True))
    np.savez_compressed(d / "env.tmp.npz", **env)
    (d / "env.tmp.npz").replace(d / "env.npz")
    return 1


def main():
    dirs = [p.parent for p in sorted((config.DATA / "bank").glob("*/burn_long.npz"))]
    ok = 0
    with ThreadPoolExecutor(int(os.environ.get("WORKERS", 6))) as ex:
        for i, r in enumerate(ex.map(one, dirs), 1):
            ok += r
            if i % 50 == 0:
                print(i, "of", len(dirs), flush=True)
    print("done", ok, "of", len(dirs))


if __name__ == "__main__":
    main()
