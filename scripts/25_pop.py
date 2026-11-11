"""Population density around each fire (human ignition and suppression): WorldPop 1 km (2019 for 2019 cases, 2020 otherwise), local file, reprojected to each case grid. Output data/<root>/<id>/pop.npz with pop (people per km2). Safe to re-run.
Run: .venv/bin/python scripts/25_pop.py"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import reproject

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from firespread import config, grid as G  # noqa: E402

POP = config.RAW / "worldpop"


def main():
    dirs = [p.parent for root in ("cases", "bank") for p in sorted((config.DATA / root).glob("*/burn_long.npz"))]
    for d in dirs:
        out = d / "pop.npz"
        if out.exists():
            continue
        env = np.load(d / "env.npz")
        g = G.Grid(int(env["grid"][0]), *map(float, env["grid"][1:4]), int(env["grid"][4]), int(env["grid"][5]))
        year = 2019 if pd.Timestamp(pd.read_json(d / "meta.json", typ="series")["t0"]).year <= 2019 else 2020
        with rasterio.open(POP / f"ind_ppp_{year}_1km_Aggregated.tif") as ds:
            tmp = np.full(g.shape, np.nan, np.float32)
            reproject(rasterio.band(ds, 1), tmp, dst_transform=g.transform, dst_crs=f"EPSG:{g.epsg}", resampling=Resampling.bilinear, dst_nodata=np.nan)
        np.savez_compressed(out, pop=np.nan_to_num(tmp, nan=0.0).astype(np.float32))
    print("done", len(dirs))


if __name__ == "__main__":
    main()
