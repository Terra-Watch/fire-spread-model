"""Where did it burn before? MODIS burned-area dates (MCD64A1, 500 m) for the 12 months before the fire's start, on the case grid: days since the most recent burn, and the number of months with a burn.
Burned ground has little fuel left, so this is a strong input. Output data/<root>/<id>/burnhist.npz (days_since float32 with NaN = no burn in the year, n_burns int16, mapped uint8). Safe to re-run.
Run: WORKERS=6 .venv/bin/python scripts/53_burn_history.py [limit]"""
import datetime as dt
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx
import numpy as np
import pandas as pd
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import reproject

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from firespread import burnt, config, grid as G  # noqa: E402


def history(grid, t0: pd.Timestamp, client) -> dict:
    day0 = t0.date()
    start = day0 - dt.timedelta(days=365)
    last = np.zeros(grid.shape, np.int64)                 # ordinal date of the latest burn before the start, 0 = none
    count = np.zeros(grid.shape, np.int16)
    mapped = np.zeros(grid.shape, bool)
    w, s, e, n = grid.bounds_lonlat(pad_deg=0.01)
    for y, m in burnt._months(start, day0):
        r = client.post(burnt.STAC, json={"collections": ["modis-64A1-061"], "bbox": [w, s, e, n], "datetime": f"{y}-{m:02d}-01T00:00:00Z/{y}-{m:02d}-28T00:00:00Z", "limit": 8})
        for f in r.json().get("features", []):
            url = burnt._signed(client, f["assets"]["Burn_Date"]["href"])
            with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR", CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif", GDAL_HTTP_MAX_RETRY="3"), rasterio.open(url) as ds:
                tmp = np.full(grid.shape, burnt.UNMAPPED, np.int16)
                reproject(rasterio.band(ds, 1), tmp, dst_transform=grid.transform, dst_crs=f"EPSG:{grid.epsg}", resampling=Resampling.nearest, dst_nodata=burnt.UNMAPPED)
            mapped |= tmp >= 0
            ordinal = dt.date(y, 1, 1).toordinal() + tmp.astype(np.int64) - 1
            ok = (tmp > 0) & (ordinal < day0.toordinal()) & (ordinal >= start.toordinal())
            last = np.where(ok, np.maximum(last, ordinal), last)
            count += ok.astype(np.int16)
    return {"days_since": np.where(last > 0, day0.toordinal() - last, np.nan).astype(np.float32), "n_burns": count, "mapped": mapped.astype(np.uint8)}


def one(d: Path):
    out = d / "burnhist.npz"
    if out.exists():
        return 1
    env = np.load(d / "env.npz")
    g = G.Grid(int(env["grid"][0]), *map(float, env["grid"][1:4]), int(env["grid"][4]), int(env["grid"][5]))
    t0 = pd.Timestamp(pd.read_json(d / "meta.json", typ="series")["t0"])
    try:
        with httpx.Client(timeout=90) as c:
            res = history(g, t0, c)
    except Exception as e:
        print("fail", d.name, repr(e)[:120], flush=True)
        return 0
    np.savez_compressed(out, **res)
    return 1


def main(limit=None):
    dirs = [p.parent for root in ("cases", "bank") for p in sorted((config.DATA / root).glob("*/burn_long.npz"))]
    if limit:
        dirs = dirs[:limit]
    ok = 0
    with ThreadPoolExecutor(int(os.environ.get("WORKERS", 4))) as ex:
        for i, r in enumerate(ex.map(one, dirs), 1):
            ok += r
            if i % 50 == 0:
                print(i, "of", len(dirs), flush=True)
    print("done", ok, "of", len(dirs))


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else None)
