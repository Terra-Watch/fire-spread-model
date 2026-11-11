"""Greenness before the fire: MODIS NDVI (MOD13Q1 v061, 250 m, 16-day composites) from the Planetary Computer, the latest composite that ended before the start, and the same composite a year earlier.
Output data/<root>/<id>/ndvi.npz with ndvi (now) and ndvi_prev (a year earlier), float32, NaN where there is no value. Safe to re-run.
Run: WORKERS=3 .venv/bin/python scripts/22_ndvi.py"""
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

FILL = -3000


def composite(grid, client, end: pd.Timestamp):
    """NDVI on the grid from the latest composite of each tile that started 16 to 32 days before `end`."""
    w, s, e, n = grid.bounds_lonlat(pad_deg=0.01)
    start = (end - pd.Timedelta(days=32)).strftime("%Y-%m-%dT00:00:00Z")
    stop = (end - pd.Timedelta(days=16)).strftime("%Y-%m-%dT00:00:00Z")
    r = client.post(burnt.STAC, json={"collections": ["modis-13Q1-061"], "bbox": [w, s, e, n], "datetime": f"{start}/{stop}", "limit": 20})
    feats = sorted(r.json().get("features", []), key=lambda f: f["properties"].get("start_datetime") or f["id"], reverse=True)
    out, seen = np.full(grid.shape, np.nan, np.float32), set()
    for f in feats:
        tile = f["id"].split(".")[2]
        if tile in seen:
            continue
        seen.add(tile)
        url = burnt._signed(client, f["assets"]["250m_16_days_NDVI"]["href"])
        with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR", CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif", GDAL_HTTP_MAX_RETRY="3"), rasterio.open(url) as ds:
            tmp = np.full(grid.shape, FILL, np.int16)
            reproject(rasterio.band(ds, 1), tmp, dst_transform=grid.transform, dst_crs=f"EPSG:{grid.epsg}", resampling=Resampling.nearest, dst_nodata=FILL)
        out = np.where(np.isnan(out) & (tmp > FILL), tmp * 1e-4, out).astype(np.float32)
    return out


def one(d: Path):
    out = d / "ndvi.npz"
    if out.exists():
        return 1
    env = np.load(d / "env.npz")
    g = G.Grid(int(env["grid"][0]), *map(float, env["grid"][1:4]), int(env["grid"][4]), int(env["grid"][5]))
    t0 = pd.Timestamp(pd.read_json(d / "meta.json", typ="series")["t0"]).tz_localize(None)
    try:
        with httpx.Client(timeout=60) as c:
            now, prev = composite(g, c, t0), composite(g, c, t0 - pd.Timedelta(days=365))
    except Exception as e:
        print("fail", d.name, repr(e)[:120], flush=True)
        return 0
    np.savez_compressed(out, ndvi=now, ndvi_prev=prev)
    return 1


def main():
    dirs = [p.parent for root in ("cases", "bank") for p in sorted((config.DATA / root).glob("*/burn_long.npz"))]
    ok = 0
    with ThreadPoolExecutor(int(os.environ.get("WORKERS", 3))) as ex:
        for i, r in enumerate(ex.map(one, dirs), 1):
            ok += r
            if i % 50 == 0:
                print(i, "of", len(dirs), flush=True)
    print("done", ok, "of", len(dirs))


if __name__ == "__main__":
    main()
