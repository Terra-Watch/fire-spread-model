"""Sentinel-2 pictures from just before the fire. Scenes from the 45 days before the start (at least one day before, scene cloud cover under 60 %), newest first; each pixel takes its value from the newest scene where it is clear
(scene classification), until 90 % of the pixels are filled or 8 scenes were read. Bands B08 (near infrared), B11 and B12 (shortwave infrared), reflectance on the 100 m grid, read from the file's overview level nearest 80 m (fast).
Output data/<root>/<id>/s2.npz with nir, swir1, swir2, age (days since the scene, per pixel) as float32 with NaN where no clear pixel was found, and n_scenes. Safe to re-run.
Run: WORKERS=6 .venv/bin/python scripts/52_sentinel2.py [limit]"""
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

BANDS = {"nir": "B08", "swir1": "B11", "swir2": "B12"}
CLEAR = (4, 5, 6, 7, 11)             # scene classification: vegetation, bare, water, unclassified, snow; everything else (no data, shadow, cloud, cirrus) is dropped
MAX_SCENES = 8
OVERVIEW = {"B08": 2, "B11": 1, "B12": 1, "SCL": 1}      # 10 m bands: 2 = 80 m; 20 m bands: 1 = 80 m


def _read(client, href, grid, resampling, dtype, band):
    url = burnt._signed(client, href)
    with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR", CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif", GDAL_HTTP_MAX_RETRY="3"), rasterio.open(url, overview_level=OVERVIEW[band]) as ds:
        out = np.zeros(grid.shape, dtype)
        reproject(rasterio.band(ds, 1), out, dst_transform=grid.transform, dst_crs=f"EPSG:{grid.epsg}", resampling=resampling)
    return out


def scenes(client, grid, t0):
    w, s, e, n = grid.bounds_lonlat(pad_deg=0.005)
    start, stop = (t0 - pd.Timedelta(days=45)).strftime("%Y-%m-%dT00:00:00Z"), (t0 - pd.Timedelta(days=1)).strftime("%Y-%m-%dT00:00:00Z")
    r = client.post(burnt.STAC, json={"collections": ["sentinel-2-l2a"], "bbox": [w, s, e, n], "datetime": f"{start}/{stop}", "query": {"eo:cloud_cover": {"lt": 60}}, "limit": 60})
    return sorted(r.json().get("features", []), key=lambda f: f["properties"]["datetime"], reverse=True)


def one(d: Path):
    out = d / "s2.npz"
    if out.exists():
        return 1
    env = np.load(d / "env.npz")
    g = G.Grid(int(env["grid"][0]), *map(float, env["grid"][1:4]), int(env["grid"][4]), int(env["grid"][5]))
    t0 = pd.Timestamp(pd.read_json(d / "meta.json", typ="series")["t0"]).tz_localize(None)
    try:
        with httpx.Client(timeout=90) as c:
            res = {k: np.full(g.shape, np.nan, np.float32) for k in (*BANDS, "age")}; n_read = 0
            for f in scenes(c, g, t0):
                if n_read >= MAX_SCENES or np.isfinite(res["nir"]).mean() >= 0.9:
                    break
                scl = _read(c, f["assets"]["SCL"]["href"], g, Resampling.nearest, np.uint8, "SCL")
                new = np.isin(scl, CLEAR) & np.isnan(res["nir"])
                if new.sum() < 0.02 * new.size:
                    continue
                n_read += 1
                for k, b in BANDS.items():
                    v = _read(c, f["assets"][b]["href"], g, Resampling.average, np.float32, b) / 10000.0
                    res[k] = np.where(new & (v > 0), v, res[k])
                res["age"] = np.where(new & np.isfinite(res["nir"]), (t0 - pd.Timestamp(f["properties"]["datetime"]).tz_localize(None)).total_seconds() / 86400.0, res["age"])
    except Exception as e:
        print("fail", d.name, repr(e)[:120], flush=True)
        return 0
    np.savez_compressed(out, **{k: v.astype(np.float32) for k, v in res.items()}, n_scenes=np.int16(n_read))
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
