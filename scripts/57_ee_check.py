"""Cross-check of the refilled layers against Google Earth Engine, case by case: burn history (MCD64A1), NDVI (MOD13Q1) and Sentinel-2 (S2 harmonized), computed on the same 192 x 192 crop grid with the same rules as scripts 22, 52 and 53, then compared with what the Planetary Computer route gives.
Run: python scripts/57_ee_check.py N_CASES   (uses the earthengine scope of the application default login and project space-debris-507012)"""
import datetime as dt
import io
import sys
from pathlib import Path

import ee
import google.auth
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts"))
import importlib.util
from firespread import grid as G  # noqa: E402

spec = importlib.util.spec_from_file_location("fg", ROOT / "scripts" / "56_fill_gaps.py"); fg = importlib.util.module_from_spec(spec); spec.loader.exec_module(fg)
cred, _ = google.auth.default(scopes=["https://www.googleapis.com/auth/earthengine", "https://www.googleapis.com/auth/cloud-platform"])
ee.Initialize(credentials=cred, project="space-debris-507012")
DAY = 86400000.0


def pixels(img, g, bands):
    req = {"expression": img.select(bands), "fileFormat": "NUMPY_NDARRAY", "grid": {"dimensions": {"width": g.nx, "height": g.ny}, "affineTransform": {"scaleX": g.res, "shearX": 0, "translateX": g.x0, "shearY": 0, "scaleY": -g.res, "translateY": g.y0}, "crsCode": f"EPSG:{g.epsg}"}}
    a = ee.data.computePixels(req)
    return {b: np.asarray(a[b], np.float32) for b in bands}


def ee_ndvi(g, end):
    def comp(e):
        c = ee.ImageCollection("MODIS/061/MOD13Q1").merge(ee.ImageCollection("MODIS/061/MYD13Q1")).filterDate((e - pd.Timedelta(days=32)).strftime("%Y-%m-%d"), (e - pd.Timedelta(days=16)).strftime("%Y-%m-%d")).sort("system:time_start").select("NDVI")
        im = c.mosaic().updateMask(c.mosaic().gt(-3000)).multiply(1e-4)       # newest composite on top; fill value -3000 masked
        a = pixels(im.unmask(-9999), g, ["NDVI"])["NDVI"]; return np.where(a <= -9000, np.nan, a)
    return comp(end), comp(end - pd.Timedelta(days=365))


def ee_burn(g, t0):
    day0 = t0.normalize(); start = day0 - pd.Timedelta(days=365)
    d0, s0 = day0.timestamp() / 86400.0, start.timestamp() / 86400.0
    col = ee.ImageCollection("MODIS/061/MCD64A1").filterDate(start.replace(day=1).strftime("%Y-%m-%d"), (day0 + pd.Timedelta(days=31)).strftime("%Y-%m-%d"))

    def one(im):
        y = ee.Date(im.get("system:time_start")).get("year")
        jan1 = ee.Date.fromYMD(y, 1, 1).millis().divide(DAY)
        bd = im.select("BurnDate")
        ep = bd.toFloat().add(jan1).subtract(1)                                         # day number of the burn; compare with the start rule below
        ok = bd.gt(0).And(ep.lt(d0)).And(ep.gte(s0))
        return ee.Image.cat([ep.toFloat().updateMask(ok).rename("ep"), ok.toFloat().rename("ok")])
    m = col.map(one)
    last = m.select("ep").max().unmask(0)
    cnt = m.select("ok").sum().unmask(0)
    ds = ee.Image.constant(d0).subtract(last).updateMask(last.gt(0)).unmask(-9999)
    r = pixels(ee.Image.cat([ds.rename("ds"), cnt.rename("n")]), g, ["ds", "n"])
    return np.where(r["ds"] <= -9000, 366.0, r["ds"]), r["n"]


def ee_s2(g, t0):
    w, so, e, n = g.bounds_lonlat(pad_deg=0.0)
    col = ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED").filterBounds(ee.Geometry.Rectangle([w, so, e, n])).filterDate((t0 - pd.Timedelta(days=45)).strftime("%Y-%m-%d"), t0.strftime("%Y-%m-%d")).filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 60))

    def prep(im):
        scl = im.select("SCL"); clear = scl.eq(4).Or(scl.eq(5)).Or(scl.eq(6)).Or(scl.eq(7)).Or(scl.eq(11))
        age = ee.Image.constant(ee.Number(t0.timestamp() * 1000).subtract(im.get("system:time_start")).divide(DAY)).rename("age")
        return ee.Image.cat([im.select(["B8", "B11", "B12"]).toFloat().divide(10000), age.toFloat()]).updateMask(clear).updateMask(im.select("B8").gt(0))
    mos = col.sort("system:time_start").map(prep).mosaic().setDefaultProjection(col.first().select("B8").projection())   # mosaics carry no projection; 10 m is the native grid of B8
    mos = mos.reduceResolution(ee.Reducer.mean(), maxPixels=2048).reproject(crs=f"EPSG:{g.epsg}", crsTransform=[g.res, 0, g.x0, 0, -g.res, g.y0])
    r = pixels(mos.unmask(-9999), g, ["B8", "B11", "B12", "age"])
    return {k: np.where(v <= -9000, np.nan, v) for k, v in r.items()}


def stats(name, a, b):
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 20:
        return f"{name}: too few shared valid cells ({int(ok.sum())})  PC valid {np.isfinite(a).mean():.2f}, EE valid {np.isfinite(b).mean():.2f}"
    cc = np.corrcoef(a[ok], b[ok])[0, 1] if a[ok].std() > 0 and b[ok].std() > 0 else float("nan")
    return f"{name}: shared {ok.mean():.2f} | corr {cc:.3f} | mean PC {a[ok].mean():.3f} EE {b[ok].mean():.3f} | MAE {np.abs(a[ok]-b[ok]).mean():.4f}"


def main(n):
    meta = pd.read_parquet(ROOT / "data" / "dataset" / "meta_all.parquet")
    meta = meta[meta.year >= 2019].copy(); meta["yr"] = meta.year.astype(int)
    pick = meta.groupby("yr").head(max(1, n // 7)).head(n)
    for m in pick.itertuples():
        g = G.Grid(int(m.epsg), float(m.grid_x0) + int(m.crop_c0) * float(m.res), float(m.grid_y0) - int(m.crop_r0) * float(m.res), float(m.res), 192, 192)
        t0 = pd.Timestamp(m.t0).tz_localize(None) if pd.Timestamp(m.t0).tzinfo else pd.Timestamp(m.t0)
        pc, pe = fg.fill_case(m, ["ndvi", "s2", "burn"]); 
        try:
            nd_now, nd_prev = ee_ndvi(g, t0); bds, bn = ee_burn(g, t0); s2 = ee_s2(g, t0)
        except Exception as e:
            print(m.case, "EE FAILED", repr(e)[:160]); continue
        print(f"== {m.case} ({t0.date()}) PC errors: {pe}")
        if "ndvi" in pc: print("  " + stats("ndvi", pc["ndvi"], nd_now))
        if "burn_n" in pc: print("  " + stats("burn_n", pc["burn_n"], bn)); print("  " + stats("burn_days_since", pc["burn_days_since"], bds))
        if "s2_nir" in pc: print("  " + stats("s2_nir", pc["s2_nir"], s2["B8"])); print("  " + stats("s2_swir1", pc["s2_swir1"], s2["B11"])); print("  " + stats("s2_age", pc["s2_age"], s2["age"]))


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 7)
