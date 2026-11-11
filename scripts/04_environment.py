"""For each pilot case, write one bundle: terrain, land cover, forest share, the start footprint and the truth in grid cells, and hourly weather from three sources.
data/cases/<case_id>/{env.npz, weather.csv, meta.json}. Safe to re-run: finished cases are skipped.   Run: .venv/bin/python scripts/04_environment.py [limit]"""
import json
import os
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from firespread import config, events, grid as G, rasters, weather  # noqa: E402

DET = {}


def detections(year):
    if year not in DET:
        DET[year] = pd.read_parquet(config.DATA / "detections" / f"{year}.parquet")
    return DET[year]


def build(row, dem_on: bool = True, root: str = "cases") -> dict:
    cid = row.case_id
    out = config.DATA / root / cid
    if (out / "meta.json").exists():
        return json.load(open(out / "meta.json"))
    out.mkdir(parents=True, exist_ok=True)
    ev = detections(int(row.event[:4]))
    ev = ev[ev["event"] == row.event]
    start = ev[ev["epoch"] <= row.epoch_from]
    truth = ev[ev["epoch"] == row.epoch_to]
    t0, t1 = pd.Timestamp(row.t0), pd.Timestamp(row.t1)
    g = G.make_grid(float(start.latitude.mean()), float(start.longitude.mean()))
    sr, sc = g.to_rc(start.latitude.values, start.longitude.values)
    tr, tc = g.to_rc(truth.latitude.values, truth.longitude.values)
    sx, sy = events.sinusoidal_km(start.latitude.values, start.longitude.values)
    tx, ty = events.sinusoidal_km(truth.latitude.values, truth.longitude.values)
    new = cKDTree(np.column_stack([sx, sy])).query(np.column_stack([tx, ty]))[0] > config.GROWTH_TOL_KM
    dem = rasters.read_dem(g) if dem_on else np.full(g.shape, np.nan, np.float32)
    slope, up = rasters.slope_upslope(np.where(np.isnan(dem), np.nanmean(dem) if np.isfinite(dem).any() else 0, dem), g.res)
    lc = rasters.read_landcover(g)
    fs = rasters.forest_share(g)
    arrays = dict(dem=dem, slope=slope, upslope=up, forest_share=fs, **{f"lc_{k}": v for k, v in lc.items()},
                  start_rc=np.column_stack([sr, sc]).astype(np.float32), start_age_h=((t0 - start["t"]).dt.total_seconds() / 3600).values.astype(np.float32),
                  start_frp=start["frp"].values.astype(np.float32),
                  truth_rc=np.column_stack([tr, tc]).astype(np.float32), truth_t_h=((truth["t"] - t0).dt.total_seconds() / 3600).values.astype(np.float32), truth_new=new)
    np.savez_compressed(out / "env.npz", **arrays, grid=np.array([g.epsg, g.x0, g.y0, g.res, g.nx, g.ny], float), has_dem=np.array(dem_on))
    a, b = (t0 - pd.Timedelta(days=2)).strftime("%Y-%m-%d"), (t1 + pd.Timedelta(days=2)).strftime("%Y-%m-%d")
    wx = []
    for kind in (("hist", "prev1", "era5") if dem_on else ("hist", "era5")):
        try:
            w = weather.fetch(kind, float(row.lat), float(row.lon), a, b)
        except Exception as e:   # one source failing must not lose the case
            w = pd.DataFrame(columns=weather.VARS, index=pd.DatetimeIndex([], tz="UTC", name="time"))
            print("weather", kind, cid, e, flush=True)
        w = w.reset_index()
        w.insert(0, "source", kind)
        w.insert(2, "hours_from_t0", (w["time"] - t0).dt.total_seconds() / 3600.0)
        wx.append(w)
    wx = pd.concat(wx, ignore_index=True)
    wx.to_csv(out / "weather.csv", index=False)
    inside = lambda r, c: ((r >= 0) & (r < g.ny) & (c >= 0) & (c < g.nx))
    meta = {"case_id": cid, "event": row.event, "state": row.state, "split": row.split, "year": int(row.year), "t0": str(t0), "t1": str(t1), "gap_h": float(row.gap_h),
            "lat": float(row.lat), "lon": float(row.lon), "n_start": int(len(start)), "n_truth": int(len(truth)), "n_new": int(new.sum()),
            "start_inside_grid": float(inside(sr, sc).mean()), "truth_inside_grid": float(inside(tr, tc).mean()),
            "dem_valid": float(np.isfinite(dem).mean()), "slope_p95_deg": float(np.nanpercentile(slope, 95)),
            "forest_share_mean": float(fs.mean()), "tree_mean": float(lc["tree"].mean()), "crop_mean": float(lc["crop"].mean()),
            "weather_rows": {k: int(((wx.source == k) & wx.wind_speed_10m.notna()).sum()) for k in ("hist", "prev1", "era5")}}
    json.dump(meta, open(out / "meta.json", "w"))
    return meta


def main(limit=None):
    p = pd.read_parquet(config.DATA / "pilot_cases.parquet")
    if limit:
        p = p.iloc[::max(1, len(p) // limit)].head(limit)
    t = time.time()
    done, failed = 0, []

    def work(row):
        try:
            return build(row)
        except Exception:
            traceback.print_exc()
            failed.append(row.case_id)

    with ThreadPoolExecutor(int(os.environ.get("WORKERS", "8"))) as ex:
        for m in ex.map(work, [r for r in p.itertuples()]):
            done += m is not None
            if done % 20 == 0:
                print(done, "cases", round(time.time() - t), "s", flush=True)
    (config.DATA / "cases").mkdir(exist_ok=True)
    open(config.DATA / "cases" / "_failed.txt", "w").write("\n".join(failed))
    print("done", done, "failed", len(failed), round(time.time() - t), "s")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else None)
