"""Fills the layers that the first build left empty: burn history (53), NDVI (22) and Sentinel-2 (52), for the cases of one cleaned part (v2), and writes part_pN_v3.
Why they were empty: the per-file sign call of the Planetary Computer was refused when many workers ran (src/firespread/burnt.py now uses one token per container). The same functions as the original scripts do the reading; the grid is the case's 192 x 192 crop, rebuilt from meta.parquet.
Only cases with a layer fully empty are touched; a case already complete is left as it is. Resumable: progress is saved every 100 cases (gapfill_state.json next to the new part).
Run: WORKERS=6 python scripts/56_fill_gaps.py data/dataset/part_p0_v2 [limit]"""
import importlib.util
import json
import os
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx
import numpy as np
import pandas as pd
from rasterio.enums import Resampling

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "src"))
from firespread import burnt, grid as G  # noqa: E402


def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py"); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


NDVI, S2M, BH = load("22_ndvi"), load("52_sentinel2"), load("53_burn_history")
CROP = 192
GROUPS = {"ndvi": ["ndvi", "ndvi_prev", "ndvi_anom"], "s2": ["s2_nir", "s2_swir1", "s2_swir2", "s2_ndmi", "s2_nbr", "s2_age", "s2_valid"], "burn": ["burn_days_since", "burn_n"]}


def s2_fetch(c, g, t0):
    res = {k: np.full(g.shape, np.nan, np.float32) for k in (*S2M.BANDS, "age")}; n_read = 0
    for f in S2M.scenes(c, g, t0):
        if n_read >= S2M.MAX_SCENES or np.isfinite(res["nir"]).mean() >= 0.9:
            break
        scl = S2M._read(c, f["assets"]["SCL"]["href"], g, Resampling.nearest, np.uint8, "SCL")
        new = np.isin(scl, S2M.CLEAR) & np.isnan(res["nir"])
        if new.sum() < 0.02 * new.size:
            continue
        n_read += 1
        for k, b in S2M.BANDS.items():
            off = 1000.0 if float(f["properties"].get("s2:processing_baseline", "0") or 0) >= 4.0 else 0.0   # baseline 04.00 (scenes from 25 Jan 2022) stores reflectance + 1000
            v = (S2M._read(c, f["assets"][b]["href"], g, Resampling.average, np.float32, b) - off) / 10000.0
            res[k] = np.where(new & (v > 0), v, res[k])
        res["age"] = np.where(new & np.isfinite(res["nir"]), (t0 - pd.Timestamp(f["properties"]["datetime"]).tz_localize(None)).total_seconds() / 86400.0, res["age"])
    return res


def retry(fn, tries=4):
    import time
    for k in range(tries):
        try:
            return fn()
        except Exception as e:
            err = e
            time.sleep(3 * (k + 1))
    raise err


def fill_case(m, need):
    """Returns (layers, errors): each group is tried up to 4 times on its own; a group that still fails is reported and the others are kept."""
    g = G.Grid(int(m.epsg), float(m.grid_x0) + int(m.crop_c0) * float(m.res), float(m.grid_y0) - int(m.crop_r0) * float(m.res), float(m.res), CROP, CROP)
    t0 = pd.Timestamp(m.t0).tz_localize(None) if pd.Timestamp(m.t0).tzinfo else pd.Timestamp(m.t0)
    out, errs = {}, {}
    with httpx.Client(timeout=90) as c:
        if "ndvi" in need:
            try:
                now = retry(lambda: NDVI.composite(g, c, t0)); prev = retry(lambda: NDVI.composite(g, c, t0 - pd.Timedelta(days=365)))
                out.update(ndvi=now, ndvi_prev=prev, ndvi_anom=now - prev)
            except Exception as e:
                errs["ndvi"] = repr(e)[:120]
        if "s2" in need:
            try:
                r = retry(lambda: s2_fetch(c, g, t0)); nir, sw1, sw2 = r["nir"], r["swir1"], r["swir2"]
                with np.errstate(all="ignore"):
                    out.update(s2_nir=nir, s2_swir1=sw1, s2_swir2=sw2, s2_ndmi=(nir - sw1) / (nir + sw1), s2_nbr=(nir - sw2) / (nir + sw2), s2_age=r["age"], s2_valid=np.isfinite(nir).astype(np.float32))
            except Exception as e:
                errs["s2"] = repr(e)[:120]
        if "burn" in need:
            try:
                h = retry(lambda: BH.history(g, pd.Timestamp(m.t0), c))
                out.update(burn_days_since=np.where(np.isnan(h["days_since"]), 366.0, h["days_since"]), burn_n=h["n_burns"].astype(np.float32))
            except Exception as e:
                errs["burn"] = repr(e)[:120]
    return out, errs


def main(part, limit=None):
    part = Path(part); second = part.name.endswith("_v3")      # second pass (v3 -> v4): retries what is still empty and redoes Sentinel-2 for 2022 on with the offset removed
    out = part.parent / (part.name.replace("_v3", "_v4") if second else part.name.replace("_v2", "_v3")); out.mkdir(exist_ok=True)
    names = json.load(open(part / "names.json")); idx = {k: names["spatial"].index(k) for ks in GROUPS.values() for k in ks}
    meta = pd.read_parquet(part / "meta.parquet"); X = np.load(part / "spatial.npy", mmap_mode="r")
    for f in part.iterdir():
        if f.name not in ("spatial.npy", "gapfill_state.json") and not (out / f.name).exists():
            shutil.copy(f, out / f.name)
    Xn = np.load(out / "spatial.npy") if (out / "spatial.npy").exists() and (out / "gapfill_state.json").exists() else np.array(X)
    state = json.load(open(out / "gapfill_state.json")) if (out / "gapfill_state.json").exists() else {"done": [], "failed": {}}
    done = set(state["done"]); todo = []
    for i, m in enumerate(meta.itertuples()):
        need = [g for g, ks in GROUPS.items() if not np.isfinite(np.asarray(Xn[i, idx[ks[0]]], np.float32)).any() or (g == "s2" and float(np.asarray(Xn[i, idx["s2_valid"]], np.float32).sum()) == 0) or (g == "burn" and not np.isfinite(np.asarray(Xn[i, idx["burn_n"]], np.float32)).any())]
        if second and int(m.year) >= 2022 and "s2" not in need:
            need.append("s2")
        if need and m.case not in done:
            todo.append((i, m, need))
    if limit:
        todo = todo[:limit]
    print(f"{part.name}: {len(meta)} cases, {len(todo)} to fill", flush=True)

    def work(t):
        i, m, need = t
        try:
            res, errs = fill_case(m, need)
            return i, m.case, need, res, errs
        except Exception as e:
            return i, m.case, need, {}, {"all": repr(e)[:160]}
    n = 0
    with ThreadPoolExecutor(int(os.environ.get("WORKERS", 6))) as ex:
        for i, case, need, res, err in ex.map(work, todo):
            n += 1
            for k, v in res.items():
                Xn[i, idx[k]] = np.asarray(v).astype(Xn.dtype)
            if err:
                state["failed"][case] = err
            else:
                state["done"].append(case)
            if n % 100 == 0 or n == len(todo):
                np.save(out / "spatial.npy", Xn); json.dump(state, open(out / "gapfill_state.json", "w")); print(n, "of", len(todo), "| failed", len(state["failed"]), flush=True)
    np.save(out / "spatial.npy", Xn); json.dump(state, open(out / "gapfill_state.json", "w"))
    cov = {g: float(np.mean([np.isfinite(np.asarray(Xn[i, idx[ks[-1] if g != 's2' else 's2_nir']], np.float32)).any() for i in range(len(meta))])) for g, ks in GROUPS.items()}
    print("wrote", out, "| share of cases with each source now:", {k: round(v, 3) for k, v in cov.items()}, "| failed", len(state["failed"]))


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else None)
