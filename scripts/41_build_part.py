"""Build the case bundles (terrain included) and the MODIS burn dates for one share of data/big_cases.parquet; the share is chosen by a hash of the 1 x 1 degree block, so neighbouring fires stay together.
Safe to re-run. Output data/bank/<case_id>/{env.npz, weather.csv, meta.json, burn.npz}.   Run: WORKERS=6 .venv/bin/python scripts/41_build_part.py PART N_PARTS [limit]"""
import hashlib
import importlib
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from firespread import burnt, config, grid as G  # noqa: E402

env_script = importlib.import_module("04_environment")


def one(row):
    out = config.DATA / "bank" / row.case_id
    if (out / "burn.npz").exists():
        return 1
    try:
        env_script.build(row, dem_on=True, root="bank")
        env = np.load(out / "env.npz")
        g = G.Grid(int(env["grid"][0]), *map(float, env["grid"][1:4]), int(env["grid"][4]), int(env["grid"][5]))
        t0, t1 = pd.Timestamp(row.t0), pd.Timestamp(row.t1)
        with httpx.Client(timeout=60) as c:
            doy = burnt.burn_doy(g, t0.date(), (t1 + pd.Timedelta(days=1)).date(), c)
        np.savez_compressed(out / "burn.npz", doy=doy, year=t0.year)
        return 1
    except Exception as e:
        print("fail", row.case_id, repr(e)[:100], flush=True)
        return 0


def main(part, n_parts, limit=None):
    p = pd.read_parquet(config.DATA / "big_cases.parquet")
    p = p[[int(hashlib.sha1(b.encode()).hexdigest(), 16) % n_parts == part for b in p.block]]
    if limit:
        p = p.head(limit)
    print(f"part {part} of {n_parts}: {len(p)} cases", flush=True)
    t = time.time(); ok = 0
    with ThreadPoolExecutor(int(os.environ.get("WORKERS", "6"))) as ex:
        for i, r in enumerate(ex.map(one, list(p.itertuples())), 1):
            ok += r
            if i % 50 == 0:
                print(i, "of", len(p), round(time.time() - t), "s", flush=True)
    print("done", ok, "of", len(p), round(time.time() - t), "s")


if __name__ == "__main__":
    main(int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]) if len(sys.argv) > 3 else None)
