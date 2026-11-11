"""A larger bank of cases for training and scoring a learned model: no terrain (the slowest download), land cover, forest share, start and truth cells, hourly weather, MODIS burn date.
Chosen like the pilot (never the same fires), split by the same block hash. data/bank/<case_id>/.   Run: .venv/bin/python scripts/13_bank.py [n_cases]"""
import hashlib
import importlib
import json
import os
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from firespread import burnt, config, grid as G  # noqa: E402

env_script = importlib.import_module("04_environment")
LAST_BURN_DATE = pd.Timestamp("2025-08-31", tz="UTC")       # the burned-area product lags by about two months


def choose(n):
    c = pd.read_parquet(config.DATA / "cases.parquet")
    e = pd.read_parquet(config.DATA / "events.parquet")[["event", "extent_km"]]
    pilot = set(pd.read_parquet(config.DATA / "pilot_cases.parquet").event)
    c = c.merge(e, on="event")
    c = c[(c.n_start >= 3) & (c.n_new >= 1) & (c.gap_h >= 3) & (c.gap_h <= 30) & (c.extent_km <= 40) & (c.state != "") & ~c.event.isin(pilot) & (c.t1 <= LAST_BURN_DATE)].copy()
    c["block"] = [f"{int(np.floor(a))}_{int(np.floor(b))}" for a, b in zip(c.lat, c.lon)]
    c["split"] = ["test" if int(hashlib.sha1(b.encode()).hexdigest(), 16) % 100 < 30 else "dev" for b in c.block]
    c["year"] = c.t0.dt.year
    w = np.sqrt(c.state.value_counts())
    quota = (w / w.sum() * n).round().astype(int).clip(lower=2)
    rng = np.random.default_rng(7)
    picks = []
    for st, k in quota.items():
        g = c[c.state == st].sample(frac=1.0, random_state=int(rng.integers(1 << 31)))
        seen, taken = {}, []
        for i, r in g.iterrows():
            if seen.get(r.event, 0) < 2:
                taken.append(i)
                seen[r.event] = seen.get(r.event, 0) + 1
            if len(taken) >= k:
                break
        picks += taken
    p = c.loc[picks].copy()
    p["case_id"] = p.event + "_" + p.epoch_from.astype(str)
    return p.reset_index(drop=True)


def one(row):
    out = config.DATA / "bank" / row.case_id
    if (out / "burn.npz").exists():
        return 1
    try:
        meta = env_script.build(row, dem_on=False, root="bank")
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


def main(n):
    p = choose(n)
    p.to_parquet(config.DATA / "bank_cases.parquet", index=False)
    print(len(p), "cases;", p.split.value_counts().to_dict(), p.year.value_counts().sort_index().to_dict(), flush=True)
    t = time.time(); ok = 0
    with ThreadPoolExecutor(int(os.environ.get("WORKERS", "6"))) as ex:
        for i, r in enumerate(ex.map(one, list(p.itertuples())), 1):
            ok += r
            if i % 50 == 0:
                print(i, "of", len(p), round(time.time() - t), "s", flush=True)
    print("done", ok, "of", len(p), round(time.time() - t), "s")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 2000)
