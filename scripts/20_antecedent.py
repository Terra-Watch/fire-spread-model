"""Dryness before the fire: daily ERA5 rain, evaporation and temperature for the 60 days before each case's start, from the Open-Meteo archive. Output data/antecedent.parquet (one row per case). Safe to re-run.
Run: .venv/bin/python scripts/20_antecedent.py"""
import sys
import time
from pathlib import Path

import httpx
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from firespread import config  # noqa: E402

URL = "https://archive-api.open-meteo.com/v1/archive"
OUT = config.DATA / "antecedent.parquet"
DAILY = "precipitation_sum,et0_fao_evapotranspiration,temperature_2m_max,relative_humidity_2m_mean"


def one(client, d):
    meta = pd.read_json(d / "meta.json", typ="series")
    t0 = pd.Timestamp(meta["t0"])
    p = {"latitude": round(float(meta["lat"]), 3), "longitude": round(float(meta["lon"]), 3), "start_date": str((t0 - pd.Timedelta(days=60)).date()), "end_date": str(t0.date()),
         "daily": DAILY, "timezone": "GMT"}
    for k in range(6):
        r = client.get(URL, params=p)
        if r.status_code == 200:
            break
        time.sleep(min(60, 5 * 2 ** k) if r.status_code == 429 else 2 ** k)
    else:
        raise RuntimeError(f"HTTP {r.status_code}")
    h = r.json()["daily"]
    rain = np.nan_to_num(np.array(h["precipitation_sum"], float))[:-1]            # the days before the start day
    et0 = np.nan_to_num(np.array(h["et0_fao_evapotranspiration"], float))[:-1]
    tmax = np.array(h["temperature_2m_max"], float)[:-1]
    rh = np.array(h["relative_humidity_2m_mean"], float)[:-1]
    wet = np.flatnonzero(rain[::-1] >= 2.0)                                       # days back to the last day with at least 2 mm
    return {"case": meta["case_id"], "rain7": rain[-7:].sum(), "rain30": rain[-30:].sum(), "rain60": rain.sum(), "days_dry": float(wet[0]) if len(wet) else 60.0,
            "balance30": rain[-30:].sum() - et0[-30:].sum(), "tmax14": np.nanmean(tmax[-14:]), "rh14": np.nanmean(rh[-14:])}


def main():
    dirs = [p.parent for root in ("cases", "bank") for p in sorted((config.DATA / root).glob("*/burn_long.npz"))]
    done = pd.read_parquet(OUT) if OUT.exists() else pd.DataFrame()
    have = set(done["case"]) if len(done) else set()
    rows = done.to_dict("records")
    fails = 0
    with httpx.Client(timeout=60) as client:
        for i, d in enumerate(dirs, 1):
            if d.name in have:
                continue
            try:
                rows.append(one(client, d))
            except Exception as e:
                fails += 1
                print("fail", d.name, repr(e)[:100], flush=True)
            time.sleep(0.4)
            if i % 50 == 0:
                print(i, "of", len(dirs), flush=True)
                pd.DataFrame(rows).to_parquet(OUT)
    pd.DataFrame(rows).to_parquet(OUT)
    print("done", len(rows), "of", len(dirs), "fails", fails)


if __name__ == "__main__":
    main()
