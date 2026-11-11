"""Soil moisture and vapour pressure deficit just before the start: ERA5 hourly from the Open-Meteo archive. Output data/soil.parquet (one row per case). Safe to re-run.
Run: .venv/bin/python scripts/23_soil.py"""
import sys
import time
from pathlib import Path

import httpx
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from firespread import config  # noqa: E402

URL = "https://archive-api.open-meteo.com/v1/archive"
OUT = config.DATA / "soil.parquet"


def one(client, d):
    meta = pd.read_json(d / "meta.json", typ="series")
    t0 = pd.Timestamp(meta["t0"]).tz_localize(None)
    p = {"latitude": round(float(meta["lat"]), 3), "longitude": round(float(meta["lon"]), 3), "start_date": str((t0 - pd.Timedelta(days=7)).date()), "end_date": str(t0.date()),
         "hourly": "soil_moisture_0_to_7cm,soil_moisture_7_to_28cm,vapour_pressure_deficit", "timezone": "GMT"}
    for k in range(6):
        r = client.get(URL, params=p)
        if r.status_code == 200:
            break
        time.sleep(min(60, 5 * 2 ** k) if r.status_code == 429 else 2 ** k)
    else:
        raise RuntimeError(f"HTTP {r.status_code}")
    h = pd.DataFrame(r.json()["hourly"]).assign(time=lambda x: pd.to_datetime(x["time"]))
    h = h[h.time < t0].set_index("time")                                           # only what was known before the start
    last24, first = h.iloc[-24:], h.iloc[:24]
    return {"case": meta["case_id"], "sm7": last24.soil_moisture_0_to_7cm.mean(), "sm28": last24.soil_moisture_7_to_28cm.mean(),
            "sm7_change": last24.soil_moisture_0_to_7cm.mean() - first.soil_moisture_0_to_7cm.mean(), "vpd3": h.vapour_pressure_deficit.iloc[-72:].mean()}


def main():
    dirs = [p.parent for root in ("cases", "bank") for p in sorted((config.DATA / root).glob("*/burn_long.npz"))]
    done = pd.read_parquet(OUT) if OUT.exists() else pd.DataFrame()
    rows = done.to_dict("records")
    have = {r["case"] for r in rows}
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
