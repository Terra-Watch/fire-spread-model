"""Hourly weather at a fire from Open-Meteo, kept on disk so each place and period is asked for once.
Three views of the same hours:
  hist   the archived forecast stream (historical-forecast API): the best model's own short-range output, close to what was happening
  prev1  the forecast made the day before (previous-runs API): the honest forecast error for a 6 to 24 h look-ahead (not available for every year)
  era5   reanalysis (archive API): complete from 1940, smoother winds
  fcst   the live forecast API: recent past hours and the coming days (for a live job; not kept as history)
"""
import hashlib
import json
import time
from pathlib import Path

import httpx
import pandas as pd

from . import config

VARS = ["temperature_2m", "relative_humidity_2m", "dew_point_2m", "precipitation", "wind_speed_10m", "wind_direction_10m", "wind_gusts_10m"]
URL = {"hist": "https://historical-forecast-api.open-meteo.com/v1/forecast", "prev1": "https://previous-runs-api.open-meteo.com/v1/forecast", "era5": "https://archive-api.open-meteo.com/v1/archive", "fcst": "https://api.open-meteo.com/v1/forecast"}
CACHE = config.DATA / "weather_cache"
_last_call = [0.0]


def _params(kind, lat, lon, start, end):
    hourly = [f"{v}_previous_day1" for v in VARS] if kind == "prev1" else VARS
    p = {"latitude": round(lat, 3), "longitude": round(lon, 3), "start_date": start, "end_date": end, "hourly": ",".join(hourly), "wind_speed_unit": "ms", "timezone": "GMT"}
    return p


def fetch(kind: str, lat: float, lon: float, start: str, end: str, client: httpx.Client | None = None, pause: float = 0.25) -> pd.DataFrame:
    """Hourly frame indexed by UTC time with the VARS columns (NaN where the source had nothing). Raises on a failed request after retries."""
    p = _params(kind, lat, lon, start, end)
    key = hashlib.sha1((kind + json.dumps(p, sort_keys=True)).encode()).hexdigest()[:20]
    path = CACHE / f"{key}.json"
    if path.exists():
        body = json.load(open(path))
    else:
        CACHE.mkdir(parents=True, exist_ok=True)
        own = client is None
        client = client or httpx.Client(timeout=60)
        try:
            for attempt in range(6):
                wait = pause - (time.time() - _last_call[0])
                if wait > 0:
                    time.sleep(wait)
                _last_call[0] = time.time()
                r = client.get(URL[kind], params=p)
                if r.status_code == 200:
                    body = r.json()
                    break
                if r.status_code == 429:
                    time.sleep(min(60, 5 * 2 ** attempt))
                    continue
                if r.status_code == 400:        # outside what this source covers
                    body = {"hourly": {"time": []}, "error": r.text[:200]}
                    break
                time.sleep(2 ** attempt)
            else:
                raise RuntimeError(f"Open-Meteo {kind} failed: HTTP {r.status_code}")
        finally:
            if own:
                client.close()
        json.dump(body, open(path, "w"))
    h = body.get("hourly", {})
    if not h.get("time"):
        return pd.DataFrame(columns=VARS, index=pd.DatetimeIndex([], tz="UTC", name="time"))
    df = pd.DataFrame({v: h.get(f"{v}_previous_day1" if kind == "prev1" else v) for v in VARS}, index=pd.to_datetime(h["time"], utc=True))
    df.index.name = "time"
    return df.astype("float32")
