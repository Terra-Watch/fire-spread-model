"""How busy is the neighbourhood? Forest detections within 5 and 20 km in the three days before a forecast starts (the fire's own event left out): fire-season pressure,
dryness and human burning, from detections only, with nothing from the future."""
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from . import config
from .events import sinusoidal_km

_CACHE: dict = {}


def _year(year: int):
    if year not in _CACHE:
        d = pd.read_parquet(config.DATA / "detections" / f"{year}.parquet", columns=["latitude", "longitude", "t", "event"])
        x, y = sinusoidal_km(d["latitude"].values, d["longitude"].values)
        hours = ((d["t"] - pd.Timestamp(f"{year}-01-01", tz="UTC")).dt.total_seconds() / 3600.0).values
        _CACHE[year] = (cKDTree(np.column_stack([x, y])), hours, d["event"].values)
    return _CACHE[year]


def counts(lat: float, lon: float, t0: pd.Timestamp, event: str, back_h: float = 72.0, recent: pd.DataFrame | None = None) -> dict:
    """`recent`: detections (latitude, longitude, t, event) to use instead of the year files, for a live job."""
    if recent is not None:
        x, y = sinusoidal_km(recent["latitude"].values, recent["longitude"].values)
        tree, ev = cKDTree(np.column_stack([x, y])), recent["event"].values
        hours = ((recent["t"] - pd.Timestamp(f"{t0.year}-01-01", tz="UTC")).dt.total_seconds() / 3600.0).values
    else:
        tree, hours, ev = _year(t0.year)
    x, y = sinusoidal_km(np.array([lat]), np.array([lon]))
    idx = np.array(tree.query_ball_point([x[0], y[0]], r=20.0), dtype=np.int64)
    if idx.size == 0:
        return {"act5": 0.0, "act20": 0.0, "act_ev20": 0.0}
    h0 = (t0 - pd.Timestamp(f"{t0.year}-01-01", tz="UTC")).total_seconds() / 3600.0
    ok = idx[(hours[idx] >= h0 - back_h) & (hours[idx] < h0) & (ev[idx] != event)]
    if ok.size == 0:
        return {"act5": 0.0, "act20": 0.0, "act_ev20": 0.0}
    d, _ = tree.query(np.column_stack([x, y]), k=1)           # unused: keeps the tree warm
    pts = tree.data[ok]
    r = np.hypot(pts[:, 0] - x[0], pts[:, 1] - y[0])
    return {"act5": float(np.log1p((r <= 5).sum())), "act20": float(np.log1p(ok.size)), "act_ev20": float(np.log1p(len(set(ev[ok]))))}
