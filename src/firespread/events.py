"""From detections to fire events, observation epochs and forecast cases. Pure functions on DataFrames (tested in tests/test_events.py).

event   detections linked in space (<= LINK_KM) and time (<= LINK_GAP_H), as connected components.
epoch   one observation of the event: passes closer than EPOCH_MERGE_H (two satellites, neighbouring swaths) are merged.
case    a pair of consecutive epochs. The start is every detection of the event up to the first epoch; the truth is the second epoch.
        Detections of the second epoch farther than GROWTH_TOL_KM from everything in the start are new growth.
"""
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

from . import config

R_KM = 6371.0088


def sinusoidal_km(lat, lon):
    """Equal-area planar coordinates in km, good for distances of a few km anywhere in India."""
    la, lo = np.radians(np.asarray(lat, float)), np.radians(np.asarray(lon, float))
    return R_KM * lo * np.cos(la), R_KM * la


def link_events(det: pd.DataFrame, max_km: float = config.LINK_KM, max_gap_h: float = config.LINK_GAP_H) -> np.ndarray:
    """Event label per row (0..n-1, arbitrary order). Linked when within max_km AND max_gap_h of another member (single linkage, so a spreading fire stays one event)."""
    n = len(det)
    if n == 0:
        return np.zeros(0, np.int64)
    x, y = sinusoidal_km(det["latitude"].values, det["longitude"].values)
    th = ((det["t"] - det["t"].min()).dt.total_seconds() / 3600.0).values                         # hours, whatever the datetime resolution is
    pts = np.column_stack([x, y, th / max_gap_h * max_km])
    pairs = cKDTree(pts).query_pairs(r=max_km, p=np.inf, output_type="ndarray")           # box in space and time; the exact circle follows
    if len(pairs):
        keep = np.hypot(x[pairs[:, 0]] - x[pairs[:, 1]], y[pairs[:, 0]] - y[pairs[:, 1]]) <= max_km
        pairs = pairs[keep]
    graph = coo_matrix((np.ones(len(pairs), np.int8), (pairs[:, 0], pairs[:, 1])), shape=(n, n))
    return connected_components(graph, directed=False)[1].astype(np.int64)


def assign_epochs(det: pd.DataFrame, merge_h: float = config.EPOCH_MERGE_H) -> pd.Series:
    """Epoch number (0, 1, 2 ...) per row within its event: detections whose time is within merge_h of the previous detection (in time order) share an epoch."""
    out = pd.Series(0, index=det.index, dtype="int64")
    for _, g in det.groupby("event", sort=False):
        t = g["t"].sort_values()
        gap = t.diff().dt.total_seconds().fillna(0.0) / 3600.0
        out.loc[t.index] = (gap > merge_h).cumsum().values
    return out


def event_table(det: pd.DataFrame) -> pd.DataFrame:
    """One row per event: size, duration, number of epochs, centre."""
    g = det.groupby("event")
    x, y = sinusoidal_km(det["latitude"].values, det["longitude"].values)
    det = det.assign(_x=x, _y=y)
    g = det.groupby("event")
    out = pd.DataFrame({
        "n_det": g.size(), "n_epochs": g["epoch"].nunique(), "t_first": g["t"].min(), "t_last": g["t"].max(),
        "lat": g["latitude"].mean(), "lon": g["longitude"].mean(), "frp_sum": g["frp"].sum(),
        "extent_km": np.hypot(g["_x"].max() - g["_x"].min(), g["_y"].max() - g["_y"].min()),
    })
    out["duration_h"] = (out["t_last"] - out["t_first"]).dt.total_seconds() / 3600.0
    return out.reset_index()


def case_table(det: pd.DataFrame, tol_km: float = config.GROWTH_TOL_KM) -> pd.DataFrame:
    """One row per pair of consecutive epochs within an event, with how much of the second epoch is new growth and how far it advanced."""
    x, y = sinusoidal_km(det["latitude"].values, det["longitude"].values)
    det = det.assign(_x=x, _y=y)
    rows = []
    for ev, g in det.groupby("event", sort=False):
        epochs = sorted(g["epoch"].unique())
        if len(epochs) < 2:
            continue
        by = {e: g[g["epoch"] == e] for e in epochs}
        for a, b in zip(epochs[:-1], epochs[1:]):
            start = g[g["epoch"] <= a]
            nxt = by[b]
            d, _ = cKDTree(np.column_stack([start["_x"], start["_y"]])).query(np.column_stack([nxt["_x"], nxt["_y"]]))
            new = d > tol_km
            t0, t1 = by[a]["t"].mean(), nxt["t"].mean()
            rows.append({
                "event": ev, "epoch_from": int(a), "epoch_to": int(b), "t0": t0, "t1": t1, "gap_h": (t1 - t0).total_seconds() / 3600.0,
                "n_start": len(start), "n_last": len(by[a]), "n_truth": len(nxt), "n_new": int(new.sum()),
                "adv_p95_km": float(np.percentile(d[new], 95)) if new.any() else 0.0, "adv_max_km": float(d[new].max()) if new.any() else 0.0,
                "lat": float(start["latitude"].mean()), "lon": float(start["longitude"].mean()),
                "frp_last": float(by[a]["frp"].sum()),
            })
    return pd.DataFrame(rows)
