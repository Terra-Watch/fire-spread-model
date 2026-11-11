"""Score the likely next areas of an active fire. A fire is the set of its VIIRS detections; the bundle written for it has the same layout as a research case, so the features come from the
same code (rank.build). The weather source is a parameter: "fcst" live, "hist" to replay a past fire."""
import io
import json
from pathlib import Path

import httpx
import joblib
import numpy as np
import pandas as pd

from . import ancillary, cases, config, events, grid as G, rasters, rank, weather
from .forest import ForestMask

BANDS = (0.02, 0.06, 0.15)          # cumulative share of the candidate cells: band 3 (highest) = top 2%, band 2 = next 4%, band 1 = next 9%
START_ZONE_CELLS = 5                # a MODIS pixel is about 5 cells wide: cells nearer the start are not asked about


def fetch_firms_nrt(key: str, days: int = 3, bbox=(68.0, 6.0, 98.0, 36.0)) -> pd.DataFrame:
    """VIIRS near-real-time detections over India (both satellites) of the last `days` days, nominal and high confidence: latitude, longitude, t (UTC), frp."""
    w, s, e, n = bbox
    frames = []
    for sat in ("VIIRS_SNPP_NRT", "VIIRS_NOAA20_NRT"):
        r = httpx.get(f"https://firms.modaps.eosdis.nasa.gov/api/area/csv/{key}/{sat}/{w},{s},{e},{n}/{days}", timeout=120)
        if r.status_code != 200:                                      # httpx's own error text would carry the URL, and the URL carries the key
            raise RuntimeError(f"FIRMS {sat} answered HTTP {r.status_code}")
        frames.append(pd.read_csv(io.StringIO(r.text)))
    df = pd.concat(frames, ignore_index=True)
    df = df[df["confidence"].astype(str).str.lower().isin(["n", "h", "nominal", "high"])].copy()
    hhmm = df["acq_time"].astype(int)
    df["t"] = pd.to_datetime(df["acq_date"], utc=True) + pd.to_timedelta(hhmm // 100, unit="h") + pd.to_timedelta(hhmm % 100, unit="m")
    return df.drop_duplicates(["latitude", "longitude", "t"])[["latitude", "longitude", "t", "frp"]].reset_index(drop=True)


def active_fires(det: pd.DataFrame, now: pd.Timestamp, recent_h: float = 36.0, min_det: int = 3) -> tuple[dict, pd.DataFrame]:
    """Forest detections (latitude, longitude, t, frp) of the last days linked into fires. Returns ({event label: detections} for fires with a detection in the last recent_h hours and at least min_det detections (the research cases start with at least 3, so smaller ones are outside what the model was trained on), all forest detections with their event label)."""
    det = det.copy()
    share = ForestMask().share(det["latitude"].values, det["longitude"].values)
    det = det[share >= config.MIN_FOREST_SHARE].reset_index(drop=True)
    if det.empty:
        return {}, det.assign(event=pd.Series(dtype=str))
    det["event"] = [f"live-{k}" for k in events.link_events(det)]
    last = det.groupby("event")["t"].max()
    keep = last[last >= now - pd.Timedelta(hours=recent_h)].index
    return {e: g.reset_index(drop=True) for e, g in det[det.event.isin(keep)].groupby("event") if len(g) >= min_det}, det


def make_bundle(fire: pd.DataFrame, t0: pd.Timestamp, horizon_h: int, out: Path, lines: list, weather_kind: str = "fcst", client: httpx.Client | None = None) -> dict:
    """Write the case bundle for `fire` (its detections known at t0) in `out` and return the antecedent/soil row. Network: DEM, land cover are local/public; weather, rain history, soil, greenness as in the research scripts."""
    out.mkdir(parents=True, exist_ok=True)
    start = fire                                               # the caller passes the detections known at t0
    lat, lon = float(start.latitude.mean()), float(start.longitude.mean())
    g = G.make_grid(lat, lon)
    sr, sc = g.to_rc(start.latitude.values, start.longitude.values)
    dem = rasters.read_dem(g)
    slope, up = rasters.slope_upslope(np.where(np.isnan(dem), np.nanmean(dem) if np.isfinite(dem).any() else 0, dem), g.res)
    lc = rasters.read_landcover(g)
    np.savez_compressed(out / "env.npz", dem=dem, slope=slope, upslope=up, forest_share=rasters.forest_share(g), **{f"lc_{k}": v for k, v in lc.items()},
                        start_rc=np.column_stack([sr, sc]).astype(np.float32), start_age_h=((t0 - start["t"]).dt.total_seconds() / 3600).values.astype(np.float32),
                        start_frp=start["frp"].values.astype(np.float32), truth_rc=np.zeros((0, 2), np.float32), truth_t_h=np.zeros(0, np.float32), truth_new=np.zeros(0, bool),
                        grid=np.array([g.epsg, g.x0, g.y0, g.res, g.nx, g.ny], float), has_dem=np.array(True))
    wx = []
    for kind in (weather_kind, "era5"):
        try:
            w = weather.fetch(kind, lat, lon, (t0 - pd.Timedelta(days=2)).strftime("%Y-%m-%d"), (t0 + pd.Timedelta(hours=horizon_h, days=2)).strftime("%Y-%m-%d"))
        except Exception as e:                                                  # one source failing must not lose the fire
            print("weather", kind, e, flush=True)
            w = pd.DataFrame(columns=weather.VARS, index=pd.DatetimeIndex([], tz="UTC", name="time"))
        w = w.reset_index()
        w.insert(0, "source", kind)
        w.insert(2, "hours_from_t0", (w["time"] - t0).dt.total_seconds() / 3600.0)
        wx.append(w)
    pd.concat(wx, ignore_index=True).to_csv(out / "weather.csv", index=False)
    cid = out.name
    meta = {"case_id": cid, "event": str(fire["event"].iloc[0]) if "event" in fire else cid, "state": "", "split": "live", "year": int(t0.year), "t0": str(t0),
            "t1": str(t0 + pd.Timedelta(hours=horizon_h)), "gap_h": float(horizon_h), "lat": lat, "lon": lon, "n_start": int(len(start))}
    json.dump(meta, open(out / "meta.json", "w"))
    own = client is None
    client = client or httpx.Client(timeout=60)
    try:
        ant = {**ancillary.antecedent(client, lat, lon, t0, cid), **ancillary.soil(client, lat, lon, t0, cid)}
        now, prev = ancillary.ndvi(g, client, t0)
    finally:
        if own:
            client.close()
    np.savez_compressed(out / "ndvi.npz", ndvi=now, ndvi_prev=prev)
    np.savez_compressed(out / "osm.npz", **ancillary.line_distances(g, lines))
    np.savez_compressed(out / "pop.npz", pop=ancillary.population(g, int(t0.year)))
    json.dump({k: (float(v) if k != "case" else v) for k, v in ant.items()}, open(out / "ant.json", "w"))
    return ant


def score_bundle(d: Path, model_path: Path, recent: pd.DataFrame | None = None, weather_kind: str = "fcst") -> dict:
    """Score the cells around the fire in bundle `d` with a saved ranking model. Returns the score grid (NaN where not asked), the band grid (3 highest .. 1, 0 outside), the case and the model's version fields."""
    bundle = joblib.load(model_path)
    H = bundle["horizon_h"]
    c = cases.load_case(d, source=weather_kind, gap_override=H)
    c.extra["recent"] = recent
    ant = pd.Series(json.load(open(d / "ant.json")))
    full = rank.build(c, d, ant)
    cand = c.candidates & (c.dist_cells >= START_ZONE_CELLS)
    X = np.column_stack([np.broadcast_to(full[k], c.fuel.shape)[cand] for k in bundle["features"]])
    p = bundle["model"].predict_proba(X)[:, 1]
    score = np.full(c.fuel.shape, np.nan, np.float32)
    score[cand] = p
    order = (-p).argsort().argsort() / len(p)                       # 0 = highest ranked
    band = np.zeros(c.fuel.shape, np.int8)
    band[cand] = np.select([order < BANDS[0], order < BANDS[1], order < BANDS[2]], [3, 2, 1], 0)
    return {"score": score, "band": band, "case": c, "horizon_h": H, "weather_source": c.weather_source, "n_cells": int(cand.sum())}


def band_geojson(res: dict, grid: G.Grid, props: dict | None = None, simplify_m: float = 60.0) -> dict:
    """The bands (3 highest, 2, 1) of a scored fire as lon/lat polygons, one feature per band (cells of that band only), for the map."""
    from pyproj import Transformer
    from rasterio.features import shapes
    from shapely.geometry import mapping, shape
    from shapely.ops import transform, unary_union
    to_ll = Transformer.from_crs(grid.epsg, 4326, always_xy=True).transform
    feats = []
    for b in (1, 2, 3):
        mask = res["band"] == b
        if not mask.any():
            continue
        polys = [shape(g) for g, v in shapes(mask.astype(np.uint8), mask=mask, transform=grid.transform) if v == 1]
        geom = transform(to_ll, unary_union(polys).simplify(simplify_m))
        feats.append({"type": "Feature", "geometry": mapping(geom), "properties": {"band": b, **(props or {})}})
    return {"type": "FeatureCollection", "features": feats}
