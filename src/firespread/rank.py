"""The features of the ranking model for one case directory (research case or live fire): the base set (distance, fuel shares, terrain, weather, fire, activity, season) and the later groups
(dryness, soil, landscape, no-wind path distance, greenness, roads and rivers, population). One code path for the research scripts and the live scoring job."""
import numpy as np
import pandas as pd
from scipy import ndimage

from . import activity, cases

DRY = ["rain7", "rain30", "rain60", "days_dry", "balance30", "tmax14", "rh14"]
SOIL = ["sm7", "sm28", "sm7_change", "vpd3"]
LAND = ["forest_5", "forest_15", "tree_5", "tree_15", "nonburn_dist", "nonburn_frac15", "elev", "rel_elev"]
PATH = ["path_t"]
GREEN = ["ndvi", "ndvi_anom"]
HUMAN = ["road_dist", "river_dist", "pop"]
S2 = ["s2_nir", "s2_swir1", "s2_swir2", "s2_ndmi", "s2_nbr", "s2_age", "s2_valid"]
WIND = ["wind", "wind_along"]


def base_features(c: cases.Case):
    ny, nx = c.fuel.shape
    sc = c.seeds.mean(axis=0)
    rr, cc = np.mgrid[0:ny, 0:nx]
    east, north = cc + 0.5 - (sc[1] + 0.5), -(rr + 0.5 - (sc[0] + 0.5))
    bearing = np.arctan2(east, north)
    hours = max(1, int(round(c.gap_h)))
    u = np.mean(-c.wind_ms[:hours] * np.sin(np.radians(c.wind_from[:hours])))
    v = np.mean(-c.wind_ms[:hours] * np.cos(np.radians(c.wind_from[:hours])))
    wind, w_to = np.hypot(u, v), np.arctan2(u, v)
    env = np.load(c.extra["dir"] / "env.npz")
    act = activity.counts(float(c.meta["lat"]), float(c.meta["lon"]), pd.Timestamp(c.meta["t0"]), c.meta["event"], recent=c.extra.get("recent"))
    has_dem = bool(env["has_dem"]) if "has_dem" in env.files else True
    hours_t0 = pd.Timestamp(c.meta["t0"])
    f = {
        "dist": c.dist_cells, "wind": np.full((ny, nx), wind), "wind_along": wind * np.cos(bearing - w_to),
        "slope": np.degrees(np.arctan(c.tan_slope)) if has_dem else np.full((ny, nx), np.nan), "slope_along": c.tan_slope * np.cos(bearing - np.radians(c.upslope)) if has_dem else np.full((ny, nx), np.nan),
        "tree": env["lc_tree"], "shrub": env["lc_shrub"], "grass": env["lc_grass"], "crop": env["lc_crop"], "built": env["lc_built"], "water": env["lc_water"],
        "forest_share": env["forest_share"], "moist": np.full((ny, nx), c.moisture[:hours].mean()), "temp": np.full((ny, nx), c.temp_c[:hours].mean()),
        "rh": np.full((ny, nx), c.rh[:hours].mean()), "frp": np.full((ny, nx), np.log1p(float(np.mean(env["start_frp"])))), "month": np.full((ny, nx), float(hours_t0.month)), "t0_hour": np.full((ny, nx), float(hours_t0.hour)), "act5": np.full((ny, nx), act["act5"]), "act20": np.full((ny, nx), act["act20"]), "act_ev20": np.full((ny, nx), act["act_ev20"]), "n_start": np.full((ny, nx), np.log1p(c.extra["n_start"])), "gap": np.full((ny, nx), c.gap_h),
    }
    return f


def extras(c, env, ant):
    f = {k: float(ant[k]) for k in DRY + SOIL}
    fs, tr = env["forest_share"].astype(np.float32), env["lc_tree"].astype(np.float32)
    for k in (5, 15):
        f[f"forest_{k}"] = ndimage.uniform_filter(fs, k)
        f[f"tree_{k}"] = ndimage.uniform_filter(tr, k)
    burnable = (env["lc_tree"] + env["lc_shrub"] + env["lc_grass"] + env["lc_crop"] >= 0.5) & (env["lc_built"] + env["lc_water"] < 0.5)
    f["nonburn_dist"] = ndimage.distance_transform_edt(burnable)
    f["nonburn_frac15"] = ndimage.uniform_filter((~burnable).astype(np.float32), 15)
    has_dem = bool(env["has_dem"]) if "has_dem" in env.files else True
    dem = env["dem"].astype(np.float32) if has_dem else np.full(fs.shape, np.nan, np.float32)
    f["elev"] = dem
    f["rel_elev"] = dem - np.nanmean(dem[c.seed_mask]) if has_dem else dem
    arr = cases.forecast_arrival(c, wind_scale=0.0)                           # fuel-aware travel time with no wind: a path distance that goes round barriers
    f["path_t"] = np.where(np.isfinite(arr), arr, c.gap_h + 6.0)
    return f


def build(c, d, ant):
    """All feature arrays (full grid, or a constant per case) for case `c` loaded from directory `d`; `ant` is the row with the DRY and SOIL values."""
    env = np.load(d / "env.npz")
    full = {**base_features(c), **extras(c, env, ant)}
    full["road_dist"], full["river_dist"] = (np.load(d / "osm.npz")[k] for k in ("road_dist", "river_dist"))
    full["pop"] = np.load(d / "pop.npz")["pop"]
    if (d / "ndvi.npz").exists():
        z = np.load(d / "ndvi.npz")
        full["ndvi"], full["ndvi_anom"] = z["ndvi"], z["ndvi"] - z["ndvi_prev"]
    else:
        full["ndvi"] = full["ndvi_anom"] = np.full(c.fuel.shape, np.nan, np.float32)
    if (d / "s2.npz").exists():                                                  # Sentinel-2 from just before the start (scripts/52_sentinel2.py)
        z = np.load(d / "s2.npz")
        nir, sw1, sw2 = z["nir"], z["swir1"], z["swir2"]
        full.update(s2_nir=nir, s2_swir1=sw1, s2_swir2=sw2, s2_ndmi=(nir - sw1) / (nir + sw1), s2_nbr=(nir - sw2) / (nir + sw2), s2_age=z["age"], s2_valid=np.isfinite(nir).astype(np.float32))
    else:
        nan = np.full(c.fuel.shape, np.nan, np.float32)
        full.update({k: nan for k in S2 if k != "s2_valid"}, s2_valid=np.zeros(c.fuel.shape, np.float32))
    return full
