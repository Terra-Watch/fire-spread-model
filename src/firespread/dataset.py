"""The dataset of one case: every picture layer, every per-fire number and every label, in a fixed order, so the image models, the tree models and the audits all read the same thing.
Layers are on the case's 100 m grid, cropped to 192 x 192 cells around the start. Nothing here looks past the start time except the labels."""
import json

import numpy as np
import pandas as pd
from scipy import ndimage
from scipy.spatial import cKDTree

from . import burn_eval, cases, config, events, rank

CROP = 192
DISK = np.fromfunction(lambda i, j: np.hypot(i - 2, j - 2) <= cases.PIXEL_RADIUS_CELLS, (5, 5)).astype(np.float32)   # a VIIRS pixel is about 375 m: cells within 190 m belong to it
CAP = 240.0

SPATIAL = ["seed", "det_frp", "det_age_h", "dist", "tree", "shrub", "grass", "crop", "built", "water", "bare", "forest_share", "forest_5", "forest_15", "tree_5", "tree_15",
           "nonburn_dist", "nonburn_frac15", "dist_built", "dist_crop", "dist_water", "elev", "rel_elev", "tpi", "slope", "aspect_sin", "aspect_cos", "slope_along",
           "ndvi", "ndvi_prev", "ndvi_anom", "s2_nir", "s2_swir1", "s2_swir2", "s2_ndmi", "s2_nbr", "s2_age", "s2_valid",
           "road_dist", "river_dist", "pop", "path_t", "burn_days_since", "burn_n", "fire_hist_90", "fire_hist_365", "fire_days_since", "other_fires_72h"]
SCALARS = ["n_start", "frp_sum_log", "extent_km", "act5", "act20", "act_ev20", "lat", "lon", "month_sin", "month_cos", "doy_sin", "doy_cos", "hour_sin", "hour_cos",
           "wind", "wind_u", "wind_v", "wind_max", "gust_max", "temp_mean", "temp_max", "rh_mean", "rh_min", "vpd_max", "rain_sum", "moist_mean", "moist_min",
           "rain7", "rain30", "rain60", "days_dry", "balance30", "tmax14", "rh14", "sm7", "sm28", "sm7_change", "vpd3"]
LABELS = ["y_modis", "y_modis_all", "y_viirs_new", "y_viirs_any", "burn_rel_day"]      # per horizon; plus the shared cand, modis_valid and dist maps


def stamp_counts(grid, lat, lon, weights=None):
    """How many detections (or their summed weight) fall within a pixel's reach of each cell."""
    r, c = grid.to_rc(np.asarray(lat), np.asarray(lon))
    ok = (r >= -3) & (r < grid.ny + 3) & (c >= -3) & (c < grid.nx + 3)
    h = np.zeros(grid.shape, np.float32)
    ri, ci = np.clip(np.floor(r[ok]).astype(int), 0, grid.ny - 1), np.clip(np.floor(c[ok]).astype(int), 0, grid.nx - 1)
    np.add.at(h, (ri, ci), 1.0 if weights is None else np.asarray(weights)[ok])
    return ndimage.convolve(h, DISK, mode="constant")


def load_detections(meta):
    """Forest detections of the case's year and the years around it (the history looks back a year; the labels look forward)."""
    year = int(meta["year"])
    frames = [pd.read_parquet(config.DATA / "detections" / f"{y}.parquet", columns=["latitude", "longitude", "t", "frp", "event", "epoch"]) for y in (year - 1, year, year + 1) if (config.DATA / "detections" / f"{y}.parquet").exists()]
    return pd.concat(frames, ignore_index=True)


def history_layers(grid, det, meta, fire):
    t0 = pd.Timestamp(meta["t0"])
    w, s, e, n = grid.bounds_lonlat(pad_deg=0.02)
    box = det[(det.longitude >= w) & (det.longitude <= e) & (det.latitude >= s) & (det.latitude <= n) & (det.t < t0) & (det.t >= t0 - pd.Timedelta(days=365))]
    out = {"fire_hist_365": stamp_counts(grid, box.latitude.values, box.longitude.values),
           "fire_hist_90": stamp_counts(grid, *(lambda b: (b.latitude.values, b.longitude.values))(box[box.t >= t0 - pd.Timedelta(days=90)])),
           "other_fires_72h": stamp_counts(grid, *(lambda b: (b.latitude.values, b.longitude.values))(box[(box.t >= t0 - pd.Timedelta(hours=72)) & (box.event != meta["event"])]))}
    last = np.full(grid.shape, np.nan, np.float32)
    r, c = grid.to_rc(box.latitude.values, box.longitude.values)
    days = ((t0 - box.t).dt.total_seconds() / 86400.0).values
    for rr, cc, dd in sorted(zip(r, c, days), key=lambda x: -x[2]):                 # oldest first, newer overwrite
        i, j = int(np.floor(rr)), int(np.floor(cc))
        i0, i1, j0, j1 = max(i - 2, 0), min(i + 3, grid.ny), max(j - 2, 0), min(j + 3, grid.nx)
        if i1 <= i0 or j1 <= j0:
            continue
        k = DISK[i0 - (i - 2): i1 - (i - 2), j0 - (j - 2): j1 - (j - 2)] > 0
        last[i0:i1, j0:j1] = np.where(k, dd, last[i0:i1, j0:j1])
    out["fire_days_since"] = np.where(np.isnan(last), 366.0, np.minimum(last, 366.0)).astype(np.float32)
    # the fire's own footprint at the start: summed power and hours since each cell was seen
    out["det_frp"] = stamp_counts(grid, fire.latitude.values, fire.longitude.values, fire.frp.values)
    age = np.full(grid.shape, np.nan, np.float32)
    rf, cf = grid.to_rc(fire.latitude.values, fire.longitude.values)
    for rr, cc, hh in zip(rf, cf, ((t0 - fire.t).dt.total_seconds() / 3600.0).values):
        i, j = int(np.floor(rr)), int(np.floor(cc))
        if 0 <= i < grid.ny and 0 <= j < grid.nx:
            age[i, j] = hh if np.isnan(age[i, j]) else min(age[i, j], hh)
    out["det_age_h"] = np.where(np.isnan(age), 0.0, age).astype(np.float32)
    return out


def terrain_context(env, seed_mask):
    lc = {k: env[f"lc_{k}"].astype(np.float32) for k in ("tree", "shrub", "grass", "crop", "built", "water", "bare")}
    def dist(mask):
        return np.minimum(ndimage.distance_transform_edt(~mask), CAP).astype(np.float32) if mask.any() else np.full(mask.shape, CAP, np.float32)
    out = {**lc, "dist_built": dist(lc["built"] >= 0.3), "dist_crop": dist(lc["crop"] >= 0.5), "dist_water": dist(lc["water"] >= 0.3)}
    dem = env["dem"].astype(np.float32)
    ok = np.isfinite(dem)
    out["tpi"] = (dem - ndimage.uniform_filter(np.where(ok, dem, np.nanmean(dem) if ok.any() else 0.0), 11)).astype(np.float32) if ok.any() else dem
    up = np.radians(np.nan_to_num(env["upslope"]).astype(np.float32))
    out["aspect_sin"], out["aspect_cos"] = np.sin(up), np.cos(up)
    return out


def spatial_layers(c, d, ant, det, meta, fire):
    """All SPATIAL layers (full grid) of case `c` (loaded with the longest horizon) from bundle directory `d`."""
    env = np.load(d / "env.npz")
    full = rank.build(c, d, ant)
    full.update(terrain_context(env, c.seed_mask))
    full.update(history_layers(c_grid(env), det, meta, fire))
    full["seed"] = c.seed_mask.astype(np.float32)
    if (d / "ndvi.npz").exists():
        full["ndvi_prev"] = np.load(d / "ndvi.npz")["ndvi_prev"]
    else:
        full["ndvi_prev"] = np.full(c.fuel.shape, np.nan, np.float32)
    if (d / "burnhist.npz").exists():
        z = np.load(d / "burnhist.npz"); full["burn_days_since"] = np.where(np.isnan(z["days_since"]), 366.0, z["days_since"]).astype(np.float32); full["burn_n"] = z["n_burns"].astype(np.float32)
    else:
        full["burn_days_since"] = full["burn_n"] = np.full(c.fuel.shape, np.nan, np.float32)
    full["tree"], full["shrub"], full["grass"], full["crop"], full["built"], full["water"], full["bare"] = (full[k] for k in ("tree", "shrub", "grass", "crop", "built", "water", "bare"))
    return {k: np.broadcast_to(full[k], c.fuel.shape).astype(np.float32) for k in SPATIAL}


def c_grid(env):
    from . import grid as G
    return G.Grid(int(env["grid"][0]), *map(float, env["grid"][1:4]), int(env["grid"][4]), int(env["grid"][5]))


def scalar_row(c, ant, fire, meta, H, gust_max):
    """The per-fire numbers (SCALARS) for horizon H: weather over the first H hours from the start, rain history, soil, fire size, season."""
    t0 = pd.Timestamp(meta["t0"]); h = max(1, int(H))
    ws, wd = c.wind_ms[:h], np.radians(c.wind_from[:h])
    u, v = np.mean(-ws * np.sin(wd)), np.mean(-ws * np.cos(wd))
    T, RH = c.temp_c[:h], c.rh[:h]
    es = 0.6108 * np.exp(17.27 * T / (T + 237.3))
    x, y = events.sinusoidal_km(fire.latitude.values, fire.longitude.values)
    solar = (t0.hour + t0.minute / 60 + float(meta["lon"]) / 15.0) % 24
    act = rank.activity.counts(float(meta["lat"]), float(meta["lon"]), t0, meta["event"], recent=c.extra.get("recent"))
    row = {"n_start": np.log1p(len(fire)), "frp_sum_log": np.log1p(float(fire.frp.sum())), "extent_km": float(np.hypot(np.ptp(x), np.ptp(y))), **act, "lat": float(meta["lat"]), "lon": float(meta["lon"]),
           "month_sin": np.sin(2 * np.pi * t0.month / 12), "month_cos": np.cos(2 * np.pi * t0.month / 12), "doy_sin": np.sin(2 * np.pi * t0.dayofyear / 365.25), "doy_cos": np.cos(2 * np.pi * t0.dayofyear / 365.25),
           "hour_sin": np.sin(2 * np.pi * solar / 24), "hour_cos": np.cos(2 * np.pi * solar / 24),
           "wind": float(np.hypot(u, v)), "wind_u": float(u), "wind_v": float(v), "wind_max": float(ws.max()), "gust_max": float(gust_max),
           "temp_mean": float(T.mean()), "temp_max": float(T.max()), "rh_mean": float(RH.mean()), "rh_min": float(RH.min()), "vpd_max": float((es * (1 - RH / 100)).max()),
           "rain_sum": float(c.precip[:h].sum()), "moist_mean": float(c.moisture[:h].mean()), "moist_min": float(c.moisture[:h].min())}
    row.update({k: float(ant[k]) for k in ("rain7", "rain30", "rain60", "days_dry", "balance30", "tmax14", "rh14", "sm7", "sm28", "sm7_change", "vpd3")})
    return np.array([row[k] for k in SCALARS], np.float32)


def labels(c, d, det, meta, row, H):
    """The labels for horizon H, full grid: the MODIS truth (connected to this fire, and all burned cells), the VIIRS new growth and any later detection of this fire, the MODIS burn day relative to the start. Also the cells asked about and the MODIS-mapped mask."""
    import dataclasses
    t0 = pd.Timestamp(meta["t0"])
    cc = dataclasses.replace(c, meta={**c.meta, "t1": str(t0 + pd.Timedelta(hours=H))})
    b = np.load(d / "burn_long.npz")
    truth, cand, ok = burn_eval.burn_masks(cc, b["doy"], int(b["year"]))
    if truth is None:
        shape = c.fuel.shape
        truth = cand = np.zeros(shape, bool)
    truth_c, _ = burn_eval.connected_truth(cc, truth, cand)
    # VIIRS labels: later detections of this fire within H hours; "new" = farther than 0.75 km from everything seen up to the start
    ev = det[det.event == meta["event"]]
    start, later = ev[ev.epoch <= int(row.epoch_from)], ev[(ev.epoch > int(row.epoch_from)) & (ev.t <= t0 + pd.Timedelta(hours=H))]
    grid = c_grid(np.load(d / "env.npz"))
    any_m = np.zeros(c.fuel.shape, bool); new_m = np.zeros(c.fuel.shape, bool)
    if len(later):
        sx, sy = events.sinusoidal_km(start.latitude.values, start.longitude.values); lx, ly = events.sinusoidal_km(later.latitude.values, later.longitude.values)
        new = cKDTree(np.column_stack([sx, sy])).query(np.column_stack([lx, ly]))[0] > config.GROWTH_TOL_KM
        any_m = cases._stamp(c.fuel.shape, np.column_stack(grid.to_rc(later.latitude.values, later.longitude.values)))
        if new.any():
            new_m = cases._stamp(c.fuel.shape, np.column_stack(grid.to_rc(later.latitude.values[new], later.longitude.values[new])))
    doy = b["doy"].astype(np.int32); rel = doy - t0.timetuple().tm_yday
    burn_rel = np.where((doy > 0) & (rel >= 0) & (rel <= 5), rel, -1).astype(np.int8)
    return {"y_modis": truth_c, "y_modis_all": truth, "y_viirs_new": new_m & ~c.seed_mask, "y_viirs_any": any_m & ~c.seed_mask, "burn_rel_day": burn_rel,
            "cand": cand, "modis_valid": doy >= 0}


# name -> (unit, source, native resolution, how it is made). Used for the data card; every layer and number in SPATIAL and SCALARS must be here (tests/test_dataset_doc.py).
DOC = {
    "seed": ("0/1", "NASA FIRMS VIIRS 375 m", "375 m", "cells within 190 m of a detection of this fire up to the start"),
    "det_frp": ("MW", "FIRMS VIIRS", "375 m", "summed fire radiative power of the fire's detections, spread to the cells they cover"),
    "det_age_h": ("h", "FIRMS VIIRS", "375 m", "hours between the detection and the start time"),
    "dist": ("cells of 100 m", "derived", "100 m", "distance to the fire footprint"),
    "tree": ("share 0-1", "ESA WorldCover v200 (2021)", "10 m", "share of 10 m pixels of that class in the 100 m cell"),
    "shrub": ("share", "ESA WorldCover", "10 m", "as above"), "grass": ("share", "ESA WorldCover", "10 m", "as above"), "crop": ("share", "ESA WorldCover", "10 m", "as above"),
    "built": ("share", "ESA WorldCover", "10 m", "as above"), "water": ("share", "ESA WorldCover", "10 m", "as above"), "bare": ("share", "ESA WorldCover", "10 m", "as above"),
    "forest_share": ("share", "boundary-defined forest-land mask", "0.001 deg (about 110 m)", "forest land by legal boundary, not tree cover"),
    "forest_5": ("share", "derived", "500 m", "forest_share averaged over 5 x 5 cells"), "forest_15": ("share", "derived", "1.5 km", "forest_share averaged over 15 x 15 cells"),
    "tree_5": ("share", "derived", "500 m", "tree share averaged over 5 x 5 cells"), "tree_15": ("share", "derived", "1.5 km", "tree share averaged over 15 x 15 cells"),
    "nonburn_dist": ("cells", "derived", "100 m", "distance to the nearest cell that cannot burn (built, water, bare, little vegetation)"), "nonburn_frac15": ("share", "derived", "1.5 km", "share of non-burnable cells in 15 x 15"),
    "dist_built": ("cells", "WorldCover", "100 m", "distance to the nearest cell with at least 30 % built-up"), "dist_crop": ("cells", "WorldCover", "100 m", "distance to the nearest cell with at least 50 % cropland"),
    "dist_water": ("cells", "WorldCover", "100 m", "distance to the nearest cell with at least 30 % water"),
    "elev": ("m", "Copernicus DEM GLO-30", "30 m", "averaged to 100 m"), "rel_elev": ("m", "Copernicus DEM", "100 m", "elevation minus the fire's mean elevation"), "tpi": ("m", "Copernicus DEM", "1.1 km", "elevation minus its 11 x 11 cell mean"),
    "slope": ("degrees", "Copernicus DEM", "100 m", "Horn slope"), "aspect_sin": ("-1..1", "Copernicus DEM", "100 m", "sine of the uphill direction"), "aspect_cos": ("-1..1", "Copernicus DEM", "100 m", "cosine of the uphill direction"),
    "slope_along": ("tan", "Copernicus DEM", "100 m", "slope in the direction from the fire to the cell (positive uphill from the fire)"),
    "ndvi": ("-", "MODIS MOD13Q1 v061", "250 m", "latest 16-day composite that started at least 16 days before the start"), "ndvi_prev": ("-", "MODIS MOD13Q1", "250 m", "the same composite a year earlier"),
    "ndvi_anom": ("-", "MODIS", "250 m", "ndvi minus ndvi_prev (negative: drier than a year ago)"),
    "s2_nir": ("reflectance", "Sentinel-2 L2A B08", "10 m", "newest clear pixel in the 45 days before the start, averaged to 100 m"), "s2_swir1": ("reflectance", "Sentinel-2 L2A B11", "20 m", "as above"),
    "s2_swir2": ("reflectance", "Sentinel-2 L2A B12", "20 m", "as above"), "s2_ndmi": ("-", "Sentinel-2", "100 m", "(NIR - SWIR1) / (NIR + SWIR1): vegetation moisture"), "s2_nbr": ("-", "Sentinel-2", "100 m", "(NIR - SWIR2) / (NIR + SWIR2): burn ratio"),
    "s2_age": ("days", "Sentinel-2", "100 m", "days between the scene and the start"), "s2_valid": ("0/1", "Sentinel-2", "100 m", "1 where a clear pixel was found"),
    "road_dist": ("cells", "OpenStreetMap (January 2021 extract)", "100 m", "distance to the nearest motorway, trunk, primary, secondary, tertiary, unclassified road or track"), "river_dist": ("cells", "OpenStreetMap", "100 m", "distance to the nearest river"),
    "pop": ("people/km2", "WorldPop 1 km (2019 or 2020)", "1 km", "bilinear to 100 m"),
    "path_t": ("h", "fire-spread engine, no wind", "100 m", "fuel-aware travel time from the fire (goes round barriers)"),
    "burn_days_since": ("days", "MODIS MCD64A1", "500 m", "days since the latest burn in the 365 days before the start; 366 = none"), "burn_n": ("months", "MODIS MCD64A1", "500 m", "months with a burn in that year"),
    "fire_hist_90": ("detections", "FIRMS VIIRS", "375 m", "detections within reach of the cell in the 90 days before the start"), "fire_hist_365": ("detections", "FIRMS VIIRS", "375 m", "same, 365 days"),
    "fire_days_since": ("days", "FIRMS VIIRS", "375 m", "days since the latest detection at the cell, 366 = none"), "other_fires_72h": ("detections", "FIRMS VIIRS", "375 m", "detections of other fires in the last 72 hours"),
    "n_start": ("log1p count", "FIRMS", "-", "detections of the fire up to the start"), "frp_sum_log": ("log1p MW", "FIRMS", "-", "summed power"), "extent_km": ("km", "FIRMS", "-", "bounding-box diagonal of the fire"),
    "act5": ("log1p count", "FIRMS", "-", "forest detections of other fires within 5 km in the 72 h before"), "act20": ("log1p count", "FIRMS", "-", "same, 20 km"), "act_ev20": ("log1p count", "FIRMS", "-", "number of other fires within 20 km"),
    "lat": ("deg", "-", "-", "fire centre"), "lon": ("deg", "-", "-", "fire centre"), "month_sin": ("-", "-", "-", "season"), "month_cos": ("-", "-", "-", "season"), "doy_sin": ("-", "-", "-", "day of year"), "doy_cos": ("-", "-", "-", "day of year"),
    "hour_sin": ("-", "-", "-", "local solar hour of the start"), "hour_cos": ("-", "-", "-", "local solar hour of the start"),
    "wind": ("m/s", "Open-Meteo archived forecast stream", "hourly", "mean vector wind over the horizon"), "wind_u": ("m/s", "Open-Meteo", "hourly", "east component of the mean wind"), "wind_v": ("m/s", "Open-Meteo", "hourly", "north component"),
    "wind_max": ("m/s", "Open-Meteo", "hourly", "strongest hourly wind in the horizon"), "gust_max": ("m/s", "Open-Meteo", "hourly", "strongest gust in the horizon"),
    "temp_mean": ("C", "Open-Meteo", "hourly", "mean over the horizon"), "temp_max": ("C", "Open-Meteo", "hourly", "maximum"), "rh_mean": ("%", "Open-Meteo", "hourly", "mean relative humidity"), "rh_min": ("%", "Open-Meteo", "hourly", "minimum"),
    "vpd_max": ("kPa", "Open-Meteo", "hourly", "largest vapour pressure deficit"), "rain_sum": ("mm", "Open-Meteo", "hourly", "rain over the horizon"),
    "moist_mean": ("fraction", "derived from Open-Meteo", "hourly", "dead-fuel moisture (equilibrium moisture with a 10 h lag, rain pushes it up)"), "moist_min": ("fraction", "derived", "hourly", "driest hour"),
    "rain7": ("mm", "ERA5 daily via Open-Meteo", "daily", "rain in the 7 days before the start day"), "rain30": ("mm", "ERA5", "daily", "30 days"), "rain60": ("mm", "ERA5", "daily", "60 days"),
    "days_dry": ("days", "ERA5", "daily", "days since at least 2 mm of rain"), "balance30": ("mm", "ERA5", "daily", "30-day rain minus evaporation"), "tmax14": ("C", "ERA5", "daily", "mean daily maximum, 14 days"), "rh14": ("%", "ERA5", "daily", "mean humidity, 14 days"),
    "sm7": ("m3/m3", "ERA5 hourly", "hourly", "soil moisture 0-7 cm, last 24 h before the start"), "sm28": ("m3/m3", "ERA5", "hourly", "soil moisture 7-28 cm"), "sm7_change": ("m3/m3", "ERA5", "hourly", "change in 0-7 cm moisture over the last 7 days"), "vpd3": ("kPa", "ERA5", "hourly", "mean vapour pressure deficit, last 72 h"),
}
