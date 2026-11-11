"""Per-place inputs beyond terrain and land cover, each as a function of (place, start time) so the research scripts and a live scoring job build them the same way:
dryness before the start (ERA5 daily), soil moisture (ERA5 hourly), greenness (MODIS NDVI), roads and rivers (OpenStreetMap lines), population (WorldPop)."""
import json
import time

import httpx
import numpy as np
import pandas as pd
import rasterio
from pyproj import Transformer
from rasterio.enums import Resampling
from rasterio.features import rasterize
from rasterio.warp import reproject
from scipy import ndimage
from shapely.geometry import LineString

from . import burnt, config
from .grid import Grid

ARCHIVE = "https://archive-api.open-meteo.com/v1/archive"
NDVI_FILL = -3000
DIST_CAP = 240.0
ROADS = {"motorway", "trunk", "primary", "secondary", "tertiary", "unclassified", "track"}


def _archive(client, params):
    for k in range(6):
        r = client.get(ARCHIVE, params=params)
        if r.status_code == 200:
            return r.json()
        time.sleep(min(60, 5 * 2 ** k) if r.status_code == 429 else 2 ** k)
    raise RuntimeError(f"HTTP {r.status_code}")


def antecedent(client: httpx.Client, lat: float, lon: float, t0: pd.Timestamp, case_id: str) -> dict:
    """Rain, evaporation balance, hot spell and humidity over the 60 days before the start day (ERA5 daily)."""
    t0 = pd.Timestamp(t0).tz_localize(None)
    h = _archive(client, {"latitude": round(lat, 3), "longitude": round(lon, 3), "start_date": str((t0 - pd.Timedelta(days=60)).date()), "end_date": str(t0.date()),
                          "daily": "precipitation_sum,et0_fao_evapotranspiration,temperature_2m_max,relative_humidity_2m_mean", "timezone": "GMT"})["daily"]
    rain = np.nan_to_num(np.array(h["precipitation_sum"], float))[:-1]            # the days before the start day
    et0 = np.nan_to_num(np.array(h["et0_fao_evapotranspiration"], float))[:-1]
    tmax = np.array(h["temperature_2m_max"], float)[:-1]
    rh = np.array(h["relative_humidity_2m_mean"], float)[:-1]
    wet = np.flatnonzero(rain[::-1] >= 2.0)                                       # days back to the last day with at least 2 mm
    return {"case": case_id, "rain7": rain[-7:].sum(), "rain30": rain[-30:].sum(), "rain60": rain.sum(), "days_dry": float(wet[0]) if len(wet) else 60.0,
            "balance30": rain[-30:].sum() - et0[-30:].sum(), "tmax14": np.nanmean(tmax[-14:]), "rh14": np.nanmean(rh[-14:])}


def soil(client: httpx.Client, lat: float, lon: float, t0: pd.Timestamp, case_id: str) -> dict:
    """Soil moisture and vapour pressure deficit from the 7 days before the start (ERA5 hourly)."""
    t0 = pd.Timestamp(t0).tz_localize(None)
    h = pd.DataFrame(_archive(client, {"latitude": round(lat, 3), "longitude": round(lon, 3), "start_date": str((t0 - pd.Timedelta(days=7)).date()), "end_date": str(t0.date()),
                                       "hourly": "soil_moisture_0_to_7cm,soil_moisture_7_to_28cm,vapour_pressure_deficit", "timezone": "GMT"})["hourly"])
    h = h.assign(time=pd.to_datetime(h["time"]))
    h = h[h.time < t0].set_index("time")                                           # only what was known before the start
    last24, first = h.iloc[-24:], h.iloc[:24]
    return {"case": case_id, "sm7": last24.soil_moisture_0_to_7cm.mean(), "sm28": last24.soil_moisture_7_to_28cm.mean(),
            "sm7_change": last24.soil_moisture_0_to_7cm.mean() - first.soil_moisture_0_to_7cm.mean(), "vpd3": h.vapour_pressure_deficit.iloc[-72:].mean()}


def _ndvi_composite(grid: Grid, client: httpx.Client, end: pd.Timestamp) -> np.ndarray:
    """NDVI on the grid from the latest composite of each tile that started 16 to 32 days before `end`."""
    w, s, e, n = grid.bounds_lonlat(pad_deg=0.01)
    start = (end - pd.Timedelta(days=32)).strftime("%Y-%m-%dT00:00:00Z")
    stop = (end - pd.Timedelta(days=16)).strftime("%Y-%m-%dT00:00:00Z")
    r = client.post(burnt.STAC, json={"collections": ["modis-13Q1-061"], "bbox": [w, s, e, n], "datetime": f"{start}/{stop}", "limit": 20})
    feats = sorted(r.json().get("features", []), key=lambda f: f["properties"].get("start_datetime") or f["id"], reverse=True)
    out, seen = np.full(grid.shape, np.nan, np.float32), set()
    for f in feats:
        tile = f["id"].split(".")[2]
        if tile in seen:
            continue
        seen.add(tile)
        url = burnt._signed(client, f["assets"]["250m_16_days_NDVI"]["href"])
        with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR", CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif", GDAL_HTTP_MAX_RETRY="3"), rasterio.open(url) as ds:
            tmp = np.full(grid.shape, NDVI_FILL, np.int16)
            reproject(rasterio.band(ds, 1), tmp, dst_transform=grid.transform, dst_crs=f"EPSG:{grid.epsg}", resampling=Resampling.nearest, dst_nodata=NDVI_FILL)
        out = np.where(np.isnan(out) & (tmp > NDVI_FILL), tmp * 1e-4, out).astype(np.float32)
    return out


def ndvi(grid: Grid, client: httpx.Client, t0: pd.Timestamp) -> tuple[np.ndarray, np.ndarray]:
    """(greenness now, greenness a year earlier), NaN where there is no value."""
    t0 = pd.Timestamp(t0).tz_localize(None)
    return _ndvi_composite(grid, client, t0), _ndvi_composite(grid, client, t0 - pd.Timedelta(days=365))


def load_lines(path, need_tiles: set):
    """Road and river lines from a geojsonseq export of the OSM extract (see scripts/24_osm.py) that touch a quarter-degree tile in need_tiles: [(kind, lon/lat array)]."""
    kept = []
    with open(path) as f:
        for row in f:
            if '"LineString"' not in row[:200]:
                continue
            ft = json.loads(row.lstrip("\x1e"))
            p = ft["properties"]
            kind = "road" if p.get("highway") in ROADS else "river" if p.get("waterway") == "river" else None
            if kind is None:
                continue
            xy = np.array(ft["geometry"]["coordinates"])
            if set(map(tuple, np.unique(np.floor(xy * 4).astype(int), axis=0))) & need_tiles:
                kept.append((kind, xy))
    return kept


def tiles_of(grid: Grid) -> set:
    w, s, e, n = grid.bounds_lonlat(pad_deg=0.01)
    return {(i, j) for i in range(int(np.floor(w * 4)), int(np.floor(e * 4)) + 1) for j in range(int(np.floor(s * 4)), int(np.floor(n * 4)) + 1)}


def line_distances(grid: Grid, lines: list) -> dict:
    """Distance in cells to the nearest road or track and the nearest river (240 where none is in the window)."""
    w, s, e, n = grid.bounds_lonlat(pad_deg=0.01)
    tf = Transformer.from_crs(4326, grid.epsg, always_xy=True)
    out = {}
    for name, kind in (("road_dist", "road"), ("river_dist", "river")):
        geoms = [LineString(zip(*tf.transform(xy[:, 0], xy[:, 1]))) for k, xy in lines
                 if k == kind and len(xy) > 1 and xy[:, 0].min() <= e and xy[:, 0].max() >= w and xy[:, 1].min() <= n and xy[:, 1].max() >= s]
        m = rasterize(geoms, out_shape=grid.shape, transform=grid.transform, all_touched=True, dtype="uint8") if geoms else None
        out[name] = np.minimum(ndimage.distance_transform_edt(m == 0), DIST_CAP).astype(np.float32) if m is not None and m.any() else np.full(grid.shape, DIST_CAP, np.float32)
    return out


def population(grid: Grid, year: int) -> np.ndarray:
    """People per km2 (WorldPop 1 km; 2019 for 2019, 2020 for later years)."""
    with rasterio.open(config.RAW / "worldpop" / f"ind_ppp_{2019 if year <= 2019 else 2020}_1km_Aggregated.tif") as ds:
        tmp = np.full(grid.shape, np.nan, np.float32)
        reproject(rasterio.band(ds, 1), tmp, dst_transform=grid.transform, dst_crs=f"EPSG:{grid.epsg}", resampling=Resampling.bilinear, dst_nodata=np.nan)
    return np.nan_to_num(tmp, nan=0.0).astype(np.float32)
