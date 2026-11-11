"""Terrain and land cover on a case grid: Copernicus DEM (30 m, read over the network), ESA WorldCover (10 m, local tiles) and the forest-land mask."""
import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import reproject

from . import config
from .forest import ForestMask
from .grid import Grid

WC = {"tree": 10, "shrub": 20, "grass": 30, "crop": 40, "built": 50, "bare": 60, "snow": 70, "water": 80, "wetland": 90, "mangrove": 95, "moss": 100}


def _tile_name(lat, lon):
    return config.DEM_URL.format(ns="N" if lat >= 0 else "S", lat=abs(int(lat)), ew="E" if lon >= 0 else "W", lon=abs(int(lon)))


def read_dem(grid: Grid) -> np.ndarray:
    """Elevation in metres on the grid (NaN where no tile answered). Averaged from 30 m, so very steep detail is smoothed at a 100 m cell."""
    w, s, e, n = grid.bounds_lonlat(pad_deg=0.01)
    out = np.full(grid.shape, np.nan, np.float32)
    with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR", CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif", GDAL_HTTP_MAX_RETRY="3", GDAL_HTTP_RETRY_DELAY="1"):
        for la in range(int(np.floor(s)), int(np.floor(n)) + 1):
            for lo in range(int(np.floor(w)), int(np.floor(e)) + 1):
                try:
                    with rasterio.open("/vsicurl/" + _tile_name(la, lo)) as ds:
                        tmp = np.full(grid.shape, np.nan, np.float32)
                        reproject(rasterio.band(ds, 1), tmp, dst_transform=grid.transform, dst_crs=f"EPSG:{grid.epsg}", resampling=Resampling.average, dst_nodata=np.nan)
                        out = np.where(np.isnan(out), tmp, out)
                except rasterio.errors.RasterioIOError:
                    continue
    return out


def slope_upslope(dem: np.ndarray, res: float):
    """Slope in degrees and the compass direction (0 = north, clockwise) pointing uphill, by the Horn method. NaN stays NaN."""
    z = np.pad(dem, 1, mode="edge")
    dzdx = ((z[:-2, 2:] + 2 * z[1:-1, 2:] + z[2:, 2:]) - (z[:-2, :-2] + 2 * z[1:-1, :-2] + z[2:, :-2])) / (8 * res)    # east
    dzdy = ((z[:-2, :-2] + 2 * z[:-2, 1:-1] + z[:-2, 2:]) - (z[2:, :-2] + 2 * z[2:, 1:-1] + z[2:, 2:])) / (8 * res)    # north (rows grow southwards)
    slope = np.degrees(np.arctan(np.hypot(dzdx, dzdy)))
    up = (np.degrees(np.arctan2(dzdx, dzdy)) + 360.0) % 360.0
    return slope.astype(np.float32), up.astype(np.float32)


def read_landcover(grid: Grid, sub: int = 5) -> dict[str, np.ndarray]:
    """Share of each land-cover group in each cell (from 10 m WorldCover, sampled at res/sub) and the dominant class code. NaN-free; cells with no tile are all zero."""
    fine_res = grid.res / sub
    fine = np.zeros((grid.ny * sub, grid.nx * sub), np.uint8)
    w, s, e, n = grid.bounds_lonlat(pad_deg=0.01)
    from rasterio.transform import Affine
    ft = Affine(fine_res, 0, grid.x0, 0, -fine_res, grid.y0)
    for la in range(int(np.floor(s / 3)) * 3, int(np.floor(n / 3)) * 3 + 1, 3):
        for lo in range(int(np.floor(w / 3)) * 3, int(np.floor(e / 3)) * 3 + 1, 3):
            f = config.WORLDCOVER_DIR / f"ESA_WorldCover_10m_2021_v200_{'N' if la >= 0 else 'S'}{abs(la):02d}{'E' if lo >= 0 else 'W'}{abs(lo):03d}_Map.tif"
            if not f.exists():
                continue
            with rasterio.open(f) as ds:
                tmp = np.zeros(fine.shape, np.uint8)
                reproject(rasterio.band(ds, 1), tmp, dst_transform=ft, dst_crs=f"EPSG:{grid.epsg}", resampling=Resampling.nearest)
                fine = np.where(fine == 0, tmp, fine)
    blocks = fine.reshape(grid.ny, sub, grid.nx, sub).transpose(0, 2, 1, 3).reshape(grid.ny, grid.nx, sub * sub)
    out = {name: (blocks == code).mean(axis=2).astype(np.float32) for name, code in WC.items() if name in ("tree", "shrub", "grass", "crop", "built", "water", "bare")}
    counts = np.stack([(blocks == c).sum(axis=2) for c in WC.values()], axis=-1)
    codes = np.array(list(WC.values()), np.uint8)
    out["dominant"] = np.where(counts.sum(axis=2) > 0, codes[counts.argmax(axis=-1)], 0).astype(np.uint8)
    return out


def forest_share(grid: Grid, mask: ForestMask | None = None) -> np.ndarray:
    lat, lon = grid.centers_lonlat()
    return (mask or ForestMask()).share(lat.ravel(), lon.ravel(), half=1).reshape(grid.shape).astype(np.float32)
