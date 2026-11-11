"""A square metric grid (UTM) centred on a fire, and the conversions between map positions and cells."""
from dataclasses import dataclass

import numpy as np
from pyproj import Transformer
from rasterio.transform import Affine


@dataclass(frozen=True)
class Grid:
    epsg: int
    x0: float          # west edge, metres
    y0: float          # north edge, metres
    res: float         # cell size, metres
    nx: int
    ny: int

    @property
    def transform(self) -> Affine:
        return Affine(self.res, 0, self.x0, 0, -self.res, self.y0)

    @property
    def shape(self):
        return (self.ny, self.nx)

    def to_xy(self, lat, lon):
        return Transformer.from_crs(4326, self.epsg, always_xy=True).transform(np.asarray(lon, float), np.asarray(lat, float))

    def to_rc(self, lat, lon):
        """Fractional (row, col) of each position; floor() gives the cell."""
        x, y = self.to_xy(lat, lon)
        return (self.y0 - y) / self.res, (x - self.x0) / self.res

    def centers_lonlat(self):
        cols = self.x0 + (np.arange(self.nx) + 0.5) * self.res
        rows = self.y0 - (np.arange(self.ny) + 0.5) * self.res
        xx, yy = np.meshgrid(cols, rows)
        lon, lat = Transformer.from_crs(self.epsg, 4326, always_xy=True).transform(xx, yy)
        return lat, lon

    def bounds_lonlat(self, pad_deg: float = 0.0):
        xs = [self.x0, self.x0 + self.nx * self.res]
        ys = [self.y0 - self.ny * self.res, self.y0]
        lon, lat = Transformer.from_crs(self.epsg, 4326, always_xy=True).transform([xs[0], xs[1], xs[0], xs[1]], [ys[0], ys[0], ys[1], ys[1]])
        return min(lon) - pad_deg, min(lat) - pad_deg, max(lon) + pad_deg, max(lat) + pad_deg


def make_grid(lat: float, lon: float, half_km: float = 12.0, res: float = 100.0) -> Grid:
    """A grid of 2 half_km on a side, centred on (lat, lon), in the UTM zone of the point, aligned to multiples of `res` so that neighbouring grids line up."""
    zone = int((lon + 180.0) // 6.0) + 1
    epsg = (32600 if lat >= 0 else 32700) + zone
    x, y = Transformer.from_crs(4326, epsg, always_xy=True).transform(lon, lat)
    n = int(np.ceil(2.0 * half_km * 1000.0 / res))
    x0 = np.floor((x - n * res / 2.0) / res) * res
    y0 = np.ceil((y + n * res / 2.0) / res) * res
    return Grid(epsg, float(x0), float(y0), float(res), n, n)
