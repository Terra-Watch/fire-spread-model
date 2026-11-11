"""Forest-land mask (boundary-defined forest, not tree cover) at 0.001 degrees, read lazily from disk."""
import numpy as np

from . import config


class ForestMask:
    def __init__(self, path=None):
        self.m = np.load(path or config.FOREST_MASK, mmap_mode="r")

    def _index(self, lat, lon):
        i = np.floor((config.FOREST_TOP - np.asarray(lat)) / config.FOREST_STEP).astype(np.int64)
        j = np.floor((np.asarray(lon) - config.FOREST_LEFT) / config.FOREST_STEP).astype(np.int64)
        return i, j

    def share(self, lat, lon, half: int = 1):
        """Mean of the (2 half + 1)^2 pixels around each point: with half = 1 about 330 m, close to one VIIRS pixel. Points outside the mask give 0."""
        i, j = self._index(lat, lon)
        h, w = self.m.shape
        total = np.zeros(i.shape, np.float32)
        n = (2 * half + 1) ** 2
        for di in range(-half, half + 1):
            for dj in range(-half, half + 1):
                ii, jj = i + di, j + dj
                ok = (ii >= 0) & (ii < h) & (jj >= 0) & (jj < w)
                v = np.zeros(i.shape, np.float32)
                v[ok] = self.m[ii[ok], jj[ok]]
                total += v
        return total / n
