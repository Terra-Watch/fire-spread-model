"""Score forecasts against the MODIS burn date instead of VIIRS detections: cells burned after the start of the forecast (by the burn day) versus cells that stayed unburned,
leaving out the zone right around the start (a MODIS pixel is about 5 of our cells wide) and cells that burned before the forecast began."""
import datetime as dt

import numpy as np
import pandas as pd
from scipy import ndimage

from . import cases
from .burnt import UNMAPPED

START_ZONE_CELLS = 5


def doy(date: dt.date) -> int:
    return date.timetuple().tm_yday


def burn_masks(c: cases.Case, burn_doy: np.ndarray, year: int, strict: bool = False):
    """(truth, candidates, usable). truth: burned between the day of the start and the day of the second observation (strict: only after the day of the start).
    usable is False when the case has no mapped cells or no positives."""
    t0, t1 = pd.Timestamp(c.meta["t0"]), pd.Timestamp(c.meta["t1"])
    d0 = doy(t0.date()) + (1 if strict else 0)
    d1 = doy(t1.date()) + 1                                    # the product dates a burn to the day; allow the day after the second observation
    if t0.year != year:
        return None, None, False
    mapped = burn_doy >= 0
    far = c.dist_cells >= START_ZONE_CELLS
    earlier = (burn_doy > 0) & (burn_doy < doy(t0.date()))
    cand = mapped & far & ~earlier & (c.dist_cells <= cases.CANDIDATE_CELLS)
    truth = cand & (burn_doy >= d0) & (burn_doy <= d1)
    return truth, cand, bool(truth.sum() >= 5 and (cand & ~truth).sum() >= 50)


def evaluate_burn(c: cases.Case, score: np.ndarray, truth: np.ndarray, cand: np.ndarray) -> dict:
    from . import scoring
    s, y, d = score[cand], truth[cand], c.dist_cells[cand]
    edges = [5, 8, 12, 16, 22, 30, 45, 80]
    return {"auc": scoring.auc(s, y), "dir_auc": scoring.auc_within_bins(s, y, d, edges), "area50": scoring.area_for_capture(s, y, 0.5), "n_pos": int(y.sum()), "n_cand": int(cand.sum())}


BRIDGE_CELLS = 3          # burned patches closer than this are one fire (the product is 500 m and patchy)
TOUCH_CELLS = 15          # the fire we forecast must have a burned patch within 1.5 km of its start


def connected_truth(c: cases.Case, truth: np.ndarray, cand: np.ndarray):
    """Keep only the burned cells that belong to patches touching the start of THIS fire, so other fires in the same 24 km window do not count as growth.
    The candidates are narrowed the same way: cells near other fires are not 'unburned', they are somebody else's."""
    burned = truth | ((c.dist_cells < START_ZONE_CELLS) & False)
    near_start = c.dist_cells <= TOUCH_CELLS
    grown = ndimage.binary_dilation(burned | c.seed_mask, iterations=BRIDGE_CELLS)
    lab, n = ndimage.label(grown, structure=np.ones((3, 3)))
    keep = np.unique(lab[(burned | c.seed_mask) & near_start])
    keep = keep[keep > 0]
    patch = np.isin(lab, keep)
    return truth & patch, cand
