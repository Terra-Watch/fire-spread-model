"""Load a case bundle into what the engine and the scoring need, and run a forecast and its baselines on it."""
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import distance_transform_edt
from scipy.spatial import cKDTree

from . import config, engine, moisture, scoring
from .engine import CROP, FOREST, GRASS, NONBURNABLE, SHRUB, FuelTable

PIXEL_RADIUS_CELLS = 1.9          # a VIIRS pixel is about 375 m: cells within 190 m of its centre belong to it (grid cell 100 m)
CANDIDATE_CELLS = 80              # candidate area: up to 8 km from the start footprint
DIR_BIN_EDGES_CELLS = [8, 12, 16, 22, 30, 45, 80]     # 0.8 to 8 km: growth is counted new only beyond 0.75 km


class NoWeather(Exception):
    """No source had enough hourly wind around this fire."""


@dataclass
class Case:
    id: str
    meta: dict
    res: float
    fuel: np.ndarray
    tan_slope: np.ndarray
    upslope: np.ndarray
    cell_factor: np.ndarray
    seeds: np.ndarray                  # (n, 2) cells burning at hour 0
    seed_mask: np.ndarray
    truth: np.ndarray                  # cells of newly detected pixels at the second observation
    candidates: np.ndarray             # cells where growth is asked about
    dist_cells: np.ndarray             # distance to the start footprint, in cells
    wind_ms: np.ndarray                # hourly from the start, 10 m
    wind_from: np.ndarray
    temp_c: np.ndarray
    rh: np.ndarray
    precip: np.ndarray
    moisture: np.ndarray
    gap_h: float
    weather_source: str
    extra: dict = field(default_factory=dict)


def _stamp(shape, rc, radius=PIXEL_RADIUS_CELLS):
    m = np.zeros(shape, bool)
    r = int(np.ceil(radius))
    for a, b in rc:
        i0, j0 = int(np.floor(a)), int(np.floor(b))
        for di in range(-r, r + 1):
            for dj in range(-r, r + 1):
                i, j = i0 + di, j0 + dj
                if 0 <= i < shape[0] and 0 <= j < shape[1] and np.hypot(i + 0.5 - a, j + 0.5 - b) <= radius:
                    m[i, j] = True
    return m


def fuel_classes(lc_tree, lc_shrub, lc_grass, lc_crop, lc_built, lc_water, lc_bare):
    shares = np.stack([lc_tree, lc_shrub, lc_grass, lc_crop], axis=0)
    best = shares.argmax(axis=0)
    ok = shares.max(axis=0) >= 0.3
    ids = np.array([FOREST, SHRUB, GRASS, CROP])[best]
    blocked = (lc_water + lc_built + lc_bare) > 0.6
    return np.where(ok & ~blocked, ids, NONBURNABLE).astype(np.int64)


def _hourly(df: pd.DataFrame, col: str, hours: np.ndarray):
    d = df.dropna(subset=[col])
    return np.interp(hours, d["hours_from_t0"].values, d[col].values)


def load_case(case_dir: Path, source: str = "hist", gap_override: float | None = None) -> Case:
    case_dir = Path(case_dir)
    meta = json.load(open(case_dir / "meta.json"))
    env = np.load(case_dir / "env.npz")
    wx = pd.read_csv(case_dir / "weather.csv", parse_dates=["time"])
    used, w = None, None
    for src in (source, "hist", "era5"):                                   # the wanted source first, then whatever has the hours
        cand = wx[wx.source == src].sort_values("hours_from_t0")
        if cand.wind_speed_10m.notna().sum() >= 24:
            used, w = src, cand
            break
    if w is None:
        raise NoWeather(meta["case_id"])
    gap = float(gap_override or meta["gap_h"])
    hours = np.arange(0.0, gap + 12.0, 1.0)
    u = -w.wind_speed_10m * np.sin(np.radians(w.wind_direction_10m))      # wind FROM the direction: components of where it blows to, then back
    v = -w.wind_speed_10m * np.cos(np.radians(w.wind_direction_10m))
    w = w.assign(u=u, v=v)
    uu, vv = _hourly(w, "u", hours), _hourly(w, "v", hours)
    wind_ms = np.hypot(uu, vv)
    wind_from = (np.degrees(np.arctan2(-uu, -vv)) + 360.0) % 360.0
    full_t = moisture.dead_fuel(w.temperature_2m.ffill().bfill().values, w.relative_humidity_2m.ffill().bfill().values, w.precipitation.fillna(0).values)
    w = w.assign(moist=full_t)
    shape = env["dem"].shape
    start = env["start_rc"]
    seed_mask = _stamp(shape, start)
    truth = _stamp(shape, env["truth_rc"][env["truth_new"]]) & ~seed_mask
    dist = distance_transform_edt(~seed_mask)
    fuel = fuel_classes(env["lc_tree"], env["lc_shrub"], env["lc_grass"], env["lc_crop"], env["lc_built"], env["lc_water"], env["lc_bare"])
    seeds = np.argwhere(seed_mask)
    new_rc = env["truth_rc"][env["truth_new"]]
    adv = cKDTree(start).query(new_rc)[0] * float(env["grid"][3]) / 1000.0 if len(new_rc) else np.array([0.0])
    return Case(id=meta["case_id"], meta=meta, res=float(env["grid"][3]), fuel=fuel, tan_slope=np.tan(np.radians(np.nan_to_num(env["slope"]))).astype(np.float64),
                upslope=np.nan_to_num(env["upslope"]).astype(np.float64), cell_factor=np.ones(shape), seeds=seeds, seed_mask=seed_mask, truth=truth,
                candidates=(~seed_mask) & (dist <= CANDIDATE_CELLS), dist_cells=dist, wind_ms=wind_ms, wind_from=wind_from,
                temp_c=_hourly(w, "temperature_2m", hours), rh=_hourly(w, "relative_humidity_2m", hours), precip=_hourly(w, "precipitation", hours),
                moisture=_hourly(w, "moist", hours), gap_h=gap, weather_source=used,
                extra={"adv_p95_km": float(np.percentile(adv, 95)), "adv_max_km": float(adv.max()), "n_start": int(len(start)), "dir": case_dir})


def forecast_arrival(c: Case, fuels: FuelTable | None = None, beta_s: float = 3.0, wind_scale: float = 1.0, rate_scale: float = 1.0) -> np.ndarray:
    fuels = fuels or FuelTable.priors()
    if rate_scale != 1.0:
        fuels = FuelTable(fuels.r0 * rate_scale, fuels.cw, fuels.bw, fuels.mx, fuels.waf)
    return engine.arrival_times(c.fuel, c.tan_slope, c.upslope, c.cell_factor, fuels, c.wind_ms * wind_scale, c.wind_from, c.moisture, c.seeds, c.gap_h + 6.0, c.res, beta_s)


def scores_from_arrival(c: Case, arrival: np.ndarray) -> np.ndarray:
    """Higher = more likely to have grown by the second observation. Cells the fire does not reach are ranked by their distance from the start."""
    return np.where(np.isfinite(arrival), -arrival, -1e6 - c.dist_cells)


def baseline_proximity(c: Case) -> np.ndarray:
    return -c.dist_cells


def baseline_wind_ellipse(c: Case) -> np.ndarray:
    """Spread through uniform forest on the flat with the same wind and a dry fuel: the shape of a wind-driven fire, nothing else."""
    ones = np.full(c.fuel.shape, FOREST, np.int64)
    arr = engine.arrival_times(ones, np.zeros(c.fuel.shape), np.zeros(c.fuel.shape), np.ones(c.fuel.shape), FuelTable.priors(), c.wind_ms, c.wind_from,
                               np.full(len(c.wind_ms), 0.08), c.seeds, c.gap_h + 6.0, c.res)
    return scores_from_arrival(c, arr)


def evaluate(c: Case, score: np.ndarray) -> dict:
    s, y = score[c.candidates], c.truth[c.candidates]
    d = c.dist_cells[c.candidates]
    return {"auc": scoring.auc(s, y), "dir_auc": scoring.auc_within_bins(s, y, d, DIR_BIN_EDGES_CELLS), "area50": scoring.area_for_capture(s, y, 0.5), "area80": scoring.area_for_capture(s, y, 0.8), "n_pos": int(y.sum()), "n_cand": int(c.candidates.sum())}


def predicted_reach_km(c: Case, arrival: np.ndarray, q: float = 0.95) -> float:
    """How far the forecast fire gets from the start footprint by the second observation: the q-quantile of the distance of the cells it reaches (km)."""
    m = np.isfinite(arrival) & (arrival <= c.gap_h) & ~c.seed_mask
    return float(np.quantile(c.dist_cells[m], q) * c.res / 1000.0) if m.any() else 0.0
