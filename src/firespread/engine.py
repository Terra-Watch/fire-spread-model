"""Fire spread as minimum travel time on a grid: the arrival time of the fire in every cell.

Each cell lets the fire pass at an elliptical rate (fast with the wind, slower across, slowest against), scaled by the fuel, the dead-fuel moisture, the slope and a per-cell factor
(barriers). Wind and moisture change hour by hour; the conditions used on a step are those of the hour in which the fire leaves the cell. Sixteen neighbours per cell keep the grid
from imprinting itself on the shape. Pure arithmetic: same inputs, same arrival times.

Units: metres, hours, degrees (compass: 0 north, clockwise). Wind direction is where the wind blows FROM, as in every weather source.
"""
from dataclasses import dataclass

import numpy as np
from numba import njit

# 8 neighbours and 8 knight moves: (rows down, columns right)
OFFSETS = np.array([(-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1),
                    (-2, 1), (-1, 2), (1, 2), (2, 1), (2, -1), (1, -2), (-1, -2), (-2, -1)], dtype=np.int64)
DIST = np.hypot(OFFSETS[:, 0], OFFSETS[:, 1])                                   # in cells
BEARING = np.arctan2(OFFSETS[:, 1], -OFFSETS[:, 0])                             # radians, 0 north, clockwise
KMH_PER_MS = 3.6
MPH_PER_KMH = 0.621371

NONBURNABLE, FOREST, SHRUB, GRASS, CROP = 0, 1, 2, 3, 4


@dataclass
class FuelTable:
    """Per fuel class (index = class id): base spread rate with no wind or slope on dry fuel (m/h), wind coefficient and exponent, moisture of extinction (fraction), wind reduction to flame height."""
    r0: np.ndarray
    cw: np.ndarray
    bw: np.ndarray
    mx: np.ndarray
    waf: np.ndarray

    @staticmethod
    def priors() -> "FuelTable":
        """Starting values, NOT measurements: no Indian fuel model exists. Meant to be fitted on the dev cases (scripts/07_calibrate.py)."""
        return FuelTable(r0=np.array([0, 60, 90, 150, 50], float), cw=np.array([0, .35, .40, .30, .30], float), bw=np.array([1, 1.2, 1.2, 1.3, 1.2], float),
                         mx=np.array([.3, .30, .30, .20, .20], float), waf=np.array([.3, .20, .35, .45, .45], float))


def damping(moisture, mx):
    """Rothermel's moisture damping: 1 when dry, 0 at the moisture of extinction."""
    rm = np.clip(moisture / mx, 0.0, 1.0)
    return np.clip(1.0 - 2.59 * rm + 5.11 * rm ** 2 - 3.52 * rm ** 3, 0.0, 1.0)


def length_to_width(u_kmh):
    """Anderson (1983) from the wind at flame height, capped at 8."""
    u = np.asarray(u_kmh, float) * MPH_PER_KMH
    return np.clip(0.936 * np.exp(0.2566 * u) + 0.461 * np.exp(-0.1548 * u) - 0.397, 1.0, 8.0)


def head_rates(fuels: FuelTable, wind_ms, moisture):
    """Head spread rate [hour, fuel class] (m/h) and the eccentricity of the fire ellipse [hour], from hourly 10 m wind (m/s) and dead-fuel moisture (fraction)."""
    wind_ms, moisture = np.asarray(wind_ms, float), np.asarray(moisture, float)
    u = wind_ms[:, None] * KMH_PER_MS * fuels.waf[None, :]                       # km/h at flame height, per class
    r = fuels.r0[None, :] * damping(moisture[:, None], fuels.mx[None, :]) * (1.0 + fuels.cw[None, :] * u ** fuels.bw[None, :])
    r[:, NONBURNABLE] = 0.0
    lw = length_to_width(wind_ms * KMH_PER_MS * 0.30)                             # one eccentricity per hour, from the wind at a typical 0.3 reduction
    return r, np.sqrt(1.0 - 1.0 / lw ** 2)


@njit(cache=True)
def _travel_times(fuel, tan_slope, upslope, cell_factor, rhead, ecc, head_dir, seeds_r, seeds_c, horizon, res, beta_s, offsets, dist, bearing):
    ny, nx = fuel.shape
    nh = rhead.shape[0]
    n = ny * nx
    t = np.full(n, np.inf)
    cap = n * 8 + 16
    ht = np.empty(cap)
    hi = np.empty(cap, np.int64)
    size = 0
    for k in range(seeds_r.shape[0]):
        r, c = seeds_r[k], seeds_c[k]
        if 0 <= r < ny and 0 <= c < nx:
            i = r * nx + c
            if t[i] > 0.0:
                t[i] = 0.0
                ht[size] = 0.0
                hi[size] = i
                size += 1
    # binary min-heap built from the seeds (all zero, so already a heap)
    while size > 0:
        tu, u = ht[0], hi[0]
        size -= 1
        if size > 0:                                    # pop: move the last item to the root and sift down
            lt, li = ht[size], hi[size]
            p = 0
            while True:
                ch = 2 * p + 1
                if ch >= size:
                    break
                if ch + 1 < size and ht[ch + 1] < ht[ch]:
                    ch += 1
                if ht[ch] < lt:
                    ht[p], hi[p] = ht[ch], hi[ch]
                    p = ch
                else:
                    break
            ht[p], hi[p] = lt, li
        if tu > t[u] or tu > horizon:
            continue
        r, c = u // nx, u % nx
        h = min(nh - 1, int(tu))
        e = ecc[h]
        hd = head_dir[h]
        sl = tan_slope[r, c]
        up = upslope[r, c]
        for k in range(offsets.shape[0]):
            rr, cc = r + offsets[k, 0], c + offsets[k, 1]
            if rr < 0 or rr >= ny or cc < 0 or cc >= nx:
                continue
            f = fuel[rr, cc]
            rh = rhead[h, f]
            if rh <= 0.0:
                continue
            cf = cell_factor[rr, cc]
            if k >= 8:                                  # a knight move must not jump a barrier: both cells it passes between must let the fire through
                if abs(offsets[k, 0]) == 2:
                    ar, ac, br, bc = r + offsets[k, 0] // 2, c, r + offsets[k, 0] // 2, c + offsets[k, 1]
                else:
                    ar, ac, br, bc = r, c + offsets[k, 1] // 2, r + offsets[k, 0], c + offsets[k, 1] // 2
                if fuel[ar, ac] == 0 or fuel[br, bc] == 0:
                    continue
                cf = min(cf, min(cell_factor[ar, ac], cell_factor[br, bc]))
            v = rh * (1.0 - e) / (1.0 - e * np.cos(bearing[k] - hd))                # elliptical rate along this bearing (from the focus)
            sm = np.exp(beta_s * sl * np.cos(bearing[k] - up))
            v *= min(6.0, max(0.2, sm)) * cf
            if v <= 1e-9:
                continue
            tv = tu + dist[k] * res / v
            j = rr * nx + cc
            if tv < t[j] and tv <= horizon:
                t[j] = tv
                p = size                                # push and sift up
                size += 1
                while p > 0:
                    par = (p - 1) // 2
                    if ht[par] > tv:
                        ht[p], hi[p] = ht[par], hi[par]
                        p = par
                    else:
                        break
                ht[p], hi[p] = tv, j
    return t.reshape(ny, nx)


def arrival_times(fuel, tan_slope, upslope_deg, cell_factor, fuels: FuelTable, wind_ms, wind_from_deg, moisture, seeds_rc, horizon_h, res, beta_s=3.0):
    """Arrival time in hours after the start for every cell (inf = not reached within the horizon). `seeds_rc` are (row, col) cells burning at hour 0."""
    r, ecc = head_rates(fuels, wind_ms, moisture)
    head = np.radians((np.asarray(wind_from_deg, float) + 180.0) % 360.0)
    seeds = np.asarray(seeds_rc, np.int64).reshape(-1, 2)
    return _travel_times(np.ascontiguousarray(fuel, np.int64), np.ascontiguousarray(tan_slope, np.float64), np.ascontiguousarray(np.radians(upslope_deg), np.float64),
                         np.ascontiguousarray(cell_factor, np.float64), np.ascontiguousarray(r, np.float64), np.ascontiguousarray(ecc, np.float64), np.ascontiguousarray(head, np.float64),
                         np.ascontiguousarray(seeds[:, 0]), np.ascontiguousarray(seeds[:, 1]), float(horizon_h), float(res), float(beta_s), OFFSETS, DIST, BEARING)
