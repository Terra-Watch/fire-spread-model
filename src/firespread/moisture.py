"""Dead fine-fuel moisture from hourly air temperature, humidity and rain: the equilibrium moisture of Simard (1968), approached with a time lag (about 10 h for leaf litter),
pushed up while it rains. A first model: no measurements of Indian litter exist here to fit it on."""
import numpy as np


def equilibrium(temp_c, rh_pct):
    """Equilibrium moisture content (fraction of dry weight) of dead fuel."""
    t = np.asarray(temp_c, float) * 9.0 / 5.0 + 32.0
    h = np.clip(np.asarray(rh_pct, float), 1.0, 100.0)
    low = 0.03229 + 0.281073 * h - 0.000578 * h * t
    mid = 2.22749 + 0.160107 * h - 0.01478 * t
    high = 21.0606 + 0.005565 * h ** 2 - 0.00035 * h * t - 0.483199 * h
    return np.clip(np.where(h < 10, low, np.where(h <= 50, mid, high)) / 100.0, 0.02, 0.35)


def dead_fuel(temp_c, rh_pct, precip_mm, lag_h: float = 10.0, start: float | None = None):
    """Hourly moisture series. Each hour moves toward the equilibrium by 1 - exp(-1/lag); rain of 0.5 mm/h or more raises it toward 0.35 faster (lag 2 h)."""
    emc = equilibrium(temp_c, rh_pct)
    rain = np.nan_to_num(np.asarray(precip_mm, float))
    m = np.empty(len(emc))
    cur = float(emc[0] if start is None else start)
    for i in range(len(emc)):
        if rain[i] >= 0.5:
            cur += (0.35 - cur) * (1.0 - np.exp(-1.0 / 2.0))
        else:
            cur += (emc[i] - cur) * (1.0 - np.exp(-1.0 / lag_h))
        m[i] = cur
    return m
