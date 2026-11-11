"""Choose the pilot cases and freeze the split by area before any model sees them.
Eligible: at least 3 detections in the start, at least 3 new, 3 to 30 h between the two observations, event no wider than 40 km.
Sample: states in proportion to the square root of their eligible cases, at most 2 cases per event, seed 42.
Split: each 1 x 1 degree block goes wholly to dev (70 %) or test (30 %) by a hash of its name, so neighbouring fires never straddle the split.
Run: .venv/bin/python scripts/03_pilot.py [n_cases]"""
import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from firespread import config  # noqa: E402


def block_split(lat, lon):
    name = f"{int(np.floor(lat))}_{int(np.floor(lon))}"
    return name, ("test" if int(hashlib.sha1(name.encode()).hexdigest(), 16) % 100 < 30 else "dev")


def main(n):
    c = pd.read_parquet(config.DATA / "cases.parquet")
    e = pd.read_parquet(config.DATA / "events.parquet")[["event", "extent_km", "n_epochs"]]
    c = c.merge(e, on="event")
    c = c[(c.n_start >= 3) & (c.n_new >= 3) & (c.gap_h >= 3) & (c.gap_h <= 30) & (c.extent_km <= 40) & (c.state != "")].copy()
    c["block"], c["split"] = zip(*[block_split(a, b) for a, b in zip(c.lat, c.lon)])
    c["year"] = c.t0.dt.year
    w = np.sqrt(c.state.value_counts())
    quota = (w / w.sum() * n).round().astype(int).clip(lower=1)
    rng = np.random.default_rng(42)
    picks = []
    for st, k in quota.items():
        g = c[c.state == st].sample(frac=1.0, random_state=int(rng.integers(1 << 31)))
        seen, taken = {}, []
        for i, r in g.iterrows():
            if seen.get(r.event, 0) < 2:
                taken.append(i)
                seen[r.event] = seen.get(r.event, 0) + 1
            if len(taken) >= k:
                break
        picks += taken
    p = c.loc[picks].copy()
    p["case_id"] = p.event + "_" + p.epoch_from.astype(str)
    p = p.sort_values(["split", "state", "t0"]).reset_index(drop=True)
    p.to_parquet(config.DATA / "pilot_cases.parquet", index=False)
    print(len(p), "cases;", p.event.nunique(), "events;", p.split.value_counts().to_dict(), "| eligible pool", len(c))
    print(p.state.value_counts().to_dict())
    print(p.year.value_counts().sort_index().to_dict(), "gap_h:", p.gap_h.describe().round(1)[["min", "50%", "max"]].to_dict())


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 300)
