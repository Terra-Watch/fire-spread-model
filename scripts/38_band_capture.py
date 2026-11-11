"""What each band has delivered on past fires: the share of a fire's burned cells that fell inside the top 2 % (band 3), top 6 % (bands 3+2) and top 15 % (bands 3+2+1) of the cells asked about, on the frozen-test fires,
model fitted on dev only; plain nearest-first for comparison. Writes data/live/band_capture.json for the dashboard's evidence lines.   Run: .venv/bin/python scripts/38_band_capture.py"""
import json
import sys
from importlib import import_module
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from firespread import config, live, scoring  # noqa: E402

N = import_module("21_new_features")
F = N.F


def captured(score, y, share):
    k = max(1, int(round(share * len(y))))
    return float(y[np.argpartition(-score, k - 1)[:k]].sum() / max(1, y.sum()))


def main():
    ant = pd.read_parquet(config.DATA / "antecedent.parquet").set_index("case").join(pd.read_parquet(config.DATA / "soil.parquet").set_index("case"), how="inner")
    dirs = [p.parent for root in ("cases", "bank") for p in sorted((config.DATA / root).glob("*/burn_long.npz"))]
    rng = np.random.default_rng(0)
    out = {"bands": {"3": live.BANDS[0], "2": live.BANDS[1], "1": live.BANDS[2]}, "horizons": {}}
    for H in (24, 48):
        items = [x for x in (N.prepare(d, H, ant) for d in dirs) if x]
        dev, test = [x for x in items if x["split"] == "dev"], [x for x in items if x["split"] == "test"]
        names = [k for k in dev[0]["f"] if k not in F.GROUPS["wind"]]
        m = N.fit(dev, names, rng)
        res = {}
        for tag in ("prox", "model"):
            rows = []
            for it in test:
                s = -it["d"] if tag == "prox" else m.predict_proba(np.column_stack([it["f"][k] for k in names]))[:, 1]
                rows.append({"block": it["block"], **{f"c{int(100 * sh)}": captured(s, it["y"], sh) for sh in live.BANDS}})
            df = pd.DataFrame(rows)
            res[tag] = {col: scoring.bootstrap_mean(df.groupby("block")[col].mean()) for col in df.columns if col != "block"}
            print(H, "h", tag, {k: [round(x, 2) for x in v] for k, v in res[tag].items()}, flush=True)
        out["horizons"][str(H)] = {"fires": len(test), "model": res["model"], "nearest_first": res["prox"]}
    json.dump(out, open(config.DATA / "live" / "band_capture.json", "w"), indent=1)


if __name__ == "__main__":
    main()
