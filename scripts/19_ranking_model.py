"""Final ranking model (learned, without wind) per horizon. (1) Weather-error check: fit on dev, score the frozen-test cases of 2024-25 twice, once with the archived forecast stream and once with the day-before forecast
(what the dashboard will really have). (2) Fit on all cases and save data/models/rank_<H>h.joblib with its feature list.
Run: .venv/bin/python scripts/19_ranking_model.py"""
import dataclasses
import sys
from importlib import import_module
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from firespread import burn_eval, cases, config  # noqa: E402

F = import_module("14_final")
feat = import_module("11_learned")


def prepare(d, H, source="hist"):
    try:
        c = cases.load_case(d, source=source, gap_override=H)
    except cases.NoWeather:
        return None
    c = dataclasses.replace(c, meta={**c.meta, "t1": str(pd.Timestamp(c.meta["t0"]) + pd.Timedelta(hours=H))})
    b = np.load(d / "burn_long.npz")
    truth, cand, ok = burn_eval.burn_masks(c, b["doy"], int(b["year"]))
    if not ok:
        return None
    truth, cand = burn_eval.connected_truth(c, truth, cand)
    if truth.sum() < 5:
        return None
    return c, truth, cand, {k: v.astype(np.float32) for k, v in feat.features(c).items()}


def main():
    dirs = [p.parent for root in ("cases", "bank") for p in sorted((config.DATA / root).glob("*/burn_long.npz"))]
    rng = np.random.default_rng(0)
    (config.DATA / "models").mkdir(exist_ok=True)
    for H in (24, 48):
        items = [x for x in (prepare(d, H) for d in dirs) if x]
        names = [k for k in items[0][3] if k not in F.GROUPS["wind"]]
        dev = [x for x in items if x[0].meta["split"] == "dev"]
        recent = [d for d in dirs if d.name[:4] in ("2024", "2025") and json_split(d) == "test"]
        m = F.fit(dev, names, rng)
        prox = lambda it: cases.baseline_proximity(it[0])
        score = lambda it: m.predict_proba(np.column_stack([it[3][k].ravel() for k in names]))[:, 1].reshape(it[0].fuel.shape)
        rows = []
        for tag, src in (("hist", "hist"), ("prev1", "prev1")):
            sub = [x for x in (prepare(d, H, src) for d in recent) if x]
            used = sum(x[0].weather_source == "prev1" for x in sub)
            print(f"H={H} {tag}: {len(sub)} cases, {used} really on the day-before forecast", flush=True)
            if tag == "hist":
                rows += F.rows_for(sub, prox, "prox")
            rows += F.rows_for(sub, score, f"learned_{tag}")
        F.summarize(pd.DataFrame(rows), f"WEATHER SOURCE, 2024-25 test cases, {H} h")
        joblib.dump({"model": F.fit(items, names, rng), "features": names, "horizon_h": H}, config.DATA / "models" / f"rank_{H}h.joblib")
        print(f"saved rank_{H}h ({len(items)} cases)", flush=True)


def json_split(d):
    import json
    return json.load(open(d / "meta.json"))["split"]


if __name__ == "__main__":
    main()
