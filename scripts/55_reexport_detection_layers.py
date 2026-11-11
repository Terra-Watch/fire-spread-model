"""v2 of a dataset part: the detection-derived inputs rebuilt from CLEAN detections (docs/false-detections-plan.md, D1 to D3). v1 is left untouched.
Clean = detection score p_scar >= hide (classifier v2 of wildfire-dataset, threshold fixed on 2019 to 2023) or not scored (outside the forecasting study cells, about 8%: kept, never silently dropped, counted per case).
Replaced layers: fire_hist_90, fire_hist_365, fire_days_since, other_fires_72h. Replaced numbers: act5, act20, act_ev20. All past-only, as before.
NOT changed: the fire's own start footprint (seed, det_frp, det_age_h, n_start, frp_sum_log, extent_km) and the labels; instead each case gets p_real_start (share of its start detections that pass), n_start_pass, n_start_unscored and admitted_v2 (D1: at least 3 passing start detections; D2: not more than 30% of the start detections flagged).
Steps:  --build-lookup   one-time table data/detection_scores_lookup.parquet (key = lat, lon to 4 decimals + day; max score where several share a key) from the v4 audit
        PART_DIR [--check]   writes PART_DIR_v2; --check rebuilds with ALL detections and compares with the stored layers (proves the grid and the code are the same as the export's) and writes nothing.
Run: .venv/bin/python scripts/55_reexport_detection_layers.py data/dataset/part_p0"""
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "src"))
from firespread import activity, config, dataset, grid as G  # noqa: E402
V4 = ROOT.parent / "wildfire-dataset"; LOOKUP = ROOT / "data" / "detection_scores_lookup.parquet"; CROP = 192; PAD = 8      # padded grid: detections just outside the crop reach its edge cells (stamp_counts clips them onto the edge of a grid, so the crop alone would be wrong)
REPL = ["fire_hist_90", "fire_hist_365", "fire_days_since", "other_fires_72h"]; REPL_S = ["act5", "act20", "act_ev20"]
key = lambda lat, lon, t: pd.Series(np.round(lat, 4)).astype(str).values + "_" + pd.Series(np.round(lon, 4)).astype(str).values + "_" + pd.Series(pd.to_datetime(t)).dt.strftime("%Y%m%d").values


def build_lookup():
    cols = ["latitude", "longitude", "date_utc"]
    d = pd.concat([pd.read_parquet(V4 / "dataset" / "detections_forest.parquet", columns=cols), pd.read_parquet(V4 / "dataset" / "detections_new.parquet", columns=cols)], ignore_index=True)
    d["p"] = pd.read_parquet(V4 / "audit" / "false_detections" / "detection_scores_all.parquet").sort_values("rid").p_scar.values
    d["k"] = key(d.latitude.values, d.longitude.values, d.date_utc)
    out = d.groupby("k").p.max().reset_index(); out.to_parquet(LOOKUP); print("lookup", len(out), "keys")


def year_dets(year, look, hide):
    fr = []
    for y in (year - 1, year, year + 1):
        f = config.DATA / "detections" / f"{y}.parquet"
        if f.exists():
            fr.append(pd.read_parquet(f, columns=["latitude", "longitude", "t", "frp", "event", "epoch"]))
    d = pd.concat(fr, ignore_index=True); d["p"] = look.reindex(key(d.latitude.values, d.longitude.values, d.t)).values
    d["clean"] = d.p.isna() | (d.p >= hide); return d


def main(part, check):
    part = Path(part); names = json.load(open(part / "names.json")); meta = pd.read_parquet(part / "meta.parquet")
    hide = json.load(open(V4 / "audit" / "false_detections" / "classifier_report_v2.json"))["thresholds_from_2019_2023"]["hide_keeps_98pct_of_confirmed"]
    look = pd.read_parquet(LOOKUP).set_index("k").p
    X = np.load(part / "spatial.npy", mmap_mode="r"); S = {H: np.load(part / f"scalars_{H}.npy") for H in (24, 48)}
    sp = {k: names["spatial"].index(k) for k in REPL}; sc = {k: names["scalars"].index(k) for k in REPL_S}
    new_X = np.array(X) if not check else None; new_S = {H: S[H].copy() for H in S}
    extra = []; dets = {}; worst = 0.0
    for i, m in enumerate(meta.itertuples()):
        year = int(m.year)
        if year not in dets:
            dets = {year: year_dets(year, look, hide)}
        det = dets[year]; use = det if check else det[det.clean]
        ef = int(m.case.rsplit("_", 1)[1]); fire = det[(det.event == m.event) & (det.epoch <= ef)]
        g = G.Grid(int(m.epsg), float(m.grid_x0) + (int(m.crop_c0) - PAD) * float(m.res), float(m.grid_y0) - (int(m.crop_r0) - PAD) * float(m.res), float(m.res), CROP + 2 * PAD, CROP + 2 * PAD)
        mm = {"t0": m.t0, "event": m.event}
        L = {k: v[PAD:PAD + CROP, PAD:PAD + CROP] for k, v in dataset.history_layers(g, use, mm, fire).items()}
        t0 = pd.Timestamp(m.t0); recent = use[(use.t >= t0 - pd.Timedelta(hours=72)) & (use.t < t0)][["latitude", "longitude", "t", "event"]]
        act = activity.counts(float(m.lat), float(m.lon), t0, m.event, recent=recent)
        if check:
            for k in REPL:
                a, b = L[k].astype(np.float16).astype(np.float32), np.asarray(X[i, sp[k]], np.float32); dd = float(np.nanmax(np.abs(a - b))) if np.isfinite(a).any() else 0.0; print('  ', m.case, k, 'max diff', round(dd, 4), 'range', round(float(np.nanmax(b)), 2), flush=True); worst = max(worst, dd)
            for H in S:
                for k in REPL_S:
                    dd = abs(act[k] - float(S[H][i, sc[k]])); print('  ', m.case, H, k, 'diff', round(dd, 4), flush=True) if dd > 0.01 else None; worst = max(worst, dd)
            continue
        for k in REPL:
            new_X[i, sp[k]] = L[k].astype(np.float16)
        for H in new_S:
            for k in REPL_S:
                new_S[H][i, sc[k]] = act[k]
        pr = fire.p; scored = pr.notna(); n_pass = int((fire.clean).sum()); flagged = int((~fire.clean).sum())
        extra.append({"p_real_start": float((pr[scored] >= hide).mean()) if scored.any() else np.nan, "n_start_pass": n_pass, "n_start_flagged": flagged, "n_start_unscored": int((~scored).sum()),
                      "admitted_v2": bool(n_pass >= 3 and flagged <= 0.30 * len(fire))})
    if check:
        print(f"check: largest difference between the rebuild with all detections and the stored part = {worst:.5f} over {len(meta)} cases"); assert worst < 0.05, "grid or code differs from the export"; return
    out = part.parent / (part.name + "_v2"); out.mkdir(exist_ok=True)
    for f in part.iterdir():
        if f.name not in ("spatial.npy", "meta.parquet", "scalars_24.npy", "scalars_48.npy", "qa.json", "audit.json"):
            shutil.copy(f, out / f.name)
    np.save(out / "spatial.npy", new_X)
    for H in new_S:
        np.save(out / f"scalars_{H}.npy", new_S[H])
    pd.concat([meta.reset_index(drop=True), pd.DataFrame(extra)], axis=1).to_parquet(out / "meta.parquet")
    json.dump({"detection_version": "classifier_v2", "hide": hide, "replaced_layers": REPL, "replaced_numbers": REPL_S, "unscored_kept": True, "source_part": part.name}, open(out / "v2.json", "w"), indent=1)
    e = pd.DataFrame(extra); print("wrote", out, "| admitted_v2", int(e.admitted_v2.sum()), "of", len(e), "| unscored share of start detections", round(float(e.n_start_unscored.sum() / max((e.n_start_pass + e.n_start_flagged).sum(), 1)), 3))


if __name__ == "__main__":
    if sys.argv[1] == "--build-lookup":
        build_lookup()
    else:
        main(sys.argv[1], "--check" in sys.argv)
