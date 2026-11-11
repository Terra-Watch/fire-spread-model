"""Merges the audits of all exported parts into one data card and one audit report: counts by state, year and split, completeness of every layer, label statistics, agreement of the two truths, dev-versus-test comparison,
spatial leakage between dev and test fires. Writes DATASET.md (the data card), data/dataset/meta_all.parquet and data/dataset/audit_summary.json, plus figures in data/dataset/figures/.
Run: .venv/bin/python scripts/57_merge_audit.py PART_DIR [PART_DIR ...]      (each directory holds meta.parquet, audit.json, names.json)"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from firespread import config, dataset, events  # noqa: E402


def main(parts):
    parts = [Path(p) for p in parts]
    metas = [pd.read_parquet(p / "meta.parquet").assign(part=p.name) for p in parts]
    audits = [json.load(open(p / "audit.json")) for p in parts]
    meta = pd.concat(metas, ignore_index=True)
    out = config.DATA / "dataset"; (out / "figures").mkdir(parents=True, exist_ok=True)
    # spatial leakage: how close are dev fires to test fires?
    x, y = events.sinusoidal_km(meta.lat.values, meta.lon.values)
    test = meta.split.values == "test"
    dist = np.full(len(meta), np.nan)
    if test.any():
        dist[~test] = cKDTree(np.column_stack([x[test], y[test]])).query(np.column_stack([x[~test], y[~test]]))[0]
    meta["km_to_nearest_test"] = dist; meta["near_test_25km"] = dist < 25
    meta.to_parquet(out / "meta_all.parquet")
    n = len(meta)
    # layer completeness, weighted by the cases of each part
    layers = {}
    for a in audits:
        for k, row in a["layers"].items():
            for sp, v in row.items():
                d = layers.setdefault(k, {}).setdefault(sp, {"w": 0, "nan": 0.0, "mean": 0.0, "n_mean": 0})
                w = a["by_split"].get(sp, 0)
                d["w"] += w; d["nan"] += v["nan_share"] * w
                if v["mean"] is not None:
                    d["mean"] += v["mean"] * w; d["n_mean"] += w
    layer_tbl = pd.DataFrame({k: {f"{sp}_nan_share": (v[sp]["nan"] / v[sp]["w"] if v[sp]["w"] else np.nan) for sp in v} | {f"{sp}_mean": (v[sp]["mean"] / v[sp]["n_mean"] if v[sp]["n_mean"] else np.nan) for sp in v} for k, v in layers.items()}).T
    labels = {H: {k: float(np.mean([a["labels"][str(H)][k] for a in audits])) for k in audits[0]["labels"][str(H)] if not isinstance(audits[0]["labels"][str(H)][k], dict)} for H in (24, 48)}
    summary = {"cases": n, "parts": [p.name for p in parts], "t0_range": [str(meta.t0.min()), str(meta.t0.max())], "by_split": meta.split.value_counts().to_dict(), "usable": {H: int(meta[f"usable_{H}"].sum()) for H in (24, 48)},
               "usable_by_split": {H: meta[meta[f"usable_{H}"]].split.value_counts().to_dict() for H in (24, 48)}, "states": int(meta.state.nunique()), "blocks": int(meta.block.nunique()),
               "dev_within_25km_of_a_test_fire": float(np.nanmean(meta.near_test_25km[~test])) if test.any() else None, "layers_over_10pct_missing": {k: round(float(max(r.get("dev_nan_share", 0), r.get("test_nan_share", 0))), 3) for k, r in layer_tbl.iterrows() if max(r.get("dev_nan_share", 0), r.get("test_nan_share", 0)) > 0.10},
               "flags": {k: float(meta[k].mean()) for k in ("has_dem", "has_ndvi", "has_s2", "has_burnhist", "has_osm")}, "weather_source": meta.weather_source.value_counts().to_dict(), "labels": labels}
    json.dump(summary, open(out / "audit_summary.json", "w"), indent=1, default=str)
    layer_tbl.to_csv(out / "layer_completeness.csv")
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))
        ax[0].scatter(meta.lon, meta.lat, s=4, c=np.where(meta.split == "test", "tab:red", "tab:blue"), alpha=.5); ax[0].set_title("fires: dev (blue), test (red)"); ax[0].set_aspect("equal")
        meta.groupby(["year", "split"]).size().unstack().plot.bar(ax=ax[1], stacked=True, color=["tab:blue", "tab:red"]); ax[1].set_title("fires per year")
        meta.state.value_counts().head(15).sort_values().plot.barh(ax=ax[2], color="steelblue"); ax[2].set_title("fires per state (top 15)")
        plt.tight_layout(); plt.savefig(out / "figures" / "coverage.png", dpi=110); plt.close()
    except Exception as e:                                     # figures are a convenience
        print("figures skipped:", e)
    # the data card
    names = json.load(open(parts[0] / "names.json"))
    L = [f"# Fire-spread dataset: data card\n", f"Cases: **{n}** fires from {summary['t0_range'][0][:10]} to {summary['t0_range'][1][:10]} (start of the forecast), {summary['states']} states, {summary['blocks']} one-degree blocks. Splits are by 1 x 1 degree block "
         f"(dev {summary['by_split'].get('dev', 0)}, test {summary['by_split'].get('test', 0)}); five folds by block hash for tuning. Usable for scoring (at least 5 burned cells and 50 other cells): "
         f"24 h {summary['usable'][24]}, 48 h {summary['usable'][48]}.\n",
         "## What one case is\nTwo consecutive satellite looks at one forest fire; the forecast starts at the first. Each case has 48 picture layers (192 x 192 cells of 100 m, centred on the fire), 38 numbers, and labels for 24 h and 48 h.\n",
         "## Labels\n* `y_modis`: cells that burned within the horizon according to the MODIS burned-area product (500 m, dated to the day), only the patches touching this fire, outside the 5-cell start zone. The main target.\n* `y_modis_all`: all burned cells in the square.\n"
         "* `y_viirs_new`: cells within 190 m of a later VIIRS detection of this fire that is more than 0.75 km from everything seen up to the start; `y_viirs_any`: any later detection (includes the fire's own persistence).\n* `burn_rel_day`: MODIS burn day relative to the start day (0-5), -1 = none.\n"
         "* `cand` marks the cells asked about (5 to 80 cells from the start, MODIS-mapped, not burned earlier), `modis_valid` the cells MODIS mapped.\n",
         "## Layers and numbers\n| name | unit | source | resolution | how |\n|---|---|---|---|---|\n"] + [f"| {k} | {dataset.DOC[k][0]} | {dataset.DOC[k][1]} | {dataset.DOC[k][2]} | {dataset.DOC[k][3]} |" for k in names["spatial"] + names["scalars"]] + \
        [f"\n## Audit results\n* Missing data: layers with more than 10 % missing: {summary['layers_over_10pct_missing'] or 'none'}.\n* Share of cases with each source: {', '.join(f'{k} {v:.0%}' for k, v in summary['flags'].items())}.\n* Weather source used: {summary['weather_source']}.\n"
         f"* Spatial leakage: {summary['dev_within_25km_of_a_test_fire']:.1%} of dev fires lie within 25 km of a test fire (flag `near_test_25km` in `meta_all.parquet`; drop them for a clean variant).\n" if summary["dev_within_25km_of_a_test_fire"] is not None else "",
         "* Label statistics (mean over parts): " + "; ".join(f"{H} h: MODIS burned share of asked cells {labels[H]['modis_share_of_asked']:.2%}, VIIRS new {labels[H]['viirs_new_share_of_asked']:.2%}, share of VIIRS new cells inside MODIS-burned {labels[H]['viirs_new_inside_modis_burned_share']:.0%}" for H in (24, 48)) + ".\n",
         "## Known limits\n* The MODIS burned-area truth is 500 m and misses small burns; the VIIRS truth is 375 m and misses cloudy passes. The two agree only partly (see above).\n* Weather in the horizon comes from the archived forecast stream (a forecast, not observations).\n* Land cover is 2021, roads 2021, population 2019/2020, regardless of the fire year.\n"
         "* Canopy height is not included: the only open global map we could reach has no overviews and row-strip tiling, about ten minutes per case.\n* Fires larger than 40 km, with fewer than 3 detections at the start, or outside forest land are not in the set.\n* Fire seasons differ a lot between years (2021 has about 7 times the March detections of 2020).\n"]
    open(Path(__file__).resolve().parents[1] / "DATASET.md", "w").write("\n".join(L))
    print("wrote DATASET.md; cases", n, "| leakage", summary["dev_within_25km_of_a_test_fire"], "| layers >10% missing:", summary["layers_over_10pct_missing"])


if __name__ == "__main__":
    main(sys.argv[1:])
