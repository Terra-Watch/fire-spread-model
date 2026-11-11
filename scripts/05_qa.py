"""Check the case bundles and write data/QA.md: how complete they are and what is missing. Run: .venv/bin/python scripts/05_qa.py"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from firespread import config  # noqa: E402


def main():
    rows = [json.load(open(f)) for f in sorted((config.DATA / "cases").glob("*/meta.json"))]
    m = pd.DataFrame(rows)
    for k in ("hist", "prev1", "era5"):
        m[f"w_{k}"] = m.weather_rows.map(lambda d: d.get(k, 0))
    flags = {
        "start not fully inside the grid": m[m.start_inside_grid < 1.0].case_id.tolist(),
        "less than 90 % of the grid has elevation": m[m.dem_valid < 0.9].case_id.tolist(),
        "no archived-forecast weather (hist)": m[m.w_hist < 24].case_id.tolist(),
        "no ERA5 weather": m[m.w_era5 < 24].case_id.tolist(),
        "more than half cropland around the fire": m[m.crop_mean > 0.5].case_id.tolist(),
        "less than 20 % forest land around the fire": m[m.forest_share_mean < 0.2].case_id.tolist(),
    }
    q = lambda s: f"{s.quantile(.1):.1f} / {s.median():.1f} / {s.quantile(.9):.1f}"
    out = ["# Data QA (pilot cases)", "", f"Cases with a complete bundle: **{len(m)}** (dev {int((m.split == 'dev').sum())}, test {int((m.split == 'test').sum())}); events {m.event.nunique()}; states {m.state.nunique()}.", "",
           "| Measure | 10 % / median / 90 % |", "|---|---|",
           f"| detections in the start footprint | {q(m.n_start)} |", f"| new-growth detections | {q(m.n_new)} |", f"| hours between the two observations | {q(m.gap_h)} |",
           f"| 95th percentile slope in the grid (degrees) | {q(m.slope_p95_deg)} |", f"| forest-land share of the grid | {q(m.forest_share_mean)} |", f"| tree-cover share (WorldCover) | {q(m.tree_mean)} |",
           f"| crop share (WorldCover) | {q(m.crop_mean)} |", "", "## Weather rows with wind present, by year and source (a case needs about 100 to 150 hours)", "",
           m.groupby("year")[["w_hist", "w_prev1", "w_era5"]].apply(lambda g: (g > 24).mean().round(2)).rename(columns={"w_hist": "hist", "w_prev1": "prev1 (day-1 forecast)", "w_era5": "era5"}).to_markdown(), "",
           "## Flags", ""]
    for k, v in flags.items():
        out.append(f"- {k}: **{len(v)}**" + (f" ({', '.join(v[:6])}{' ...' if len(v) > 6 else ''})" if v else ""))
    failed = (config.DATA / "cases" / "_failed.txt")
    out += ["", f"Failed to build: {len([x for x in failed.read_text().splitlines() if x]) if failed.exists() else 'unknown'}"]
    (config.DATA / "QA.md").write_text("\n".join(out) + "\n")
    print("\n".join(out))


if __name__ == "__main__":
    main()
