"""Score the likely next areas of every active fire at a moment in time and write the files the dashboard serves.
Replay (--at): fires are the forest detections of the 3 days before that moment, weather from the archived forecast stream; the files say "replay". Live (--live): the last 3 days of FIRMS near-real-time detections and the weather forecast; the files say "live".
Output data/live/<run_id>/events.json plus <event>_<H>h.geojson (bands) and <event>_pixels.geojson (the fire's detections).
Run: .venv/bin/python scripts/34_score_active.py (--at 2025-03-12T12:00 | --live) --model-tag rank3 --lines data/osm_lines.geojsonseq [--max-fires 8]"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from firespread import ancillary, config, events, grid as G, live  # noqa: E402


def firms_key() -> str:
    """The FIRMS key from the environment or the website's .env; never printed."""
    import os
    key = os.environ.get("FIRMS_MAP_KEY", "").strip()
    env = Path(__file__).resolve().parents[2] / "website" / "backend" / ".env"
    if not key and env.exists():
        key = next((ln.split("=", 1)[1].strip().strip("\"'") for ln in env.read_text().splitlines() if ln.startswith("FIRMS_MAP_KEY=")), "")
    if not key:
        raise SystemExit("FIRMS_MAP_KEY is not set")
    return key


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--at", help="replay: the moment to score (UTC); leave out with --live")
    ap.add_argument("--live", action="store_true", help="fetch the last 3 days of VIIRS near-real-time detections (FIRMS_MAP_KEY in the environment or website/backend/.env) and score with the live weather forecast")
    ap.add_argument("--model-tag", default="rank3")
    ap.add_argument("--lines", required=True)
    ap.add_argument("--max-fires", type=int, default=8)
    ap.add_argument("--recent-h", type=float, default=36.0)
    a = ap.parse_args()
    if a.live:
        det = live.fetch_firms_nrt(firms_key())
        now = det["t"].max()
        wx_kind, mode = "fcst", "live"
        fires, det = live.active_fires(det, now, a.recent_h)
        size = pd.Series({e: len(g) for e, g in fires.items()}).sort_values(ascending=False)
        run = f"live-{now:%Y%m%dT%H%M}"
    else:
        now = pd.Timestamp(a.at, tz="UTC")
        wx_kind, mode = "hist", "replay"
        det = pd.read_parquet(config.DATA / "detections" / f"{now.year}.parquet")
        det = det[(det["t"] > now - pd.Timedelta(hours=72)) & (det["t"] <= now)][["latitude", "longitude", "t", "frp"]].reset_index(drop=True)
        det["event"] = [f"{now:%Y%m%d%H}-{k}" for k in events.link_events(det)]
        last = det.groupby("event")["t"].max()
        live_ev = last[last >= now - pd.Timedelta(hours=a.recent_h)].index
        fires = {e: g.reset_index(drop=True) for e, g in det[det.event.isin(live_ev)].groupby("event")}
        size = det[det.event.isin(live_ev)].groupby("event").size().sort_values(ascending=False)
        run = f"replay-{now:%Y%m%dT%H%M}"
    out = config.DATA / "live" / run
    out.mkdir(parents=True, exist_ok=True)
    print(f"{len(size)} active fires at {now}; scoring the {min(len(size), a.max_fires)} largest", flush=True)
    summary = []
    for ev in size.index[:a.max_fires]:
        fire = fires[ev]
        t0 = fire["t"].mean()
        g = G.make_grid(float(fire.latitude.mean()), float(fire.longitude.mean()))
        lines = ancillary.load_lines(a.lines, ancillary.tiles_of(g))
        entry = {"event": ev, "lat": float(fire.latitude.mean()), "lon": float(fire.longitude.mean()), "n_detections": int(len(fire)), "first_detection": str(fire.t.min()), "last_detection": str(fire.t.max()),
                 "frp_sum": float(fire.frp.sum()), "horizons": {}}
        pix = {"type": "FeatureCollection", "features": [{"type": "Feature", "geometry": {"type": "Point", "coordinates": [float(r.longitude), float(r.latitude)]}, "properties": {"t": str(r.t), "frp": float(r.frp)}} for r in fire.itertuples()]}
        json.dump(pix, open(out / f"{ev}_pixels.geojson", "w"))
        for H in (24, 48):
            work = config.DATA / "live_work" / f"{run}-{ev}-{H}h"
            if not (work / "ant.json").exists():
                live.make_bundle(fire, t0, H, work, lines, weather_kind=wx_kind)
            recent = det[(det.t >= t0 - pd.Timedelta(hours=72)) & (det.t < t0)][["latitude", "longitude", "t", "event"]]
            res = live.score_bundle(work, config.DATA / "models" / f"{a.model_tag}_{H}h.joblib", recent=recent, weather_kind=wx_kind)
            gj = live.band_geojson(res, g, {"event": ev, "horizon_h": H, "run": run, "mode": mode})
            json.dump(gj, open(out / f"{ev}_{H}h.geojson", "w"))
            entry["horizons"][str(H)] = {"file": f"{ev}_{H}h.geojson", "cells_scored": res["n_cells"], "weather_source": res["weather_source"], "bands": {str(b): int((res["band"] == b).sum()) for b in (3, 2, 1)}}
        summary.append(entry)
        print("scored", ev, flush=True)
    json.dump({"run": run, "mode": mode, "at": str(now), "model_tag": a.model_tag, "issued": str(pd.Timestamp.now(tz="UTC")), "fires": summary,
               "note": ("Archived replay with archived weather. " if mode == "replay" else "Live: satellite detections up to the time shown and the weather forecast. ") + "Ranking of cells within 8 km of the fire, not a calibrated probability, not wind-driven."}, open(out / "events.json", "w"), indent=1)
    print("wrote", out, flush=True)


if __name__ == "__main__":
    main()
