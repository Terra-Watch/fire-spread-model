"""Score fires with the gated U-Net (mean of the available seeds) and write a run folder for the dashboard:
events.json, <event>_<H>h.geojson (bands), <event>_burned_<H>h.geojson (what burned, archived fires), <event>_pixels.geojson (start footprint),
<event>_places_<H>h.geojson and <event>_roads_<H>h.geojson (what the bands run over), skill.json is written by hand from the paper's tables.
Replay mode: fires are 2025 test fires of the data set. Live scoring needs the 46 layers built from live feeds (see docs/dashboard-plan.md).
Run: .venv/bin/python scripts/62_outlook_run.py --release RELEASE_DIR --models MODEL_DIR --cases CASES_CSV --out website/backend/runtime_data/fire_spread [--n 10]"""
import argparse, glob, importlib.util, json, sys, datetime as dt
from pathlib import Path
import numpy as np, pandas as pd, torch
from pyproj import Transformer
from rasterio.features import shapes
from shapely.geometry import mapping, shape
from shapely.ops import transform, unary_union
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "src")); sys.argv, argv = sys.argv[:1], sys.argv[1:]
spec = importlib.util.spec_from_file_location("tr", ROOT / "scripts" / "61_train.py"); tr = importlib.util.module_from_spec(spec); spec.loader.exec_module(tr)
from firespread import affected as AF, grid as G, imgmodels
PREVALENCE = {24: 0.0218, 48: 0.0293}      # share of the cells asked about that burned, mean over the data set (docs/dataset.md): the top band holds about as many cells as usually burn at that horizon
BANDS = {h: (p, 3 * p, 7 * p) for h, p in PREVALENCE.items()}      # cumulative shares: top band = 1x, next = up to 3x, last = up to 7x the usual burned share
COMPASS = AF.COMPASS
ap = argparse.ArgumentParser(); ap.add_argument("--release", required=True); ap.add_argument("--models", required=True); ap.add_argument("--cases", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--n", type=int, default=10); ap.add_argument("--since", default="2025-04-08"); ap.add_argument("--places", default=str(ROOT / "data/places.geojsonseq")); ap.add_argument("--roads", default=str(ROOT / "data/osm_lines.geojsonseq")); a = ap.parse_args(argv)
cases = pd.read_csv(a.cases); cases["t"] = pd.to_datetime(cases.t0, utc=True, format="ISO8601")
pick = cases[(cases.study_split == "test") & (cases.t >= a.since) & (cases.n_start >= 9) & (cases.truth_n_24 >= 40) & (cases.truth_n_48 >= 40)].sort_values("n_start", ascending=False)
pick = pick.groupby("state").head(3).head(a.n).sort_values("t"); at = pick.t.max()
run = "replay-gated-" + at.strftime("%Y%m%dT%H%M"); out = Path(a.out) / run; out.mkdir(parents=True, exist_ok=True)
dirs = [f"{a.release}/part_p{i}" for i in range(6)]; fires = {}
def grid_of(r): return G.Grid(int(r.epsg), float(r.grid_x0) + int(r.crop_c0) * float(r.res), float(r.grid_y0) - int(r.crop_r0) * float(r.res), float(r.res), 192, 192)
def box_of(g):
    ll = Transformer.from_crs(g.epsg, 4326, always_xy=True).transform; xs = [g.x0, g.x0 + 192 * g.res]; ys = [g.y0, g.y0 - 192 * g.res]
    pts = [ll(x, y) for x in xs for y in ys]; return (min(p[0] for p in pts), min(p[1] for p in pts), max(p[0] for p in pts), max(p[1] for p in pts))
grids = [grid_of(r) for _, r in pick.iterrows()]; boxes = [box_of(g) for g in grids]
print("reading places and roads ...", flush=True); PL, RD, SRC = [], [], []
for bx in boxes:
    try: p_, r_ = AF.fetch_overpass(bx, ROOT / "data/overpass_cache"); SRC.append("OpenStreetMap, current (Overpass)")
    except Exception as ex: print("Overpass failed, using the local extract:", repr(ex)[:80]); p_ = r_ = None; SRC.append("OpenStreetMap, January 2021 extract")
    PL.append(p_); RD.append(r_)
if any(x is None for x in PL):
    lp, lr = AF.read_places(Path(a.places), boxes), AF.read_roads(Path(a.roads), boxes); PL = [x if x is not None else lp[i] for i, x in enumerate(PL)]; RD = [x if x is not None else lr[i] for i, x in enumerate(RD)]
def polys_to_geojson(mask, g, props):
    if not mask.any(): return None
    to_ll = Transformer.from_crs(g.epsg, 4326, always_xy=True).transform
    ps = [shape(s) for s, v in shapes(mask.astype(np.uint8), mask=mask, transform=g.transform) if v == 1]
    return {"type": "Feature", "geometry": mapping(transform(to_ll, unary_union(ps).simplify(60.0))), "properties": props}
fc = lambda feats: {"type": "FeatureCollection", "features": feats}
for H in (24, 48):
    data = tr.Data(dirs, H); ckpts = [torch.load(p, weights_only=False, map_location="cpu") for p in sorted(glob.glob(f"{a.models}/fsn_{H}_s*.pt"))]; assert ckpts, "no models"
    models = []
    for ck in ckpts:
        m_ = imgmodels.make("firesensenet", len(ck["sp_names"]), len(ck["sc_names"]), pretrained=False); m_.load_state_dict(ck["model"]); m_.eval(); models.append(m_)
    data.sp_names, data.sc_names = ckpts[0]["sp_names"], ckpts[0]["sc_names"]
    for k, (_, r) in enumerate(pick.iterrows()):
        i = int(data.meta.index[data.meta.case == r.case][0]); raw = data.raw(i); sp = data.sp_names; names = {n: j for j, n in enumerate(json.load(open(dirs[0] + "/names.json"))["spatial"])}
        scores = []
        for ck, mod in zip(ckpts, models):
            data.mu, data.sd, data.smu, data.ssd = ck["mu"], ck["sd"], ck["smu"], ck["ssd"]; x, s, y, cand, dist = data.get(i)
            with torch.no_grad(): scores.append(torch.sigmoid(mod(torch.from_numpy(x)[None], torch.from_numpy(s)[None]))[0].numpy())
        score = np.mean(scores, axis=0)
        m = cand > 0; idx = np.flatnonzero(m.ravel()); rank = np.empty(len(idx)); rank[np.argsort(-score.ravel()[idx])] = np.arange(len(idx)) / len(idx)
        band = np.zeros(m.shape, np.int8).ravel(); band[idx] = np.select([rank < BANDS[H][0], rank < BANDS[H][1], rank < BANDS[H][2]], [3, 2, 1], 0); band = band.reshape(m.shape)
        g = grids[k]; ev = f"replay-{r.case.replace('_', '-')}"; seed = raw[0][sp.index("seed")] > 0
        feats = [f for b in (1, 2, 3) if (f := polys_to_geojson(band == b, g, {"band": b, "event": ev, "horizon_h": H, "run": run, "mode": "replay"}))]
        json.dump(fc(feats), open(out / f"{ev}_{H}h.geojson", "w"))
        burned = polys_to_geojson(y > 0, g, {"band": 0, "burned": True, "event": ev, "horizon_h": H}); json.dump(fc([burned] if burned else []), open(out / f"{ev}_burned_{H}h.geojson", "w"))
        rr, cc = np.nonzero(seed); top = band == 3; tr_, tc_ = np.nonzero(top)
        dy, dx = -(tr_.mean() - rr.mean()), tc_.mean() - cc.mean(); deg = (np.degrees(np.arctan2(dx, dy)) + 360) % 360
        cover = {k_: float(raw[0][names[k_]][top | (band == 2)].mean()) for k_ in ("tree", "shrub", "grass", "crop")}
        up = float((raw[0][names["slope_along"]][top] > 0).mean())
        summ = {"bearing_deg": round(float(deg)), "bearing": COMPASS[int(((deg + 22.5) % 360) // 45)], "cover": max(cover, key=cover.get), "cover_share": round(max(cover.values()), 2), "uphill_share": round(up, 2), "slope_deg": round(float(raw[0][names["slope"]][top].mean()), 1)}
        layers = {k_: raw[0][names[k_]] for k_ in ("pop", "tree", "shrub", "grass", "crop", "built", "water")}
        af = AF.affected(band, layers, g, PL[k], RD[k], seed)
        near = [p for p in af["places"] if p["kind"] in ("village", "town")]
        pl = [p for p in af["places"] if p["band"] > 0 or p["distance_km"] <= 3][:60]
        json.dump(fc([{"type": "Feature", "geometry": {"type": "Point", "coordinates": [p["lon"], p["lat"]]}, "properties": {kk: p[kk] for kk in ("name", "kind", "band", "distance_km", "bearing")}} for p in pl]), open(out / f"{ev}_places_{H}h.geojson", "w"))
        rd = [d for d in af["roads"] if d["band"] > 0]
        json.dump(fc([{"type": "Feature", "geometry": {"type": "LineString", "coordinates": d["coords"]}, "properties": {"name": d["name"], "highway": d["highway"], "band": d["band"]}} for d in rd]), open(out / f"{ev}_roads_{H}h.geojson", "w"))
        road_names = sorted({d["name"] for d in rd if d["name"]})[:8]
        f = fires.setdefault(ev, {"event": ev, "case": r.case, "state": r.state, "lat": float(r.lat), "lon": float(r.lon), "n_detections": int(r.n_start), "first_detection": str(r.t), "last_detection": str(r.t),
                                  "frp_sum": round(float(np.expm1(float(data.S[int(data.meta.part.iloc[i])][int(data.meta.row.iloc[i])][json.load(open(dirs[0] + "/names.json"))["scalars"].index("frp_sum_log")]))), 1), "horizons": {}})
        f["horizons"][str(H)] = {"file": f"{ev}_{H}h.geojson", "burned_file": f"{ev}_burned_{H}h.geojson", "places_file": f"{ev}_places_{H}h.geojson", "roads_file": f"{ev}_roads_{H}h.geojson", "cells_scored": int(m.sum()), "weather_source": str(r.weather_source),
                                 "bands": {str(b): int((band == b).sum()) for b in (3, 2, 1)}, "summary": summ, "affected": {**af["summary"], "nearest_settlement": near[0] if near else None, "roads_in_bands": len(rd), "road_names": road_names, "models": len(models)}}
        if H == 24:      # the fire at the start: its footprint as one outline (the cells within 190 m of its detections), not one marker per cell
            foot = polys_to_geojson(seed, g, {"footprint": True, "event": ev, "detections": int(r.n_start)}); json.dump(fc([foot] if foot else []), open(out / f"{ev}_pixels.geojson", "w"))
        print(H, ev, len(models), "seeds", len(pl), "places", len(rd), "roads", flush=True)
json.dump({"run": run, "mode": "replay", "at": str(at), "model_tag": "gated-unet-3seeds", "issued": dt.datetime.now(dt.timezone.utc).isoformat(), "fires": list(fires.values()),
           "note": "A ranking of the cells within 8 km of the fire, not a probability. Wind is not used. Archived fires from 2025, scored as they would have been at the time. Places and roads from OpenStreetMap (fetched when the run was made; the January 2021 extract where the service was busy); population from WorldPop 2019-2020."}, open(out / "events.json", "w"), indent=1)
print(run, len(fires), "fires")
