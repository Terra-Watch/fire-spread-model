"""Score fires with the gated U-Net and write a run folder for the dashboard. One outlook per fire ("the next one to two days"): the 24 h and the 48 h models
rank the cells almost identically and their difference carries no timing information (docs/results.md, "One outlook, not two horizons"), so their rankings are averaged
(mean of the seeds of each, then the mean of the two ranks). Files, all under the horizon key 48: events.json, <event>_48h.geojson (bands), <event>_burned_48h.geojson
(what burned within 48 h, archived fires), <event>_pixels.geojson (start footprint), <event>_places_48h.geojson and <event>_roads_48h.geojson (what the bands run over),
<event>_overlay.json (the ranking as a heat picture and a hillshade picture with their corner coordinates). skill.json is written from the measured numbers.
Replay mode: fires are 2025 test fires of the data set. Live scoring needs the 46 layers built from live feeds (see docs/dashboard-plan.md).
Run: .venv/bin/python scripts/62_outlook_run.py --release RELEASE_DIR --models MODEL_DIR --cases CASES_CSV --out website/backend/runtime_data/fire_spread [--n 10]"""
import argparse, base64, glob, importlib.util, io, json, sys, datetime as dt
from pathlib import Path
import numpy as np, pandas as pd, torch
from pyproj import Transformer
from PIL import Image
from rasterio.features import shapes
from scipy.ndimage import gaussian_filter
from scipy.stats import rankdata
from shapely.geometry import mapping, shape
from shapely.ops import transform, unary_union
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "src")); sys.argv, argv = sys.argv[:1], sys.argv[1:]
spec = importlib.util.spec_from_file_location("tr", ROOT / "scripts" / "61_train.py"); tr = importlib.util.module_from_spec(spec); spec.loader.exec_module(tr)
from firespread import affected as AF, grid as G, imgmodels
PREVALENCE = {24: 0.0218, 48: 0.0293}      # share of the cells asked about that burned, mean over the data set (docs/dataset.md): the top band holds about as many cells as usually burn at that horizon
BANDS = {h: (p, 3 * p, 7 * p) for h, p in PREVALENCE.items()}      # cumulative shares: top band = 1x, next = up to 3x, last = up to 7x the usual burned share
COMPASS = AF.COMPASS
ap = argparse.ArgumentParser(); ap.add_argument("--release", required=True); ap.add_argument("--models", required=True); ap.add_argument("--cases", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--n", type=int, default=10); ap.add_argument("--only", default="", help="comma-separated case ids to score instead of the default pick"); ap.add_argument("--since", default="2025-04-08"); ap.add_argument("--places", default=str(ROOT / "data/places.geojsonseq")); ap.add_argument("--roads", default=str(ROOT / "data/osm_lines.geojsonseq")); a = ap.parse_args(argv)
cases = pd.read_csv(a.cases); cases["t"] = pd.to_datetime(cases.t0, utc=True, format="ISO8601")
pick = cases[(cases.study_split == "test") & (cases.t >= a.since) & (cases.n_start >= 9) & (cases.truth_n_24 >= 40) & (cases.truth_n_48 >= 40)].sort_values("n_start", ascending=False)
pick = pick.groupby("state").head(3).head(a.n).sort_values("t")
if a.only: pick = cases[cases.case.isin(a.only.split(","))].sort_values("t")      # chosen examples (docs/23): the case ids are looked up in the data, not filtered
at = pick.t.max()
run = "replay-outlook-" + at.strftime("%Y%m%dT%H%M"); out = Path(a.out) / run; out.mkdir(parents=True, exist_ok=True)
dirs = [f"{a.release}/part_p{i}" for i in range(6)]; fires = {}
NAMED_FILE = ROOT.parent / "wildfire-ignition-forecasting/results/named_places.parquet"; NAMED = pd.read_parquet(NAMED_FILE) if NAMED_FILE.exists() else None
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
H = 48                                        # the one outlook is stored under the 48 h key (truth = burned within 48 h)
FIRE = np.array([[255, 242, 58], [255, 184, 28], [255, 122, 18], [224, 54, 15], [179, 18, 28]], float)      # most likely (bright) to possible (deep red); the dashboard's --tw-wi-fire-1..5 tokens
TINT = {"tree": (47, 125, 67, 0.22), "crop": (201, 154, 58, 0.16), "built": (184, 184, 200, 0.20), "water": (58, 134, 200, 0.30)}      # --tw-wi-tint-*: land cover under the shading
def picture_url(rgba, scale=4, quality=82):
    """A small WebP data URL (the dashboard draws it as a map image; the JSON API carries it, so no separate image route or token is needed)."""
    im = Image.fromarray(rgba, "RGBA").resize((rgba.shape[1] * scale, rgba.shape[0] * scale), Image.BILINEAR); b = io.BytesIO(); im.save(b, "WEBP", quality=quality, method=6)
    return "data:image/webp;base64," + base64.b64encode(b.getvalue()).decode()
def heat_picture(rank, shown):
    """The ranking as a picture: the top cells bright yellow, falling through orange to deep red at the edge of what is shown, then nothing. A rank, not a probability."""
    t = np.clip(rank / shown, 0, 1); x = t * (len(FIRE) - 1); i = np.minimum(x.astype(int), len(FIRE) - 2); f = (x - i)[..., None]
    rgb = FIRE[i] * (1 - f) + FIRE[i + 1] * f; a = np.where(rank < shown, 0.95 - 0.55 * t, 0.0)
    a = gaussian_filter(a, 0.7); out = np.zeros(rank.shape + (4,), np.uint8); out[..., :3] = rgb.astype(np.uint8); out[..., 3] = (np.clip(a, 0, 1) * 255).astype(np.uint8); return out
def terrain_picture(elev, cover, res, az=315.0, alt=45.0, z=3.0):
    """Hillshade (Horn, light from the north-west) over a faint land-cover tint, as in the earlier dark terrain view. Transparent where the ground is flat and untinted."""
    e = np.where(np.isfinite(elev), elev, np.nanmean(elev) if np.isfinite(elev).any() else 0.0); gy, gx = np.gradient(e, res)      # gy: southward (rows grow south)
    nx, ny = -gx * z, gy * z; inv = 1 / np.sqrt(nx * nx + ny * ny + 1); a_, h_ = np.radians(az), np.radians(alt)
    shade = np.clip((nx * np.sin(a_) * np.cos(h_) + ny * np.cos(a_) * np.cos(h_) + np.sin(h_)) * inv, 0, 1); d = (shade - np.sin(h_)) * 2.2
    sa = np.minimum(0.85, np.abs(d)); sc = np.where(d[..., None] >= 0, 255.0, 0.0) * np.ones(3)
    tint = np.zeros(e.shape + (3,)); ta = np.zeros(e.shape)
    for k_, (r_, g_, b_, al) in TINT.items():
        m_ = (cover[k_] > 0.5) & (ta == 0); tint[m_] = (r_, g_, b_); ta[m_] = al
    a = sa + ta * (1 - sa); rgb = (sc * sa[..., None] + tint * (ta * (1 - sa))[..., None]) / np.maximum(a, 1e-6)[..., None]
    h_, w_ = e.shape; yy, xx = np.mgrid[0:h_, 0:w_]; edge = np.minimum.reduce([yy, xx, h_ - 1 - yy, w_ - 1 - xx]) / (0.12 * min(h_, w_)); edge = np.clip(edge, 0, 1); edge = edge * edge * (3 - 2 * edge)      # fade to nothing at the border, so the picture has no hard square edge on the map
    a = a * edge
    out = np.zeros(e.shape + (4,), np.uint8); out[..., :3] = np.clip(rgb, 0, 255).astype(np.uint8); out[..., 3] = (a * 255).astype(np.uint8); return out
def quad_of(g):
    ll = Transformer.from_crs(g.epsg, 4326, always_xy=True).transform; x1, y1 = g.x0 + 192 * g.res, g.y0 - 192 * g.res
    return [list(map(float, ll(x, y))) for x, y in ((g.x0, g.y0), (x1, g.y0), (x1, y1), (g.x0, y1))]      # top-left, top-right, bottom-right, bottom-left
D = {h: tr.Data(dirs, h) for h in (24, 48)}; MODELS = {}
for h in (24, 48):
    ckpts = [torch.load(p, weights_only=False, map_location="cpu") for p in sorted(glob.glob(f"{a.models}/fsn_{h}_s*.pt"))]; assert ckpts, f"no {h} h models"
    ms = []
    for ck in ckpts:
        m_ = imgmodels.make("firesensenet", len(ck["sp_names"]), len(ck["sc_names"]), pretrained=False); m_.load_state_dict(ck["model"]); m_.eval(); ms.append(m_)
    D[h].sp_names, D[h].sc_names = ckpts[0]["sp_names"], ckpts[0]["sc_names"]; MODELS[h] = (ckpts, ms)
def score_of(h, i):
    cks, ms = MODELS[h]; d = D[h]; out = []
    for ck, mod in zip(cks, ms):
        d.mu, d.sd, d.smu, d.ssd = ck["mu"], ck["sd"], ck["smu"], ck["ssd"]; x, s, _, _, _ = d.get(i)
        with torch.no_grad(): out.append(torch.sigmoid(mod(torch.from_numpy(x)[None], torch.from_numpy(s)[None]))[0].numpy())
    return np.mean(out, axis=0)
data = D[48]; n_models = sum(len(v[1]) for v in MODELS.values()); names = {n: j for j, n in enumerate(json.load(open(dirs[0] + "/names.json"))["spatial"])}; sc_all = json.load(open(dirs[0] + "/names.json"))["scalars"]
for k, (_, r) in enumerate(pick.iterrows()):
    i = int(data.meta.index[data.meta.case == r.case][0]); raw = data.raw(i); sp = data.sp_names; y, cand = raw[2], raw[3]
    m = cand > 0; idx = np.flatnonzero(m.ravel()); both = rankdata(-score_of(24, i).ravel()[idx]) + rankdata(-score_of(48, i).ravel()[idx])      # mean of the two ranks (1 = most likely)
    rank = np.empty(len(idx)); rank[np.argsort(both, kind="stable")] = np.arange(len(idx)) / len(idx)
    band = np.zeros(m.shape, np.int8).ravel(); band[idx] = np.select([rank < BANDS[H][0], rank < BANDS[H][1], rank < BANDS[H][2]], [3, 2, 1], 0); band = band.reshape(m.shape)
    rank_map = np.ones(m.shape).ravel(); rank_map[idx] = rank; rank_map = rank_map.reshape(m.shape)
    g = grids[k]; ev = f"replay-{r.case.replace('_', '-')}"; seed = raw[0][sp.index("seed")] > 0
    feats = [f for b in (1, 2, 3) if (f := polys_to_geojson(band == b, g, {"band": b, "event": ev, "horizon_h": H, "run": run, "mode": "replay"}))]
    json.dump(fc(feats), open(out / f"{ev}_{H}h.geojson", "w"))
    burned = polys_to_geojson(y > 0, g, {"band": 0, "burned": True, "event": ev, "horizon_h": H}); json.dump(fc([burned] if burned else []), open(out / f"{ev}_burned_{H}h.geojson", "w"))
    cover_layers = {k_: raw[0][sp.index(k_)] for k_ in ("tree", "crop", "built", "water")}
    json.dump({"coordinates": quad_of(g), "heat": picture_url(heat_picture(rank_map, BANDS[H][2])), "terrain": picture_url(terrain_picture(raw[0][sp.index("elev")], cover_layers, float(g.res)), scale=2),
               "note": "heat: rank of the cells asked about, brightest = most likely, shown down to the edge of the lowest band; not a probability"}, open(out / f"{ev}_overlay.json", "w"))
    rr, cc = np.nonzero(seed); top = band == 3; tr_, tc_ = np.nonzero(top)
    dy, dx = -(tr_.mean() - rr.mean()), tc_.mean() - cc.mean(); deg = (np.degrees(np.arctan2(dx, dy)) + 360) % 360
    cover = {k_: float(raw[0][names[k_]][top | (band == 2)].mean()) for k_ in ("tree", "shrub", "grass", "crop")}
    up = float((raw[0][names["slope_along"]][top] > 0).mean())
    summ = {"bearing_deg": round(float(deg)), "bearing": COMPASS[int(((deg + 22.5) % 360) // 45)], "cover": max(cover, key=cover.get), "cover_share": round(max(cover.values()), 2), "uphill_share": round(up, 2), "slope_deg": round(float(raw[0][names["slope"]][top].mean()), 1)}
    layers = {k_: raw[0][names[k_]] for k_ in ("pop", "tree", "shrub", "grass", "crop", "built", "water")}
    af = AF.affected(band, layers, g, PL[k], RD[k], seed)
    near = [p for p in af["places"] if p["kind"] in ("village", "town")]
    pl = [p for p in af["places"] if p["band"] > 0 or p["distance_km"] <= 3][:60]
    if NAMED is not None:      # the named-places table of the forecast work adds villages the OpenStreetMap extract of the run lacks (distance and direction from the fire's start, not in a band)
        have = {p["name"] for p in pl}; kx = np.cos(np.radians(r.lat)); near_ = NAMED[(abs(NAMED.lat - r.lat) < 0.06) & (abs(NAMED.lon - r.lon) < 0.06 / kx)]
        for q_ in near_.itertuples():
            d_ = float(np.hypot((q_.lat - r.lat) * 111.32, (q_.lon - r.lon) * 111.32 * kx))
            if d_ <= 6 and q_.name not in have and q_.kind in ("village", "town", "hamlet"):
                b_ = (np.degrees(np.arctan2((q_.lon - r.lon) * kx, q_.lat - r.lat)) + 360) % 360; pl.append({"name": q_.name, "kind": q_.kind, "lat": float(q_.lat), "lon": float(q_.lon), "band": 0, "distance_km": round(d_, 1), "bearing": COMPASS[int(((b_ + 22.5) % 360) // 45)]}); have.add(q_.name)
        pl.sort(key=lambda p: (p["band"] == 0, p["distance_km"])); pl = pl[:60]
    json.dump(fc([{"type": "Feature", "geometry": {"type": "Point", "coordinates": [p["lon"], p["lat"]]}, "properties": {kk: p[kk] for kk in ("name", "kind", "band", "distance_km", "bearing")}} for p in pl]), open(out / f"{ev}_places_{H}h.geojson", "w"))
    rd = [d for d in af["roads"] if d["band"] > 0]
    json.dump(fc([{"type": "Feature", "geometry": {"type": "LineString", "coordinates": d["coords"]}, "properties": {"name": d["name"], "highway": d["highway"], "band": d["band"]}} for d in rd]), open(out / f"{ev}_roads_{H}h.geojson", "w"))
    road_names = sorted({d["name"] for d in rd if d["name"]})[:8]
    frp = float(np.expm1(float(data.S[int(data.meta.part.iloc[i])][int(data.meta.row.iloc[i])][sc_all.index("frp_sum_log")])))
    fires[ev] = {"event": ev, "case": r.case, "state": r.state, "lat": float(r.lat), "lon": float(r.lon), "n_detections": int(r.n_start), "first_detection": str(r.t), "last_detection": str(r.t), "frp_sum": round(frp, 1), "overlay_file": f"{ev}_overlay.json",
                 "horizons": {str(H): {"file": f"{ev}_{H}h.geojson", "burned_file": f"{ev}_burned_{H}h.geojson", "places_file": f"{ev}_places_{H}h.geojson", "roads_file": f"{ev}_roads_{H}h.geojson", "cells_scored": int(m.sum()), "weather_source": str(r.weather_source),
                                       "bands": {str(b): int((band == b).sum()) for b in (3, 2, 1)}, "summary": summ, "affected": {**af["summary"], "nearest_settlement": near[0] if near else None, "roads_in_bands": len(rd), "road_names": road_names, "models": n_models}}}}
    foot = polys_to_geojson(seed, g, {"footprint": True, "event": ev, "detections": int(r.n_start)}); json.dump(fc([foot] if foot else []), open(out / f"{ev}_pixels.geojson", "w"))      # the fire at the start: one outline, not one marker per cell
    print(ev, n_models, "models", len(pl), "places", len(rd), "roads", flush=True)
CHK_FILE = ROOT / "data/65_outlook_check.csv"; CHK = pd.read_csv(CHK_FILE) if CHK_FILE.exists() else None
if CHK is not None:
    for ev, f in fires.items():
        r = CHK[CHK.case == f["case"]]
        if len(r):
            r = r.iloc[0]; km2 = (r.res_m / 1000) ** 2
            f["check"] = {"burned_km2": round(float(r.burned_cells * km2), 1), "top_band_km2": round(float(r.model_b1_cells * km2), 1), "top_band_capture": round(float(r.model_b1_capture), 2), "top_band_precision": round(float(r.model_b1_precision), 2),
                          "nearest_capture": round(float(r.nearest_b1_capture), 2), "three_bands_km2": round(float(r.model_b3_cells * km2), 1), "three_bands_capture": round(float(r.model_b3_capture), 2)}
json.dump({"run": run, "mode": "replay", "at": str(at), "model_tag": f"gated-unet-{n_models}models", "outlook": "next 1 to 2 days", "issued": dt.datetime.now(dt.timezone.utc).isoformat(), "fires": list(fires.values()),
           "note": "A ranking of the cells within 8 km of the fire for the next one to two days, not a probability and not a timing. Wind is not used. Archived fires from 2025, scored as they would have been at the time. Places and roads from OpenStreetMap (fetched when the run was made; the January 2021 extract where the service was busy); population from WorldPop 2019-2020."}, open(out / "events.json", "w"), indent=1)
json.dump({"source": "gated U-Net, the 24 h and 48 h models averaged (three seeds each), 250 test fires (2025), cells within 8 km of the fire, scored against what burned within 48 h; reference: ranking cells by distance to the fire",
           "horizons": {"48": {"model_iou": 0.248, "baseline_iou": 0.123, "model_direction_auc": 0.812, "baseline_direction_auc": 0.599, "cases": 250}},
           **({"check": {"fires": int(len(CHK)), "burned_km2_median": round(float((CHK.burned_cells * (CHK.res_m / 1000) ** 2).median()), 1), "top_band_km2_median": round(float((CHK.model_b1_cells * (CHK.res_m / 1000) ** 2).median()), 1),
                         "top_band_capture_median": round(float(CHK.model_b1_capture.median()), 2), "nearest_capture_median": round(float(CHK.nearest_b1_capture.median()), 2), "share_capture_80": round(float((CHK.model_b1_capture >= 0.8).mean()), 2),
                         "share_capture_under_30": round(float((CHK.model_b1_capture < 0.3).mean()), 2), "share_better_than_nearest": round(float((CHK.model_b1_capture > CHK.nearest_b1_capture).mean()), 2),
                         "three_bands_km2_median": round(float((CHK.model_b3_cells * (CHK.res_m / 1000) ** 2).median()), 1), "three_bands_capture_median": round(float(CHK.model_b3_capture.median()), 2)}} if CHK is not None else {})}, open(out / "skill.json", "w"), indent=1)
print(run, len(fires), "fires")
