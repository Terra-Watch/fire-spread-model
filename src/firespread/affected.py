"""What a fire's bands run over: people, settlements, roads and public buildings inside each band, from the layers of the case plus the local OpenStreetMap extract (places.geojsonseq, osm_lines.geojsonseq).
Used by scripts/62_outlook_run.py. Population is the 100 m layer 'pop' (people per km2), so people in a band = sum(pop) x 0.01 km2 per cell."""
import json
from pathlib import Path
import numpy as np
from pyproj import Transformer
from shapely.geometry import shape

KIND = {"village": "village", "hamlet": "village", "town": "town", "city": "town", "suburb": "village", "locality": "village", "school": "school", "hospital": "health", "clinic": "health", "fire_station": "fire_station", "police": "police", "place_of_worship": "worship"}
COMPASS = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]


def read_places(path: Path, boxes: list[tuple[float, float, float, float]]) -> list[list[dict]]:
    """Named places and public buildings inside each (west, south, east, north) box, one pass over the file."""
    out = [[] for _ in boxes]
    for line in open(path, encoding="utf8"):
        line = line.lstrip("\x1e").strip()
        if not line:
            continue
        f = json.loads(line); p = f["properties"]; k = KIND.get(p.get("place") or p.get("amenity") or "")
        if not k or not p.get("name"):
            continue
        g = f["geometry"]; c = g["coordinates"] if g["type"] == "Point" else list(shape(g).centroid.coords)[0]
        for i, (w, s, e, n) in enumerate(boxes):
            if w <= c[0] <= e and s <= c[1] <= n:
                out[i].append({"name": p.get("name:en") or p["name"], "kind": k, "lon": c[0], "lat": c[1]})
    return out


def read_roads(path: Path, boxes: list[tuple[float, float, float, float]]) -> list[list[dict]]:
    """Roads and tracks that touch each box, one pass over the file."""
    out = [[] for _ in boxes]
    for line in open(path, encoding="utf8"):
        if '"highway"' not in line:
            continue
        f = json.loads(line.lstrip("\x1e")); g = f["geometry"]
        if g["type"] != "LineString":
            continue
        xs = [c[0] for c in g["coordinates"]]; ys = [c[1] for c in g["coordinates"]]
        for i, (w, s, e, n) in enumerate(boxes):
            if max(xs) >= w and min(xs) <= e and max(ys) >= s and min(ys) <= n:
                p = f["properties"]; out[i].append({"coords": g["coordinates"], "name": p.get("name") or p.get("ref") or "", "highway": p["highway"]})
    return out


def affected(band: np.ndarray, layers: dict, grid, places: list[dict], roads: list[dict], seed: np.ndarray, res_m: float = 100.0) -> dict:
    """Numbers, places and roads for one horizon. `band` is 3 (most likely) .. 1, 0 outside; `layers` holds the case's 'pop', 'tree', 'shrub', 'grass', 'crop', 'built', 'water', 'bare'."""
    cell_km2 = (res_m / 1000) ** 2; to_xy = Transformer.from_crs(4326, grid.epsg, always_xy=True).transform
    rr, cc = np.nonzero(seed); cy, cx = rr.mean(), cc.mean(); summary = {"bands": {}}
    for b in (3, 2, 1):
        m = band == b
        summary["bands"][str(b)] = {"area_km2": round(float(m.sum() * cell_km2), 1), "people": int(round(float(np.nansum(layers["pop"][m]) * cell_km2), -1)) if m.any() else 0,
                                    "cover": {k: round(float(np.nanmean(layers[k][m])), 2) for k in ("tree", "shrub", "grass", "crop", "built", "water")} if m.any() else {}}
    out_places = []
    for p in places:
        x, y = to_xy(p["lon"], p["lat"]); r, c = (grid.y0 - y) / grid.res, (x - grid.x0) / grid.res
        if not (0 <= r < band.shape[0] and 0 <= c < band.shape[1]):
            continue
        bd = int(band[int(r), int(c)]); dk = float(np.hypot(r - cy, c - cx) * res_m / 1000)
        if bd == 0 and dk > 8:
            continue
        deg = (np.degrees(np.arctan2(c - cx, -(r - cy))) + 360) % 360
        out_places.append({**p, "band": bd, "distance_km": round(dk, 1), "bearing": COMPASS[int(((deg + 22.5) % 360) // 45)]})
    out_places.sort(key=lambda p: (-p["band"], p["distance_km"]))
    out_roads = []
    for rd in roads:
        xy = np.array([to_xy(lo, la) for lo, la in rd["coords"]]); rs = ((grid.y0 - xy[:, 1]) / grid.res).astype(int); cs = ((xy[:, 0] - grid.x0) / grid.res).astype(int)
        ok = (rs >= 0) & (rs < band.shape[0]) & (cs >= 0) & (cs < band.shape[1])
        if not ok.any():
            continue
        bmax = int(band[rs[ok], cs[ok]].max())
        out_roads.append({**rd, "band": bmax})
    summary["places"] = {"by_band": {str(b): {k: sum(1 for p in out_places if p["band"] == b and p["kind"] == k) for k in ("village", "town", "school", "health", "worship", "fire_station", "police")} for b in (3, 2, 1)}}
    return {"summary": summary, "places": out_places, "roads": out_roads}


def fetch_overpass(box: tuple[float, float, float, float], cache: Path | None = None, tries: int = 4) -> tuple[list[dict], list[dict]]:
    """Current OpenStreetMap places, public buildings, roads and tracks for a (west, south, east, north) box from the public Overpass service, with a cache file and back-off.
    The service is often busy: callers fall back to the local extract when this raises."""
    import hashlib, time
    import httpx
    w, s, e, n = box; key = hashlib.sha1(f"{w:.4f},{s:.4f},{e:.4f},{n:.4f}".encode()).hexdigest()[:12]; f = cache / f"{key}.json" if cache else None
    if f and f.is_file():
        d = json.loads(f.read_text())
    else:
        q = (f'[out:json][timeout:60];(node["place"~"village|hamlet|town|city|suburb|locality"]({s},{w},{n},{e});node["amenity"~"school|hospital|clinic|fire_station|police"]({s},{w},{n},{e});'
             f'way["highway"~"motorway|trunk|primary|secondary|tertiary|unclassified|track"]({s},{w},{n},{e}););out geom tags;')
        for k in range(tries):
            try:
                r = httpx.post("https://overpass-api.de/api/interpreter", data={"data": q}, headers={"User-Agent": "TerraWatch-outlook/1.0", "Accept": "application/json"}, timeout=90)
                if r.status_code == 200:
                    d = r.json(); break
            except httpx.HTTPError:
                pass
            time.sleep(8 * 2 ** k)
        else:
            raise RuntimeError("Overpass unavailable")
        if f:
            f.parent.mkdir(parents=True, exist_ok=True); f.write_text(json.dumps(d))
    places, roads = [], []
    for el in d["elements"]:
        t = el.get("tags", {})
        if el["type"] == "node":
            k = KIND.get(t.get("place") or t.get("amenity") or "")
            if k and t.get("name"):
                places.append({"name": t.get("name:en") or t["name"], "kind": k, "lon": el["lon"], "lat": el["lat"]})
        elif el["type"] == "way" and "geometry" in el:
            roads.append({"coords": [[g["lon"], g["lat"]] for g in el["geometry"]], "name": t.get("name") or t.get("ref") or "", "highway": t["highway"]})
    return places, roads
