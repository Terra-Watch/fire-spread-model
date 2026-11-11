"""Roads, tracks and rivers around each fire (firebreaks and access) from the local OpenStreetMap extract (wildfire-dataset/inputs_raw/osm, January 2021).
Prepare the lines once:  osmium tags-filter -o lines.osm.pbf INDIA.pbf w/highway=motorway,trunk,primary,secondary,tertiary,unclassified,track w/waterway=river
                         osmium export lines.osm.pbf -f geojsonseq -o lines.geojsonseq
Output data/<root>/<id>/osm.npz with road_dist and river_dist in cells (240 where there are none). Safe to re-run.
Run: .venv/bin/python scripts/24_osm.py LINES.geojsonseq"""
import json
import sys
from pathlib import Path

import numpy as np
from pyproj import Transformer
from rasterio.features import rasterize
from scipy import ndimage
from shapely.geometry import LineString

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from firespread import config, grid as G  # noqa: E402

CAP = 240.0
ROADS = {"motorway", "trunk", "primary", "secondary", "tertiary", "unclassified", "track"}


def grid_of(d: Path) -> G.Grid:
    env = np.load(d / "env.npz")
    return G.Grid(int(env["grid"][0]), *map(float, env["grid"][1:4]), int(env["grid"][4]), int(env["grid"][5]))


def load_lines(path, need):
    """Lines (kind, lon/lat array) that touch a quarter-degree tile some case window touches."""
    kept = []
    with open(path) as f:
        for row in f:
            if '"LineString"' not in row[:200]:
                continue
            ft = json.loads(row.lstrip("\x1e"))
            p = ft["properties"]
            kind = "road" if p.get("highway") in ROADS else "river" if p.get("waterway") == "river" else None
            if kind is None:
                continue
            xy = np.array(ft["geometry"]["coordinates"])
            tiles = set(map(tuple, np.unique(np.floor(xy * 4).astype(int), axis=0)))
            if tiles & need:
                kept.append((kind, xy))
    return kept


def main(lines_path):
    dirs = [p.parent for root in ("cases", "bank") for p in sorted((config.DATA / root).glob("*/burn_long.npz"))]
    grids, need = {}, set()
    for d in dirs:
        g = grids[d] = grid_of(d)
        w, s, e, n = g.bounds_lonlat(pad_deg=0.01)
        need |= {(i, j) for i in range(int(np.floor(w * 4)), int(np.floor(e * 4)) + 1) for j in range(int(np.floor(s * 4)), int(np.floor(n * 4)) + 1)}
    lines = load_lines(lines_path, need)
    print("lines kept", len(lines), "of tiles", len(need), flush=True)
    bbox = np.array([[xy[:, 0].min(), xy[:, 1].min(), xy[:, 0].max(), xy[:, 1].max()] for _, xy in lines])
    kinds = np.array([k for k, _ in lines])
    for i, d in enumerate(dirs, 1):
        out = d / "osm.npz"
        if out.exists():
            continue
        g = grids[d]
        w, s, e, n = g.bounds_lonlat(pad_deg=0.01)
        near = np.flatnonzero((bbox[:, 0] <= e) & (bbox[:, 2] >= w) & (bbox[:, 1] <= n) & (bbox[:, 3] >= s))
        tf = Transformer.from_crs(4326, g.epsg, always_xy=True)
        res = {}
        for name, kind in (("road_dist", "road"), ("river_dist", "river")):
            geoms = [LineString(zip(*tf.transform(lines[j][1][:, 0], lines[j][1][:, 1]))) for j in near if kinds[j] == kind and len(lines[j][1]) > 1]
            if geoms:
                m = rasterize(geoms, out_shape=g.shape, transform=g.transform, all_touched=True, dtype="uint8")
                res[name] = np.minimum(ndimage.distance_transform_edt(m == 0), CAP).astype(np.float32) if m.any() else np.full(g.shape, CAP, np.float32)
            else:
                res[name] = np.full(g.shape, CAP, np.float32)
        np.savez_compressed(out, **res)
        if i % 100 == 0:
            print(i, "of", len(dirs), flush=True)
    print("done", len(dirs))


if __name__ == "__main__":
    main(sys.argv[1])
