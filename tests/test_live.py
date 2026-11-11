import numpy as np

from firespread import grid as G, live


def test_band_geojson_has_one_feature_per_band_in_lonlat():
    g = G.make_grid(17.4, 78.5)
    band = np.zeros(g.shape, np.int8)
    band[100:110, 100:110] = 3
    band[100:110, 110:130] = 1
    gj = live.band_geojson({"band": band}, g, {"event": "e"})
    assert [f["properties"]["band"] for f in gj["features"]] == [1, 3]
    assert all(f["properties"]["event"] == "e" for f in gj["features"])
    geom = gj["features"][0]["geometry"]                    # a single patch comes out as a Polygon, several as a MultiPolygon
    ring = geom["coordinates"][0] if geom["type"] == "Polygon" else geom["coordinates"][0][0]
    lon, lat = ring[0]
    assert 78.3 < lon < 78.7 and 17.2 < lat < 17.6          # the fire's own neighbourhood, in degrees
