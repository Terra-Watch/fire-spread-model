import numpy as np

from firespread import grid, rasters


def test_grid_is_centred_aligned_and_in_the_right_utm_zone():
    g = grid.make_grid(21.5, 86.3, half_km=12.0, res=100.0)
    assert g.epsg == 32645 and g.nx == g.ny == 240
    assert g.x0 % 100 == 0 and g.y0 % 100 == 0
    r, c = g.to_rc(21.5, 86.3)
    assert abs(r - 120) <= 1.0 and abs(c - 120) <= 1.0
    lat, lon = g.centers_lonlat()
    assert lat.shape == g.shape and abs(lat.mean() - 21.5) < 0.01 and abs(lon.mean() - 86.3) < 0.01
    w, s, e, n = g.bounds_lonlat()
    assert w < 86.3 < e and s < 21.5 < n


def test_neighbouring_grids_line_up():
    a = grid.make_grid(21.5, 86.30)
    b = grid.make_grid(21.5, 86.32)
    assert (b.x0 - a.x0) % 100 == 0 and (b.y0 - a.y0) % 100 == 0


def test_slope_and_uphill_direction_on_planes():
    n, res = 30, 100.0
    cols = np.arange(n)[None, :].repeat(n, 0)
    rows = np.arange(n)[:, None].repeat(n, 1)
    east = 20.0 * cols            # rises 20 m per 100 m eastwards
    s, up = rasters.slope_upslope(east.astype(float), res)
    assert abs(s[15, 15] - np.degrees(np.arctan(0.2))) < 0.05 and abs(up[15, 15] - 90.0) < 0.5
    north = -20.0 * rows           # rows grow southwards, so this rises to the north
    s, up = rasters.slope_upslope(north.astype(float), res)
    assert abs(s[15, 15] - np.degrees(np.arctan(0.2))) < 0.05 and (up[15, 15] < 0.5 or up[15, 15] > 359.5)
    flat, _ = rasters.slope_upslope(np.full((n, n), 100.0), res)
    assert float(flat.max()) == 0.0
