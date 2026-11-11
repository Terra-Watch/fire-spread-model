"""Invariants every exported dataset part must satisfy (run on data/dataset/part_test, or on the part named in DATASET_PART)."""
import json
import os

import numpy as np
import pandas as pd
import pytest

from firespread import config

PART = config.DATA / "dataset" / os.environ.get("DATASET_PART", "part_test")
pytestmark = pytest.mark.skipif(not (PART / "meta.parquet").exists(), reason="no exported dataset part")


@pytest.fixture(scope="module")
def d():
    names = json.load(open(PART / "names.json"))
    return {"names": names, "meta": pd.read_parquet(PART / "meta.parquet"), "X": np.load(PART / "spatial.npy", mmap_mode="r"), "shared": np.load(PART / "shared.npz"),
            **{f"lab{H}": np.load(PART / f"labels_{H}.npz") for H in (24, 48)}, **{f"sc{H}": np.load(PART / f"scalars_{H}.npy") for H in (24, 48)}}


def layer(d, name):
    return np.asarray(d["X"][:, d["names"]["spatial"].index(name)], np.float32)


def test_shapes_and_ids(d):
    n = len(d["meta"]); c = d["names"]["crop"]
    assert d["X"].shape == (n, len(d["names"]["spatial"]), c, c)
    assert d["sc24"].shape == d["sc48"].shape == (n, len(d["names"]["scalars"]))
    assert d["meta"].case.is_unique and d["shared"]["cand"].shape == (n, c, c)


def test_no_infinities_and_scalars_complete(d):
    assert not np.isinf(np.asarray(d["X"], np.float32)).any()
    assert np.isfinite(d["sc24"]).all() and np.isfinite(d["sc48"]).all()


def test_labels_are_inside_the_cells_asked_about(d):
    cand = d["shared"]["cand"] > 0
    for H in (24, 48):
        assert not (d[f"lab{H}"]["y_modis"].astype(bool) & ~cand).any()
        assert not (d[f"lab{H}"]["y_modis"].astype(bool) & ~d[f"lab{H}"]["y_modis_all"].astype(bool)).any()    # the connected truth is part of all burned cells
    dist = d["shared"]["dist"].astype(np.float32)
    assert dist[cand].min() >= 5 and dist[cand].max() <= 80 + 1e-3                                         # 0.5 to 8 km from the start footprint


def test_seed_and_distance_agree(d):
    seed = layer(d, "seed") > 0
    assert (d["shared"]["dist"].astype(np.float32)[seed] == 0).all()
    assert not (d["shared"]["cand"] > 0)[seed].any()
    for i in range(len(seed)):                                                                              # the crop is centred on the fire
        r, c = np.argwhere(seed[i]).mean(axis=0)
        assert abs(r - 96) < 40 and abs(c - 96) < 40


def test_physical_ranges(d):
    for k in ("tree", "shrub", "grass", "crop", "built", "water", "bare", "forest_share"):
        v = layer(d, k); assert np.nanmin(v) >= -1e-3 and np.nanmax(v) <= 1.01, k
    assert np.nanmin(layer(d, "dist")) >= 0 and np.nanmin(layer(d, "road_dist")) >= 0 and np.nanmin(layer(d, "pop")) >= 0
    assert np.nanmin(layer(d, "slope")) >= 0 and np.nanmax(layer(d, "slope")) <= 90
    s2 = ((layer(d, "s2_valid") > 0) == np.isfinite(layer(d, "s2_nir")))
    assert s2.all()


def test_nothing_comes_from_the_future(d):
    assert np.nanmin(layer(d, "fire_days_since")) >= 0                                                     # history layers are measured back from the start
    has = d["meta"].has_burnhist.values
    if has.any():
        assert np.nanmin(layer(d, "burn_days_since")[has]) >= 0
    assert np.isnan(layer(d, "burn_days_since")[~has]).all()                                                # a case without the history says "unknown", never a made-up value
    if (d["meta"].has_s2).any():
        assert np.nanmin(layer(d, "s2_age")) >= 1 - 1e-3                                                   # scenes are at least a day old
    anom = layer(d, "ndvi_anom"); ref = layer(d, "ndvi") - layer(d, "ndvi_prev"); ok = np.isfinite(anom) & np.isfinite(ref)
    assert ok.any() and np.nanmax(np.abs(anom[ok] - ref[ok])) < 2e-2


def test_splits_by_block(d):
    m = d["meta"]
    assert set(m.split) <= {"dev", "test"} and m.fold.between(0, 4).all()
    assert (m.groupby("block").split.nunique() == 1).all()                                                  # one split per 1 x 1 degree block
