"""The shared input builders reproduce what the research scripts saved (needs network for the first two; skipped offline)."""
import json
from pathlib import Path

import httpx
import numpy as np
import pandas as pd
import pytest

from firespread import ancillary, config, grid as G

CASE = config.DATA / "cases" / "2019-002106_0"


def _grid():
    env = np.load(CASE / "env.npz")
    return G.Grid(int(env["grid"][0]), *map(float, env["grid"][1:4]), int(env["grid"][4]), int(env["grid"][5]))


@pytest.mark.skipif(not CASE.exists(), reason="research data not built")
def test_population_matches_saved():
    assert np.allclose(ancillary.population(_grid(), 2019), np.load(CASE / "pop.npz")["pop"])


@pytest.mark.skipif(not CASE.exists(), reason="research data not built")
def test_antecedent_and_soil_match_saved():
    meta = json.load(open(CASE / "meta.json"))
    ant = pd.read_parquet(config.DATA / "antecedent.parquet").set_index("case").loc[meta["case_id"]]
    sm = pd.read_parquet(config.DATA / "soil.parquet").set_index("case").loc[meta["case_id"]]
    try:
        with httpx.Client(timeout=60) as c:
            a = ancillary.antecedent(c, float(meta["lat"]), float(meta["lon"]), meta["t0"], meta["case_id"])
            s = ancillary.soil(c, float(meta["lat"]), float(meta["lon"]), meta["t0"], meta["case_id"])
    except httpx.HTTPError:
        pytest.skip("offline")
    assert all(np.isclose(a[k], ant[k], rtol=1e-6, atol=1e-6) for k in ant.index)
    assert all(np.isclose(s[k], sm[k], rtol=1e-6, atol=1e-6) for k in sm.index)
