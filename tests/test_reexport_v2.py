"""v2 export (scripts/54b): the rebuild with all detections equals the stored part, and with clean detections no history count goes up. Needs data/detection_scores_lookup.parquet and data/dataset/part_test."""
import importlib.util
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

R = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.skipif(not (R / "data" / "detection_scores_lookup.parquet").exists() or not (R / "data" / "dataset" / "part_test").exists(), reason="needs the lookup and the test part")


def load():
    sp = importlib.util.spec_from_file_location("reexport", R / "scripts" / "54b_reexport_detection_layers.py"); m = importlib.util.module_from_spec(sp); sp.loader.exec_module(m); return m


def test_check_and_v2(tmp_path):
    m = load(); part = R / "data" / "dataset" / "part_test"
    m.main(part, True)                                              # asserts: rebuild with all detections == stored layers (float16 tolerance)
    dst = tmp_path / "part_t"; shutil.copytree(part, dst); m.main(dst, False)
    v1, v2 = np.load(dst / "spatial.npy", mmap_mode="r"), np.load(tmp_path / "part_t_v2" / "spatial.npy", mmap_mode="r"); names = json.load(open(dst / "names.json"))["spatial"]
    assert v1.shape == v2.shape
    for k in ("fire_hist_90", "fire_hist_365", "other_fires_72h"):
        i = names.index(k); assert (v2[:, i].astype(np.float32) <= v1[:, i].astype(np.float32) + 1e-3).all(), k      # removing detections can only lower a count
    for k in ("seed", "det_frp", "dist", "tree", "elev"):
        i = names.index(k); assert np.array_equal(np.nan_to_num(v1[:, i]), np.nan_to_num(v2[:, i])), k      # nothing else changes
    meta = pd.read_parquet(tmp_path / "part_t_v2" / "meta.parquet"); assert {"p_real_start", "admitted_v2", "n_start_unscored"} <= set(meta.columns) and len(meta) == len(v2)
