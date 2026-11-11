import numpy as np
import pandas as pd

from firespread import events

T0 = pd.Timestamp("2024-03-01 08:00", tz="UTC")


def det(rows):
    """rows: (east_km, north_km, hours) around 22 N, 80 E."""
    la0, lo0 = 22.0, 80.0
    k_lat = 111.195
    k_lon = 111.195 * np.cos(np.radians(la0))
    return pd.DataFrame({"latitude": [la0 + n / k_lat for _, n, _ in rows], "longitude": [lo0 + e / k_lon for e, _, _ in rows],
                         "t": [T0 + pd.Timedelta(hours=h) for _, _, h in rows], "sat": "N20", "frp": 5.0})


def test_close_in_space_and_time_is_one_event_far_is_not():
    d = det([(0, 0, 0), (0.5, 0, 10), (0.9, 0.3, 30), (30, 0, 1), (30.4, 0, 2), (0, 0, 200)])
    lab = events.link_events(d)
    assert lab[0] == lab[1] == lab[2] and lab[3] == lab[4] and lab[0] != lab[3] and lab[5] != lab[0]


def test_a_spreading_fire_stays_one_event_by_chaining():
    d = det([(i * 0.9, 0, i * 6) for i in range(8)])
    assert len(set(events.link_events(d))) == 1


def test_epochs_merge_nearby_passes_and_split_far_ones():
    d = det([(0, 0, 0), (0.1, 0, 0.8), (0.2, 0, 12), (0.3, 0, 12.5), (0.5, 0, 25)])
    d["event"] = 0
    ep = events.assign_epochs(d)
    assert ep.tolist() == [0, 0, 1, 1, 2]


def test_case_counts_new_growth_and_advance():
    # epoch 0: a 3-pixel start; epoch 1: one pixel in place and two 2.0 km and 3.0 km east
    d = det([(0, 0, 0), (0.4, 0, 0), (0, 0.4, 0), (0.1, 0.1, 12), (2.0, 0, 12), (3.0, 0, 12)])
    d["event"] = 0
    d["epoch"] = events.assign_epochs(d)
    cases = events.case_table(d)
    assert len(cases) == 1
    c = cases.iloc[0]
    assert c.n_start == 3 and c.n_truth == 3 and c.n_new == 2 and abs(c.gap_h - 12.0) < 0.1
    assert 2.4 < c.adv_max_km < 3.1 and c.adv_p95_km > 2.0


def test_event_table_summary():
    d = det([(0, 0, 0), (1.5, 0, 12), (3.0, 0, 24)])
    d["event"] = 0
    d["epoch"] = events.assign_epochs(d)
    e = events.event_table(d).iloc[0]
    assert e.n_det == 3 and e.n_epochs == 3 and abs(e.duration_h - 24) < 0.01 and 2.9 < e.extent_km < 3.1
