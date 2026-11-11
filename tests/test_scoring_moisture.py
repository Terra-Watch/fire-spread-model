import numpy as np

from firespread import moisture, scoring


def test_auc_perfect_random_and_tied():
    lab = np.array([1, 1, 0, 0, 0], bool)
    assert scoring.auc([5, 4, 3, 2, 1], lab) == 1.0
    assert scoring.auc([1, 2, 3, 4, 5], lab) == 0.0
    assert scoring.auc([1, 1, 1, 1, 1], lab) == 0.5
    assert np.isnan(scoring.auc([1, 2], np.array([False, False])))


def test_area_for_capture():
    lab = np.array([1, 0, 1, 0, 0, 0, 0, 0, 0, 0], bool)
    perfect = np.array([9, 1, 8, 1, 1, 1, 1, 1, 1, 1], float)
    assert abs(scoring.area_for_capture(perfect, lab, 0.5) - 0.1) < 1e-9          # one cell of ten holds half the growth
    assert abs(scoring.area_for_capture(perfect, lab, 1.0) - 0.2) < 1e-9
    assert abs(scoring.area_for_capture(np.ones(10), lab, 0.5) - 0.5) < 1e-9      # no information: half the area for half the cells


def test_bootstrap_interval_contains_the_mean():
    m, lo, hi = scoring.bootstrap_mean([0.2, 0.3, 0.1, 0.4, 0.25])
    assert lo <= m <= hi and abs(m - 0.25) < 1e-9


def test_equilibrium_moisture_is_low_when_hot_and_dry_and_high_when_humid():
    hot_dry = moisture.equilibrium(38.0, 15.0)
    cool_humid = moisture.equilibrium(18.0, 85.0)
    assert 0.02 <= hot_dry < 0.06 and cool_humid > 0.15


def test_dead_fuel_lags_follows_and_rain_wets_it():
    n = 48
    m = moisture.dead_fuel(np.full(n, 35.0), np.full(n, 20.0), np.zeros(n), start=0.25)
    assert m[0] > m[10] > m[-1] and m[-1] < 0.2                                   # dries toward the equilibrium
    wet = moisture.dead_fuel(np.full(n, 35.0), np.full(n, 20.0), np.where(np.arange(n) < 3, 5.0, 0.0), start=0.05)
    assert wet[3] > 0.2 and wet[3] > m[3] - 0.3


def test_iou_at_true_area():
    truth = np.array([1, 1, 0, 0, 0, 0], bool)
    assert scoring.iou_at_true_area(np.array([9, 8, 1, 2, 3, 4.0]), truth) == 1.0
    assert abs(scoring.iou_at_true_area(np.array([9, 1, 8, 2, 3, 4.0]), truth) - 1 / 3) < 1e-9      # one of two hit: 1 / (4 - 1)
