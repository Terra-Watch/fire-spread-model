import numpy as np

from firespread import engine
from firespread.engine import FOREST, FuelTable, arrival_times

N, RES = 121, 100.0
C = N // 2


def run(wind=0.0, wind_from=270.0, moisture=0.05, slope=None, upslope=0.0, fuel=None, factor=None, horizon=400.0, hours=420, table=None, beta=3.0):
    fuel = np.full((N, N), FOREST) if fuel is None else fuel
    tan = np.zeros((N, N)) if slope is None else slope
    up = np.full((N, N), upslope)
    cf = np.ones((N, N)) if factor is None else factor
    return arrival_times(fuel, tan, up, cf, table or FuelTable.priors(), np.full(hours, wind), np.full(hours, wind_from), np.full(hours, moisture), [(C, C)], horizon, RES, beta)


def test_no_wind_gives_round_fire_with_the_base_rate():
    t = run()
    r0 = FuelTable.priors().r0[FOREST] * float(engine.damping(0.05, FuelTable.priors().mx[FOREST]))
    # 30 cells = 3 km straight in every compass direction takes 3000 / rate hours
    want = 3000.0 / r0
    for r, c in [(C - 30, C), (C + 30, C), (C, C - 30), (C, C + 30)]:
        assert abs(t[r, c] - want) / want < 0.02
    diag = t[C - 21, C + 21]                                   # 21 cells on both axes = 2.97 km
    assert abs(diag - 2969.0 / r0) / (2969.0 / r0) < 0.03      # sixteen neighbours: the diagonal is not slower than the axes


def test_wind_pushes_the_head_downwind_and_the_back_is_slowest():
    t = run(wind=5.0, wind_from=270.0)                       # from the west: the head runs east
    head, back, flank_n, flank_s = t[C, C + 20], t[C, C - 20], t[C - 20, C], t[C + 20, C]
    assert head < flank_n < back and abs(flank_n - flank_s) / flank_n < 0.02
    assert back / head > 4.0                                   # a real ellipse, not a slight skew
    t2 = run(wind=5.0, wind_from=0.0)                         # from the north: the head runs south
    assert t2[C + 20, C] < t2[C, C + 20] < t2[C - 20, C]
    assert np.allclose(t2, np.rot90(t, k=-1), rtol=0.02, equal_nan=False) or abs(t2[C + 20, C] - t[C, C + 20]) < 0.05 * t[C, C + 20]


def test_slope_speeds_the_fire_uphill_and_slows_it_downhill():
    tan = np.full((N, N), np.tan(np.radians(15)))
    t = run(slope=tan, upslope=90.0)                          # the ground rises to the east
    flat = run()
    assert t[C, C + 20] < flat[C, C + 20] < t[C, C - 20]


def test_drier_fuel_burns_faster_and_wet_fuel_stops_the_fire():
    dry, damp, wet = run(moisture=0.03), run(moisture=0.18), run(moisture=0.40)
    assert dry[C, C + 15] < damp[C, C + 15]
    assert np.isinf(wet[C, C + 5])                              # above the moisture of extinction nothing spreads


def test_barriers_and_unburnable_cells_stop_the_fire_but_a_gap_lets_it_through():
    fuel = np.full((N, N), FOREST)
    fuel[:, C + 10] = engine.NONBURNABLE                        # a continuous firebreak 10 cells east
    t = run(fuel=fuel)
    assert np.isinf(t[C, C + 15]) and np.isfinite(t[C, C - 15])
    fuel[C, C + 10] = FOREST                                    # one gap
    t = run(fuel=fuel)
    open_forest = run()
    assert np.isfinite(t[C, C + 15]) and abs(t[C, C + 15] - open_forest[C, C + 15]) < 0.5       # straight through the gap: no delay
    assert t[C + 8, C + 15] > open_forest[C + 8, C + 15] * 1.05                                  # off the axis it has to funnel through the gap, later


def test_changing_wind_changes_the_direction_of_the_run():
    hours = 420
    dirn = np.where(np.arange(hours) < 12, 270.0, 0.0)          # west wind for 12 h, then north wind
    t = arrival_times(np.full((N, N), FOREST), np.zeros((N, N)), np.zeros((N, N)), np.ones((N, N)), FuelTable.priors(), np.full(hours, 5.0), dirn, np.full(hours, 0.05), [(C, C)], 400.0, RES)
    only_west = run(wind=5.0, wind_from=270.0)
    assert t[C + 20, C] < only_west[C + 20, C]                  # the south run exists only because the wind turned


def test_same_inputs_same_answer_and_the_horizon_is_respected():
    a, b = run(wind=3.0), run(wind=3.0)
    assert np.array_equal(a, b)
    short = run(wind=3.0, horizon=5.0)
    assert np.isfinite(short).sum() < np.isfinite(a).sum() and np.nanmax(short[np.isfinite(short)]) <= 5.0
