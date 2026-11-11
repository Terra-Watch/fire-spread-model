"""Builds notebooks/01_raw_to_inputs.ipynb: one fire followed from the raw sources to the 47 layers the models see. Executes it so the figures are saved inside.   Run: .venv/bin/python scripts/50_make_notebook_inputs.py"""
import nbformat as nbf
from nbconvert.preprocessors import ExecutePreprocessor
from pathlib import Path

C = []
md = lambda s: C.append(nbf.v4.new_markdown_cell(s.strip()))
code = lambda s: C.append(nbf.v4.new_code_cell(s.strip()))

md("""# From raw data to the model inputs: one fire, step by step

This notebook follows **one real fire** from the raw sources to the 47 picture layers the image models are trained on.
Each section shows the raw data on the left and what it becomes on the right. All maps are the same 19.2 km square around the fire's start, 100 m per pixel (192 x 192 cells).

| Section | Raw source | Becomes |
|---|---|---|
| A | VIIRS satellite fire detections (375 m) | fire footprint, distance-to-fire map |
| B | ESA WorldCover 10 m land cover, forest-land mask | tree / shrub / grass / crop / built / water shares, forest share |
| C | Copernicus 30 m terrain | elevation, slope, uphill direction |
| D | MODIS greenness (NDVI, 250 m) | greenness and its anomaly against last year |
| E | Open-Meteo weather, ERA5 rain and soil | weather over the horizon, dead-fuel moisture, rain history |
| F | OpenStreetMap roads and rivers, WorldPop people | distance to road / river, population |
| G | MODIS burned-area product (500 m, day of burning) | the truth: which cells burned within 24 or 48 hours |
| H | all of the above | the 47-layer stack and the target |
""")
code("""
import sys, json, warnings
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
sys.path.insert(0, '../src')
warnings.filterwarnings('ignore')
from firespread import config, grid as G, cases, burn_eval, rank, ancillary, rasters
plt.rcParams.update({'figure.dpi': 110, 'axes.titlesize': 8, 'font.size': 8})
import textwrap
from matplotlib.axes import Axes
_st = Axes.set_title
Axes.set_title = lambda self, label, *a, **k: _st(self, textwrap.fill(str(label), 40), *a, **k)   # long titles wrap instead of overlapping

z = np.load('../data/maps_48h.npz')
info = pd.DataFrame({k: z[k] for k in ('case', 'split', 'block', 'year', 'state')})
burned = z['y'].reshape(len(z['y']), -1).sum(1)
pick = info[(info.split == 'test') & (burned > 220) & (burned < 340) & (info.year == 2023)]
CASE = pick.case.iloc[2]; idx = int(np.flatnonzero(z['case'] == CASE)[0])
d = next(p for p in (config.DATA / 'bank' / CASE, config.DATA / 'cases' / CASE) if p.exists())
meta = json.load(open(d / 'meta.json')); env = np.load(d / 'env.npz')
grid = G.Grid(int(env['grid'][0]), *map(float, env['grid'][1:4]), int(env['grid'][4]), int(env['grid'][5]))
c = cases.load_case(d, gap_override=48)
print(f"fire {CASE}: {meta['state']}, start {meta['t0'][:16]} UTC, centre {meta['lat']:.3f} N {meta['lon']:.3f} E, {meta['n_start']} satellite detections at the start")
print(f"grid: {grid.nx} x {grid.ny} cells of {grid.res:.0f} m in UTM zone EPSG:{grid.epsg}; the models see a 192 x 192 crop around the fire")
""")
code("""
def show(ax, a, title, cmap='viridis', vmin=None, vmax=None, mark=True, cbar=True):
    im = ax.imshow(a, cmap=cmap, vmin=vmin, vmax=vmax, interpolation='nearest')
    ax.set_title(title); ax.set_xticks([]); ax.set_yticks([])
    if mark:
        r, cc = np.argwhere(c.seed_mask).mean(axis=0); ax.plot(cc, r, marker='+', color='red', ms=8, mew=1.2)
    if cbar: plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    return im
""")
md("## A. The fire itself: satellite detections")
code("""
cp = pd.read_parquet(config.DATA / 'cases.parquet'); row = cp[(cp.event == meta['event']) & (cp.t0 == pd.Timestamp(meta['t0']))].iloc[0]
det = pd.read_parquet(config.DATA / 'detections' / f"{meta['year']}.parquet")
fire = det[(det.event == meta['event']) & (det.epoch <= int(row.epoch_from))]
fig, ax = plt.subplots(1, 3, figsize=(13, 4))
h = (fire.t - fire.t.min()).dt.total_seconds() / 3600
sc = ax[0].scatter(fire.longitude, fire.latitude, c=h, s=30 + fire.frp * 3, cmap='plasma', edgecolor='k', lw=.4)
ax[0].set_title(f'RAW: {len(fire)} VIIRS detections (colour = hours since first, size = fire power)'); ax[0].set_xlabel('lon'); ax[0].set_ylabel('lat'); plt.colorbar(sc, ax=ax[0])
show(ax[1], c.seed_mask.astype(float), 'INPUT: footprint of the fire at the start (cells within 190 m of a detection)', 'Reds', cbar=False)
show(ax[2], np.minimum(c.dist_cells, 120), 'INPUT: distance to the fire (cells of 100 m)', 'magma_r')
plt.tight_layout(); plt.show()
""")
md("## B. Land cover and forest land")
code("""
lat0, lon0 = meta['lat'], meta['lon']
tile = config.WORLDCOVER_DIR / f"ESA_WorldCover_10m_2021_v200_N{3*int(np.floor(lat0/3)):02d}E{3*int(np.floor(lon0/3)):03d}_Map.tif"
import rasterio
from rasterio.windows import from_bounds
w, s, e, n = grid.bounds_lonlat()
with rasterio.open(tile) as ds:
    raw = ds.read(1, window=from_bounds(w, s, e, n, ds.transform), boundless=True, fill_value=0)
classes = {10: ('tree', '#006400'), 20: ('shrub', '#ffbb22'), 30: ('grass', '#ffff4c'), 40: ('crop', '#f096ff'), 50: ('built-up', '#fa0000'), 60: ('bare', '#b4b4b4'), 80: ('water', '#0064c8'), 90: ('wetland', '#0096a0'), 95: ('mangrove', '#00cf75'), 100: ('moss', '#fae6a0')}
keys = sorted(classes); cm = ListedColormap(['#ffffff'] + [classes[k][1] for k in keys]); idxmap = {0: 0, **{k: i + 1 for i, k in enumerate(keys)}}
rawi = np.vectorize(lambda v: idxmap.get(int(v), 0))(raw[::4, ::4])
fig, ax = plt.subplots(2, 4, figsize=(14, 7))
ax[0, 0].imshow(rawi, cmap=cm, vmin=0, vmax=len(keys), interpolation='nearest'); ax[0, 0].set_title(f'RAW: WorldCover classes, 10 m ({raw.shape[1]} x {raw.shape[0]} pixels, shown 1 in 4)'); ax[0, 0].axis('off')
ax[0, 0].legend(handles=[plt.Rectangle((0, 0), 1, 1, color=classes[k][1]) for k in keys], labels=[classes[k][0] for k in keys], fontsize=6, loc='lower left', ncol=2)
for a, k in zip(ax.ravel()[1:], ['lc_tree', 'lc_shrub', 'lc_grass', 'lc_crop', 'lc_built', 'lc_water']): show(a, env[k], f'INPUT: {k[3:]} share per 100 m cell', 'YlGn' if k != 'lc_water' else 'Blues', 0, 1)
show(ax[1, 3], env['forest_share'], 'INPUT: forest-land share (boundary-defined forest mask, not tree cover)', 'Greens', 0, 1)
plt.tight_layout(); plt.show()
""")
md("## C. Terrain")
code("""
dem = env['dem']; rel = dem - np.nanmean(dem[c.seed_mask])
fig, ax = plt.subplots(1, 4, figsize=(15, 3.6))
show(ax[0], dem, 'INPUT: elevation (m), Copernicus 30 m averaged to 100 m', 'terrain')
show(ax[1], rel, 'INPUT: elevation relative to the fire (m)', 'RdBu_r', -np.nanmax(np.abs(rel)), np.nanmax(np.abs(rel)))
show(ax[2], env['slope'], 'INPUT: slope (degrees)', 'YlOrBr', 0, 35)
show(ax[3], env['upslope'], 'uphill direction (degrees); turned into slope-along-the-line-from-the-fire', 'twilight', 0, 360)
plt.tight_layout(); plt.show()
""")
md("## D. Greenness (MODIS NDVI) and dryness of the vegetation")
code("""
nd = np.load(d / 'ndvi.npz') if (d / 'ndvi.npz').exists() else None
fig, ax = plt.subplots(1, 3, figsize=(12, 3.6))
if nd is not None:
    show(ax[0], nd['ndvi'], 'RAW-ish: NDVI now (latest 16-day composite before the start, 250 m)', 'YlGn', 0, 0.9)
    show(ax[1], nd['ndvi_prev'], 'the same composite a year earlier', 'YlGn', 0, 0.9)
    a = nd['ndvi'] - nd['ndvi_prev']; show(ax[2], a, 'INPUT: greenness anomaly = now minus last year (negative = drier than usual)', 'BrBG', -0.4, 0.4)
plt.tight_layout(); plt.show()
""")
md("## E. Weather, dead-fuel moisture, rain history and soil")
code("""
wx = pd.read_csv(d / 'weather.csv', parse_dates=['time']); H = 48
fig, ax = plt.subplots(1, 5, figsize=(17, 3.2))
for src, st in (('hist', '-'), ('era5', ':')):
    w_ = wx[(wx.source == src) & (wx.hours_from_t0.between(-24, H + 6))]
    ax[0].plot(w_.hours_from_t0, w_.wind_speed_10m, st, label=src); ax[1].plot(w_.hours_from_t0, w_.temperature_2m, st); ax[2].plot(w_.hours_from_t0, w_.relative_humidity_2m, st); ax[3].plot(w_.hours_from_t0, w_.precipitation, st)
hrs = np.arange(len(c.moisture)); ax[4].plot(hrs, c.moisture, 'k'); 
for a, t_ in zip(ax, ['RAW: wind speed 10 m (m/s)', 'RAW: temperature (C)', 'RAW: relative humidity (%)', 'RAW: rain (mm/h)', 'INPUT: dead-fuel moisture computed from them']): a.set_title(t_); a.set_xlabel('hours from the start'); a.axvline(0, color='r', lw=.6)
ax[0].legend(); plt.tight_layout(); plt.show()
ant = json.load(open(d / 'ant.json')) if (d / 'ant.json').exists() else None
antt = pd.read_parquet(config.DATA / 'antecedent.parquet').set_index('case'); sl = pd.read_parquet(config.DATA / 'soil.parquet').set_index('case')
print('INPUTS (one number per fire, painted on the whole map):'); print(pd.concat([antt.loc[CASE], sl.loc[CASE]]).round(2).to_string())
print('wind is computed but NOT given to the final models: it added nothing in every test.')
""")
md("## F. Roads, rivers and people")
code("""
tiles = ancillary.tiles_of(grid)
lines = ancillary.load_lines(config.DATA / 'osm_lines.geojsonseq', tiles)
from pyproj import Transformer
tf = Transformer.from_crs(4326, grid.epsg, always_xy=True)
fig, ax = plt.subplots(1, 4, figsize=(16, 3.8))
for kind, col in (('road', 'k'), ('river', 'tab:blue')):
    for k, xy in lines:
        if k == kind:
            x, y = tf.transform(xy[:, 0], xy[:, 1]); ax[0].plot((np.array(x) - grid.x0) / grid.res, (grid.y0 - np.array(y)) / grid.res, col, lw=.6)
ax[0].set_xlim(0, grid.nx); ax[0].set_ylim(grid.ny, 0); ax[0].set_title('RAW: OpenStreetMap roads/tracks (black) and rivers (blue)'); ax[0].set_aspect('equal')
osm = np.load(d / 'osm.npz'); pop = np.load(d / 'pop.npz')['pop']
show(ax[1], osm['road_dist'], 'INPUT: distance to the nearest road or track (cells)', 'viridis', 0, 100)
show(ax[2], osm['river_dist'], 'INPUT: distance to the nearest river (cells)', 'viridis', 0, 100)
show(ax[3], pop, 'INPUT: people per km2 (WorldPop 1 km)', 'Oranges')
plt.tight_layout(); plt.show()
""")
md("## G. The truth: which cells burned")
code("""
b = np.load(d / 'burn_long.npz'); doy = b['doy'].astype(float); t0 = pd.Timestamp(meta['t0']); d0 = t0.timetuple().tm_yday
c48 = __import__('dataclasses').replace(c, meta={**c.meta, 't1': str(t0 + pd.Timedelta(hours=48))})
truth, cand, ok = burn_eval.burn_masks(c48, b['doy'], int(b['year'])); truth_c, cand_c = burn_eval.connected_truth(c48, truth, cand)
fig, ax = plt.subplots(1, 4, figsize=(16, 3.8))
dd = np.where(doy > 0, doy - d0, np.nan); show(ax[0], dd, 'RAW: MODIS burn date (days after the start day; 500 m blocks)', 'turbo', -30, 30)
show(ax[1], cand_c.astype(float), 'cells the models are asked about: 5 to 80 cells from the start, mapped, not burned earlier', 'Greys', cbar=False)
show(ax[2], truth.astype(float), 'burned within 48 h (all patches in the square)', 'Reds', cbar=False)
show(ax[3], truth_c.astype(float), 'TARGET: only the patches touching this fire (what the models learn)', 'Reds', cbar=False)
plt.tight_layout(); plt.show()
print(f'burned cells in the target: {int(truth_c.sum())} of {int(cand_c.sum())} asked about ({100*truth_c.sum()/cand_c.sum():.1f} %)')
""")
md("## H. Everything the models see: the 47 layers")
code("""
names = list(z['names']); X = z['X'][idx].astype(np.float32)
fig, axs = plt.subplots(7, 7, figsize=(18, 18))
for i, (a, nm) in enumerate(zip(axs.ravel(), names)):
    v = X[i]
    if np.nanstd(v) < 1e-6: a.text(.5, .5, f'constant\\n{np.nanmean(v):.3g}', ha='center', va='center'); a.set_title(nm); a.axis('off'); continue
    a.imshow(v, cmap='viridis', interpolation='nearest'); a.set_title(nm); a.set_xticks([]); a.set_yticks([])
for a in axs.ravel()[len(names):]: a.axis('off')
plt.suptitle(f'All {len(names)} input layers for fire {CASE} (192 x 192 crop). Constant layers are numbers that apply to the whole fire (weather, fire size, season, rain history ...).', y=.995)
plt.tight_layout(); plt.show()
print('target (burned within 48 h) and mask of cells asked about, same crop:', z['y'][idx].sum(), 'burned cells,', z['cand'][idx].sum(), 'asked about')
""")
md("""## What the models receive

* **Input:** a 47 x 192 x 192 array per fire (the layers above), each layer standardised with the training fires' mean and spread; missing values (for example, no cloud-free greenness) become 0.
* **Target:** the 192 x 192 map of cells that burned within 24 or 48 hours (section G).
* **Scored only on** the cells asked about, with the burned patches that touch the fire.
* **Tree models** see the same layers one cell at a time; **image models** see the whole square at once.
* Wind is deliberately left out of the final models, because it added nothing in any test.
""")
nb = nbf.v4.new_notebook(); nb.cells = C; nb.metadata['kernelspec'] = {'name': 'firespread', 'display_name': 'fire-spread', 'language': 'python'}
out = Path(__file__).resolve().parents[1] / 'notebooks' / '01_raw_to_inputs.ipynb'
ExecutePreprocessor(timeout=1800, kernel_name='firespread').preprocess(nb, {'metadata': {'path': str(out.parent)}})
nbf.write(nb, out); print('wrote', out)
