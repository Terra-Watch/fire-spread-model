"""Where things are. Existing raw data stays where it is (read only); everything this project makes goes under data/."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REPO = ROOT.parent
V4 = REPO / "wildfire-dataset"
RAW = V4 / "inputs_raw"
DATA = ROOT / "data"

FIRMS_DIRS = (RAW / "firms_main_south", RAW / "firms_hpn", RAW / "firms_ne")
FOREST_MASK = V4 / "dataset" / "forest_union_0p001.npy"          # uint8, 0.001 degree pixels, top-left corner 35.0 N, 68.0 E
FOREST_TOP, FOREST_LEFT, FOREST_STEP = 35.0, 68.0, 0.001
WORLDCOVER_DIR = V4 / "_tiles"                                   # ESA WorldCover 10 m 2021 v200, 3 degree tiles
ADM1 = RAW / "geoBoundaries_IND_ADM1.geojson"
DEM_URL = "https://copernicus-dem-30m.s3.amazonaws.com/Copernicus_DSM_COG_10_{ns}{lat:02d}_00_{ew}{lon:03d}_00_DEM/Copernicus_DSM_COG_10_{ns}{lat:02d}_00_{ew}{lon:03d}_00_DEM.tif"

# Event rules (stated here, not tuned after looking at results)
LINK_KM = 1.0            # detections closer than this in space ...
LINK_GAP_H = 36.0        # ... and in time belong to one fire event
EPOCH_MERGE_H = 3.0      # passes closer than this in time (two satellites, adjacent swaths) are one observation
GROWTH_TOL_KM = 0.75     # a detection farther than this from everything seen earlier is new growth
MIN_FOREST_SHARE = 0.5   # share of the 3 x 3 forest-mask pixels (~330 m) around a detection that must be forest land
