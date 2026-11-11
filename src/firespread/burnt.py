"""Burn date from the MODIS burned-area product (MCD64A1 v061, 500 m, one file per tile and month) read from the Planetary Computer's public archive.
Independent of the VIIRS detections used to start the forecasts, area-based, and dated to the day: a finer truth than 'a new detection appeared'."""
import datetime as dt

import httpx
import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import reproject

from .grid import Grid

STAC = "https://planetarycomputer.microsoft.com/api/stac/v1/search"
SIGN = "https://planetarycomputer.microsoft.com/api/sas/v1/sign"
UNMAPPED = -3                    # no value written for this cell (outside every tile that answered)
_TOKENS = {}                     # (storage account, container) -> (token, expiry as epoch seconds); one token serves every file of the container
TOKEN_URL = "https://planetarycomputer.microsoft.com/api/sas/v1/token/{acct}/{cont}"


def _token(client, acct, cont):
    import time
    from datetime import datetime
    hit = _TOKENS.get((acct, cont))
    if hit and hit[1] - time.time() > 300:
        return hit[0]
    r = client.get(TOKEN_URL.format(acct=acct, cont=cont))
    if r.status_code != 200 or "token" not in r.json():
        return None
    j = r.json(); exp = datetime.fromisoformat(j.get("msft:expiry", "2000-01-01T00:00:00Z").replace("Z", "+00:00")).timestamp()
    _TOKENS[(acct, cont)] = (j["token"], exp)
    return j["token"]


def _signed(client, href, tries=6):
    """Signed URL. First choice: one container token per hour (a handful of requests in total); the per-file sign call was refused (rate limit) when many workers ran, which left most cases without burn history, Sentinel-2 and part of the NDVI."""
    import time
    from urllib.parse import urlparse
    u = urlparse(href); acct = u.netloc.split(".")[0]; cont = u.path.strip("/").split("/")[0]
    try:
        tok = _token(client, acct, cont)
        if tok:
            return href + "?" + tok
    except Exception:
        pass
    for k in range(tries):
        r = client.get(SIGN, params={"href": href})
        if r.status_code == 200 and "href" in r.json():
            return r.json()["href"]
        time.sleep(2 ** k)
    raise RuntimeError(f"could not sign {href}: HTTP {r.status_code}")


def _months(d0: dt.date, d1: dt.date):
    y, m = d0.year, d0.month
    while (y, m) <= (d1.year, d1.month):
        yield y, m
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)


def burn_doy(grid: Grid, d0: dt.date, d1: dt.date, client: httpx.Client | None = None) -> np.ndarray:
    """Day of year of burning on the grid for the months from d0 to d1: 0 not burned, 1..366 burned that day, -1 not mapped (too little data), -2 water, UNMAPPED where no tile answered."""
    own = client is None
    client = client or httpx.Client(timeout=60)
    out = np.full(grid.shape, UNMAPPED, np.int16)
    w, s, e, n = grid.bounds_lonlat(pad_deg=0.01)
    try:
        for y, m in _months(d0, d1):
            r = client.post(STAC, json={"collections": ["modis-64A1-061"], "bbox": [w, s, e, n], "datetime": f"{y}-{m:02d}-01T00:00:00Z/{y}-{m:02d}-28T00:00:00Z", "limit": 8})
            for f in r.json().get("features", []):
                url = _signed(client, f["assets"]["Burn_Date"]["href"])
                with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR", CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif", GDAL_HTTP_MAX_RETRY="3"), rasterio.open(url) as ds:
                    tmp = np.full(grid.shape, UNMAPPED, np.int16)
                    reproject(rasterio.band(ds, 1), tmp, dst_transform=grid.transform, dst_crs=f"EPSG:{grid.epsg}", resampling=Resampling.nearest, dst_nodata=UNMAPPED)
                # a month's file says 0 for cells that did not burn that month: keep a real burn date from any month
                out = np.where(tmp > 0, tmp, np.where(out == UNMAPPED, tmp, out))
    finally:
        if own:
            client.close()
    return out


def doy_to_date(year: int, doy):
    return dt.date(year, 1, 1) + dt.timedelta(days=int(doy) - 1)
