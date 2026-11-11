"""VIIRS 375 m active-fire detections (NASA FIRMS standard processing, Suomi-NPP and NOAA-20) from the CSVs already on disk."""
import re
from pathlib import Path

import numpy as np
import pandas as pd

from . import config

COLS = ["latitude", "longitude", "acq_date", "acq_time", "satellite", "confidence", "frp", "daynight", "type"]
_DATE = re.compile(r"(\d{4}-\d{2}-\d{2})")


def list_files(year: int, dirs=None) -> list[Path]:
    """FIRMS files whose start date falls in `year` (each file covers 5 days; the 5 days that run into the next year are included by the date filter later)."""
    out = []
    for d in (dirs or config.FIRMS_DIRS):
        for f in sorted(Path(d).glob("*.csv")):
            m = _DATE.search(f.name)
            if m and m.group(1).startswith(str(year)):
                out.append(f)
    return out


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Vegetation fires only (type 0), nominal or high confidence, one row per detection (the regional exports overlap), with a UTC time."""
    df = df[(df["type"] == 0) & df["confidence"].isin(["n", "h", "nominal", "high"])].copy()
    hhmm = df["acq_time"].astype(int)
    df["t"] = pd.to_datetime(df["acq_date"], utc=True) + pd.to_timedelta(hhmm // 100, unit="h") + pd.to_timedelta(hhmm % 100, unit="m")
    df["sat"] = df["satellite"].astype(str).str.upper().map(lambda s: "N20" if s.startswith("N20") or s == "1" else "SNPP")
    df = df.drop_duplicates(["latitude", "longitude", "t", "sat"])
    return df[["latitude", "longitude", "t", "sat", "confidence", "frp", "daynight"]].reset_index(drop=True)


def load_year(year: int, dirs=None) -> pd.DataFrame:
    frames = [pd.read_csv(f, usecols=COLS, dtype={"confidence": "string", "satellite": "string"}) for f in list_files(year, dirs)]
    if not frames:
        return pd.DataFrame(columns=["latitude", "longitude", "t", "sat", "confidence", "frp", "daynight"])
    df = clean(pd.concat(frames, ignore_index=True))
    return df[df["t"].dt.year == year].reset_index(drop=True)
