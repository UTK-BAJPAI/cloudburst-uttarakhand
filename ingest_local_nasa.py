"""
Ingest per-location NASA POWER CSV exports from a local folder
and produce a unified, labelled cloudburst dataset.

Usage:
    python ingest_local_nasa.py --local-dir "C:\\Users\\ACER\\Downloads\\ML\\cloudburst_project\\6_location"
"""
from __future__ import annotations

import argparse
import logging
import re
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

# Re-use the same labelling + balancing logic from data_pipeline.py
from data_pipeline import (
    KNOWN_EVENTS, RAIN_THRESHOLD_MM, Site,
    label_cloudburst, balance_classes, _season,
)

logging.basicConfig(format="%(asctime)s | %(levelname)-7s | %(message)s",
                    datefmt="%H:%M:%S", level=logging.INFO)
log = logging.getLogger("cloudburst.ingest")

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"; DATA_DIR.mkdir(exist_ok=True)
OUT_CSV = DATA_DIR / "cloudburst_dataset.csv"

# ---- 6 known Uttarakhand hotspots with their NASA POWER coordinates --------
SITE_REGISTRY = [
    Site("Arakot (Uttarkashi)",            "Uttarkashi",  30.88,  78.20,  date(1990,1,1), date(2026,5,3)),
    Site("Badrinath (Chamoli)",            "Chamoli",     30.74,  79.49,  date(1990,1,1), date(2026,5,3)),
    Site("Dharali (Uttarkashi)",           "Uttarkashi",  31.04,  78.73,  date(1990,1,1), date(2026,5,3)),
    Site("Kedarnath (Rudraprayag)",        "Rudraprayag", 30.735, 79.066, date(1990,1,1), date(2026,5,3)),
    Site("Malpa (Pithoragarh)",            "Pithoragarh", 30.23,  80.72,  date(1990,1,1), date(2026,5,3)),
    Site("Mandakini Valley (Rudraprayag)", "Rudraprayag", 30.45,  79.20,  date(1990,1,1), date(2026,5,3)),
]

LAT_LON_RE = re.compile(r"latitude\s+(-?\d+\.?\d*)\s+longitude\s+(-?\d+\.?\d*)", re.I)
END_HEADER = "-END HEADER-"


def parse_one_file(path: Path):
    """Parse a single NASA POWER per-location CSV export."""
    text_lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()

    # 1. Find lat/lon and end-of-header position
    lat = lon = None
    end_idx = None
    for i, line in enumerate(text_lines):
        m = LAT_LON_RE.search(line)
        if m and lat is None:
            lat, lon = float(m.group(1)), float(m.group(2))
        if END_HEADER in line:
            end_idx = i
            break
    if lat is None or end_idx is None:
        log.warning("Could not parse header in %s - skipped.", path.name)
        return None

    # 2. Match this lat/lon to one of the 6 registered sites
    site = min(SITE_REGISTRY,
               key=lambda s: (s.lat - lat) ** 2 + (s.lon - lon) ** 2)
    distance = ((site.lat - lat) ** 2 + (site.lon - lon) ** 2) ** 0.5
    if distance > 0.5:
        log.warning("File %s lat/lon (%.2f,%.2f) doesn't match any registered site (closest=%s).",
                    path.name, lat, lon, site.name)
        return None

    # 3. Read the data rows (everything after -END HEADER-)
    df = pd.read_csv(path, skiprows=end_idx + 1)

    # Normalise column names
    rename_map = {"PRECTOTCORR": "PRECTOT", "WS10M": "WS2M",
                  "MM": "MO", "DD": "DY"}
    df = df.rename(columns=rename_map)

    # Derive T2M from T2M_MIN / T2M_MAX if T2M itself wasn't downloaded
    if "T2M" not in df.columns and {"T2M_MIN", "T2M_MAX"}.issubset(df.columns):
        df["T2M"] = (df["T2M_MIN"].replace(-999, np.nan) +
                     df["T2M_MAX"].replace(-999, np.nan)) / 2.0
        log.info("  -> derived T2M from T2M_MIN/T2M_MAX")

    # Use IMERG_PRECTOT as fallback if PRECTOTCORR was not exported
    if "PRECTOT" not in df.columns and "IMERG_PRECTOT" in df.columns:
        df["PRECTOT"] = df["IMERG_PRECTOT"]
        log.info("  -> used IMERG_PRECTOT as PRECTOT")

    needed = {"YEAR", "MO", "DY", "PRECTOT", "T2M", "RH2M", "WS2M"}
    missing = needed - set(df.columns)
    if missing:
        log.warning("File %s is missing columns %s - skipped.", path.name, missing)
        return None

    # 4. Build Date and clean -999 sentinels
    df["Date"] = pd.to_datetime(dict(year=df.YEAR, month=df.MO, day=df.DY),
                                errors="coerce")
    df = df.dropna(subset=["Date"])
    df = df.replace(-999, np.nan).dropna(subset=["PRECTOT", "T2M", "RH2M", "WS2M"])
    df = df[["Date", "PRECTOT", "T2M", "RH2M", "WS2M"]].reset_index(drop=True)

    log.info("Parsed %s -> site=%s, rows=%d", path.name, site.name, len(df))
    return site, df


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--local-dir", required=True,
                   help=r"Folder containing per-location NASA POWER CSVs")
    args = p.parse_args()

    local_dir = Path(args.local_dir)
    if not local_dir.exists():
        raise SystemExit(f"Folder not found: {local_dir}")

    # Case-insensitive .csv match without double-counting on Windows
    csvs = sorted({p.resolve() for p in local_dir.glob("*")
                   if p.suffix.lower() == ".csv"})
    if not csvs:
        raise SystemExit(f"No CSV files found in {local_dir}")
    log.info("Found %d CSV files in %s", len(csvs), local_dir)

    frames = []
    for f in csvs:
        result = parse_one_file(f)
        if result is None:
            continue
        site, weather = result
        weather["Location"] = site.name
        weather["District"] = site.district
        weather["Latitude"] = site.lat
        weather["Longitude"] = site.lon
        weather = label_cloudburst(weather, site)
        frames.append(weather)

    if not frames:
        raise SystemExit("No usable data parsed - aborting.")

    full = pd.concat(frames, ignore_index=True)
    log.info("Raw merged frame: %d rows, %d positives, %d negatives",
             len(full), int(full.Cloudburst.sum()),
             int((full.Cloudburst == 0).sum()))

    full = full.dropna(subset=["PRECTOT", "T2M", "RH2M", "WS2M"]).reset_index(drop=True)
    balanced = balance_classes(full, ratio=2.0)
    balanced["Year"] = balanced["Date"].dt.year
    balanced["Month"] = balanced["Date"].dt.month
    balanced["MonthName"] = balanced["Date"].dt.strftime("%b")
    balanced["Season"] = balanced["Month"].map(_season)

    log.info("Balanced dataset: %d rows (positives=%d, negatives=%d)",
             len(balanced), int(balanced.Cloudburst.sum()),
             int((balanced.Cloudburst == 0).sum()))

    balanced.to_csv(OUT_CSV, index=False)
    log.info("Saved %s", OUT_CSV)


if __name__ == "__main__":
    main()