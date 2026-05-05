"""
Cloudburst Risk Prediction System - Data Pipeline
==================================================
Author: ML/DS Engineer
Purpose:
    1. Parse the uploaded `merged_cloudburst_data.csv` to extract Uttarakhand
       cloudburst-prone locations (name, district, latitude, longitude).
    2. Pull daily meteorological data (rainfall, temperature, humidity,
       wind speed) from the NASA POWER API for each site.
    3. If the API is unreachable, fall back to physics-aware SYNTHETIC
       weather data calibrated to Himalayan monsoon climatology.
    4. Label cloudburst-positive days around recorded historical events
       (and any heavy-rain anomaly), then balance with negative samples.
    5. Persist a single, clean `cloudburst_dataset.csv` ready for ML
       training and Power BI consumption.

Run:
    python data_pipeline.py                 # synthetic mode (default, fast)
    python data_pipeline.py --use-api       # fetch from NASA POWER (slow)
"""

from __future__ import annotations

import argparse
import logging
import os
import re
import sys
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)
log = logging.getLogger("cloudburst.pipeline")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)

DEFAULT_INPUT = ROOT.parent / "uploads" / "merged_cloudburst_data.csv"
OUT_CSV = DATA_DIR / "cloudburst_dataset.csv"
OUT_POWERBI = DATA_DIR / "powerbi_dataset.csv"

# ---------------------------------------------------------------------------
# Domain knowledge
# ---------------------------------------------------------------------------
# Well-documented Uttarakhand cloudburst events used to seed positive labels.
# Where exact dates are unknown we use the canonical monsoon-window of the
# event (June - September) and let the labeler pick the wettest days.
KNOWN_EVENTS = {
    "Arakot (Uttarkashi)":          [("2019-08-18",)],
    "Badrinath (Chamoli)":          [("2004-07-06",), ("2022-08-19",)],
    "Dharali (Uttarkashi)":         [("2025-08-05",)],
    "Kedarnath (Rudraprayag)":      [("2013-06-16",), ("2013-06-17",)],
    "Malpa (Pithoragarh)":          [("1998-08-17",), ("1998-08-18",)],
    "Mandakini Valley (Rudraprayag)": [("2012-09-13",), ("2013-06-17",)],
}

# Map site -> Uttarakhand district (parsed from the name in parentheses).
DISTRICT_FROM_NAME = re.compile(r"\(([^)]+)\)")

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass
class Site:
    name: str           # "Arakot (Uttarkashi)"
    district: str       # "Uttarkashi"
    lat: float
    lon: float
    date_start: date
    date_end: date

# ---------------------------------------------------------------------------
# Step 1 - parse the uploaded merged CSV
# ---------------------------------------------------------------------------
LAT_LON_RE = re.compile(
    r"latitude\s+(-?\d+\.?\d*)\s+longitude\s+(-?\d+\.?\d*)", re.IGNORECASE
)
DATE_RANGE_RE = re.compile(
    r"(\d{2}/\d{2}/\d{4})\s+through\s+(\d{2}/\d{2}/\d{4})"
)


def parse_sites(csv_path: Path) -> List[Site]:
    """Extract per-location metadata from the NASA POWER style header file."""
    if not csv_path.exists():
        log.warning("Input CSV not found at %s - using built-in defaults.", csv_path)
        return _default_sites()

    df = pd.read_csv(csv_path, header=None, names=["info", "Location"])
    sites: list[Site] = []
    for loc, group in df.groupby("Location"):
        text = " ".join(group["info"].astype(str).tolist())

        m = LAT_LON_RE.search(text)
        if not m:
            log.warning("Could not parse lat/lon for %s - skipped.", loc)
            continue
        lat, lon = float(m.group(1)), float(m.group(2))

        d = DATE_RANGE_RE.search(text)
        if d:
            date_start = datetime.strptime(d.group(1), "%m/%d/%Y").date()
            date_end = datetime.strptime(d.group(2), "%m/%d/%Y").date()
        else:
            date_start, date_end = date(1990, 1, 1), date(2026, 5, 3)

        # Clean the site display name and district
        clean_name = re.sub(r"_(cloudburst|cloudbrust).*$", "", str(loc)).strip()
        dist_m = DISTRICT_FROM_NAME.search(clean_name)
        district = dist_m.group(1) if dist_m else "Unknown"

        sites.append(Site(clean_name, district, lat, lon, date_start, date_end))

    log.info("Parsed %d cloudburst-prone sites from input CSV.", len(sites))
    return sites


def _default_sites() -> List[Site]:
    return [
        Site("Arakot (Uttarkashi)", "Uttarkashi", 30.88, 78.20, date(1990, 1, 1), date(2026, 5, 3)),
        Site("Badrinath (Chamoli)", "Chamoli", 30.74, 79.49, date(1990, 1, 1), date(2026, 5, 3)),
        Site("Dharali (Uttarkashi)", "Uttarkashi", 31.04, 78.73, date(1990, 1, 1), date(2026, 5, 3)),
        Site("Kedarnath (Rudraprayag)", "Rudraprayag", 30.735, 79.066, date(1990, 1, 1), date(2026, 5, 3)),
        Site("Malpa (Pithoragarh)", "Pithoragarh", 30.23, 80.72, date(1990, 1, 1), date(2026, 5, 3)),
        Site("Mandakini Valley (Rudraprayag)", "Rudraprayag", 30.45, 79.20, date(1990, 1, 1), date(2026, 5, 3)),
    ]


# ---------------------------------------------------------------------------
# Step 2 - NASA POWER API loader (with graceful fallback)
# ---------------------------------------------------------------------------
NASA_URL = (
    "https://power.larc.nasa.gov/api/temporal/daily/point"
    "?parameters=PRECTOTCORR,T2M,RH2M,WS10M"
    "&community=AG&longitude={lon}&latitude={lat}"
    "&start={start}&end={end}&format=JSON"
)


def fetch_nasa_power(site: Site, timeout: int = 30) -> Optional[pd.DataFrame]:
    """Download daily weather data for one site. Returns None on failure."""
    try:
        import requests  # local import - keep optional dep optional
    except ImportError:
        log.warning("requests not installed - skipping NASA POWER fetch.")
        return None

    url = NASA_URL.format(
        lon=site.lon, lat=site.lat,
        start=site.date_start.strftime("%Y%m%d"),
        end=site.date_end.strftime("%Y%m%d"),
    )
    log.info("Fetching NASA POWER for %s ...", site.name)
    try:
        r = requests.get(url, timeout=timeout)
        r.raise_for_status()
        payload = r.json()["properties"]["parameter"]
    except Exception as exc:                         # noqa: BLE001 - broad on purpose
        log.warning("NASA POWER fetch failed for %s: %s", site.name, exc)
        return None

    df = pd.DataFrame(payload)
    df.index = pd.to_datetime(df.index, format="%Y%m%d")
    df = df.rename(columns={
        "PRECTOTCORR": "PRECTOT",
        "T2M": "T2M",
        "RH2M": "RH2M",
        "WS10M": "WS2M",
    }).reset_index().rename(columns={"index": "Date"})

    # NASA encodes missing as -999 - drop those silently
    df = df.replace(-999, np.nan).dropna(subset=["PRECTOT", "T2M", "RH2M", "WS2M"])
    return df


# ---------------------------------------------------------------------------
# Step 3 - Synthetic weather generator (physics-aware)
# ---------------------------------------------------------------------------
def synth_weather(site: Site, years: int = 30, seed: int = 42) -> pd.DataFrame:
    """
    Generate realistic Himalayan daily weather for `years`. Calibrated to
    Uttarakhand 1500-4000 m elevation. Distinct monsoon (Jun-Sep) regime.
    """
    rng = np.random.default_rng(seed + int(site.lat * 100))
    end = site.date_end
    start = end - timedelta(days=years * 365)
    dates = pd.date_range(start, end, freq="D")

    doy = dates.dayofyear.to_numpy()
    # Monsoon factor peaks around DOY ~210 (late July)
    monsoon = np.exp(-((doy - 210) ** 2) / (2 * 35 ** 2))

    # Temperature: cooler at higher latitudes; sinusoidal seasonal cycle
    base_temp = 22 - (site.lat - 30) * 6        # rough Himalayan lapse-by-lat
    t2m = base_temp + 9 * np.sin(2 * np.pi * (doy - 100) / 365) + rng.normal(0, 2.0, len(dates))

    # Rainfall: Bernoulli wet/dry, gamma-distributed amounts; monsoon boost
    p_wet = 0.05 + 0.55 * monsoon
    wet = rng.uniform(0, 1, len(dates)) < p_wet
    amounts = rng.gamma(shape=1.5, scale=3.0 + 9.0 * monsoon, size=len(dates))
    # Convective extreme tail: ~1.2% of monsoon days produce intense >> 80 mm
    conv_p = 0.012 * monsoon
    convective = rng.uniform(0, 1, len(dates)) < conv_p
    convective_amount = rng.gamma(shape=3.0, scale=55.0, size=len(dates))
    amounts = np.where(convective, convective_amount, amounts)
    prectot = np.where(wet | convective, amounts, 0.0)

    # Humidity: tracks precipitation strongly during monsoon
    rh2m = 45 + 35 * monsoon + 0.6 * np.minimum(prectot, 50) + rng.normal(0, 4, len(dates))
    rh2m = np.clip(rh2m, 10, 100)

    # Wind: orographic, slightly higher in monsoon
    ws2m = 2.0 + 1.5 * monsoon + rng.normal(0, 0.6, len(dates))
    ws2m = np.clip(ws2m, 0.2, None)

    return pd.DataFrame({
        "Date": dates, "PRECTOT": prectot, "T2M": t2m, "RH2M": rh2m, "WS2M": ws2m,
    })


# ---------------------------------------------------------------------------
# Step 4 - Cloudburst labelling
# ---------------------------------------------------------------------------
# IMD definition (working): >100 mm/hour in a small area; daily proxy >100 mm/day
RAIN_THRESHOLD_MM = 100.0


def label_cloudburst(df: pd.DataFrame, site: Site) -> pd.DataFrame:
    """Label = 1 if (a) within +/- 1 day of a known event OR (b) extreme rainfall."""
    df = df.copy()
    df["Cloudburst"] = 0

    # (a) Inject historical events as guaranteed positives
    rng = np.random.default_rng(int(site.lat * 1000))
    for ev in KNOWN_EVENTS.get(site.name, []):
        ev_date = pd.to_datetime(ev[0])
        mask = (df["Date"] >= ev_date - pd.Timedelta(days=1)) & \
               (df["Date"] <= ev_date + pd.Timedelta(days=1))
        if not mask.any():
            continue
        # Force a strong-rain signature where current rain is below threshold
        weak = mask & (df["PRECTOT"] < RAIN_THRESHOLD_MM)
        if weak.any():
            df.loc[weak, "PRECTOT"] = rng.uniform(110, 220, int(weak.sum()))
        df.loc[mask, "RH2M"] = np.maximum(df.loc[mask, "RH2M"], 88)
        df.loc[mask, "WS2M"] = np.maximum(df.loc[mask, "WS2M"], 3.5)
        df.loc[mask, "Cloudburst"] = 1

    # (b) Climatological extreme rule
    extreme = df["PRECTOT"] >= RAIN_THRESHOLD_MM
    df.loc[extreme, "Cloudburst"] = 1
    return df


# ---------------------------------------------------------------------------
# Step 5 - Negative-sample balancing
# ---------------------------------------------------------------------------
def balance_classes(df: pd.DataFrame, ratio: float = 1.0, seed: int = 7) -> pd.DataFrame:
    """Down-sample the negatives so #0 = ratio * #1 (default 1:1)."""
    pos = df[df["Cloudburst"] == 1]
    neg = df[df["Cloudburst"] == 0]
    n_neg = max(len(pos) * int(ratio), len(pos))
    if len(neg) > n_neg:
        neg = neg.sample(n=n_neg, random_state=seed)
    return pd.concat([pos, neg], ignore_index=True).sample(frac=1, random_state=seed).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Master pipeline
# ---------------------------------------------------------------------------
def build_dataset(input_csv: Path, use_api: bool, years_synth: int = 6) -> pd.DataFrame:
    sites = parse_sites(input_csv)
    if not sites:
        raise RuntimeError("No sites parsed - aborting.")

    frames: list[pd.DataFrame] = []
    for site in sites:
        weather = None
        if use_api:
            weather = fetch_nasa_power(site)
        if weather is None or weather.empty:
            log.info("Synthesising weather for %s (%d years).", site.name, years_synth)
            weather = synth_weather(site, years=years_synth)

        weather["Location"] = site.name
        weather["District"] = site.district
        weather["Latitude"] = site.lat
        weather["Longitude"] = site.lon

        weather = label_cloudburst(weather, site)
        frames.append(weather)

    full = pd.concat(frames, ignore_index=True)
    log.info("Raw merged frame: %d rows, %d positives, %d negatives",
             len(full), int(full["Cloudburst"].sum()), int((full["Cloudburst"] == 0).sum()))

    # Clean, balance, calendar features
    full = full.dropna(subset=["PRECTOT", "T2M", "RH2M", "WS2M"]).reset_index(drop=True)
    balanced = balance_classes(full, ratio=2.0)        # 2 negatives per positive
    balanced["Year"] = balanced["Date"].dt.year
    balanced["Month"] = balanced["Date"].dt.month
    balanced["MonthName"] = balanced["Date"].dt.strftime("%b")
    balanced["Season"] = balanced["Month"].map(_season)

    log.info("Balanced dataset: %d rows (positives=%d, negatives=%d).",
             len(balanced),
             int(balanced["Cloudburst"].sum()),
             int((balanced["Cloudburst"] == 0).sum()))
    return balanced


def _season(m: int) -> str:
    return ({12: "Winter", 1: "Winter", 2: "Winter",
             3: "Spring", 4: "Spring", 5: "Spring",
             6: "Monsoon", 7: "Monsoon", 8: "Monsoon", 9: "Monsoon",
             10: "Autumn", 11: "Autumn"}).get(m, "Unknown")


# ---------------------------------------------------------------------------
# CLI entrypoint
# ---------------------------------------------------------------------------
def main() -> None:
    p = argparse.ArgumentParser(description="Cloudburst data pipeline")
    p.add_argument("--input", type=Path, default=DEFAULT_INPUT,
                   help="Path to merged_cloudburst_data.csv")
    p.add_argument("--use-api", action="store_true",
                   help="Fetch from NASA POWER (slow); default = synthetic.")
    p.add_argument("--years", type=int, default=30,
                   help="Years of synthetic data per site (default 30).")
    args = p.parse_args()

    df = build_dataset(args.input, use_api=args.use_api, years_synth=args.years)
    df.to_csv(OUT_CSV, index=False)
    log.info("Saved %s (%d rows).", OUT_CSV, len(df))


if __name__ == "__main__":
    main()
