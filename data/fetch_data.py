"""Download the public data used by the models.

- CAMMESA: 5-minute SADI demand from the public demand API (only the most
  recent ~6 months are served). Resampled to hourly.
- Energy-Charts (Fraunhofer ISE, CC BY 4.0, source SMARD/Bundesnetzagentur):
  DE-LU day-ahead prices for the international benchmark.

Usage:
    python data/fetch_data.py               # both
    python data/fetch_data.py --only cammesa
"""
from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path

import pandas as pd
import requests

RAW = Path(__file__).parent / "raw"
CAMMESA_URL = (
    "https://api.cammesa.com/demanda-svc/demanda/"
    "ObtieneDemandaYTemperaturaRegionByFecha"
)
SADI_REGION = 1002  # whole interconnected system
EC_URL = "https://api.energy-charts.info/price"


def fetch_cammesa(days_back: int = 200, region: int = SADI_REGION) -> Path:
    today = dt.date.today()
    frames = []
    for d in range(days_back, 0, -1):
        day = today - dt.timedelta(days=d)
        r = requests.get(CAMMESA_URL, params={"fecha": day.isoformat(), "id_region": region}, timeout=30)
        r.raise_for_status()
        rows = r.json()
        if rows:
            frames.append(pd.DataFrame(rows))
    if not frames:
        raise RuntimeError("CAMMESA API returned no data")
    df = pd.concat(frames, ignore_index=True)
    df["time"] = pd.to_datetime(df["fecha"], format="%Y-%m-%dT%H:%M:%S.%f%z")
    hourly = (
        df.set_index("time")[["dem", "temp"]]
        .sort_index()
        .resample("1h")
        .mean()
        .dropna(subset=["dem"])
        .rename(columns={"dem": "demand_mw", "temp": "temp_c"})
    )
    out = RAW / "cammesa_sadi_demand_hourly.csv"
    hourly.to_csv(out)
    print(f"CAMMESA: {len(hourly)} hours, {hourly.index.min()} -> {hourly.index.max()}")
    return out


def fetch_de_lu(start: str = "2023-01-01", end: str = "2024-12-31") -> Path:
    r = requests.get(EC_URL, params={"bzn": "DE-LU", "start": start, "end": end}, timeout=120)
    r.raise_for_status()
    js = r.json()
    s = pd.Series(js["price"], index=pd.to_datetime(js["unix_seconds"], unit="s", utc=True), name="price_eur_mwh")
    s = s.resample("1h").mean().dropna()  # 15-min products from Oct 2025 are averaged to hourly
    out = RAW / "de_lu_day_ahead_2023_2024.csv"
    s.to_csv(out)
    print(f"DE-LU: {len(s)} hours, mean {s.mean():.1f} EUR/MWh")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=["cammesa", "delu"])
    a = ap.parse_args()
    RAW.mkdir(parents=True, exist_ok=True)
    if a.only in (None, "cammesa"):
        fetch_cammesa()
    if a.only in (None, "delu"):
        fetch_de_lu()
