"""Price series: real DE-LU day-ahead prices and a CMg proxy for Argentina."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from .config import PriceProxy

RAW = Path(__file__).resolve().parents[2] / "data" / "raw"


def load_de_lu() -> pd.Series:
    s = pd.read_csv(RAW / "de_lu_day_ahead_2023_2024.csv", index_col=0, parse_dates=True).iloc[:, 0]
    s.index = pd.to_datetime(s.index, utc=True)
    return s.rename("price")


def load_cammesa_demand() -> pd.Series:
    df = pd.read_csv(RAW / "cammesa_sadi_demand_hourly.csv", index_col=0)
    df.index = pd.to_datetime(df.index, utc=True).tz_convert("America/Argentina/Buenos_Aires")
    return df["demand_mw"]


def cmg_proxy(demand: pd.Series, p: PriceProxy | None = None) -> pd.Series:
    """Convex merit-order proxy, rescaled so each month averages its reference CMg."""
    p = p or PriceProxy()
    month = demand.index.month
    rel = demand / demand.groupby(month).transform("mean")
    shape = rel**p.elasticity
    shape = shape / shape.groupby(month).transform("mean")
    level = pd.Series(month, index=demand.index).map(p.monthly_cmg_usd)
    price = level * shape.clip(lower=p.floor_share)
    return price.rename("price")


def load_price_csv(path: str | Path) -> pd.Series:
    """Load a real hourly price export (two columns: timestamp, price)."""
    s = pd.read_csv(path, index_col=0, parse_dates=True).iloc[:, 0]
    return s.rename("price")
