"""Project cash flows for an AlmaSADI battery: contract phase, spot tail, debt, tax."""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd

from .config import AlmaSADIContract, Battery, Finance


def irr(cf: np.ndarray, lo: float = -0.99, hi: float = 1.0) -> float:
    npv = lambda r: float(np.sum(cf / (1 + r) ** np.arange(len(cf))))
    if npv(lo) * npv(hi) > 0:
        return float("nan")
    for _ in range(200):
        mid = (lo + hi) / 2
        if npv(lo) * npv(mid) <= 0:
            hi = mid
        else:
            lo = mid
    return (lo + hi) / 2


def npv(cf: np.ndarray, rate: float) -> float:
    return float(np.sum(cf / (1 + rate) ** np.arange(len(cf))))


def cash_flows(
    bat: Battery,
    con: AlmaSADIContract,
    fin: Finance,
    contract_ops: dict[str, float],
    merchant_margin_per_mw_year: float,
) -> pd.DataFrame:
    """Yearly cash flows in USD. Year 0 is construction.

    contract_ops: annual MWh charged and discharged under CAMMESA dispatch
                  (keys 'mwh_charged', 'mwh_discharged' per year, at full size).
    merchant_margin_per_mw_year: spot arbitrage margin once the contract
                  pays energy at spot (from con.spot_from_year) or expires.
    """
    capex = fin.capex_usd_per_kwh * bat.energy_mwh * 1_000
    years = np.arange(0, fin.life_years + 1)
    cal = con.first_year - 1 + years
    rows = []
    for y, yr in zip(years, cal):
        r = dict(year=int(y), calendar=int(yr))
        if y == 0:
            r.update(capex=-capex)
            rows.append(r)
            continue
        fade = 1 - fin.degradation_per_year * ((y - 1) % fin.augmentation_year)
        in_contract = y <= con.contract_years
        spot_energy = yr >= con.spot_from_year or not in_contract
        cap = 0.0
        if in_contract:
            pay = con.capacity_price * con.annual_factor(yr) * bat.power_mw * 12
            shortfall = max(0.0, 1 - con.availability)
            penalty = min(pay * shortfall, pay * con.penalty_cap)
            cap = pay - penalty
        if spot_energy:
            energy = merchant_margin_per_mw_year * bat.power_mw * fade
            rpf = 0.0
        else:
            energy = (
                con.discharge_price * contract_ops["mwh_discharged"] * fade
                - con.charge_price * contract_ops["mwh_charged"] * fade
            )
            rpf = con.rpf_price * con.rpf_share * bat.power_mw * 8_760 * con.availability
        opex = -fin.opex_share * capex
        aug = -fin.augmentation_share * capex if y == fin.augmentation_year else 0.0
        ebitda = cap + energy + rpf + opex
        dep = capex / fin.depreciation_years if y <= fin.depreciation_years else 0.0
        r.update(capacity=cap, energy=energy, rpf=rpf, opex=opex, augmentation=aug, ebitda=ebitda, depreciation=dep)
        rows.append(r)
    df = pd.DataFrame(rows).fillna(0.0).set_index("year")

    # Debt sculpted to a target DSCR on unlevered cash flow, capped by maximum gearing
    unlev_tax = -np.maximum(0.0, df.ebitda - df.depreciation) * fin.tax_rate
    cfads = (df.ebitda + df.augmentation + unlev_tax).clip(lower=0.0)
    tenor = (df.index >= 1) & (df.index <= fin.debt_tenor)
    service = np.where(tenor, cfads / fin.target_dscr, 0.0)
    disc = (1 + fin.debt_rate) ** -df.index.to_numpy()
    debt = min(float(np.sum(service * disc)), fin.gearing * capex)
    service *= debt / max(float(np.sum(service * disc)), 1e-9)

    balance, interest, principal = debt, [], []
    for y in df.index:
        i = balance * fin.debt_rate if tenor[y] else 0.0
        p = service[y] - i if tenor[y] else 0.0
        balance -= p
        interest.append(-i)
        principal.append(-p)
    df["debt_draw"] = np.where(df.index == 0, debt, 0.0)
    df["interest"] = interest
    df["principal"] = principal
    df["tax"] = -np.maximum(0.0, df.ebitda - df.depreciation + df.interest) * fin.tax_rate
    df.loc[0, "tax"] = 0.0
    df["project_cf"] = df.capex + df.ebitda + df.augmentation + unlev_tax.where(df.index > 0, 0.0)
    df["equity_cf"] = df.capex + df.ebitda + df.augmentation + df.tax + df.debt_draw + df.interest + df.principal
    ds = -(df.interest + df.principal)
    df["dscr"] = np.where(ds > 0, (df.ebitda + df.augmentation + df.tax) / ds.where(ds > 0), np.nan)
    df["gearing"] = debt / capex
    return df


def metrics(df: pd.DataFrame, bat: Battery, fin: Finance) -> dict[str, float]:
    disc = (1 + fin.discount_rate) ** -df.index.to_numpy()
    costs = -(df.capex + df.opex + df.augmentation).to_numpy()
    return {
        "project_irr": irr(df.project_cf.to_numpy()),
        "equity_irr": irr(df.equity_cf.to_numpy()),
        "npv_musd": npv(df.project_cf.to_numpy(), fin.discount_rate) / 1e6,
        "min_dscr": float(np.nanmin(df.dscr)),
        "avg_dscr": float(np.nanmean(df.dscr)),
        "gearing": float(df.gearing.iloc[0]),
        "capex_musd": float(-df.capex.sum()) / 1e6,
        "pv_costs_musd": float(np.sum(costs * disc)) / 1e6,
    }


def lcos(df: pd.DataFrame, fin: Finance, mwh_per_year: float) -> float:
    """Levelised cost per MWh discharged (capex, opex, augmentation; no charging energy)."""
    t = df.index.to_numpy()
    disc = (1 + fin.discount_rate) ** -t
    cost = -(df.capex + df.opex + df.augmentation).to_numpy()
    energy = np.where(t > 0, mwh_per_year, 0.0)
    return float(np.sum(cost * disc) / np.sum(energy * disc))


def breakeven_capacity_price(
    bat: Battery, con: AlmaSADIContract, fin: Finance, ops: dict[str, float], merchant: float
) -> float:
    """Capacity price (USD/MW-month) that makes project NPV zero at fin.discount_rate."""
    lo, hi = 0.0, 40_000.0
    f = lambda p: npv(cash_flows(bat, replace(con, capacity_price=p), fin, ops, merchant).project_cf.to_numpy(), fin.discount_rate)
    for _ in range(60):
        mid = (lo + hi) / 2
        if f(mid) > 0:
            hi = mid
        else:
            lo = mid
    return (lo + hi) / 2
