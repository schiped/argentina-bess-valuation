"""Interactive AlmaSADI battery valuation.  streamlit run app.py

Uses the dispatch results saved by run_argentina.py, so it answers in milliseconds.
"""
from __future__ import annotations

import json
import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

import numpy as np
import pandas as pd
import streamlit as st

from bess.config import AlmaSADIContract, Battery, Finance
from bess.finance import breakeven_capacity_price, cash_flows, irr, lcos, metrics

ROOT = Path(__file__).parent
stats = json.loads((ROOT / "results" / "dispatch_stats.json").read_text())

st.set_page_config(page_title="AlmaSADI BESS valuation", layout="wide")
st.title("AlmaSADI battery valuation")
st.caption(
    "100 MW / 400 MWh battery under the AlmaSADI storage contract (Res. SE 50/2026, awarded by Res. 155/2026), "
    "with energy and regulation paid at spot from 2037. Contract cycling is set by need (CAMMESA AlmaGBA operating model, Oct 2026); the spot tail comes from a Pyomo dispatch on real CAMMESA demand."
)

with st.sidebar:
    st.header("Bid and contract")
    cap_price = st.slider("Capacity price, USD/MW-month", 5_000, 12_500, 8_427, 100)
    availability = st.slider("Availability", 0.90, 1.00, 0.97, 0.005)
    first_year = st.selectbox("Commercial operation", [2027, 2028, 2029], index=1)
    cycles = st.slider("CAMMESA cycles per year", 30, 400, 150, 10,
                       help="Dispatch by need (weekly plan, network loading > 85%). Cycling is a net cost under the contract.")
    st.header("Cost and finance")
    capex = st.slider("Capex, USD/kWh", 120, 320, 190, 5)
    hurdle = st.slider("Hurdle rate (USD)", 0.06, 0.16, 0.10, 0.005)
    tax = st.radio("Tax regime", ["RIGI 25%", "General 35%"], horizontal=True)
    dscr = st.slider("Target DSCR", 1.10, 1.60, 1.30, 0.05)
    debt_rate = st.slider("Debt rate", 0.05, 0.12, 0.085, 0.005)
    st.header("Spot tail")
    merchant_k = st.slider("Spot margin vs base case", 0.0, 2.0, 1.0, 0.05)

bat = Battery()
con = replace(AlmaSADIContract(), capacity_price=cap_price, availability=availability, first_year=first_year)
fin = replace(
    Finance(), capex_usd_per_kwh=capex, discount_rate=hurdle,
    tax_rate=0.25 if tax.startswith("RIGI") else 0.35, target_dscr=dscr, debt_rate=debt_rate,
)
rte = bat.eta_charge * bat.eta_discharge
ops = {"mwh_discharged": cycles * bat.energy_mwh, "mwh_charged": cycles * bat.energy_mwh / rte}
merchant = stats["merchant_margin_usd_mw_year"] * merchant_k

cf = cash_flows(bat, con, fin, ops, merchant)
m = metrics(cf, bat, fin)
be = breakeven_capacity_price(bat, con, fin, ops, merchant)

c = st.columns(5)
c[0].metric("Project IRR", f"{m['project_irr']:.1%}")
c[1].metric("Equity IRR", f"{m['equity_irr']:.1%}")
c[2].metric("NPV at hurdle", f"{m['npv_musd']:.1f} MUSD")
c[3].metric("Breakeven bid", f"{be:,.0f} USD/MW-mo")
c[4].metric("LCOS", f"{lcos(cf, fin, ops['mwh_discharged']):.0f} USD/MWh")
st.caption(f"Capex {m['capex_musd']:.0f} MUSD · sculpted gearing {m['gearing']:.0%} · min DSCR {m['min_dscr']:.2f}")

left, right = st.columns([1.4, 1])
with left:
    st.subheader("Cash flow by year, MUSD")
    show = cf.loc[1:, ["capacity", "rpf", "energy", "opex", "augmentation", "tax"]] / 1e6
    show.index = cf.loc[1:, "calendar"]
    st.bar_chart(show)
with right:
    st.subheader("Project IRR against bid")
    bids = np.arange(5_000, 14_001, 250)
    curve = pd.DataFrame(
        {"Project IRR %": [irr(cash_flows(bat, replace(con, capacity_price=b), fin, ops, merchant).project_cf.to_numpy()) * 100 for b in bids]},
        index=bids,
    )
    st.line_chart(curve)
    st.caption("AlmaSADI floor 7,397 · AlmaSADI award average 8,427 · AlmaGBA award average 11,619 · ceiling 12,500 USD/MW-month")

with st.expander("Cash flow table"):
    st.dataframe((cf / 1e6).round(2).assign(calendar=cf.calendar, dscr=cf.dscr.round(2)))
st.caption("Price shape is a merit-order proxy built from real CAMMESA demand. Replace with CAMMESA hourly CMg to use for a real bid.")
