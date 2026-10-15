"""Valuation of a 100 MW / 400 MWh AlmaSADI battery. Writes figures and JSON to results/.

    python run_argentina.py
"""
from __future__ import annotations

import json
import sys
from dataclasses import asdict, replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from bess.config import AlmaSADIContract, Battery, Finance, PriceProxy
from bess.dispatch import rolling_daily, summarise
from bess.finance import breakeven_capacity_price, cash_flows, irr, lcos, metrics
from bess.plotting import COLORS, style
from bess.prices import cmg_proxy, load_cammesa_demand

OUT = Path(__file__).parent / "results"
TZ = "America/Argentina/Buenos_Aires"
AWARDS = {  # USD/MW-month
    "AlmaSADI floor (Bragado)": 7_397,
    "AlmaSADI award avg": 8_427,
    "AlmaGBA award avg": 11_619,
    "Ceiling": 12_500,
}


def annualise_margin(df: pd.DataFrame, bat: Battery, pp: PriceProxy) -> float:
    """Arbitrage margin per MW-year at long-run price levels.

    Spreads scale with the price level in the proxy, so margin per (hour x USD of
    monthly CMg) observed in the 2026 window is applied to the long-run monthly levels.
    """
    local = df.tz_convert(TZ)
    trade = local.price * (local.discharge - local.charge) - bat.degradation_per_mwh * local.discharge
    by_m = trade.groupby(local.index.month).agg(["sum", "size"])
    level = pd.Series(pp.monthly_cmg_usd)
    k = (by_m["sum"] / (by_m["size"] * level.loc[by_m.index])).mean()
    hours = pd.Series({m: pd.Period(f"2027-{m:02d}").days_in_month * 24 for m in range(1, 13)})
    long_run = pd.Series(pp.long_run_cmg_usd)  # the spot tail starts in 2037
    return float(k * (long_run * hours).sum() / bat.power_mw)


def main() -> None:
    OUT.mkdir(exist_ok=True)
    style()
    pp, con, fin = PriceProxy(), AlmaSADIContract(), Finance()

    demand = load_cammesa_demand()
    price = cmg_proxy(demand, pp).tz_convert("UTC")
    price.to_csv(OUT / "cmg_proxy_hourly.csv")

    # Contract phase: CAMMESA dispatches for system value; 30% of power held for regulation
    bat_c = Battery(rpf_share=con.rpf_share)
    disp_c = rolling_daily(price, bat_c, TZ)
    s_c = summarise(disp_c, bat_c)
    scale = 8_760 / s_c["hours"]
    ops_lp = {"mwh_charged": s_c["mwh_charged"] * scale, "mwh_discharged": s_c["mwh_discharged"] * scale}
    # Base case cycles by need; the price-driven LP above is the upper bound on cycling
    rte = bat_c.eta_charge * bat_c.eta_discharge
    ops_for = lambda cyc: {"mwh_discharged": cyc * bat_c.energy_mwh, "mwh_charged": cyc * bat_c.energy_mwh / rte}
    ops = ops_for(con.cycles_per_year)
    lp_cycles = ops_lp["mwh_discharged"] / bat_c.energy_mwh

    # Spot phase (from 2037): owner trades at marginal price, full power
    bat_m = Battery()
    disp_m = rolling_daily(price, bat_m, TZ)
    merchant = annualise_margin(disp_m, bat_m, pp)

    cf = cash_flows(bat_m, con, fin, ops, merchant)
    met = metrics(cf, bat_m, fin)
    met["lcos_usd_mwh"] = lcos(cf, fin, ops["mwh_discharged"])
    met["breakeven_usd_mw_month"] = breakeven_capacity_price(bat_m, con, fin, ops, merchant)
    met["merchant_margin_usd_mw_year"] = merchant
    met["contract_cycles_per_year"] = con.cycles_per_year
    met["lp_cycles_per_day"] = s_c["cycles_per_day"]
    met["merchant_cycles_per_day"] = summarise(disp_m, bat_m)["cycles_per_day"]
    met["demand_window"] = f"{demand.index.min():%Y-%m-%d} to {demand.index.max():%Y-%m-%d}"

    # IRR against bid price
    bids = np.arange(5_000, 14_001, 250)
    irr_bid = [irr(cash_flows(bat_m, replace(con, capacity_price=b), fin, ops, merchant).project_cf.to_numpy()) for b in bids]
    met["irr_at_awards"] = {
        k: irr(cash_flows(bat_m, replace(con, capacity_price=v), fin, ops, merchant).project_cf.to_numpy())
        for k, v in AWARDS.items()
    }

    # Breakeven grid: capex x hurdle
    capexes = np.arange(140, 301, 20)
    hurdles = np.array([0.08, 0.09, 0.10, 0.11, 0.12, 0.13, 0.14])
    grid = np.array([
        [breakeven_capacity_price(bat_m, con, replace(fin, capex_usd_per_kwh=c, discount_rate=h), ops, merchant) for c in capexes]
        for h in hurdles
    ])

    # Monte Carlo on the awarded average price
    rng = np.random.default_rng(7)
    n = 2_000
    draws = []
    for _ in range(n):
        f = replace(
            fin,
            capex_usd_per_kwh=float(rng.normal(fin.capex_usd_per_kwh, 25)),
            debt_rate=float(rng.uniform(0.07, 0.10)),
        )
        c = replace(con, availability=float(rng.uniform(0.94, 0.995)))
        m_level = merchant * float(rng.lognormal(0, 0.35))
        o = ops_for(float(rng.uniform(60, lp_cycles)))
        draws.append(irr(cash_flows(bat_m, c, f, o, m_level).project_cf.to_numpy()))
    draws = np.array(draws)
    met["mc_irr_p10_p50_p90"] = [float(x) for x in np.nanpercentile(draws, [10, 50, 90])]
    met["mc_prob_irr_below_hurdle"] = float(np.mean(draws < fin.discount_rate))

    # How often CAMMESA actually cycles the battery. At the AlmaGBA operating workshop
    # (CAMMESA/SACME, 8 Oct 2026) dispatch is set weekly and triggered by network loading
    # above 85%, in seasonal peak windows, so a price-driven daily cycle is an upper bound.
    # Under the contract cycling is a net cost (charge 20 > discharge 10 USD/MWh).
    cycling = {}
    for label, cyc in [("Critical days only", 60), ("Seasonal peak windows", 150),
                       ("Most weekdays", 250), ("Price-driven daily (LP)", lp_cycles)]:
        o = ops_for(cyc)
        cycling[label] = {
            "cycles_per_year": float(cyc),
            "project_irr": irr(cash_flows(bat_m, con, fin, o, merchant).project_cf.to_numpy()),
            "breakeven_usd_mw_month": breakeven_capacity_price(bat_m, con, fin, o, merchant),
        }
    met["contract_cycling_sensitivity"] = cycling

    # ---------- figures ----------
    # 1. A winter week: proxy price and battery state of charge
    wk = disp_c.tz_convert(TZ)
    start = wk.index[wk.index.month == 7][0].normalize() + pd.Timedelta(days=7)
    w = wk.loc[start : start + pd.Timedelta(days=7)]
    fig, ax = plt.subplots(2, 1, figsize=(10, 5.6), sharex=True, height_ratios=[1.3, 1])
    ax[0].plot(w.index, w.price, color=COLORS[0], lw=1.6)
    ax[0].set_ylabel("CMg proxy\nUSD/MWh")
    ax[0].set_title("A July week: price proxy from CAMMESA demand and price-driven dispatch (upper bound on cycling)", loc="left")
    ax[1].fill_between(w.index, w.soc / bat_c.energy_mwh * 100, color=COLORS[1], alpha=0.35, lw=0)
    ax[1].plot(w.index, w.soc / bat_c.energy_mwh * 100, color=COLORS[1], lw=1.4)
    ax[1].set_ylabel("State of charge\n%")
    ax[1].set_ylim(0, 105)
    fig.tight_layout()
    fig.savefig(OUT / "fig1_week_dispatch.png", dpi=160)
    plt.close(fig)

    # 2. Revenue stack by year
    ops_y = cf.loc[1:]
    fig, ax = plt.subplots(figsize=(10, 4.8))
    bottom = np.zeros(len(ops_y))
    for col, label, color in [
        ("capacity", "Capacity payment (AlmaSADI)", COLORS[0]),
        ("rpf", "Primary regulation", COLORS[2]),
        ("energy", "Energy (contract net, then spot)", COLORS[1]),
    ]:
        v = ops_y[col].to_numpy() / 1e6
        pos = np.clip(v, 0, None)
        ax.bar(ops_y.calendar, pos, bottom=bottom, color=color, label=label, width=0.75)
        bottom += pos
        neg = np.clip(v, None, 0)
        if neg.any():
            ax.bar(ops_y.calendar, neg, color=color, alpha=0.55, width=0.75)
    ax.bar(ops_y.calendar, (ops_y.opex + ops_y.augmentation) / 1e6, color=COLORS[3], width=0.75, label="Opex and augmentation", alpha=0.8)
    ax.axhline(0, color="#999", lw=0.8)
    ax.set_ylabel("MUSD per year")
    ax.set_title("Revenue stack, 100 MW / 400 MWh at the awarded average of 8,427 USD/MW-month", loc="left")
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax.text(ops_y.calendar.iloc[0] - 0.4, -6.5, "Under contract, charging (20 USD/MWh) costs more\nthan discharging earns (10 USD/MWh): cycling is a cost", fontsize=8.5, color="#4b5664")
    ax.set_ylim(bottom=min(-11, ax.get_ylim()[0]))
    ax.legend(ncols=4, fontsize=9, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.08))
    fig.tight_layout()
    fig.savefig(OUT / "fig2_revenue_stack.png", dpi=160)
    plt.close(fig)

    # 3. Project IRR vs bid
    fig, ax = plt.subplots(figsize=(10, 4.6))
    ax.plot(bids, np.array(irr_bid) * 100, color=COLORS[0], lw=2)
    ax.axhline(fin.discount_rate * 100, color="#999", lw=1, ls="--")
    ax.text(bids[0], fin.discount_rate * 100 + 0.3, f"hurdle {fin.discount_rate:.0%}", color="#666", fontsize=9)
    offsets = [(8, -30), (8, -30), (-150, 10), (8, -30)]
    for (k, v), off in zip(AWARDS.items(), offsets):
        y = met["irr_at_awards"][k] * 100
        ax.scatter([v], [y], color=COLORS[1], zorder=3, s=30)
        ax.annotate(f"{k}\n{v:,} → {y:.1f}%", (v, y), textcoords="offset points", xytext=off, fontsize=8.5)
    ax.set_xlabel("Capacity price bid, USD/MW-month")
    ax.set_ylabel("Project IRR, %")
    ax.set_title(f"What each bid earns at {fin.capex_usd_per_kwh:.0f} USD/kWh capex", loc="left")
    fig.tight_layout()
    fig.savefig(OUT / "fig3_irr_vs_bid.png", dpi=160)
    plt.close(fig)

    # 4. Breakeven heatmap
    fig, ax = plt.subplots(figsize=(10, 4.8))
    im = ax.imshow(grid, cmap="Blues", aspect="auto", origin="lower")
    ax.set_xticks(range(len(capexes)), [f"{c}" for c in capexes])
    ax.set_yticks(range(len(hurdles)), [f"{h:.0%}" for h in hurdles])
    for i in range(len(hurdles)):
        for j in range(len(capexes)):
            v = grid[i, j]
            ax.text(j, i, f"{v/1000:.1f}k", ha="center", va="center", fontsize=8,
                    color="white" if v > grid.mean() else "#1a1a1a")
    ax.set_xlabel("Capex, USD/kWh")
    ax.set_ylabel("Hurdle rate")
    ax.set_title("Breakeven capacity price, USD/MW-month (NPV = 0)", loc="left")
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(OUT / "fig4_breakeven_grid.png", dpi=160)
    plt.close(fig)

    # 5. Monte Carlo
    fig, ax = plt.subplots(figsize=(10, 4.2))
    ax.hist(draws * 100, bins=50, color=COLORS[0], alpha=0.85)
    ax.axvline(fin.discount_rate * 100, color=COLORS[1], lw=1.5, ls="--")
    p10, p50, p90 = met["mc_irr_p10_p50_p90"]
    ax.set_title(f"Project IRR at 8,427 USD/MW-month, {n:,} draws: P10 {p10:.1%} · P50 {p50:.1%} · P90 {p90:.1%}", loc="left")
    ax.set_xlabel("Project IRR, %")
    ax.set_ylabel("Draws")
    fig.tight_layout()
    fig.savefig(OUT / "fig5_montecarlo_irr.png", dpi=160)
    plt.close(fig)

    cf.round(0).to_csv(OUT / "cash_flows_base.csv")
    stats = {"ops": ops, "merchant_margin_usd_mw_year": merchant}
    (OUT / "dispatch_stats.json").write_text(json.dumps(stats, indent=2))
    summary = {
        "metrics": met,
        "assumptions": {"battery": asdict(bat_m), "contract": asdict(con), "finance": asdict(fin), "price_proxy": asdict(pp)},
    }
    (OUT / "summary_argentina.json").write_text(json.dumps(summary, indent=2, default=str))
    print(json.dumps(met, indent=2, default=str))


if __name__ == "__main__":
    main()
