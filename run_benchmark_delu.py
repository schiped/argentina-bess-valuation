"""International benchmark: the same dispatch engine on DE-LU day-ahead prices, 2023-2024.

A 10 MW / 20 MWh battery committed one gate per delivery day (rolling horizon)
against the same two years solved with perfect foresight in one window.

    python run_benchmark_delu.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

import matplotlib.pyplot as plt
import numpy as np

from bess.config import Battery
from bess.dispatch import perfect_foresight, rolling_daily, summarise
from bess.plotting import COLORS, style
from bess.prices import load_de_lu

OUT = Path(__file__).parent / "results"
TZ = "Europe/Berlin"


def main() -> None:
    OUT.mkdir(exist_ok=True)
    style()
    price = load_de_lu()
    bat = Battery(power_mw=10, duration_h=2)

    pf = perfect_foresight(price, bat)
    s_pf = summarise(pf, bat)
    runs = {}
    for la in (0, 6, 12, 24):
        df = rolling_daily(price, bat, TZ, lookahead_h=la)
        runs[la] = (df, summarise(df, bat))
        print(f"lookahead {la:>2} h: capture {runs[la][1]['trading'] / s_pf['trading']:.2%}")

    years = s_pf["hours"] / 8_760
    res = {
        "perfect_foresight_eur": s_pf["trading"],
        "rolling": {
            la: {
                "trading_eur": s["trading"],
                "capture": s["trading"] / s_pf["trading"],
                "eur_per_mw_year": s["trading"] / bat.power_mw / years,
                "cycles_per_day": s["cycles_per_day"],
            }
            for la, (_, s) in runs.items()
        },
        "mean_price": float(price.mean()),
    }
    (OUT / "summary_delu.json").write_text(json.dumps(res, indent=2))

    roll = runs[24][0]
    cum = lambda d: (d.price * (d.discharge - d.charge) - bat.degradation_per_mwh * d.discharge).cumsum() / 1e6
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2), width_ratios=[1.7, 1])
    ax[0].plot(pf.index, cum(pf), color=COLORS[3], lw=1.4, label="Perfect foresight (ceiling)")
    ax[0].plot(roll.index, cum(roll), color=COLORS[0], lw=1.6, label="Daily gate, 24 h lookahead")
    ax[0].set_ylabel("Cumulative trading profit, MEUR")
    ax[0].legend(frameon=False, fontsize=9)
    ax[0].set_title("DE-LU 2023-24, 10 MW / 20 MWh", loc="left")
    las = list(runs)
    cap = [res["rolling"][k]["capture"] * 100 for k in las]
    ax[1].plot(las, cap, "o-", color=COLORS[1])
    for x, y in zip(las, cap):
        ax[1].annotate(f"{y:.2f}%", (x, y), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=8.5)
    ax[1].set_xticks(las)
    ax[1].set_xlabel("Lookahead past the committed day, h")
    ax[1].set_ylabel("Capture vs ceiling, %")
    ax[1].set_title("Value of lookahead", loc="left")
    ax[1].set_ylim(min(cap) - 1, 100.5)
    fig.tight_layout()
    fig.savefig(OUT / "fig6_delu_benchmark.png", dpi=160)
    plt.close(fig)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
