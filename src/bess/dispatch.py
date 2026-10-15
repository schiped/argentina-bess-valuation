"""Battery dispatch LP (Pyomo + HiGHS), perfect foresight and rolling daily gates."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pyomo.environ as pyo

from .config import Battery

SOLVER = "appsi_highs"


def solve_window(price: np.ndarray, bat: Battery, soc0: float = 0.0) -> dict[str, np.ndarray]:
    """Maximise price*(discharge - charge) - degradation over one window of hourly prices.

    Charge and discharge are made mutually exclusive only in negative-price hours,
    the only hours where doing both at once could pay (burning energy in losses).
    """
    n = len(price)
    p_max = bat.power_mw * (1 - bat.rpf_share)
    m = pyo.ConcreteModel()
    m.T = pyo.RangeSet(0, n - 1)
    m.c = pyo.Var(m.T, bounds=(0, p_max))
    m.d = pyo.Var(m.T, bounds=(0, p_max))
    m.e = pyo.Var(m.T, bounds=(0, bat.energy_mwh))

    def balance(m, t):
        prev = soc0 if t == 0 else m.e[t - 1]
        return m.e[t] == prev + bat.eta_charge * m.c[t] - m.d[t] / bat.eta_discharge

    m.balance = pyo.Constraint(m.T, rule=balance)

    neg = [t for t in range(n) if price[t] < 0]
    if neg:
        m.N = pyo.Set(initialize=neg)
        m.u = pyo.Var(m.N, domain=pyo.Binary)
        m.only_c = pyo.Constraint(m.N, rule=lambda m, t: m.c[t] <= p_max * m.u[t])
        m.only_d = pyo.Constraint(m.N, rule=lambda m, t: m.d[t] <= p_max * (1 - m.u[t]))

    m.obj = pyo.Objective(
        expr=sum(price[t] * (m.d[t] - m.c[t]) - bat.degradation_per_mwh * m.d[t] for t in m.T),
        sense=pyo.maximize,
    )
    res = pyo.SolverFactory(SOLVER).solve(m)
    if res.solver.termination_condition != pyo.TerminationCondition.optimal:
        raise RuntimeError(f"solver: {res.solver.termination_condition}")
    get = lambda v: np.array([pyo.value(v[t]) for t in m.T])
    return {"charge": get(m.c), "discharge": get(m.d), "soc": get(m.e)}


def perfect_foresight(price: pd.Series, bat: Battery) -> pd.DataFrame:
    out = solve_window(price.to_numpy(), bat)
    return pd.DataFrame(out, index=price.index).assign(price=price)


def rolling_daily(price: pd.Series, bat: Battery, tz: str, lookahead_h: int = 24) -> pd.DataFrame:
    """One gate per local delivery day; the lookahead tail only shapes the closing SoC.

    Working on a UTC/aware index keeps 23 h and 25 h clock-change days correct.
    """
    local = price.index.tz_convert(tz)
    days = pd.Index(local.normalize()).unique()
    soc, parts = 0.0, []
    pos = np.arange(len(price))
    for day in days:
        idx = pos[local.normalize() == day]
        end = min(idx[-1] + 1 + lookahead_h, len(price))
        win = price.iloc[idx[0]:end].to_numpy()
        r = solve_window(win, bat, soc0=soc)
        k = len(idx)
        parts.append(pd.DataFrame({key: v[:k] for key, v in r.items()}, index=price.index[idx]))
        soc = r["soc"][k - 1]
    return pd.concat(parts).assign(price=price)


def summarise(df: pd.DataFrame, bat: Battery) -> dict[str, float]:
    gross = float((df.price * df.discharge).sum())
    cost = float((df.price * df.charge).sum())
    deg = float(bat.degradation_per_mwh * df.discharge.sum())
    hours = len(df)
    return {
        "hours": hours,
        "gross": gross,
        "charge_cost": cost,
        "degradation": deg,
        "trading": gross - cost - deg,
        "mwh_charged": float(df.charge.sum()),
        "mwh_discharged": float(df.discharge.sum()),
        "cycles": float(df.discharge.sum() / bat.energy_mwh),
        "cycles_per_day": float(df.discharge.sum() / bat.energy_mwh / (hours / 24)),
    }
