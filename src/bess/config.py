"""Default assumptions. Everything here is an input, not a fact: change it.

Contract terms follow the AlmaSADI call (Res. SE 50/2026, awarded by
Res. SE 155/2026) as summarised by Marval O'Farrell Mairal and Tavarone:
capacity price capped at 12,500 USD/MW-month, discharged energy paid at
10 USD/MWh, charging energy charged at 20 USD/MWh (losses), primary frequency
regulation (>=30% of power) at 5 USD/MWh, and an annual factor on the capacity
payment that starts at 1.30 and steps down to 0.50 for 2037-2041. The exact
intermediate steps of the factor are not public in the summaries, so they are
interpolated linearly here.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Battery:
    power_mw: float = 100.0
    duration_h: float = 4.0
    eta_charge: float = 0.92
    eta_discharge: float = 0.92
    degradation_per_mwh: float = 3.0  # cost per MWh discharged (currency of the price series)
    rpf_share: float = 0.0  # share of power held back for primary frequency regulation

    @property
    def energy_mwh(self) -> float:
        return self.power_mw * self.duration_h


@dataclass
class AlmaSADIContract:
    capacity_price: float = 8_427.0  # USD/MW-month, average awarded price (Res. 155/2026)
    capacity_ceiling: float = 12_500.0
    discharge_price: float = 10.0  # USD/MWh delivered
    charge_price: float = 20.0  # USD/MWh consumed (losses charge)
    rpf_price: float = 5.0  # USD/MWh of primary regulation band
    rpf_share: float = 0.30
    availability: float = 0.97  # monthly availability achieved
    penalty_cap: float = 0.80  # penalties capped at 80% of monthly payment
    first_year: int = 2028  # commercial operation
    contract_years: int = 15
    factor_start_year: int = 2026
    factor_start: float = 1.30
    factor_floor_year: int = 2037
    factor_floor: float = 0.50
    spot_from_year: int = 2037  # energy and regulation paid at spot from this year
    # CAMMESA dispatches contracted storage by need (weekly plan, network loading > 85%,
    # seasonal peak windows; AlmaGBA workshop, 8 Oct 2026), not by daily price arbitrage
    cycles_per_year: float = 150.0

    def annual_factor(self, year: int) -> float:
        if year >= self.factor_floor_year:
            return self.factor_floor
        span = self.factor_floor_year - self.factor_start_year
        k = (year - self.factor_start_year) / span
        return self.factor_start + k * (self.factor_floor - self.factor_start)


@dataclass
class Finance:
    # AlmaGBA: ~540 MUSD announced at award; CAMMESA reports 683 MW firm in 11 plants
    # (Oct 2026), so about 190-200 USD/kWh all-in for 4 h
    capex_usd_per_kwh: float = 190.0
    opex_share: float = 0.015  # of capex per year
    augmentation_year: int = 10
    augmentation_share: float = 0.12  # of capex
    life_years: int = 20
    tax_rate: float = 0.25  # RIGI rate; use 0.35 for the general regime
    depreciation_years: int = 10
    gearing: float = 0.70  # maximum; actual debt is sculpted to target_dscr
    target_dscr: float = 1.30
    debt_rate: float = 0.085
    debt_tenor: int = 12
    discount_rate: float = 0.10  # project hurdle, USD real
    degradation_per_year: float = 0.02  # usable energy fade before augmentation


@dataclass
class PriceProxy:
    """Hourly marginal-cost proxy for the SADI.

    The shape comes from real CAMMESA demand through a convex merit-order curve
    price ~ demand**elasticity; the level comes from a monthly reference CMg.
    Replace with real CAMMESA hourly CMg (report "Precios horarios") when available.
    """

    elasticity: float = 3.0
    floor_share: float = 0.25
    monthly_cmg_usd: dict[int, float] = field(
        default_factory=lambda: {
            1: 85, 2: 85, 3: 95, 4: 140, 5: 150, 6: 165,
            7: 155, 8: 120, 9: 95, 10: 85, 11: 80, 12: 85,
        }
    )
    # Long-run levels for valuing energy beyond the next few years (the 2026 levels above
    # carry the Q2-2026 fuel price spike). Average about 72 USD/MWh, winter premium kept.
    long_run_cmg_usd: dict[int, float] = field(
        default_factory=lambda: {
            1: 65, 2: 65, 3: 65, 4: 70, 5: 80, 6: 95,
            7: 95, 8: 85, 9: 70, 10: 60, 11: 60, 12: 65,
        }
    )
