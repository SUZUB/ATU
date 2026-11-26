"""Unit-carrying type aliases shared by all layers.

Units are part of the type name so that basis points, fractions and annual values
cannot be mixed silently. The data contracts from the plan (PriceFrame, FeatureFrame,
Batch, TargetPositions, Orders, ReturnSeries, GateResult) are added here by the
milestones that introduce them.
"""

from typing import NewType

Bps = NewType("Bps", float)
"""Cost or threshold in basis points: 1 bp = 0.0001 of notional."""

Fraction = NewType("Fraction", float)
"""A plain fraction, e.g. a daily return of 0.01 = 1 %."""

AnnualFraction = NewType("AnnualFraction", float)
"""An annualised fraction, e.g. a volatility target of 0.15 = 15 % per year."""

TradingDays = NewType("TradingDays", int)
"""A count of exchange trading days (not calendar days)."""


def bps_to_fraction(bps: Bps | float) -> Fraction:
    """Converts basis points to a fraction (5 bps -> 0.0005)."""
    return Fraction(float(bps) / 10_000.0)
