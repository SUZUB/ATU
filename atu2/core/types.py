"""Unit-carrying type aliases shared by all layers.

Units are part of the type name so that basis points, fractions and annual values
cannot be mixed silently. The data contracts from the plan (PriceFrame, FeatureFrame,
Batch, TargetPositions, Orders, ReturnSeries, GateResult) are added here by the
milestones that introduce them.
"""

from dataclasses import dataclass
from typing import NewType

import pandas as pd

from atu2.core.errors import ContractError

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


PRICE_FIELDS: tuple[str, ...] = ("open", "high", "low", "close", "volume", "adj_close")


@dataclass(frozen=True)
class PriceFrame:
    """
    Interface contract data -> features (docs/BUILD_PLAN.md, "Interface contracts").

    Every field is a DataFrame indexed by NYSE trading dates (timezone-naive, sorted,
    unique) with one column per asset, all sharing the same index and columns.
    open/high/low/close are raw prices, adj_close is the split- and dividend-adjusted
    (total-return) close, volume is shares. `tradable` is False before an asset's warm-up
    ends and on days without data. `adjusted` says adj_close is total-return adjusted.
    `data_hash` is the content hash of all fields (NFR-09).
    """

    open: pd.DataFrame
    high: pd.DataFrame
    low: pd.DataFrame
    close: pd.DataFrame
    volume: pd.DataFrame
    adj_close: pd.DataFrame
    tradable: pd.DataFrame
    adjusted: bool
    data_hash: str

    def __post_init__(self) -> None:
        ref = self.adj_close
        if not isinstance(ref.index, pd.DatetimeIndex):
            raise ContractError("PriceFrame: index must be a DatetimeIndex of trading dates")
        if not ref.index.is_monotonic_increasing or ref.index.has_duplicates:
            raise ContractError("PriceFrame: dates must be sorted and unique")
        if ref.index.tz is not None:
            raise ContractError("PriceFrame: dates must be timezone-naive exchange dates")
        for name in (*PRICE_FIELDS, "tradable"):
            frame = getattr(self, name)
            if not frame.index.equals(ref.index) or not frame.columns.equals(ref.columns):
                raise ContractError(f"PriceFrame: field '{name}' does not share the index and columns of adj_close")
        if not all(pd.api.types.is_bool_dtype(dtype) for dtype in self.tradable.dtypes):
            raise ContractError("PriceFrame: tradable must be boolean")

    @property
    def dates(self) -> pd.DatetimeIndex:
        """Trading dates of the frame."""
        return pd.DatetimeIndex(self.adj_close.index)

    @property
    def tickers(self) -> list[str]:
        """Asset columns, in order."""
        return [str(c) for c in self.adj_close.columns]
