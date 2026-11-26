"""Pure performance-metric functions on return series (ADR 0001).

These live in the lowest layer so that peers in the same layer (training's early stopping
and evaluation's reports) both import them from here and never from each other.
No I/O, no state, no imports from atu2 above core.

Units: `returns` are simple per-period returns as fractions (0.01 = 1 %); `periods_per_year`
is 252 for daily trading data. Annualised values are fractions per year. Non-finite
entries are dropped before every calculation.
"""

from collections.abc import Sequence

import numpy as np
import numpy.typing as npt

TRADING_DAYS_PER_YEAR = 252
ArrayLike = Sequence[float] | npt.NDArray[np.floating]


def _clean(returns: ArrayLike) -> npt.NDArray[np.float64]:
    x = np.asarray(returns, dtype=np.float64).ravel()
    return x[np.isfinite(x)]


def annualised_return(returns: ArrayLike, periods_per_year: int = TRADING_DAYS_PER_YEAR) -> float:
    """Arithmetic mean return times periods_per_year (fraction per year); 0.0 when empty."""
    x = _clean(returns)
    return float(np.mean(x) * periods_per_year) if len(x) else 0.0


def annualised_volatility(returns: ArrayLike, periods_per_year: int = TRADING_DAYS_PER_YEAR) -> float:
    """Sample standard deviation (ddof=1) times sqrt(periods_per_year); 0.0 with < 2 points."""
    x = _clean(returns)
    return float(np.std(x, ddof=1) * np.sqrt(periods_per_year)) if len(x) > 1 else 0.0


def sharpe_ratio(returns: ArrayLike, periods_per_year: int = TRADING_DAYS_PER_YEAR) -> float:
    """Annualised Sharpe ratio, zero risk-free rate: mean / std * sqrt(periods_per_year); 0.0 if std is 0."""
    x = _clean(returns)
    if len(x) < 2:
        return 0.0
    sd = float(np.std(x, ddof=1))
    return float(np.mean(x) / sd * np.sqrt(periods_per_year)) if sd > 0 else 0.0


def sortino_ratio(returns: ArrayLike, periods_per_year: int = TRADING_DAYS_PER_YEAR) -> float:
    """
    Annualised Sortino ratio: mean / downside deviation * sqrt(periods_per_year), where the
    downside deviation is sqrt(mean(min(r, 0)^2)) over all periods. 0.0 when there is no downside.
    """
    x = _clean(returns)
    if len(x) < 2:
        return 0.0
    dd = float(np.sqrt(np.mean(np.minimum(x, 0.0) ** 2)))
    return float(np.mean(x) / dd * np.sqrt(periods_per_year)) if dd > 0 else 0.0


def max_drawdown(returns: ArrayLike) -> float:
    """
    Largest peak-to-trough fall of the compounded equity curve, as a positive fraction
    (0.2 = 20 %). The curve starts at 1.0 before the first return.
    """
    x = _clean(returns)
    if len(x) == 0:
        return 0.0
    equity = np.concatenate([[1.0], np.cumprod(1.0 + x)])
    peaks = np.maximum.accumulate(equity)
    return float(np.max(1.0 - equity / peaks))


def hit_rate(returns: ArrayLike) -> float:
    """Share of non-zero periods with a positive return (0..1); 0.0 when every period is zero."""
    x = _clean(returns)
    active = x[x != 0]
    return float(np.mean(active > 0)) if len(active) else 0.0


def turnover(positions: npt.ArrayLike) -> float:
    """
    Mean absolute position change per period, summed across assets. `positions` is
    (periods,) or (periods, assets); the first row counts as a change from flat (0).
    """
    p = np.asarray(positions, dtype=np.float64)
    if p.ndim == 1:
        p = p[:, None]
    if p.shape[0] == 0:
        return 0.0
    changes = np.abs(np.diff(np.vstack([np.zeros((1, p.shape[1])), np.nan_to_num(p)]), axis=0))
    return float(np.mean(changes.sum(axis=1)))
