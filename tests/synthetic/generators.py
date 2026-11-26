"""
Seeded synthetic markets whose true answer is known (docs/BUILD_PLAN.md, "Synthetic data
generators"):

- random_walk      no edge: i.i.d. zero-mean log returns
- planted_trend    known edge: each asset follows a persistent +/- drift that flips rarely,
                   so time-series momentum earns a positive return
- regime_switching market-wide calm / turbulent regimes (Markov chain) with different
                   drift and volatility, to exercise changepoint and attention features

All return a SyntheticMarket with close prices on NYSE trading days (atu2.core.calendars).
`to_ohlcv` turns closes into consistent open/high/low/close/volume bars.
Units: volatilities and drifts are per trading day, as fractions (0.01 = 1 %).
"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from atu2.core.calendars import trading_days

DEFAULT_START = "2010-01-04"


@dataclass(frozen=True)
class SyntheticMarket:
    """Close prices (dates x assets), simple daily returns, optional regime labels."""

    prices: pd.DataFrame
    returns: pd.DataFrame
    regimes: pd.Series | None = None
    trend_state: pd.DataFrame | None = None
    params: dict[str, float | int | str] = field(default_factory=dict)


def _dates(n_days: int, start: str) -> pd.DatetimeIndex:
    # ~252 trading days per 365 calendar days; over-allocate, then cut
    end = pd.Timestamp(start) + pd.Timedelta(days=int(n_days * 1.6) + 30)
    days = trading_days(start, end)[:n_days]
    if len(days) < n_days:
        raise ValueError("calendar range too short")
    return pd.DatetimeIndex(days, name="date")


def _assets(n_assets: int) -> list[str]:
    return [f"A{i:02d}" for i in range(n_assets)]


def _market(
    log_ret: np.ndarray, n_days: int, start: str, n_assets: int, start_price: float
) -> tuple[pd.DataFrame, pd.DataFrame]:
    dates = _dates(n_days, start)
    cols = _assets(n_assets)
    # first day has no return: price starts at start_price
    log_ret = log_ret.copy()
    log_ret[0] = 0.0
    prices = pd.DataFrame(start_price * np.exp(np.cumsum(log_ret, axis=0)), index=dates, columns=cols)
    returns = prices.pct_change().iloc[1:]
    return prices, returns


def random_walk(
    n_assets: int = 5,
    n_days: int = 2000,
    *,
    daily_vol: float = 0.01,
    seed: int = 0,
    start: str = DEFAULT_START,
    start_price: float = 100.0,
) -> SyntheticMarket:
    """Pure random walk: no strategy has an edge."""
    rng = np.random.default_rng(seed)
    log_ret = rng.normal(0.0, daily_vol, size=(n_days, n_assets))
    prices, returns = _market(log_ret, n_days, start, n_assets, start_price)
    return SyntheticMarket(prices, returns, params={"kind": "random_walk", "seed": seed, "daily_vol": daily_vol})


def planted_trend(
    n_assets: int = 5,
    n_days: int = 2000,
    *,
    daily_vol: float = 0.01,
    drift_to_vol: float = 0.15,
    mean_trend_days: int = 150,
    seed: int = 0,
    start: str = DEFAULT_START,
    start_price: float = 100.0,
) -> SyntheticMarket:
    """
    Persistent trends: each asset's drift is +/- drift_to_vol * daily_vol and flips sign
    with probability 1 / mean_trend_days per day. A trend follower that knows the state
    has an annualised Sharpe of about drift_to_vol * sqrt(252) (2.4 with the defaults);
    a 63-day momentum signal recovers a good part of it.
    """
    rng = np.random.default_rng(seed)
    flips = rng.random((n_days, n_assets)) < 1.0 / mean_trend_days
    initial = rng.choice([-1.0, 1.0], size=n_assets)
    state = initial[None, :] * np.where(np.cumsum(flips, axis=0) % 2 == 0, 1.0, -1.0)
    mu = drift_to_vol * daily_vol
    log_ret = state * mu + rng.normal(0.0, daily_vol, size=(n_days, n_assets))
    prices, returns = _market(log_ret, n_days, start, n_assets, start_price)
    trend = pd.DataFrame(state, index=prices.index, columns=prices.columns)
    return SyntheticMarket(
        prices,
        returns,
        trend_state=trend,
        params={
            "kind": "planted_trend",
            "seed": seed,
            "daily_vol": daily_vol,
            "drift_to_vol": drift_to_vol,
            "mean_trend_days": mean_trend_days,
        },
    )


def regime_switching(
    n_assets: int = 5,
    n_days: int = 2000,
    *,
    calm_vol: float = 0.006,
    turbulent_vol: float = 0.02,
    calm_drift: float = 0.0004,
    turbulent_drift: float = -0.0008,
    stay_probability: float = 0.98,
    seed: int = 0,
    start: str = DEFAULT_START,
    start_price: float = 100.0,
) -> SyntheticMarket:
    """
    Two market-wide regimes (0 = calm, 1 = turbulent) following a Markov chain with the
    given stay probability (mean regime length 1 / (1 - stay_probability) days).
    """
    rng = np.random.default_rng(seed)
    regime = np.zeros(n_days, dtype=int)
    switches = rng.random(n_days) > stay_probability
    for t in range(1, n_days):
        regime[t] = 1 - regime[t - 1] if switches[t] else regime[t - 1]
    vol = np.where(regime == 1, turbulent_vol, calm_vol)[:, None]
    drift = np.where(regime == 1, turbulent_drift, calm_drift)[:, None]
    log_ret = drift + vol * rng.normal(0.0, 1.0, size=(n_days, n_assets))
    prices, returns = _market(log_ret, n_days, start, n_assets, start_price)
    labels = pd.Series(regime, index=prices.index, name="regime")
    return SyntheticMarket(
        prices,
        returns,
        regimes=labels,
        params={
            "kind": "regime_switching",
            "seed": seed,
            "calm_vol": calm_vol,
            "turbulent_vol": turbulent_vol,
            "stay_probability": stay_probability,
        },
    )


def to_ohlcv(
    prices: pd.DataFrame, seed: int = 0, intraday_vol: float = 0.005, base_volume: float = 1e6
) -> dict[str, pd.DataFrame]:
    """
    Builds consistent daily bars from closes: open = previous close times a small gap,
    high >= max(open, close), low <= min(open, close), low > 0, volume > 0.
    Returns {ticker: DataFrame[open, high, low, close, volume]}.
    """
    rng = np.random.default_rng(seed)
    out: dict[str, pd.DataFrame] = {}
    for col in prices.columns:
        close = prices[col].to_numpy(dtype=float)
        prev = np.concatenate([[close[0]], close[:-1]])
        open_ = prev * np.exp(rng.normal(0.0, intraday_vol / 2, size=len(close)))
        top = np.maximum(open_, close)
        bottom = np.minimum(open_, close)
        high = top * np.exp(np.abs(rng.normal(0.0, intraday_vol, size=len(close))))
        low = bottom * np.exp(-np.abs(rng.normal(0.0, intraday_vol, size=len(close))))
        volume = np.round(base_volume * np.exp(rng.normal(0.0, 0.3, size=len(close))))
        out[col] = pd.DataFrame(
            {"open": open_, "high": high, "low": low, "close": close, "volume": volume}, index=prices.index
        )
    return out
