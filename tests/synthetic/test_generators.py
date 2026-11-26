"""Statistical properties of the synthetic generators (the pipeline's known answers)."""

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from atu2.core.calendars import trading_days
from tests.synthetic.generators import planted_trend, random_walk, regime_switching, to_ohlcv

GENERATORS = [random_walk, planted_trend, regime_switching]


def _tsmom_t_stat(market_returns: pd.DataFrame, lookback: int = 63) -> float:
    """t-statistic of sign(past `lookback`-day return) x next-day return, pooled over assets."""
    log_r = np.log1p(market_returns)
    signal = np.sign(log_r.rolling(lookback).sum())
    pnl = (signal.shift(1) * market_returns).stack().dropna().to_numpy()
    return float(pnl.mean() / pnl.std(ddof=1) * np.sqrt(len(pnl)))


@pytest.mark.parametrize("gen", GENERATORS)
def test_shape_calendar_and_positive_prices(gen):
    m = gen(n_assets=4, n_days=600, seed=3)
    assert m.prices.shape == (600, 4)
    assert m.returns.shape == (599, 4)
    assert (m.prices > 0).all().all()
    expected_days = trading_days(m.prices.index[0], m.prices.index[-1])
    assert m.prices.index.equals(pd.DatetimeIndex(expected_days, name="date"))


@pytest.mark.parametrize("gen", GENERATORS)
def test_seeded_and_deterministic(gen):
    a, b, c = gen(seed=7), gen(seed=7), gen(seed=8)
    pd.testing.assert_frame_equal(a.prices, b.prices)
    assert not np.allclose(a.prices.to_numpy(), c.prices.to_numpy())


def test_random_walk_has_no_edge():
    m = random_walk(n_assets=8, n_days=3000, seed=1)
    log_r = np.log1p(m.returns).stack().to_numpy()
    t_mean = log_r.mean() / log_r.std(ddof=1) * np.sqrt(len(log_r))
    assert abs(t_mean) < 3.0
    lag1 = np.mean([m.returns[c].autocorr(1) for c in m.returns])
    assert abs(lag1) < 0.03
    assert abs(_tsmom_t_stat(m.returns)) < 3.0
    assert m.returns.std().between(0.009, 0.011).all()


def test_planted_trend_rewards_momentum():
    m = planted_trend(n_assets=8, n_days=3000, seed=1)
    assert _tsmom_t_stat(m.returns) > 3.0
    # the true state is persistent: mean run length close to mean_trend_days
    assert m.trend_state is not None
    flips = (m.trend_state.diff().abs() > 0).to_numpy().sum()
    mean_run = m.trend_state.size / max(flips, 1)
    assert 75 < mean_run < 300
    # knowing the state gives a high Sharpe (about 0.15 * sqrt(252) = 2.4)
    oracle = (m.trend_state.shift(0).iloc[1:] * m.returns).stack().to_numpy()
    assert oracle.mean() / oracle.std() * np.sqrt(252) > 1.5


def test_regime_switching_regimes_differ_and_persist():
    m = regime_switching(n_assets=5, n_days=3000, seed=2)
    assert m.regimes is not None
    reg = m.regimes.iloc[1:]
    assert set(reg.unique()) == {0, 1}
    vol_calm = m.returns[reg.to_numpy() == 0].stack().std()
    vol_turb = m.returns[reg.to_numpy() == 1].stack().std()
    assert vol_turb / vol_calm > 2.0
    switches = int((m.regimes.diff().abs() > 0).sum())
    assert len(m.regimes) / max(switches, 1) > 20


@settings(max_examples=25, deadline=None)
@given(seed=st.integers(0, 10_000), vol=st.floats(0.001, 0.05))
def test_ohlcv_bars_are_consistent(seed, vol):
    m = random_walk(n_assets=2, n_days=120, seed=seed, daily_vol=vol)
    for ticker, bars in to_ohlcv(m.prices, seed=seed, intraday_vol=vol).items():
        assert (bars["low"] > 0).all()
        assert (bars["high"] >= bars[["open", "close"]].max(axis=1)).all()
        assert (bars["low"] <= bars[["open", "close"]].min(axis=1)).all()
        assert (bars["volume"] > 0).all()
        np.testing.assert_allclose(bars["close"].to_numpy(), m.prices[ticker].to_numpy())
