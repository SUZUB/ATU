"""core/metrics against hand calculations."""

import numpy as np
import pytest

from atu2.core.metrics import (
    annualised_return,
    annualised_volatility,
    hit_rate,
    max_drawdown,
    sharpe_ratio,
    sortino_ratio,
    turnover,
)

R = [0.01, -0.02, 0.03, 0.0, -0.01]


def test_hand_calculated_values():
    mean = 0.002
    sd = np.sqrt(sum((r - mean) ** 2 for r in R) / 4)
    assert annualised_return(R) == pytest.approx(mean * 252)
    assert annualised_volatility(R) == pytest.approx(sd * np.sqrt(252))
    assert sharpe_ratio(R) == pytest.approx(mean / sd * np.sqrt(252))
    downside = np.sqrt((0.02**2 + 0.01**2) / 5)
    assert sortino_ratio(R) == pytest.approx(mean / downside * np.sqrt(252))
    assert hit_rate(R) == pytest.approx(2 / 4)


def test_max_drawdown_hand_calculation():
    # equity 1.0 -> 1.1 -> 0.88 -> 0.968: worst fall from 1.1 to 0.88 = 20 %
    assert max_drawdown([0.1, -0.2, 0.1]) == pytest.approx(0.2)
    assert max_drawdown([0.05, 0.05]) == 0.0
    assert max_drawdown([-0.5]) == pytest.approx(0.5)  # loss from the starting capital


def test_turnover():
    assert turnover([1.0, 1.0, -1.0]) == pytest.approx((1 + 0 + 2) / 3)
    assert turnover(np.array([[0.5, -0.5], [0.5, 0.5]])) == pytest.approx((1.0 + 1.0) / 2)
    assert turnover(np.zeros((0, 2))) == 0.0


def test_degenerate_inputs_and_nan_dropping():
    for f in (annualised_return, annualised_volatility, sharpe_ratio, sortino_ratio, max_drawdown, hit_rate):
        assert f([]) == 0.0
    assert sharpe_ratio([0.01, 0.01, 0.01]) == 0.0  # zero volatility
    assert sortino_ratio([0.01, 0.02]) == 0.0  # no downside
    assert hit_rate([0.0, 0.0]) == 0.0
    assert sharpe_ratio([0.01, np.nan, -0.02, 0.03, np.inf]) == pytest.approx(sharpe_ratio([0.01, -0.02, 0.03]))
    assert sharpe_ratio([0.01]) == 0.0 and sortino_ratio([0.01]) == 0.0
