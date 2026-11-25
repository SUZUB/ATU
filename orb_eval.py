"""
orb_eval.py
===========
Statistics that separate skill from luck in a backtest (numpy / scipy only).

- block_bootstrap_ci   moving-block bootstrap CI for the mean or the Sharpe ratio of
                       daily returns (blocks keep short-range autocorrelation intact)
- deflated_sharpe      Bailey & Lopez de Prado (2014) Deflated Sharpe Ratio: probability
                       that the true Sharpe is > 0 after accounting for the number of
                       variants tried, non-normal returns and sample length
- random_direction_test  are the chosen trade directions better than coin flips on the
                       very same trades?
- breakeven_fee        fee per side at which the average trade earns exactly 0
"""

from typing import Optional, Tuple

import numpy as np
from scipy import stats

EULER_GAMMA = 0.5772156649015329


def _sharpe(x: np.ndarray, annualize: float = 1.0) -> float:
    sd = np.std(x, ddof=1) if len(x) > 1 else 0.0
    return float(np.mean(x) / sd * np.sqrt(annualize)) if sd > 0 else 0.0


def block_bootstrap_ci(
    returns,
    kind: str = "mean",
    block: int = 5,
    n_boot: int = 2000,
    alpha: float = 0.05,
    annualize: float = 252.0,
    seed: int = 0,
) -> Tuple[float, float, float]:
    """
    Moving-block bootstrap of `returns` (1-D, e.g. daily portfolio returns).
    kind="mean"   -> statistic is the mean return
    kind="sharpe" -> statistic is the Sharpe ratio, annualized with `annualize`
    Returns (lo, hi, p_le_0): the (alpha/2, 1-alpha/2) percentile interval and the
    share of bootstrap statistics <= 0.
    """
    x = np.asarray(returns, dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 2:
        return float("nan"), float("nan"), 1.0
    if kind not in ("mean", "sharpe"):
        raise ValueError("kind must be 'mean' or 'sharpe'")
    block = int(max(1, min(block, n)))
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block))
    starts = rng.integers(0, n - block + 1, size=(n_boot, n_blocks))
    idx = (starts[:, :, None] + np.arange(block)[None, None, :]).reshape(n_boot, -1)[:, :n]
    samples = x[idx]
    if kind == "mean":
        stat = samples.mean(axis=1)
    else:
        sd = samples.std(axis=1, ddof=1)
        stat = np.where(sd > 0, samples.mean(axis=1) / np.where(sd > 0, sd, 1.0) * np.sqrt(annualize), 0.0)
    lo, hi = np.percentile(stat, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi), float(np.mean(stat <= 0))


def deflated_sharpe(returns, n_trials: int, var_sr_trials: Optional[float] = None) -> float:
    """
    Deflated Sharpe Ratio (Bailey & Lopez de Prado, 2014) on per-period (daily) returns.

        SR     = mean / std                                   (not annualized)
        sigma2 = 1 - skew*SR + (kurt - 1)/4 * SR^2            (kurt non-excess)
        v      = max(sigma2 / (T - 1), var_sr_trials)         variance of the SR estimate
        SR0    = sqrt(v) * ((1-g) * Phi^-1(1 - 1/N) + g * Phi^-1(1 - 1/(N e)))   g = Euler-Mascheroni
        DSR    = Phi((SR - SR0) * sqrt(T - 1) / sqrt(sigma2))

    SR0 is the Sharpe a best-of-N search would show by luck alone. DSR >= 0.95 means a
    95% probability that the true Sharpe exceeds that bar. n_trials <= 1 -> SR0 = 0
    (the plain Probabilistic Sharpe Ratio).
    """
    x = np.asarray(returns, dtype=float)
    x = x[np.isfinite(x)]
    T = len(x)
    if T < 3 or np.std(x, ddof=1) == 0:
        return 0.0
    sr = float(np.mean(x) / np.std(x, ddof=1))
    skew = float(stats.skew(x, bias=False))
    kurt = float(stats.kurtosis(x, fisher=False, bias=False))
    sigma2 = 1.0 - skew * sr + (kurt - 1.0) / 4.0 * sr ** 2
    sigma2 = max(sigma2, 1e-12)
    if n_trials and n_trials > 1:
        v = max(sigma2 / (T - 1), float(var_sr_trials or 0.0))
        N = float(n_trials)
        sr0 = np.sqrt(v) * ((1 - EULER_GAMMA) * stats.norm.ppf(1 - 1 / N)
                            + EULER_GAMMA * stats.norm.ppf(1 - 1 / (N * np.e)))
    else:
        sr0 = 0.0
    return float(stats.norm.cdf((sr - sr0) * np.sqrt(T - 1) / np.sqrt(sigma2)))


def random_direction_test(r_long, r_short, direction, n_random: int = 2000, seed: int = 0) -> float:
    """
    Compares the average R of the chosen directions with `n_random` coin-flip direction
    assignments on the SAME trades (rows with direction != 0 and both outcomes known).
    Returns the p-value: share of random runs whose average R >= the observed one.
    A small p-value means the direction choice itself adds value.
    """
    rl = np.asarray(r_long, dtype=float)
    rs = np.asarray(r_short, dtype=float)
    d = np.asarray(direction, dtype=float)
    keep = (d != 0) & np.isfinite(rl) & np.isfinite(rs)
    if keep.sum() == 0:
        return 1.0
    rl, rs, d = rl[keep], rs[keep], d[keep]
    observed = float(np.mean(np.where(d > 0, rl, rs)))
    rng = np.random.default_rng(seed)
    flips = rng.integers(0, 2, size=(n_random, len(d))).astype(bool)
    random_means = np.where(flips, rl[None, :], rs[None, :]).mean(axis=1)
    return float(np.mean(random_means >= observed))


def breakeven_fee(gross_r, cost_units) -> float:
    """
    Fee per side (fraction of price) at which the average trade nets 0R:
    net R = gross R - fee * cost_unit  =>  fee* = mean(gross) / mean(cost_unit).
    Negative when the gross edge itself is negative.
    """
    g = np.asarray(gross_r, dtype=float)
    c = np.asarray(cost_units, dtype=float)
    keep = np.isfinite(g) & np.isfinite(c)
    if keep.sum() == 0 or np.mean(c[keep]) <= 0:
        return float("nan")
    return float(np.mean(g[keep]) / np.mean(c[keep]))
