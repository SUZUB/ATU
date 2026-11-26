# ADR 0006: Loss definition and cost model

Status: Draft (M0), awaiting review · Date: 2026-11-26

## Context
The papers' pre-cost Sharpe ratios do not survive realistic costs.

## Decision
- Training loss (plan, "models and losses"): negative annualised Sharpe of the net portfolio
  return, where the net return subtracts `c` times the volatility-scaled absolute position
  change, plus `lambda_turn * mean|z_t - z_(t-1)|` plus `lambda_corr * |corr(r_p, r_long)|`.
  Setting `c`, `lambda_turn` and `lambda_corr` to zero recovers the plain Sharpe loss.
- Backtest costs: linear basis points on the absolute position change per asset class, plus
  an optional short-borrow cost; all values come from config. Every result is reported at
  0, 1, 2, 3 and 5 bps; 2 bps is the central case.

## Consequences
- The cost assumption is raised if paper trading measures higher slippage (R16).
