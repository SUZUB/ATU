# ADR 0007: Evaluation protocol: splits, holdout, trial counting

Status: Draft (M0), awaiting review · Date: 2026-11-26

## Decision
- Expanding-window walk-forward: an initial training period, then test blocks of about two
  years, retraining at each step.
- An embargo of at least the longest feature lookback separates train and test.
- The last two years are a frozen holdout. `atu2/datasets/holdout_guard.py` raises
  `HoldoutAccessError` unless the final flag is set, and only `atu2 evaluate --final` sets it,
  and only when `holdout_unlock.txt` exists (created by the user, never by tooling).
- Every run and every tuning trial, including failed ones, is counted in the registry; the
  deflated Sharpe uses that count. Tuning is capped at 30 trials.
- Every reported number is the median over at least five seeds, with the spread shown.

## Consequences
- The holdout can be used once. Any later model change needs a new untouched period.
