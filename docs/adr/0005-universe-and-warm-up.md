# ADR 0005: Universe rule and warm-up rule

Status: Draft (M0), awaiting review · Date: 2026-11-26

## Context
A universe picked after seeing performance is selection bias (R10).

## Decision (recommended default, to confirm in M1)
- About 40 liquid ETFs with at least 12 years of history and high daily volume, spread over
  equities, bonds, credit, gold, oil, commodities, real estate and currencies.
- The rule and the candidate list live in `configs/universe.yaml` and are committed before any backtest runs.
- An asset becomes tradable only after 252 trading days of history; the warm-up rows are
  dropped explicitly, never filled with zeros.

## Consequences
- Changing the universe after the holdout is opened is not allowed (change control).
