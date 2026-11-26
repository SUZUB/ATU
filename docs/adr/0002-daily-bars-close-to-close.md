# ADR 0002: Daily bars and close-to-close timing

Status: Draft (M0), awaiting review · Date: 2026-11-26

## Context
The reference papers use daily data; costs at daily frequency stay small enough to test honestly.

## Decision
- Daily bars only (D3). The signal for day t is computed after the close of t from data up
  to t; the position earns the close-to-close return from t to t+1; costs are charged on
  the absolute position change at t.
- One exchange calendar for every asset: the NYSE calendar in `atu2/core/calendars.py`.
  Other assets are aligned to it, forward-filling at most one day, otherwise marked missing.

## Alternatives
- Next-open execution: closer to reality but needs open prices of uneven quality and
  differs from the papers. Paper trading (M7) measures the gap instead.

## Consequences
- Backtests ignore the close-to-fill slippage; risk R16 tracks it in the paper phase.
