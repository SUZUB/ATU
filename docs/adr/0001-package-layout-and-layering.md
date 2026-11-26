# ADR 0001: Package layout and the downward-only import rule

Status: Draft (M0), awaiting review · Date: 2026-11-26

## Context
ATU 2 must let one part change (data source, model) without touching the backtest or the
evaluation, and it lives in the same repository as the legacy ATU 1 code (ORB v2, RL).

## Decision
- New package `atu2/` in the existing ATU repo (locked decision D1). Legacy files stay untouched.
- Layers, top to bottom: `cli > experiments > (paper, evaluation, training) > (execution, backtest, models) > (datasets, features) > data > core`.
- A module imports only from layers strictly below its own. Packages in the same layer
  (for example `backtest` and `models`) may not import each other; anything two peers
  share moves down a layer, usually into `core/types.py`.
- `atu2` never imports legacy top-level modules (`orb_eval`, `alpaca_data`, `src`, ...).
  Useful legacy code is ported, with its tests.
- Enforced by `tests/architecture/test_layering.py`, which has negative tests proving it
  catches upward, peer, root-package and legacy imports.

## Alternatives
- A separate repository: cleaner, but loses the history and the Colab setup.
- Allowing peer imports: simpler at first, but cycles creep in (for example training <-> evaluation).

## Consequences
- The training loop's early stopping cannot import `evaluation.metrics` (peer layer). The
  Sharpe computation it needs must live lower (models/losses or core) or be duplicated
  with a test that both agree. Revisit with an ADR if this becomes awkward in M4.
