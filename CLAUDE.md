# ATU 2 project rules

Read docs/BUILD_PLAN.md before any task. It is the source of truth for requirements (FR-xx, NFR-xx), tests (T-xxx), risks (Rxx) and milestones (M0-M8).

## Hard rules
1. Layering: modules import only from layers below them: cli > experiments > (paper, evaluation, training) > (execution, backtest, models) > (datasets, features) > data > core. A test enforces it.
2. No look-ahead: a feature or position at date t may use data up to t only. Never weaken or delete a leakage test.
3. Costs are always applied in backtests and reported at 0, 1, 2, 3 and 5 bps.
4. Never read holdout dates. Only `atu2 evaluate --final` may, and only when the file holdout_unlock.txt exists. Never create that file yourself.
5. Never edit configs/gate.yaml or its recorded hash. Never lower a threshold to make a result pass.
6. No secrets in code, notebooks, logs, tests or commits. Keys come from environment variables only. Tests make no network calls (use recorded fixtures).
7. Never modify a test just to make it pass. If a test seems wrong, stop and explain.
8. Every random source is seeded from config. Every experiment run is logged in the registry, including failed ones (they count as trials).
9. Do not touch legacy files outside atu2/, configs/, tests/, docs/, .github/ unless the task says so.
10. Do not add scope. Extra ideas go in docs/ideas.md, not in code.

## Commands
- Lint: ruff check . && ruff format --check .
- Types: mypy atu2
- Tests: pytest -q --cov=atu2
- All three must pass before you say a task is done.

## Done means
Code merged on a branch, tests for the requirement exist and pass, type hints and docstrings with units, config instead of constants, docs/ADR updated, and a stop report written.

## Commits
Conventional messages that name the requirement ID, for example: feat(FR-12): backtest engine.

## When stuck
If the same test fails after three honest attempts, stop and report what you tried. Do not work around it.

## Repository notes
- ATU 2 lives in atu2/, configs/, tests/ (except tests/test_orb_v2.py), docs/, scripts/ and .github/. Everything else is legacy ATU 1 (ORB v2, RL) and stays untouched.
- Synthetic markets for tests: tests/synthetic/generators.py (random_walk, planted_trend, regime_switching, to_ohlcv).
- Frozen gate hash check: python scripts/gate_hash.py
