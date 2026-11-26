# Licences

Every dependency and every piece of reference code, with its licence (NFR-11, R19).
Checked from the installed package metadata on 2026-11-26.

## Runtime (atu2)
| Package | Licence |
|---|---|
| numpy | BSD-3-Clause (bundled parts: 0BSD, MIT, Zlib) |
| pandas | BSD-3-Clause |
| PyYAML | MIT |

## Development
| Package | Licence |
|---|---|
| ruff | MIT |
| mypy | MIT |
| pytest | MIT |
| pytest-cov | MIT |
| hypothesis | MPL-2.0 (used as a test tool, not distributed) |
| types-PyYAML | Apache-2.0 |
| pandas-stubs | BSD-3-Clause |
| detect-secrets | Apache-2.0 |
| pre-commit | MIT |

## Later milestones (to record when added)
| Package | Expected licence | Milestone |
|---|---|---|
| pyarrow | Apache-2.0 | M1 |
| scipy | BSD-3-Clause | M2 |
| torch | BSD-3-Clause | M4 |

## Reference code
| Repository | Licence | Status |
|---|---|---|
| kieranjwood/trading-momentum-transformer | **unchecked** | Check before M3. Until confirmed, we only run it in `ref/`; no code is copied into `atu2/`. |
| X-Trend reference repository | MIT according to the plan | Not yet verified by us; check before M6. |
