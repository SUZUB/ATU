# ATU 2: Build Plan (requirements, design, tests, risks)

Nov 26, 2026 · @Pranav

## Purpose, scope and locked decisions

This plan defines ATU 2 end to end so the build is a series of small, pre-agreed tasks: a cost-aware deep trend-following system, built on the published Oxford-Man line of work, judged by one strict gate before any money is involved. It follows the study doc (ATU 2: Research Study and System Design) and borrows its structure from standard requirements-and-design practice: requirements with acceptance tests, a layered architecture, a test strategy, a risk register, a work breakdown with gates, and change control.

### Goal

Produce a reproducible research platform that trains and evaluates daily-bar trend-following models on a liquid ETF universe, and answers one question honestly: does a cost-aware Momentum Transformer-style model earn a positive net Sharpe ratio at 1 to 3 bps of cost on data it has never seen?

### In scope (version 1)

- Daily bars for a fixed universe of about 40 liquid ETFs, from Alpaca, cached locally.
- Baselines, the Momentum Transformer family, a cost-aware loss, a no-trade band, volatility targeting.
- A backtest engine, a cost model, and an evaluation module (bootstrap, deflated Sharpe, placebo, cost sweep).
- Paper trading through Alpaca after the gate is passed.

### Out of scope (version 1)

- Real-money trading, leverage beyond the configured cap, options and futures execution.
- Intraday models (the ORB work stays a separate satellite, not part of this build).
- Reinforcement learning, LLM sentiment, and foundation-model alpha.
- Indian-market instruments and Indian broker integration.
- Any claim of reproducing the papers' exact numbers.

### Locked decisions

These are fixed so the build does not reopen them. Changing one needs an architecture decision record (section 9).

| ID | Decision | Reason | Reversible? |
|---|---|---|---|
| D1 | New package `atu2/` in the existing ATU repo, legacy code untouched | Keeps history and Colab setup, avoids breaking ORB v2 | Yes |
| D2 | Python 3.11, PyTorch for our model code | One framework we control; the reference repo runs in its own isolated environment | Costly later |
| D3 | Daily bars only, signal at close, return earned over the next close-to-close | Matches the papers; costs stay small | Yes |
| D4 | Primary data: Alpaca daily bars with split and dividend adjustment, stored as Parquet | Free, consistent with ORB v2 | Yes |
| D5 | Reproduction data: the free Nasdaq Data Link continuous futures set that the reference repo uses (availability to confirm in week 1) | Lets us check the reference code against the paper | Yes |
| D6 | Success metric: net Sharpe at 1, 2 and 3 bps on a frozen holdout, deflated for trial count | The papers' pre-cost numbers do not survive costs | No (frozen) |
| D7 | Gate thresholds are those in the study doc, frozen in an ADR before the holdout is touched | Prevents moving the goalposts | No (frozen) |
| D8 | No secrets in files, notebooks or logs; keys only via environment variables or a prompt | Standing security rule | No |
| D9 | Paper trading only until the gate and three months of paper results pass | Base rates and Indian regulatory exposure | No |

### Assumptions to verify in week 1

- The reference repo still runs on current Python (it is a few years old) and the free futures dataset is still downloadable.
- Alpaca's free plan still serves daily history for the whole ETF list.
- ETF history is long enough (many start around 2004 to 2008) to hold a meaningful validation and a 2-year holdout.

## Requirements

There are 22 functional and 12 non-functional requirements. Each has an ID, a MoSCoW priority (Must, Should, Could) and an acceptance test, so every task in the build can point at the requirement it satisfies.

### Functional requirements

| ID | Requirement | Priority | Acceptance test |
|---|---|---|---|
| FR-01 | Download daily bars for the universe from Alpaca, split and dividend adjusted, cached as Parquet, with incremental updates | Must | Second run downloads nothing new; a known split date shows no price jump |
| FR-02 | Validate data: no gaps beyond market holidays, no non-positive prices, high at least low, flagged outliers; fail loudly | Must | Injected bad rows each raise a named error |
| FR-03 | Point-in-time universe: an asset joins only after it has enough history; list fixed in config, not chosen by hindsight | Must | Asset with 100 days of history is excluded until day 253 |
| FR-04 | Feature builder: vol-scaled returns (1, 21, 63, 126, 252 days), MACD pairs, 60-day EWM volatility; deterministic and causal | Must | Truncating future data leaves every earlier feature unchanged |
| FR-05 | Changepoint module: severity and location scores at 21 and 126 days | Should | Planted regime break scores higher than calm data |
| FR-06 | Cross-asset rank features | Should | Ranks sum correctly and ignore assets not yet listed |
| FR-07 | Dataset builder: sliding windows, expanding-window splits, embargo, frozen holdout | Must | No train sample overlaps a test sample; holdout never read before the final run |
| FR-08 | Baselines: vol-scaled time-series momentum, long-only, MACD rule, equal-volatility buy and hold | Must | Baseline output equals a hand calculation on synthetic data |
| FR-09 | Models: LSTM position network and decoder-only TFT-style (LSTM plus multi-head attention) with tanh position output | Must | Output shape and bounds (-1 to 1) hold for random input |
| FR-10 | Losses: Sharpe, portfolio Sharpe, turnover penalty, long-only correlation penalty, each weight configurable | Must | Gradient check passes; perfect signal gives the lowest loss |
| FR-11 | Training loop: seeded, early stopping on validation Sharpe, checkpointing | Must | Same seed reproduces the same weights on CPU |
| FR-12 | Backtest engine: decide at close t, earn return t to t+1, charge costs on position change | Must | Matches a hand calculation on a 5-day example |
| FR-13 | Cost model: linear basis points per asset class, optional short-borrow cost, all configurable | Must | Zero turnover gives zero cost; doubling bps doubles cost |
| FR-14 | Execution rules: no-trade band, volatility target, per-asset and total caps | Must | Caps never exceeded in a property test |
| FR-15 | Evaluation: Sharpe, Sortino, drawdown, turnover, hit rate, cost sweep, break-even cost, block bootstrap, deflated Sharpe, placebo | Must | Noise data fails; planted edge passes |
| FR-16 | Reference harness: run the reference repo's code and compare with our baselines | Should | Report shows the gap to the paper's numbers |
| FR-17 | Experiment registry: every run logs config hash, data hash, git commit, seed, metrics and the running trial count | Must | Any past result can be re-run from its log entry |
| FR-18 | One-page run report with a PASS or FAIL against the frozen gate | Must | Report renders for a finished run with no manual steps |
| FR-19 | Paper-trading adapter for Alpaca: daily rebalance orders and reconciliation | Should | Dry run against recorded responses matches expected orders |
| FR-20 | Kill switch and alerts for drawdown breach, data fault, order rejection | Should | Simulated faults trigger the switch |
| FR-21 | Optional feature plug-ins (levels, seasonality, Kronos, X-Trend context) behind flags with an ablation runner | Could | Each plug-in can be switched off with identical baseline results |
| FR-22 | Command line: data, features, train, backtest, evaluate, report, paper | Must | Each command runs end to end on the synthetic sample |

### Non-functional requirements

| ID | Requirement | Priority | Acceptance test |
|---|---|---|---|
| NFR-01 | Reproducibility: same config, data and seed give the same metrics on CPU | Must | Two runs differ by under 1e-6 |
| NFR-02 | No look-ahead by construction, enforced by automated tests | Must | Leakage tests in CI all pass |
| NFR-03 | Performance targets (to be measured): feature build under 2 minutes, backtest under 10 seconds, one training run under 30 minutes on a free Colab GPU | Should | Timed in the benchmark test |
| NFR-04 | Security: keys only in environment variables or a prompt, never logged, secret scan on every commit | Must | Scanner finds nothing; logs contain no key patterns |
| NFR-05 | Maintainability: type hints, linter clean, short modules, docstrings with units | Should | Lint and type checks pass in CI |
| NFR-06 | Test coverage at least 85% on data, features, losses, backtest, evaluation | Must | Coverage gate in CI |
| NFR-07 | Portability: runs on Linux, Colab and Windows through WSL | Should | CI on Linux; Colab smoke notebook |
| NFR-08 | Observability: structured logs with a run ID on every line | Should | Log sample parsed in a test |
| NFR-09 | Data versioning: every dataset has a content hash recorded with each run | Must | Hash changes when one price changes |
| NFR-10 | Documentation: README, architecture decision records, runbook | Must | Docs checklist in the definition of done |
| NFR-11 | No paid service required in version 1 | Must | Fresh checkout runs with free accounts only |
| NFR-12 | Compliance: research and paper trading only; live trading needs a written review | Must | Live order path disabled by default and tested |

## Architecture

The code is a stack of layers in which every module may import only from the layers below it. That one rule keeps changes local: swapping the data source or adding a model never touches the backtest or the evaluation.

module dependencies · 7 rows, downward imports only

Read it top to bottom: a box may import from any box below it and never from one above. The command line sits on top and `core` at the bottom, so each lower layer can be tested on its own.

### Package layout

```
atu2/
  core/         config, types, hashing, logging, calendars, errors
  data/         alpaca_client, cache, validators, universe
  features/     returns, macd, changepoint, ranks, extras/ (optional plug-ins)
  datasets/     windows, splits, scalers, holdout guard
  models/       lstm, tft, losses, registry
  training/     trainer, checkpoints, seeds
  execution/    no_trade_band, vol_target, caps, kill_switch
  backtest/     engine, costs, baselines
  evaluation/   metrics, bootstrap, dsr, placebo, cost_sweep, report
  experiments/  registry, runner, trial_counter
  paper/        broker_adapter, reconcile, scheduler
  cli.py        command line entry point
configs/        YAML files, one per experiment
tests/          mirrors atu2/, plus synthetic/ data generators
docs/           architecture decision records (adr/), runbook, this plan
```

### Interface contracts

Every module talks to the next through a small typed object, so a module can be replaced without changing its neighbours.

| Boundary | Object passed | Guarantees |
|---|---|---|
| data to features | `PriceFrame`: date index, one column per asset, fields open, high, low, close, volume, adjusted flag | Sorted, no duplicate dates, no gaps except holidays, validated |
| features to datasets | `FeatureFrame`: date index by asset, named columns, plus `feature_spec` listing each feature and its lookback | Causal: value at date t uses data up to t only |
| datasets to models | `Batch`: tensors of shape (batch, window, assets or features), plus mask and dates | Windows never cross a split boundary |
| models to execution | `TargetPositions`: date by asset, values in -1 to 1 | Bounded, finite, no NaN |
| execution to backtest | `Orders`: date by asset, position after band, vol target and caps | Caps respected, turnover recorded |
| backtest to evaluation | `ReturnSeries`: gross and net daily returns, positions, turnover, cost per day | Net equals gross minus cost exactly |
| evaluation to report | `Metrics` dict plus `GateResult` (PASS or FAIL with reasons) | Thresholds read from the frozen gate config |

### Data schemas

| Dataset | Format and location | Key columns |
|---|---|---|
| Raw bars | Parquet, `data/raw/<ticker>_1d.parquet` | date, open, high, low, close, volume, adj_close |
| Clean bars | Parquet, `data/clean/prices.parquet` | date, ticker, adj_close, return, valid flag |
| Features | Parquet, `data/features/<feature_set_hash>.parquet` | date, ticker, one column per feature |
| Run record | JSON lines, `runs/registry.jsonl` | run_id, git_sha, config_hash, data_hash, seed, trial_index, metrics |
| Run artefacts | Folder `runs/<run_id>/` | config.yaml, weights, returns.parquet, report.html |

Rules: raw files are never edited after download; every derived file's name contains the hash of what it was built from; timestamps are stored in UTC and dates in the exchange calendar.

## Detailed design per module

Each module below states its job, its key rules and the edge cases the build must handle, so none of them are discovered mid-build.

### data

- **Download:** Alpaca daily bars with split and dividend adjustment; request in chunks, pause between calls, back off on rate-limit errors, never write today's unfinished bar.
- **Cache:** one Parquet file per ticker; append only; on every update re-fetch the last few days and compare, because a mismatch means a split or dividend rewrote history and the file must be rebuilt.
- **Validation:** reject non-positive prices, high below low, zero volume runs, daily moves beyond a configured threshold (flag, then require a human decision), and calendar gaps. A second source (a free daily-price provider) cross-checks closes; disagreement above a tolerance fails the build.
- **Universe:** about 40 ETFs spanning US and international equities, bonds, credit, gold, oil, broad commodities, real estate and currency proxies. The list and each asset's start date live in config. An asset becomes tradable only after 252 days of history.
- **Edge cases:** ETF closures or mergers (document the limitation: free data has no delisted funds), holiday misalignment between assets (use one exchange calendar, forward-fill at most one day, otherwise mark missing), dividends (use total-return adjusted prices throughout).

### features

- **Returns:** daily simple returns, then normalised by 60-day EWM volatility, at horizons 1, 21, 63, 126 and 252 days, as in the reference paper.
- **MACD:** several fast and slow pairs, each divided by recent price volatility. The paper lists inconsistent pairs between its text and exhibits, so the pairs are a config choice we fix and then test, not a copied constant.
- **Changepoint:** severity and location at 21 and 126 days. The reference uses a Gaussian-process method; plan to run theirs first, then decide with an ablation whether a lighter substitute (for example a likelihood-ratio change score) gives the same benefit.
- **Ranks:** for each date, the rank of each asset's momentum among assets already listed.
- **Rule:** every feature declares its lookback in `feature_spec`; a test proves that the value at date t is unchanged if data after t is deleted.

### datasets and splits

- **Windows:** fixed-length sequences per asset (length is a config value taken from the reference repo's setting at build time).
- **Walk-forward:** expanding window. An initial training period, then successive test blocks of about 2 years, retraining at each step. The last 2 years are the frozen holdout and are excluded from every step.
- **Embargo:** a gap of at least the longest feature lookback between train and test so overlapping windows cannot leak.
- **Holdout guard:** a function that raises an error if any code path reads holdout dates before the final evaluation flag is set.

### models and losses

- **Models:** (a) LSTM position network; (b) decoder-only TFT-style model with an LSTM, multi-head attention and a tanh output giving a position between -1 and +1 per asset. Hyperparameter ranges are taken from the reference repo's configs; search is capped at 30 trials and every trial is counted for the deflated Sharpe.
- **Loss:** the training loss combines three terms, described below.

```
r^{p}_t = \frac{1}{N}\sum_{i=1}^{N} z_{i,t}\,\frac{\sigma_{tgt}}{\sigma_{i,t}}\,R_{i,t+1}, \qquad r^{net}_t = r^{p}_t - c\,\frac{1}{N}\sum_{i=1}^{N}\frac{\sigma_{tgt}}{\sigma_{i,t}}\,|z_{i,t}-z_{i,t-1}|

L = -\sqrt{252}\,\frac{\mathrm{mean}(r^{net})}{\mathrm{std}(r^{net})} + \lambda_{turn}\,\mathrm{mean}|z_{t}-z_{t-1}| + \lambda_{corr}\,\bigl|\mathrm{corr}(r^{p}, r^{long})\bigr|
```

Here z is the model's position, R the next-day return, sigma the ex-ante volatility, c the cost per unit of turnover and r^long a long-only benchmark. Setting c, lambda_turn and lambda_corr to zero recovers the papers' plain Sharpe loss, which is what the reference run uses.

### execution and risk

- **No-trade band:** keep yesterday's position unless the new target differs by more than a band width (a config value tuned on validation only).
- **Volatility target:** scale the whole book to a fixed annual volatility; leverage capped.
- **Caps:** per-asset weight cap and total gross exposure cap, applied after the band.
- **Kill switch:** flatten to cash if drawdown exceeds a limit, or if data is stale or invalid.

### backtest

- **Timing:** the signal is computed at the close on day t from data up to t; the position earns the close-to-close return of day t+1. This is a simplification of real fills, so the paper-trading phase measures the slippage it ignores.
- **Costs:** linear basis points on the absolute position change, per asset class; short positions also pay a borrow cost parameter. Always report results at 0, 1, 2, 3 and 5 bps.
- **Baselines:** vol-scaled time-series momentum (sign of 12-month return), MACD rule, long-only, equal-volatility buy and hold. Every model is compared with these on identical dates.

### evaluation

- **Metrics:** annualised return and volatility, Sharpe, Sortino, maximum drawdown, turnover, hit rate, break-even cost.
- **Statistics:** block bootstrap confidence interval; deflated Sharpe using the registry's trial count; placebo test (shuffled or random-direction signal must show no skill). The bootstrap, deflated Sharpe and placebo logic are ported from the existing `orb_eval.py`, which already has tests.
- **Seeds:** every reported number is the median over at least 5 seeds, with the spread shown.
- **Gate:** the PASS or FAIL rule reads thresholds from a frozen config file whose hash is stored in every report.

## Configuration, reproducibility and environments

Every result must be re-creatable from three things: a config file, a data hash and a git commit. Nothing else is allowed to influence a run.

### Configuration

- One YAML file per experiment under `configs/`, validated by a typed schema at load time; unknown keys are errors, not warnings.
- The config is hashed and the hash is stored in the run record. The gate thresholds live in a separate frozen file with its own hash.
- No magic numbers in code: lookbacks, band width, caps, cost levels, loss weights and split dates are all config values.

### Reproducibility rules

- Seed every random source (Python, NumPy, PyTorch) from the config; log the seed.
- Pin dependencies in a lock file and record the Python version; the CI environment is built from that lock.
- Use deterministic settings on CPU; on GPU report the seed-to-seed spread instead of promising exact equality.
- Never train on data whose hash is not in the registry; the trainer refuses unknown data.
- Save the config, weights, return series and report with every run so it can be audited later.
- Count every trial in the registry (including failed and discarded ones); the deflated Sharpe depends on that count.

### Environments

| Task | Where | Why |
|---|---|---|
| Data, features, backtests, tests | Local machine or any CPU | Cheap and fast; no GPU needed |
| Model training sweeps | Colab GPU (free tier), checkpoints to Google Drive | GPU helps training; sessions can disconnect, so checkpoint every epoch |
| Reference repo run | Separate virtual environment | Keeps its old dependencies from contaminating ours |
| Scheduled paper trading | A scheduled job (for example GitHub Actions) or a small always-on machine | Colab sessions are not reliable schedulers |

### Colab-specific rules

- All code lives in the repository; notebooks only call the command line. The existing Colab setup cell hard-resets to the remote main branch, so anything not pushed is lost.
- API keys are typed into a hidden prompt each session and never saved in the notebook or Drive.
- Mount Drive for the data cache and run folders so a disconnect does not cost a download or a training run.

### Dependencies and licences

Record the licence of every dependency and of any reference code we copy from. The X-Trend repository carries an MIT licence; the Momentum Transformer repository's licence is not yet confirmed and is checked in week 1 before any code is copied.

## Test strategy

The tests exist to catch the failures that fool trading projects most: look-ahead leakage, cost mistakes and selection luck. Most of the suite runs in seconds on synthetic data, so it can run on every commit.

### Test levels

| Level | What it covers | Runs when |
|---|---|---|
| Unit | One function: a feature, a loss term, a cost calculation | Every commit |
| Property-based | Invariants over random inputs: positions stay within caps, scaling volatility scales positions | Every commit |
| Leakage | Proves the future cannot influence the past | Every commit |
| Golden | Output matches a hand-calculated or independently computed reference | Every commit |
| Integration | A full pipeline on a 3-asset synthetic market, under 60 seconds | Every commit |
| Statistical | Placebo on noise fails, planted edge passes | Every commit (seeded) |
| Regression | Metrics of the baselines on frozen sample data stay within tolerance | Before merge |
| Slow | Real-data training runs, multi-seed sweeps | Manually, before a milestone gate |

### Named tests the build must include

| ID | Test | Passes when |
|---|---|---|
| T-LEAK-1 | Truncate data after date t, rebuild features | Features up to t are identical |
| T-LEAK-2 | Scramble all prices after t, run the model | Positions up to t are identical |
| T-LEAK-3 | Request holdout dates before the final flag | The holdout guard raises an error |
| T-SPLIT-1 | Build walk-forward splits | No train window overlaps a test window; embargo respected |
| T-DATA-1 | Feed a series with a known split | Adjusted prices show no jump; unadjusted data is rejected |
| T-DATA-2 | Inject each bad-data type | Each raises its named validation error |
| T-LOSS-1 | Numerical gradient check on each loss term | Analytic and numeric gradients agree |
| T-LOSS-2 | Perfect-foresight positions on synthetic data | Loss is lower than for random positions |
| T-BT-1 | Constant position, known returns | Equity curve equals the hand calculation |
| T-BT-2 | Zero turnover; then double the cost | Cost is zero; then exactly doubles |
| T-EXE-1 | Random targets and extreme volatility | Caps and the band are never violated |
| T-EVAL-1 | Pure-noise returns through the gate | Result is FAIL and placebo shows no skill |
| T-EVAL-2 | Planted trend through the gate | Result is PASS |
| T-DET-1 | Run twice with the same seed on CPU | Metrics match to 1e-6 |
| T-REG-1 | Registry after a failed run | The trial count still increases |
| T-SEC-1 | Scan repository and logs | No key patterns found |
| T-E2E-1 | Command line from data to report on synthetic data | Report produced, no manual step |

### Synthetic data generators

Three generators live in `tests/synthetic/`: a pure random walk (no edge), a market with a planted persistent trend (known edge), and a market with regime switches (to exercise the changepoint and attention parts). Because the true answer is known, they let us prove the pipeline finds an edge when one exists and finds nothing when it does not.

### Continuous integration

- On every pull request: formatting, linter, type check, secret scan, unit, property, leakage, golden, integration and statistical tests, and a coverage gate (85% on core modules).
- A failing check blocks the merge; no one merges around it.
- Slow tests are tagged and run manually, and their output is attached to the milestone report.
- Optional later: mutation testing on the backtest and loss modules, to check the tests really detect bugs.

### Review checklist (used on every pull request)

- Could this use information from after the decision time?
- Are costs applied, and in the right units?
- Is every random source seeded and logged?
- Does it count as a trial in the registry?
- Is anything hard-coded that should be config?
- Is there a test, and does it fail without the change?

## Risk register and complications

The biggest risk is that no model beats the simple baseline after costs. The plan treats that as an acceptable answer: the gates are built so the project stops cheaply, with a clear result, instead of tuning forever.

| ID | Risk | Likelihood | Impact | Early warning | Mitigation |
|---|---|---|---|---|---|
| R01 | No edge after costs on ETFs: the deep model does not beat plain time-series momentum | High | High | Baseline net Sharpe at 2 bps already near zero at milestone M2 | Cost sweep from day one; stop rule at gate B; report the negative result as the deliverable |
| R02 | Overfitting through many trials and tuning | High | High | Best validation Sharpe far above holdout; trial count climbing | Frozen holdout, deflated Sharpe, hard cap of 30 tuning trials, registry counts everything |
| R03 | Look-ahead leakage | Medium | Critical | A result that looks too good (Sharpe above 2 net) | Leakage tests T-LEAK-1 to 3, review checklist, treat any implausibly good result as a bug first |
| R04 | Bad or mis-adjusted data (splits, dividends, gaps) | Medium | High | Return spikes on known corporate-action dates | Validators, second-source cross-check, total-return adjusted prices, test T-DATA-1 |
| R05 | ETF history too short for enough market regimes | High | Medium | Fewer than 15 years for most assets | Use the longest-history ETFs for the core, report regime coverage, make narrower claims; use the futures set only for reproducing the paper |
| R06 | Reference repo is stale: old dependencies, dataset removed | Medium | Medium | Install or download fails in week 1 | Isolated environment; time-box reproduction to 4 days; fall back to implementing from the paper and compare on our own baselines |
| R07 | Colab limits: disconnects, GPU quota, slow training | High | Low | Sessions ending mid-run | Small models, per-epoch checkpoints to Drive, CPU-capable code path, runs resumable |
| R08 | Colab setup cell resets the repo and wipes unpushed work | Medium | Medium | Missing local edits after a restart | All work committed and pushed; notebooks only call the command line |
| R09 | GPU non-determinism makes runs irreproducible | Medium | Medium | Same seed gives different metrics | Report median and spread over at least 5 seeds; exact reproduction only claimed on CPU |
| R10 | Survivorship and selection bias in the universe | Medium | Medium | Universe chosen after seeing performance | Universe fixed by written rule before any backtest; limitation noted (no delisted funds in free data) |
| R11 | Trend crash or long drawdown (the index of trend funds fell 20.4% over May 2024 to May 2025) | Medium | High | Drawdown exceeding the limit in backtest or paper | Position and gross caps, kill switch, vol target, set expectations in advance |
| R12 | Broker API changes or rate limits | Medium | Low | Contract tests fail; 429 errors | API client behind an interface, recorded-response tests, cache, backoff |
| R13 | API keys leaked in code, notebook or logs | Low | Critical | Secret scanner hit | Env vars or prompt only, pre-commit and CI secret scan, rotate keys at once if exposed |
| R14 | Scope creep: adding Kronos, X-Trend, LLM ideas before the core works | High | Medium | New work items outside the plan | Change control (section 9); extras are flags, allowed only after gate B |
| R15 | Moving the goalposts after seeing results | Medium | High | Thresholds edited after the holdout run | Thresholds frozen by hash in an ADR before the holdout is opened; any change resets the trial count |
| R16 | Execution gap: signal at close but real fills differ | High | Medium | Paper slippage far above modelled cost | Slippage tracked in paper phase; cost assumption raised and the backtest re-run |
| R17 | Regulatory, tax or remittance issues for the user in India | Medium | High | Any step moving from paper to real money | Paper only (D9); take professional advice before any live step |
| R18 | Limited time and energy (studies, internship) | High | Medium | A milestone slipping by more than a week | Small gated milestones; each ends in a usable, documented state; pause points are explicit |
| R19 | Dependency or licence conflict | Low | Medium | Unknown licence on code we want to reuse | Check licences in week 1; reimplement rather than copy when unclear |
| R20 | Bugs in the evaluation code itself producing false confidence | Medium | High | Placebo test passes on noise | Statistical tests T-EVAL-1 and T-EVAL-2; ported code keeps its existing tests; independent check by hand on a small case |

### Complications to design for from the start

- **Calendars:** assets trade on different holidays; use one calendar and a stated fill rule.
- **Corporate actions:** use adjusted prices consistently and never mix adjusted and unadjusted series.
- **Short selling ETFs:** shorting has borrow costs and limits; the backtest includes a borrow cost and a long-only variant is always reported alongside.
- **Warm-up periods:** the first 252 days of any asset are unusable; the code drops them explicitly rather than filling with zeros.
- **Volatility floor:** when volatility is near zero, scaling by it explodes positions; enforce a floor and a leverage cap.
- **Rebalance timing:** month-end and holiday effects, half-days, and daylight-saving shifts for the scheduler.
- **Fractional shares and minimum order sizes** in paper and live trading.
- **Model drift:** a retrain schedule and a check that live feature distributions match training.

## Work breakdown, milestones and gates

The build is nine milestones, about 34 working days of focused effort plus a 3-month paper-trading period, with three hard gates where the project can stop. Estimates are rough (plus or minus 50%) and assume one person working with Claude Code on small tasks.

milestones and gates · 9 milestones, 3 gates

The top row is the critical path. M3 runs in parallel with M2, and after Gate B the plan splits into paper trading (M7) and optional extras (M6).

### Milestones

| ID | Milestone | Main tasks | Depends on | Estimate (days) | Exit criteria |
|---|---|---|---|---|---|
| M0 | Foundation | Repo skeleton, config loader, logging, hashing, CI, secret scan, first ADRs, licence check | none | 3 | CI green on an empty pipeline; ADRs 1 to 8 merged |
| M1 | Data layer | Alpaca client, cache, validators, universe, calendar, second-source check | M0 | 4 | 40 ETFs cached; validation report clean; T-DATA tests pass |
| M2 | Baselines, backtest, evaluation | Baselines, cost model, backtest engine, metrics, bootstrap, deflated Sharpe, placebo, cost sweep, report | M1 | 5 | Baseline table at 0 to 5 bps; T-BT and T-EVAL tests pass. **Gate A** |
| M3 | Reference reproduction (parallel with M2) | Isolated environment, run reference code, import its positions, evaluate with our harness | M0 | 4 | Documented gap to the paper; or documented decision to skip after the time box |
| M4 | Features, models, losses | Features, changepoint, datasets and splits, holdout guard, models, losses, trainer | M2 | 7 | Model finds the planted trend and ignores noise; all leakage tests pass |
| M5 | Experiments | LSTM vs TFT, with and without changepoints, cost-aware loss, no-trade band, 5 seeds each, trial counter | M4 | 6 | Results table at 1 to 3 bps; holdout opened once. **Gate B** |
| M6 | Optional extras | Ablate levels, seasonality, Kronos, X-Trend context, one at a time | Gate B passed | 5 | Each extra kept or dropped with a measured reason |
| M7 | Paper trading | Broker adapter, reconciliation, scheduler, monitoring, runbook; then 3 months of paper running | Gate B passed | 3 build plus 3 months | Paper results within the backtest band. **Gate C** |
| M8 | Release and handoff | Final report, docs, tag v1.0, lessons learned | Gate C or an early stop | 2 | Fresh checkout reproduces the headline table |

### Gates and stop rules

| Gate | Pass condition | If it fails |
|---|---|---|
| A (after M2) | Evaluation code proven on noise and planted data; baselines reproducible | Fix the harness; nothing else proceeds |
| B (after M5) | Net Sharpe positive at 2 bps on the frozen holdout, deflated Sharpe probability at least 0.95, placebo clean, break-even cost at least 2x assumed cost, neighbouring settings keep half the Sharpe | Stop. Publish the negative result and keep the baselines; do not tune further on the holdout |
| C (after M7) | Three months of paper results inside the backtest confidence band | Return to M5 with the paper data as a new diagnostic; no live money |

### Critical path

M0, M1, M2, M4, M5, M7, M8 (about 30 days). M3 and M6 are off the critical path: M3 can run alongside M2, and M6 is optional.

### Definition of done

A task is done only when all of these hold:

- Code merged through a reviewed pull request with a green CI run.
- Tests exist for the requirement it satisfies, and the requirement ID is in the commit message.
- Type hints, docstrings (with units) and config entries in place, no hard-coded constants.
- Any new experiment is logged in the registry.
- Docs updated: README section, ADR if a decision was made.

A milestone is done when its exit criteria hold, a tag is created, and a short milestone report (what was built, what changed from the plan, numbers) is added to the docs.

## Engineering process

The process is deliberately light: small tasks, short-lived branches, automated checks and written decisions. It is meant to keep changes minimal during the build, not to add paperwork.

### Work flow

- **Issue:** every task is an issue that names the requirement IDs it satisfies and the tests that prove it.
- **Branch:** one short-lived branch per issue (`feat/FR-12-backtest-engine`), merged within days, never long-running.
- **Commit:** small commits with conventional messages (`feat`, `fix`, `test`, `docs`, `refactor`) that mention the requirement ID.
- **Pull request:** a template with the review checklist from the test strategy; CI must be green.
- **Review:** every change is reviewed against the checklist (by you, and by a second Claude pass focused on leakage and costs) before merging into main.
- **Merge and tag:** squash-merge into main; tag at each milestone.

### Using Claude Code on this plan

- Give it one task at a time: the issue text, the requirement IDs, the interface contract and the acceptance test. Ask for the failing test first, then the code.
- Do not let it change a contract, a frozen threshold or the gate; those require an ADR.
- Require it to run the full test suite before it says a task is done, and to show the output.
- Keep the repository instructions file (CLAUDE.md) short: layering rule, test commands, the no-leak rule, the no-secrets rule.

### Change control

| Change type | Process |
|---|---|
| Bug fix inside a module | Normal pull request |
| New feature or plug-in | Issue linked to a requirement; if not in this plan, add the requirement first |
| Change to an interface contract | ADR plus update of every dependent test |
| Change to gate thresholds, universe, holdout dates | ADR, and the trial count is reset; not allowed after the holdout is opened |
| Change to scope (new milestone, new data source) | Update this plan, note the effect on the critical path |

### Architecture decision records

Each decision is a short file in `docs/adr/` with context, decision, alternatives and consequences. The first eight, written in M0:

| ADR | Decision |
|---|---|
| 0001 | Package layout and the downward-only import rule |
| 0002 | Daily bars and close-to-close timing |
| 0003 | PyTorch for our models; reference repo isolated |
| 0004 | Data sources, adjustment policy and cache format |
| 0005 | Universe rule and warm-up rule |
| 0006 | Loss definition and cost model |
| 0007 | Evaluation protocol: splits, holdout, trial counting |
| 0008 | Frozen gate thresholds (with their hash) |

### Versioning and releases

Semantic versions for the package (`0.x` until the first gate passes, `1.0` at release). Data and configs are versioned by content hash, not by name. Every report prints the package version, commit, config hash and data hash.

### Code quality rules

- Layering: imports go downward only; a lint rule enforces it.
- Small modules, one responsibility each; functions pure where possible (no hidden global state).
- Type hints on all public functions; docstrings state units (basis points versus fractions, daily versus annual).
- No silent failures: invalid data or config raises a named error.
- Logging, not printing; no secrets in any log line.

### Documentation

README (how to run), this plan, ADRs, the runbook (next section), and a one-page glossary of terms (Sharpe, deflated Sharpe, turnover, bps) so the project is readable later.

## Operations: paper-trading runbook

The paper phase runs one job a day after the US close, compares reality with the backtest, and can flatten everything with one command. It uses simulated money only.

### Daily cycle

- **Trigger:** after the US market close (about 02:30 IST in winter, 01:30 IST in summer; the scheduler works in US Eastern time so daylight-saving shifts are handled).
- **Data:** pull the day's bars, validate, stop on any failure.
- **Signals:** build features, run the frozen model, apply band, volatility target and caps.
- **Orders:** compute the difference from current paper positions and submit orders for the next open.
- **Reconcile:** compare broker positions with the intended ones; log any gap.
- **Record:** write positions, orders, fills, slippage and the day's return to the registry.
- **Report:** a short daily summary; a weekly comparison with the backtest.

### Scheduling

Use a scheduled job (for example a GitHub Actions workflow or a small always-on machine), not a Colab notebook, because Colab sessions end unpredictably. The model weights are loaded from a versioned file; the job never retrains on its own.

### Monitoring

| Signal | Watch for | Action |
|---|---|---|
| Data freshness | Missing or stale bars | Skip trading, alert, keep positions |
| Position drift | Broker differs from intended | Re-send orders once, then alert |
| Turnover vs backtest | More than 2x expected | Investigate before the next run |
| Slippage vs model | Paper cost above assumption | Raise cost assumption, re-run backtest |
| Drawdown | Beyond the configured limit | Kill switch: flatten to cash, require manual restart |
| Feature drift | Live feature distribution far from training | Alert; consider retrain through the normal process |

### Incident playbooks

- **Data provider down or API errors:** hold positions, retry with backoff, alert after three failures; never trade on partial data.
- **Model output invalid (NaN, out of range):** fall back to flat positions for that run and alert.
- **Order rejected or partially filled:** log, retry once, reconcile next day.
- **Suspected key exposure:** revoke keys immediately, rotate, review logs.
- **Unexpected large loss:** trigger the kill switch, freeze changes, write a short post-mortem.

### Rollback and change freeze

Every deployed model has a version and hash; rolling back means pointing the job at the previous version and flattening positions if needed. During the 3-month paper period nothing changes except bug fixes, and each fix is logged, so the paper results stay comparable with the backtest.

### Exit from paper to anything real

Not part of this plan. It needs Gate C, a written review of costs, tax and regulatory rules for the user's country, and a separate decision with a small size limit.

## Open decisions and first-week checklist

Seven decisions need your confirmation before M0 starts; each has a recommended default, so a one-word "agreed" is enough.

### Decisions to confirm

| # | Question | Recommended default |
|---|---|---|
| 1 | Where does the code live? | New `atu2/` package in the ATU repo (D1) |
| 2 | Universe rule | Liquid ETFs with at least 12 years of history and high daily volume, spread across equities, bonds, credit, gold, oil, commodities, real estate and currencies; about 40, final list fixed by rule in week 1 |
| 3 | Shorting | Allowed in backtest with a borrow-cost parameter; always report a long-only variant too |
| 4 | Volatility target and caps | Annual volatility target set in config (the reference paper uses 15%); caps decided in M2 from baseline behaviour |
| 5 | Cost assumptions | Report 0, 1, 2, 3 and 5 bps; plan on 2 bps as the central case until paper trading measures real slippage |
| 6 | Where heavy runs happen | CPU locally for everything except model sweeps, which go to Colab GPU |
| 7 | Time budget | Which milestones you want to attempt first (suggested: M0 to M2, then stop and review at Gate A) |

### Example universe (to be finalised by the rule in decision 2)

Broad US equities (SPY, QQQ, IWM), international equities (EFA, EEM), US Treasuries at several maturities (SHY, IEF, TLT), credit (LQD, HYG), gold and silver (GLD, SLV), oil and broad commodities (USO, DBC), real estate (VNQ), and currency proxies (UUP, FXE, FXY). These are placeholders; any fund that fails the history or volume rule is replaced, and the list is committed before any backtest runs.

### First-week checklist (M0 and the start of M1)

- [ ] Confirm the seven decisions above.
- [ ] Create the `atu2/` skeleton, linter, type checker, test runner and CI workflow.
- [ ] Add the secret scanner and a pre-commit hook; confirm no keys exist in the repository history.
- [ ] Write ADRs 0001 to 0008, including the frozen gate thresholds and their hash.
- [ ] Check the licence of the Momentum Transformer repository and of every dependency.
- [ ] Try installing the reference repo in an isolated environment and confirm whether the free futures dataset still downloads (start of the 4-day time box for M3).
- [ ] Confirm Alpaca daily history for a sample of the universe on the free plan.
- [ ] Generate the three synthetic datasets and commit them with their tests.
- [ ] Tag `v0.0.1` when CI is green on the empty pipeline.

After this checklist, each later task is a single issue with its requirement IDs and acceptance tests, so building is a matter of working down the milestone table.

## Claude Code prompts, phase by phase

Claude Code builds one milestone at a time from the prompts below, and stops at the end of each one so you can review before the next. I have assumed the recommended defaults for the seven open decisions above; tell me if any differ and I will adjust the prompts.

### Before you start (one time, about 20 minutes)

- Open Claude Code in a local clone of the ATU repository (Python 3.11 installed). Claude Code cannot drive Colab, but this build is CPU-friendly, so the heavy steps run locally; the Colab notebook only calls the command line after you push.
- Download this doc as Markdown (the export option on the doc) and commit it as `docs/BUILD_PLAN.md`. Claude Code cannot open the claude.ai link, so the file is how it reads the plan.
- Save the CLAUDE.md below in the repository root.
- Set your keys as environment variables in your own shell (`ALPACA_API_KEY`, `ALPACA_API_SECRET`). Never paste them into chat or any file. For M3 you also need a free Nasdaq Data Link key, set the same way.
- Work on a branch per milestone and merge only after you have read the stop report.

### CLAUDE.md (repository root)

```
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
```

### Rules for every phase

- Paste exactly one phase prompt at a time, in order. Do not skip a gate.
- Ask Claude Code to write its task plan first and then proceed; if the plan differs from the prompt, ask why before it writes code.
- Every phase ends with a stop report in this format: what changed, requirement IDs satisfied, the exact test output, deviations from the plan, open questions. Read it before merging.
- A phase that needs your keys or your data ends by telling you the exact command to run yourself; you paste back only the summary output, never the keys.

### Prompt M0: Foundation

```
Read CLAUDE.md and docs/BUILD_PLAN.md (sections: Architecture, Configuration, Test strategy, Engineering process). First write a numbered task plan, then carry it out.

Goal: milestone M0, the project foundation. No data or model code yet.

Tasks:
1. Create the atu2/ package skeleton exactly as in the plan's package layout, with module docstrings only.
2. pyproject.toml for Python 3.11 with ruff, mypy, pytest, pytest-cov, hypothesis; pin dependencies in a lock file.
3. core/: typed config loader (YAML, unknown keys raise an error), content hashing for files, dataframes and configs, structured logging with a run id, named error classes, a single exchange-calendar helper.
4. A test that parses every import in atu2/ and fails if any module imports from a layer above it (rule 1 in CLAUDE.md). Add a negative test that proves it fails on a deliberate upward import.
5. tests/synthetic/: three generators (pure random walk, planted persistent trend, regime-switching market), seeded, with tests that check their statistical properties.
6. configs/gate.yaml with the frozen thresholds from the plan (net Sharpe positive at 2 bps on holdout, deflated Sharpe probability at least 0.95, break-even cost at least 2x assumed cost, placebo p-value above 0.10, neighbouring settings keep at least half the Sharpe) and a script that prints its hash.
7. GitHub Actions workflow: ruff, mypy, pytest with a coverage gate of 85 percent on core, plus a secret scan. Add a pre-commit config with the secret scan.
8. docs/adr/0001 to 0008 as short drafts following the list in the plan. docs/licences.md listing each dependency and its licence; note that the Momentum Transformer repo licence is still unchecked.
9. README with a quickstart.

Constraints: no network calls; do not touch legacy files; do not install anything not needed.

Acceptance: ruff check, mypy atu2 and pytest -q --cov=atu2 all pass; the layering test and its negative test pass; the workflow file is valid YAML.

STOP when done. Write the stop report (changes, requirement IDs NFR-01, NFR-02, NFR-04, NFR-05, NFR-06, NFR-10, exact test output, deviations, questions). Do not tag a release; I will tag v0.0.1 after review.
```

### Prompt M1: Data layer

```
Read CLAUDE.md and docs/BUILD_PLAN.md (sections: Detailed design > data, Interface contracts, Data schemas, Risk register R04, R10, R12). Plan first, then build.

Goal: milestone M1, requirements FR-01, FR-02, FR-03, NFR-09. Use TDD: write each test before the code.

Tasks:
1. data/alpaca_client: daily bars with split and dividend adjustment, chunked requests, pause between calls, backoff on rate-limit errors, never write today's unfinished bar. Reuse the ideas (not the intraday specifics) from the existing alpaca_data.py in the repo. Keys only from environment variables; fail with a clear message if missing.
2. data/cache: one Parquet file per ticker, append-only; on update re-fetch the last few days and compare; a mismatch means a split or dividend rewrote history and the file must be rebuilt. Record a content hash per file.
3. data/validators: reject non-positive prices, high below low, runs of zero volume, daily moves above a configured threshold (flag them, require a decision), calendar gaps beyond holidays. Each failure type has its own named error.
4. data/universe: read configs/universe.yaml (rule plus candidate list), enforce the 252-day warm-up per asset, and expose a PriceFrame object exactly as defined in the interface contracts.
5. A second-source cross-check behind an adapter interface (free daily-price provider); disagreement above a tolerance fails. Implement the adapter, but tests use recorded fixtures only.
6. CLI: atu2 data (download, validate, report) and atu2 data universe-check.
7. Tests T-DATA-1 (known split) and T-DATA-2 (each bad-data type), plus recorded-response tests for the client. No network in tests.

Acceptance: all three CLAUDE.md commands pass; coverage on data/ at least 85 percent.

STOP. In the report, give me the exact command to run the real download myself with my environment keys, and say what output summary to paste back (counts, date ranges, validation findings, never keys). Propose the final universe from the rule, and wait for my confirmation before fixing it in configs/universe.yaml.
```

### Prompt M2: Baselines, backtest and evaluation (ends at Gate A)

```
Read CLAUDE.md and docs/BUILD_PLAN.md (sections: Detailed design > backtest, evaluation; Test strategy; Gates). Plan first, then build with TDD.

Goal: milestone M2, requirements FR-08, FR-12, FR-13, FR-15, FR-17 (registry skeleton), FR-18, FR-22 (commands backtest, evaluate, report).

Tasks:
1. backtest/costs: linear basis points per asset class plus an optional short-borrow cost, all from config.
2. backtest/engine: decide at close t from data up to t, earn the close-to-close return of day t+1, charge costs on the absolute position change. Output the ReturnSeries object from the interface contracts (net equals gross minus cost exactly).
3. backtest/baselines: vol-scaled time-series momentum (sign of the 12-month return), MACD rule, long-only, equal-volatility buy and hold.
4. evaluation/: metrics (annualised return and volatility, Sharpe, Sortino, maximum drawdown, turnover, hit rate, break-even cost), cost sweep at 0, 1, 2, 3 and 5 bps. Port the block bootstrap, deflated Sharpe and random-direction placebo from the existing orb_eval.py in the repo, keeping their existing tests and adding tests for the new package path.
5. experiments/registry skeleton: append-only JSON lines with run id, git sha, config hash, data hash, seed, trial index, metrics. Failed runs also increase the trial count.
6. report: a one-page run report (tables and plots) with a PASS or FAIL read from configs/gate.yaml and the gate hash printed.
7. Tests: T-BT-1, T-BT-2, T-EVAL-1 (noise fails), T-EVAL-2 (planted trend passes), T-REG-1, and a golden test of the baselines against a hand calculation.

Constraints: do not read holdout dates; the engine takes dates from the splits layer once it exists, so for now accept an explicit date range.

Acceptance: all three commands pass; T-EVAL-1 and T-EVAL-2 pass on the synthetic generators.

STOP at Gate A. Report the baseline table at all cost levels on synthetic data, and give me the command to run it on the real cache myself and what to paste back. Do not start M4.
```

### Prompt M3: Reference reproduction (time-boxed, can run alongside M2)

```
Read CLAUDE.md and docs/BUILD_PLAN.md (sections: Locked decisions D5, Risk register R06 and R19, Milestones M3). Time box: 4 working days. If it is not working by then, stop and document why.

Goal: run the published Momentum Transformer reference code (github.com/kieranjwood/trading-momentum-transformer) in isolation and measure it with our own evaluation harness.

Tasks:
1. Create an isolated virtual environment under ref/ (git-ignored). Clone the reference repo there. Record its licence in docs/licences.md. If the licence is missing or unclear, do not copy any of its code into atu2/; we only run it.
2. Follow its README: download the free futures dataset (needs my Nasdaq Data Link key from an environment variable; tell me when you need it, never print it), optionally run its changepoint step, create features, run one experiment.
3. If the install or dataset fails, record exactly what failed and what version issues exist, then stop. Do not patch the world to make it work for more than half a day.
4. If it runs, export its daily positions to CSV and evaluate them with our harness (cost sweep, bootstrap, placebo). Compare the numbers with the paper's published ones (for the version with changepoints: Sharpe 2.62 over 1995 to 2020 and 2.00 over 2015 to 2020 before costs; 1.22, 0.44 and -0.35 at 1, 2 and 3 bps over 2015 to 2020).
5. Write docs/reference_repro.md: what ran, what differed, the gap to the paper, and the settings we can reuse (window length, hyperparameter ranges, MACD pairs, changepoint lookbacks).

Constraints: nothing in ref/ is imported by atu2/; no keys in files.

STOP and report. Say whether the reference reproduces, partly reproduces or fails, and what that means for M4.
```

### Prompt M4: Features, datasets, models, losses, training

```
Read CLAUDE.md and docs/BUILD_PLAN.md (sections: Detailed design > features, datasets and splits, models and losses; Test strategy; docs/reference_repro.md if it exists). Plan first, then TDD.

Goal: milestone M4, requirements FR-04 to FR-07, FR-09 to FR-11, NFR-02, NFR-01.

Tasks:
1. features/: vol-scaled returns (1, 21, 63, 126, 252 days, 60-day EWM volatility), MACD pairs from config, cross-asset ranks, and a changepoint module (start with a likelihood-ratio change score at 21 and 126 days; keep it swappable). Each feature declares its lookback in feature_spec. Volatility floor and warm-up handling as in the plan.
2. datasets/: sliding windows, expanding-window walk-forward splits with an embargo of at least the longest lookback, and the holdout guard (the last 2 years raise an error unless the final flag is set).
3. models/: an LSTM position network and a decoder-only TFT-style model (LSTM plus multi-head attention) in PyTorch, tanh output between -1 and +1. Hyperparameter defaults from docs/reference_repro.md if available, otherwise from the plan, written down in an ADR.
4. models/losses: portfolio Sharpe with the turnover and long-only-correlation penalties exactly as in the plan's formula; each weight configurable; weights at zero must recover the plain Sharpe loss.
5. training/: seeded trainer with early stopping on validation Sharpe, checkpoints, resume.
6. Tests: T-LEAK-1, T-LEAK-2, T-LEAK-3, T-SPLIT-1, T-LOSS-1, T-LOSS-2, T-DET-1, plus shape and bounds tests for models.
7. A synthetic experiment: on the planted-trend market the trained model must beat the baselines net of costs, and on the random-walk market it must not (use the placebo and the gate logic). Report both results.

Constraints: CPU must work; GPU is optional. Do not read holdout dates. Do not tune on anything but validation periods.

Acceptance: all three commands pass; all T-LEAK tests pass; the synthetic experiment behaves as described.

STOP. Report the synthetic results and the training time on CPU for one run, so I can decide whether to use Colab for M5.
```

### Prompt M5a: Experiments on validation data only

```
Read CLAUDE.md and docs/BUILD_PLAN.md (sections: Evaluation, Gates, Risk register R01, R02, R15). Plan first.

Goal: milestone M5, development half. Build the experiment runner and run all comparisons on validation periods only. The holdout must stay closed.

Tasks:
1. experiments/runner: take a config, run walk-forward training and evaluation, log every run and every tuning trial in the registry (the trial count feeds the deflated Sharpe). Hard cap of 30 tuning trials in total.
2. Experiment configs for: the four baselines; the LSTM; the TFT-style model; TFT with changepoint features; TFT with the cost-aware loss; TFT with cost-aware loss and the no-trade band. Five seeds each; report the median and the spread.
3. execution/: no-trade band, volatility target, per-asset and gross caps, kill switch, with T-EXE-1 and property tests.
4. Cost sweep at 0, 1, 2, 3 and 5 bps and the break-even cost for every configuration; deflated Sharpe using the registry trial count; placebo on the best configuration.
5. A comparison report: one table of net Sharpe by configuration and cost level, plus the trial count.
6. atu2 evaluate --final: implemented but it refuses to run unless the file holdout_unlock.txt exists. Test that it refuses without the file. You must never create that file.

Constraints: no changes to configs/gate.yaml; no new ideas beyond the listed configurations.

STOP. Give me the comparison report and your honest reading of it (does anything beat the best baseline at 2 bps?). Do not open the holdout.
```

### Prompt M5b: Final holdout run (only after I create holdout_unlock.txt)

```
I have created holdout_unlock.txt. Run atu2 evaluate --final once, on the single configuration I name here: <CONFIG NAME>. Do not run it on any other configuration and do not run it twice.

Produce docs/gate_b_report.md with: the configuration and its hash, net Sharpe at 0, 1, 2, 3 and 5 bps, deflated Sharpe probability with the trial count, placebo result, break-even cost, the neighbouring-settings check, and the PASS or FAIL against configs/gate.yaml with its hash.

Then delete nothing, change nothing in the gate, and STOP. If the result is FAIL, write the negative-result summary instead of suggesting more tuning. After this, the holdout is spent: any further model change needs a new untouched evaluation period.
```

### Prompt M6: Optional extras (only if Gate B passed)

```
Read CLAUDE.md and docs/BUILD_PLAN.md (sections: Optional extras, Risk register R14). The holdout is already spent, so nothing in this phase may use it.

Goal: test ideas one at a time as optional feature plug-ins behind flags (FR-21): support and resistance levels, seasonality flags, Kronos embeddings, X-Trend-style context set. One branch per idea.

Rules:
- Each plug-in must leave the baseline results identical when switched off (test it).
- Evaluate on validation periods only, with every trial counted in the registry and the deflated Sharpe recomputed.
- Keep an idea only if it beats the same system without it at 2 bps on validation, with the placebo clean. Otherwise drop it and record why in docs/ideas.md.
- Anything kept is labelled unconfirmed until it passes a fresh forward period (the paper-trading phase).

STOP after each idea with a one-paragraph verdict and the numbers.
```

### Prompt M7: Paper trading

```
Read CLAUDE.md and docs/BUILD_PLAN.md (sections: Operations, Risk register R12, R13, R16, R17). Plan first.

Goal: milestone M7 build, requirements FR-19 and FR-20, NFR-12. Paper money only.

Tasks:
1. paper/broker_adapter: Alpaca paper-trading endpoint only, behind an interface; a dry-run mode that is the default; the live endpoint must be impossible to select (tested).
2. paper/reconcile: compare broker positions with intended positions and log differences.
3. Daily job: after the US close, pull data, validate, build features, run the frozen model version, apply band, volatility target and caps, compute orders for the next open, submit, reconcile, record positions, fills and slippage in the registry. The model is loaded from a versioned file and the job never retrains.
4. Kill switch and alerts for drawdown breach, stale data, invalid output, order rejection, with tests that simulate each fault.
5. A GitHub Actions scheduled workflow (US Eastern time) that uses encrypted repository secrets for keys; ship it disabled and tell me how to enable it.
6. docs/runbook.md covering the daily cycle, monitoring table, incident playbooks and rollback from the plan.
7. Tests with recorded broker responses only; no network.

Acceptance: all three commands pass; a dry run on recorded data produces the expected orders.

STOP. Tell me exactly how to run one dry run myself and how to enable the schedule. I decide when paper trading starts. During the 3-month period only bug fixes are allowed, each logged.
```

### Prompt M8: Release and handoff

```
Read CLAUDE.md and docs/BUILD_PLAN.md. Goal: milestone M8.

Tasks:
1. Final report in docs/final_report.md: what was built, the headline results table (baselines and models at 0 to 5 bps), the Gate B and Gate C outcomes, what did not work, the limits of the evidence (short ETF history, adjusted data, spent holdout), and lessons learned.
2. A script that reproduces the headline table from a fresh checkout and the recorded data hashes; run it in a clean directory and show the output matches.
3. Update README, ADRs and runbook so they match the final code; remove dead code and stale configs.
4. Prepare the v1.0 changelog.

STOP. Do not tag; I will tag v1.0 after reading the report.
```

### What you do between phases

| After | You do |
|---|---|
| M0 | Read the report, merge, tag `v0.0.1` |
| M1 | Run the real download with your keys, paste back the summary, confirm the universe |
| M2 | Run the baselines on real data, paste the table; this is Gate A, and the decision whether to continue |
| M3 | Provide the Nasdaq Data Link key as an environment variable if asked; read the reproduction verdict |
| M4 | Read the synthetic-test results; decide CPU or Colab for M5 |
| M5a | Read the comparison report; pick at most one configuration to send to the final run |
| M5b | Create `holdout_unlock.txt` yourself, once, then paste the M5b prompt with the chosen configuration; this is Gate B |
| M7 | Run one dry run, then enable the schedule; check weekly against the backtest band |
| M8 | Read the final report and tag `v1.0` |

Two deliberate frictions: you are the only one who can unlock the holdout, and only you start paper trading. Those are the two steps where a tool in a hurry could quietly spoil the result.
