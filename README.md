# ATU - Automated Trading Unit: Intraday Reinforcement Learning System

An intraday RL trading system built with **Stable-Baselines3 (PPO + A2C ensemble)**, **Gymnasium**, and **FinBERT** news sentiment. It trains on **real candles** (yfinance; default 1-hour bars over ~2 years) and **real headlines** (Google News RSS + yfinance news), with realistic intraday frictions and risk controls: square-off before the close, ATR-based stops with volatility sizing, trailing stop / take-profit, a daily loss cap, a 3-trade daily cap and per-leg fees.

---

## 1. Project Structure

```text
XD/
├── ATU_GPU.ipynb               # Colab notebook (cells 3-4 call src/train.py)
├── notebooks/
│   ├── ATU_GPU.ipynb           # Identical copy of the Colab notebook
│   └── training.ipynb          # Small local walkthrough on generated demo data
├── src/
│   ├── data_processing.py      # Archive loader, demo data generator, re-exports
│   ├── model.py                # Env / PPO factory helpers
│   └── train.py                # THE pipeline: data -> features -> training -> backtest + CLI
├── market_data_processor.py    # File 1: OHLCV, indicators, leak-free scaling, news fetcher, sentiment merge
├── sentiment_processor.py      # File 2: FinBERT scoring
├── constrained_trading_env.py  # File 3: Gymnasium trading environment
├── train_and_backtest.py       # File 4: CLI entry point (thin wrapper over src/train.py)
├── research_strategies.py      # Rule-based 5-min Opening Range Breakout (paper-based), edge check
├── orb_model.py                # Trained + saved ORB model (learns which ORB trades to take)
└── requirements.txt
```

The CLI and the notebook run the **same functions** in `src/train.py`, so they cannot drift apart.

---

## 2. Workflow

```mermaid
flowchart LR
    A["Local IDE (Windows)"] <-->|"git push / pull"| B[("GitHub<br/>SUZUB/ATU")]
    B -->|"git clone / pull"| C["Google Colab"]
    C -->|"checkpoints, news cache, results"| D[("Google Drive<br/>/MyDrive/ATU_Workspace/")]
```

Workspace layout (local folder or Drive):

| Folder | Contents |
| :--- | :--- |
| `datasets/` | `news_cache_<TICKER>.csv` (accumulated headlines) |
| `checkpoints/seed_<n>/` | `ppo_trading_agent_v2.zip`, `a2c_trading_agent_v2.zip`, `*_vecnormalize_v2.pkl` |
| `training_logs/` | TensorBoard logs |
| `results/` | `seed_summary.txt/json`, `equity_by_seed.png`; per seed in `seed_<n>/`: `backtest_report.txt`, `evaluation_metrics.json`, `daily_pnl.csv`, `backtest_portfolio_history.csv`, `equity_curve.png` |

---

## 3. Pipeline

```mermaid
flowchart TD
    P["yfinance OHLCV (default 1h, ~2 years)"] --> F["Indicators: RSI, MACD, ATR, BB, ADX"]
    F --> S["MinMax scaling fit on first 80% only<br/>raw_open/high/low/close kept for P&L"]
    N["Google News RSS + yfinance news"] --> B["FinBERT: P(pos) - P(neg)"]
    B --> L["Daily mean, dated in exchange tz,<br/>lagged 1 day, expires after 4 days"]
    S --> M["final_state_vector"]
    L --> M
    M --> X["Chronological 80/20 split"]
    P --> R["Price-mirrored copy<br/>(training only)"]
    X --> T["PPO + A2C per seed, shaped reward,<br/>fixed budget of 20 passes"]
    R --> T
    T --> E["Confidence-gated ensemble backtest<br/>(raw env) + daily stats, B&H, seed summary"]
```

- **No synthetic data in the pipeline.** A failed price download raises an error.
- **No look-ahead:** scaler fit on the training slice only; sentiment from day D is first used on day D+1.
- **No direction bias:** each agent also trains on a mirrored price history (every return negated), so "always long" or "always short" cannot score.
- **No memorization by repetition:** a fixed budget of 20 passes over the training data, fresh models every run (`--resume` to continue).

---

## 4. Environment Rules (`ConstrainedTradingEnv`)

| Rule | Implementation |
| :--- | :--- |
| **Actions** | `Discrete(3)`: 0 = Short, 1 = Flat, 2 = Long (target position) |
| **Observation** | Scaled market features + unrealized P&L, position direction, time of day, trades remaining, lockout flag |
| **Reward** | Shaped (default): step net P&L, minus 0.5x each deepening of the day's max drawdown, +/-0.05% of equity for a green/red day. Staying flat is free. `reward_mode="pnl"` = plain P&L. Normalized only during training via `VecNormalize` |
| **Stop (1R)** | 2 x ATR(14) / price, clipped to 0.3%-2%. Detected on the candle high/low; fills at the stop, or at the open if the candle gapped through it. A **losing** stop locks the agent out for the rest of the day |
| **Position size** | `equity x 0.5% / 1R`, capped at 1x, so a stop costs ~0.5% of equity |
| **Trailing / take-profit** | From +1R the stop trails the best price at 1R (break-even first); take-profit at +3R |
| **Daily loss cap** | At -1% of the day's starting equity: close and stop trading until the next session |
| **Churn control** | Min holding time and re-entry cooldown (~15 min on 5m bars, 1 bar on 1h) |
| **Fees** | 0.03% of traded notional on entry and on exit |
| **Square-off** | NSE/BSE (`.NS`/`.BO`): session 09:15-15:30 IST, square-off 15:15. US: 09:30-16:00 ET, square-off 15:45. Always flat at the close of each day's last bar |
| **Daily cap** | Max 3 entries per day |
| **Episode** | One unbroken episode over the whole dataset; terminates only at the last bar |

---

## 5. Local Development

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# Smoke test (real data, tiny training, one seed, no news)
python train_and_backtest.py --ticker AAPL --passes 1 --seeds 0 --no-news

# Full run (1h bars over ~2 years, 3 seeds x 20 passes)
python train_and_backtest.py --ticker AAPL
```

Key CLI options (`python train_and_backtest.py --help`):

| Flag | Default | Meaning |
| :--- | :--- | :--- |
| `--ticker` | `AAPL` | yfinance symbol (`RELIANCE.NS` for NSE) |
| `--workspace` | `ATU_Workspace` | Output folder (alias: `--model-save-path`) |
| `--interval` / `--period` | `1h` / `730d` | Bar size and history (`5m` is limited to `60d`) |
| `--passes` | `20` | Training passes over the (original + mirrored) training data, per agent |
| `--seeds` | `0 1 2` | One independently trained ensemble per seed |
| `--chunks` | `5` | Checkpoint chunks per agent |
| `--no-mirror` | off | Train without the mirrored price copy |
| `--resume` | off | Continue from checkpoints instead of fresh models |
| `--device` | `auto` | FinBERT device |
| `--news-query` | company + symbol | Override news search queries (repeatable) |
| `--no-finbert` | off | Neutral sentiment |
| `--no-news` | off | Price data only: skip news fetching and FinBERT (notebook: `USE_NEWS_SENTIMENT = False`) |
| `--require-sentiment` | off | Fail if no bar gets real sentiment |
| `--include-archive` | off | Also scan old archive CSVs in `datasets/` |
| `--no-tensorboard` | off | Disable TensorBoard logs |

---

## 6. Google Colab

1. Open `ATU_GPU.ipynb` from `https://github.com/SUZUB/ATU` in Colab and select a GPU runtime (used by FinBERT; the small SB3 agents train on CPU).
2. **Cell 1** clones/pulls the repo, installs requirements and mounts Drive.
3. **Cell 2** imports the shared pipeline.
4. **Cell 3** builds the real-data state vector. Change `TICKER`, `INTERVAL`, `PERIOD` here.
5. **Cell 4** trains one PPO + A2C ensemble per seed (`SEEDS`, `TRAIN_PASSES`), backtests each and prints the per-seed reports and the cross-seed summary.
6. **Cell 5** runs the rule-based 5-minute Opening Range Breakout on a stock universe (no RL) and saves `results/research/`.
7. **Cell 6** loads the saved ORB model from Drive, updates it with new data, saves it again and prints today's signals.

Equivalent CLI inside Colab:
```bash
!python train_and_backtest.py --ticker AAPL --workspace /content/drive/MyDrive/ATU_Workspace
```

---

## 7. Reading the Report

Each seed's report includes net/gross P&L, fees, trade-level win rate and profit factor, exit types (stop / trail / take-profit / daily cap), drawdown, Sharpe/Sortino, the **buy & hold** return over the same window, **green / red / flat days**, sub-window returns and each agent's **signal mix**. A `WARNING: ... collapsed` line means an agent outputs the same signal on >95% of test bars.

The **seed summary** compares the seeds: a real edge is profitable in most seeds and holds up across sub-windows; one good seed is luck.

**Caveat:** with 1h bars the test window is ~145 trading days; with 5m bars (60 days max) it is only ~12 days and results are anecdotal. Past backtests do not guarantee future profit.

---

## 8. Research Strategy: 5-Minute Opening Range Breakout (`research_strategies.py`)

Rule-based, parameters taken from the papers (nothing tuned on our data):

- **ORB** - Zarattini & Aziz (2023), *Can Day Trading Really Be Profitable?*, SSRN 4416622: the first 5-minute candle sets the direction (green -> long, red -> short, doji -> none); entry at the next bar's open; stop at the other side of the opening range; exit at **3R (1:3 risk:reward)** or the session close. Each trade risks 1% of its capital share, leverage capped at 4x.
- **Stocks in Play** filter - Zarattini, Barbon & Aziz (2024), SFI RP 24-98: trade only the top-N stocks by relative opening volume (`top_n`).
- **Intraday momentum** - Gao, Han, Li & Zhou (2018), *JFE* 129(2): first-bar return from the previous close predicts the last half hour.
- **Edge check** - rank correlation of opening features with the rest-of-day return, train vs. test.

```powershell
python research_strategies.py --fee 0.0001          # 13 US stocks, 5m bars, 1:3 RR
python research_strategies.py --tickers RELIANCE.NS TCS.NS INFY.NS --fee 0.0002
```

What we found (Nov 2026, 12-13 large US stocks):

| Test | Result |
| :--- | :--- |
| 1-hour opening range, 2 years | Loses even before fees in the test period - a 60-minute range is too slow |
| Intraday momentum, 2 years | Loses before fees |
| **5-minute ORB 1:3, last 60 days** | **+1% to +6% net at 0.003-0.01% fees per side, ~-2% at 0.03%**; profitable second half, slightly negative first half |
| RL agents (Cells 3-4), 2 years | Learn to stay flat: indicators alone do not beat fees |

Costs decide the outcome: the paper assumes ~0.002% per side. Yahoo serves only 60 days of 5-minute bars, so every run **merges new bars into `datasets/intraday/`** and the out-of-sample record grows over time. Trust the strategy only when both halves stay positive at your real costs, and paper-trade it before risking money.

### 8.1 Trained ORB model (`orb_model.py`, notebook Cell 6)

- **Sample** = one stock on one day, at 09:35 ET (first 5-minute candle closed). 12 features: candle direction/body/range/wicks, relative opening volume, gap, previous-day return and range, 10-day average range, SPY's first candle.
- **Labels** = net R (after fees) of a 1:3 bracket trade LONG and SHORT (stop = other side of the first candle).
- **Model** = two small gradient-boosted tree regressors; trade the better side if its predicted R > +0.10R.
- **Leak guard** = before training, prices after 09:35 are scrambled and every feature must stay identical (this caught a real look-ahead bug during development).
- **Evaluation** = walk-forward, retrained every 5 sessions on the past only, compared with plain ORB on the same days.
- **Persistence** = `<workspace>/models/orb_model.joblib` (+ dated copies, `orb_model.json` metadata). Each run loads it, retrains on all accumulated history and saves a new version.
- **Live signals** use the model only if it beat plain ORB in both halves of the out-of-sample test; otherwise plain ORB.

First result (40 US stocks, 59 sessions, 39 out-of-sample, 0.01% fees/side): model +2.77% vs plain ORB +2.76%; plain ORB was positive in both halves (+1.6%, +1.1%, max DD -0.7%), the model was not (-0.0%, +2.8%) -> live signals currently use plain ORB.

```powershell
python orb_model.py --workspace ATU_Workspace --fee 0.0001
```
