"""
src/train.py
============
Main Execution Pipeline for the RL Trading System.

Data Flow & Lifecycle:
----------------------
1. Ingestion & Indicators (File 1):
   - MarketDataProcessor computes OHLCV + Technical Indicators (RSI, MACD, ATR, BB, ADX).
2. Sentiment Scoring (File 2):
   - SentimentProcessor scores financial news headlines via ProsusAI/finbert.
3. Feature Merging:
   - Merges both feature streams into one massive table: `final_state_vector`.
4. Train/Test Chronological Split:
   - 80% Training Set, 20% Out-of-Sample Testing Set.
5. RL Environment (File 3):
   - ConstrainedTradingEnv enforces continuous compounding, 15:15 MIS auto-square-off,
     day-lock stop-loss penalty, 3-trade daily cap, and round-trip transaction fees.
6. Training & Continuous Backtesting:
   - Trains PPO for 200,000 steps via Stable-Baselines3.
   - Saves model to disk (`ppo_trading_agent_v1.zip`).
   - Backtests on the remaining 20% test set using an unbroken continuous loop.
   - Outputs institutional-grade financial audit metrics (Sortino, Gross vs. Net PnL, Fees).
"""

from pathlib import Path
import sys
from typing import List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

# Package imports
try:
    from src.data_processing import (
        MarketDataProcessor,
        SentimentProcessor,
        create_final_state_vector,
        generate_intraday_5min_market_data,
        load_archive_headlines,
    )
    from src.model import ConstrainedTradingEnv, create_vectorized_env
except ImportError:
    # Support direct script execution
    sys.path.append(str(Path(__file__).resolve().parent.parent))
    from src.data_processing import (
        MarketDataProcessor,
        SentimentProcessor,
        create_final_state_vector,
        generate_intraday_5min_market_data,
        load_archive_headlines,
    )
    from src.model import ConstrainedTradingEnv, create_vectorized_env


def calculate_sortino_ratio(
    returns: Union[List[float], np.ndarray],
    risk_free_rate: float = 0.0,
    target_return: float = 0.0,
    periods_per_year: int = 252,
) -> float:
    """
    Calculates the annualized Sortino Ratio:
    Sortino Ratio = (Annualized Mean Return - Risk-Free Rate) / Annualized Downside Deviation
    """
    returns_arr = np.asarray(returns, dtype=np.float64)
    if len(returns_arr) == 0:
        return 0.0

    mean_return = float(np.mean(returns_arr))
    annualized_return = mean_return * periods_per_year

    downside_diff = returns_arr - target_return
    downside_returns = np.minimum(0.0, downside_diff)
    downside_variance = np.mean(np.square(downside_returns))
    downside_deviation = np.sqrt(downside_variance) * np.sqrt(periods_per_year)

    if downside_deviation < 1e-9:
        if annualized_return > risk_free_rate:
            return float(np.inf)
        elif annualized_return < risk_free_rate:
            return float(-np.inf)
        else:
            return 0.0

    return float((annualized_return - risk_free_rate) / downside_deviation)


def build_pipeline_data(
    ticker: str = "AAPL",
    use_live_market_data: bool = False,
    use_finbert: bool = False,
    n_candles: int = 1500,
    device: str = "auto",
) -> pd.DataFrame:
    """
    Executes the sequential data pipeline:
    File 1 (MarketDataProcessor) + File 2 (SentimentProcessor) -> final_state_vector.

    Args:
        ticker: Target asset symbol.
        use_live_market_data: If True, fetches live 5-minute data from Yahoo Finance.
                              If False, synthesizes realistic 5-min intraday market data.
        use_finbert: If True, executes FinBERT model inference on news headlines.
                     If False, uses fast cached/scored headline sentiments.
        n_candles: Total number of 5-minute bars to generate if simulating.
        device: Compute device ('auto', 'cuda', 'cpu').

    Returns:
        pd.DataFrame: Merged `final_state_vector` DataFrame.
    """
    print("[Pipeline Step 1] Fetching & Engineering Market Data (File 1)...")
    market_processor = MarketDataProcessor()

    if use_live_market_data:
        try:
            print(f"  - Downloading 5-minute data for '{ticker}' via Yahoo Finance...")
            market_state_df = market_processor.process(ticker=ticker, period="60d", interval="5m")
        except Exception as e:
            print(f"  - Yahoo Finance fetch failed ({e}). Falling back to 5-min intraday generator...")
            raw_market = generate_intraday_5min_market_data(n_rows=n_candles, initial_price=150.0)
            market_state_df = market_processor.process_dataframe(raw_market)
    else:
        print(f"  - Synthesizing {n_candles:,} intraday 5-min candles across trading sessions...")
        raw_market = generate_intraday_5min_market_data(n_rows=n_candles, initial_price=150.0)
        market_state_df = market_processor.process_dataframe(raw_market)

    print(f"  - Market Features Engineered: {market_state_df.shape[0]} rows, {market_state_df.shape[1]} columns")

    print("\n[Pipeline Step 2] Scoring News Sentiment via FinBERT (File 2)...")
    headlines_df = load_archive_headlines(archive_dir="archive", ticker=ticker, max_rows=500)
    sentiment_df: Optional[pd.DataFrame] = None

    if use_finbert and not headlines_df.empty:
        try:
            device_arg = None if device == "auto" else device
            sentiment_processor = SentimentProcessor(device=device_arg)
            print(f"  - FinBERT device: {sentiment_processor.device.type.upper()}")
            sentiment_df = sentiment_processor.process_news_dataframe(headlines_df, batch_size=32)
            print(f"  - Headlines Scored & Aggregated: {len(sentiment_df)} calendar days")
        except Exception as e:
            print(f"  - FinBERT scoring skipped ({e}). Using simulated sentiment distribution.")
            sentiment_df = None

    if sentiment_df is None or sentiment_df.empty:
        # Create realistic sentiment stream matching market dates
        unique_dates = sorted(list(set(pd.to_datetime(market_state_df.index).strftime("%Y-%m-%d"))))
        rng = np.random.default_rng(42)
        sentiment_df = pd.DataFrame(
            {
                "date": unique_dates,
                "daily_sentiment": np.clip(rng.normal(0.05, 0.35, size=len(unique_dates)), -1.0, 1.0),
            }
        )

    print("\n[Pipeline Step 3] Merging into 'final_state_vector'...")
    final_state_vector = create_final_state_vector(market_state_df, sentiment_df)
    print(f"  - final_state_vector created: {final_state_vector.shape[0]} rows, {final_state_vector.shape[1]} features")
    print(f"  - Columns: {list(final_state_vector.columns)}")

    return final_state_vector


def run_pipeline(
    total_timesteps: int = 200000,
    ticker: str = "AAPL",
    use_live_market_data: bool = False,
    use_finbert: bool = False,
    model_save_path: str = "ppo_trading_agent_v1",
    device: str = "auto",
) -> Tuple[PPO, float, float, float]:
    """
    Executes the complete Sequential Data Pipeline:
    1. Runs File 1 & File 2 to build `final_state_vector`.
    2. Splits data into 80% Train and 20% Test chronologically.
    3. Feeds training data into File 3 (`ConstrainedTradingEnv`) via `DummyVecEnv`.
    4. Trains PPO agent for 200,000 steps.
    5. Saves trained model to disk.
    6. Executes continuous backtesting loop on remaining 20% test set.
    7. Logs comprehensive financial performance report.
    """
    print("=" * 70)
    print("     REINFORCEMENT LEARNING INTRADAY TRADING PIPELINE (PPO)     ")
    print("=" * 70)

    # 1. Build unified final_state_vector
    final_state_vector = build_pipeline_data(
        ticker=ticker,
        use_live_market_data=use_live_market_data,
        use_finbert=use_finbert,
        n_candles=1500,
        device=device,
    )

    # 2. Chronological 80% Train / 20% Test Split
    print("\n[Pipeline Step 4] Splitting final_state_vector (80% Train, 20% Test)...")
    split_idx = int(len(final_state_vector) * 0.8)
    train_df = final_state_vector.iloc[:split_idx].copy()
    test_df = final_state_vector.iloc[split_idx:].copy()

    train_days = len(set(pd.to_datetime(train_df.index).date))
    test_days = len(set(pd.to_datetime(test_df.index).date))

    print(f"  - Training Set (80%): {train_df.shape[0]:,} candles across {train_days} trading days")
    print(f"  - Testing Set (20%):  {test_df.shape[0]:,} candles across {test_days} trading days")

    # 3. Vectorize Training Environment (File 3)
    print("\n[Pipeline Step 5] Vectorizing ConstrainedTradingEnv (File 3)...")
    train_env = create_vectorized_env(df=train_df)
    print(f"  - Action Space:      {train_env.action_space}")
    print(f"  - Observation Space: {train_env.observation_space}")

    # 4. Model Training
    print(f"\n[Pipeline Step 6] Initializing PPO (Device: {device}) & Training for {total_timesteps:,} Timesteps...")
    model = PPO(
        policy="MlpPolicy",
        env=train_env,
        learning_rate=0.0003,
        n_steps=2048,
        batch_size=64,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        verbose=1,
        seed=42,
        device=device,
    )

    model.learn(total_timesteps=total_timesteps)

    # 5. Persist Model
    save_path_obj = Path(model_save_path)
    if save_path_obj.parent and not save_path_obj.parent.exists():
        save_path_obj.parent.mkdir(parents=True, exist_ok=True)

    print(f"\n[Pipeline Step 7] Saving Model to '{model_save_path}'...")
    model.save(model_save_path)
    print("  - Model weights successfully saved.")

    # 6. Continuous Walk-Forward Backtesting Loop on Remaining 20% Test Data
    print("\n[Pipeline Step 8] Running Continuous Backtesting Loop on 20% Test Data...")
    loaded_model = PPO.load(model_save_path, device=device)
    test_env = ConstrainedTradingEnv(df=test_df)
    state, info = test_env.reset(seed=42)

    initial_balance: float = test_env.initial_balance
    portfolio_values: List[float] = [initial_balance]
    step_returns: List[float] = []

    done = False
    step_count = 0
    profitable_closed_trades = 0

    while not done:
        action, _states = loaded_model.predict(state, deterministic=True)
        state, reward, terminated, truncated, step_info = test_env.step(action)
        done = terminated or truncated
        step_count += 1

        portfolio_values.append(float(step_info["balance"]))
        step_returns.append(float(step_info["step_profit"]))

        if step_info.get("trade_closed_profit", False):
            profitable_closed_trades += 1

    final_compounded_balance = float(step_info["account_balance"])
    total_days_processed = step_info.get("days_processed", test_days)
    total_trades_executed = step_info.get("total_trades", 0)
    total_fees_paid = step_info.get("total_fees_paid", 0.0)

    net_profit = final_compounded_balance - initial_balance
    net_return_pct = (net_profit / initial_balance) * 100.0
    gross_profit = net_profit + total_fees_paid
    gross_return_pct = (gross_profit / initial_balance) * 100.0

    # 7. Financial Performance Audit Report
    print("\n[Pipeline Step 9] Calculating Sortino Ratio & Audit Metrics...")
    sortino = calculate_sortino_ratio(
        returns=step_returns,
        risk_free_rate=0.0,
        target_return=0.0,
        periods_per_year=252,
    )

    print("\n" + "=" * 70)
    print("             INTRADAY BACKTEST PERFORMANCE REPORT               ")
    print("=" * 70)
    print(f" Initial Account Balance:        ₹{initial_balance:,.2f}")
    print(f" Final Compounded Balance:       ₹{final_compounded_balance:,.2f}")
    print(f" Total Days Processed:           {total_days_processed} days ({step_count} candles)")
    print(f" Total Trades Executed:          {total_trades_executed} trades (max 3/day)")
    print(f" Profitable Closed Trades (>2%): {profitable_closed_trades} trades")
    print(f" Total Brokerage & STT Paid:     ₹{total_fees_paid:,.2f} (Round-trip)")
    print(f" Gross Profit (Before Fees):     ₹{gross_profit:+,.2f} ({gross_return_pct:+.2f}%)")
    print(f" Net Profit (After Fees):        ₹{net_profit:+,.2f} ({net_return_pct:+.2f}%)")
    print(f" Sortino Ratio (Ann. 252d):      {sortino:.4f}")
    print(f" Continuous Episode Status:      Completed Entire Test Horizon ({terminated})")
    print("=" * 70 + "\n")

    return loaded_model, final_compounded_balance, net_return_pct, sortino


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="ATU RL Intraday Trading Pipeline")
    parser.add_argument(
        "--timesteps",
        type=int,
        default=200000,
        help="Total PPO training timesteps (default: 200000)",
    )
    parser.add_argument(
        "--ticker",
        type=str,
        default="AAPL",
        help="Stock ticker symbol (default: AAPL)",
    )
    parser.add_argument(
        "--live-data",
        action="store_true",
        help="Download live Yahoo Finance market data instead of synthesized candles",
    )
    parser.add_argument(
        "--finbert",
        action="store_true",
        help="Score news sentiment using FinBERT model (CUDA/CPU)",
    )
    parser.add_argument(
        "--model-save-path",
        type=str,
        default="ppo_trading_agent_v1",
        help="Path or directory to save the trained model checkpoint",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Compute device for PPO and FinBERT ('auto', 'cuda', 'cpu')",
    )

    args = parser.parse_args()

    run_pipeline(
        total_timesteps=args.timesteps,
        ticker=args.ticker,
        use_live_market_data=args.live_data,
        use_finbert=args.finbert,
        model_save_path=args.model_save_path,
        device=args.device,
    )

