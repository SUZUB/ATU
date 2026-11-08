"""
train_and_backtest.py (File 4)
==============================
Main Execution Pipeline for the RL Trading System.

Sequential Data Flow:
---------------------
1. Imports File 1 (market_data_processor.py) & File 2 (sentiment_processor.py).
2. Merges outputs into one massive table: `final_state_vector`.
3. Splits data chronologically: 80% Train, 20% Test.
4. Imports File 3 (constrained_trading_env.py) and vectorizes the training environment.
5. Trains PPO for 200,000 steps, saves model as `ppo_trading_agent_v1.zip`.
6. Executes continuous backtesting loop on the remaining 20% test set.
7. Logs performance audit report (Total Days, Trades, Brokerage/STT in ₹, Gross vs Net PnL, Sortino).
"""

from pathlib import Path
import sys
from typing import List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

# File 1: MarketDataProcessor
from market_data_processor import MarketDataProcessor

# File 2: SentimentProcessor
from sentiment_processor import SentimentProcessor

# File 3: ConstrainedTradingEnv
from constrained_trading_env import ConstrainedTradingEnv

# Data utilities from src
from src.data_processing import (
    create_final_state_vector,
    generate_intraday_5min_market_data,
    load_archive_headlines,
)


def calculate_sortino_ratio(
    returns: Union[List[float], np.ndarray],
    risk_free_rate: float = 0.0,
    target_return: float = 0.0,
    periods_per_year: int = 252,
) -> float:
    """
    Calculates the annualized Sortino Ratio from periodic returns.
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


def build_final_state_vector(
    ticker: str = "AAPL",
    n_candles: int = 1500,
    use_live_market_data: bool = False,
    use_finbert: bool = False,
    device: str = "auto",
) -> pd.DataFrame:
    """
    Merges File 1 (Market Features) + File 2 (Sentiment Scores) -> `final_state_vector`.
    """
    print("[Pipeline 1/4] Processing Market Data (File 1)...")
    market_proc = MarketDataProcessor()

    if use_live_market_data:
        try:
            print(f"  - Downloading live 5-minute data for '{ticker}'...")
            market_df = market_proc.process(ticker=ticker, period="60d", interval="5m")
        except Exception as e:
            print(f"  - Yahoo Finance fetch failed ({e}). Generating 5-min intraday market data...")
            raw_market = generate_intraday_5min_market_data(n_rows=n_candles, initial_price=150.0)
            market_df = market_proc.process_dataframe(raw_market)
    else:
        print(f"  - Generating {n_candles:,} intraday 5-min candles across trading sessions...")
        raw_market = generate_intraday_5min_market_data(n_rows=n_candles, initial_price=150.0)
        market_df = market_proc.process_dataframe(raw_market)

    print(f"  - Market Data Processed: {market_df.shape[0]} candles, {market_df.shape[1]} features")

    print("\n[Pipeline 2/4] Processing News Sentiment (File 2)...")
    headlines_df = load_archive_headlines(archive_dir="archive", ticker=ticker, max_rows=500)
    sentiment_df: Optional[pd.DataFrame] = None

    if use_finbert and not headlines_df.empty:
        try:
            device_arg = None if device == "auto" else device
            sentiment_proc = SentimentProcessor(device=device_arg)
            sentiment_df = sentiment_proc.process_news_dataframe(headlines_df, batch_size=32)
            print(f"  - FinBERT Headlines Processed: {len(sentiment_df)} trading days")
        except Exception as e:
            print(f"  - FinBERT scoring skipped ({e}). Using simulated sentiment distribution.")
            sentiment_df = None

    if sentiment_df is None or sentiment_df.empty:
        unique_dates = sorted(list(set(pd.to_datetime(market_df.index).strftime("%Y-%m-%d"))))
        rng = np.random.default_rng(42)
        sentiment_df = pd.DataFrame(
            {
                "date": unique_dates,
                "daily_sentiment": np.clip(rng.normal(0.05, 0.35, size=len(unique_dates)), -1.0, 1.0),
            }
        )

    print("\n[Pipeline 3/4] Merging Processors into 'final_state_vector'...")
    final_state_vector = create_final_state_vector(market_df, sentiment_df)
    print(f"  - final_state_vector Shape: {final_state_vector.shape}")
    print(f"  - Features: {list(final_state_vector.columns)}")

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
    Full Sequential Execution Pipeline:
    File 1 + File 2 -> final_state_vector -> 80/20 Split -> File 3 -> Train 200k -> Continuous Backtest.
    """
    print("=" * 70)
    print("   FILE 4: MAIN PIPELINE RUNNER (TRAINING & CONTINUOUS BACKTEST)   ")
    print("=" * 70)

    # 1. Merge File 1 and File 2 into final_state_vector
    final_state_vector = build_final_state_vector(
        ticker=ticker,
        n_candles=1500,
        use_live_market_data=use_live_market_data,
        use_finbert=use_finbert,
        device=device,
    )

    # 2. Chronological Split: 80% Train, 20% Test
    print("\n[Pipeline 4/4] Splitting final_state_vector (80% Train, 20% Test)...")
    split_idx = int(len(final_state_vector) * 0.8)
    train_df = final_state_vector.iloc[:split_idx].copy()
    test_df = final_state_vector.iloc[split_idx:].copy()

    train_days = len(set(pd.to_datetime(train_df.index).date))
    test_days = len(set(pd.to_datetime(test_df.index).date))

    print(f"  - Training Set (80%): {train_df.shape[0]:,} candles across {train_days} trading days")
    print(f"  - Testing Set (20%):  {test_df.shape[0]:,} candles across {test_days} trading days")

    # 3. Vectorize File 3 (ConstrainedTradingEnv)
    print("\nVectorizing ConstrainedTradingEnv (File 3) with DummyVecEnv...")
    train_env = DummyVecEnv([lambda: ConstrainedTradingEnv(df=train_df)])
    print(f"  - Action Space:      {train_env.action_space}")
    print(f"  - Observation Space: {train_env.observation_space}")

    # 4. Train PPO Agent
    print(f"\nInitializing PPO Agent (Device: {device}) & Training for {total_timesteps:,} Timesteps...")
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

    print(f"\nSaving Trained Policy to disk as '{model_save_path}'...")
    model.save(model_save_path)
    print("  - Model successfully saved.")

    # 6. Continuous Walk-Forward Backtesting Loop
    print("\nRestoring Model & Running Continuous Backtest on Remaining 20% Test Data...")
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

    # 7. Final Financial Performance Audit Report
    print("\nCalculating Evaluation Metrics...")
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
