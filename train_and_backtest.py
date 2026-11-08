"""
train_and_backtest.py (File 4)
==============================
Main Execution Pipeline for the ATU Intraday Reinforcement Learning Trading System.

Sequential Data Flow:
---------------------
1. Imports File 1 (market_data_processor.py) & File 2 (sentiment_processor.py).
2. Merges outputs into one unified table: `final_state_vector`.
3. Splits data chronologically: 80% Train, 20% Test.
4. Imports File 3 (constrained_trading_env.py) and vectorizes the training environment.
5. Trains PPO agent (with periodic CheckpointCallback support), saves to Google Drive or local storage.
6. Executes continuous backtesting loop on the remaining 20% test set.
7. Logs and persists institutional-grade performance metrics:
   - Initial & Final Balance, Net Return %, Gross Return %
   - Round-trip Brokerage & STT Fees
   - Annualized Sharpe & Sortino Ratios
   - Maximum Drawdown %
   - Win Rate % & Trade Count
   - Equity Curves & JSON Metrics
"""

import json
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
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


def calculate_financial_metrics(
    portfolio_values: List[float],
    step_returns: List[float],
    total_trades: int,
    profitable_trades: int,
    initial_balance: float,
    final_balance: float,
    total_fees_paid: float,
    bars_per_day: int = 75,
) -> Dict[str, Any]:
    """
    Computes institutional risk-adjusted return and drawdown metrics:
    Sortino Ratio, Sharpe Ratio, Maximum Drawdown, Win Rate, and Gross vs Net Returns.
    """
    pv_arr = np.asarray(portfolio_values, dtype=np.float64)
    ret_arr = np.asarray(step_returns, dtype=np.float64)
    annual_factor = np.sqrt(252 * bars_per_day)

    net_profit = final_balance - initial_balance
    net_return_pct = (net_profit / initial_balance) * 100.0
    gross_profit = net_profit + total_fees_paid
    gross_return_pct = (gross_profit / initial_balance) * 100.0

    # Maximum Drawdown calculation
    if len(pv_arr) > 0:
        peak = np.maximum.accumulate(pv_arr)
        drawdown = (pv_arr - peak) / np.maximum(peak, 1e-9)
        max_drawdown_pct = float(np.min(drawdown)) * 100.0
    else:
        max_drawdown_pct = 0.0

    # Sharpe Ratio
    mean_ret = float(np.mean(ret_arr)) if len(ret_arr) > 0 else 0.0
    std_ret = float(np.std(ret_arr)) if len(ret_arr) > 0 else 0.0
    sharpe = float((mean_ret / std_ret) * annual_factor) if std_ret > 1e-9 else 0.0

    # Sortino Ratio (Downside deviation only)
    downside_returns = np.minimum(0.0, ret_arr)
    downside_variance = np.mean(np.square(downside_returns)) if len(ret_arr) > 0 else 0.0
    downside_dev = np.sqrt(downside_variance)
    sortino = float((mean_ret / downside_dev) * annual_factor) if downside_dev > 1e-9 else 0.0

    # Win Rate
    win_rate = (profitable_trades / total_trades * 100.0) if total_trades > 0 else 0.0

    return {
        "initial_balance": float(initial_balance),
        "final_balance": float(final_balance),
        "net_profit": float(net_profit),
        "net_return_pct": float(net_return_pct),
        "gross_profit": float(gross_profit),
        "gross_return_pct": float(gross_return_pct),
        "total_fees_paid": float(total_fees_paid),
        "total_trades": int(total_trades),
        "profitable_trades": int(profitable_trades),
        "win_rate_pct": float(win_rate),
        "max_drawdown_pct": float(max_drawdown_pct),
        "sharpe_ratio": float(sharpe),
        "sortino_ratio": float(sortino),
    }


def build_final_state_vector(
    ticker: str = "AAPL",
    n_candles: int = 1500,
    use_live_market_data: bool = False,
    use_finbert: bool = True,
    device: str = "auto",
    data_dir: Optional[str] = None,
    allow_synthetic_fallback: bool = False,
) -> pd.DataFrame:
    """
    Merges File 1 (Market Features) + File 2 (Sentiment Scores) -> `final_state_vector`.
    Guarantees that real sentiment headlines are loaded, scored via FinBERT, and aligned
    date-for-date with the market data without silent zero or random substitution.
    """
    print(f"\n[Pipeline 1/4] Ingesting Financial News Headlines for '{ticker}' (File 2)...")
    headlines_df = load_archive_headlines(
        archive_dir=data_dir,
        ticker=ticker,
        max_rows=1000,
        allow_synthetic_fallback=allow_synthetic_fallback,
    )
    print(f"  - Actual Headlines Ingested: {len(headlines_df):,} records for ticker '{ticker}'")

    sentiment_df: Optional[pd.DataFrame] = None

    if use_finbert:
        print("\n[Pipeline 2/4] Scoring News Sentiment via FinBERT (File 2)...")
        device_arg = None if device == "auto" else device
        sentiment_proc = SentimentProcessor(device=device_arg)
        print(f"  - FinBERT Compute Device: {sentiment_proc.device.type.upper()}")
        sentiment_df = sentiment_proc.process_news_dataframe(headlines_df, batch_size=32)
        print(f"  - FinBERT Output: {len(headlines_df):,} headlines scored into {len(sentiment_df):,} distinct trading days")
    elif allow_synthetic_fallback:
        print("\n[Pipeline 2/4] FinBERT disabled. Generating synthetic sentiment (fallback explicitly allowed)...")
        unique_dates = sorted(list(set(headlines_df["date"]))) if not headlines_df.empty else ["2024-01-02", "2024-01-03"]
        rng = np.random.default_rng(42)
        sentiment_df = pd.DataFrame(
            {
                "date": unique_dates,
                "daily_sentiment": np.clip(rng.normal(0.05, 0.35, size=len(unique_dates)), -1.0, 1.0),
            }
        )
    else:
        raise ValueError(
            "FinBERT sentiment scoring is disabled (use_finbert=False) and allow_synthetic_fallback is False.\n"
            "To guarantee training integrity, please enable FinBERT scoring or pass --allow-synthetic-fallback."
        )

    if sentiment_df is None or sentiment_df.empty:
        if not allow_synthetic_fallback:
            raise ValueError(
                f"No valid sentiment scores were generated for ticker '{ticker}'.\n"
                f"Silent synthetic fallback is disabled. Verify that your headlines dataset contains valid records."
            )
        else:
            print("WARNING: Empty sentiment scores. Generating simulated sentiment fallback...")
            rng = np.random.default_rng(42)
            sentiment_df = pd.DataFrame(
                {
                    "date": ["2024-01-02", "2024-01-03", "2024-01-04"],
                    "daily_sentiment": [0.15, -0.20, 0.05],
                }
            )

    # Align market data starting date with real headline dates
    market_start_date = sentiment_df["date"].min() if not sentiment_df.empty else "2024-01-02"
    print(f"\n[Pipeline 3/4] Processing Market Data & Technical Indicators (File 1)...")
    print(f"  - Synchronizing market time series start date to sentiment anchor: {market_start_date}")
    market_proc = MarketDataProcessor()

    if use_live_market_data:
        try:
            print(f"  - Downloading live 5-minute data for '{ticker}'...")
            market_df = market_proc.process(ticker=ticker, period="60d", interval="5m")
        except Exception as e:
            print(f"  - Yahoo Finance fetch failed ({e}). Generating aligned 5-min intraday market data...")
            raw_market = generate_intraday_5min_market_data(n_rows=n_candles, start_date=market_start_date, initial_price=150.0)
            market_df = market_proc.process_dataframe(raw_market)
    else:
        print(f"  - Generating {n_candles:,} intraday 5-min candles starting from {market_start_date}...")
        raw_market = generate_intraday_5min_market_data(n_rows=n_candles, start_date=market_start_date, initial_price=150.0)
        market_df = market_proc.process_dataframe(raw_market)

    print(f"  - Market Data Processed: {market_df.shape[0]:,} candles, {market_df.shape[1]} features")

    print("\n[Pipeline 4/4] Merging Processors into 'final_state_vector'...")
    final_state_vector = create_final_state_vector(market_df, sentiment_df)
    sentiment_series = final_state_vector["daily_sentiment"]
    non_zero_pct = float((sentiment_series != 0.0).mean()) * 100.0

    print(f"  - final_state_vector Shape: {final_state_vector.shape}")
    print(f"  - Features: {list(final_state_vector.columns)}")
    print(f"  - Sentiment Feature Statistics:")
    print(f"      * Mean:     {sentiment_series.mean():+.4f}")
    print(f"      * Std Dev:  {sentiment_series.std():.4f}")
    print(f"      * Min/Max:  [{sentiment_series.min():+.4f}, {sentiment_series.max():+.4f}]")
    print(f"      * Non-Zero: {non_zero_pct:.1f}% of state vectors")

    if non_zero_pct == 0.0 and not allow_synthetic_fallback:
        raise ValueError(
            "CRITICAL INTEGRITY FAILURE: All sentiment features in final_state_vector are 0.0!\n"
            "Real sentiment dates did not overlap with market candles. Check date alignment."
        )

    return final_state_vector


def run_pipeline(
    total_timesteps: int = 200000,
    ticker: str = "AAPL",
    use_live_market_data: bool = False,
    use_finbert: bool = True,
    model_save_path: str = "ppo_trading_agent_v1",
    device: str = "auto",
    data_dir: Optional[str] = None,
    checkpoint_freq: Optional[int] = None,
    allow_synthetic_fallback: bool = False,
) -> Tuple[PPO, float, float, float]:
    """
    Full Sequential Execution Pipeline:
    File 1 + File 2 -> final_state_vector -> 80/20 Split -> File 3 -> Train -> Continuous Backtest.
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
        data_dir=data_dir,
        allow_synthetic_fallback=allow_synthetic_fallback,
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

    # Setup Organized Directory Structure for Persistence (Phase 7)
    base_output_dir = Path(model_save_path)
    if base_output_dir.suffix:
        # User passed a file path like ppo_trading_agent_v1.zip or /dir/model
        base_dir = base_output_dir.parent if base_output_dir.parent != Path("") else Path(".")
        model_name = base_output_dir.stem
    else:
        # User passed a directory like /content/drive/MyDrive/ATU_Models
        base_dir = base_output_dir
        model_name = "ppo_trading_agent_v1"

    checkpoints_dir = base_dir / "checkpoints"
    final_models_dir = base_dir / "final_models"
    training_logs_dir = base_dir / "training_logs"
    backtest_results_dir = base_dir / "backtest_results"
    evaluation_metrics_dir = base_dir / "evaluation_metrics"

    for d in [checkpoints_dir, final_models_dir, training_logs_dir, backtest_results_dir, evaluation_metrics_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # Setup Checkpoint Callback
    callbacks = []
    freq = checkpoint_freq if checkpoint_freq else max(1000, total_timesteps // 5)
    callbacks.append(
        CheckpointCallback(
            save_freq=freq,
            save_path=str(checkpoints_dir),
            name_prefix="ppo_checkpoint",
            save_replay_buffer=False,
            save_vecnormalize=True,
        )
    )

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

    model.learn(total_timesteps=total_timesteps, callback=callbacks)

    # 5. Persist Model in organized folders
    final_zip_path = final_models_dir / f"{model_name}.zip"
    compat_zip_path = base_dir / f"{model_name}.zip"

    print(f"\nSaving Final Trained Policy to:")
    print(f"  - Primary:       {final_zip_path}")
    print(f"  - Compatibility: {compat_zip_path}")
    model.save(str(final_zip_path.with_suffix("")))
    model.save(str(compat_zip_path.with_suffix("")))

    # Save training configuration
    train_config = {
        "ticker": ticker,
        "total_timesteps": total_timesteps,
        "device": device,
        "use_live_market_data": use_live_market_data,
        "use_finbert": use_finbert,
        "train_candles": len(train_df),
        "test_candles": len(test_df),
        "learning_rate": 0.0003,
        "batch_size": 64,
        "gamma": 0.99,
    }
    with open(training_logs_dir / "training_config.json", "w", encoding="utf-8") as f:
        json.dump(train_config, f, indent=2)

    # 6. Continuous Walk-Forward Backtesting Loop
    print("\nRestoring Model & Running Continuous Backtest on Remaining 20% Test Data...")
    loaded_model = PPO.load(str(final_zip_path), device=device)
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

    # 7. Final Financial Performance Audit Report
    print("\nCalculating Institutional Evaluation Metrics...")
    metrics = calculate_financial_metrics(
        portfolio_values=portfolio_values,
        step_returns=step_returns,
        total_trades=total_trades_executed,
        profitable_trades=profitable_closed_trades,
        initial_balance=initial_balance,
        final_balance=final_compounded_balance,
        total_fees_paid=total_fees_paid,
    )
    metrics["total_days_processed"] = total_days_processed
    metrics["total_steps"] = step_count
    metrics["continuous_episode_completed"] = bool(terminated)

    # Persist metrics JSON
    with open(evaluation_metrics_dir / "backtest_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    # Persist portfolio history CSV
    portfolio_df = pd.DataFrame({"step": range(len(portfolio_values)), "portfolio_value": portfolio_values})
    portfolio_df.to_csv(backtest_results_dir / "portfolio_history.csv", index=False)

    # Generate equity curve plot via matplotlib if available
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        plt.figure(figsize=(10, 5))
        plt.plot(portfolio_values, label="Portfolio Balance (₹)", color="#2563EB", linewidth=1.5)
        plt.axhline(initial_balance, color="gray", linestyle="--", alpha=0.7, label="Initial Capital")
        plt.title(f"ATU Walk-Forward Out-of-Sample Equity Curve ({ticker})")
        plt.xlabel("Intraday 5-Min Steps")
        plt.ylabel("Balance (₹)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(backtest_results_dir / "equity_curve.png", dpi=150)
        plt.close()
        print(f"  - Equity curve plot saved: {backtest_results_dir / 'equity_curve.png'}")
    except Exception as e:
        print(f"  - Note: Matplotlib plot generation skipped ({e})")

    # Format human-readable audit report
    report_text = f"""======================================================================
             INTRADAY BACKTEST PERFORMANCE REPORT               
======================================================================
 Initial Account Balance:        ₹{metrics['initial_balance']:,.2f}
 Final Compounded Balance:       ₹{metrics['final_balance']:,.2f}
 Total Days Processed:           {metrics['total_days_processed']} days ({metrics['total_steps']} candles)
 Total Trades Executed:          {metrics['total_trades']} trades (max 3/day)
 Profitable Closed Trades (>2%): {metrics['profitable_trades']} trades
 Win Rate:                       {metrics['win_rate_pct']:.2f}%
 Total Brokerage & STT Paid:     ₹{metrics['total_fees_paid']:,.2f} (Round-trip)
 Gross Profit (Before Fees):     ₹{metrics['gross_profit']:+,.2f} ({metrics['gross_return_pct']:+.2f}%)
 Net Profit (After Fees):        ₹{metrics['net_profit']:+,.2f} ({metrics['net_return_pct']:+.2f}%)
 Maximum Drawdown:               {metrics['max_drawdown_pct']:.2f}%
 Sharpe Ratio (Annualized):      {metrics['sharpe_ratio']:.4f}
 Sortino Ratio (Annualized):     {metrics['sortino_ratio']:.4f}
 Continuous Episode Status:      Completed Entire Test Horizon ({metrics['continuous_episode_completed']})
======================================================================"""

    print("\n" + report_text + "\n")
    with open(backtest_results_dir / "backtest_report.txt", "w", encoding="utf-8") as f:
        f.write(report_text)

    return loaded_model, final_compounded_balance, metrics["net_return_pct"], metrics["sortino_ratio"]


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
        default=True,
        help="Enable FinBERT news sentiment scoring (default: True)",
    )
    parser.add_argument(
        "--no-finbert",
        action="store_false",
        dest="finbert",
        help="Disable FinBERT news sentiment scoring",
    )
    parser.add_argument(
        "--allow-synthetic-fallback",
        action="store_true",
        default=False,
        help="Explicitly permit fallback to synthetic headlines/sentiment if real dataset is absent (default: False)",
    )
    parser.add_argument(
        "--model-save-path",
        type=str,
        default="ppo_trading_agent_v1",
        help="Path or directory to save model checkpoints and artifacts",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Compute device for PPO and FinBERT ('auto', 'cuda', 'cpu')",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Directory containing historical news datasets (e.g. /content/drive/MyDrive/ATU_Data)",
    )
    parser.add_argument(
        "--checkpoint-freq",
        type=int,
        default=None,
        help="Frequency in timesteps to save intermediate PPO checkpoints",
    )

    args = parser.parse_args()

    run_pipeline(
        total_timesteps=args.timesteps,
        ticker=args.ticker,
        use_live_market_data=args.live_data,
        use_finbert=args.finbert,
        model_save_path=args.model_save_path,
        device=args.device,
        data_dir=args.data_dir,
        checkpoint_freq=args.checkpoint_freq,
        allow_synthetic_fallback=args.allow_synthetic_fallback,
    )
