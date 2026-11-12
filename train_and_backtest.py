"""
train_and_backtest.py (File 4)
==============================
CLI entry point for the ATU pipeline. All logic lives in `src/train.py`, which the
Colab notebook (`ATU_GPU.ipynb`) also calls, so the CLI and notebook stay identical:

    real yfinance candles (default 1h, ~2y) + real news (FinBERT, 1-day lag)
    -> leak-free features -> 80/20 split
    -> PPO + A2C per seed on original + mirrored data (shaped, normalized rewards)
    -> confidence-gated ensemble backtest per seed (unscaled P&L, $ or ₹ by ticker)

Examples:
    python train_and_backtest.py --ticker AAPL
    python train_and_backtest.py --ticker RELIANCE.NS --workspace /content/drive/MyDrive/ATU_Workspace
    python train_and_backtest.py --passes 1 --seeds 0 --no-news   # smoke test
"""

from src.train import (  # noqa: F401  (re-exported for backward compatibility)
    BacktestResult,
    ExperimentResult,
    LiveData,
    build_live_state_vector,
    calculate_financial_metrics,
    consensus_action,
    main,
    run_ensemble_backtest,
    run_experiment,
    run_pipeline,
    save_backtest_results,
    session_rules,
    split_train_test,
    train_ensemble,
)
from market_data_processor import create_final_state_vector  # noqa: F401

__all__ = [
    "BacktestResult", "ExperimentResult", "LiveData", "build_live_state_vector", "calculate_financial_metrics",
    "consensus_action", "create_final_state_vector", "main", "run_ensemble_backtest", "run_experiment",
    "run_pipeline", "save_backtest_results", "session_rules", "split_train_test", "train_ensemble",
]

if __name__ == "__main__":
    main()
