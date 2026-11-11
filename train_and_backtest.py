"""
train_and_backtest.py (File 4)
==============================
CLI entry point for the ATU pipeline. All logic lives in `src/train.py`, which the
Colab notebook (`ATU_GPU.ipynb`) also calls, so the CLI and notebook stay identical:

    real yfinance 5-min candles + real news (FinBERT, 1-day lag)
    -> leak-free features -> 80/20 split
    -> PPO + A2C on VecNormalize'd rewards (chunked, resumable checkpoints)
    -> consensus walk-forward backtest in unscaled rupees

Examples:
    python train_and_backtest.py --ticker AAPL --timesteps 500000
    python train_and_backtest.py --ticker RELIANCE.NS --workspace /content/drive/MyDrive/ATU_Workspace
    python train_and_backtest.py --timesteps 4096 --chunks 2 --no-tensorboard   # smoke test
"""

from src.train import (  # noqa: F401  (re-exported for backward compatibility)
    BacktestResult,
    LiveData,
    build_live_state_vector,
    calculate_financial_metrics,
    consensus_action,
    main,
    run_ensemble_backtest,
    run_pipeline,
    save_backtest_results,
    session_rules,
    split_train_test,
    train_ensemble,
)
from market_data_processor import create_final_state_vector  # noqa: F401

__all__ = [
    "BacktestResult", "LiveData", "build_live_state_vector", "calculate_financial_metrics",
    "consensus_action", "create_final_state_vector", "main", "run_ensemble_backtest",
    "run_pipeline", "save_backtest_results", "session_rules", "split_train_test", "train_ensemble",
]

if __name__ == "__main__":
    main()
