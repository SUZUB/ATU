"""
Trading Project Source Package
==============================
Modular reinforcement learning trading system with FinBERT sentiment and MIS friction modeling.
"""

from src.data_processing import MarketDataProcessor, SentimentProcessor, create_final_state_vector
from src.model import ConstrainedTradingEnv, create_ppo_agent
from src.train import run_pipeline

__all__ = [
    "MarketDataProcessor",
    "SentimentProcessor",
    "create_final_state_vector",
    "ConstrainedTradingEnv",
    "create_ppo_agent",
    "run_pipeline",
]

