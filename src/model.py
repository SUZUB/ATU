"""
src/model.py
============
Reinforcement Learning Environment & Model Definitions.

Components:
-----------
1. ConstrainedTradingEnv (File 3):
   - Gymnasium rules engine with Continuous Compounding.
   - Day-Lock Stop-Loss (-1% threshold with slippage penalty).
   - MIS 15:15 Auto-Square-Off without premature termination.
   - 3-Trade Daily Quota & Overtrading Penalty (-0.1).
   - Round-trip Brokerage & STT transaction fees (0.05% entry + 0.05% exit).
   - Dynamic 1% risk manager scaling against live `account_balance`.
2. create_trading_env:
   - Helper to instantiate and validate the Gymnasium environment.
3. create_ppo_agent:
   - Factory to build a Stable-Baselines3 PPO agent tuned for financial time series.
"""

from typing import Optional
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

# Import ConstrainedTradingEnv from root module
try:
    from constrained_trading_env import ConstrainedTradingEnv
except ImportError:
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).resolve().parent.parent))
    from constrained_trading_env import ConstrainedTradingEnv


def create_trading_env(
    df,
    initial_balance: float = 100000.0,
    max_risk_per_trade: float = 0.01,
    profit_target: float = 0.02,
    loss_limit: float = -0.01,
    commission_rate: float = 0.0005,
    max_daily_trades: int = 3,
) -> ConstrainedTradingEnv:
    """
    Creates an instance of ConstrainedTradingEnv with configured risk parameters.
    """
    return ConstrainedTradingEnv(
        df=df,
        initial_balance=initial_balance,
        max_risk_per_trade=max_risk_per_trade,
        profit_target=profit_target,
        loss_limit=loss_limit,
        commission_rate=commission_rate,
        max_daily_trades=max_daily_trades,
    )


def create_vectorized_env(df, **env_kwargs) -> DummyVecEnv:
    """
    Wraps ConstrainedTradingEnv in Stable-Baselines3 DummyVecEnv.
    """
    return DummyVecEnv([lambda: create_trading_env(df=df, **env_kwargs)])


def create_ppo_agent(
    env: DummyVecEnv,
    learning_rate: float = 0.0003,
    n_steps: int = 2048,
    batch_size: int = 64,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    clip_range: float = 0.2,
    ent_coef: float = 0.01,
    verbose: int = 1,
    seed: int = 42,
    device: str = "auto",
    tensorboard_log: Optional[str] = None,
) -> PPO:
    """
    Instantiates a PPO model with an MlpPolicy tuned for financial time series.
    Configurable compute device ('auto', 'cuda', 'cpu').
    """
    return PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=batch_size,
        gamma=gamma,
        gae_lambda=gae_lambda,
        clip_range=clip_range,
        ent_coef=ent_coef,
        verbose=verbose,
        seed=seed,
        device=device,
        tensorboard_log=tensorboard_log,
    )

