"""
src/model.py
============
Reinforcement Learning Environment & Model Definitions.

Components:
-----------
1. ConstrainedTradingEnv (File 3):
   - Gymnasium rules engine with continuous compounding on raw (unscaled) prices.
   - Discrete(3) actions (0 short, 1 flat, 2 long); reward = net rupee P&L per step.
   - ATR-based stop with volatility sizing, trailing stop, take-profit and a daily
     loss cap; intra-candle and gap-slippage fills; day lockout after a losing stop.
   - Session square-off (15:15 IST for NSE MIS) without terminating the episode.
   - 3-trade daily cap; 0.03% brokerage per leg (entry + exit).
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


def create_trading_env(df, **env_kwargs) -> ConstrainedTradingEnv:
    """
    Creates an instance of ConstrainedTradingEnv. Any ConstrainedTradingEnv argument
    (risk, stops, sizing, session times, ...) can be overridden via keyword; the env's
    own defaults are the single source of truth.
    """
    return ConstrainedTradingEnv(df=df, **env_kwargs)


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

