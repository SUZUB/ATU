"""
src/train.py
============
Single source of truth for the ATU pipeline. Both the CLI (`train_and_backtest.py`,
`python -m src.train`) and the Colab notebook (`ATU_GPU.ipynb` cells 3-4) call these
functions, so the two entry points cannot drift apart.

Flow:
-----
1. Real 5-minute OHLCV from yfinance (no synthetic fallback; failures raise).
2. Real recent headlines (Google News RSS + yfinance news, optional archive),
   scored by FinBERT, lagged one day in the exchange timezone.
3. Leak-free features (scaler fit on the first 80%) -> chronological 80/20 split.
4. PPO + A2C trained on VecNormalize'd (reward-only) envs in chunks, with model
   and normalization stats checkpointed to the workspace after every chunk.
5. Consensus walk-forward backtest on a raw env (unscaled rupee P&L).

Workspace layout (local folder or Google Drive):
    <workspace>/datasets/        news cache, optional archive CSVs
    <workspace>/checkpoints/     *_trading_agent_v1.zip, *_vecnormalize.pkl
    <workspace>/training_logs/   TensorBoard logs (if tensorboard is installed)
    <workspace>/results/         metrics JSON, portfolio CSV, equity curve, report
"""

from dataclasses import dataclass, field
from datetime import time as dtime
import importlib.util
import json
import math
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from stable_baselines3 import A2C, PPO
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

try:
    from market_data_processor import (
        MarketDataProcessor,
        compute_split_index,
        create_final_state_vector,
        fetch_recent_headlines,
        is_indian_ticker,
    )
    from constrained_trading_env import ConstrainedTradingEnv
    from src.data_processing import SentimentProcessor, load_archive_headlines
except ImportError:
    sys.path.append(str(Path(__file__).resolve().parent.parent))
    from market_data_processor import (
        MarketDataProcessor,
        compute_split_index,
        create_final_state_vector,
        fetch_recent_headlines,
        is_indian_ticker,
    )
    from constrained_trading_env import ConstrainedTradingEnv
    from src.data_processing import SentimentProcessor, load_archive_headlines


# =====================================================================
# Configuration
# =====================================================================
PPO_KWARGS: Dict[str, Any] = dict(
    learning_rate=0.0003, n_steps=2048, batch_size=64, gamma=0.99,
    gae_lambda=0.95, clip_range=0.2, ent_coef=0.01,
)
A2C_KWARGS: Dict[str, Any] = dict(
    learning_rate=0.0007, n_steps=5, gamma=0.99, gae_lambda=1.0,
    ent_coef=0.01, max_grad_norm=0.5,
)
# name -> (algorithm, hyperparameters, checkpoint file, normalization stats file)
AGENT_SPECS: Dict[str, Tuple[type, Dict[str, Any], str, str]] = {
    "PPO": (PPO, PPO_KWARGS, "ppo_trading_agent_v1.zip", "ppo_vecnormalize.pkl"),
    "A2C": (A2C, A2C_KWARGS, "a2c_trading_agent_v1.zip", "a2c_vecnormalize.pkl"),
}
REWARD_CLIP: float = 10.0
ACTION_SHORT, ACTION_FLAT, ACTION_LONG = 0, 1, 2  # ConstrainedTradingEnv Discrete(3)
COLLAPSE_THRESHOLD: float = 0.95  # warn when one agent emits one signal on >95% of test bars


def session_rules(ticker: str) -> Dict[str, dtime]:
    """
    Exchange session rules passed to ConstrainedTradingEnv. Positions are squared off
    15 minutes before the close (NSE MIS = 15:15 IST). yfinance timestamps are already
    in the exchange's local timezone, so wall-clock times compare directly.
    """
    if is_indian_ticker(ticker):  # NSE / BSE, Asia/Kolkata
        return dict(session_start_time=dtime(9, 15), session_end_time=dtime(15, 30), auto_square_off_time=dtime(15, 15))
    # US equities, America/New_York
    return dict(session_start_time=dtime(9, 30), session_end_time=dtime(16, 0), auto_square_off_time=dtime(15, 45))


@dataclass
class WorkspacePaths:
    root: Path

    @property
    def datasets(self) -> Path:
        return self.root / "datasets"

    @property
    def checkpoints(self) -> Path:
        return self.root / "checkpoints"

    @property
    def training_logs(self) -> Path:
        return self.root / "training_logs"

    @property
    def results(self) -> Path:
        return self.root / "results"

    def create(self) -> "WorkspacePaths":
        for d in (self.datasets, self.checkpoints, self.training_logs, self.results):
            d.mkdir(parents=True, exist_ok=True)
        return self


# =====================================================================
# Metrics
# =====================================================================
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
    Return, drawdown and risk-adjusted metrics. `step_returns` are per-bar portfolio
    returns; `profit_factor` here is bar-level (the ensemble backtest overrides it with
    the trade-level value).
    """
    pv_arr = np.asarray(portfolio_values, dtype=np.float64)
    ret_arr = np.asarray(step_returns, dtype=np.float64)
    annual_factor = np.sqrt(252 * bars_per_day)

    net_profit = final_balance - initial_balance
    net_return_pct = (net_profit / initial_balance) * 100.0
    gross_profit = net_profit + total_fees_paid
    gross_return_pct = (gross_profit / initial_balance) * 100.0

    if len(pv_arr) > 0:
        peak = np.maximum.accumulate(pv_arr)
        drawdown = (pv_arr - peak) / np.maximum(peak, 1e-9)
        max_drawdown_pct = float(np.min(drawdown)) * 100.0
    else:
        max_drawdown_pct = 0.0

    mean_ret = float(np.mean(ret_arr)) if len(ret_arr) > 0 else 0.0
    std_ret = float(np.std(ret_arr)) if len(ret_arr) > 0 else 0.0
    sharpe = float((mean_ret / std_ret) * annual_factor) if std_ret > 1e-9 else 0.0

    downside_returns = np.minimum(0.0, ret_arr)
    downside_variance = np.mean(np.square(downside_returns)) if len(ret_arr) > 0 else 0.0
    downside_dev = np.sqrt(downside_variance)
    sortino = float((mean_ret / downside_dev) * annual_factor) if downside_dev > 1e-9 else 0.0

    win_rate = (profitable_trades / total_trades * 100.0) if total_trades > 0 else 0.0

    pos_profits = float(np.sum(ret_arr[ret_arr > 0])) if len(ret_arr) > 0 else 0.0
    neg_losses = float(np.abs(np.sum(ret_arr[ret_arr < 0]))) if len(ret_arr) > 0 else 0.0
    profit_factor = (pos_profits / neg_losses) if neg_losses > 1e-9 else (float("inf") if pos_profits > 0.0 else 0.0)

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
        "profit_factor": float(profit_factor),
    }


# =====================================================================
# 1-3. Real data: prices + news sentiment -> final_state_vector
# =====================================================================
@dataclass
class LiveData:
    ticker: str
    final_state_vector: pd.DataFrame
    market_processor: MarketDataProcessor
    env_kwargs: Dict[str, dtime]
    bars_per_day: int
    headlines: pd.DataFrame
    sentiment: Optional[pd.DataFrame]


def _score_recent_news(
    ticker: str,
    market_start: Any,
    market_end: Any,
    exchange_tz: Any,
    data_dir: Optional[str],
    use_finbert: bool,
    device: str,
    news_queries: Optional[Sequence[str]],
    include_archive: bool,
) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    """Fetches recent headlines and scores them with FinBERT. Returns (headlines, daily sentiment or None)."""
    news_start = pd.Timestamp(market_start) - pd.Timedelta(days=7)  # lagged sentiment for the first sessions
    cache_path = None
    if data_dir is not None:
        safe_ticker = "".join(c if c.isalnum() else "_" for c in ticker.upper())
        cache_path = Path(data_dir) / f"news_cache_{safe_ticker}.csv"
    headlines = fetch_recent_headlines(
        ticker, start=news_start, end=market_end, tz=exchange_tz,
        queries=news_queries, cache_path=cache_path,
    )

    if include_archive:
        try:
            archive = load_archive_headlines(
                archive_dir=data_dir, ticker=ticker, max_rows=50000, allow_synthetic_fallback=False,
            )
            archive_day = pd.to_datetime(archive["date"])
            archive = archive[(archive_day >= news_start) & (archive_day <= pd.Timestamp(market_end))]
            if not archive.empty:
                print(f"  - Archive headlines inside the market window: {len(archive):,}")
                headlines = pd.concat([headlines, archive.assign(source="archive")], ignore_index=True)
        except FileNotFoundError:
            pass

    if headlines.empty:
        print("  - WARNING: no real headlines found for the market window -> neutral sentiment (0.0).")
        return headlines, None
    if not use_finbert:
        print("  - FinBERT disabled (--no-finbert) -> neutral sentiment (0.0).")
        return headlines, None

    sentiment_processor = SentimentProcessor(device=None if device == "auto" else device)
    print(f"  - Scoring {len(headlines):,} headlines with FinBERT on {sentiment_processor.device.type.upper()}...")
    sentiment_df = sentiment_processor.process_news_dataframe(headlines[["date", "headline"]], batch_size=32)
    print(f"  - Daily sentiment: {len(sentiment_df):,} days | mean {sentiment_df['daily_sentiment'].mean():+.3f}")
    return headlines, sentiment_df


def build_live_state_vector(
    ticker: str = "AAPL",
    period: str = "60d",
    interval: str = "5m",
    data_dir: Optional[str] = None,
    use_finbert: bool = True,
    device: str = "auto",
    news_queries: Optional[Sequence[str]] = None,
    include_archive: bool = False,
    require_sentiment: bool = False,
    use_news: bool = True,
) -> LiveData:
    """
    Downloads real candles and real headlines, scores headlines with FinBERT, and
    merges them (one-day lag, exchange timezone). Raises if price data is unavailable;
    never substitutes synthetic data. `include_archive` also scans the large 2009-2020
    archive CSVs (off by default: they cannot overlap a recent 60-day window).
    `use_news=False` skips news and FinBERT entirely (price data only, neutral sentiment).
    Any failure in the optional news step degrades to neutral sentiment instead of raising.
    """
    print(f"\n[Data 1/3] Downloading real {interval} candles for {ticker} (period={period})...")
    market_processor = MarketDataProcessor()  # scaler fit on the first 80% only
    market_df = market_processor.process(ticker=ticker, period=period, interval=interval)
    exchange_tz = market_df.index.tz
    bar_dates = pd.Series(market_df.index.date)
    bars_per_day = int(bar_dates.value_counts().median())
    market_start, market_end = bar_dates.min(), bar_dates.max()
    print(f"  - {market_df.shape[0]:,} candles | {market_start} -> {market_end} | "
          f"{bar_dates.nunique()} sessions | ~{bars_per_day} bars/day | tz={exchange_tz}")

    sentiment_df: Optional[pd.DataFrame] = None
    headlines = pd.DataFrame(columns=["date", "headline"])
    if not use_news:
        print("\n[Data 2/3] News sentiment disabled (price data only) -> neutral sentiment (0.0).")
    else:
        print(f"\n[Data 2/3] Collecting real headlines for {ticker}...")
        try:
            headlines, sentiment_df = _score_recent_news(
                ticker, market_start, market_end, exchange_tz, data_dir,
                use_finbert, device, news_queries, include_archive,
            )
        except Exception as e:  # news is optional: never let it stop the price pipeline
            print(f"  - WARNING: news/sentiment step failed ({type(e).__name__}: {e}) -> neutral sentiment (0.0).")
            sentiment_df = None

    print("\n[Data 3/3] Merging into final_state_vector (sentiment lagged 1 day, expires after 4)...")
    final_state_vector = create_final_state_vector(market_df, sentiment_df)
    sentiment_series = final_state_vector["daily_sentiment"]
    split_idx = compute_split_index(len(final_state_vector))
    coverage_all = float((sentiment_series != 0.0).mean()) * 100.0
    coverage_test = float((sentiment_series.iloc[split_idx:] != 0.0).mean()) * 100.0
    print(f"  - Shape: {final_state_vector.shape}")
    print(f"  - Non-zero sentiment: {coverage_all:.1f}% of all bars, {coverage_test:.1f}% of test bars")
    if require_sentiment and coverage_all == 0.0:
        raise ValueError("No bar received non-zero sentiment (require_sentiment=True).")

    return LiveData(
        ticker=ticker,
        final_state_vector=final_state_vector,
        market_processor=market_processor,
        env_kwargs=session_rules(ticker),
        bars_per_day=bars_per_day,
        headlines=headlines,
        sentiment=sentiment_df,
    )


def split_train_test(final_state_vector: pd.DataFrame, market_processor: MarketDataProcessor) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Chronological split on the exact row the scaler was fit up to."""
    split_idx = compute_split_index(len(final_state_vector), market_processor.train_ratio)
    if market_processor.fit_end_index != split_idx:
        raise ValueError(f"Scaler fit window ({market_processor.fit_end_index}) != train split ({split_idx}).")
    train_df = final_state_vector.iloc[:split_idx].copy()
    test_df = final_state_vector.iloc[split_idx:].copy()
    print(f"\nTrain: {len(train_df):,} candles ({len(set(train_df.index.date))} days) | "
          f"Test: {len(test_df):,} candles ({len(set(test_df.index.date))} days)")
    return train_df, test_df


# =====================================================================
# 4. Training (VecNormalize rewards, chunked, resumable)
# =====================================================================
def has_finite_weights(model: BaseAlgorithm) -> bool:
    return all(torch.isfinite(p).all().item() for p in model.policy.parameters())


def make_train_env(train_df: pd.DataFrame, env_kwargs: Dict[str, Any], stats_path: Path) -> VecNormalize:
    """Reward-normalized training env; restores running reward stats when resuming."""
    venv = DummyVecEnv([lambda: ConstrainedTradingEnv(df=train_df, **env_kwargs)])
    if stats_path.exists():
        try:
            env = VecNormalize.load(str(stats_path), venv)
            env.training = True
            env.norm_reward = True
            print(f"  Restored reward-normalization stats: {stats_path}")
            return env
        except Exception as e:
            print(f"  Could not restore {stats_path} ({e}); using fresh stats.")
    return VecNormalize(venv, norm_obs=False, norm_reward=True, clip_reward=REWARD_CLIP, gamma=0.99)


def load_or_create(
    name: str,
    env: VecNormalize,
    checkpoint_path: Path,
    tensorboard_log: Optional[str],
) -> BaseAlgorithm:
    """Resumes a compatible, finite checkpoint; otherwise builds a fresh model. Never deletes files."""
    algo_cls, hyperparams, _, _ = AGENT_SPECS[name]
    if checkpoint_path.exists():
        try:
            model = algo_cls.load(str(checkpoint_path), env=env, device="cpu")
            if not has_finite_weights(model):
                raise ValueError("checkpoint contains NaN/Inf weights")
            model.tensorboard_log = tensorboard_log
            print(f"[Persistent Memory] Resumed {name} from {checkpoint_path} ({model.num_timesteps:,} timesteps so far)")
            return model
        except Exception as e:
            print(f"[Fresh Initialization] Cannot resume {name} ({e}). Starting a fresh model.")
    else:
        print(f"[Fresh Initialization] No {name} checkpoint at {checkpoint_path}.")
    return algo_cls(
        policy="MlpPolicy", env=env, verbose=1, seed=42, device="cpu",
        tensorboard_log=tensorboard_log, **hyperparams,
    )


def train_ensemble(
    train_df: pd.DataFrame,
    env_kwargs: Dict[str, Any],
    workspace: str,
    chunks: int = 10,
    steps_per_chunk: int = 50_000,
    tensorboard: bool = True,
    log_interval: int = 1000,
) -> Dict[str, BaseAlgorithm]:
    """Trains PPO then A2C in chunks, saving model + VecNormalize stats after each chunk."""
    paths = WorkspacePaths(Path(workspace)).create()
    if tensorboard and importlib.util.find_spec("tensorboard") is None:
        print("TensorBoard is not installed; training without TensorBoard logs.")
        tensorboard = False

    models: Dict[str, BaseAlgorithm] = {}
    for name, (_, _, checkpoint_file, stats_file) in AGENT_SPECS.items():
        print(f"\nBuilding {name} training environment...")
        checkpoint_path = paths.checkpoints / checkpoint_file
        stats_path = paths.checkpoints / stats_file
        env = make_train_env(train_df, env_kwargs, stats_path)
        tb_dir = str(paths.training_logs / name.lower()) if tensorboard else None
        model = load_or_create(name, env, checkpoint_path, tb_dir)

        print(f"[Training] {name}: {chunks} chunks x {steps_per_chunk:,} steps = {chunks * steps_per_chunk:,} timesteps")
        for chunk in range(chunks):
            model.learn(total_timesteps=steps_per_chunk, reset_num_timesteps=False, log_interval=log_interval)
            model.save(str(checkpoint_path))
            env.save(str(stats_path))
            print(f"  {name} chunk {chunk + 1}/{chunks} done ({model.num_timesteps:,} total timesteps) -> saved")
        if not has_finite_weights(model):
            raise RuntimeError(f"{name} weights became NaN/Inf during training.")
        models[name] = model
    return models


# =====================================================================
# 5. Consensus walk-forward backtest (raw rupee P&L)
# =====================================================================
@dataclass
class BacktestResult:
    ticker: str
    metrics: Dict[str, Any]
    portfolio_values: List[float] = field(default_factory=list)
    timestamps: List[pd.Timestamp] = field(default_factory=list)
    step_returns: List[float] = field(default_factory=list)
    trade_pnls: List[float] = field(default_factory=list)
    initial_balance: float = 100000.0

    @property
    def report(self) -> str:
        return format_report(self.metrics)


def consensus_action(action_ppo: Any, action_a2c: Any) -> int:
    """Discrete consensus: trade only when both agents pick the same direction, else Flat."""
    a, b = int(np.asarray(action_ppo).reshape(-1)[0]), int(np.asarray(action_a2c).reshape(-1)[0])
    return a if a == b and a != ACTION_FLAT else ACTION_FLAT


def signal_mix(actions: Sequence[int]) -> Dict[str, float]:
    """Share of short / flat / long signals."""
    arr = np.asarray(actions, dtype=int)
    n = max(len(arr), 1)
    return {
        "short": float((arr == ACTION_SHORT).sum() / n),
        "flat": float((arr == ACTION_FLAT).sum() / n),
        "long": float((arr == ACTION_LONG).sum() / n),
    }


def collapse_warnings(mixes: Dict[str, Dict[str, float]], threshold: float = COLLAPSE_THRESHOLD) -> List[str]:
    """Flags agents that emit a single signal on more than `threshold` of bars."""
    out = []
    for name, mix in mixes.items():
        label, share = max(mix.items(), key=lambda kv: kv[1])
        if share > threshold:
            out.append(f"{name} collapsed: '{label}' on {share:.0%} of test bars (policy ignores the market state)")
    return out


def run_ensemble_backtest(
    models: Dict[str, BaseAlgorithm],
    test_df: pd.DataFrame,
    env_kwargs: Dict[str, Any],
    bars_per_day: int,
    ticker: str,
) -> BacktestResult:
    """
    Runs the full test horizon on an UN-normalized env, so balances, fees and P&L are
    real rupees. Training used norm_obs=False, so agents consume raw observations.
    Records each agent's signal mix and warns when a policy has collapsed to one signal.
    """
    test_env = ConstrainedTradingEnv(df=test_df, **env_kwargs)
    state, info = test_env.reset(seed=42)
    print(f"\n[Backtest] Consensus walk-forward over {len(test_df):,} candles...")

    result = BacktestResult(
        ticker=ticker,
        metrics={},
        portfolio_values=[float(info["equity"])],
        timestamps=[test_df.index[0]],
        initial_balance=float(test_env.initial_balance),
    )
    stop_loss_hits = 0
    done = False
    step_info: Dict[str, Any] = info
    signals: Dict[str, List[int]] = {"PPO": [], "A2C": [], "Ensemble": []}
    while not done:
        action_ppo, _ = models["PPO"].predict(state, deterministic=True)
        action_a2c, _ = models["A2C"].predict(state, deterministic=True)
        action = consensus_action(action_ppo, action_a2c)
        signals["PPO"].append(int(np.asarray(action_ppo).reshape(-1)[0]))
        signals["A2C"].append(int(np.asarray(action_a2c).reshape(-1)[0]))
        signals["Ensemble"].append(action)
        state, _, terminated, truncated, step_info = test_env.step(action)
        done = terminated or truncated

        result.portfolio_values.append(float(step_info["equity"]))
        result.timestamps.append(test_df.index[step_info["step"]])
        result.step_returns.append(float(step_info["step_profit"]))
        result.trade_pnls.extend(step_info["closed_trade_pnls"])
        stop_loss_hits += int(step_info["stop_loss_hit"])

    metrics = calculate_financial_metrics(
        portfolio_values=result.portfolio_values,
        step_returns=result.step_returns,
        total_trades=int(step_info["closed_trades"]),
        profitable_trades=int(step_info["winning_trades"]),
        initial_balance=result.initial_balance,
        final_balance=float(step_info["equity"]),
        total_fees_paid=float(step_info["total_fees_paid"]),
        bars_per_day=bars_per_day,
    )
    gross_wins = sum(p for p in result.trade_pnls if p > 0.0)
    gross_losses = abs(sum(p for p in result.trade_pnls if p < 0.0))
    metrics["profit_factor"] = (gross_wins / gross_losses) if gross_losses > 0.0 else (float("inf") if gross_wins > 0.0 else 0.0)
    metrics["stop_loss_exits"] = stop_loss_hits
    metrics["ticker"] = ticker
    metrics["test_start"] = str(test_df.index[0])
    metrics["test_end"] = str(test_df.index[-1])
    metrics["total_steps"] = len(result.step_returns)
    metrics["total_days_processed"] = int(step_info["days_processed"])
    metrics["signal_mix"] = {name: signal_mix(acts) for name, acts in signals.items()}
    metrics["collapse_warnings"] = collapse_warnings({k: metrics["signal_mix"][k] for k in ("PPO", "A2C")})
    result.metrics = metrics
    print(f"Backtest completed: {len(result.step_returns):,} steps, {metrics['total_trades']} closed trades, "
          f"{stop_loss_hits} stop-loss exits.")
    return result


def format_report(metrics: Dict[str, Any]) -> str:
    pf = metrics["profit_factor"]
    pf_text = f"{pf:.2f}" if math.isfinite(pf) else "N/A (No Losing Trades)"
    return "\n".join([
        "=" * 65,
        "      ENSEMBLE (PPO + A2C) INSTITUTIONAL PERFORMANCE REPORT    ",
        "=" * 65,
        f" Ticker / Test Window:     {metrics['ticker']} | {metrics['test_start'][:10]} -> {metrics['test_end'][:10]}",
        f" Initial Balance:          ₹{metrics['initial_balance']:,.2f}",
        f" Final Compounded Balance: ₹{metrics['final_balance']:,.2f}",
        f" Net Profit (After Fees):  ₹{metrics['net_profit']:+,.2f} ({metrics['net_return_pct']:+.2f}%)",
        f" Gross Profit (Pre-Fees):  ₹{metrics['gross_profit']:+,.2f} ({metrics['gross_return_pct']:+.2f}%)",
        f" Total Brokerage & STT:    ₹{metrics['total_fees_paid']:,.2f}",
        f" Closed Trades:            {metrics['total_trades']:,} (Win Rate: {metrics['win_rate_pct']:.1f}%, "
        f"Stop-Loss Exits: {metrics['stop_loss_exits']})",
        f" Maximum Drawdown:         {metrics['max_drawdown_pct']:.2f}%",
        f" Sharpe Ratio (Ann.):      {metrics['sharpe_ratio']:.4f}",
        f" Sortino Ratio (Ann.):     {metrics['sortino_ratio']:.4f}",
        f" Profit Factor (Trades):   {pf_text}",
        "-" * 65,
        " Signal Mix (short / flat / long):",
        *[
            f"   {name:<9} {mix['short']:5.0%} / {mix['flat']:5.0%} / {mix['long']:5.0%}"
            for name, mix in metrics.get("signal_mix", {}).items()
        ],
        *[f" WARNING: {w}" for w in metrics.get("collapse_warnings", [])],
        "=" * 65,
    ])


def save_backtest_results(result: BacktestResult, results_dir: str, show_plot: bool = False) -> Dict[str, str]:
    """Writes portfolio CSV, metrics JSON, text report and equity curve PNG."""
    out = Path(results_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = {
        "csv": out / "backtest_portfolio_history.csv",
        "metrics": out / "evaluation_metrics.json",
        "report": out / "backtest_report.txt",
        "plot": out / "equity_curve.png",
    }
    pd.DataFrame({
        "timestamp": result.timestamps,
        "portfolio_value": result.portfolio_values,
        "step_return": [0.0] + result.step_returns,
    }).to_csv(paths["csv"], index=False)
    with open(paths["metrics"], "w", encoding="utf-8") as f:
        json.dump(result.metrics, f, indent=2)
    with open(paths["report"], "w", encoding="utf-8") as f:
        f.write(result.report)

    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(12, 5))
    plt.plot(result.portfolio_values, color="#2563EB", linewidth=1.8, label="Ensemble Portfolio Equity (₹)")
    plt.axhline(result.initial_balance, color="gray", linestyle="--", alpha=0.7, label="Initial Capital")
    plt.title(f"ATU Multi-Agent Ensemble Out-of-Sample Equity Curve ({result.ticker})", fontsize=13)
    plt.xlabel("Intraday 5-Min Steps", fontsize=11)
    plt.ylabel("Portfolio Balance (₹)", fontsize=11)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(paths["plot"], dpi=150)
    if show_plot:
        plt.show()
    else:
        plt.close(fig)
    return {k: str(v) for k, v in paths.items()}


# =====================================================================
# End-to-end pipeline + CLI
# =====================================================================
def run_pipeline(
    ticker: str = "AAPL",
    workspace: str = "ATU_Workspace",
    total_timesteps: int = 500_000,
    chunks: int = 10,
    period: str = "60d",
    data_dir: Optional[str] = None,
    use_finbert: bool = True,
    device: str = "auto",
    news_queries: Optional[Sequence[str]] = None,
    tensorboard: bool = True,
    require_sentiment: bool = False,
    include_archive: bool = False,
    use_news: bool = True,
) -> Tuple[Dict[str, BaseAlgorithm], BacktestResult]:
    """Same steps as ATU_GPU.ipynb cells 3-4. `total_timesteps` is per agent."""
    print("=" * 70)
    print("   ATU PIPELINE: REAL DATA -> PPO + A2C ENSEMBLE -> WALK-FORWARD BACKTEST   ")
    print("=" * 70)
    paths = WorkspacePaths(Path(workspace)).create()
    data = build_live_state_vector(
        ticker=ticker, period=period,
        data_dir=str(data_dir) if data_dir else str(paths.datasets),
        use_finbert=use_finbert, device=device, news_queries=news_queries,
        require_sentiment=require_sentiment, include_archive=include_archive, use_news=use_news,
    )
    train_df, test_df = split_train_test(data.final_state_vector, data.market_processor)
    steps_per_chunk = max(1, math.ceil(total_timesteps / chunks))
    models = train_ensemble(
        train_df, data.env_kwargs, workspace=str(paths.root),
        chunks=chunks, steps_per_chunk=steps_per_chunk, tensorboard=tensorboard,
    )
    result = run_ensemble_backtest(models, test_df, data.env_kwargs, data.bars_per_day, ticker)
    saved = save_backtest_results(result, str(paths.results))
    print("\n" + result.report)
    for label, path in saved.items():
        print(f"  {label:<8} {path}")
    return models, result


def main(argv: Optional[Sequence[str]] = None) -> None:
    import argparse
    import matplotlib

    matplotlib.use("Agg")

    parser = argparse.ArgumentParser(description="ATU real-data PPO + A2C ensemble pipeline")
    parser.add_argument("--ticker", default="AAPL", help="yfinance symbol, e.g. AAPL or RELIANCE.NS (default: AAPL)")
    parser.add_argument("--workspace", "--model-save-path", dest="workspace", default="ATU_Workspace",
                        help="Folder for datasets/, checkpoints/, training_logs/, results/ "
                             "(Colab: /content/drive/MyDrive/ATU_Workspace)")
    parser.add_argument("--timesteps", type=int, default=500_000, help="Training timesteps PER AGENT (default: 500000)")
    parser.add_argument("--chunks", type=int, default=10, help="Checkpoint chunks per agent (default: 10)")
    parser.add_argument("--period", default="60d", help="yfinance period for 5m candles (max 60d)")
    parser.add_argument("--data-dir", default=None, help="Headline cache/archive folder (default: <workspace>/datasets)")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"], help="FinBERT device")
    parser.add_argument("--no-finbert", dest="finbert", action="store_false", help="Skip FinBERT (neutral sentiment)")
    parser.add_argument("--no-news", dest="news", action="store_false",
                        help="Price data only: skip news fetching and FinBERT entirely")
    parser.add_argument("--news-query", action="append", default=None,
                        help="Override news search query (repeatable). Default: '<company> stock', '<symbol> stock'")
    parser.add_argument("--require-sentiment", action="store_true", help="Fail if no bar receives real sentiment")
    parser.add_argument("--include-archive", action="store_true",
                        help="Also scan the 2009-2020 archive CSVs for headlines (slow; rarely overlaps)")
    parser.add_argument("--no-tensorboard", dest="tensorboard", action="store_false", help="Disable TensorBoard logging")
    parser.add_argument("--live-data", action="store_true", help=argparse.SUPPRESS)  # always on; kept for old commands
    args = parser.parse_args(argv)

    run_pipeline(
        ticker=args.ticker,
        workspace=args.workspace,
        total_timesteps=args.timesteps,
        chunks=args.chunks,
        period=args.period,
        data_dir=args.data_dir,
        use_finbert=args.finbert,
        device=args.device,
        news_queries=args.news_query,
        tensorboard=args.tensorboard,
        require_sentiment=args.require_sentiment,
        include_archive=args.include_archive,
        use_news=args.news,
    )


if __name__ == "__main__":
    main()
