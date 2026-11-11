"""
ConstrainedTradingEnv Module
============================
Intraday 5-minute trading environment (Gymnasium API) with continuous compounding
and Margin Intraday Square-off (MIS) frictions.

Design (v4):
------------
1. Raw-price accounting:
   P&L, account balance and stop-loss checks use the UNSCALED close price
   (`raw_close` column). Scaled features are used only for the observation.
2. Discrete execution:
   action > +threshold -> target Long (+1); action < -threshold -> target Short (-1);
   otherwise -> Flat (0). Position size is fixed at entry and never resized while
   held. Brokerage is charged only on entry and exit legs (a flip = exit + entry).
3. Reward = net profit of the step in currency:
   (equity after step) - (equity before step), where equity = cash + mark-to-market
   of the open position. Fees are already deducted from cash, so they are included.
   No multipliers, no clipping.
4. Position sizing:
   Notional = account_balance * max_risk_per_trade / |loss_limit| at entry, so a
   stop-loss hit loses ~max_risk_per_trade of equity (defaults: 1% / 1% -> 1.0x).
5. Rules:
   - Max `max_daily_trades` entries per calendar day.
   - Stop-loss: the stop price is entry * (1 + loss_limit) for longs and
     entry * (1 - loss_limit) for shorts. If the next candle's raw low (long) or raw
     high (short) breaches it, the position is closed EXACTLY at the stop price and the
     agent is locked out for the rest of the day. Without `raw_high`/`raw_low` columns
     the check falls back to the close price.
   - 15:15 MIS square-off: positions are closed on the first bar at/after 15:15; no
     new entries until the next session.
   - Positions are never carried overnight (safety close on date rollover).
   - One unbroken episode over the whole dataset; terminates only at the final bar.

Observation:
------------
[market features (scaled; `raw_close` excluded)] +
[unrealized_profit, current_position_direction, time_of_day, trades_remaining, locked_out_status]
"""

from datetime import time
from typing import Any, Dict, List, Optional, Tuple, Union
import warnings
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd


RAW_PRICE_COLUMN: str = "raw_close"
RAW_HIGH_COLUMN: str = "raw_high"
RAW_LOW_COLUMN: str = "raw_low"


def _seconds(t: time) -> int:
    return t.hour * 3600 + t.minute * 60 + t.second


class ConstrainedTradingEnv(gym.Env):
    """
    Action Space:
        Box(-1.0, 1.0, shape=(1,)), interpreted as a discrete target position.

    Observation Space:
        Box(num_features + 5,):
            market features...,
            unrealized_profit          (fraction of entry notional, gross of fees),
            current_position_direction (-1, 0, 1),
            time_of_day                (fraction of session elapsed, 0..1),
            trades_remaining           (integer, 0..max_daily_trades),
            locked_out_status          (0 or 1).
    """

    metadata = {"render_modes": ["human"], "version": "4.0.0"}

    STATE_FEATURE_NAMES: Tuple[str, ...] = (
        "unrealized_profit",
        "current_position_direction",
        "time_of_day",
        "trades_remaining",
        "locked_out_status",
    )

    def __init__(
        self,
        df: pd.DataFrame,
        initial_balance: float = 100000.0,
        max_risk_per_trade: float = 0.01,
        profit_target: float = 0.02,
        loss_limit: float = -0.01,
        commission_rate: float = 0.0003,
        max_daily_trades: int = 3,
        auto_square_off_time: time = time(15, 15, 0),
        price_col: str = "Close",
        raw_price_col: str = RAW_PRICE_COLUMN,
        raw_high_col: str = RAW_HIGH_COLUMN,
        raw_low_col: str = RAW_LOW_COLUMN,
        session_start_time: time = time(9, 15, 0),
        session_end_time: time = time(15, 30, 0),
        action_threshold: float = 0.1,
    ) -> None:
        """
        Args:
            df: Feature DataFrame with DatetimeIndex (or 'date'/'Date' column). Must contain
                unscaled prices in `raw_price_col`; otherwise `price_col` is used and must
                itself be unscaled.
            initial_balance: Starting equity.
            max_risk_per_trade: Fraction of equity lost if the stop-loss is hit (drives sizing).
            profit_target: Reserved; not enforced (the agent chooses its exits).
            loss_limit: Stop-loss threshold as a price-move fraction of the position (negative).
            commission_rate: Fee fraction charged on each leg's traded notional.
            max_daily_trades: Maximum entries per calendar day.
            auto_square_off_time: MIS cut-off time.
            price_col: Fallback price column if `raw_price_col` is absent.
            raw_price_col: Column holding unscaled close prices (excluded from observations).
            raw_high_col / raw_low_col: Optional unscaled high/low columns used for
                intra-candle stop-loss detection (excluded from observations).
            session_start_time / session_end_time: Session bounds for `time_of_day`.
            action_threshold: |action| above this opens/holds a directional position.
        """
        super().__init__()

        if df.empty or len(df) < 3:
            raise ValueError("Input DataFrame must contain at least 3 rows of market data.")
        if loss_limit >= 0:
            raise ValueError(f"loss_limit must be negative, got {loss_limit}")
        if max_risk_per_trade <= 0:
            raise ValueError(f"max_risk_per_trade must be positive, got {max_risk_per_trade}")
        if _seconds(session_end_time) <= _seconds(session_start_time):
            raise ValueError("session_end_time must be after session_start_time")

        # Timestamps
        if isinstance(df.index, pd.DatetimeIndex):
            self.dates = df.index
        elif "date" in df.columns or "Date" in df.columns:
            date_col = "date" if "date" in df.columns else "Date"
            self.dates = pd.DatetimeIndex(pd.to_datetime(df[date_col]))
        else:
            self.dates = pd.DatetimeIndex(pd.to_datetime(df.index))

        self.df = df.copy().reset_index(drop=True)
        self.initial_balance: float = float(initial_balance)
        self.max_risk_per_trade: float = float(max_risk_per_trade)
        self.profit_target: float = float(profit_target)
        self.loss_limit: float = float(loss_limit)
        self.commission_rate: float = float(commission_rate)
        self.max_daily_trades: int = int(max_daily_trades)
        self.auto_square_off_time: time = auto_square_off_time
        self.action_threshold: float = float(action_threshold)
        self._session_start_s: int = _seconds(session_start_time)
        self._session_len_s: int = _seconds(session_end_time) - self._session_start_s

        # Fixed leverage so that a stop-loss hit costs ~max_risk_per_trade of equity
        self.position_leverage: float = self.max_risk_per_trade / abs(self.loss_limit)

        # Unscaled price series for all accounting
        if raw_price_col in self.df.columns:
            self.raw_price_col: str = raw_price_col
        else:
            fallback = next(
                (c for c in (price_col, price_col.lower(), "Close", "close") if c in self.df.columns),
                None,
            )
            if fallback is None:
                raise KeyError(f"Neither '{raw_price_col}' nor '{price_col}' found in DataFrame.")
            warnings.warn(
                f"'{raw_price_col}' column not found; using '{fallback}' for P&L. "
                "It MUST contain unscaled prices.",
                stacklevel=2,
            )
            self.raw_price_col = fallback

        self.prices: np.ndarray = self.df[self.raw_price_col].to_numpy(dtype=np.float64)
        if not np.all(np.isfinite(self.prices)) or np.any(self.prices <= 0.0):
            raise ValueError(
                f"Price column '{self.raw_price_col}' must contain strictly positive, finite, "
                "UNSCALED prices. MinMax-scaled prices (which reach 0.0) produce meaningless returns."
            )

        # Optional unscaled high/low for intra-candle stop-loss detection
        self.highs: Optional[np.ndarray] = None
        self.lows: Optional[np.ndarray] = None
        if raw_high_col in self.df.columns and raw_low_col in self.df.columns:
            self.highs = self.df[raw_high_col].to_numpy(dtype=np.float64)
            self.lows = self.df[raw_low_col].to_numpy(dtype=np.float64)
            for name, arr in ((raw_high_col, self.highs), (raw_low_col, self.lows)):
                if not np.all(np.isfinite(arr)) or np.any(arr <= 0.0):
                    raise ValueError(f"'{name}' must contain strictly positive, finite, UNSCALED prices.")

        # Observation features: all numeric columns except the dedicated raw price columns
        numeric_df = self.df.select_dtypes(include=[np.number])
        excluded = [raw_high_col, raw_low_col]
        if self.raw_price_col == raw_price_col:
            excluded.append(self.raw_price_col)
        numeric_df = numeric_df.drop(columns=[c for c in excluded if c in numeric_df.columns])
        self.feature_columns: List[str] = list(numeric_df.columns)
        self.features: np.ndarray = numeric_df.to_numpy(dtype=np.float32)
        self.num_rows: int = len(self.df)
        self.num_features: int = self.features.shape[1]

        self.action_space: spaces.Box = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        self.observation_space: spaces.Box = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.num_features + len(self.STATE_FEATURE_NAMES),),
            dtype=np.float32,
        )

        self._reset_state()

    # ------------------------------------------------------------------
    # State helpers
    # ------------------------------------------------------------------
    def _reset_state(self) -> None:
        self.current_step: int = 0
        self.current_date = self.dates[0].date()
        self.unique_dates_seen: set = {self.current_date}
        self.account_balance: float = self.initial_balance  # realized cash
        self.daily_trades_count: int = 0
        self.total_trades: int = 0
        self.closed_trades: int = 0
        self.winning_trades: int = 0
        self.total_fees_paid: float = 0.0
        self.locked_out_for_day: bool = False
        self.last_action: float = 0.0
        self._clear_position()

    def _clear_position(self) -> None:
        self.position_direction: int = 0
        self.shares: float = 0.0
        self.entry_price: float = 0.0
        self.entry_notional: float = 0.0
        self.entry_fee: float = 0.0
        self.unrealized_pnl: float = 0.0  # currency, gross of fees

    @property
    def is_active_position(self) -> bool:
        return self.position_direction != 0

    @property
    def unrealized_profit(self) -> float:
        """Open-position price move as a fraction of entry notional (gross of fees)."""
        if not self.is_active_position or self.entry_notional <= 0.0:
            return 0.0
        return self.unrealized_pnl / self.entry_notional

    @property
    def current_unrealized_profit(self) -> float:
        """Alias for backward compatibility."""
        return self.unrealized_profit

    @property
    def equity(self) -> float:
        """Mark-to-market equity: realized cash + open-position P&L."""
        return self.account_balance + self.unrealized_pnl

    @property
    def balance(self) -> float:
        """Backward-compatible alias for mark-to-market equity."""
        return self.equity

    @property
    def trades_remaining(self) -> int:
        return max(0, self.max_daily_trades - self.daily_trades_count)

    def _time_of_day(self) -> float:
        t = self.dates[self.current_step].time()
        frac = (_seconds(t) - self._session_start_s) / self._session_len_s
        return float(np.clip(frac, 0.0, 1.0))

    def _get_observation(self) -> np.ndarray:
        state = np.array(
            [
                self.unrealized_profit,
                float(self.position_direction),
                self._time_of_day(),
                float(self.trades_remaining),
                float(self.locked_out_for_day),
            ],
            dtype=np.float32,
        )
        return np.concatenate([self.features[self.current_step], state]).astype(np.float32)

    def _action_to_target(self, action: Union[float, np.ndarray]) -> Tuple[float, int]:
        act_val = float(np.asarray(action, dtype=np.float64).reshape(-1)[0])
        if not np.isfinite(act_val):
            act_val = 0.0
        act_val = float(np.clip(act_val, -1.0, 1.0))
        if act_val > self.action_threshold:
            return act_val, 1
        if act_val < -self.action_threshold:
            return act_val, -1
        return act_val, 0

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------
    def _open_position(self, direction: int, price: float) -> None:
        notional = self.account_balance * self.position_leverage
        fee = notional * self.commission_rate
        self.account_balance -= fee
        self.total_fees_paid += fee

        self.position_direction = int(direction)
        self.shares = notional / price
        self.entry_price = float(price)
        self.entry_notional = float(notional)
        self.entry_fee = float(fee)
        self.unrealized_pnl = 0.0

        self.daily_trades_count += 1
        self.total_trades += 1

    def _close_position(self, price: float) -> float:
        """Closes at `price`, realizes P&L into cash, returns the trade's net P&L (both fees)."""
        gross_pnl = self.position_direction * self.shares * (price - self.entry_price)
        exit_fee = self.shares * price * self.commission_rate
        self.account_balance += gross_pnl - exit_fee
        self.total_fees_paid += exit_fee

        trade_net_pnl = gross_pnl - self.entry_fee - exit_fee
        self.closed_trades += 1
        if trade_net_pnl > 0.0:
            self.winning_trades += 1

        self._clear_position()
        return float(trade_net_pnl)

    @property
    def stop_price(self) -> float:
        """Price at which the open position loses exactly |loss_limit| of its entry notional."""
        if self.position_direction == 1:
            return self.entry_price * (1.0 + self.loss_limit)
        if self.position_direction == -1:
            return self.entry_price * (1.0 - self.loss_limit)
        return 0.0

    def _stop_breached(self, idx: int) -> bool:
        """True if candle `idx` traded through the stop price (intra-candle when high/low are known)."""
        close = self.prices[idx]
        if self.position_direction == 1:
            low = min(close, self.lows[idx]) if self.lows is not None else close
            return low <= self.stop_price
        if self.position_direction == -1:
            high = max(close, self.highs[idx]) if self.highs is not None else close
            return high >= self.stop_price
        return False

    def _mark_to_market(self, price: float) -> None:
        if self.is_active_position:
            self.unrealized_pnl = self.position_direction * self.shares * (price - self.entry_price)

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        self._reset_state()
        info = self._build_info(
            step_net_pnl=0.0,
            step_return=0.0,
            closed_trade_pnls=[],
            stop_loss_hit=False,
        )
        return self._get_observation(), info

    def step(
        self, action: Union[float, np.ndarray]
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        equity_before = self.equity
        price_now = float(self.prices[self.current_step])
        time_now = self.dates[self.current_step].time()
        closed_trade_pnls: List[float] = []
        stop_loss_hit = False

        # 1. Decide target position at the current bar's close
        act_val, target = self._action_to_target(action)
        if self.locked_out_for_day or time_now >= self.auto_square_off_time:
            target = 0

        # 2. Exit leg (close or first half of a flip)
        if self.is_active_position and target != self.position_direction:
            closed_trade_pnls.append(self._close_position(price_now))

        # 3. Entry leg (subject to daily cap)
        if target != 0 and not self.is_active_position and self.daily_trades_count < self.max_daily_trades:
            self._open_position(target, price_now)

        # 4. Advance one bar and mark the position to market on raw prices
        self.current_step += 1
        price_next = float(self.prices[self.current_step])
        next_ts = self.dates[self.current_step]
        self._mark_to_market(price_next)

        # 5. In-trade risk rules at the new bar
        new_day = next_ts.date() != self.current_date
        if self.is_active_position:
            if new_day:
                # Safety: never carry an intraday position overnight
                closed_trade_pnls.append(self._close_position(price_next))
            elif self._stop_breached(self.current_step):
                # Exit exactly at the stop price: loss is capped at |loss_limit| + fees
                closed_trade_pnls.append(self._close_position(self.stop_price))
                stop_loss_hit = True
                self.locked_out_for_day = True
            elif next_ts.time() >= self.auto_square_off_time:
                closed_trade_pnls.append(self._close_position(price_next))

        # 6. Date rollover: reset daily counters for the new session
        self.unique_dates_seen.add(next_ts.date())
        if new_day:
            self.current_date = next_ts.date()
            self.daily_trades_count = 0
            self.locked_out_for_day = False

        # 7. Single unbroken episode: terminate only at the last bar
        terminated = self.current_step >= self.num_rows - 1
        truncated = False
        if terminated and self.is_active_position:
            closed_trade_pnls.append(self._close_position(price_next))

        # 8. Reward = net currency profit of this step (fees included)
        step_net_pnl = float(self.equity - equity_before)
        step_return = step_net_pnl / equity_before if equity_before > 0.0 else 0.0
        self.last_action = act_val

        info = self._build_info(
            step_net_pnl=step_net_pnl,
            step_return=step_return,
            closed_trade_pnls=closed_trade_pnls,
            stop_loss_hit=stop_loss_hit,
        )
        info["action"] = float(act_val)
        return self._get_observation(), step_net_pnl, terminated, truncated, info

    def _build_info(
        self,
        step_net_pnl: float,
        step_return: float,
        closed_trade_pnls: List[float],
        stop_loss_hit: bool,
    ) -> Dict[str, Any]:
        ts = self.dates[self.current_step]
        return {
            "step": self.current_step,
            "date": str(self.current_date),
            "time": str(ts.time()),
            "price": float(self.prices[self.current_step]),
            "account_balance": float(self.account_balance),
            "balance": float(self.equity),
            "equity": float(self.equity),
            "unrealized_profit": float(self.unrealized_profit),
            "unrealized_pnl": float(self.unrealized_pnl),
            "position_direction": int(self.position_direction),
            "step_net_pnl": float(step_net_pnl),
            "step_profit": float(step_return),  # net portfolio return of this step
            "daily_trades_count": int(self.daily_trades_count),
            "trades_remaining": int(self.trades_remaining),
            "total_trades": int(self.total_trades),
            "closed_trades": int(self.closed_trades),
            "winning_trades": int(self.winning_trades),
            "total_fees_paid": float(self.total_fees_paid),
            "closed_trade_pnls": list(closed_trade_pnls),
            "trade_closed": bool(closed_trade_pnls),
            "trade_closed_profit": bool(any(p > 0.0 for p in closed_trade_pnls)),
            "stop_loss_hit": bool(stop_loss_hit),
            "locked_out_for_day": bool(self.locked_out_for_day),
            "days_processed": int(len(self.unique_dates_seen)),
        }


if __name__ == "__main__":
    np.random.seed(42)

    # 5 trading days of 5-minute bars (09:15-15:25, 75 bars/day)
    n_rows = 375
    timestamps = []
    curr_date = pd.Timestamp("2024-01-02")
    while len(timestamps) < n_rows:
        if curr_date.weekday() < 5:
            for ts in pd.date_range(
                start=curr_date.replace(hour=9, minute=15),
                end=curr_date.replace(hour=15, minute=25),
                freq="5min",
            ):
                timestamps.append(ts)
                if len(timestamps) == n_rows:
                    break
        curr_date += pd.Timedelta(days=1)

    close_prices = 150.0 * np.cumprod(1.0 + np.random.normal(0.0001, 0.003, n_rows))
    mock_df = pd.DataFrame(
        {
            "Close": (close_prices - close_prices.min()) / (close_prices.max() - close_prices.min()),
            "RSI": np.random.uniform(0.3, 0.7, n_rows),
            "daily_sentiment": np.random.uniform(-0.5, 0.5, n_rows),
            RAW_PRICE_COLUMN: close_prices,
            RAW_HIGH_COLUMN: close_prices * (1.0 + np.abs(np.random.normal(0, 0.002, n_rows))),
            RAW_LOW_COLUMN: close_prices * (1.0 - np.abs(np.random.normal(0, 0.002, n_rows))),
        },
        index=pd.DatetimeIndex(timestamps),
    )

    env = ConstrainedTradingEnv(df=mock_df)
    obs, info = env.reset(seed=42)
    print(f"Observation shape: {obs.shape} (features: {env.feature_columns} + {list(env.STATE_FEATURE_NAMES)})")
    print(f"Initial equity: {env.equity:,.2f}\n")

    done = False
    total_reward = 0.0
    step_idx = 0
    while not done:
        action = np.array([0.9 if (step_idx // 20) % 2 == 0 else 0.0])
        obs, reward, terminated, truncated, step_info = env.step(action)
        total_reward += reward
        done = terminated or truncated
        step_idx += 1
        if step_info["trade_closed"] and step_idx < 120:
            print(
                f"{step_info['date']} {step_info['time']} | closed {step_info['closed_trade_pnls']} | "
                f"stop={step_info['stop_loss_hit']} | equity={step_info['equity']:,.2f}"
            )

    print(f"\nSteps: {step_idx} | Days: {step_info['days_processed']} | Trades: {step_info['total_trades']}")
    print(f"Fees: {step_info['total_fees_paid']:,.2f} | Final equity: {step_info['equity']:,.2f}")
    print(f"Sum of rewards: {total_reward:,.2f} == net P&L: {step_info['equity'] - env.initial_balance:,.2f}")
