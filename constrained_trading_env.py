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
   Default action space is Discrete(3): 0 = Short, 1 = Flat, 2 = Long. Every choice
   gets equal exploration, which prevents the "always one direction" collapse seen
   with a clipped Gaussian on Box(-1, 1), where the flat band |a| <= 0.1 was almost
   never sampled. `discrete_actions=False` restores the legacy Box(-1, 1) interface
   (action > +threshold -> Long, < -threshold -> Short, else Flat).
   Position size is fixed at entry and never resized while held. Brokerage is
   charged only on entry and exit legs (a flip = exit + entry).
3. Reward (reward_mode="shaped", default), in currency:
   step net P&L (equity change incl. fees)
   - 0.5 x each deepening of the day's maximum drawdown below its peak equity
   +/- 0.05% of the day's starting equity when a day closes green / red (0 if flat).
   Staying flat is free. reward_mode="pnl" gives the plain step net P&L. The info
   dict always reports the true P&L (`step_net_pnl`), so backtests are unaffected.
4. Risk & sizing:
   1R (initial stop distance) = 2 x ATR(14) / price, clipped to [0.3%, 2%] (fixed
   |loss_limit| during ATR warm-up). Notional = equity * max_risk_per_trade / 1R, capped
   at 1x, so an initial-stop hit costs ~0.5% of equity. Trailing stop: from +1R the stop
   follows the best price at 1R distance (break-even first). Take-profit at +3R.
   Daily loss cap: at -1% of the day's starting equity the position is closed and the
   agent is locked out until the next session.
5. Rules:
   - Max `max_daily_trades` entries per calendar day.
   - Stop-loss: if a candle's raw low (long) or raw high (short) breaches the current
     stop, the position is closed at the stop price. A losing stop locks the agent out
     for the rest of the day; a trailed stop that exits in profit does not. Gap slippage: if that candle's raw open is
     already strictly beyond the stop (below it for a long, above it for a short), the
     fill is the raw open instead. Without `raw_high`/`raw_low` the breach check uses
     the close; without `raw_open` the fill is the stop price.
   - Minimum holding period (`min_hold_bars`) and post-exit re-entry cooldown
     (`reentry_cooldown_bars`) to stop fee-burning churn.
   - 15:15 MIS square-off: positions are closed on the first bar at/after 15:15; no
     new entries until the next session.
   - Positions are closed at the close of each session's last bar (also covers 1h bars
     and half-days) and never carried overnight (safety close on date rollover).
   - One unbroken episode over the whole dataset; terminates only at the final bar.

Observation:
------------
[market features (scaled; `raw_close` excluded)] +
[unrealized_profit, current_position_direction, time_of_day, trades_remaining, locked_out_status,
 action_blocked]
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
RAW_OPEN_COLUMN: str = "raw_open"


def _seconds(t: time) -> int:
    return t.hour * 3600 + t.minute * 60 + t.second


class ConstrainedTradingEnv(gym.Env):
    """
    Action Space:
        Discrete(3): 0 = Short, 1 = Flat, 2 = Long (target position).
        Legacy (discrete_actions=False): Box(-1.0, 1.0, shape=(1,)).

    Observation Space:
        Box(num_features + 6,):
            market features...,
            unrealized_profit          (fraction of entry notional, gross of fees),
            current_position_direction (-1, 0, 1),
            time_of_day                (fraction of session elapsed, 0..1),
            trades_remaining           (integer, 0..max_daily_trades),
            locked_out_status          (0 or 1),
            action_blocked             (1 while a min-hold or re-entry cooldown blocks the agent).
    """

    metadata = {"render_modes": ["human"], "version": "4.0.0"}

    STATE_FEATURE_NAMES: Tuple[str, ...] = (
        "unrealized_profit",
        "current_position_direction",
        "time_of_day",
        "trades_remaining",
        "locked_out_status",
        "action_blocked",
    )

    def __init__(
        self,
        df: pd.DataFrame,
        initial_balance: float = 100000.0,
        max_risk_per_trade: float = 0.005,
        profit_target: float = 0.02,
        loss_limit: float = -0.01,
        commission_rate: float = 0.0003,
        max_daily_trades: int = 3,
        auto_square_off_time: time = time(15, 15, 0),
        price_col: str = "Close",
        raw_price_col: str = RAW_PRICE_COLUMN,
        raw_high_col: str = RAW_HIGH_COLUMN,
        raw_low_col: str = RAW_LOW_COLUMN,
        raw_open_col: str = RAW_OPEN_COLUMN,
        session_start_time: time = time(9, 15, 0),
        session_end_time: time = time(15, 30, 0),
        action_threshold: float = 0.1,
        discrete_actions: bool = True,
        min_hold_bars: int = 3,
        reentry_cooldown_bars: int = 3,
        stop_atr_multiple: Optional[float] = 2.0,
        atr_window: int = 14,
        min_stop_pct: float = 0.003,
        max_stop_pct: float = 0.02,
        max_position_leverage: float = 1.0,
        trail_activation_r: Optional[float] = 1.0,
        trail_distance_r: float = 1.0,
        take_profit_r: Optional[float] = 3.0,
        daily_loss_limit: Optional[float] = 0.01,
        reward_mode: str = "shaped",
        drawdown_penalty: float = 0.5,
        day_result_bonus: float = 0.0005,
    ) -> None:
        """
        Args:
            df: Feature DataFrame with DatetimeIndex (or 'date'/'Date' column). Must contain
                unscaled prices in `raw_price_col`; otherwise `price_col` is used and must
                itself be unscaled.
            initial_balance: Starting equity.
            max_risk_per_trade: Fraction of equity lost if the initial stop is hit (drives sizing).
            profit_target: Legacy; unused (see take_profit_r).
            loss_limit: Fixed stop distance (negative price-move fraction) used when
                `stop_atr_multiple` is None or ATR is not yet available.
            commission_rate: Fee fraction charged on each leg's traded notional.
            max_daily_trades: Maximum entries per calendar day.
            auto_square_off_time: MIS cut-off time.
            price_col: Fallback price column if `raw_price_col` is absent.
            raw_price_col: Column holding unscaled close prices (excluded from observations).
            raw_high_col / raw_low_col: Optional unscaled high/low columns used for
                intra-candle stop-loss detection (excluded from observations).
            raw_open_col: Optional unscaled open column used for gap-slippage stop fills
                (excluded from observations).
            session_start_time / session_end_time: Session bounds for `time_of_day`.
            action_threshold: Legacy Box mode only: |action| above this is directional.
            discrete_actions: True (default) -> Discrete(3) actions; False -> Box(-1, 1).
            min_hold_bars: A position cannot be closed or flipped by the agent until it has
                been held this many bars (stops, square-off and rollover still apply).
            reentry_cooldown_bars: After any exit, no new entry for this many bars.
            stop_atr_multiple: Initial stop distance = this x ATR(atr_window) / price,
                clipped to [min_stop_pct, max_stop_pct]. None -> fixed |loss_limit|.
            max_position_leverage: Cap on notional / equity.
            trail_activation_r: Once the best price is this many stop-distances (R) in
                profit, the stop trails the best price at `trail_distance_r` R
                (1.0 / 1.0 -> break-even at +1R). None disables trailing.
            take_profit_r: Close the trade at entry +/- this many R. None disables it.
            daily_loss_limit: Close and lock out for the day once equity falls this
                fraction below the day's starting equity. None disables it.
            reward_mode: "shaped" (default) or "pnl" (reward = step net P&L only).
            drawdown_penalty: Shaped mode: extra penalty per unit by which the day's
                maximum drawdown (below the day's peak equity) deepens.
            day_result_bonus: Shaped mode: at each day's close, +/- this fraction of the
                day's starting equity for a green / red day (0 for a flat day).
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
        self.discrete_actions: bool = bool(discrete_actions)
        self.min_hold_bars: int = max(0, int(min_hold_bars))
        self.reentry_cooldown_bars: int = max(0, int(reentry_cooldown_bars))
        self._session_start_s: int = _seconds(session_start_time)
        self._session_len_s: int = _seconds(session_end_time) - self._session_start_s

        self.stop_atr_multiple: Optional[float] = None if stop_atr_multiple is None else float(stop_atr_multiple)
        self.min_stop_pct: float = float(min_stop_pct)
        self.max_stop_pct: float = float(max_stop_pct)
        self.max_position_leverage: float = float(max_position_leverage)
        self.trail_activation_r: Optional[float] = None if trail_activation_r is None else float(trail_activation_r)
        self.trail_distance_r: float = float(trail_distance_r)
        self.take_profit_r: Optional[float] = None if take_profit_r is None else float(take_profit_r)
        self.daily_loss_limit: Optional[float] = None if daily_loss_limit is None else float(daily_loss_limit)
        if not 0.0 < self.min_stop_pct <= self.max_stop_pct:
            raise ValueError("Require 0 < min_stop_pct <= max_stop_pct")
        if self.max_position_leverage <= 0.0:
            raise ValueError("max_position_leverage must be positive")
        if reward_mode not in ("shaped", "pnl"):
            raise ValueError(f"reward_mode must be 'shaped' or 'pnl', got {reward_mode!r}")
        self.reward_mode: str = reward_mode
        self.drawdown_penalty: float = float(drawdown_penalty)
        self.day_result_bonus: float = float(day_result_bonus)

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

        # Optional unscaled open for gap-slippage stop fills
        self.opens: Optional[np.ndarray] = None
        if raw_open_col in self.df.columns:
            self.opens = self.df[raw_open_col].to_numpy(dtype=np.float64)
            if not np.all(np.isfinite(self.opens)) or np.any(self.opens <= 0.0):
                raise ValueError(f"'{raw_open_col}' must contain strictly positive, finite, UNSCALED prices.")

        # Causal ATR as a fraction of price (uses bars <= i only), for stops and sizing
        prev_close = np.concatenate([[self.prices[0]], self.prices[:-1]])
        if self.highs is not None and self.lows is not None:
            true_range = np.maximum.reduce([
                self.highs - self.lows,
                np.abs(self.highs - prev_close),
                np.abs(self.lows - prev_close),
            ])
        else:
            true_range = np.abs(self.prices - prev_close)
        atr = pd.Series(true_range).rolling(int(atr_window), min_periods=int(atr_window)).mean().to_numpy()
        self.atr_frac: np.ndarray = atr / self.prices  # NaN during warm-up -> fixed stop

        # Observation features: all numeric columns except the dedicated raw price columns
        numeric_df = self.df.select_dtypes(include=[np.number])
        excluded = [raw_high_col, raw_low_col, raw_open_col]
        if self.raw_price_col == raw_price_col:
            excluded.append(self.raw_price_col)
        numeric_df = numeric_df.drop(columns=[c for c in excluded if c in numeric_df.columns])
        self.feature_columns: List[str] = list(numeric_df.columns)
        self.features: np.ndarray = numeric_df.to_numpy(dtype=np.float32)
        self.num_rows: int = len(self.df)
        # Last bar of each session (next bar is another day): positions are squared off at
        # its close, so coarse bars (1h) and half-days never carry a position overnight.
        day_index = np.array([d.toordinal() for d in self.dates.date])
        self.last_bar_of_day: np.ndarray = np.append(day_index[1:] != day_index[:-1], True)
        self.num_features: int = self.features.shape[1]

        self.action_space: spaces.Space = (
            spaces.Discrete(3)
            if self.discrete_actions
            else spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        )
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
        self.bars_since_exit: int = 10**9  # no exit yet -> no cooldown
        self.day_start_equity: float = self.initial_balance
        self.day_peak_equity: float = self.initial_balance
        self.day_max_drawdown: float = 0.0  # deepest drop below the day's peak so far (currency)
        self._clear_position()

    def _clear_position(self) -> None:
        self.position_direction: int = 0
        self.shares: float = 0.0
        self.entry_price: float = 0.0
        self.entry_notional: float = 0.0
        self.entry_fee: float = 0.0
        self.unrealized_pnl: float = 0.0  # currency, gross of fees
        self.bars_held: int = 0
        self.stop_distance: float = 0.0  # initial stop distance as a price fraction (1R)
        self.stop_level: float = 0.0
        self.take_profit_level: Optional[float] = None
        self.best_price: float = 0.0

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

    @property
    def exit_blocked(self) -> bool:
        """Agent-initiated exits are blocked during the minimum holding period."""
        return self.is_active_position and self.bars_held < self.min_hold_bars

    @property
    def entry_blocked(self) -> bool:
        """New entries are blocked during the post-exit cooldown."""
        return not self.is_active_position and self.bars_since_exit < self.reentry_cooldown_bars

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
                float(self.exit_blocked or self.entry_blocked),
            ],
            dtype=np.float32,
        )
        return np.concatenate([self.features[self.current_step], state]).astype(np.float32)

    def _action_to_target(self, action: Union[int, float, np.ndarray]) -> Tuple[float, int]:
        """Returns (signal in [-1, 1], target direction in {-1, 0, 1})."""
        act_val = float(np.asarray(action, dtype=np.float64).reshape(-1)[0])
        if not np.isfinite(act_val):
            act_val = 1.0 if self.discrete_actions else 0.0
        if self.discrete_actions:
            target = int(np.clip(round(act_val), 0, 2)) - 1  # 0 short, 1 flat, 2 long
            return float(target), target
        act_val = float(np.clip(act_val, -1.0, 1.0))
        if act_val > self.action_threshold:
            return act_val, 1
        if act_val < -self.action_threshold:
            return act_val, -1
        return act_val, 0

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------
    def initial_stop_distance(self, idx: int) -> float:
        """1R as a price fraction: k x ATR / price clipped to [min, max], else |loss_limit|."""
        if self.stop_atr_multiple is not None and idx < len(self.atr_frac) and np.isfinite(self.atr_frac[idx]):
            return float(np.clip(self.stop_atr_multiple * self.atr_frac[idx], self.min_stop_pct, self.max_stop_pct))
        return abs(self.loss_limit)

    def _open_position(self, direction: int, price: float) -> None:
        stop_distance = self.initial_stop_distance(self.current_step)
        leverage = min(self.max_risk_per_trade / stop_distance, self.max_position_leverage)
        notional = self.account_balance * leverage
        fee = notional * self.commission_rate
        self.account_balance -= fee
        self.total_fees_paid += fee

        self.position_direction = int(direction)
        self.shares = notional / price
        self.entry_price = float(price)
        self.entry_notional = float(notional)
        self.entry_fee = float(fee)
        self.unrealized_pnl = 0.0
        self.stop_distance = stop_distance
        self.stop_level = price * (1.0 - direction * stop_distance)
        self.take_profit_level = (
            price * (1.0 + direction * self.take_profit_r * stop_distance) if self.take_profit_r is not None else None
        )
        self.best_price = float(price)

        self.bars_held = 0
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
        self.bars_since_exit = 0
        return float(trade_net_pnl)

    @property
    def stop_price(self) -> float:
        """Current protective stop (initial ATR stop, ratcheted by the trailing rule)."""
        return self.stop_level if self.is_active_position else 0.0

    def _update_trailing_stop(self, idx: int) -> None:
        """After bar `idx` closes: track the best price and ratchet the stop (never loosens)."""
        if not self.is_active_position:
            return
        if self.position_direction == 1:
            extreme = self.highs[idx] if self.highs is not None else self.prices[idx]
            self.best_price = max(self.best_price, float(extreme))
        else:
            extreme = self.lows[idx] if self.lows is not None else self.prices[idx]
            self.best_price = min(self.best_price, float(extreme))
        if self.trail_activation_r is None:
            return
        r = self.stop_distance * self.entry_price
        favorable = self.position_direction * (self.best_price - self.entry_price)
        if favorable >= self.trail_activation_r * r:
            trail = self.best_price - self.position_direction * self.trail_distance_r * r
            if self.position_direction == 1:
                self.stop_level = max(self.stop_level, trail)
            else:
                self.stop_level = min(self.stop_level, trail)

    def _take_profit_hit(self, idx: int) -> bool:
        if self.take_profit_level is None or not self.is_active_position:
            return False
        if self.position_direction == 1:
            high = max(self.prices[idx], self.highs[idx]) if self.highs is not None else self.prices[idx]
            return high >= self.take_profit_level
        low = min(self.prices[idx], self.lows[idx]) if self.lows is not None else self.prices[idx]
        return low <= self.take_profit_level

    def _take_profit_fill_price(self, idx: int) -> float:
        """Take-profit level, or the open when the candle gapped beyond it (better fill)."""
        tp = float(self.take_profit_level)
        if self.opens is None:
            return tp
        candle_open = float(self.opens[idx])
        if self.position_direction == 1 and candle_open > tp:
            return candle_open
        if self.position_direction == -1 and candle_open < tp:
            return candle_open
        return tp

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

    def _stop_fill_price(self, idx: int) -> float:
        """Stop price, or the candle's open when the candle gapped strictly through the stop."""
        stop = self.stop_price
        if self.opens is None:
            return stop
        candle_open = float(self.opens[idx])
        if self.position_direction == 1 and candle_open < stop:
            return candle_open
        if self.position_direction == -1 and candle_open > stop:
            return candle_open
        return stop

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
        forced_flat = (
            self.locked_out_for_day
            or time_now >= self.auto_square_off_time
            or bool(self.last_bar_of_day[self.current_step])
        )
        if forced_flat:
            target = 0

        if self.exit_blocked and not forced_flat:
            target = self.position_direction  # minimum holding period: keep the position

        # 2. Exit leg (close or first half of a flip)
        if self.is_active_position and target != self.position_direction:
            closed_trade_pnls.append(self._close_position(price_now))

        # 3. Entry leg (subject to daily cap and post-exit cooldown)
        if (
            target != 0
            and not self.is_active_position
            and not self.entry_blocked
            and self.daily_trades_count < self.max_daily_trades
        ):
            self._open_position(target, price_now)

        # 4. Advance one bar and mark the position to market on raw prices
        self.current_step += 1
        if self.is_active_position:
            self.bars_held += 1
        else:
            self.bars_since_exit += 1
        price_next = float(self.prices[self.current_step])
        next_ts = self.dates[self.current_step]
        self._mark_to_market(price_next)

        # 5. In-trade risk rules at the new bar (stop checked before take-profit: conservative)
        new_day = next_ts.date() != self.current_date
        trailing_stop_hit = take_profit_hit = daily_cap_hit = False
        if self.is_active_position:
            idx = self.current_step
            if new_day:
                # Safety: never carry an intraday position overnight
                closed_trade_pnls.append(self._close_position(price_next))
            elif self._stop_breached(idx):
                # Exit at the stop, or at the open if the candle gapped through it
                trade_pnl = self._close_position(self._stop_fill_price(idx))
                closed_trade_pnls.append(trade_pnl)
                if trade_pnl < 0.0:
                    stop_loss_hit = True
                    self.locked_out_for_day = True  # losing stop -> done for the day
                else:
                    trailing_stop_hit = True  # trailed stop locked in a gain
            elif self._take_profit_hit(idx):
                closed_trade_pnls.append(self._close_position(self._take_profit_fill_price(idx)))
                take_profit_hit = True
            elif next_ts.time() >= self.auto_square_off_time:
                closed_trade_pnls.append(self._close_position(price_next))
            else:
                self._update_trailing_stop(idx)

        # 5b. Daily loss cap on mark-to-market equity
        if (
            not new_day
            and self.daily_loss_limit is not None
            and not self.locked_out_for_day
            and self.equity <= self.day_start_equity * (1.0 - self.daily_loss_limit)
        ):
            if self.is_active_position:
                closed_trade_pnls.append(self._close_position(price_next))
            self.locked_out_for_day = True
            daily_cap_hit = True

        # 6. Date rollover: score the finished day, then reset daily counters
        terminated = self.current_step >= self.num_rows - 1
        if terminated and self.is_active_position:
            closed_trade_pnls.append(self._close_position(price_next))
        day_closed_pnl: Optional[float] = None
        day_base = self.day_start_equity
        if new_day or terminated:
            day_closed_pnl = self.equity - day_base
        drawdown_now = max(self.day_peak_equity, self.equity) - self.equity
        new_drawdown = 0.0 if new_day else max(0.0, drawdown_now - self.day_max_drawdown)
        self.unique_dates_seen.add(next_ts.date())
        if new_day:
            self.current_date = next_ts.date()
            self.daily_trades_count = 0
            self.locked_out_for_day = False
            self.day_start_equity = self.equity
            self.day_peak_equity = self.equity
            self.day_max_drawdown = 0.0
        else:
            self.day_peak_equity = max(self.day_peak_equity, self.equity)
            self.day_max_drawdown = max(self.day_max_drawdown, drawdown_now)

        # 7. Single unbroken episode: terminates only at the last bar (handled above)
        truncated = False

        # 8. Reward: net currency profit of this step (fees included), plus shaping
        step_net_pnl = float(self.equity - equity_before)
        step_return = step_net_pnl / equity_before if equity_before > 0.0 else 0.0
        reward = step_net_pnl
        shaping = 0.0
        if self.reward_mode == "shaped":
            shaping -= self.drawdown_penalty * new_drawdown  # only when the day's worst drawdown deepens
            if day_closed_pnl is not None and abs(day_closed_pnl) > 0.01:
                shaping += float(np.sign(day_closed_pnl)) * self.day_result_bonus * max(day_base, 0.0)
            reward += shaping
        self.last_action = act_val

        info = self._build_info(
            step_net_pnl=step_net_pnl,
            step_return=step_return,
            closed_trade_pnls=closed_trade_pnls,
            stop_loss_hit=stop_loss_hit,
        )
        info["action"] = float(act_val)
        info["reward_shaping"] = float(shaping)
        info["day_closed_pnl"] = day_closed_pnl
        info["trailing_stop_hit"] = trailing_stop_hit
        info["take_profit_hit"] = take_profit_hit
        info["daily_cap_hit"] = daily_cap_hit
        return self._get_observation(), float(reward), terminated, truncated, info

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
            "trailing_stop_hit": False,
            "take_profit_hit": False,
            "daily_cap_hit": False,
            "locked_out_for_day": bool(self.locked_out_for_day),
            "bars_held": int(self.bars_held),
            "exit_blocked": bool(self.exit_blocked),
            "entry_blocked": bool(self.entry_blocked),
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
        action = 2 if (step_idx // 20) % 2 == 0 else 1  # alternate Long / Flat
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
    print(f"Sum of shaped rewards: {total_reward:,.2f} | net P&L: {step_info['equity'] - env.initial_balance:,.2f}")
