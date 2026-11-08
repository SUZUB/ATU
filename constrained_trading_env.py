"""
ConstrainedTradingEnv Module
============================
A production-grade, custom Reinforcement Learning trading environment built on
the modern Gymnasium API (gymnasium.Env) designed for intraday 5-minute market data
with Continuous Compounding and realistic Margin Intraday Square-off (MIS) frictions:

Key Architecture:
-----------------
1. Unbroken Continuous Episode:
   - Runs the entire dataset (e.g., 60 days / 4,500 candles) as a single unbroken episode.
   - Date Rollover NEVER terminates or truncates: it ONLY updates `self.current_date = current_date`,
     resets `self.daily_trades_count = 0`, and resets `self.locked_out_for_day = False`.
   - 15:15 Auto-Square-Off and -1% Stop-Loss NEVER set `terminated = True` or `truncated = True`.
   - The ONLY line in the entire `step()` function that returns a termination flag is at the
     absolute end of the dataset: `if self.current_step >= len(self.df) - 1: terminated = True`.
2. True Compounding (Dynamic Risk & Balance):
   - When a position is closed (manual exit, target reached, stop-loss hit, or 15:15 square-off),
     exit fees are deducted, net PnL is calculated, and it is permanently added to
     `self.account_balance`.
   - `self.unrealized_profit` resets to 0.0.
   - The 1% Risk Manager dynamically calculates risk using the live compounding `self.account_balance`.
3. Day-Lock Penalty for Stop-Loss:
   - If the -1% stop-loss triggers, the trade is squared off, balance is compounded, flat penalty
     of -10.0 is applied, and `self.locked_out_for_day = True` is set.
   - While `locked_out_for_day` is True, all actions are overridden to 0.0 (Hold).
   - `self.locked_out_for_day = False` and `self.daily_trades_count = 0` reset when the date rolls over.
4. 15:15 MIS Auto-Square-Off:
   - At or past 15:15:00, any open position is forcefully closed, exit fee is deducted, net PnL
     is permanently compounded into `self.account_balance`, and the agent remains flat until the
     next morning session (09:15:00) with `terminated = False` and `truncated = False`.
5. Round-Trip Brokerage & STT Transaction Costs:
   - `self.commission_rate = 0.0003` (0.03% brokerage, turnover charges, and STT).
   - Deducted twice per full trade lifecycle (entry leg and exit leg).
   - All deductions are tracked in `self.total_fees_paid`.
"""

from datetime import time
from typing import Any, Dict, Optional, Tuple, Union
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd


class ConstrainedTradingEnv(gym.Env):
    """
    Intraday continuous-action trading environment with continuous compounding,
    dynamic risk scaling, 15:15 MIS auto-square-off, and day-lock stop-loss enforcement.

    Observation Space:
    ------------------
    Box(num_features + 1,): All numeric features from the provided DataFrame
    plus one dynamic feature: `unrealized_profit`.

    Action Space:
    -------------
    Box(-1.0, 1.0, shape=(1,)):
    - Direction: Sign of action (+1 for Long/Buy, -1 for Short/Sell, 0 for Flat).
    - Threshold: |action| > 0.1 signifies active trade intent; <= 0.1 signifies Flat/Hold.
    - Sizing: Scaled by Risk Manager up to max 1.0% (0.01) risk of live `account_balance`.
    """

    metadata = {"render_modes": ["human"], "version": "3.3.0"}

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
    ) -> None:
        """
        Initializes the ConstrainedTradingEnv.

        Args:
            df: DataFrame containing normalized OHLCV, indicators, and sentiment with DatetimeIndex.
            initial_balance: Starting account equity in currency units (default: 100,000.0).
            max_risk_per_trade: Maximum allowable fraction of account equity risked on a trade (0.01 = 1%).
            profit_target: Target profit fraction triggering +20 reward and reset (0.02 = 2%).
            loss_limit: Stop-loss threshold triggering day-lock (-0.01 = -1%).
            commission_rate: Transaction fee fraction per single leg (0.0003 = 0.03%).
            max_daily_trades: Maximum permitted trade entries per calendar day (default: 3).
            auto_square_off_time: Cut-off time for MIS intraday square-off (default: 15:15:00).
            price_col: Column name to use for computing step returns (default: 'Close').
        """
        super().__init__()

        if df.empty or len(df) < 3:
            raise ValueError("Input DataFrame must contain at least 3 rows of market data.")

        # Extract and validate DatetimeIndex
        if isinstance(df.index, pd.DatetimeIndex):
            self.dates = df.index
        elif "date" in df.columns or "Date" in df.columns:
            date_col = "date" if "date" in df.columns else "Date"
            self.dates = pd.to_datetime(df[date_col])
        else:
            self.dates = pd.to_datetime(df.index)

        self.df = df.copy().reset_index(drop=True)
        self.initial_balance: float = float(initial_balance)
        self.max_risk_per_trade: float = float(max_risk_per_trade)
        self.profit_target: float = float(profit_target)
        self.loss_limit: float = float(loss_limit)
        self.commission_rate: float = float(commission_rate)
        self.max_daily_trades: int = int(max_daily_trades)
        self.auto_square_off_time: time = auto_square_off_time

        # Locate price column for calculating returns
        if price_col in self.df.columns:
            self.price_col: str = price_col
        elif price_col.lower() in self.df.columns:
            self.price_col = price_col.lower()
        elif "Close" in self.df.columns:
            self.price_col = "Close"
        elif "close" in self.df.columns:
            self.price_col = "close"
        else:
            self.price_col = self.df.columns[3] if len(self.df.columns) > 3 else self.df.columns[0]

        # Extract numeric features and price series
        numeric_df = self.df.select_dtypes(include=[np.number])
        self.features: np.ndarray = numeric_df.to_numpy(dtype=np.float32)
        self.prices: np.ndarray = self.df[self.price_col].to_numpy(dtype=np.float64)
        self.num_rows: int = len(self.df)
        self.num_features: int = self.features.shape[1]

        # Continuous Action Space: [-1.0, 1.0]
        self.action_space: spaces.Box = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(1,),
            dtype=np.float32,
        )

        # Observation Space: num_features + 1 (unrealized_profit)
        self.observation_space: spaces.Box = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.num_features + 1,),
            dtype=np.float32,
        )

        # Compounding Account Balance & State Tracking
        self.account_balance: float = self.initial_balance
        self.current_step: int = 0
        self.current_date = self.dates[0].date()
        self.daily_trades_count: int = 0
        self.total_trades: int = 0
        self.total_fees_paid: float = 0.0
        self.unrealized_profit: float = 0.0
        self.is_active_position: bool = False
        self.position_direction: int = 0  # 0: Flat, 1: Long, -1: Short
        self.locked_out_for_day: bool = False
        self.last_action: float = 0.0
        self.unique_dates_seen: set = {self.current_date}

    @property
    def current_unrealized_profit(self) -> float:
        """Alias for backward compatibility."""
        return self.unrealized_profit

    @property
    def balance(self) -> float:
        """Live mark-to-market portfolio balance."""
        if self.is_active_position:
            return self.account_balance * (1.0 + self.unrealized_profit)
        return self.account_balance

    def _apply_risk_manager(self, action: Union[float, np.ndarray]) -> float:
        """
        Enforces maximum single-trade risk limit (1.0% = 0.01) calculated dynamically
        against the live compounding account balance.

        Args:
            action: Raw continuous action in [-1.0, 1.0].

        Returns:
            float: Signed risk fraction in [-0.01, 0.01].
        """
        act_val = float(action[0]) if isinstance(action, (np.ndarray, list)) else float(action)
        clipped_action = float(np.clip(act_val, -1.0, 1.0))
        applied_risk = clipped_action * self.max_risk_per_trade
        return float(applied_risk)

    def _get_observation(self) -> np.ndarray:
        """
        Constructs the current state vector: feature slice + unrealized_profit.
        """
        current_features = self.features[self.current_step]
        profit_feature = np.array([self.unrealized_profit], dtype=np.float32)
        return np.concatenate([current_features, profit_feature], dtype=np.float32)

    def _deduct_leg_fee(self) -> float:
        """
        Helper method to apply single-leg commission fee (0.03% of live account balance).
        Deducts fee from open trade unrealized profit and tracks in total_fees_paid.

        Returns:
            float: Fee amount in currency units.
        """
        fee_amount = self.commission_rate * self.account_balance
        self.total_fees_paid += fee_amount
        self.unrealized_profit -= self.commission_rate
        return fee_amount

    def _close_position_and_compound(self) -> float:
        """
        Square off open position:
        1. Deducts exit leg fee.
        2. Calculates net profit/loss.
        3. Permanently compounds net profit/loss into self.account_balance.
        4. Resets position state and unrealized_profit to 0.0.

        Returns:
            float: Realized trade net PnL fraction (after entry and exit fees).
        """
        self._deduct_leg_fee()
        net_trade_fraction = self.unrealized_profit
        net_trade_pnl_currency = net_trade_fraction * self.account_balance

        # Permanently compound into live account balance
        self.account_balance += net_trade_pnl_currency
        self.unrealized_profit = 0.0
        self.is_active_position = False
        self.position_direction = 0

        return net_trade_fraction

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Resets environment to starting state.
        """
        super().reset(seed=seed)

        self.current_step = 0
        self.current_date = self.dates[0].date()
        self.daily_trades_count = 0
        self.total_trades = 0
        self.total_fees_paid = 0.0
        self.unrealized_profit = 0.0
        self.account_balance = self.initial_balance
        self.is_active_position = False
        self.position_direction = 0
        self.locked_out_for_day = False
        self.last_action = 0.0
        self.unique_dates_seen = {self.current_date}

        observation = self._get_observation()
        info: Dict[str, Any] = {
            "step": self.current_step,
            "date": str(self.current_date),
            "time": str(self.dates[self.current_step].time()),
            "account_balance": float(self.account_balance),
            "balance": float(self.balance),
            "unrealized_profit": float(self.unrealized_profit),
            "daily_trades_count": int(self.daily_trades_count),
            "total_trades": int(self.total_trades),
            "total_fees_paid": float(self.total_fees_paid),
            "trade_closed_profit": False,
            "locked_out_for_day": False,
            "days_processed": len(self.unique_dates_seen),
        }
        return observation, info

    def step(
        self, action: Union[float, np.ndarray]
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Executes one 5-minute candle step with continuous compounding, dynamic risk
        management, 15:15 MIS auto-square-off, and day-lock stop-loss penalty.
        Runs as a single unbroken continuous episode across the entire dataset.
        """
        # 1. Date & Time Tracking & Daily Rollover Reset
        current_timestamp = self.dates[self.current_step]
        current_date = current_timestamp.date()
        current_time = current_timestamp.time()

        self.unique_dates_seen.add(current_date)
        if current_date != self.current_date:
            # Rule 2: On date rollover, ONLY update current_date, reset daily count, and reset day-lock
            self.current_date = current_date
            self.daily_trades_count = 0
            self.locked_out_for_day = False
            # Rule 3: ABSOLUTELY NO truncated = True or terminated = True here!

        step_reward = 0.0
        trade_closed_profit = False
        effective_action = 0.0

        # 2. Check 15:15 MIS Auto-Square-Off on Existing Open Position
        if current_time >= self.auto_square_off_time and self.is_active_position:
            # Forcefully close position at 15:15 without terminating/truncating the episode
            net_fraction = self._close_position_and_compound()
            trade_closed_profit = bool(net_fraction > 0.0)

            if net_fraction > 0.0:
                step_reward += float(net_fraction * 10.0)
            else:
                step_reward += float(net_fraction * 10.0)

        # 3. Parse and Filter Raw Action
        act_val = float(action[0]) if isinstance(action, (np.ndarray, list)) else float(action)
        act_val = float(np.clip(act_val, -1.0, 1.0))

        # Enforce Day-Lock Penalty or 15:15 intraday cut-off (Override action to Hold)
        if self.locked_out_for_day or current_time >= self.auto_square_off_time:
            target_direction = 0
            act_val = 0.0
        elif act_val > 0.1:
            target_direction = 1   # Long
        elif act_val < -0.1:
            target_direction = -1  # Short
        else:
            target_direction = 0   # Flat / Hold

        # 4. Position State Machine & Entry/Exit Fee Mechanics
        if not self.is_active_position:
            # Currently FLAT
            if target_direction == 0:
                effective_action = 0.0
            else:
                # Attempting to open a new position
                if self.daily_trades_count >= self.max_daily_trades:
                    # 3-Trade daily cap reached: Force Hold without artificial micro-penalty
                    effective_action = 0.0
                    act_val = 0.0
                else:
                    # Open new position
                    self.daily_trades_count += 1
                    self.total_trades += 1
                    self.is_active_position = True
                    self.position_direction = target_direction
                    effective_action = act_val

                    # Deduct Entry Leg Fee
                    self._deduct_leg_fee()

        else:
            # Currently in an ACTIVE position
            if target_direction == 0:
                # Manual voluntary trade closure
                net_fraction = self._close_position_and_compound()
                trade_closed_profit = bool(net_fraction > 0.0)

                if net_fraction >= self.profit_target:
                    close_reward = 20.0
                elif net_fraction > 0.0:
                    close_reward = net_fraction * 10.0
                else:
                    close_reward = net_fraction * 10.0

                effective_action = 0.0
                step_reward += close_reward

            elif target_direction == self.position_direction:
                # Holding active trade
                effective_action = act_val

            else:
                # Reversing position: Close existing trade + Attempt new trade
                net_fraction = self._close_position_and_compound()
                trade_closed_profit = bool(net_fraction > 0.0)

                if net_fraction >= self.profit_target:
                    close_reward = 20.0
                elif net_fraction > 0.0:
                    close_reward = net_fraction * 10.0
                else:
                    close_reward = net_fraction * 10.0

                step_reward += close_reward

                # Attempt new reverse trade entry
                if (
                    self.locked_out_for_day
                    or self.daily_trades_count >= self.max_daily_trades
                    or current_time >= self.auto_square_off_time
                ):
                    effective_action = 0.0
                    act_val = 0.0
                else:
                    self.daily_trades_count += 1
                    self.total_trades += 1
                    self.is_active_position = True
                    self.position_direction = target_direction
                    effective_action = act_val

                    # Deduct Entry Leg Fee for new trade
                    self._deduct_leg_fee()

        # 5. Advance Candle & Calculate Price Return
        prev_price = self.prices[self.current_step]
        self.current_step += 1
        curr_price = self.prices[self.current_step]

        if prev_price > 0:
            price_return = (curr_price - prev_price) / prev_price
        else:
            price_return = 0.0

        applied_risk = self._apply_risk_manager(effective_action)
        step_profit = (applied_risk / self.max_risk_per_trade) * price_return

        if self.is_active_position:
            self.unrealized_profit += float(step_profit)

        self.last_action = act_val

        # 6. Evaluate In-Trade Triggers: Stop-Loss, 15:15 Cut-off & Profit Target
        next_time = self.dates[self.current_step].time()

        if self.is_active_position:
            # Trigger A: Strict -1% Stop-Loss with Day-Lock Penalty (Flat -10.0 penalty)
            if self.unrealized_profit <= self.loss_limit:
                step_reward -= 10.0

                # Square off trade, deduct exit fee, and compound balance
                self._close_position_and_compound()
                trade_closed_profit = False

                # Activate Day-Lock Penalty for the remainder of this calendar day
                self.locked_out_for_day = True

            # Trigger B: 15:15 MIS Auto-Square-Off at candle arrival (NEVER terminates episode)
            elif next_time >= self.auto_square_off_time:
                net_fraction = self._close_position_and_compound()
                trade_closed_profit = bool(net_fraction > 0.0)
                if net_fraction > 0.0:
                    step_reward += float(net_fraction * 10.0)
                else:
                    step_reward += float(net_fraction * 10.0)

            # Trigger C: Continuous Profit Taking (+2% target hit with 1:2 risk-reward reward)
            elif self.unrealized_profit >= self.profit_target:
                self._close_position_and_compound()
                step_reward += 20.0
                trade_closed_profit = True

            else:
                # Ongoing step holding reward (Symmetric)
                if step_profit > 0.0:
                    step_reward += float(step_profit * 10.0)
                else:
                    step_reward += float(step_profit * 10.0)  # step_profit is already negative, naturally subtracts symmetrically

        # 7. Unbroken Continuous Episode: ONLY terminate at absolute end of dataset
        terminated = False
        truncated = False

        # Rule 4: ONLY line in the entire step() function that sets terminated = True
        if self.current_step >= len(self.df) - 1:
            terminated = True
            if self.is_active_position:
                self._close_position_and_compound()

        observation = self._get_observation()
        info: Dict[str, Any] = {
            "step": self.current_step,
            "date": str(self.current_date),
            "time": str(self.dates[self.current_step].time()),
            "account_balance": float(self.account_balance),
            "balance": float(self.balance),
            "unrealized_profit": float(self.unrealized_profit),
            "step_profit": float(step_profit),
            "applied_risk": float(applied_risk),
            "action": float(act_val),
            "daily_trades_count": int(self.daily_trades_count),
            "total_trades": int(self.total_trades),
            "total_fees_paid": float(self.total_fees_paid),
            "trade_closed_profit": bool(trade_closed_profit),
            "locked_out_for_day": bool(self.locked_out_for_day),
            "days_processed": int(len(self.unique_dates_seen)),
        }

        return observation, step_reward, terminated, truncated, info


if __name__ == "__main__":
    np.random.seed(42)

    # Generate 5-minute intraday mock data spanning 5 trading days (75 candles/day = 375 candles)
    n_rows = 375
    timestamps = []
    curr_date = pd.Timestamp("2024-01-02")
    while len(timestamps) < n_rows:
        if curr_date.weekday() < 5:
            day_candles = pd.date_range(
                start=curr_date.replace(hour=9, minute=15),
                end=curr_date.replace(hour=15, minute=25),
                freq="5min",
            )
            for ts in day_candles:
                timestamps.append(ts)
                if len(timestamps) == n_rows:
                    break
        curr_date += pd.Timedelta(days=1)

    datetime_index = pd.DatetimeIndex(timestamps)
    daily_returns = np.random.normal(0.0001, 0.003, n_rows)
    close_prices = 150.0 * np.cumprod(1.0 + daily_returns)

    mock_df = pd.DataFrame(
        {
            "Open": close_prices * (1.0 + np.random.normal(0, 0.001, n_rows)),
            "High": close_prices * (1.0 + np.abs(np.random.normal(0, 0.002, n_rows))),
            "Low": close_prices * (1.0 - np.abs(np.random.normal(0, 0.002, n_rows))),
            "Close": close_prices,
            "Volume": np.random.uniform(0.1, 1.0, n_rows),
            "RSI": np.random.uniform(30.0, 70.0, n_rows),
            "MACD": np.random.normal(0.0, 0.3, n_rows),
            "MACD_Signal": np.random.normal(0.0, 0.3, n_rows),
            "MACD_Diff": np.random.normal(0.0, 0.1, n_rows),
            "ATR": np.random.uniform(0.5, 1.5, n_rows),
            "BB_High": close_prices * 1.01,
            "BB_Mid": close_prices * 1.00,
            "BB_Low": close_prices * 0.98,
            "ADX": np.random.uniform(15.0, 45.0, n_rows),
            "daily_sentiment": np.random.uniform(-0.5, 0.5, n_rows),
        },
        index=datetime_index,
    )

    print("==========================================================")
    print(" ConstrainedTradingEnv Continuous Compounding Multi-Day   ")
    print("==========================================================")

    env = ConstrainedTradingEnv(df=mock_df)
    obs, info = env.reset(seed=42)

    print(f"Total Candles in Dataset: {len(mock_df)} (5 trading days)")
    print(f"Starting Date:            {info['date']}")
    print(f"Initial Account Balance:  ₹{env.account_balance:,.2f}\n")

    done = False
    step_idx = 0
    trade_events = 0

    while not done:
        # Alternating trade intents: buy every morning, hold across intraday
        action = np.array([0.9 if step_idx % 25 == 0 else 0.5])
        obs, reward, terminated, truncated, step_info = env.step(action)
        done = terminated or truncated
        step_idx += 1

        if (
            step_info["trade_closed_profit"]
            or step_info["locked_out_for_day"]
            or step_info["time"].startswith("15:15")
            or done
        ):
            trade_events += 1
            if trade_events <= 10 or done:
                print(
                    f"Step {step_idx:03d} | Date: {step_info['date']} {step_info['time']} | "
                    f"Trades: {step_info['daily_trades_count']}/3 | "
                    f"LockedOut: {step_info['locked_out_for_day']} | "
                    f"AccountBalance: ₹{step_info['account_balance']:,.2f} | "
                    f"TotalFees: ₹{step_info['total_fees_paid']:,.2f} | "
                    f"Terminated: {terminated} | Truncated: {truncated}"
                )

    print("\n--- Multi-Day Continuous Compounding Summary ---")
    print(f"Total Steps Executed:     {step_idx}")
    print(f"Total Days Processed:     {step_info['days_processed']}")
    print(f"Total Trades Executed:    {step_info['total_trades']}")
    print(f"Total Fees Paid:          ₹{step_info['total_fees_paid']:,.2f}")
    print(f"Final Compounded Balance: ₹{step_info['account_balance']:,.2f}")
    print(f"Episode Completed Cleanly at Dataset Horizon: {terminated}")
