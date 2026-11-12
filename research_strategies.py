"""
research_strategies.py
======================
Rule-based intraday strategies taken from published research, a multi-stock
backtester with realistic fills and fees, and a feature "edge check".

Strategies (parameters fixed from the papers; nothing is tuned on our test data):

1. Opening Range Breakout (ORB)
   Zarattini & Aziz (2023), "Can Day Trading Really Be Profitable? Evidence of
   Sustainable Long-term Profits from Opening Range Breakout (ORB) Day Trading
   Strategy vs. Benchmark in the US Stock Market", SSRN 4416622.
   - The first bar of the session is the opening range (OR). A green first bar ->
     long at the next bar's open; a red first bar -> short; a doji -> no trade.
   - Stop: the other side of the OR (OR low for longs, OR high for shorts).
   - Exit: profit target at `rr` x risk (default 3 -> 1:3 risk:reward) or the close
     of the session, whichever comes first. (The paper uses 10R, i.e. mostly
     end-of-day exits; rr=None reproduces "end of day only".)
   - Sizing: each trade risks `risk_per_trade` of the capital allocated to it,
     capped at `max_leverage` x that capital.

2. ORB on "Stocks in Play"
   Zarattini, Barbon & Aziz (2024), "A Profitable Day Trading Strategy For The
   U.S. Equity Market", Swiss Finance Institute RP 24-98.
   - Same trade rules, but each day only the `top_n` stocks with the highest
     relative opening volume (first-bar volume / mean first-bar volume of the
     previous 14 sessions) and relative volume >= `min_rel_volume` are traded.

3. Intraday momentum
   Gao, Han, Li & Zhou (2018), "Market intraday momentum", Journal of Financial
   Economics 129(2), 394-414.
   - The return from the previous close to the end of the first bar predicts the
     return of the session's last bar (the last half hour on 1h US data).
     Trade the last bar in that direction, exit at the close.

All trades are opened and closed within the same session (no overnight risk).
Stops and targets are detected on the bar high/low; when one bar touches both,
the stop is assumed to fill first (conservative). Gaps through a level fill at
the bar's open. Fees are charged on both legs.

Data note: yfinance keeps 5-minute bars for only 60 days, so the default is 1-hour
bars over ~2 years (first bar 09:30-10:30 ET = a 60-minute opening range).
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd

from market_data_processor import MarketDataProcessor

DEFAULT_UNIVERSE: List[str] = [
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "AMD", "NFLX", "AVGO", "JPM", "SPY", "QQQ",
]
REL_VOLUME_LOOKBACK: int = 14


# =====================================================================
# Session features (all causal: known at the close of the bar they sit on)
# =====================================================================
def add_session_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds per-bar session features to a raw OHLCV frame (DatetimeIndex in exchange time):
      session, bar_in_day, is_last_bar, prev_close,
      gap_pct            open of the day vs previous close (%), known from bar 0
      or_high / or_low   opening range = first bar of the session
      or_dir             +1 green first bar, -1 red, 0 doji
      first_bar_ret_pct  previous close -> first bar close (%) (Gao et al. signal)
      rel_volume         first-bar volume / mean first-bar volume of prior 14 sessions
      vwap, vwap_dist_pct  session VWAP and close's distance from it (%)
      or_pos             (close - OR mid) / OR range
      day_ret_pct        day open -> close of this bar (%)
    """
    out = df.copy()
    session = pd.Series(out.index.date, index=out.index)
    out["session"] = session
    out["bar_in_day"] = out.groupby("session").cumcount()
    out["is_last_bar"] = session.ne(session.shift(-1)).to_numpy()

    day = out.groupby("session").agg(
        day_open=("Open", "first"), day_close=("Close", "last"),
        or_open=("Open", "first"), or_high=("High", "first"), or_low=("Low", "first"),
        or_close=("Close", "first"), or_volume=("Volume", "first"),
    )
    day["prev_close"] = day["day_close"].shift(1)
    day["gap_pct"] = (day["day_open"] / day["prev_close"] - 1.0) * 100.0
    day["or_dir"] = np.sign(day["or_close"] - day["or_open"])
    day["first_bar_ret_pct"] = (day["or_close"] / day["prev_close"] - 1.0) * 100.0
    prior_mean = day["or_volume"].shift(1).rolling(REL_VOLUME_LOOKBACK, min_periods=5).mean()
    day["rel_volume"] = day["or_volume"] / prior_mean
    out = out.join(day[["day_open", "prev_close", "gap_pct", "or_high", "or_low", "or_dir",
                        "first_bar_ret_pct", "rel_volume"]], on="session")

    typical = (out["High"] + out["Low"] + out["Close"]) / 3.0
    pv = (typical * out["Volume"]).groupby(out["session"]).cumsum()
    vol = out["Volume"].groupby(out["session"]).cumsum()
    out["vwap"] = (pv / vol.replace(0, np.nan)).fillna(out["Close"])
    out["vwap_dist_pct"] = (out["Close"] / out["vwap"] - 1.0) * 100.0
    or_range = (out["or_high"] - out["or_low"]).replace(0, np.nan)
    out["or_pos"] = ((out["Close"] - (out["or_high"] + out["or_low"]) / 2.0) / or_range).fillna(0.0)
    out["day_ret_pct"] = (out["Close"] / out["day_open"] - 1.0) * 100.0
    return out


SESSION_FEATURE_COLUMNS: List[str] = [
    "gap_pct", "first_bar_ret_pct", "rel_volume", "vwap_dist_pct", "or_pos", "day_ret_pct",
]


def session_features_for_state(raw_df: pd.DataFrame) -> pd.DataFrame:
    """
    Causal session features for the RL state vector, aligned to `raw_df`'s index.
    Day-level values that are only known after the first bar closes (first_bar_ret,
    rel_volume) are already known at every bar where they appear (bar 0 onward,
    observed at the bar's close). NaNs (first sessions) become 0; values are clipped.
    """
    feats = add_session_features(raw_df)[SESSION_FEATURE_COLUMNS].copy()
    feats["rel_volume"] = np.log(feats["rel_volume"].clip(lower=1e-3))  # log ratio: 0 = normal volume
    return feats.replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(-10.0, 10.0)


# =====================================================================
# Trade simulation
# =====================================================================
@dataclass
class Trade:
    ticker: str
    session: Any
    strategy: str
    direction: int
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    entry: float
    exit: float
    stop: Optional[float]
    target: Optional[float]
    notional: float
    pnl: float
    fees: float
    r_multiple: float
    exit_reason: str


def _simulate_bracket(
    bars: pd.DataFrame, direction: int, entry: float, stop: Optional[float], target: Optional[float]
) -> tuple:
    """
    Walks the bars after entry. Returns (exit_price, exit_time, reason). The entry bar
    itself is included (entry at its open). Stop before target when both are touched.
    """
    for ts, b in bars.iterrows():
        o, h, l = float(b["Open"]), float(b["High"]), float(b["Low"])
        if stop is not None:
            if direction == 1 and l <= stop:
                return (min(o, stop), ts, "stop")
            if direction == -1 and h >= stop:
                return (max(o, stop), ts, "stop")
        if target is not None:
            if direction == 1 and h >= target:
                return (max(o, target), ts, "target")
            if direction == -1 and l <= target:
                return (min(o, target), ts, "target")
    last = bars.iloc[-1]
    return (float(last["Close"]), bars.index[-1], "close")


def _book(trades: List[Trade], **kw: Any) -> Trade:
    gross = kw["direction"] * kw["notional"] / kw["entry"] * (kw["exit"] - kw["entry"])
    fees = kw.pop("fee_rate") * (kw["notional"] + kw["notional"] / kw["entry"] * kw["exit"])
    risk_cash = kw.pop("risk_cash")
    t = Trade(pnl=gross - fees, fees=fees, r_multiple=(gross - fees) / risk_cash if risk_cash > 0 else 0.0, **kw)
    trades.append(t)
    return t


def orb_trade(
    day_bars: pd.DataFrame, ticker: str, capital: float, risk_per_trade: float, max_leverage: float,
    rr: Optional[float], fee_rate: float,
) -> Optional[dict]:
    """One ORB trade on a single session (or None)."""
    if len(day_bars) < 2:
        return None
    first = day_bars.iloc[0]
    direction = int(np.sign(first["Close"] - first["Open"]))
    if direction == 0:
        return None
    after = day_bars.iloc[1:]
    entry = float(after.iloc[0]["Open"])
    stop = float(first["Low"]) if direction == 1 else float(first["High"])
    risk_per_share = direction * (entry - stop)
    if risk_per_share <= 0:
        return None  # opened beyond the stop
    notional = min(capital * risk_per_trade / (risk_per_share / entry), capital * max_leverage)
    target = entry + direction * rr * risk_per_share if rr is not None else None
    exit_price, exit_time, reason = _simulate_bracket(after, direction, entry, stop, target)
    return dict(ticker=ticker, session=day_bars.index[0].date(), direction=direction,
                entry_time=after.index[0], exit_time=exit_time, entry=entry, exit=exit_price,
                stop=stop, target=target, notional=notional, exit_reason=reason, fee_rate=fee_rate,
                risk_cash=notional * risk_per_share / entry)


def momentum_trade(day_bars: pd.DataFrame, prev_close: float, ticker: str, capital: float,
                   max_leverage: float, fee_rate: float) -> Optional[dict]:
    """Gao et al.: first-bar return (from previous close) -> trade the last bar in that direction."""
    if len(day_bars) < 3 or not np.isfinite(prev_close) or prev_close <= 0:
        return None
    signal = float(day_bars.iloc[0]["Close"]) / prev_close - 1.0
    direction = int(np.sign(signal))
    if direction == 0:
        return None
    last = day_bars.iloc[[-1]]
    entry = float(last.iloc[0]["Open"])
    notional = capital * max_leverage
    exit_price, exit_time, reason = _simulate_bracket(last, direction, entry, None, None)
    return dict(ticker=ticker, session=day_bars.index[0].date(), direction=direction,
                entry_time=last.index[0], exit_time=exit_time, entry=entry, exit=exit_price,
                stop=None, target=None, notional=notional, exit_reason=reason, fee_rate=fee_rate,
                risk_cash=notional * 0.01)  # R measured as 1% of notional (no stop in this strategy)


# =====================================================================
# Portfolio backtest
# =====================================================================
@dataclass
class StrategyResult:
    name: str
    trades: pd.DataFrame
    daily: pd.DataFrame  # one row per session: equity, pnl
    initial_capital: float
    params: Dict[str, Any] = field(default_factory=dict)


def run_strategy(
    data: Dict[str, pd.DataFrame],
    strategy: str = "orb",
    top_n: Optional[int] = None,
    min_rel_volume: float = 1.0,
    rr: Optional[float] = 3.0,
    risk_per_trade: float = 0.01,
    max_leverage: float = 2.0,
    fee_rate: float = 0.0003,
    initial_capital: float = 100_000.0,
    sessions: Optional[Iterable[Any]] = None,
) -> StrategyResult:
    """
    Simulates `strategy` ("orb" or "momentum") over all tickers in `data` (raw OHLCV
    frames). Capital is compounded daily and split equally across the day's trades.
    `top_n` (with `min_rel_volume`) selects "Stocks in Play" each day; None trades every
    ticker. `sessions` restricts the simulation to those dates (e.g. the test period).
    """
    feats = {t: add_session_features(df) for t, df in data.items()}
    by_day: Dict[Any, Dict[str, pd.DataFrame]] = {}
    for t, f in feats.items():
        for d, g in f.groupby("session", sort=True):
            by_day.setdefault(d, {})[t] = g
    days = sorted(by_day)
    if sessions is not None:
        keep = set(sessions)
        days = [d for d in days if d in keep]

    equity = float(initial_capital)
    trades: List[Trade] = []
    rows = []
    for d in days:
        candidates = by_day[d]
        if top_n is not None:
            rv = {t: float(g["rel_volume"].iloc[0]) for t, g in candidates.items()}
            ranked = sorted((t for t in rv if np.isfinite(rv[t]) and rv[t] >= min_rel_volume),
                            key=lambda t: rv[t], reverse=True)
            candidates = {t: candidates[t] for t in ranked[:top_n]}
        specs = []
        for t, g in candidates.items():
            if strategy == "orb":
                spec = orb_trade(g, t, 1.0, risk_per_trade, max_leverage, rr, fee_rate)
            elif strategy == "momentum":
                spec = momentum_trade(g, float(g["prev_close"].iloc[0]), t, 1.0, max_leverage, fee_rate)
            else:
                raise ValueError(f"unknown strategy {strategy!r}")
            if spec is not None:
                specs.append(spec)
        day_pnl = 0.0
        if specs:
            alloc = equity / len(specs)
            for spec in specs:
                spec["notional"] *= alloc
                spec["risk_cash"] *= alloc
                day_pnl += _book(trades, strategy=strategy, **spec).pnl
        equity += day_pnl
        rows.append({"session": d, "pnl": day_pnl, "equity": equity, "n_trades": len(specs)})

    daily = pd.DataFrame(rows).set_index("session") if rows else pd.DataFrame(columns=["pnl", "equity", "n_trades"])
    trade_df = pd.DataFrame([t.__dict__ for t in trades])
    name = {"orb": "ORB", "momentum": "Intraday momentum"}[strategy]
    if strategy == "orb":
        name += f" 1:{rr:g}" if rr is not None else " (exit at close)"
    if top_n is not None:
        name += f" | Stocks in Play top {top_n}"
    return StrategyResult(name=name, trades=trade_df, daily=daily, initial_capital=float(initial_capital),
                          params=dict(strategy=strategy, top_n=top_n, rr=rr, risk_per_trade=risk_per_trade,
                                      max_leverage=max_leverage, fee_rate=fee_rate))


def summarize(result: StrategyResult) -> Dict[str, Any]:
    """Headline statistics from a StrategyResult."""
    daily, trades = result.daily, result.trades
    cap0 = result.initial_capital
    if daily.empty:
        return {"name": result.name, "trades": 0}
    start_equity = daily["equity"] - daily["pnl"]
    day_ret = daily["pnl"] / start_equity
    traded = daily[daily["n_trades"] > 0]
    green = int((traded["pnl"] > 0).sum())
    red = int((traded["pnl"] < 0).sum())
    eq = np.concatenate([[cap0], daily["equity"].to_numpy()])
    peak = np.maximum.accumulate(eq)
    wins = trades.loc[trades["pnl"] > 0, "pnl"] if len(trades) else pd.Series(dtype=float)
    losses = trades.loc[trades["pnl"] < 0, "pnl"] if len(trades) else pd.Series(dtype=float)
    return {
        "name": result.name,
        "sessions": int(len(daily)),
        "trades": int(len(trades)),
        "total_return_pct": (float(daily["equity"].iloc[-1]) / cap0 - 1.0) * 100.0,
        "win_rate_pct": float((trades["pnl"] > 0).mean() * 100.0) if len(trades) else 0.0,
        "avg_r": float(trades["r_multiple"].mean()) if len(trades) else 0.0,
        "profit_factor": float(wins.sum() / -losses.sum()) if len(losses) and losses.sum() < 0 else float("inf"),
        "green_day_rate_pct": float(green / (green + red) * 100.0) if green + red else 0.0,
        "green_days": green,
        "red_days": red,
        "max_drawdown_pct": float(((eq - peak) / peak).min() * 100.0),
        "sharpe": float(day_ret.mean() / day_ret.std() * np.sqrt(252)) if day_ret.std() > 0 else 0.0,
        "fees": float(trades["fees"].sum()) if len(trades) else 0.0,
        "gross_return_pct": (float(daily["equity"].iloc[-1]) + (float(trades["fees"].sum()) if len(trades) else 0.0)
                             - cap0) / cap0 * 100.0,
        "target_exits": int((trades["exit_reason"] == "target").sum()) if len(trades) else 0,
        "stop_exits": int((trades["exit_reason"] == "stop").sum()) if len(trades) else 0,
        "close_exits": int((trades["exit_reason"] == "close").sum()) if len(trades) else 0,
    }


def buy_and_hold_pct(data: Dict[str, pd.DataFrame], sessions: Sequence[Any]) -> float:
    """Equal-weight buy & hold of the universe over the given sessions (close to close)."""
    first, last = min(sessions), max(sessions)
    rets = []
    for df in data.values():
        d = pd.Series(df.index.date, index=df.index)
        part = df[(d >= first) & (d <= last)]
        if len(part) > 1:
            rets.append(float(part["Close"].iloc[-1] / part["Open"].iloc[0] - 1.0))
    return float(np.mean(rets) * 100.0) if rets else 0.0


# =====================================================================
# Edge check: does a feature predict the rest of the day after costs?
# =====================================================================
def edge_check(data: Dict[str, pd.DataFrame], split_session: Any) -> pd.DataFrame:
    """
    For each ticker-day, features known at the end of the first bar vs. two targets:
      rest_of_day_pct  next bar's open -> session close (the ORB holding period)
      last_bar_pct     last bar's open -> close (the momentum holding period)
    Reports Spearman rank correlation (IC) for the train and test periods and the
    average return of trading the feature's sign (net of 2 x fee legs, 0.03% each).
    """
    rows = []
    for t, df in data.items():
        f = add_session_features(df)
        for d, g in f.groupby("session", sort=True):
            if len(g) < 3:
                continue
            first = g.iloc[0]
            rows.append({
                "ticker": t, "session": d,
                "or_dir": float(first["or_dir"]),
                "first_bar_ret_pct": float(first["first_bar_ret_pct"]),
                "gap_pct": float(first["gap_pct"]),
                "vwap_dist_pct": float(first["vwap_dist_pct"]),
                "log_rel_volume": float(np.log(first["rel_volume"])) if first["rel_volume"] > 0 else np.nan,
                "rest_of_day_pct": (float(g["Close"].iloc[-1]) / float(g["Open"].iloc[1]) - 1.0) * 100.0,
                "last_bar_pct": (float(g["Close"].iloc[-1]) / float(g["Open"].iloc[-1]) - 1.0) * 100.0,
            })
    panel = pd.DataFrame(rows).replace([np.inf, -np.inf], np.nan)
    cost_pct = 2 * 0.03
    out = []
    for feat in ["or_dir", "first_bar_ret_pct", "gap_pct", "vwap_dist_pct"]:
        for target in ["rest_of_day_pct", "last_bar_pct"]:
            for period, part in (("train", panel[panel["session"] < split_session]),
                                 ("test", panel[panel["session"] >= split_session])):
                p = part[[feat, target]].dropna()
                if len(p) < 30:
                    continue
                ic = float(p[feat].rank().corr(p[target].rank()))
                signed = np.sign(p[feat]) * p[target]
                out.append({"feature": feat, "target": target, "period": period, "n": len(p), "ic": ic,
                            "t_stat": ic * np.sqrt(max(len(p) - 2, 1)) / np.sqrt(max(1 - ic ** 2, 1e-9)),
                            "sign_trade_gross_pct": float(signed.mean()),
                            "sign_trade_net_pct": float(signed.mean() - cost_pct)})
    return pd.DataFrame(out)


# =====================================================================
# Data
# =====================================================================
def _exchange_tz(ticker: str) -> str:
    return "Asia/Kolkata" if ticker.upper().endswith((".NS", ".BO")) else "America/New_York"


def download_universe(tickers: Sequence[str], period: str = "730d", interval: str = "1h",
                      cache_dir: Optional[str] = None, refresh: bool = True,
                      verbose: bool = True) -> Dict[str, pd.DataFrame]:
    """
    Raw OHLCV per ticker. With `cache_dir`, bars are stored in
    <cache_dir>/ohlcv_<TICKER>_<interval>.csv and every fresh download is MERGED into
    that file, so history accumulates beyond Yahoo's limits (5m bars: 60 days) each
    time this runs. `refresh=False` uses the cache only. Failed tickers are skipped.
    The current (still running) session is dropped so incomplete days never enter a backtest.
    """
    import os
    proc = MarketDataProcessor()
    data: Dict[str, pd.DataFrame] = {}
    for t in tickers:
        path = os.path.join(cache_dir, f"ohlcv_{t}_{interval}.csv") if cache_dir else None
        cached = None
        if path and os.path.exists(path):
            cached = pd.read_csv(path, index_col=0)
            cached.index = pd.to_datetime(cached.index, utc=True).tz_convert(_exchange_tz(t))
        fresh = None
        if refresh or cached is None:
            try:
                fresh = proc.fetch_data(ticker=t, period=period, interval=interval)
            except Exception as e:  # noqa: BLE001
                if verbose:
                    print(f"  {t:<8} download failed ({type(e).__name__}); {'using cache' if cached is not None else 'skipped'}")
        parts = [p for p in (cached, fresh) if p is not None and len(p)]
        if not parts:
            continue
        df = pd.concat(parts)
        df = df[~df.index.duplicated(keep="last")].sort_index()
        now = pd.Timestamp.now(tz=df.index.tz)
        if df.index[-1].date() == now.date() and now.time() < pd.Timestamp("16:30").time():
            df = df[df.index.date != now.date()]  # drop today's incomplete session
        if path:
            os.makedirs(cache_dir, exist_ok=True)
            df.to_csv(path)
        data[t] = df
        if verbose:
            print(f"  {t:<8} {len(df):>6} bars  {df.index[0].date()} -> {df.index[-1].date()}"
                  f"  ({len(set(df.index.date))} sessions)")
    return data


# =====================================================================
# Report
# =====================================================================
def research_report(
    data: Dict[str, pd.DataFrame],
    rr: Optional[float] = 3.0,
    risk_per_trade: float = 0.01,
    max_leverage: float = 4.0,
    fee_rate: float = 0.0001,
) -> tuple:
    """
    ORB backtest over everything in `data` plus robustness views: first vs second half,
    fee sensitivity and per-ticker results. Returns (report_text, StrategyResult).
    """
    sessions = sorted(set().union(*[set(df.index.date) for df in data.values()]))
    kw = dict(rr=rr, risk_per_trade=risk_per_trade, max_leverage=max_leverage)
    main = run_strategy(data, fee_rate=fee_rate, **kw)
    s = summarize(main)
    half = len(sessions) // 2
    lines = [
        "=" * 72,
        f" OPENING RANGE BREAKOUT  {main.name}",
        f" {len(data)} tickers | {len(sessions)} sessions {sessions[0]} -> {sessions[-1]}",
        f" Risk {risk_per_trade:.1%} of allocated capital per trade, leverage cap {max_leverage:g}x, "
        f"fees {fee_rate:.3%} per side",
        "=" * 72,
        f" Net return:        {s['total_return_pct']:+.2f}%   (gross {s['gross_return_pct']:+.2f}%, "
        f"equal-weight buy & hold {buy_and_hold_pct(data, sessions):+.2f}%)",
        f" Trades:            {s['trades']}  win rate {s['win_rate_pct']:.1f}%  avg {s['avg_r']:+.2f}R  "
        f"profit factor {s['profit_factor']:.2f}",
        f" Exits:             target {s['target_exits']} | stop {s['stop_exits']} | close {s['close_exits']}",
        f" Days:              green {s['green_days']} / red {s['red_days']} -> green-day rate "
        f"{s['green_day_rate_pct']:.1f}%",
        f" Max drawdown:      {s['max_drawdown_pct']:.2f}%   Sharpe (daily, ann.) {s['sharpe']:.2f}",
        "-" * 72,
        " Consistency (net return):",
    ]
    for label, part in (("first half", sessions[:half]), ("second half", sessions[half:])):
        p = summarize(run_strategy(data, sessions=part, fee_rate=fee_rate, **kw))
        lines.append(f"   {label:<12} {part[0]} -> {part[-1]}: {p['total_return_pct']:+.2f}%  "
                     f"green days {p['green_day_rate_pct']:.0f}%  sharpe {p['sharpe']:.2f}")
    lines.append(" Fee sensitivity (net return):")
    lines.append("   " + " | ".join(
        f"{f:.3%}/side {summarize(run_strategy(data, fee_rate=f, **kw))['total_return_pct']:+.2f}%"
        for f in (0.00003, 0.0001, 0.0002, 0.0003)))
    lines.append(" Per ticker (net return):")
    per = sorted(((t, summarize(run_strategy({t: df}, fee_rate=fee_rate, **kw))["total_return_pct"])
                  for t, df in data.items()), key=lambda x: x[1])
    lines.append("   " + ", ".join(f"{t} {r:+.1f}%" for t, r in per))
    lines += [
        "-" * 72,
        " Read this as evidence, not a promise: trust it only when BOTH halves are",
        " positive at your real fee level, and keep re-running as 5-min history accumulates.",
        "=" * 72,
    ]
    return "\n".join(lines), main


def main(argv: Optional[Sequence[str]] = None) -> None:
    import argparse
    import os

    parser = argparse.ArgumentParser(description="ATU research strategies: 5-min ORB backtest")
    parser.add_argument("--tickers", nargs="+", default=DEFAULT_UNIVERSE)
    parser.add_argument("--interval", default="5m", help="5m reproduces the paper (first 5-minute candle)")
    parser.add_argument("--period", default="60d")
    parser.add_argument("--rr", type=float, default=3.0, help="Reward:risk target (default 3 -> 1:3); 0 = exit at close")
    parser.add_argument("--risk", type=float, default=0.01, help="Risk per trade, fraction of allocated capital")
    parser.add_argument("--leverage", type=float, default=4.0, help="Max notional / allocated capital")
    parser.add_argument("--fee", type=float, default=0.0001, help="Fee + slippage per side (0.0001 = 0.01%%)")
    parser.add_argument("--workspace", default="ATU_Workspace", help="Bars accumulate in <workspace>/datasets/intraday")
    args = parser.parse_args(argv)

    data = download_universe(args.tickers, period=args.period, interval=args.interval,
                             cache_dir=os.path.join(args.workspace, "datasets", "intraday"))
    text, result = research_report(data, rr=args.rr or None, risk_per_trade=args.risk,
                                   max_leverage=args.leverage, fee_rate=args.fee)
    out = os.path.join(args.workspace, "results", "research")
    os.makedirs(out, exist_ok=True)
    result.trades.to_csv(os.path.join(out, "orb_trades.csv"), index=False)
    result.daily.to_csv(os.path.join(out, "orb_daily.csv"))
    with open(os.path.join(out, "orb_report.txt"), "w", encoding="utf-8") as f:
        f.write(text)
    print(text)
    print(f"Saved to {out}")


if __name__ == "__main__":
    main()
