"""
orb_model.py
============
A trained, persistent model on top of the 5-minute Opening Range Breakout
(research_strategies.py, after Zarattini & Aziz 2023).

What it learns
--------------
One sample per stock per session, taken when the first 5-minute candle closes
(09:35 ET). For each sample we simulate BOTH bracket trades:
    long : entry at the next bar's open, stop at the opening-range LOW
    short: entry at the next bar's open, stop at the opening-range HIGH
    target at `rr` x risk (default 3 -> 1:3), otherwise exit at the session close,
and record the result in R units net of fees (R = the distance to the stop).
Two gradient-boosted regressors learn E[R | features] for long and for short.

Decision rule
-------------
Trade the direction with the higher predicted R if it exceeds `min_edge`
(default +0.10R); otherwise skip that stock today. Plain ORB (direction = colour
of the first candle, always trade) is the baseline the model must beat: live
signals come from the model only if it beat plain ORB out-of-sample in BOTH
halves of the walk-forward test; otherwise they fall back to plain ORB.

Leak guard
----------
`assert_no_lookahead` scrambles all prices after 09:35 and verifies that no
feature changes; training refuses to run if one does.

Honest evaluation
-----------------
Walk-forward: the model is retrained every `step` sessions on past sessions only
and trades the next block it has never seen. Only those out-of-sample trades are
reported, next to plain ORB on exactly the same sessions.

Persistence
-----------
<workspace>/models/orb_model.joblib (latest) plus a dated copy per training run.
Every run reloads it, retrains on ALL accumulated 5-minute history (which keeps
growing on Drive) and saves the new version, so the model improves as data grows.
`todays_signals` uses the saved model on the newest session.
"""

from dataclasses import dataclass, field
from datetime import datetime
import json
import os
from typing import Any, Dict, List, Optional, Sequence

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from research_strategies import _simulate_bracket, download_universe

MODEL_UNIVERSE: List[str] = [
    "SPY", "QQQ", "IWM", "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "AMD", "NFLX",
    "AVGO", "JPM", "BAC", "XOM", "CVX", "UNH", "LLY", "JNJ", "V", "MA", "COST", "WMT", "HD",
    "ORCL", "CRM", "ADBE", "INTC", "QCOM", "MU", "PLTR", "UBER", "BA", "DIS", "NKE", "PYPL",
    "COIN", "SHOP", "SMCI",
]
FEATURES: List[str] = [
    "or_dir", "or_body_pct", "or_range_pct", "or_upper_wick", "or_lower_wick", "log_rel_volume",
    "gap_pct", "prev_day_ret_pct", "prev_day_range_pct", "atr_daily_pct", "mkt_or_dir", "mkt_or_body_pct",
]
MARKET_TICKER = "SPY"
R_CLIP = (-3.0, 3.5)


# =====================================================================
# Dataset
# =====================================================================
def _bracket_r(after: pd.DataFrame, direction: int, stop: float, rr: Optional[float], fee_rate: float,
               risk_per_trade: float, max_leverage: float) -> tuple:
    """Net R and risk fraction (of allocated capital lost at -1R, after the leverage cap) of one bracket."""
    entry = float(after.iloc[0]["Open"])
    rps = direction * (entry - stop)
    if rps <= 0 or not np.isfinite(rps):
        return np.nan, np.nan
    target = entry + direction * rr * rps if rr is not None else None
    exit_price, _, _ = _simulate_bracket(after, direction, entry, stop, target)
    r_net = direction * (exit_price - entry) / rps - fee_rate * (entry + exit_price) / rps
    risk_frac = min(risk_per_trade, max_leverage * rps / entry)
    return float(r_net), float(risk_frac)


def build_dataset(
    data: Dict[str, pd.DataFrame],
    rr: Optional[float] = 3.0,
    fee_rate: float = 0.0001,
    risk_per_trade: float = 0.01,
    max_leverage: float = 4.0,
    include_labels: bool = True,
) -> pd.DataFrame:
    """One row per (ticker, session): features at the first candle's close + long/short outcomes."""
    rows: List[Dict[str, Any]] = []
    for ticker, df in data.items():
        sessions = pd.Series(df.index.date, index=df.index)
        daily = df.groupby(sessions).agg(o=("Open", "first"), h=("High", "max"), l=("Low", "min"),
                                         c=("Close", "last"), v0=("Volume", "first"))
        daily["prev_c"] = daily["c"].shift(1)
        # Everything below must be known at 09:35 of day d: only PREVIOUS sessions' closes,
        # highs and lows (shift(1)); today's full-day values would leak the outcome.
        daily["prev_ret"] = ((daily["c"] / daily["c"].shift(1) - 1.0) * 100.0).shift(1)
        daily["range_pct"] = (daily["h"] - daily["l"]) / daily["c"] * 100.0
        daily["prev_range_pct"] = daily["range_pct"].shift(1)
        daily["atr_pct"] = daily["range_pct"].shift(1).rolling(10, min_periods=3).mean()
        daily["rel_vol"] = daily["v0"] / daily["v0"].shift(1).rolling(14, min_periods=5).mean()
        for d, g in df.groupby(sessions, sort=True):
            if len(g) < 2:
                continue
            first = g.iloc[0]
            o, h, l, c = (float(first[k]) for k in ("Open", "High", "Low", "Close"))
            rng = max(h - l, 1e-9)
            day = daily.loc[d]
            row = {
                "ticker": ticker, "session": d,
                "or_dir": float(np.sign(c - o)),
                "or_body_pct": (c / o - 1.0) * 100.0,
                "or_range_pct": rng / o * 100.0,
                "or_upper_wick": (h - max(o, c)) / rng,
                "or_lower_wick": (min(o, c) - l) / rng,
                "log_rel_volume": float(np.log(day["rel_vol"])) if day["rel_vol"] > 0 else np.nan,
                "gap_pct": (o / day["prev_c"] - 1.0) * 100.0 if day["prev_c"] > 0 else np.nan,
                "prev_day_ret_pct": day["prev_ret"],
                "prev_day_range_pct": day["prev_range_pct"],
                "atr_daily_pct": day["atr_pct"],
                "entry_time": g.index[1] if len(g) > 1 else pd.NaT,
            }
            if include_labels:
                after = g.iloc[1:]
                row["r_long"], row["risk_long"] = _bracket_r(after, 1, l, rr, fee_rate, risk_per_trade, max_leverage)
                row["r_short"], row["risk_short"] = _bracket_r(after, -1, h, rr, fee_rate, risk_per_trade, max_leverage)
            rows.append(row)
    ds = pd.DataFrame(rows)
    if ds.empty:
        return ds
    mkt = ds[ds["ticker"] == MARKET_TICKER].set_index("session")[["or_dir", "or_body_pct"]]
    ds["mkt_or_dir"] = ds["session"].map(mkt["or_dir"]) if len(mkt) else np.nan
    ds["mkt_or_body_pct"] = ds["session"].map(mkt["or_body_pct"]) if len(mkt) else np.nan
    return ds.replace([np.inf, -np.inf], np.nan).sort_values(["session", "ticker"]).reset_index(drop=True)


def assert_no_lookahead(data: Dict[str, pd.DataFrame], n_checks: int = 5, seed: int = 0) -> None:
    """
    Leak guard: for random (ticker, session) pairs, scramble every bar AFTER the first
    5-minute candle of that session (and all later sessions) and check that the
    session's features do not change. Raises AssertionError if any feature peeks ahead.
    """
    rng = np.random.default_rng(seed)
    tickers = [t for t in data if t != MARKET_TICKER and len(data[t]) > 200]
    for _ in range(n_checks):
        t = tickers[int(rng.integers(len(tickers)))]
        df = data[t]
        sessions = sorted(set(df.index.date))
        d = sessions[int(rng.integers(len(sessions) // 2, len(sessions)))]
        cut = df.index[df.index.date == d][0]  # first candle of day d
        scrambled = df.copy()
        future = scrambled.index > cut
        scrambled.loc[future, ["Open", "High", "Low", "Close"]] *= rng.uniform(0.8, 1.2, (int(future.sum()), 1))
        scrambled.loc[future, "Volume"] *= rng.uniform(0.5, 2.0, int(future.sum()))
        base = {t: df, MARKET_TICKER: data[MARKET_TICKER]} if MARKET_TICKER in data else {t: df}
        alt = {**base, t: scrambled}
        a = build_dataset(base, include_labels=False)
        b = build_dataset(alt, include_labels=False)
        fa = a[(a.ticker == t) & (a.session == d)][FEATURES].to_numpy(dtype=float)
        fb = b[(b.ticker == t) & (b.session == d)][FEATURES].to_numpy(dtype=float)
        assert np.allclose(fa, fb, equal_nan=True), f"look-ahead leak in features for {t} {d}"


# =====================================================================
# Model
# =====================================================================
def _regressor(seed: int = 0) -> HistGradientBoostingRegressor:
    # Small, heavily regularized trees: the dataset is small and noisy.
    return HistGradientBoostingRegressor(
        max_depth=3, learning_rate=0.05, max_iter=150, min_samples_leaf=50,
        l2_regularization=1.0, random_state=seed,
    )


@dataclass
class ORBModel:
    long_model: Any
    short_model: Any
    features: List[str]
    params: Dict[str, Any]
    trained_sessions: List[str] = field(default_factory=list)
    n_samples: int = 0
    trained_at: str = ""
    evaluation: Dict[str, Any] = field(default_factory=dict)

    def predict(self, ds: pd.DataFrame) -> pd.DataFrame:
        """Adds pred_long / pred_short (expected net R) and the decision (+1, -1, 0)."""
        X = ds[self.features].to_numpy(dtype=float)
        out = ds.copy()
        out["pred_long"] = self.long_model.predict(X)
        out["pred_short"] = self.short_model.predict(X)
        best = np.where(out["pred_long"] >= out["pred_short"], 1, -1)
        best_r = np.maximum(out["pred_long"], out["pred_short"])
        out["decision"] = np.where(best_r > self.params["min_edge"], best, 0)
        return out


def fit_model(ds: pd.DataFrame, params: Dict[str, Any], seed: int = 0) -> ORBModel:
    """Fits the long and short regressors on rows with valid labels."""
    models = {}
    for side in ("long", "short"):
        part = ds.dropna(subset=[f"r_{side}"])
        y = part[f"r_{side}"].clip(*R_CLIP).to_numpy()
        m = _regressor(seed)
        m.fit(part[FEATURES].to_numpy(dtype=float), y)
        models[side] = m
    return ORBModel(models["long"], models["short"], list(FEATURES), dict(params),
                    trained_sessions=sorted(str(s) for s in ds["session"].unique()),
                    n_samples=int(len(ds)), trained_at=datetime.now().isoformat(timespec="seconds"))


def _portfolio(decided: pd.DataFrame, initial_capital: float = 100_000.0) -> pd.DataFrame:
    """Daily compounding: equity split equally across the day's trades; pnl = alloc x risk x R."""
    equity, rows = initial_capital, []
    for d, g in decided.groupby("session", sort=True):
        trades = g[g["direction"] != 0]
        pnl = 0.0
        if len(trades):
            alloc = equity / len(trades)
            r = np.where(trades["direction"] == 1, trades["r_long"], trades["r_short"])
            risk = np.where(trades["direction"] == 1, trades["risk_long"], trades["risk_short"])
            pnl = float(np.nansum(alloc * risk * r))
        equity += pnl
        rows.append({"session": d, "pnl": pnl, "equity": equity, "n_trades": int(len(trades))})
    return pd.DataFrame(rows).set_index("session")


def _stats(daily: pd.DataFrame, decided: pd.DataFrame, initial_capital: float = 100_000.0) -> Dict[str, Any]:
    trades = decided[decided["direction"] != 0]
    r = np.where(trades["direction"] == 1, trades["r_long"], trades["r_short"]).astype(float)
    traded = daily[daily["n_trades"] > 0]
    green, red = int((traded["pnl"] > 0).sum()), int((traded["pnl"] < 0).sum())
    ret = daily["pnl"] / (daily["equity"] - daily["pnl"])
    eq = np.concatenate([[initial_capital], daily["equity"].to_numpy()])
    return {
        "net_return_pct": (float(daily["equity"].iloc[-1]) / initial_capital - 1.0) * 100.0 if len(daily) else 0.0,
        "trades": int(len(trades)),
        "win_rate_pct": float(np.mean(r > 0) * 100.0) if len(r) else 0.0,
        "avg_r": float(np.nanmean(r)) if len(r) else 0.0,
        "green_days": green, "red_days": red,
        "green_day_rate_pct": float(green / (green + red) * 100.0) if green + red else 0.0,
        "max_drawdown_pct": float(((eq - np.maximum.accumulate(eq)) / np.maximum.accumulate(eq)).min() * 100.0),
        "sharpe": float(ret.mean() / ret.std() * np.sqrt(252)) if len(ret) > 1 and ret.std() > 0 else 0.0,
    }


def walk_forward(ds: pd.DataFrame, params: Dict[str, Any], min_train_sessions: int = 20, step: int = 5,
                 seed: int = 0) -> Dict[str, Any]:
    """Out-of-sample evaluation: retrain every `step` sessions on past data only."""
    sessions = sorted(ds["session"].unique())
    if len(sessions) <= min_train_sessions:
        return {"error": f"need more than {min_train_sessions} sessions, have {len(sessions)}"}
    parts = []
    for start in range(min_train_sessions, len(sessions), step):
        block = sessions[start:start + step]
        model = fit_model(ds[ds["session"] < block[0]], params, seed)
        parts.append(model.predict(ds[ds["session"].isin(block)]))
    oos = pd.concat(parts)
    oos_sessions = sorted(oos["session"].unique())
    model_dec = oos.assign(direction=oos["decision"])
    base_dec = oos.assign(direction=oos["or_dir"].astype(int))
    model_daily, base_daily = _portfolio(model_dec), _portfolio(base_dec)
    half = len(oos_sessions) // 2
    halves = {}
    for name, part in (("first_half", oos_sessions[:half]), ("second_half", oos_sessions[half:])):
        md = model_dec[model_dec["session"].isin(part)]
        bd = base_dec[base_dec["session"].isin(part)]
        halves[name] = {"model": _stats(_portfolio(md), md)["net_return_pct"],
                        "baseline": _stats(_portfolio(bd), bd)["net_return_pct"],
                        "sessions": f"{part[0]} -> {part[-1]}"}
    return {
        "oos_sessions": f"{oos_sessions[0]} -> {oos_sessions[-1]} ({len(oos_sessions)})",
        "model": _stats(model_daily, model_dec),
        "baseline_orb": _stats(base_daily, base_dec),
        "halves": halves,
        "model_daily": model_daily,
        "baseline_daily": base_daily,
    }


# =====================================================================
# Persistence + workflow
# =====================================================================
def model_paths(workspace: str) -> Dict[str, str]:
    d = os.path.join(workspace, "models")
    return {"dir": d, "latest": os.path.join(d, "orb_model.joblib"), "meta": os.path.join(d, "orb_model.json")}


def save_model(model: ORBModel, workspace: str) -> str:
    p = model_paths(workspace)
    os.makedirs(p["dir"], exist_ok=True)
    # Store a plain dict (not the dataclass) so the file loads from any entry point
    # (CLI __main__, notebook, other scripts) regardless of where ORBModel was defined.
    payload = {"format": 1, **model.__dict__}
    joblib.dump(payload, p["latest"])
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    joblib.dump(payload, os.path.join(p["dir"], f"orb_model_{stamp}.joblib"))
    meta = {k: v for k, v in model.__dict__.items() if k not in ("long_model", "short_model")}
    meta["trained_sessions"] = f"{model.trained_sessions[0]} -> {model.trained_sessions[-1]} ({len(model.trained_sessions)})"
    with open(p["meta"], "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, default=str)
    return p["latest"]


def load_model(workspace: str) -> Optional[ORBModel]:
    p = model_paths(workspace)["latest"]
    if not os.path.exists(p):
        return None
    try:
        payload = joblib.load(p)
        if isinstance(payload, ORBModel):
            return payload
        payload = dict(payload)
        payload.pop("format", None)
        return ORBModel(**payload)
    except Exception as e:  # noqa: BLE001
        print(f"Could not load {p} ({type(e).__name__}: {e}); a new model will be trained.")
        return None


def format_evaluation(ev: Dict[str, Any], model: ORBModel) -> str:
    if "error" in ev:
        return f"Walk-forward skipped: {ev['error']}"
    m, b = ev["model"], ev["baseline_orb"]
    row = lambda name, s: (f" {name:<14} {s['net_return_pct']:+7.2f}%  {s['trades']:>5}  {s['win_rate_pct']:5.1f}%  "
                           f"{s['avg_r']:+.2f}R  {s['green_days']:>3}/{s['green_days'] + s['red_days']:<3} "
                           f"({s['green_day_rate_pct']:3.0f}%)  {s['max_drawdown_pct']:6.2f}%  {s['sharpe']:5.2f}")
    h = ev["halves"]
    return "\n".join([
        "=" * 78,
        " ORB MODEL - walk-forward, OUT-OF-SAMPLE only (retrained every 5 sessions on the past)",
        f" Test sessions: {ev['oos_sessions']} | fees {model.params['fee_rate']:.3%}/side | RR 1:{model.params['rr']:g}"
        f" | min edge {model.params['min_edge']:+.2f}R",
        "=" * 78,
        " Strategy           Net  Trades    Win   Avg R  Green days        MaxDD  Sharpe",
        row("Model", m),
        row("Plain ORB", b),
        "-" * 78,
        f" First half  ({h['first_half']['sessions']}):  model {h['first_half']['model']:+.2f}% | "
        f"plain ORB {h['first_half']['baseline']:+.2f}%",
        f" Second half ({h['second_half']['sessions']}): model {h['second_half']['model']:+.2f}% | "
        f"plain ORB {h['second_half']['baseline']:+.2f}%",
        " The model is only useful if it beats plain ORB in BOTH halves, and keeps doing so",
        " as more 5-minute history accumulates.",
        "=" * 78,
    ])


def train_and_save(
    workspace: str,
    tickers: Sequence[str] = MODEL_UNIVERSE,
    rr: Optional[float] = 3.0,
    fee_rate: float = 0.0001,
    risk_per_trade: float = 0.01,
    max_leverage: float = 4.0,
    min_edge: float = 0.10,
    refresh_data: bool = True,
) -> Dict[str, Any]:
    """
    Loads/extends the 5-minute history on disk, evaluates walk-forward, retrains on ALL
    history, saves the model (latest + dated copy) and returns everything.
    """
    previous = load_model(workspace)
    if previous is not None:
        print(f"Loaded saved model: trained {previous.trained_at} on {len(previous.trained_sessions)} sessions "
              f"({previous.n_samples} samples).")
    data = download_universe(tickers, period="60d", interval="5m",
                             cache_dir=os.path.join(workspace, "datasets", "intraday"),
                             refresh=refresh_data, verbose=False)
    print(f"5-minute history: {len(data)} tickers, "
          f"{len(set().union(*[set(df.index.date) for df in data.values()]))} sessions")
    assert_no_lookahead(data)  # refuses to train on features that peek into the future
    params = dict(rr=rr, fee_rate=fee_rate, risk_per_trade=risk_per_trade, max_leverage=max_leverage, min_edge=min_edge)
    ds = build_dataset(data, rr=rr, fee_rate=fee_rate, risk_per_trade=risk_per_trade, max_leverage=max_leverage)
    ev = walk_forward(ds, params)
    model = fit_model(ds, params)
    model.evaluation = {k: v for k, v in ev.items() if not k.endswith("_daily")}
    # Use the model's filter only if it beat plain ORB out-of-sample in BOTH halves.
    halves = ev.get("halves", {})
    model.params["use_model"] = bool(halves) and all(h["model"] > h["baseline"] for h in halves.values())
    print("Live signal source: " + ("MODEL (beat plain ORB in both halves)" if model.params["use_model"]
                                    else "PLAIN ORB (model has not beaten it in both halves yet)"))
    path = save_model(model, workspace)
    report = format_evaluation(ev, model)
    new_sessions = (len(model.trained_sessions) - len(previous.trained_sessions)) if previous else len(model.trained_sessions)
    print(report)
    print(f"Saved model -> {path}  ({model.n_samples} samples, {len(model.trained_sessions)} sessions, "
          f"{new_sessions:+d} sessions vs. previous model)")
    return {"model": model, "evaluation": ev, "report": report, "data": data, "dataset": ds}


def todays_signals(model: ORBModel, data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Applies the saved model to the newest session in `data` that has its first 5-minute
    candle. Returns ticker, direction (LONG/SHORT/SKIP), predicted R, entry reference,
    stop (opening-range low/high) and the 1:RR target from the first candle.
    """
    latest: Dict[str, pd.DataFrame] = {}
    last_day = max(df.index[-1].date() for df in data.values())
    for t, df in data.items():
        part = df[df.index.date >= (pd.Timestamp(last_day) - pd.Timedelta(days=40)).date()]
        if len(part) and part.index[-1].date() == last_day:
            latest[t] = part
    ds = build_dataset(latest, include_labels=False)
    ds = ds[ds["session"] == last_day]
    if ds.empty:
        return pd.DataFrame()
    pred = model.predict(ds)
    rr = model.params.get("rr") or 0.0
    use_model = bool(model.params.get("use_model", False))
    out = []
    for _, r in pred.iterrows():
        first = latest[r["ticker"]]
        first = first[first.index.date == last_day].iloc[0]
        d = int(r["decision"]) if use_model else int(r["or_dir"])
        ref = float(first["Close"])
        stop = float(first["Low"]) if d == 1 else float(first["High"]) if d == -1 else np.nan
        risk = abs(ref - stop) if d else np.nan
        out.append({"ticker": r["ticker"], "session": last_day,
                    "source": "model" if use_model else "plain ORB",
                    "signal": {1: "LONG", -1: "SHORT", 0: "SKIP"}[d],
                    # model's expected net R for the direction actually signalled
                    "pred_r": float(r["pred_long"] if d == 1 else r["pred_short"] if d == -1
                                    else max(r["pred_long"], r["pred_short"])),
                    "ref_price": ref, "stop": stop,
                    "target": ref + d * rr * risk if d and rr else np.nan})
    return pd.DataFrame(out).sort_values("pred_r", ascending=False).reset_index(drop=True)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Train / update the persistent ORB model")
    ap.add_argument("--workspace", default="ATU_Workspace")
    ap.add_argument("--fee", type=float, default=0.0001)
    ap.add_argument("--min-edge", type=float, default=0.10)
    ap.add_argument("--no-refresh", dest="refresh", action="store_false", help="Use cached bars only")
    a = ap.parse_args()
    res = train_and_save(a.workspace, fee_rate=a.fee, min_edge=a.min_edge, refresh_data=a.refresh)
    sig = todays_signals(res["model"], res["data"])
    if len(sig):
        print("\nSignals for the latest session:\n" + sig.to_string(index=False))
