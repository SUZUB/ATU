"""
orb_model.py  (ORB v2)
======================
A trained, persistent model on top of the 5-minute Opening Range Breakout
(Zarattini & Aziz 2023, SSRN 4416622), built to be statistically honest.

Samples and labels
------------------
One sample per stock per session, taken when the first 5-minute candle closes.
For each sample BOTH bracket trades are simulated with identical fills to
research_strategies._simulate_bracket (vectorized with numpy):
    entry  = open of the bar after the first candle
    stop   = first candle's low (long) / high (short); checked BEFORE the target on
             the same bar; a gap through a level fills at that bar's open
    target = entry +/- rr x risk (default rr=3 -> 1:3), else exit at the session close
Stored per side: gross R and cost_unit = (entry + exit) / risk_per_share, so
net R = gross R - fee * cost_unit (`apply_fee`) at any fee level, for free.
Sessions whose first bar is not the ticker's usual open time, or whose second bar
comes more than 1.5 bar lengths later (missing prints), are skipped.

Variants (all evaluated walk-forward on identical out-of-sample sessions)
------------------------------------------------------------------------
    plain     first candle's colour, every stock (the paper's rule)
    sip       plain, only "Stocks in Play": top_n by relative opening volume that
              session (default 10% of the universe, fixed a priori) with rel. volume >= 1
    free      two regressors predict net R of LONG and SHORT; trade the better side
              if > min_edge
    meta      meta-labelling: one regressor predicts net R of the PLAIN trade; take
              it only if > min_edge
    meta_sip  meta AND the Stocks-in-Play filter

Statistics per variant (orb_eval.py)
------------------------------------
net return, trades, win rate, avg R, green-day rate, max drawdown, annualized Sharpe,
block-bootstrap CI of the mean daily return (and p(mean <= 0)), Deflated Sharpe
(n_trials = 5 variants), break-even fee, random-direction p-value, first / second
half returns, and net return / avg R at 1, 2, 3, 5 bp per side.
Best = highest Sharpe among variants with >= 30 trades. Verdict:
    SUPPORTED  best DSR >= 0.95 AND profitable at 3 bp AND both halves positive
    WEAK       avg R > 0 AND bootstrap p(mean <= 0) <= 0.10
    NO EDGE    otherwise

Leak guard: `assert_no_lookahead` scrambles all prices after the first candle and
requires identical features. Labels only use the same session -> no purging needed.

Persistence: <workspace>/models/orb_model.joblib (+ dated copies, orb_model.json),
stored as a plain dict; payloads from older versions (no meta_model) still load.
"""

from dataclasses import dataclass, field
from datetime import datetime
import json
import os
from typing import Any, Callable, Dict, List, Optional, Sequence

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from orb_eval import block_bootstrap_ci, breakeven_fee, deflated_sharpe, random_direction_test
from research_strategies import download_universe, split_incomplete_session

MODEL_UNIVERSE: List[str] = [
    "SPY", "QQQ", "IWM", "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "AMD", "NFLX",
    "AVGO", "JPM", "BAC", "XOM", "CVX", "UNH", "LLY", "JNJ", "V", "MA", "COST", "WMT", "HD",
    "ORCL", "CRM", "ADBE", "INTC", "QCOM", "MU", "PLTR", "UBER", "BA", "DIS", "NKE", "PYPL",
    "COIN", "SHOP", "SMCI",
]
LARGE_UNIVERSE: List[str] = MODEL_UNIVERSE + [
    "DIA", "XLF", "XLE", "XLK", "SMH", "GLD", "TLT", "ARKK",
    "ABBV", "MRK", "PFE", "TMO", "ABT", "DHR", "AMGN", "GILD", "ISRG", "VRTX", "REGN", "BMY", "CVS", "MRNA",
    "GS", "MS", "WFC", "C", "SCHW", "AXP", "BLK",
    "KO", "PEP", "PG", "MCD", "SBUX", "TGT", "LOW", "CMCSA", "TMUS", "VZ", "T",
    "CAT", "DE", "GE", "HON", "UPS", "FDX", "LMT", "RTX", "UNP",
    "TXN", "AMAT", "LRCX", "KLAC", "ADI", "MRVL", "NOW", "INTU", "PANW", "CRWD", "SNOW", "DDOG", "NET",
    "ABNB", "DASH", "ROKU", "F", "GM", "RIVN", "SOFI", "HOOD", "MSTR", "DKNG", "OXY", "COP", "SLB",
]
FEATURES: List[str] = [
    "or_dir", "or_body_pct", "or_range_pct", "or_upper_wick", "or_lower_wick", "log_rel_volume",
    "gap_pct", "prev_day_ret_pct", "prev_day_range_pct", "atr_daily_pct", "mkt_or_dir", "mkt_or_body_pct",
    "rv_pct",
]
VARIANTS = ("plain", "sip", "free", "meta", "meta_sip")
EVAL_FEES_BP = (1, 2, 3, 5)
DEFAULT_FEE = 0.0002  # 2 bp per side: commissions + spread + slippage for liquid US stocks
MARKET_TICKER = "SPY"
R_CLIP = (-3.0, 3.5)
MIN_TRADES_FOR_BEST = 30


def apply_fee(gross_r, cost_unit, fee_rate: float):
    """Net R = gross R - fee per side x (entry + exit) / risk_per_share."""
    return np.asarray(gross_r, dtype=float) - float(fee_rate) * np.asarray(cost_unit, dtype=float)


# =====================================================================
# Vectorized session grids and bracket simulation
# =====================================================================
def _session_grid(df: pd.DataFrame) -> Dict[str, Any]:
    """Reshapes one ticker's bars into (session x bar) arrays padded with NaN."""
    idx = df.index
    codes, uniques = pd.factorize(idx.normalize())
    pos = pd.Series(codes).groupby(codes).cumcount().to_numpy()
    S, M = len(uniques), max(int(pos.max()) + 1, 2)

    def grid(values: np.ndarray) -> np.ndarray:
        g = np.full((S, M), np.nan)
        g[codes, pos] = values
        return g

    minutes = (idx.hour * 60 + idx.minute + idx.second / 60.0).to_numpy(dtype=float)
    return {
        "O": grid(df["Open"].to_numpy(float)), "H": grid(df["High"].to_numpy(float)),
        "L": grid(df["Low"].to_numpy(float)), "C": grid(df["Close"].to_numpy(float)),
        "V": grid(df["Volume"].to_numpy(float)), "T": grid(minutes),
        "n": np.bincount(codes, minlength=S), "sessions": [u.date() for u in uniques],
    }


def bracket_numpy(O: np.ndarray, H: np.ndarray, L: np.ndarray, C: np.ndarray, n_bars: np.ndarray,
                  direction: int, stop: np.ndarray, rr: Optional[float]) -> tuple:
    """
    Bracket trade per session row, entering at the open of column 1. Returns
    (entry, exit_price, risk_per_share, valid). Fills match _simulate_bracket exactly:
    first bar touching stop or target decides, stop wins ties, gaps fill at the open,
    otherwise the session's last close.
    """
    S, M = O.shape
    rows = np.arange(S)
    entry = O[:, 1]
    with np.errstate(invalid="ignore"):
        rps = direction * (entry - stop)
        valid = np.isfinite(rps) & (rps > 0) & (n_bars >= 2)
        O1, H1, L1 = O[:, 1:], H[:, 1:], L[:, 1:]
        s_hit = (L1 <= stop[:, None]) if direction == 1 else (H1 >= stop[:, None])
        if rr is not None:
            target = entry + direction * rr * rps
            t_hit = (H1 >= target[:, None]) if direction == 1 else (L1 <= target[:, None])
        else:
            target = np.full(S, np.nan)
            t_hit = np.zeros_like(s_hit)
    big = M + 1
    s_idx = np.where(s_hit.any(axis=1), s_hit.argmax(axis=1), big)
    t_idx = np.where(t_hit.any(axis=1), t_hit.argmax(axis=1), big)
    is_stop = (s_idx < big) & (s_idx <= t_idx)
    is_tgt = ~is_stop & (t_idx < big)
    o_s = O1[rows, np.clip(s_idx, 0, M - 2)]
    o_t = O1[rows, np.clip(t_idx, 0, M - 2)]
    if direction == 1:
        stop_fill, tgt_fill = np.minimum(o_s, stop), np.maximum(o_t, target)
    else:
        stop_fill, tgt_fill = np.maximum(o_s, stop), np.minimum(o_t, target)
    close_px = C[rows, np.clip(n_bars - 1, 0, M - 1)]
    exit_price = np.where(is_stop, stop_fill, np.where(is_tgt, tgt_fill, close_px))
    return entry, exit_price, rps, valid


# =====================================================================
# Dataset
# =====================================================================
def _shift(x: np.ndarray, k: int = 1) -> np.ndarray:
    out = np.full_like(x, np.nan, dtype=float)
    if k < len(x):
        out[k:] = x[:-k]
    return out


def _ticker_rows(ticker: str, df: pd.DataFrame, rr: Optional[float], risk_per_trade: float,
                 max_leverage: float, include_labels: bool, live: bool) -> Optional[pd.DataFrame]:
    if df is None or len(df) == 0:
        return None
    g = _session_grid(df.sort_index())
    O, H, L, C, V, T, n = g["O"], g["H"], g["L"], g["C"], g["V"], g["T"], g["n"]
    S = len(g["sessions"])
    rows = np.arange(S)

    # Data-quality filter: usual open time, no missing second bar
    first_t = T[:, 0]
    vals, counts = np.unique(first_t[np.isfinite(first_t)], return_counts=True)
    usual_open = vals[np.argmax(counts)] if len(vals) else np.nan
    diffs = np.diff(T, axis=1)
    pos_diffs = diffs[np.isfinite(diffs) & (diffs > 0)]
    bar_len = float(np.median(pos_diffs)) if len(pos_diffs) else 5.0
    ok_open = first_t == usual_open
    with np.errstate(invalid="ignore"):
        ok_second = (n >= 2) & ((T[:, 1] - T[:, 0]) <= 1.5 * bar_len)
    sample_ok = ok_open & ok_second
    if live and S:
        sample_ok[-1] = ok_open[-1]  # live: the first candle alone is enough

    # Daily context (only PREVIOUS sessions are used for today's features)
    day_close = C[rows, np.clip(n - 1, 0, C.shape[1] - 1)]
    with np.errstate(invalid="ignore", divide="ignore"):
        day_high, day_low = np.nanmax(H, axis=1), np.nanmin(L, axis=1)
        prev_c = _shift(day_close)
        prev_ret = _shift((day_close / prev_c - 1.0) * 100.0)
        range_pct = (day_high - day_low) / day_close * 100.0
        prev_range = _shift(range_pct)
        atr = pd.Series(prev_range).rolling(10, min_periods=3).mean().to_numpy()
        v0 = V[:, 0]
        rel_vol = v0 / pd.Series(_shift(v0)).rolling(14, min_periods=5).mean().to_numpy()
        o, h, l, c = O[:, 0], H[:, 0], L[:, 0], C[:, 0]
        rng = np.maximum(h - l, 1e-9)
        feats = pd.DataFrame({
            "ticker": ticker, "session": g["sessions"],
            "or_dir": np.sign(c - o),
            "or_body_pct": (c / o - 1.0) * 100.0,
            "or_range_pct": rng / o * 100.0,
            "or_upper_wick": (h - np.maximum(o, c)) / rng,
            "or_lower_wick": (np.minimum(o, c) - l) / rng,
            "log_rel_volume": np.where(rel_vol > 0, np.log(np.where(rel_vol > 0, rel_vol, 1.0)), np.nan),
            "gap_pct": np.where(prev_c > 0, (o / prev_c - 1.0) * 100.0, np.nan),
            "prev_day_ret_pct": prev_ret,
            "prev_day_range_pct": prev_range,
            "atr_daily_pct": atr,
            "or_high": h, "or_low": l, "or_close": c,
        })
    if include_labels:
        for side, d, stop in (("long", 1, l), ("short", -1, h)):
            entry, exit_price, rps, valid = bracket_numpy(O, H, L, C, n, d, stop, rr)
            with np.errstate(invalid="ignore", divide="ignore"):
                feats[f"g_{side}"] = np.where(valid, d * (exit_price - entry) / rps, np.nan)
                feats[f"c_{side}"] = np.where(valid, (entry + exit_price) / rps, np.nan)
                feats[f"risk_{side}"] = np.where(valid, np.minimum(risk_per_trade, max_leverage * rps / entry), np.nan)
        # Symmetric bracket (stop distance = full opening range on BOTH sides): used only by
        # the random-direction test, so stop placement cannot make a direction look skilful.
        with np.errstate(invalid="ignore"):
            sym_dist = h - l
            for side, d in (("long", 1), ("short", -1)):
                entry, exit_price, rps, valid = bracket_numpy(O, H, L, C, n, d, O[:, 1] - d * sym_dist, rr)
                feats[f"gs_{side}"] = np.where(valid, d * (exit_price - entry) / rps, np.nan)
    return feats[sample_ok]


def build_dataset(
    data: Dict[str, pd.DataFrame],
    rr: Optional[float] = 3.0,
    fee_rate: float = DEFAULT_FEE,
    risk_per_trade: float = 0.01,
    max_leverage: float = 4.0,
    include_labels: bool = True,
    live: bool = False,
) -> pd.DataFrame:
    """
    One row per valid (ticker, session) with features known at the first candle's close.
    With labels: g_/c_/risk_ for long and short, plus r_long / r_short / r_plain net of
    `fee_rate`. `live=True` also keeps the newest session when only its first candle exists.
    """
    parts = [_ticker_rows(t, df, rr, risk_per_trade, max_leverage, include_labels, live) for t, df in data.items()]
    parts = [p for p in parts if p is not None and len(p)]
    if not parts:
        return pd.DataFrame(columns=["ticker", "session", *FEATURES])
    ds = pd.concat(parts, ignore_index=True)
    mkt = ds[ds["ticker"] == MARKET_TICKER].set_index("session")
    ds["mkt_or_dir"] = ds["session"].map(mkt["or_dir"]) if len(mkt) else np.nan
    ds["mkt_or_body_pct"] = ds["session"].map(mkt["or_body_pct"]) if len(mkt) else np.nan
    by_session = ds.groupby("session")["log_rel_volume"]
    ds["rv_pct"] = by_session.rank(pct=True)
    ds["rv_rank"] = by_session.rank(ascending=False, method="first")
    if include_labels:
        ds["r_long"] = apply_fee(ds["g_long"], ds["c_long"], fee_rate)
        ds["r_short"] = apply_fee(ds["g_short"], ds["c_short"], fee_rate)
        ds["r_plain"] = np.where(ds["or_dir"] > 0, ds["r_long"], np.where(ds["or_dir"] < 0, ds["r_short"], np.nan))
    ds = ds.replace([np.inf, -np.inf], np.nan)
    return ds.sort_values(["session", "ticker"]).reset_index(drop=True)


def assert_no_lookahead(
    data: Dict[str, pd.DataFrame],
    n_checks: int = 5,
    seed: int = 0,
    builder: Optional[Callable[..., pd.DataFrame]] = None,
    features: Optional[Sequence[str]] = None,
) -> None:
    """
    Leak guard: for random (ticker, session) pairs, scramble every price/volume AFTER the
    session's first candle (including all later sessions) and require that the session's
    features stay identical. Raises AssertionError on any look-ahead.
    """
    builder = builder or (lambda d: build_dataset(d, include_labels=False))
    features = list(features or FEATURES)
    rng = np.random.default_rng(seed)
    tickers = [t for t in data if t != MARKET_TICKER and len(data[t]) > 100]
    if not tickers:
        return
    for _ in range(n_checks):
        t = tickers[int(rng.integers(len(tickers)))]
        df = data[t]
        sessions = sorted(set(df.index.date))
        if len(sessions) < 4:
            continue
        d = sessions[int(rng.integers(len(sessions) // 2, len(sessions)))]
        cut = df.index[df.index.date == d][0]
        scrambled = df.copy()
        future = scrambled.index > cut
        k = int(future.sum())
        scrambled.loc[future, ["Open", "High", "Low", "Close"]] *= rng.uniform(0.8, 1.2, (k, 1))
        scrambled.loc[future, "Volume"] *= rng.uniform(0.5, 2.0, k)
        base = {t: df}
        if MARKET_TICKER in data and t != MARKET_TICKER:
            base[MARKET_TICKER] = data[MARKET_TICKER]
        a = builder(base)
        b = builder({**base, t: scrambled})
        fa = a[(a["ticker"] == t) & (a["session"] == d)][features].to_numpy(dtype=float)
        fb = b[(b["ticker"] == t) & (b["session"] == d)][features].to_numpy(dtype=float)
        if fa.size == 0 and fb.size == 0:
            continue
        assert fa.shape == fb.shape and np.allclose(fa, fb, equal_nan=True), f"look-ahead leak in features for {t} {d}"


# =====================================================================
# Model
# =====================================================================
def _regressor(seed: int = 0) -> HistGradientBoostingRegressor:
    # Small, heavily regularized trees: the target is noisy.
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
    meta_model: Any = None  # v2; older payloads load with None

    def predict(self, ds: pd.DataFrame) -> pd.DataFrame:
        """Adds pred_long / pred_short / pred_meta (expected net R) and the free-model decision."""
        out = ds.copy()
        if len(out) == 0:
            for c in ("pred_long", "pred_short", "pred_meta", "decision"):
                out[c] = pd.Series(dtype=float)
            return out
        X = out.reindex(columns=self.features).to_numpy(dtype=float)
        out["pred_long"] = self.long_model.predict(X)
        out["pred_short"] = self.short_model.predict(X)
        out["pred_meta"] = self.meta_model.predict(X) if self.meta_model is not None else np.nan
        best = np.where(out["pred_long"] >= out["pred_short"], 1, -1)
        best_r = np.maximum(out["pred_long"], out["pred_short"])
        out["decision"] = np.where(best_r > self.params.get("min_edge", 0.10), best, 0)
        return out


def fit_model(ds: pd.DataFrame, params: Dict[str, Any], seed: int = 0) -> ORBModel:
    """Fits the long, short (free) and meta regressors on labels net of params['fee_rate']."""
    fee = float(params.get("fee_rate", DEFAULT_FEE))
    X_all = ds[FEATURES].to_numpy(dtype=float)
    models = {}
    for name, y in (
        ("long", apply_fee(ds["g_long"], ds["c_long"], fee)),
        ("short", apply_fee(ds["g_short"], ds["c_short"], fee)),
        ("meta", np.where(ds["or_dir"] > 0, apply_fee(ds["g_long"], ds["c_long"], fee),
                          np.where(ds["or_dir"] < 0, apply_fee(ds["g_short"], ds["c_short"], fee), np.nan))),
    ):
        y = np.asarray(y, dtype=float)
        keep = np.isfinite(y)
        m = _regressor(seed)
        if keep.sum() >= 2:
            m.fit(X_all[keep], np.clip(y[keep], *R_CLIP))
        else:  # degenerate tiny dataset: predict 0
            m.fit(np.zeros((2, len(FEATURES))), np.zeros(2))
        models[name] = m
    return ORBModel(models["long"], models["short"], list(FEATURES), dict(params),
                    trained_sessions=sorted(str(s) for s in ds["session"].unique()),
                    n_samples=int(len(ds)), trained_at=datetime.now().isoformat(timespec="seconds"),
                    meta_model=models["meta"])


def default_top_n(n_tickers: int) -> int:
    return max(1, int(round(0.10 * n_tickers)))


def variant_directions(pred: pd.DataFrame, variant: str, top_n: int, min_edge: float) -> np.ndarray:
    """Trade direction per row (+1 / -1 / 0) for one of VARIANTS."""
    plain = pred["or_dir"].fillna(0).to_numpy(dtype=int)
    sip = ((pred["rv_rank"] <= top_n) & (pred["log_rel_volume"] >= 0)).to_numpy()
    meta_ok = (pred["pred_meta"] > min_edge).to_numpy() if "pred_meta" in pred else np.zeros(len(pred), bool)
    if variant == "plain":
        d = plain
    elif variant == "sip":
        d = np.where(sip, plain, 0)
    elif variant == "free":
        d = pred["decision"].to_numpy(dtype=int)
    elif variant == "meta":
        d = np.where(meta_ok, plain, 0)
    elif variant == "meta_sip":
        d = np.where(meta_ok & sip, plain, 0)
    else:
        raise ValueError(f"unknown variant {variant!r}")
    return d.astype(int)


# =====================================================================
# Evaluation
# =====================================================================
def _chosen(df: pd.DataFrame, d: np.ndarray, fee: float) -> Dict[str, np.ndarray]:
    long_ = d == 1
    g = np.where(long_, df["g_long"], df["g_short"]).astype(float)
    c = np.where(long_, df["c_long"], df["c_short"]).astype(float)
    risk = np.where(long_, df["risk_long"], df["risk_short"]).astype(float)
    trade = (d != 0) & np.isfinite(g) & np.isfinite(c) & np.isfinite(risk)
    return {"trade": trade, "gross": g, "cost": c, "risk": risk, "net": apply_fee(g, c, fee)}


def _daily_returns(df: pd.DataFrame, d: np.ndarray, fee: float, sessions: Sequence[Any]) -> pd.Series:
    """Equal split of equity across the day's trades -> day return = mean(risk x net R)."""
    ch = _chosen(df, d, fee)
    contrib = pd.Series(np.where(ch["trade"], ch["risk"] * ch["net"], np.nan), index=df.index)
    ret = contrib.groupby(df["session"].to_numpy()).mean()
    return ret.reindex(list(sessions)).fillna(0.0)


def _compound_pct(ret: pd.Series) -> float:
    return float((np.prod(1.0 + ret.to_numpy()) - 1.0) * 100.0) if len(ret) else 0.0


def variant_stats(df: pd.DataFrame, d: np.ndarray, fee: float, sessions: Sequence[Any]) -> Dict[str, Any]:
    ch = _chosen(df, d, fee)
    t = ch["trade"]
    net, gross, cost = ch["net"][t], ch["gross"][t], ch["cost"][t]
    ret = _daily_returns(df, d, fee, sessions)
    traded_days = df.loc[t, "session"].unique()
    day_ret = ret.reindex(traded_days)
    green, red = int((day_ret > 0).sum()), int((day_ret < 0).sum())
    eq = np.concatenate([[1.0], np.cumprod(1.0 + ret.to_numpy())])
    peak = np.maximum.accumulate(eq)
    sd = ret.std(ddof=1)
    half = len(sessions) // 2
    lo, hi, p_le_0 = block_bootstrap_ci(ret.to_numpy(), kind="mean")
    s_lo, s_hi, _ = block_bootstrap_ci(ret.to_numpy(), kind="sharpe")
    # Random-direction test on a SYMMETRIC bracket (same stop distance long and short).
    # With the ORB stops, going against the first candle puts the stop at the short wick:
    # tiny risk, frequent same-bar stop+target touches (stop counted first) and a huge fee
    # per R, so the candle direction would beat coin flips even without any edge.
    r_long = df["gs_long"].to_numpy(dtype=float)[t]
    r_short = df["gs_short"].to_numpy(dtype=float)[t]
    by_fee = {}
    for bp in EVAL_FEES_BP:
        f = bp / 10_000.0
        by_fee[bp] = {"net_return_pct": _compound_pct(_daily_returns(df, d, f, sessions)),
                      "avg_r": float(np.mean(apply_fee(gross, cost, f))) if t.sum() else 0.0}
    return {
        "trades": int(t.sum()),
        "net_return_pct": _compound_pct(ret),
        "win_rate_pct": float(np.mean(net > 0) * 100.0) if t.sum() else 0.0,
        "avg_r": float(np.mean(net)) if t.sum() else 0.0,
        "avg_gross_r": float(np.mean(gross)) if t.sum() else 0.0,
        "green_days": green, "red_days": red,
        "green_day_rate_pct": float(green / (green + red) * 100.0) if green + red else 0.0,
        "max_drawdown_pct": float(((eq - peak) / peak).min() * 100.0),
        "sharpe": float(ret.mean() / sd * np.sqrt(252)) if sd > 0 else 0.0,
        "sr_daily": float(ret.mean() / sd) if sd > 0 else 0.0,
        "mean_daily_ci_bp": (lo * 1e4, hi * 1e4),
        "p_mean_le_0": p_le_0,
        "sharpe_ci": (s_lo, s_hi),
        "breakeven_fee_bp": breakeven_fee(gross, cost) * 1e4 if t.sum() else float("nan"),
        "p_random_direction": random_direction_test(r_long, r_short, d[t]) if t.sum() else 1.0,
        "first_half_pct": _compound_pct(ret.iloc[:half]),
        "second_half_pct": _compound_pct(ret.iloc[half:]),
        "by_fee_bp": by_fee,
        "_returns": ret,
    }


def verdict_for(best: Optional[Dict[str, Any]]) -> str:
    if best is None:
        return "NO EDGE"
    if (best["dsr"] >= 0.95 and best["by_fee_bp"][3]["net_return_pct"] > 0
            and best["first_half_pct"] > 0 and best["second_half_pct"] > 0):
        return "SUPPORTED"
    if best["avg_r"] > 0 and best["p_mean_le_0"] <= 0.10:
        return "WEAK"
    return "NO EDGE"


def walk_forward(
    ds: pd.DataFrame,
    params: Dict[str, Any],
    min_train: Optional[int] = None,
    step: Optional[int] = None,
    top_n: Optional[int] = None,
    seed: int = 0,
) -> Dict[str, Any]:
    """
    Retrains on past sessions only and trades the next block; evaluates all VARIANTS on
    the identical out-of-sample sessions. Defaults: 20 / 5 sessions for <= 120 sessions,
    else min(250, n // 3) / 21.
    """
    sessions = sorted(ds["session"].unique())
    n = len(sessions)
    if min_train is None or step is None:
        dm, dstep = (20, 5) if n <= 120 else (min(250, n // 3), 21)
        min_train = dm if min_train is None else min_train
        step = dstep if step is None else step
    if n <= min_train:
        return {"error": f"need more than {min_train} sessions, have {n}"}
    fee = float(params.get("fee_rate", DEFAULT_FEE))
    min_edge = float(params.get("min_edge", 0.10))
    top_n = top_n or default_top_n(ds["ticker"].nunique())

    parts = []
    for start in range(min_train, n, step):
        block = sessions[start:start + step]
        model = fit_model(ds[ds["session"] < block[0]], params, seed)
        parts.append(model.predict(ds[ds["session"].isin(block)]))
    oos = pd.concat(parts).reset_index(drop=True)
    oos_sessions = sorted(oos["session"].unique())

    variants: Dict[str, Dict[str, Any]] = {}
    for v in VARIANTS:
        variants[v] = variant_stats(oos, variant_directions(oos, v, top_n, min_edge), fee, oos_sessions)
    srs = [s["sr_daily"] for s in variants.values()]
    var_sr = float(np.var(srs, ddof=1)) if len(srs) > 1 else 0.0
    for s in variants.values():
        s["dsr"] = deflated_sharpe(s["_returns"].to_numpy(), n_trials=len(VARIANTS), var_sr_trials=var_sr)
    eligible = {k: s for k, s in variants.items() if s["trades"] >= MIN_TRADES_FOR_BEST}
    best = max(eligible, key=lambda k: eligible[k]["sharpe"]) if eligible else None
    daily = {k: pd.DataFrame({"ret": s["_returns"], "equity": np.cumprod(1.0 + s["_returns"].to_numpy())})
             for k, s in variants.items()}
    for s in variants.values():
        s.pop("_returns")
    return {
        "oos_sessions": f"{oos_sessions[0]} -> {oos_sessions[-1]} ({len(oos_sessions)})",
        "n_oos_sessions": len(oos_sessions),
        "min_train": min_train, "step": step, "top_n": top_n,
        "variants": variants,
        "best": best,
        "verdict": verdict_for(variants[best] if best else None),
        "daily": daily,
    }


def format_evaluation(ev: Dict[str, Any], model: Optional["ORBModel"] = None, params: Optional[Dict[str, Any]] = None) -> str:
    if "error" in ev:
        return f"Walk-forward skipped: {ev['error']}"
    p = params or (model.params if model is not None else {})
    fmt_pf = lambda x: f"{x:+7.2f}%"
    lines = [
        "=" * 112,
        " ORB v2 - walk-forward, OUT-OF-SAMPLE only "
        f"(train >= {ev['min_train']} sessions, retrain every {ev['step']})",
        f" Test sessions: {ev['oos_sessions']} | fee {p.get('fee_rate', DEFAULT_FEE) * 1e4:.1f} bp/side | "
        f"RR 1:{p.get('rr', 3):g} | min edge {p.get('min_edge', 0.10):+.2f}R | Stocks in Play top {ev['top_n']}",
        "=" * 112,
        " Variant    Trades     Net   Win%   AvgR  Green%   MaxDD  Sharpe  Sharpe95%CI      DSR  BE-fee  p(rand)"
        "   Half1   Half2",
    ]
    for k, s in ev["variants"].items():
        mark = "*" if k == ev["best"] else " "
        lines.append(
            f"{mark}{k:<9} {s['trades']:>7} {fmt_pf(s['net_return_pct'])} {s['win_rate_pct']:6.1f} "
            f"{s['avg_r']:+6.3f} {s['green_day_rate_pct']:6.1f} {s['max_drawdown_pct']:7.2f} {s['sharpe']:7.2f} "
            f"[{s['sharpe_ci'][0]:+5.2f},{s['sharpe_ci'][1]:+5.2f}] {s['dsr']:7.3f} {s['breakeven_fee_bp']:5.1f}bp "
            f"{s['p_random_direction']:7.3f} {fmt_pf(s['first_half_pct'])} {fmt_pf(s['second_half_pct'])}"
        )
    lines += ["-" * 112, " Net return / avg R by fee per side:"]
    for k, s in ev["variants"].items():
        lines.append(f"   {k:<9} " + " | ".join(
            f"{bp} bp: {s['by_fee_bp'][bp]['net_return_pct']:+7.2f}% {s['by_fee_bp'][bp]['avg_r']:+.3f}R"
            for bp in EVAL_FEES_BP))
    best = ev["best"]
    lines += [
        "-" * 112,
        f" Best variant (highest Sharpe, >= {MIN_TRADES_FOR_BEST} trades): {best or 'none'}",
        f" Mean daily return 95% CI: [{ev['variants'][best]['mean_daily_ci_bp'][0]:+.2f}, "
        f"{ev['variants'][best]['mean_daily_ci_bp'][1]:+.2f}] bp, p(mean <= 0) = "
        f"{ev['variants'][best]['p_mean_le_0']:.3f}" if best else " (no variant with enough trades)",
        f" VERDICT: {ev['verdict']}",
        "   SUPPORTED = DSR >= 0.95 and profitable at 3 bp and both halves positive;"
        " WEAK = avg R > 0 and p(mean <= 0) <= 0.10",
        "=" * 112,
    ]
    return "\n".join(lines)


# =====================================================================
# Persistence + workflow
# =====================================================================
def model_paths(workspace: str) -> Dict[str, str]:
    d = os.path.join(workspace, "models")
    return {"dir": d, "latest": os.path.join(d, "orb_model.joblib"), "meta": os.path.join(d, "orb_model.json")}


def save_model(model: ORBModel, workspace: str) -> str:
    p = model_paths(workspace)
    os.makedirs(p["dir"], exist_ok=True)
    # A plain dict (not the dataclass) loads from any entry point (CLI __main__, notebook).
    payload = {"format": 2, **model.__dict__}
    joblib.dump(payload, p["latest"])
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    joblib.dump(payload, os.path.join(p["dir"], f"orb_model_{stamp}.joblib"))
    meta = {k: v for k, v in model.__dict__.items() if k not in ("long_model", "short_model", "meta_model")}
    if model.trained_sessions:
        meta["trained_sessions"] = (f"{model.trained_sessions[0]} -> {model.trained_sessions[-1]} "
                                    f"({len(model.trained_sessions)})")
    with open(p["meta"], "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, default=str)
    return p["latest"]


def load_model(workspace: str) -> Optional[ORBModel]:
    """Loads the latest model; payloads without meta_model (v1) still load."""
    p = model_paths(workspace)["latest"]
    if not os.path.exists(p):
        return None
    try:
        payload = joblib.load(p)
        if isinstance(payload, ORBModel):
            return payload
        payload = dict(payload)
        payload.pop("format", None)
        known = set(ORBModel.__dataclass_fields__)
        return ORBModel(**{k: v for k, v in payload.items() if k in known})
    except Exception as e:  # noqa: BLE001
        print(f"Could not load {p} ({type(e).__name__}: {e}); a new model will be trained.")
        return None


def _complete_only(live: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    out = {}
    for t, df in live.items():
        complete, _ = split_incomplete_session(df)
        if len(complete):
            out[t] = complete
    return out


def train_and_save(
    workspace: str,
    tickers: Sequence[str] = MODEL_UNIVERSE,
    rr: Optional[float] = 3.0,
    fee_rate: float = DEFAULT_FEE,
    risk_per_trade: float = 0.01,
    max_leverage: float = 4.0,
    min_edge: float = 0.10,
    refresh_data: bool = True,
    source: str = "yfinance",
    start: Any = "2016-01-01",
    feed: str = "sip",
    top_n: Optional[int] = None,
    key: Optional[str] = None,
    secret: Optional[str] = None,
    http: Any = None,
    min_train: Optional[int] = None,
    step: Optional[int] = None,
    live_data: Optional[Dict[str, pd.DataFrame]] = None,
) -> Dict[str, Any]:
    """
    Loads/extends the 5-minute history, checks for look-ahead, evaluates all variants
    walk-forward, retrains on ALL complete history, saves, and returns
    {model, evaluation, report, data, live_data, dataset}.
    source="yfinance": last 60 days, merged into <workspace>/datasets/intraday/
    source="alpaca":   history from `start`, cached per feed in
                       <workspace>/datasets/intraday_alpaca_<feed>/ (never mixed with yfinance bars)
    `live_data` (dict of frames) skips downloading entirely (tests / offline use).
    """
    previous = load_model(workspace)
    if previous is not None:
        print(f"Loaded saved model: trained {previous.trained_at} on {len(previous.trained_sessions)} sessions "
              f"({previous.n_samples} samples).")
    used_feed = None
    if live_data is not None:
        live = dict(live_data)
    elif source == "alpaca":
        from alpaca_data import resolve_feed, update_cache_from_alpaca

        used_feed = resolve_feed(key, secret, feed, http)
        cache = os.path.join(workspace, "datasets", f"intraday_alpaca_{used_feed}")
        print(f"Alpaca feed: {used_feed} -> cache {cache}")
        live = update_cache_from_alpaca(tickers, cache, start=start, key=key, secret=secret, feed=used_feed, http=http)
    elif source == "yfinance":
        live = download_universe(tickers, period="60d", interval="5m",
                                 cache_dir=os.path.join(workspace, "datasets", "intraday"),
                                 refresh=refresh_data, verbose=False, keep_incomplete=True)
    else:
        raise ValueError("source must be 'yfinance' or 'alpaca'")
    data = _complete_only(live)
    if not data:
        raise ValueError("No 5-minute data available.")
    n_sessions = len(set().union(*[set(df.index.date) for df in data.values()]))
    print(f"5-minute history: {len(data)} tickers, {n_sessions} complete sessions"
          + (f" (feed {used_feed})" if used_feed else ""))

    assert_no_lookahead(data)  # refuses to train on features that peek into the future
    params = dict(rr=rr, fee_rate=fee_rate, risk_per_trade=risk_per_trade, max_leverage=max_leverage,
                  min_edge=min_edge, source=source, feed=used_feed, n_tickers=len(data))
    ds = build_dataset(data, rr=rr, fee_rate=fee_rate, risk_per_trade=risk_per_trade, max_leverage=max_leverage)
    top_n = top_n or default_top_n(len(data))
    ev = walk_forward(ds, params, min_train=min_train, step=step, top_n=top_n)
    model = fit_model(ds, params)
    model.params.update(top_n=top_n, variant=ev.get("best") or "plain", verdict=ev.get("verdict", "NO EDGE"))
    model.evaluation = {k: v for k, v in ev.items() if k != "daily"}
    path = save_model(model, workspace)
    report = format_evaluation(ev, model)
    print(report)
    prev_n = len(previous.trained_sessions) if previous else 0
    print(f"Saved model -> {path}  ({model.n_samples} samples, {len(model.trained_sessions)} sessions, "
          f"{len(model.trained_sessions) - prev_n:+d} vs. previous) | live variant: {model.params['variant']}")
    return {"model": model, "evaluation": ev, "report": report, "data": data, "live_data": live, "dataset": ds}


def todays_signals(model: ORBModel, live_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Applies the saved model's winning variant to the newest session in `live_data`
    (works with only the first 5-minute candle of that session). Columns: ticker,
    session, variant, verdict, signal (LONG/SHORT/SKIP), pred_r, ref_price (first candle
    close; real entry = next bar's open), stop (other side of the first candle), target (1:RR).
    """
    if not live_data:
        return pd.DataFrame()
    last_day = max(df.index[-1].date() for df in live_data.values() if len(df))
    since = (pd.Timestamp(last_day) - pd.Timedelta(days=45)).date()
    recent = {t: df[df.index.date >= since] for t, df in live_data.items()
              if len(df) and df.index[-1].date() == last_day}
    ds = build_dataset(recent, include_labels=False, live=True)
    ds = ds[ds["session"] == last_day]
    if ds.empty:
        return pd.DataFrame()
    pred = model.predict(ds)
    p = model.params
    variant = p.get("variant") or ("free" if p.get("use_model") else "plain")
    if variant in ("meta", "meta_sip") and model.meta_model is None:
        variant = "plain"
    top_n = p.get("top_n") or default_top_n(len(live_data))
    dirs = variant_directions(pred, variant, top_n, float(p.get("min_edge", 0.10)))
    rr = p.get("rr") or 0.0
    out = []
    for (_, r), d in zip(pred.iterrows(), dirs):
        ref = float(r["or_close"])
        stop = float(r["or_low"]) if d == 1 else float(r["or_high"]) if d == -1 else np.nan
        if variant == "free":
            pr = r["pred_long"] if d == 1 else r["pred_short"] if d == -1 else max(r["pred_long"], r["pred_short"])
        else:
            pr = r["pred_meta"]
        out.append({
            "ticker": r["ticker"], "session": last_day, "variant": variant, "verdict": p.get("verdict", "n/a"),
            "signal": {1: "LONG", -1: "SHORT", 0: "SKIP"}[int(d)],
            "pred_r": float(pr) if pr is not None and np.isfinite(pr) else np.nan,
            "ref_price": ref, "stop": stop,
            "target": ref + d * rr * abs(ref - stop) if d and rr else np.nan,
        })
    res = pd.DataFrame(out)
    order = res["signal"].eq("SKIP").astype(int)
    return res.assign(_o=order).sort_values(["_o", "pred_r"], ascending=[True, False]).drop(columns="_o").reset_index(drop=True)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Train / update the persistent ORB v2 model")
    ap.add_argument("--workspace", default="ATU_Workspace")
    ap.add_argument("--fee", type=float, default=DEFAULT_FEE, help="Fees + slippage per side (0.0002 = 2 bp)")
    ap.add_argument("--min-edge", type=float, default=0.10)
    ap.add_argument("--universe", choices=["default", "large"], default="default")
    ap.add_argument("--source", choices=["yfinance", "alpaca"], default="yfinance")
    ap.add_argument("--start", default="2016-01-01", help="Alpaca history start")
    ap.add_argument("--feed", default="sip", choices=["sip", "iex"])
    ap.add_argument("--top-n", type=int, default=None, help="Stocks in Play per day (default 10%% of universe)")
    ap.add_argument("--no-refresh", dest="refresh", action="store_false", help="yfinance: use cached bars only")
    a = ap.parse_args()
    import orb_model as _om  # run through the importable module so saved objects resolve everywhere

    res = _om.train_and_save(a.workspace, tickers=_om.LARGE_UNIVERSE if a.universe == "large" else _om.MODEL_UNIVERSE,
                             fee_rate=a.fee, min_edge=a.min_edge, refresh_data=a.refresh, source=a.source,
                             start=a.start, feed=a.feed, top_n=a.top_n)
    sig = _om.todays_signals(res["model"], res["live_data"])
    if len(sig):
        print("\nSignals for the latest session:\n" + sig.to_string(index=False))
