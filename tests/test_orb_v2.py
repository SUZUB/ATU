"""
Offline tests for ORB v2 (no network). Run:  python -m pytest tests/test_orb_v2.py -v

yfinance / ta are stubbed when not installed: the ORB code paths under test never
call them (data is synthetic or served by a mocked Alpaca HTTP object).
"""

import importlib.util
import os
import sys
import types

import numpy as np
import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def _stub_missing_modules() -> None:
    if importlib.util.find_spec("yfinance") is None:
        yf = types.ModuleType("yfinance")
        yf.download = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("network disabled in tests"))
        yf.Ticker = lambda *a, **k: None
        sys.modules["yfinance"] = yf
    if importlib.util.find_spec("ta") is None:
        class _Ind:
            def __init__(self, *a, **k):
                pass
        ta = types.ModuleType("ta")
        for sub, names in (("momentum", ["RSIIndicator"]), ("trend", ["ADXIndicator", "MACD"]),
                           ("volatility", ["AverageTrueRange", "BollingerBands"])):
            m = types.ModuleType(f"ta.{sub}")
            for n in names:
                setattr(m, n, _Ind)
            setattr(ta, sub, m)
            sys.modules[f"ta.{sub}"] = m
        sys.modules["ta"] = ta


_stub_missing_modules()

import alpaca_data as ad  # noqa: E402
import orb_eval as oe  # noqa: E402
import orb_model as om  # noqa: E402
import research_strategies as rs  # noqa: E402

NY = "America/New_York"
BARS_PER_DAY = 78


# ---------------------------------------------------------------------------
# Synthetic market
# ---------------------------------------------------------------------------
def make_universe(n_tickers=16, n_sessions=260, drift=0.0, seed=0, start="2023-01-03", last_day_bars=None):
    """
    5-minute bars, 09:30-15:55 NY. drift > 0 plants an edge: every bar after the first
    drifts in the direction of the first candle (what ORB tries to capture).
    """
    rng = np.random.default_rng(seed)
    days = pd.bdate_range(start, periods=n_sessions)
    offsets = pd.to_timedelta(np.arange(BARS_PER_DAY) * 5 + 9 * 60 + 30, unit="m")
    data = {}
    names = ["SPY"] + [f"T{i:02d}" for i in range(n_tickers - 1)]
    for name in names:
        r = np.empty((n_sessions, BARS_PER_DAY))
        r[:, 0] = rng.normal(0, 0.004, n_sessions)
        r[:, 1:] = rng.normal(0, 0.0012, (n_sessions, BARS_PER_DAY - 1)) + drift * np.sign(r[:, :1])
        gaps = rng.normal(0, 0.005, n_sessions)
        price = 100.0 * np.exp(rng.normal(0, 0.3))
        opens, closes = np.empty_like(r), np.empty_like(r)
        for s in range(n_sessions):
            day_open = price * np.exp(gaps[s])
            path = day_open * np.exp(np.cumsum(r[s]))
            opens[s, 0] = day_open
            opens[s, 1:] = path[:-1]
            closes[s] = path
            price = path[-1]
        wick = np.abs(rng.normal(0, 0.0006, r.shape))
        highs = np.maximum(opens, closes) * (1 + wick)
        lows = np.minimum(opens, closes) * (1 - np.abs(rng.normal(0, 0.0006, r.shape)))
        vol = np.exp(rng.normal(11, 0.5, r.shape))
        vol[:, 0] = np.exp(rng.normal(13, 0.6, n_sessions))
        idx = pd.DatetimeIndex((days.values[:, None] + offsets.values[None, :]).ravel()).tz_localize(NY)
        df = pd.DataFrame({"Open": opens.ravel(), "High": highs.ravel(), "Low": lows.ravel(),
                           "Close": closes.ravel(), "Volume": vol.ravel()}, index=idx)
        df.index.name = "Date"
        if last_day_bars is not None:
            last = df.index.normalize() == df.index.normalize()[-1]
            keep = ~last | (np.cumsum(last) <= last_day_bars)
            df = df[keep]
        data[name] = df
    return data


# ---------------------------------------------------------------------------
# a. numpy bracket == reference _simulate_bracket + old R formula
# ---------------------------------------------------------------------------
def test_numpy_bracket_matches_reference():
    rng = np.random.default_rng(42)
    S, M, fee = 1000, 12, 0.0002
    n_bars = rng.integers(2, M + 1, S)
    O, H, L, C = (np.full((S, M), np.nan) for _ in range(4))
    for s in range(S):
        p = 100.0
        for k in range(n_bars[s]):
            gap = rng.normal(0, 0.01) if rng.random() < 0.15 else 0.0  # gaps through levels
            o = p * (1 + gap)
            c = o * (1 + rng.normal(0, 0.006))
            wide = 0.02 if rng.random() < 0.1 else 0.003  # bars touching stop AND target
            H[s, k] = max(o, c) * (1 + abs(rng.normal(0, wide)))
            L[s, k] = min(o, c) * (1 - abs(rng.normal(0, wide)))
            O[s, k], C[s, k], p = o, c, c
    checked = 0
    for rr in (3.0, None, 1.0):
        for d, stop in ((1, L[:, 0]), (-1, H[:, 0])):
            entry, exit_np, rps, valid = om.bracket_numpy(O, H, L, C, n_bars, d, stop, rr)
            g = np.where(valid, d * (exit_np - entry) / rps, np.nan)
            c = np.where(valid, (entry + exit_np) / rps, np.nan)
            net_np = om.apply_fee(g, c, fee)
            for s in np.flatnonzero(valid):
                k = n_bars[s]
                after = pd.DataFrame({"Open": O[s, 1:k], "High": H[s, 1:k], "Low": L[s, 1:k], "Close": C[s, 1:k]},
                                     index=pd.date_range("2024-01-02 09:35", periods=k - 1, freq="5min"))
                e = float(after.iloc[0]["Open"])
                r_ps = d * (e - stop[s])
                target = e + d * rr * r_ps if rr is not None else None
                exit_ref, _, _ = rs._simulate_bracket(after, d, e, float(stop[s]), target)
                r_old = d * (exit_ref - e) / r_ps - fee * (e + exit_ref) / r_ps
                assert exit_np[s] == pytest.approx(exit_ref, rel=1e-12, abs=1e-12), (rr, d, s)
                assert net_np[s] == pytest.approx(r_old, rel=1e-9, abs=1e-9)
                checked += 1
    assert checked >= 1500, checked
    print(f"\n[a] numpy bracket identical to reference on {checked} trades")


# ---------------------------------------------------------------------------
# b. leak guard
# ---------------------------------------------------------------------------
def test_leak_guard_passes_clean_and_catches_leak():
    data = make_universe(n_tickers=4, n_sessions=40, seed=1)
    om.assert_no_lookahead(data, n_checks=8)  # clean features: must pass

    def leaky_builder(d):
        ds = om.build_dataset(d, include_labels=False)
        closes = {t: df.groupby(df.index.date)["Close"].last() for t, df in d.items()}
        ds["leak"] = [closes[t].get(s, np.nan) for t, s in zip(ds["ticker"], ds["session"])]  # day's CLOSE
        return ds

    with pytest.raises(AssertionError, match="look-ahead"):
        om.assert_no_lookahead(data, n_checks=8, builder=leaky_builder, features=om.FEATURES + ["leak"])
    print("\n[b] leak guard: clean data passes, leaky feature caught")


# ---------------------------------------------------------------------------
# c. planted edge -> SUPPORTED; no edge -> never SUPPORTED
# ---------------------------------------------------------------------------
def _verdict(tmp_path, drift, seed):
    data = make_universe(n_tickers=16, n_sessions=260, drift=drift, seed=seed)
    res = om.train_and_save(str(tmp_path / f"ws_{seed}_{drift}"), live_data=data, fee_rate=0.0002)
    ev = res["evaluation"]
    best = ev["variants"][ev["best"]] if ev["best"] else {}
    return ev["verdict"], ev["best"], best


def test_planted_edge_is_supported(tmp_path):
    data = make_universe(n_tickers=16, n_sessions=260, drift=0.0001, seed=7)
    ev = om.train_and_save(str(tmp_path / "ws_edge"), live_data=data, fee_rate=0.0002)["evaluation"]
    s, p_plain = ev["variants"][ev["best"]], ev["variants"]["plain"]["p_random_direction"]
    print(f"\n[c] planted edge: verdict {ev['verdict']}, best {ev['best']}, DSR {s['dsr']:.3f}, "
          f"avg R {s['avg_r']:+.3f}, net@3bp {s['by_fee_bp'][3]['net_return_pct']:+.1f}%, "
          f"plain p(random direction) {p_plain:.3f}")
    assert ev["verdict"] == "SUPPORTED"
    assert p_plain < 0.01  # the candle direction really carries the planted information


@pytest.mark.parametrize("seed", [11, 12, 13, 14])
def test_no_edge_is_never_supported(tmp_path, seed):
    verdict, best, s = _verdict(tmp_path, drift=0.0, seed=seed)
    print(f"\n[c] no edge seed {seed}: verdict {verdict}, best {best}, "
          f"DSR {s.get('dsr', float('nan')):.3f}, avg R {s.get('avg_r', float('nan')):+.3f}, "
          f"p(random direction) {s.get('p_random_direction', float('nan')):.3f}")
    assert verdict != "SUPPORTED"


def test_random_direction_not_fooled_by_stop_geometry(tmp_path):
    """No edge: the plain variant's direction must not look better than coin flips."""
    data = make_universe(n_tickers=16, n_sessions=260, drift=0.0, seed=21)
    res = om.train_and_save(str(tmp_path / "ws_geom"), live_data=data, fee_rate=0.0002)
    p_plain = res["evaluation"]["variants"]["plain"]["p_random_direction"]
    print(f"\n[c] no edge: plain p(random direction) = {p_plain:.3f} (must not be tiny)")
    assert p_plain > 0.01


# ---------------------------------------------------------------------------
# d. save / load, old payload
# ---------------------------------------------------------------------------
def test_save_load_roundtrip_and_old_payload(tmp_path):
    data = make_universe(n_tickers=6, n_sessions=60, seed=3)
    ds = om.build_dataset(data)
    params = dict(rr=3.0, fee_rate=0.0002, risk_per_trade=0.01, max_leverage=4.0, min_edge=0.1,
                  variant="meta", top_n=1, verdict="NO EDGE")
    model = om.fit_model(ds, params)
    om.save_model(model, str(tmp_path))
    loaded = om.load_model(str(tmp_path))
    a, b = model.predict(ds), loaded.predict(ds)
    for col in ("pred_long", "pred_short", "pred_meta", "decision"):
        np.testing.assert_allclose(a[col].to_numpy(float), b[col].to_numpy(float))

    import joblib

    v1_features = [f for f in om.FEATURES if f != "rv_pct"]  # the v1 model had 12 features
    X1 = ds[v1_features].to_numpy(float)
    keep = np.isfinite(ds["r_long"].to_numpy(float)) & np.isfinite(ds["r_short"].to_numpy(float))
    v1_long, v1_short = om._regressor(), om._regressor()
    v1_long.fit(X1[keep], ds["r_long"].to_numpy(float)[keep])
    v1_short.fit(X1[keep], ds["r_short"].to_numpy(float)[keep])
    old = {"format": 1, "long_model": v1_long, "short_model": v1_short,
           "features": v1_features,
           "params": {"rr": 3.0, "fee_rate": 0.0001, "min_edge": 0.1, "use_model": False},
           "trained_sessions": ["2024-01-02"], "n_samples": 10, "trained_at": "2026-11-12T00:00:00",
           "evaluation": {}}
    joblib.dump(old, om.model_paths(str(tmp_path))["latest"])
    legacy = om.load_model(str(tmp_path))
    assert legacy is not None and legacy.meta_model is None
    sig = om.todays_signals(legacy, data)
    assert len(sig) and set(sig["variant"]) == {"plain"}
    print("\n[d] round trip identical; v1 payload (no meta_model) loads and falls back to 'plain'")


# ---------------------------------------------------------------------------
# e. live signals with only the first candle of the last day
# ---------------------------------------------------------------------------
def test_todays_signals_first_candle_only(tmp_path):
    data = make_universe(n_tickers=8, n_sessions=60, seed=4, last_day_bars=1)
    last_day = max(df.index[-1].date() for df in data.values())
    assert all((df.index.date == last_day).sum() == 1 for df in data.values())
    hist = {t: df[df.index.date < last_day] for t, df in data.items()}
    model = om.fit_model(om.build_dataset(hist), dict(rr=3.0, fee_rate=0.0002, min_edge=-10.0,
                                                      variant="meta", top_n=2, verdict="WEAK"))
    sig = om.todays_signals(model, data)
    assert len(sig) == len(data) and set(sig["session"]) == {last_day}
    assert {"variant", "verdict", "signal", "stop", "target"} <= set(sig.columns)
    traded = sig[sig["signal"] != "SKIP"]
    assert len(traded) > 0
    long_ = traded[traded["signal"] == "LONG"]
    assert (long_["stop"] < long_["ref_price"]).all() and (long_["target"] > long_["ref_price"]).all()
    print(f"\n[e] {len(sig)} signals from a first-candle-only session ({len(traded)} trades)")


# ---------------------------------------------------------------------------
# f. Alpaca loader with a mocked HTTP object
# ---------------------------------------------------------------------------
class FakeResp:
    def __init__(self, status, payload=None, text=""):
        self.status_code, self._payload, self.text = status, payload, text

    def json(self):
        return self._payload


class FakeAlpaca:
    """Serves bars from `truth` (DataFrame per symbol) with paging, optional SIP denial and 429s."""

    def __init__(self, truth, page=40, deny_sip=False, n_429=0):
        self.truth, self.page, self.deny_sip, self.n_429 = truth, page, deny_sip, n_429
        self.calls = []

    def get(self, url, headers=None, params=None, timeout=None):
        self.calls.append(dict(params))
        assert headers["APCA-API-KEY-ID"] == "k" and url == ad.ALPACA_BARS_URL
        if self.n_429 > 0:
            self.n_429 -= 1
            return FakeResp(429, text="too many requests")
        if self.deny_sip and params["feed"] == "sip":
            return FakeResp(403, text="subscription does not permit querying recent SIP data")
        sym = params["symbols"]
        df = self.truth[sym]
        s, e = pd.Timestamp(params["start"]), pd.Timestamp(params["end"])
        part = df[(df.index >= s) & (df.index <= e)]
        off = int(params.get("page_token") or 0)
        chunk = part.iloc[off:off + self.page]
        bars = [{"t": ts.tz_convert("UTC").strftime("%Y-%m-%dT%H:%M:%SZ"), "o": r.Open, "h": r.High,
                 "l": r.Low, "c": r.Close, "v": r.Volume} for ts, r in chunk.iterrows()]
        nxt = str(off + self.page) if off + self.page < len(part) else None
        return FakeResp(200, {"bars": {sym: bars}, "next_page_token": nxt})


def _alpaca_truth(days=6, scale=1.0, start="2025-03-03"):
    """Bars incl. pre/post-market (04:00-19:55) like Alpaca returns them."""
    idx = []
    for d in pd.bdate_range(start, periods=days):
        idx += list(pd.date_range(d + pd.Timedelta(hours=4), d + pd.Timedelta(hours=19, minutes=55), freq="5min"))
    idx = pd.DatetimeIndex(idx).tz_localize(NY)
    px = (100 + np.arange(len(idx)) * 0.01) * scale
    return pd.DataFrame({"Open": px, "High": px + 0.05, "Low": px - 0.05, "Close": px + 0.01,
                         "Volume": np.full(len(idx), 1000.0)}, index=idx)


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch):
    monkeypatch.setattr(ad.time, "sleep", lambda s: None)


def test_alpaca_pagination_session_filter_and_429():
    truth = {"AAA": _alpaca_truth(days=3)}
    http = FakeAlpaca(truth, page=50, n_429=2)
    df = ad.fetch_alpaca_bars("AAA", "2025-03-01", "2025-03-10", key="k", secret="s", http=http)
    times = df.index.strftime("%H:%M")
    assert times.min() == "09:30" and times.max() == "15:55"
    assert len(df) == 3 * 78 and df.index.is_unique and df.index.tz is not None
    assert sum(1 for c in http.calls if c.get("page_token")) >= 5  # followed next_page_token
    assert df.attrs["feed"] == "sip"
    print(f"\n[f] pagination: {len(http.calls)} requests (2 rate-limited), {len(df)} regular-session bars kept "
          f"of {len(truth['AAA'])}")


def test_alpaca_sip_falls_back_to_iex():
    http = FakeAlpaca({"AAA": _alpaca_truth(days=2)}, deny_sip=True)
    df = ad.fetch_alpaca_bars("AAA", "2025-03-01", "2025-03-10", key="k", secret="s", http=http)
    assert df.attrs["feed"] == "iex" and len(df) == 2 * 78
    assert http.calls[0]["feed"] == "sip" and http.calls[-1]["feed"] == "iex"
    print("\n[f] SIP denied -> IEX fallback")


def test_alpaca_incremental_merge_and_split(tmp_path):
    cache = str(tmp_path / "alp")
    first = _alpaca_truth(days=6)
    http = FakeAlpaca({"AAA": first.iloc[: first.index.normalize().isin(first.index.normalize().unique()[:4]).sum()]})
    out1 = ad.update_cache_from_alpaca(["AAA"], cache, start="2025-03-01", end="2025-03-20",
                                       key="k", secret="s", http=http, verbose=False)
    assert len(set(out1["AAA"].index.date)) == 4

    # incremental: same prices, 6 days available -> fetch starts 3 days before last cached date
    http2 = FakeAlpaca({"AAA": first})
    out2 = ad.update_cache_from_alpaca(["AAA"], cache, start="2025-03-01", end="2025-03-20",
                                       key="k", secret="s", http=http2, verbose=False)
    df2 = out2["AAA"]
    assert len(set(df2.index.date)) == 6 and df2.index.is_unique and len(df2) == 6 * 78
    last_cached = pd.Timestamp(out1["AAA"].index[-1].date())
    assert pd.Timestamp(http2.calls[0]["start"]).tz_convert(NY).date() == (last_cached - pd.Timedelta(days=3)).date()
    on_disk = ad._read_cache(os.path.join(cache, "ohlcv_AAA_5m.csv"))
    assert len(on_disk) == 6 * 78

    # split: provider now returns history divided by 2 -> full re-download replaces the cache
    adjusted = _alpaca_truth(days=7, scale=0.5)
    http3 = FakeAlpaca({"AAA": adjusted})
    out3 = ad.update_cache_from_alpaca(["AAA"], cache, start="2025-03-01", end="2025-03-20",
                                       key="k", secret="s", http=http3, verbose=False)
    df3 = out3["AAA"]
    full_start = [c for c in http3.calls if pd.Timestamp(c["start"]).tz_convert(NY).date() <= pd.Timestamp("2025-03-01").date()]
    assert full_start, "split must trigger a download from `start`"
    expected = ad.regular_session(adjusted)
    np.testing.assert_allclose(df3["Close"].to_numpy(), expected["Close"].to_numpy())
    assert len(set(df3.index.date)) == 7 and df3.index.is_unique
    print("\n[f] incremental merge (no duplicates, overlap start ok) and split re-download")


def test_download_universe_never_caches_today(tmp_path, monkeypatch):
    today = pd.Timestamp("2026-03-11", tz=NY)  # a Wednesday; the clock is frozen at 10:00 below
    hist_day = today - pd.Timedelta(days=1)
    idx = pd.DatetimeIndex(list(pd.date_range(hist_day + pd.Timedelta(hours=9, minutes=30), periods=78, freq="5min"))
                           + [today + pd.Timedelta(hours=9, minutes=30)])
    frame = pd.DataFrame({"Open": 1.0, "High": 1.0, "Low": 1.0, "Close": 1.0, "Volume": 1.0}, index=idx)

    class FakeProc:
        def fetch_data(self, **kw):
            return frame.copy()

    real_split = rs.split_incomplete_session
    monkeypatch.setattr(rs, "MarketDataProcessor", FakeProc)
    monkeypatch.setattr(rs, "split_incomplete_session",
                        lambda df, close_buffer="16:30", now=None: real_split(df, close_buffer,
                                                                             now=today + pd.Timedelta(hours=10)))
    live = rs.download_universe(["ZZZ"], period="60d", interval="5m", cache_dir=str(tmp_path), verbose=False,
                                keep_incomplete=True)
    cached = pd.read_csv(tmp_path / "ohlcv_ZZZ_5m.csv", index_col=0)
    assert len(live["ZZZ"]) == 79 and len(cached) == 78
    plain = rs.download_universe(["ZZZ"], period="60d", interval="5m", cache_dir=str(tmp_path), verbose=False)
    assert len(plain["ZZZ"]) == 78
    print("\n[2] today's session returned only with keep_incomplete=True, never cached")


# ---------------------------------------------------------------------------
# orb_eval sanity
# ---------------------------------------------------------------------------
def test_orb_eval_basics():
    rng = np.random.default_rng(0)
    good = rng.normal(0.002, 0.01, 1000)
    noise = rng.normal(0.0, 0.01, 1000)
    lo, hi, p = oe.block_bootstrap_ci(good, kind="mean")
    assert lo > 0 and p < 0.05
    assert oe.deflated_sharpe(good, n_trials=5, var_sr_trials=0.0) > 0.95
    assert oe.deflated_sharpe(noise, n_trials=5, var_sr_trials=0.0) < 0.95
    rl, rsh = rng.normal(0.5, 1, 500), rng.normal(-0.5, 1, 500)
    assert oe.random_direction_test(rl, rsh, np.ones(500)) < 0.01
    assert oe.random_direction_test(rl, rsh, -np.ones(500)) > 0.99
    assert oe.breakeven_fee([0.2, 0.2], [400, 400]) == pytest.approx(0.0005)
