"""
alpaca_data.py
==============
Years of 5-minute US stock bars from the Alpaca Market Data API (v2), cached in the
same CSV format `research_strategies.download_universe` reads:
    <cache_dir>/ohlcv_<TICKER>_5m.csv   (Date index in America/New_York, OHLCV columns)

Free ("Basic") plan facts this module is built around:
- 200 requests / minute  -> ~0.32 s pause per request, retry with backoff on HTTP 429
- SIP (all US exchanges) history is available, but not the latest 15 minutes
  -> default end = now - 20 minutes
- some accounts only get IEX (one exchange, much smaller volume); a 403/422 about the
  subscription or SIP switches the request to feed="iex". Never mix feeds in one cache:
  volumes differ by an order of magnitude.

Only regular-session bars (09:30-15:59 New York time) are kept; Alpaca also returns
pre- and post-market bars. Keys come from APCA_API_KEY_ID / APCA_API_SECRET_KEY.
"""

import os
import time
from typing import Any, Dict, Iterable, Optional

import numpy as np
import pandas as pd

ALPACA_BARS_URL = "https://data.alpaca.markets/v2/stocks/bars"
NY_TZ = "America/New_York"
SESSION_OPEN = "09:30"
SESSION_LAST_BAR = "15:59"
REQUEST_PAUSE_S = 0.32  # 200 requests / minute on the free plan


class AlpacaError(RuntimeError):
    pass


def _keys(key: Optional[str], secret: Optional[str]) -> tuple:
    key = key or os.environ.get("APCA_API_KEY_ID")
    secret = secret or os.environ.get("APCA_API_SECRET_KEY")
    if not key or not secret:
        raise AlpacaError("Alpaca keys missing: set APCA_API_KEY_ID and APCA_API_SECRET_KEY "
                          "(free account at alpaca.markets -> API keys).")
    return key, secret


def _rfc3339(ts: Any) -> str:
    t = pd.Timestamp(ts)
    t = t.tz_localize(NY_TZ) if t.tzinfo is None else t
    return t.tz_convert("UTC").strftime("%Y-%m-%dT%H:%M:%SZ")


def _http_default():
    import requests

    return requests


def _is_feed_denied(status: int, text: str) -> bool:
    t = (text or "").lower()
    return status in (403, 422) and ("subscription" in t or "sip" in t)


def regular_session(df: pd.DataFrame) -> pd.DataFrame:
    """Keeps bars that start between 09:30 and 15:59 New York time."""
    if df.empty:
        return df
    return df.between_time(SESSION_OPEN, SESSION_LAST_BAR)


def fetch_alpaca_bars(
    symbol: str,
    start: Any,
    end: Any = None,
    key: Optional[str] = None,
    secret: Optional[str] = None,
    timeframe: str = "5Min",
    feed: str = "sip",
    adjustment: str = "split",
    http: Any = None,
    pause: float = REQUEST_PAUSE_S,
    max_retries: int = 6,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Downloads bars for one symbol, following next_page_token. Returns a DataFrame with
    Open/High/Low/Close/Volume, a tz-aware (America/New_York) index named "Date", only
    regular-session bars. df.attrs["feed"] holds the feed actually used.
    """
    key, secret = _keys(key, secret)
    http = http or _http_default()
    if end is None:
        end = pd.Timestamp.now(tz="UTC") - pd.Timedelta(minutes=20)  # free plan: no latest 15 min
    headers = {"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": secret}
    base = {"symbols": symbol, "timeframe": timeframe, "start": _rfc3339(start), "end": _rfc3339(end),
            "limit": 10000, "adjustment": adjustment, "sort": "asc"}

    rows: list = []
    token: Optional[str] = None
    retries = 0
    while True:
        params = {**base, "feed": feed}
        if token:
            params["page_token"] = token
        resp = http.get(ALPACA_BARS_URL, headers=headers, params=params, timeout=30)
        status = int(getattr(resp, "status_code", 200))
        if status == 429:
            retries += 1
            if retries > max_retries:
                raise AlpacaError(f"{symbol}: still rate-limited after {max_retries} retries")
            time.sleep(min(2.0 ** retries, 30.0))
            continue
        if _is_feed_denied(status, getattr(resp, "text", "")) and feed == "sip":
            if verbose:
                print(f"  {symbol}: SIP not available on this subscription -> switching to IEX")
            feed, token, rows, retries = "iex", None, [], 0
            continue
        if status != 200:
            raise AlpacaError(f"{symbol}: HTTP {status}: {str(getattr(resp, 'text', ''))[:200]}")
        retries = 0
        payload = resp.json() or {}
        bars = payload.get("bars") or {}
        if isinstance(bars, dict):
            bars = bars.get(symbol) or []
        rows.extend(bars)
        token = payload.get("next_page_token")
        if pause:
            time.sleep(pause)
        if not token:
            break

    if not rows:
        df = pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"],
                          index=pd.DatetimeIndex([], tz=NY_TZ, name="Date"), dtype=float)
    else:
        raw = pd.DataFrame(rows)
        df = pd.DataFrame({
            "Open": raw["o"].astype(float), "High": raw["h"].astype(float), "Low": raw["l"].astype(float),
            "Close": raw["c"].astype(float), "Volume": raw["v"].astype(float),
        })
        df.index = pd.DatetimeIndex(pd.to_datetime(raw["t"], utc=True)).tz_convert(NY_TZ)
        df.index.name = "Date"
        df = regular_session(df[~df.index.duplicated(keep="last")].sort_index())
    df.attrs["feed"] = feed
    return df


def resolve_feed(key: Optional[str] = None, secret: Optional[str] = None, feed: str = "sip",
                 http: Any = None, probe_symbol: str = "SPY") -> str:
    """Feed this account can actually use (one tiny request); 'sip' falls back to 'iex'."""
    end = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=7)
    df = fetch_alpaca_bars(probe_symbol, end - pd.Timedelta(days=1), end, key=key, secret=secret,
                           feed=feed, http=http, pause=0)
    return str(df.attrs.get("feed", feed))


def _read_cache(path: str) -> Optional[pd.DataFrame]:
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path, index_col=0)
    if df.empty:
        return None
    df.index = pd.to_datetime(df.index, utc=True).tz_convert(NY_TZ)
    df.index.name = "Date"
    return df[["Open", "High", "Low", "Close", "Volume"]].astype(float)


def split_detected(cached: pd.DataFrame, fresh: pd.DataFrame, tolerance: float = 0.002) -> bool:
    """
    True when the overlapping closes differ by more than `tolerance` (median relative
    difference). With adjustment="split", a new split rescales all earlier bars, so the
    cached history no longer matches and the ticker must be re-downloaded in full.
    """
    common = cached.index.intersection(fresh.index)
    if len(common) == 0:
        return False
    a = cached.loc[common, "Close"].to_numpy(dtype=float)
    b = fresh.loc[common, "Close"].to_numpy(dtype=float)
    rel = np.abs(b - a) / np.where(a != 0, np.abs(a), np.nan)
    return bool(np.nanmedian(rel) > tolerance)


def update_cache_from_alpaca(
    tickers: Iterable[str],
    cache_dir: str,
    start: Any = "2016-01-01",
    end: Any = None,
    key: Optional[str] = None,
    secret: Optional[str] = None,
    feed: str = "sip",
    adjustment: str = "split",
    http: Any = None,
    overlap_days: int = 3,
    split_tolerance: float = 0.002,
    pause: float = REQUEST_PAUSE_S,
    verbose: bool = True,
) -> Dict[str, pd.DataFrame]:
    """
    Incrementally fills <cache_dir>/ohlcv_<TICKER>_5m.csv:
      - no cache: download from `start`
      - cache: download from (last cached date - overlap_days), merge without duplicates
      - split detected on the overlap: re-download that ticker from `start`
    Today's still-running session is never written to the CSV, but it IS included in
    the returned frames (for live signals). Failed tickers fall back to their cache.
    Each returned frame has df.attrs["feed"].
    """
    from research_strategies import split_incomplete_session

    key, secret = _keys(key, secret)
    os.makedirs(cache_dir, exist_ok=True)
    out: Dict[str, pd.DataFrame] = {}
    kw = dict(end=end, key=key, secret=secret, adjustment=adjustment, http=http, pause=pause)
    for t in tickers:
        path = os.path.join(cache_dir, f"ohlcv_{t}_5m.csv")
        cached = _read_cache(path)
        try:
            if cached is None:
                merged = fetch_alpaca_bars(t, start, feed=feed, **kw)
                action = "full"
            else:
                since = pd.Timestamp(cached.index[-1].date()) - pd.Timedelta(days=overlap_days)
                fresh = fetch_alpaca_bars(t, since, feed=feed, **kw)
                if split_detected(cached, fresh, split_tolerance):
                    merged = fetch_alpaca_bars(t, start, feed=feed, **kw)
                    action = "split -> full re-download"
                else:
                    merged = pd.concat([cached, fresh])
                    merged = merged[~merged.index.duplicated(keep="last")].sort_index()
                    merged.attrs["feed"] = fresh.attrs.get("feed", feed)
                    action = "incremental"
        except Exception as e:  # noqa: BLE001
            if verbose:
                print(f"  {t:<6} Alpaca failed ({type(e).__name__}: {str(e)[:120]})"
                      f"{' -> using cache' if cached is not None else ' -> skipped'}")
            if cached is None:
                continue
            merged, action = cached, "cache only"
            merged.attrs["feed"] = feed
        used_feed = merged.attrs.get("feed", feed)
        if used_feed != feed:
            feed = used_feed  # keep the rest of the universe on the same feed
        complete, today = split_incomplete_session(merged)
        if len(complete):
            complete.to_csv(path)
        full = pd.concat([complete, today]) if today is not None and len(today) else complete
        full.attrs["feed"] = used_feed
        if len(full):
            out[t] = full
        if verbose and len(full):
            print(f"  {t:<6} {len(full):>7} bars  {full.index[0].date()} -> {full.index[-1].date()}  "
                  f"({action}, feed {used_feed})")
    return out
