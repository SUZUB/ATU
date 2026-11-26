"""Offline stand-ins for Alpaca and Stooq, plus synthetic daily bars with known corporate actions."""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from atu2.core.calendars import trading_days
from atu2.core.credentials import AlpacaCredentials
from atu2.data.alpaca_client import BAR_COLUMNS, AlpacaDailyClient

NY = "America/New_York"
FIXTURES = Path(__file__).parent / "fixtures"
CREDS = AlpacaCredentials(key_id="test-key-id-000", secret="test-secret-000")  # pragma: allowlist secret
# Friday 2024-06-14, 22:00 New York: that day's bar is final.
NOW = pd.Timestamp("2024-06-14 22:00", tz=NY)


@dataclass
class FakeResponse:
    status_code: int
    payload: Any = None
    text: str = ""

    def json(self) -> Any:
        return self.payload


def make_bars(
    start: str = "2023-01-03",
    n_days: int = 300,
    *,
    seed: int = 0,
    price: float = 100.0,
    split: tuple[str, float] | None = None,
    dividends: tuple[tuple[str, float], ...] = (),
) -> pd.DataFrame:
    """
    Daily bars on NYSE days with raw OHLCV and a total-return adj_close.
    split=("2023-06-01", 2.0): raw prices divide by 2 from that date (2-for-1).
    dividends=(("2023-03-15", 0.5),): cash dividend per share on the ex-date.
    """
    dates = pd.DatetimeIndex(
        trading_days(start, pd.Timestamp(start) + pd.Timedelta(days=n_days * 2))[:n_days], name="date"
    )
    rng = np.random.default_rng(seed)
    true_close = price * np.exp(np.cumsum(rng.normal(0.0003, 0.01, n_days)))  # economic value per original share
    ratio = np.ones(n_days)
    if split:
        ratio[dates >= pd.Timestamp(split[0])] = split[1]
    close = true_close / ratio
    open_ = np.concatenate([[close[0]], close[:-1]]) * np.exp(rng.normal(0, 0.002, n_days))
    open_[ratio != np.concatenate([[ratio[0]], ratio[:-1]])] = close[ratio != np.concatenate([[ratio[0]], ratio[:-1]])]
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.003, n_days)))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.003, n_days)))
    volume = np.round(1e6 * np.exp(rng.normal(0, 0.2, n_days)))
    # adjustment factor: splits make earlier raw prices comparable, dividends scale earlier prices down
    factor = 1.0 / ratio[-1] * ratio  # split part: adj = raw * ratio / ratio_last
    for ex_date, amount in dividends:
        i = int(np.searchsorted(dates, pd.Timestamp(ex_date)))
        prev_close = close[i - 1]
        factor[:i] *= 1 - amount / prev_close
    adj_close = close * factor
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume, "adj_close": adj_close}, index=dates
    )[list(BAR_COLUMNS)]


@dataclass
class FakeAlpaca:
    """
    Serves /v2/stocks/bars like Alpaca: multi-symbol 'bars' dict, next_page_token paging,
    adjustment=raw|all, feed=sip|iex. Bars are stamped at midnight New York time in UTC.
    """

    data: dict[str, pd.DataFrame]
    page_size: int = 50
    rate_limited: int = 0  # answer HTTP 429 this many times first
    sip_denied: bool = False
    fail_status: int | None = None
    calls: list[dict[str, Any]] = field(default_factory=list)
    headers_seen: list[dict[str, str]] = field(default_factory=list)

    def get(self, url: str, *, headers: dict[str, str], params: dict[str, Any], timeout: float) -> FakeResponse:
        self.calls.append(dict(params))
        self.headers_seen.append(dict(headers))
        if self.rate_limited > 0:
            self.rate_limited -= 1
            return FakeResponse(429, text="too many requests")
        if self.fail_status:
            return FakeResponse(self.fail_status, text="internal error")
        if self.sip_denied and params["feed"] == "sip":
            return FakeResponse(403, text=(FIXTURES / "alpaca_sip_denied.json").read_text(encoding="utf-8"))
        sym = params["symbols"]
        df = self.data.get(sym)
        if df is None:
            return FakeResponse(200, {"bars": {}, "next_page_token": None})
        stamps = pd.DatetimeIndex(df.index).tz_localize(NY).tz_convert("UTC")
        lo, hi = pd.Timestamp(params["start"]), pd.Timestamp(params["end"])
        sel = df[(stamps >= lo) & (stamps <= hi)]
        sel_stamps = stamps[(stamps >= lo) & (stamps <= hi)]
        offset = int(params.get("page_token") or 0)
        chunk = sel.iloc[offset : offset + self.page_size]
        chunk_stamps = sel_stamps[offset : offset + self.page_size]
        f = 1.0 if params["adjustment"] == "raw" else 0.0
        bars = []
        for ts, (_, row) in zip(chunk_stamps, chunk.iterrows(), strict=True):
            k = 1.0 if f else row["adj_close"] / row["close"]
            bars.append(
                {
                    "t": ts.strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "o": row["open"] * k,
                    "h": row["high"] * k,
                    "l": row["low"] * k,
                    "c": row["close"] * k,
                    "v": row["volume"],
                    "n": 1000,
                    "vw": row["close"] * k,
                }
            )
        nxt = offset + self.page_size
        token = str(nxt) if nxt < len(sel) else None
        return FakeResponse(200, {"bars": {sym: bars} if bars else {}, "next_page_token": token})


@dataclass
class RecordedHttp:
    """Replays the static fixture files: raw pages by page_token, one adjusted page."""

    calls: list[dict[str, Any]] = field(default_factory=list)

    def get(self, url: str, *, headers: dict[str, str], params: dict[str, Any], timeout: float) -> FakeResponse:
        self.calls.append(dict(params))
        if params["adjustment"] == "all":
            name = "alpaca_spy_all_p1.json"
        else:
            name = "alpaca_spy_raw_p2.json" if params.get("page_token") else "alpaca_spy_raw_p1.json"
        return FakeResponse(200, json.loads((FIXTURES / name).read_text(encoding="utf-8")))


@dataclass
class FakeStooq:
    """Second source that returns given closes, optionally perturbed."""

    closes: dict[str, pd.Series]
    name: str = "fake-stooq"

    def daily_closes(self, ticker: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
        s = self.closes[ticker]
        return s[(s.index >= start) & (s.index <= end)]


def client(fake: Any, now: pd.Timestamp = NOW, **kw: Any) -> AlpacaDailyClient:
    """Client wired to a fake server, a fixed clock and a no-op sleep (sleeps are recorded)."""
    sleeps: list[float] = []
    c = AlpacaDailyClient(credentials=CREDS, http=fake, sleep=sleeps.append, clock=lambda: now, pause_seconds=0.0, **kw)
    c.sleeps = sleeps  # type: ignore[attr-defined]
    return c
