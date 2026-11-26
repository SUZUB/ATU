"""Alpaca daily bars (FR-01): split and dividend adjusted, chunked, paced, retried on HTTP 429.

Each date range is requested twice: adjustment="raw" gives open/high/low/close/volume, and
adjustment="all" (splits and dividends) gives adj_close, the total-return close used for
returns. Long ranges are split into chunks of `chunk_years`; every chunk follows
next_page_token. HTTP 429 backs off exponentially (2, 4, 8, ... s, at most 30 s); after
`max_retries` RateLimitError is raised. If the account cannot use the SIP feed (HTTP 403/422
mentioning the subscription or SIP) and the fallback is allowed, the client switches to IEX
for the rest of its life and reports it in `feed`; the caller must keep feeds in separate caches.

Bars dated after the last finished session (`last_complete_session`) are dropped, so today's
unfinished bar is never returned and never cached. Keys come from atu2.core.credentials only
and are sent as request headers, never logged.

Ideas reused from the legacy alpaca_data.py (pagination, pacing, 429 backoff, SIP->IEX
fallback); the code is new and daily-only.
"""

import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol

import pandas as pd

from atu2.core.calendars import is_trading_day, previous_trading_day
from atu2.core.credentials import AlpacaCredentials
from atu2.core.errors import DataSourceError, RateLimitError
from atu2.core.logging import get_logger

ALPACA_BARS_URL = "https://data.alpaca.markets/v2/stocks/bars"
NY_TZ = "America/New_York"
PAGE_LIMIT = 10_000
FREE_PLAN_DELAY = pd.Timedelta(minutes=20)  # free plan: no SIP data for the latest 15 minutes
BAR_COLUMNS: tuple[str, ...] = ("open", "high", "low", "close", "volume", "adj_close")

log = get_logger("data.alpaca")


class HttpResponse(Protocol):
    """The part of a requests.Response the client uses."""

    status_code: int
    text: str

    def json(self) -> Any:
        """Decoded JSON body."""


class HttpClient(Protocol):
    """Anything with a requests-style get(); tests pass a recorded-response fake."""

    def get(self, url: str, *, headers: Mapping[str, str], params: Mapping[str, Any], timeout: float) -> HttpResponse:
        """Performs a GET request."""


def requests_client() -> HttpClient:  # pragma: no cover - real network only
    """The real HTTP client (requests), imported only when a download actually runs."""
    import requests  # noqa: PLC0415  (imported lazily: tests never need it)

    return requests  # type: ignore[return-value]


def last_complete_session(now: pd.Timestamp | None = None, settle_time: str = "20:00") -> pd.Timestamp:
    """
    Latest NYSE session whose daily bar is final at `now`: today once New York time has
    passed `settle_time` on a trading day, otherwise the previous trading day.
    """
    t = pd.Timestamp.now(tz="UTC") if now is None else pd.Timestamp(now)
    t = t.tz_localize(NY_TZ) if t.tzinfo is None else t.tz_convert(NY_TZ)
    today = t.tz_localize(None).normalize()
    settle = pd.Timestamp(f"{today.date()} {settle_time}")
    if is_trading_day(today) and t.tz_localize(None) >= settle:
        return today
    return previous_trading_day(today)


def _rfc3339(ts: pd.Timestamp) -> str:
    return ts.tz_convert("UTC").strftime("%Y-%m-%dT%H:%M:%SZ")


def _is_feed_denied(status: int, text: str) -> bool:
    t = (text or "").lower()
    return status in (403, 422) and ("subscription" in t or "sip" in t)


@dataclass
class AlpacaDailyClient:
    """Downloads daily bars for one symbol at a time. `feed` changes to "iex" after a fallback."""

    credentials: AlpacaCredentials
    feed: str = "sip"
    allow_iex_fallback: bool = True
    chunk_years: int = 5
    pause_seconds: float = 0.35
    max_retries: int = 6
    settle_time: str = "20:00"
    http: HttpClient | None = None
    sleep: Callable[[float], None] = time.sleep
    clock: Callable[[], pd.Timestamp] = field(default=lambda: pd.Timestamp.now(tz="UTC"))
    requests_made: int = 0

    def _headers(self) -> dict[str, str]:
        return {"APCA-API-KEY-ID": self.credentials.key_id, "APCA-API-SECRET-KEY": self.credentials.secret}

    def _get_pages(self, symbol: str, start: pd.Timestamp, end: pd.Timestamp, adjustment: str) -> list[dict[str, Any]]:
        http = self.http or requests_client()
        rows: list[dict[str, Any]] = []
        token: str | None = None
        retries = 0
        while True:
            params: dict[str, Any] = {
                "symbols": symbol,
                "timeframe": "1Day",
                "start": _rfc3339(start),
                "end": _rfc3339(end),
                "adjustment": adjustment,
                "feed": self.feed,
                "limit": PAGE_LIMIT,
                "sort": "asc",
            }
            if token:
                params["page_token"] = token
            resp = http.get(ALPACA_BARS_URL, headers=self._headers(), params=params, timeout=30)
            self.requests_made += 1
            status = int(resp.status_code)
            if status == 429:
                retries += 1
                if retries > self.max_retries:
                    raise RateLimitError(f"{symbol}: still rate-limited after {self.max_retries} retries")
                self.sleep(min(2.0**retries, 30.0))
                continue
            if _is_feed_denied(status, resp.text) and self.feed == "sip" and self.allow_iex_fallback:
                log.warning(
                    "SIP feed not available on this account; switching to IEX", extra={"fields": {"symbol": symbol}}
                )
                self.feed, token, rows, retries = "iex", None, [], 0
                continue
            if status != 200:
                raise DataSourceError(f"{symbol}: HTTP {status} from Alpaca: {str(resp.text)[:200]}")
            retries = 0
            payload = resp.json() or {}
            bars = payload.get("bars") or {}
            if isinstance(bars, dict):
                bars = bars.get(symbol) or []
            if not isinstance(bars, list):
                raise DataSourceError(f"{symbol}: unexpected 'bars' payload of type {type(bars).__name__}")
            rows.extend(bars)
            token = payload.get("next_page_token")
            if self.pause_seconds:
                self.sleep(self.pause_seconds)
            if not token:
                return rows

    def _chunks(self, start: pd.Timestamp, end: pd.Timestamp) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
        out = []
        s = start
        while s <= end:
            e = min(s + pd.DateOffset(years=self.chunk_years) - pd.Timedelta(days=1), end)
            out.append((s, e))
            s = e + pd.Timedelta(days=1)
        return out

    def _frame(self, symbol: str, rows: list[dict[str, Any]], fields: dict[str, str]) -> pd.DataFrame:
        if not rows:
            return pd.DataFrame(columns=list(fields.values()), index=pd.DatetimeIndex([], name="date"), dtype=float)
        raw = pd.DataFrame(rows)
        missing = [k for k in ("t", *fields) if k not in raw.columns]
        if missing:
            raise DataSourceError(f"{symbol}: bars are missing fields {missing}")
        dates = pd.DatetimeIndex(pd.to_datetime(raw["t"], utc=True)).tz_convert(NY_TZ).tz_localize(None).normalize()
        df = pd.DataFrame({name: pd.to_numeric(raw[k], errors="coerce").to_numpy() for k, name in fields.items()})
        df.index = pd.DatetimeIndex(dates, name="date")
        return df[~df.index.duplicated(keep="last")].sort_index()

    def _fetch_chunks(
        self, symbol: str, start_date: pd.Timestamp, end_date: pd.Timestamp
    ) -> tuple[list[pd.DataFrame], list[pd.DataFrame]]:
        latest_allowed = (self.clock() - FREE_PLAN_DELAY).tz_convert(NY_TZ)
        raw_parts, adj_parts = [], []
        for s, e in self._chunks(start_date, end_date):
            s_ts = s.tz_localize(NY_TZ)
            e_ts = min((e + pd.Timedelta(hours=23, minutes=59)).tz_localize(NY_TZ), latest_allowed)
            if e_ts < s_ts:
                continue
            raw_rows = self._get_pages(symbol, s_ts, e_ts, "raw")
            raw_feed = self.feed
            adj_rows = self._get_pages(symbol, s_ts, e_ts, "all")
            if self.feed != raw_feed:  # fell back during the adjusted pass: refetch raw on the new feed
                raw_rows = self._get_pages(symbol, s_ts, e_ts, "raw")
            raw = self._frame(symbol, raw_rows, {"o": "open", "h": "high", "l": "low", "c": "close", "v": "volume"})
            adj = self._frame(symbol, adj_rows, {"c": "adj_close"})
            raw.attrs["feed"] = adj.attrs["feed"] = self.feed
            raw_parts.append(raw)
            adj_parts.append(adj)
        return raw_parts, adj_parts

    def fetch(self, symbol: str, start: str | pd.Timestamp, end: str | pd.Timestamp | None = None) -> pd.DataFrame:
        """
        Daily bars for `symbol` from `start` to `end` (dates, inclusive; default and cap: the
        last complete session). Columns open, high, low, close, volume (raw) and adj_close
        (split and dividend adjusted); index "date" (timezone-naive NYSE dates).
        df.attrs["feed"] is the feed that served the data.
        """
        last_done = last_complete_session(self.clock(), self.settle_time)
        end_date = last_done if end is None else min(pd.Timestamp(end).normalize(), last_done)
        start_date = pd.Timestamp(start).normalize()
        if end_date < start_date:
            out = pd.DataFrame(columns=list(BAR_COLUMNS), index=pd.DatetimeIndex([], name="date"), dtype=float)
            out.attrs["feed"] = self.feed
            return out
        raw_parts, adj_parts = self._fetch_chunks(symbol, start_date, end_date)
        feed_used = self.feed
        if any(p.attrs.get("feed") != feed_used for p in raw_parts + adj_parts):
            # the SIP -> IEX fallback happened part-way: start again so one feed serves every row
            raw_parts, adj_parts = self._fetch_chunks(symbol, start_date, end_date)
        raw_df = pd.concat(raw_parts) if raw_parts else self._frame(symbol, [], {"c": "close"})
        adj_df = pd.concat(adj_parts) if adj_parts else self._frame(symbol, [], {"c": "adj_close"})
        if not raw_df.index.equals(adj_df.index):
            raise DataSourceError(f"{symbol}: raw and adjusted bars cover different dates")
        out = raw_df.join(adj_df)[list(BAR_COLUMNS)]
        out = out[out.index <= end_date]
        out.attrs["feed"] = self.feed
        log.info(
            "fetched",
            extra={"fields": {"symbol": symbol, "rows": len(out), "feed": self.feed, "through": str(end_date.date())}},
        )
        return out
