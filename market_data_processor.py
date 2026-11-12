"""
MarketDataProcessor Module
===========================
Data ingestion, technical-indicator engineering, leak-free feature scaling and
sentiment alignment for the ATU intraday RL trading system.

Look-ahead protections:
-----------------------
1. The MinMaxScaler is fit ONLY on the chronological training slice (first
   `train_ratio` of rows, default 80%). The remaining test rows are transformed
   with that already-fitted scaler, so test-set extremes never influence the
   training features. Test values may therefore fall slightly outside [0, 1];
   that is expected and correct.
2. Daily sentiment is lagged: a headline dated D only becomes visible to bars on
   the first trading day on or after D + `sentiment_lag_days` (default 1), so
   afternoon news cannot leak into same-day morning decisions.

Raw prices:
-----------
Unscaled open/high/low/close prices are preserved in `raw_open`, `raw_high`,
`raw_low` and `raw_close`. The trading environment uses them for P&L, balance and
stop-loss fills (including gap slippage), and excludes them from the observation.

Recent news:
------------
`fetch_recent_headlines()` pulls free, recent headlines (Google News RSS searched in
date windows + yfinance `get_news`) so FinBERT sentiment covers the same 60-day
window as yfinance 5-minute candles. Headline dates are taken in the EXCHANGE's
local timezone before the one-day sentiment lag is applied.
"""

from email.utils import parsedate_to_datetime
from pathlib import Path
import re
import time
from typing import Any, Dict, List, Optional, Sequence, Union
import urllib.error
import urllib.parse
import urllib.request
import warnings
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import yfinance as yf

# Technical Analysis library imports
from ta.momentum import RSIIndicator
from ta.trend import ADXIndicator, MACD
from ta.volatility import AverageTrueRange, BollingerBands


# Unscaled price columns consumed by ConstrainedTradingEnv (P&L and stop-loss fills).
RAW_PRICE_COLUMN: str = "raw_close"
RAW_HIGH_COLUMN: str = "raw_high"
RAW_LOW_COLUMN: str = "raw_low"
RAW_OPEN_COLUMN: str = "raw_open"
RAW_PRICE_SOURCES = {
    RAW_PRICE_COLUMN: "Close",
    RAW_HIGH_COLUMN: "High",
    RAW_LOW_COLUMN: "Low",
    RAW_OPEN_COLUMN: "Open",
}

# Columns MinMax-scaled to [0, 1] using statistics from the training slice only.
DEFAULT_SCALE_COLUMNS: List[str] = [
    "Open", "High", "Low", "Close", "Volume",
    "MACD", "MACD_Signal", "MACD_Diff",
    "ATR",
    "BB_High", "BB_Mid", "BB_Low",
]

# Oscillators bounded to [0, 100] by definition: divided by 100 (a fixed transform,
# no fitted statistics, so no leakage) to share the same 0-1 range.
BOUNDED_0_100_COLUMNS: List[str] = ["RSI", "ADX"]

DEFAULT_TRAIN_RATIO: float = 0.8


# =====================================================================
# yfinance HTTP session (robust browser impersonation)
# =====================================================================
# yfinance always asks curl_cffi for the alias "chrome", which resolves to the NEWEST
# profile the curl_cffi Python layer knows (e.g. chrome152). If the loaded native
# library is older (common on Colab after in-place upgrades), every request fails with
# ImpersonateError, Yahoo's cookie/crumb step is skipped and downloads come back empty
# ("possibly delisted"). We probe profiles from newest to oldest and keep the first one
# the installed library actually supports.
_YF_IMPERSONATE_CANDIDATES = (
    "chrome", "chrome146", "chrome145", "chrome142", "chrome136", "chrome133a", "chrome131",
    "chrome124", "chrome123", "chrome120", "chrome116", "chrome110", "edge101", "safari",
)
_YF_SESSION_STATE: Dict[str, Any] = {}


_BROWSER_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
)


def get_yfinance_session(probe_url: str = "https://fc.yahoo.com") -> Optional[Any]:
    """
    Returns an HTTP session for yfinance, cached for the process:
      1. a curl_cffi Session with the newest browser profile the installed build supports;
      2. otherwise a plain requests.Session with a browser User-Agent (yfinance accepts it),
         so a broken curl_cffi install never reaches yfinance's own (failing) default;
      3. None only if neither library is importable.
    """
    if "session" in _YF_SESSION_STATE:
        return _YF_SESSION_STATE["session"]

    session, profile = None, None
    try:
        from curl_cffi import requests as curl_requests
    except Exception as e:  # noqa: BLE001 - ImportError or a broken native library
        print(f"  - curl_cffi unavailable ({type(e).__name__}: {e}).")
        curl_requests = None

    if curl_requests is not None:
        rejected: List[str] = []
        first_error = ""
        for candidate in _YF_IMPERSONATE_CANDIDATES:
            try:
                trial = curl_requests.Session(impersonate=candidate)
                trial.get(probe_url, timeout=10)
            except Exception as e:  # noqa: BLE001 - classify by name across curl_cffi versions
                if "Impersonat" in type(e).__name__ or "not supported" in str(e):
                    rejected.append(candidate)
                    first_error = first_error or f"{type(e).__name__}: {e}"
                    continue
                # Profile accepted; failure was ordinary network trouble (keep the profile)
            session, profile = trial, candidate
            break
        if rejected:
            print(f"  - curl_cffi rejected browser profiles {rejected}; first error: {first_error}")

    if session is None:
        try:
            import requests

            session = requests.Session()
            session.headers["User-Agent"] = _BROWSER_USER_AGENT
            profile = "requests (curl_cffi fallback)"
        except ImportError:
            session, profile = None, "yfinance default"
    print(f"  - Yahoo Finance HTTP session: {profile}")

    _YF_SESSION_STATE.update(session=session, profile=profile)
    return session


def _yf_download(**kwargs: Any) -> pd.DataFrame:
    """
    Calls yfinance's own download function (yfinance.multi.download), not the
    `yf.download` attribute, so notebook monkey-patches of yf.download cannot wrap or
    recurse into it. `session` is only passed when one exists.
    """
    from yfinance.multi import download as _download

    session = get_yfinance_session()
    if session is not None:
        kwargs["session"] = session
    return _download(**kwargs)


def mirror_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """
    Price-mirrored copy of an OHLCV frame for training augmentation: every log return is
    negated (p' = p0^2 / p, with p0 the first close), so an uptrend becomes an identical
    downtrend. High and Low swap roles; Volume is unchanged. An agent trained on both the
    original and the mirror cannot profit from a fixed long or short bias.
    """
    pivot_sq = float(df["Close"].iloc[0]) ** 2
    mirrored = df.copy()
    mirrored["Open"] = pivot_sq / df["Open"]
    mirrored["Close"] = pivot_sq / df["Close"]
    mirrored["High"] = pivot_sq / df["Low"]
    mirrored["Low"] = pivot_sq / df["High"]
    return mirrored


def compute_split_index(n_rows: int, train_ratio: float = DEFAULT_TRAIN_RATIO) -> int:
    """
    Chronological train/test boundary. Training pipelines MUST split with this
    same index so that the scaler's fit window matches the training set exactly.
    """
    if not 0.0 < train_ratio <= 1.0:
        raise ValueError(f"train_ratio must be in (0, 1], got {train_ratio}")
    return int(n_rows * train_ratio)


class MarketDataProcessor:
    """
    Ingests and transforms financial market data into structured state vectors
    for Reinforcement Learning agents.

    Pipeline: fetch OHLCV -> technical indicators -> clean -> leak-free scaling.
    """

    def __init__(
        self,
        scaler: Optional[MinMaxScaler] = None,
        scale_columns: Optional[List[str]] = None,
        train_ratio: float = DEFAULT_TRAIN_RATIO,
    ) -> None:
        """
        Args:
            scaler: Optional pre-configured MinMaxScaler. Defaults to MinMaxScaler((0, 1)).
            scale_columns: Columns to MinMax-scale. Defaults to DEFAULT_SCALE_COLUMNS
                           (OHLCV, MACD family, ATR, Bollinger Bands).
            train_ratio: Fraction of rows (chronological head) the scaler is fit on.
        """
        self.scaler: MinMaxScaler = scaler if scaler is not None else MinMaxScaler(feature_range=(0, 1))
        self.scale_columns: List[str] = (
            list(scale_columns) if scale_columns is not None else list(DEFAULT_SCALE_COLUMNS)
        )
        compute_split_index(1, train_ratio)  # validates the ratio
        self.train_ratio: float = float(train_ratio)
        self._is_fitted: bool = False
        self.fit_end_index: Optional[int] = None

    def fetch_data(
        self,
        ticker: str,
        start: Optional[Union[str, pd.Timestamp]] = None,
        end: Optional[Union[str, pd.Timestamp]] = None,
        period: str = "2y",
        interval: str = "1d",
    ) -> pd.DataFrame:
        """
        Downloads historical OHLCV market data for a given ticker and date range.

        Returns:
            pd.DataFrame: Raw OHLCV DataFrame with columns ['Open', 'High', 'Low', 'Close', 'Volume'].
        """
        download_kwargs = {
            "tickers": ticker,
            "interval": interval,
            "auto_adjust": False,
            "progress": False,
        }

        if start is not None:
            download_kwargs["start"] = start
            if end is not None:
                download_kwargs["end"] = end
        else:
            download_kwargs["period"] = period

        df: pd.DataFrame = _yf_download(**download_kwargs)

        if df.empty:
            raise ValueError(
                f"No data returned for ticker '{ticker}' with parameters: {download_kwargs}. "
                f"If the log shows 'possibly delisted' or a Cookie/crumb ImpersonateError, Yahoo "
                f"rejected the request (browser profile: {_YF_SESSION_STATE.get('profile')}); the "
                f"symbol itself is usually fine. Check the yfinance smoke-test line printed by Cell 1."
            )

        # Flatten MultiIndex columns introduced in recent yfinance versions
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df.columns = [str(col).strip().title() for col in df.columns]
        df.columns.name = None

        required_cols = ["Open", "High", "Low", "Close", "Volume"]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise KeyError(
                f"Fetched data missing required OHLCV columns: {missing_cols}. "
                f"Available columns: {list(df.columns)}"
            )

        df = df[required_cols].astype(np.float64).sort_index()
        df.index = pd.to_datetime(df.index)
        df.index.name = "Date"
        return df

    def add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Computes causal (backward-looking) indicators with the `ta` library:
        RSI(14), MACD(12, 26, 9) + signal + histogram, ATR(14), Bollinger Bands(20, 2), ADX(14).
        """
        df_feat = df.copy()

        df_feat["RSI"] = RSIIndicator(close=df_feat["Close"], window=14, fillna=False).rsi()

        macd_indicator = MACD(
            close=df_feat["Close"],
            window_slow=26,
            window_fast=12,
            window_sign=9,
            fillna=False,
        )
        df_feat["MACD"] = macd_indicator.macd()
        df_feat["MACD_Signal"] = macd_indicator.macd_signal()
        df_feat["MACD_Diff"] = macd_indicator.macd_diff()

        df_feat["ATR"] = AverageTrueRange(
            high=df_feat["High"],
            low=df_feat["Low"],
            close=df_feat["Close"],
            window=14,
            fillna=False,
        ).average_true_range()

        bb_indicator = BollingerBands(close=df_feat["Close"], window=20, window_dev=2, fillna=False)
        df_feat["BB_High"] = bb_indicator.bollinger_hband()
        df_feat["BB_Mid"] = bb_indicator.bollinger_mavg()
        df_feat["BB_Low"] = bb_indicator.bollinger_lband()

        df_feat["ADX"] = ADXIndicator(
            high=df_feat["High"],
            low=df_feat["Low"],
            close=df_feat["Close"],
            window=14,
            fillna=False,
        ).adx()

        return df_feat

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Forward-fills sporadic gaps (causal) and drops the indicator burn-in rows.
        """
        return df.copy().ffill().dropna()

    def normalize_features(
        self,
        df: pd.DataFrame,
        columns_to_scale: Optional[List[str]] = None,
        fit_scaler: bool = True,
        train_ratio: Optional[float] = None,
    ) -> pd.DataFrame:
        """
        Leak-free normalization.

        - Preserves unscaled OHLC in `raw_open`/`raw_high`/`raw_low`/`raw_close`
          (for environment P&L and stop-loss fills).
        - fit_scaler=True: fits the MinMaxScaler on rows [0, split_index) ONLY,
          then transforms ALL rows (train + test) with that fitted scaler.
        - fit_scaler=False: transforms all rows with the previously fitted scaler
          (e.g. scoring a brand-new out-of-sample period).
        - RSI and ADX are divided by 100.

        Args:
            df: Cleaned feature DataFrame (chronologically sorted).
            columns_to_scale: Columns to MinMax-scale. Defaults to `self.scale_columns`.
            fit_scaler: Whether to fit the scaler on the training slice.
            train_ratio: Override for `self.train_ratio`.
        """
        target_cols = columns_to_scale if columns_to_scale is not None else self.scale_columns
        missing = [col for col in target_cols if col not in df.columns]
        if missing:
            raise KeyError(f"Columns specified for scaling not found in DataFrame: {missing}")

        normalized_df = df.copy()

        # Preserve unscaled prices before any transformation
        if RAW_PRICE_COLUMN not in normalized_df.columns and "Close" not in normalized_df.columns:
            raise KeyError("A 'Close' column is required to preserve raw prices.")
        for raw_col, source_col in RAW_PRICE_SOURCES.items():
            if raw_col not in normalized_df.columns and source_col in normalized_df.columns:
                normalized_df[raw_col] = df[source_col].astype(np.float64)

        if fit_scaler:
            ratio = self.train_ratio if train_ratio is None else float(train_ratio)
            fit_end = compute_split_index(len(normalized_df), ratio)
            if fit_end < 2:
                raise ValueError(
                    f"Training slice too small to fit the scaler ({fit_end} rows from "
                    f"{len(normalized_df)} total at train_ratio={ratio})."
                )
            self.scaler.fit(normalized_df[target_cols].iloc[:fit_end])
            self._is_fitted = True
            self.fit_end_index = fit_end
        elif not self._is_fitted:
            raise RuntimeError(
                "MinMaxScaler is not fitted yet. Call normalize_features with fit_scaler=True "
                "on data whose head is the training period first."
            )

        normalized_df[target_cols] = self.scaler.transform(normalized_df[target_cols])

        for col in BOUNDED_0_100_COLUMNS:
            if col in normalized_df.columns:
                normalized_df[col] = df[col] / 100.0

        return normalized_df

    def process(
        self,
        ticker: str,
        start: Optional[Union[str, pd.Timestamp]] = None,
        end: Optional[Union[str, pd.Timestamp]] = None,
        period: str = "2y",
        interval: str = "1d",
        fit_scaler: bool = True,
        train_ratio: Optional[float] = None,
    ) -> pd.DataFrame:
        """
        Download -> indicators -> clean -> leak-free normalization.
        """
        raw_df = self.fetch_data(ticker=ticker, start=start, end=end, period=period, interval=interval)
        return self.process_dataframe(raw_df, fit_scaler=fit_scaler, train_ratio=train_ratio)

    def process_dataframe(
        self,
        df: pd.DataFrame,
        fit_scaler: bool = True,
        train_ratio: Optional[float] = None,
    ) -> pd.DataFrame:
        """
        Processes an existing OHLCV DataFrame without re-downloading.
        """
        engineered_df = self.add_technical_indicators(df.sort_index())
        cleaned_df = self.clean_data(engineered_df)
        return self.normalize_features(cleaned_df, fit_scaler=fit_scaler, train_ratio=train_ratio)


# =====================================================================
# Sentiment alignment (lagged to prevent same-day look-ahead)
# =====================================================================
def _to_naive_calendar_dates(values) -> pd.Series:
    """Parses dates/timestamps to timezone-naive midnight datetimes (local wall-clock date)."""
    parsed = pd.to_datetime(pd.Series(values))
    if parsed.dt.tz is not None:
        parsed = parsed.dt.tz_localize(None)
    return parsed.dt.normalize().astype("datetime64[ns]")


def create_final_state_vector(
    market_df: pd.DataFrame,
    sentiment_df: Optional[pd.DataFrame] = None,
    sentiment_lag_days: int = 1,
    max_staleness_days: Optional[int] = 4,
) -> pd.DataFrame:
    """
    Merges market features with daily FinBERT sentiment into `final_state_vector`.

    Sentiment dated D becomes available on calendar day D + `sentiment_lag_days`.
    Each bar receives the most recent sentiment that was available on or before its
    own date (so Friday's news reaches Monday's bars). Sentiment older than
    `max_staleness_days` after it became available is dropped, so bars with no
    recent sentiment receive a neutral 0.0 instead of a months-old score. Row count and order of `market_df` are preserved, so the
    chronological train/test split index is unchanged.

    Args:
        market_df: Output of MarketDataProcessor with a DatetimeIndex.
        sentiment_df: DataFrame with ['date', 'daily_sentiment'].
        sentiment_lag_days: Calendar-day delay before a day's sentiment is usable (default 1).
        max_staleness_days: Max calendar days a score is carried forward after it becomes
                            available (default 4, covers long weekends). None = unlimited.
    """
    if sentiment_lag_days < 0:
        raise ValueError("sentiment_lag_days must be >= 0")

    final_df = market_df.copy()

    if isinstance(final_df.index, pd.DatetimeIndex):
        bar_timestamps = final_df.index
    elif "Date" in final_df.columns:
        bar_timestamps = pd.DatetimeIndex(pd.to_datetime(final_df["Date"]))
    else:
        bar_timestamps = pd.DatetimeIndex(pd.to_datetime(final_df.index))

    if sentiment_df is None or sentiment_df.empty or "daily_sentiment" not in sentiment_df.columns:
        final_df["daily_sentiment"] = 0.0
        return final_df

    bars = pd.DataFrame(
        {
            "_bar_date": _to_naive_calendar_dates(bar_timestamps).to_numpy(),
            "_pos": np.arange(len(final_df)),
        }
    ).sort_values("_bar_date", kind="stable")

    sentiment = pd.DataFrame(
        {
            "_available_date": _to_naive_calendar_dates(sentiment_df["date"]).to_numpy()
            + np.timedelta64(int(sentiment_lag_days), "D"),
            "daily_sentiment": pd.to_numeric(sentiment_df["daily_sentiment"], errors="coerce").to_numpy(),
        }
    ).dropna()
    sentiment = (
        sentiment.groupby("_available_date", as_index=False)["daily_sentiment"]
        .mean()
        .sort_values("_available_date")
    )

    aligned = pd.merge_asof(
        bars,
        sentiment,
        left_on="_bar_date",
        right_on="_available_date",
        direction="backward",
        tolerance=None if max_staleness_days is None else pd.Timedelta(days=int(max_staleness_days)),
    ).sort_values("_pos")

    final_df["daily_sentiment"] = aligned["daily_sentiment"].fillna(0.0).to_numpy(dtype=np.float64)
    return final_df


# =====================================================================
# Recent news headlines (free sources) for FinBERT
# =====================================================================
NEWS_COLUMNS: List[str] = ["date", "headline", "source", "published"]
_USER_AGENT = "Mozilla/5.0 (compatible; ATU-news-fetcher/1.0)"
_COMPANY_SUFFIX = re.compile(
    r"[\s,]+(inc\.?|incorporated|corp\.?|corporation|ltd\.?|limited|plc|co\.?|company|n\.?v\.?|s\.?a\.?|ag)$",
    re.IGNORECASE,
)


def is_indian_ticker(ticker: str) -> bool:
    return ticker.upper().endswith((".NS", ".BO"))


def lookup_company_name(ticker: str) -> Optional[str]:
    """Company name without legal suffixes (e.g. 'Apple Inc.' -> 'Apple'), or None if unavailable."""
    try:
        info = yf.Ticker(ticker, session=get_yfinance_session()).info or {}
    except Exception:
        return None
    name = str(info.get("longName") or info.get("shortName") or "").strip()
    previous = None
    while name and previous != name:
        previous = name
        name = _COMPANY_SUFFIX.sub("", name).strip()
    return name or None


def _to_utc(value: Any) -> Optional[pd.Timestamp]:
    try:
        ts = pd.Timestamp(value)
    except (ValueError, TypeError):
        return None
    if pd.isna(ts):
        return None
    return ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")


def _naive_day(value: Union[str, pd.Timestamp]) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    return (ts.tz_localize(None) if ts.tzinfo is not None else ts).normalize()


def _fetch_yfinance_news(ticker: str, count: int = 100) -> List[Dict[str, Any]]:
    """Most recent headlines from Yahoo Finance (typically only the last few days)."""
    try:
        yf_ticker = yf.Ticker(ticker, session=get_yfinance_session())
        try:
            items = yf_ticker.get_news(count=count)
        except TypeError:  # older yfinance without `count`
            items = yf_ticker.news
    except Exception as e:
        warnings.warn(f"yfinance news fetch failed for {ticker}: {e}", stacklevel=2)
        return []

    rows: List[Dict[str, Any]] = []
    for item in items or []:
        content = item.get("content", item) if isinstance(item, dict) else {}
        title = str(content.get("title") or "").strip()
        published = content.get("pubDate") or content.get("displayTime")
        if published is None and item.get("providerPublishTime"):
            published = pd.Timestamp(item["providerPublishTime"], unit="s", tz="UTC")
        published = _to_utc(published)
        if title and published is not None:
            rows.append({"headline": title, "published": published, "source": "yfinance"})
    return rows


def _fetch_google_news_rss(
    query: str,
    window_start: pd.Timestamp,
    window_end_exclusive: pd.Timestamp,
    country: str = "US",
    timeout: float = 15.0,
) -> List[Dict[str, Any]]:
    """Google News RSS search restricted to [window_start, window_end_exclusive)."""
    search = f"{query} after:{window_start:%Y-%m-%d} before:{window_end_exclusive:%Y-%m-%d}"
    url = (
        "https://news.google.com/rss/search?q="
        f"{urllib.parse.quote(search)}&hl=en-{country}&gl={country}&ceid={country}:en"
    )
    request = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        root = ET.fromstring(response.read())

    rows: List[Dict[str, Any]] = []
    for item in root.findall("./channel/item"):
        title = (item.findtext("title") or "").strip()
        publisher = (item.findtext("source") or "").strip()
        if publisher and title.endswith(f" - {publisher}"):
            title = title[: -len(publisher) - 3].strip()
        try:
            published = _to_utc(parsedate_to_datetime(item.findtext("pubDate") or ""))
        except (TypeError, ValueError):
            published = None
        if title and published is not None:
            rows.append({"headline": title, "published": published, "source": "google_news"})
    return rows


def _read_news_cache(cache_path: Optional[Path]) -> pd.DataFrame:
    if cache_path is None or not cache_path.exists():
        return pd.DataFrame(columns=["headline", "published", "source"])
    cached = pd.read_csv(cache_path)
    cached["published"] = pd.to_datetime(cached["published"], utc=True, errors="coerce")
    return cached.dropna(subset=["headline", "published"])


def fetch_recent_headlines(
    ticker: str,
    start: Union[str, pd.Timestamp],
    end: Union[str, pd.Timestamp],
    tz: Optional[Any] = None,
    queries: Optional[Sequence[str]] = None,
    window_days: int = 3,
    use_google_news: bool = True,
    use_yfinance: bool = True,
    cache_path: Optional[Union[str, Path]] = None,
    request_pause: float = 0.5,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Fetches real recent headlines for `ticker` between `start` and `end` (inclusive dates).

    Sources:
      - Google News RSS, searched in `window_days` windows so coverage spans the whole
        range (each RSS query returns at most ~100 items).
      - yfinance `get_news` (only the most recent days, but finer timestamps).

    Dates are assigned in `tz` (pass the exchange timezone, e.g. market_df.index.tz) so
    that the downstream one-day lag maps e.g. Tuesday post-market news to Wednesday.
    When `cache_path` is given, headlines are accumulated there and later runs only
    re-fetch from shortly before the newest cached headline.

    Returns:
        DataFrame with columns ['date', 'headline', 'source', 'published'] sorted by time.
    """
    start_day, end_day = _naive_day(start), _naive_day(end)
    if end_day < start_day:
        raise ValueError(f"end ({end_day.date()}) is before start ({start_day.date()})")

    if queries is None:
        base_symbol = ticker.split(".")[0].upper()
        company = lookup_company_name(ticker)
        queries = [f"{company} stock"] if company else []
        queries.append(f"{base_symbol} stock")
    queries = list(dict.fromkeys(q for q in queries if q and q.strip()))
    country = "IN" if is_indian_ticker(ticker) else "US"

    cache_file = Path(cache_path) if cache_path is not None else None
    cached = _read_news_cache(cache_file)
    fetch_from = start_day
    if not cached.empty:
        cached_days = cached["published"].dt.tz_convert(None).dt.normalize()
        if cached_days.min() <= start_day + pd.Timedelta(days=3):
            fetch_from = max(start_day, cached_days.max() - pd.Timedelta(days=2))

    rows: List[Dict[str, Any]] = []
    failures = 0
    requests_made = 0
    if use_google_news:
        window_start = fetch_from
        while window_start <= end_day:
            window_end_exclusive = window_start + pd.Timedelta(days=window_days + 1)  # 1-day overlap
            for query in queries:
                try:
                    rows.extend(_fetch_google_news_rss(query, window_start, window_end_exclusive, country))
                except (urllib.error.URLError, ET.ParseError, TimeoutError, OSError) as e:
                    failures += 1
                    if failures <= 3:
                        warnings.warn(f"Google News RSS request failed ({query!r}, {window_start.date()}): {e}", stacklevel=2)
                requests_made += 1
                time.sleep(request_pause)
            window_start += pd.Timedelta(days=window_days)
    if use_yfinance:
        rows.extend(_fetch_yfinance_news(ticker))

    fresh = pd.DataFrame(rows, columns=["headline", "published", "source"])
    combined = pd.concat([cached, fresh], ignore_index=True) if not cached.empty else fresh
    if combined.empty:
        if verbose:
            print(f"  - News fetch: 0 headlines ({requests_made} RSS requests, {failures} failed)")
        return pd.DataFrame(columns=NEWS_COLUMNS)

    combined["published"] = pd.to_datetime(combined["published"], utc=True)
    combined["_key"] = combined["headline"].str.lower().str.replace(r"\W+", " ", regex=True).str.strip()
    combined = (
        combined.sort_values("published")
        .drop_duplicates(subset="_key", keep="first")  # earliest publication wins
        .drop(columns="_key")
        .reset_index(drop=True)
    )

    if cache_file is not None:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        combined.assign(published=combined["published"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")).to_csv(cache_file, index=False)

    local_time = combined["published"].dt.tz_convert(tz) if tz is not None else combined["published"]
    combined["date"] = local_time.dt.strftime("%Y-%m-%d")
    local_day = pd.to_datetime(combined["date"])
    result = combined[(local_day >= start_day) & (local_day <= end_day)][NEWS_COLUMNS].reset_index(drop=True)

    if verbose:
        print(
            f"  - News fetch: {len(result):,} unique headlines over {result['date'].nunique()} days "
            f"({requests_made} RSS requests, {failures} failed; queries={queries})"
        )
    return result


if __name__ == "__main__":
    processor = MarketDataProcessor()

    print("Fetching and processing 2 years of daily data for 'AAPL'...")
    state_vector_df = processor.process(ticker="AAPL", period="2y", interval="1d")
    split_idx = compute_split_index(len(state_vector_df), processor.train_ratio)

    print(f"\nFinal State Vector Shape: {state_vector_df.shape}")
    print(f"Scaler fit on rows [0, {processor.fit_end_index}) | split index: {split_idx}")
    print(f"Feature Columns ({len(state_vector_df.columns)}): {list(state_vector_df.columns)}\n")
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 1000)
    print(state_vector_df.head(5))
