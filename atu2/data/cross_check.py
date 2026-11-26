"""Second-source cross-check of closes (FR-02; plan: "disagreement above a tolerance fails").

Sources sit behind the `DailyPriceSource` interface. The built-in `StooqSource` reads Stooq's
free daily CSV (no key). Comparison is on daily returns, not price levels, so a different
adjustment base between providers does not matter; a day disagrees when the absolute
return difference exceeds `return_tolerance`, and the check fails when more than
`max_mismatch_share` of the overlapping days disagree.

Not verified against the live Stooq service yet (tests use recorded CSV fixtures only):
Stooq's adjustment convention for US ETFs is to be confirmed on the first real run.
"""

import io
from dataclasses import dataclass
from typing import Protocol

import numpy as np
import pandas as pd

from atu2.core.errors import CrossCheckError, DataSourceError
from atu2.data.alpaca_client import HttpClient, requests_client
from atu2.data.settings import CrossCheckSettings

STOOQ_URL = "https://stooq.com/q/d/l/"


class DailyPriceSource(Protocol):
    """A second provider of daily closes."""

    name: str

    def daily_closes(self, ticker: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
        """Closes indexed by timezone-naive dates."""


@dataclass
class StooqSource:
    """Stooq daily CSV: https://stooq.com/q/d/l/?s=spy.us&i=d (columns Date,Open,High,Low,Close,Volume)."""

    http: HttpClient | None = None
    name: str = "stooq"

    def daily_closes(self, ticker: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
        """Closes from Stooq between start and end."""
        http = self.http or requests_client()
        params = {"s": f"{ticker.lower()}.us", "i": "d", "d1": start.strftime("%Y%m%d"), "d2": end.strftime("%Y%m%d")}
        resp = http.get(STOOQ_URL, headers={}, params=params, timeout=30)
        if int(resp.status_code) != 200:
            raise DataSourceError(f"{ticker}: HTTP {resp.status_code} from Stooq")
        text = resp.text or ""
        if not text.lower().startswith("date,"):
            raise DataSourceError(f"{ticker}: Stooq returned no CSV ({text[:60]!r})")
        df = pd.read_csv(io.StringIO(text))
        s = pd.Series(
            df["Close"].astype(float).to_numpy(), index=pd.DatetimeIndex(pd.to_datetime(df["Date"]), name="date")
        )
        return s.sort_index()


@dataclass(frozen=True)
class CrossCheckResult:
    """Outcome for one ticker; shares are fractions of overlapping days."""

    ticker: str
    source: str
    overlap_days: int
    mismatch_days: int
    mismatch_share: float
    max_abs_return_diff: float
    passed: bool

    def as_dict(self) -> dict[str, object]:
        """JSON-ready form for reports."""
        return self.__dict__.copy()


def compare_closes(
    ticker: str, primary_adj_close: pd.Series, secondary_close: pd.Series, settings: CrossCheckSettings, source: str
) -> CrossCheckResult:
    """Compares daily returns on the dates both sources share."""
    common = primary_adj_close.index.intersection(secondary_close.index)
    a = primary_adj_close.loc[common].pct_change().iloc[1:]
    b = secondary_close.loc[common].pct_change().iloc[1:]
    diff = (a - b).abs().to_numpy(dtype=float)
    diff = diff[np.isfinite(diff)]
    n = len(diff)
    mism = int((diff > settings.return_tolerance).sum())
    share = mism / n if n else 1.0
    passed = n >= settings.min_overlap_days and share <= settings.max_mismatch_share
    return CrossCheckResult(
        ticker, source, n, mism, round(share, 4), float(np.max(diff)) if n else float("nan"), passed
    )


def cross_check(
    ticker: str, bars: pd.DataFrame, source: DailyPriceSource, settings: CrossCheckSettings
) -> CrossCheckResult:
    """Fetches the second source and raises CrossCheckError when the comparison fails."""
    closes = source.daily_closes(ticker, bars.index[0], bars.index[-1])
    result = compare_closes(ticker, bars["adj_close"], closes, settings, source.name)
    if result.overlap_days < settings.min_overlap_days:
        raise CrossCheckError(
            f"{ticker}: only {result.overlap_days} overlapping days with {source.name} "
            f"(need {settings.min_overlap_days})"
        )
    if not result.passed:
        raise CrossCheckError(
            f"{ticker}: {result.mismatch_share:.1%} of days disagree with {source.name} by more than "
            f"{settings.return_tolerance:.2%} (allowed {settings.max_mismatch_share:.1%})"
        )
    return result
