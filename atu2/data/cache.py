"""Append-only Parquet cache of daily bars, one file per ticker and feed (FR-01, NFR-09).

Layout:  <data_dir>/raw/<feed>/<TICKER>_1d.parquet  plus  manifest.json in the same folder.
Columns: date, open, high, low, close, volume (raw) and adj_close (split + dividend adjusted).

Update rule per ticker:
- no file: download from `start`                                         -> action "full"
- file already covers the last complete session: no request at all       -> action "current"
- otherwise: re-fetch from `overlap_days` rows before the last cached date
  - overlap identical (relative tolerance): append only the new rows      -> action "incremental"
  - overlap differs: a split or dividend rewrote history; the whole file
    is rebuilt from `start`                                               -> action "rebuilt"
Existing rows are never edited in place. Files are written atomically (temp file + rename),
and the manifest records rows, date range, feed and two hashes: `data_hash` (content of
the table, stable across Parquet versions) and `file_hash` (bytes on disk).
"""

import json
import os
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import pandas as pd

from atu2.core.errors import CacheRewriteError, DataSourceError
from atu2.core.hashing import hash_dataframe, hash_file
from atu2.core.logging import get_logger
from atu2.data.alpaca_client import BAR_COLUMNS, last_complete_session

MANIFEST = "manifest.json"
log = get_logger("data.cache")


class BarSource(Protocol):
    """What the cache needs from a client (AlpacaDailyClient satisfies it)."""

    feed: str
    settle_time: str

    def fetch(self, symbol: str, start: str | pd.Timestamp, end: str | pd.Timestamp | None = None) -> pd.DataFrame:
        """Daily bars with BAR_COLUMNS, date index, attrs["feed"]."""


@dataclass(frozen=True)
class UpdateResult:
    """Outcome of updating one ticker."""

    ticker: str
    action: str  # full | current | incremental | rebuilt
    rows: int
    new_rows: int
    first: str | None
    last: str | None
    feed: str
    data_hash: str | None


class BarCache:
    """One folder per feed; never mixes feeds. `feed` is the only feed this folder accepts."""

    def __init__(self, root: str | Path, feed: str) -> None:
        self.root = Path(root)
        self.feed = feed

    def path(self, ticker: str) -> Path:
        """Parquet file of `ticker`."""
        return self.root / f"{ticker}_1d.parquet"

    def tickers(self) -> list[str]:
        """Tickers with a cached file, sorted."""
        return sorted(p.name.removesuffix("_1d.parquet") for p in self.root.glob("*_1d.parquet"))

    def read(self, ticker: str) -> pd.DataFrame | None:
        """Cached bars (date index, BAR_COLUMNS as float) or None."""
        p = self.path(ticker)
        if not p.exists():
            return None
        df = pd.read_parquet(p)
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date")[list(BAR_COLUMNS)].astype(float)
        df.index = pd.DatetimeIndex(df.index, name="date")
        return df

    def manifest(self) -> dict[str, dict[str, Any]]:
        """Per-ticker metadata and hashes."""
        p = self.root / MANIFEST
        if not p.exists():
            return {}
        data: dict[str, dict[str, Any]] = json.loads(p.read_text(encoding="utf-8"))
        return data

    def _write_manifest(self, manifest: dict[str, dict[str, Any]]) -> None:
        tmp = self.root / (MANIFEST + ".tmp")
        tmp.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
        os.replace(tmp, self.root / MANIFEST)

    def write(self, ticker: str, bars: pd.DataFrame, feed: str) -> str:
        """Writes the whole table atomically, updates the manifest, returns the data hash."""
        if feed != self.feed:
            raise DataSourceError(f"{ticker}: bars from feed '{feed}' cannot go into the '{self.feed}' cache")
        self.root.mkdir(parents=True, exist_ok=True)
        df = bars[list(BAR_COLUMNS)].astype(float).sort_index()
        df.index = pd.DatetimeIndex(df.index, name="date")
        data_hash = hash_dataframe(df)
        tmp = self.path(ticker).with_suffix(".parquet.tmp")
        df.reset_index().to_parquet(tmp, index=False)
        os.replace(tmp, self.path(ticker))
        manifest = self.manifest()
        manifest[ticker] = {
            "rows": len(df),
            "first": str(df.index[0].date()) if len(df) else None,
            "last": str(df.index[-1].date()) if len(df) else None,
            "feed": feed,
            "adjustment": "raw ohlcv + adj_close (splits and dividends)",
            "data_hash": data_hash,
            "file_hash": hash_file(self.path(ticker)),
            "updated_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        }
        self._write_manifest(manifest)
        return data_hash


def history_rewritten(cached: pd.DataFrame, fresh: pd.DataFrame, tolerance: float) -> bool:
    """True when close or adj_close on the overlapping dates differ by more than `tolerance` (relative)."""
    common = cached.index.intersection(fresh.index)
    if len(common) == 0:
        return False
    for col in ("close", "adj_close"):
        a = cached.loc[common, col].to_numpy(dtype=float)
        b = fresh.loc[common, col].to_numpy(dtype=float)
        rel = np.abs(b - a) / np.maximum(np.abs(a), 1e-12)
        if float(np.nanmax(rel)) > tolerance:
            return True
    return False


def update_ticker(
    source: BarSource,
    cache: BarCache,
    ticker: str,
    *,
    start: str,
    overlap_days: int,
    tolerance: float,
    now: pd.Timestamp | None = None,
) -> UpdateResult:
    """Brings one ticker's file up to the last complete session (see module docstring)."""
    target = last_complete_session(now, source.settle_time)
    cached = cache.read(ticker)

    def done(action: str, df: pd.DataFrame, new_rows: int) -> UpdateResult:
        h = (
            cache.write(ticker, df, source.feed)
            if action != "current"
            else cache.manifest().get(ticker, {}).get("data_hash")
        )
        return UpdateResult(
            ticker,
            action,
            len(df),
            new_rows,
            str(df.index[0].date()) if len(df) else None,
            str(df.index[-1].date()) if len(df) else None,
            source.feed,
            h,
        )

    if cached is None or len(cached) == 0:
        fresh = source.fetch(ticker, start, target)
        if len(fresh) == 0:
            raise DataSourceError(f"{ticker}: no bars returned from {start} to {target.date()}")
        return done("full", fresh, len(fresh))
    if cached.index[-1] >= target:
        return done("current", cached, 0)
    since = cached.index[max(0, len(cached) - overlap_days)]
    fresh = source.fetch(ticker, since, target)
    if history_rewritten(cached, fresh, tolerance):
        log.warning("history rewritten (split or dividend); rebuilding", extra={"fields": {"ticker": ticker}})
        rebuilt = source.fetch(ticker, start, target)
        if len(rebuilt) == 0:
            raise CacheRewriteError(f"{ticker}: history changed but the full re-download returned nothing")
        return done("rebuilt", rebuilt, len(rebuilt))
    new = fresh[fresh.index > cached.index[-1]]
    return done("incremental", pd.concat([cached, new]), len(new))
