"""Typed schemas for configs/data.yaml and configs/universe.yaml (unknown keys are errors)."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import pandas as pd

from atu2.core.config import load_config
from atu2.core.errors import ConfigTypeError

AssetClass = Literal[
    "us_equity",
    "us_sector",
    "intl_equity",
    "govt_bond",
    "credit",
    "gold_silver",
    "commodity",
    "real_estate",
    "currency",
]


def _check_date(value: str, where: str) -> None:
    try:
        pd.Timestamp(value)
    except ValueError as e:
        raise ConfigTypeError(f"{where}: '{value}' is not a date") from e


# --------------------------------------------------------------------------- data.yaml
@dataclass(frozen=True)
class DownloadSettings:
    """Alpaca download settings. Times are New York time; pauses in seconds."""

    start: str
    feed: Literal["sip", "iex"]
    allow_iex_fallback: bool
    chunk_years: int
    pause_seconds: float
    max_retries: int
    overlap_days: int
    rewrite_tolerance: float
    settle_time: str
    data_dir: str

    def validate(self) -> None:
        """Range checks."""
        _check_date(self.start, "download.start")
        if self.chunk_years < 1 or self.max_retries < 0 or self.overlap_days < 1 or self.pause_seconds < 0:
            raise ConfigTypeError("download: chunk_years >= 1, max_retries >= 0, overlap_days >= 1, pause >= 0")
        try:
            pd.Timestamp(f"2000-01-01 {self.settle_time}")
        except ValueError as e:
            raise ConfigTypeError(f"download.settle_time: '{self.settle_time}' is not HH:MM") from e


@dataclass(frozen=True)
class ApprovedMove:
    """A human decision that a flagged daily move is real (e.g. a crash), with the reason."""

    ticker: str
    date: str
    reason: str

    def validate(self) -> None:
        """The date must parse and a reason must be given."""
        _check_date(self.date, f"approved_moves[{self.ticker}]")
        if not self.reason.strip():
            raise ConfigTypeError(f"approved_moves[{self.ticker} {self.date}]: a reason is required")


@dataclass(frozen=True)
class ValidationSettings:
    """FR-02 thresholds; returns as fractions, runs in trading days."""

    max_abs_daily_return: float
    max_zero_volume_run: int
    max_consecutive_missing_days: int
    approved_moves: tuple[ApprovedMove, ...] = ()

    def is_approved(self, ticker: str, date: pd.Timestamp) -> bool:
        """True when a human approved the move of `ticker` on `date`."""
        return any(m.ticker == ticker and pd.Timestamp(m.date) == pd.Timestamp(date) for m in self.approved_moves)


@dataclass(frozen=True)
class CrossCheckSettings:
    """Second-source comparison on daily returns (fractions)."""

    enabled: bool
    source: Literal["stooq"]
    return_tolerance: float
    max_mismatch_share: float
    min_overlap_days: int


@dataclass(frozen=True)
class DataConfig:
    """configs/data.yaml."""

    download: DownloadSettings
    validation: ValidationSettings
    cross_check: CrossCheckSettings

    @property
    def raw_dir(self) -> Path:
        """Folder of the raw cache for the configured feed (feeds are never mixed)."""
        return Path(self.download.data_dir) / "raw" / self.download.feed


# --------------------------------------------------------------------------- universe.yaml
@dataclass(frozen=True)
class UniverseRule:
    """Written rule applied to the candidate list as of a fixed date (ADR 0005)."""

    as_of: str
    min_history_years: float
    min_median_dollar_volume: float
    volume_lookback_days: int
    warm_up_days: int
    required_asset_classes: tuple[AssetClass, ...] = field(default_factory=tuple)

    def validate(self) -> None:
        """Range checks."""
        _check_date(self.as_of, "rule.as_of")
        if self.warm_up_days < 0 or self.volume_lookback_days < 1:
            raise ConfigTypeError("rule: warm_up_days >= 0 and volume_lookback_days >= 1")


@dataclass(frozen=True)
class Candidate:
    """One candidate ETF; inception is the fund launch date."""

    ticker: str
    asset_class: AssetClass
    inception: str
    name: str

    def validate(self) -> None:
        """Ticker format and date."""
        if not self.ticker.isupper() or not self.ticker.replace(".", "").isalnum():
            raise ConfigTypeError(f"candidate ticker '{self.ticker}' must be upper-case letters/digits")
        _check_date(self.inception, f"candidate {self.ticker}.inception")


@dataclass(frozen=True)
class UniverseConfig:
    """configs/universe.yaml."""

    status: Literal["proposed", "confirmed"]
    rule: UniverseRule
    candidates: tuple[Candidate, ...]

    def validate(self) -> None:
        """Tickers are unique and every required class has a candidate."""
        tickers = [c.ticker for c in self.candidates]
        dupes = sorted({t for t in tickers if tickers.count(t) > 1})
        if dupes:
            raise ConfigTypeError(f"universe: duplicate tickers {dupes}")
        missing = set(self.rule.required_asset_classes) - {c.asset_class for c in self.candidates}
        if missing:
            raise ConfigTypeError(f"universe: no candidate for required asset classes {sorted(missing)}")

    @property
    def tickers(self) -> list[str]:
        """Candidate tickers in file order."""
        return [c.ticker for c in self.candidates]


def load_data_config(path: str | Path) -> DataConfig:
    """Loads and validates configs/data.yaml."""
    return load_config(path, DataConfig)


def load_universe(path: str | Path) -> UniverseConfig:
    """Loads and validates configs/universe.yaml."""
    return load_config(path, UniverseConfig)
