"""Data validation (FR-02): every failure type raises its own named error (atu2.core.errors).

Checks, in this order:
  MissingColumnError       a required column is absent
  UnsortedDatesError       dates not increasing
  DuplicateDateError       a date appears twice
  NonTradingDayError       a bar on a weekend or NYSE holiday
  MissingValueError        NaN or infinite price/volume
  NonPositivePriceError    price <= 0
  HighBelowLowError        high < low
  PriceOutsideRangeError   open or close outside [low, high]
  NegativeVolumeError      volume < 0
  ZeroVolumeRunError       more than `max_zero_volume_run` consecutive zero-volume days
  CalendarGapError         more than `max_consecutive_missing_days` consecutive NYSE days missing
  UnadjustedDataError      data not marked as split- and dividend-adjusted
  OutlierMoveError         |adj_close daily return| above the threshold and not approved

`findings()` collects everything (for reports); `validate_bars()` raises the first failure.
A single missing trading day is a warning, not a failure (it may be forward-filled once).
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from atu2.core.calendars import trading_days
from atu2.core.errors import (
    CalendarGapError,
    DataValidationError,
    DuplicateDateError,
    HighBelowLowError,
    MissingColumnError,
    MissingValueError,
    NegativeVolumeError,
    NonPositivePriceError,
    NonTradingDayError,
    OutlierMoveError,
    PriceOutsideRangeError,
    UnadjustedDataError,
    UnsortedDatesError,
    ZeroVolumeRunError,
)
from atu2.data.alpaca_client import BAR_COLUMNS
from atu2.data.settings import ValidationSettings

PRICE_COLUMNS = ("open", "high", "low", "close", "adj_close")


@dataclass(frozen=True)
class Finding:
    """One validation result. severity "error" fails validation, "warning" does not."""

    ticker: str
    check: str
    severity: str
    message: str
    error: type[DataValidationError] | None = None

    def as_dict(self) -> dict[str, str]:
        """JSON-ready form for reports."""
        return {"ticker": self.ticker, "check": self.check, "severity": self.severity, "message": self.message}


def _dates_list(dates: pd.Index, limit: int = 5) -> str:
    shown = ", ".join(str(pd.Timestamp(d).date()) for d in list(dates)[:limit])
    return shown + (f" (+{len(dates) - limit} more)" if len(dates) > limit else "")


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """(start, length) of each run of True values."""
    runs, start = [], None
    for i, v in enumerate(mask):
        if v and start is None:
            start = i
        elif not v and start is not None:
            runs.append((start, i - start))
            start = None
    if start is not None:
        runs.append((start, len(mask) - start))
    return runs


def _value_findings(
    ticker: str, bars: pd.DataFrame, idx: pd.DatetimeIndex, settings: ValidationSettings
) -> list[Finding]:
    """Price and volume checks on every row."""
    out: list[Finding] = []

    def _err(check: str, error: type[DataValidationError], message: str) -> None:
        out.append(Finding(ticker, check, "error", message, error))

    values = bars[list(BAR_COLUMNS)].to_numpy(dtype=float)
    bad = ~np.isfinite(values)
    if bad.any():
        _err(
            "missing_values",
            MissingValueError,
            f"{int(bad.sum())} NaN/inf values on {_dates_list(idx[bad.any(axis=1)])}",
        )
    prices = bars[list(PRICE_COLUMNS)]
    nonpos = (prices <= 0).any(axis=1)
    if nonpos.any():
        _err("positive_prices", NonPositivePriceError, f"price <= 0 on {_dates_list(idx[nonpos.to_numpy()])}")
    hbl = bars["high"] < bars["low"]
    if hbl.any():
        _err("high_low", HighBelowLowError, f"high below low on {_dates_list(idx[hbl.to_numpy()])}")
    outside = (bars[["open", "close"]].max(axis=1) > bars["high"]) | (bars[["open", "close"]].min(axis=1) < bars["low"])
    outside &= ~hbl
    if outside.any():
        _err(
            "ohlc_range",
            PriceOutsideRangeError,
            f"open/close outside high-low on {_dates_list(idx[outside.to_numpy()])}",
        )
    negvol = bars["volume"] < 0
    if negvol.any():
        _err("volume_sign", NegativeVolumeError, f"negative volume on {_dates_list(idx[negvol.to_numpy()])}")
    for start, length in _runs((bars["volume"] == 0).to_numpy()):
        if length > settings.max_zero_volume_run:
            _err("zero_volume", ZeroVolumeRunError, f"{length} zero-volume days from {idx[start].date()}")
    return out


def findings(ticker: str, bars: pd.DataFrame, settings: ValidationSettings, adjusted: bool = True) -> list[Finding]:
    """Runs every check and returns all findings (errors and warnings)."""
    out: list[Finding] = []

    def err(check: str, error: type[DataValidationError], message: str) -> None:
        out.append(Finding(ticker, check, "error", message, error))

    missing_cols = [c for c in BAR_COLUMNS if c not in bars.columns]
    if missing_cols:
        err("columns", MissingColumnError, f"missing columns {missing_cols}")
        return out
    if len(bars) == 0:
        return out
    idx = pd.DatetimeIndex(bars.index)
    if not idx.is_monotonic_increasing:
        err("sorted", UnsortedDatesError, "dates are not in increasing order")
        return out
    if idx.has_duplicates:
        err("duplicates", DuplicateDateError, f"duplicate dates: {_dates_list(idx[idx.duplicated()])}")
        return out
    expected = trading_days(idx[0], idx[-1])
    off_calendar = idx.difference(expected)
    if len(off_calendar):
        err("calendar", NonTradingDayError, f"bars on non-trading days: {_dates_list(off_calendar)}")

    out += _value_findings(ticker, bars, idx, settings)

    observed = np.isin(expected, idx)
    for start, length in _runs(~observed):
        first = expected[start].date()
        if length > settings.max_consecutive_missing_days:
            err("calendar_gap", CalendarGapError, f"{length} consecutive trading days missing from {first}")
        else:
            out.append(Finding(ticker, "calendar_gap", "warning", f"{length} trading day(s) missing at {first}"))

    if not adjusted:
        err("adjusted", UnadjustedDataError, "data is not split- and dividend-adjusted (adjustment='all' required)")
    ret = bars["adj_close"].pct_change()
    big = ret.abs() > settings.max_abs_daily_return
    for d in idx[big.to_numpy()]:
        r = float(ret.loc[d])
        if settings.is_approved(ticker, d):
            out.append(Finding(ticker, "outlier", "warning", f"approved move {r:+.1%} on {d.date()}"))
        else:
            err(
                "outlier",
                OutlierMoveError,
                f"daily move {r:+.1%} on {d.date()} exceeds {settings.max_abs_daily_return:.0%}: check it, then "
                "approve it in configs/data.yaml validation.approved_moves (ticker, date, reason) if it is real",
            )
    return out


def validate_bars(
    ticker: str, bars: pd.DataFrame, settings: ValidationSettings, adjusted: bool = True
) -> list[Finding]:
    """Raises the first failure's named error; returns the (warning-only) findings otherwise."""
    found = findings(ticker, bars, settings, adjusted)
    errors = [f for f in found if f.severity == "error"]
    if errors:
        first = errors[0]
        assert first.error is not None
        extra = f" (+{len(errors) - 1} more error(s))" if len(errors) > 1 else ""
        raise first.error(f"{ticker}: {first.message}{extra}")
    return found
