"""T-DATA-1 (known split) and T-DATA-2 (each bad-data type raises its named error)."""

import numpy as np
import pandas as pd
import pytest

from atu2.core import errors as errs
from atu2.data.settings import ApprovedMove, ValidationSettings
from atu2.data.validators import findings, validate_bars
from tests.data.fakes import make_bars

S = ValidationSettings(max_abs_daily_return=0.25, max_zero_volume_run=2, max_consecutive_missing_days=1)


def test_clean_bars_pass():
    assert validate_bars("AAA", make_bars(n_days=250), S) == []


# --------------------------------------------------------------------------- T-DATA-1
def test_t_data_1_known_split_adjusted_has_no_jump_unadjusted_is_rejected():
    bars = make_bars("2023-01-03", 250, split=("2023-06-01", 4.0))
    d = pd.Timestamp("2023-06-01")
    raw_move = bars["close"].pct_change().loc[d]
    adj_move = bars["adj_close"].pct_change().loc[d]
    assert raw_move < -0.7  # raw price falls ~75 % on a 4-for-1 split
    assert abs(adj_move) < 0.05  # adjusted series: no jump
    validate_bars("AAA", bars, S)  # adjusted data passes

    # unadjusted data passed off as adjusted: the split shows up as a huge move and is rejected
    fake_adjusted = bars.assign(adj_close=bars["close"])
    with pytest.raises(errs.OutlierMoveError, match="2023-06-01"):
        validate_bars("AAA", fake_adjusted, S)
    # data marked as unadjusted is rejected outright
    with pytest.raises(errs.UnadjustedDataError):
        validate_bars("AAA", bars, S, adjusted=False)


# --------------------------------------------------------------------------- T-DATA-2
def _inject(kind: str) -> pd.DataFrame:
    b = make_bars("2023-01-03", 120)
    i = 50
    if kind == "missing_column":
        return b.drop(columns=["volume"])
    if kind == "unsorted":
        return b.iloc[::-1]
    if kind == "duplicate":
        return pd.concat([b.iloc[:i], b.iloc[[i]], b.iloc[i:]])
    if kind == "non_trading_day":
        extra = b.iloc[[i]].copy()
        extra.index = pd.DatetimeIndex([pd.Timestamp("2023-07-04")], name="date")  # holiday
        return pd.concat([b, extra]).sort_index()
    if kind == "nan":
        b.iloc[i, b.columns.get_loc("close")] = np.nan
    elif kind == "non_positive":
        b.iloc[i, b.columns.get_loc("low")] = 0.0
    elif kind == "high_below_low":
        b.iloc[i, b.columns.get_loc("high")] = b.iloc[i]["low"] * 0.9
    elif kind == "outside_range":
        b.iloc[i, b.columns.get_loc("close")] = b.iloc[i]["high"] * 1.01
    elif kind == "negative_volume":
        b.iloc[i, b.columns.get_loc("volume")] = -5
    elif kind == "zero_volume_run":
        b.iloc[i : i + 3, b.columns.get_loc("volume")] = 0
    elif kind == "calendar_gap":
        return b.drop(b.index[i : i + 2])
    elif kind == "outlier":
        b.iloc[i:, b.columns.get_loc("adj_close")] *= 1.4
    return b


@pytest.mark.parametrize(
    "kind,error",
    [
        ("missing_column", errs.MissingColumnError),
        ("unsorted", errs.UnsortedDatesError),
        ("duplicate", errs.DuplicateDateError),
        ("non_trading_day", errs.NonTradingDayError),
        ("nan", errs.MissingValueError),
        ("non_positive", errs.NonPositivePriceError),
        ("high_below_low", errs.HighBelowLowError),
        ("outside_range", errs.PriceOutsideRangeError),
        ("negative_volume", errs.NegativeVolumeError),
        ("zero_volume_run", errs.ZeroVolumeRunError),
        ("calendar_gap", errs.CalendarGapError),
        ("outlier", errs.OutlierMoveError),
    ],
)
def test_t_data_2_each_bad_data_type_raises_its_named_error(kind, error):
    bad = _inject(kind)
    with pytest.raises(error):
        validate_bars("AAA", bad, S)
    assert issubclass(error, errs.DataValidationError)
    reported = {f.error for f in findings("AAA", bad, S) if f.severity == "error"}
    assert error in reported


def test_tolerated_cases_are_warnings():
    b = make_bars("2023-01-03", 120)
    one_missing = b.drop(b.index[40])
    found = validate_bars("AAA", one_missing, S)
    assert [f.check for f in found] == ["calendar_gap"] and found[0].severity == "warning"
    two_zero = b.copy()
    two_zero.iloc[10:12, two_zero.columns.get_loc("volume")] = 0
    assert validate_bars("AAA", two_zero, S) == []


def test_approved_outlier_becomes_a_warning():
    bad = _inject("outlier")
    day = str(bad.index[50].date())
    approved = ValidationSettings(0.25, 2, 1, approved_moves=(ApprovedMove("AAA", day, "verified crash"),))
    found = validate_bars("AAA", bad, approved)
    assert any(f.check == "outlier" and f.severity == "warning" for f in found)
    with pytest.raises(errs.OutlierMoveError):
        validate_bars("BBB", bad, approved)  # the approval is per ticker


def test_multiple_errors_are_counted_and_empty_frame_is_fine():
    b = _inject("negative_volume")
    b.iloc[60, b.columns.get_loc("low")] = -1.0
    with pytest.raises(errs.NonPositivePriceError, match=r"\+1 more"):
        validate_bars("AAA", b, S)
    assert findings("AAA", make_bars(n_days=5).iloc[:0], S) == []
    assert findings("AAA", make_bars(n_days=5), S, adjusted=True) == []
