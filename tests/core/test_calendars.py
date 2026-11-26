"""The single NYSE calendar: known trading-day counts, holidays and special closures."""

import pandas as pd
import pytest

from atu2.core.calendars import (
    holidays,
    is_trading_day,
    next_trading_day,
    previous_trading_day,
    trading_days,
)
from atu2.core.errors import CalendarError

# Published NYSE trading-day counts.
KNOWN_COUNTS = {2016: 252, 2018: 251, 2019: 252, 2020: 253, 2021: 252, 2022: 251, 2023: 250, 2024: 252}


@pytest.mark.parametrize("year,count", sorted(KNOWN_COUNTS.items()))
def test_trading_days_per_year(year, count):
    assert len(trading_days(f"{year}-01-01", f"{year}-12-31")) == count


@pytest.mark.parametrize(
    "day",
    [
        "2024-03-29",  # Good Friday
        "2022-06-20",  # Juneteenth observed (Sunday -> Monday)
        "2021-12-24",  # Christmas observed (Saturday -> Friday)
        "2023-01-02",  # New Year observed (Sunday -> Monday)
        "2024-11-28",  # Thanksgiving
        "2012-10-29",  # Hurricane Sandy
        "2025-01-09",  # national day of mourning
        "2001-09-11",
    ],
)
def test_closed_days(day):
    assert not is_trading_day(day)


@pytest.mark.parametrize(
    "day",
    [
        "2021-12-31",  # New Year on Saturday: Friday stays open
        "2021-06-18",  # Juneteenth not yet a market holiday in 2021
        "2024-11-29",  # day after Thanksgiving (early close, still a trading day)
        "2024-07-03",
    ],
)
def test_open_days(day):
    assert is_trading_day(day)


def test_weekends_closed_and_holidays_are_weekdays():
    assert not is_trading_day("2024-06-15")
    h = holidays("2016-01-01", "2026-12-31")
    assert (h.day_of_week < 5).all()


def test_next_and_previous_trading_day():
    assert next_trading_day("2024-03-28") == pd.Timestamp("2024-04-01")
    assert previous_trading_day("2024-04-01") == pd.Timestamp("2024-03-28")
    assert next_trading_day(pd.Timestamp("2024-12-24 15:00", tz="America/New_York")) == pd.Timestamp("2024-12-26")


def test_range_edges():
    assert len(trading_days("2024-01-10", "2024-01-01")) == 0
    with pytest.raises(CalendarError):
        trading_days("1999-12-01", "2000-01-10")
    with pytest.raises(CalendarError):
        previous_trading_day("2000-01-03")
