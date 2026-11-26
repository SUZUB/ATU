"""The single exchange calendar used everywhere: NYSE full-day closures.

Every layer takes trading dates from here, so assets with different holidays are
aligned on one calendar (plan: "use one calendar and a stated fill rule"). Early
closes (half days) are irrelevant for daily bars and are not modelled.

Supported range: 2000-01-01 onwards. Holiday rules (NYSE Rule 7.2):
- New Year's Day; Sunday -> Monday, Saturday -> no weekday closure
- Martin Luther King Jr. Day (third Monday of January)
- Washington's Birthday (third Monday of February)
- Good Friday
- Memorial Day (last Monday of May)
- Juneteenth (from 2022); July 4; Christmas: Saturday -> Friday, Sunday -> Monday
- Labor Day (first Monday of September); Thanksgiving (fourth Thursday of November)
plus the one-off closures in SPECIAL_CLOSURES.
"""

from datetime import date

import pandas as pd
from pandas.tseries.holiday import (
    AbstractHolidayCalendar,
    GoodFriday,
    Holiday,
    USLaborDay,
    USMartinLutherKingJr,
    USMemorialDay,
    USPresidentsDay,
    USThanksgivingDay,
    nearest_workday,
    sunday_to_monday,
)

from atu2.core.errors import CalendarError

CALENDAR_START = pd.Timestamp("2000-01-01")
DateLike = str | date | pd.Timestamp

SPECIAL_CLOSURES: tuple[str, ...] = (
    "2001-09-11",  # September 11 attacks
    "2001-09-12",
    "2001-09-13",
    "2001-09-14",
    "2004-06-11",  # President Reagan's funeral
    "2007-01-02",  # President Ford's funeral
    "2012-10-29",  # Hurricane Sandy
    "2012-10-30",
    "2018-12-05",  # President G. H. W. Bush's funeral
    "2025-01-09",  # President Carter's funeral
)


class NYSEHolidayCalendar(AbstractHolidayCalendar):
    """Rule-based NYSE holidays (without the one-off closures)."""

    rules = [  # noqa: RUF012  (pandas expects a plain class attribute)
        Holiday("New Year's Day", month=1, day=1, observance=sunday_to_monday),
        USMartinLutherKingJr,
        USPresidentsDay,
        GoodFriday,
        USMemorialDay,
        Holiday("Juneteenth", month=6, day=19, start_date="2022-01-01", observance=nearest_workday),
        Holiday("Independence Day", month=7, day=4, observance=nearest_workday),
        USLaborDay,
        USThanksgivingDay,
        Holiday("Christmas Day", month=12, day=25, observance=nearest_workday),
    ]


def _ts(d: DateLike) -> pd.Timestamp:
    t = pd.Timestamp(d)
    if t.tzinfo is not None:
        t = t.tz_convert("America/New_York").tz_localize(None)
    t = t.normalize()
    if t < CALENDAR_START:
        raise CalendarError(f"{t.date()} is before the supported calendar start {CALENDAR_START.date()}")
    return t


def holidays(start: DateLike, end: DateLike) -> pd.DatetimeIndex:
    """Weekday full-day closures between start and end (inclusive)."""
    s, e = _ts(start), _ts(end)
    rule_based = NYSEHolidayCalendar().holidays(start=s, end=e)
    special = pd.DatetimeIndex(pd.to_datetime(list(SPECIAL_CLOSURES)))
    special = special[(special >= s) & (special <= e)]
    days = rule_based.union(special)
    return days[days.day_of_week < 5]


def trading_days(start: DateLike, end: DateLike) -> pd.DatetimeIndex:
    """NYSE trading days between start and end (inclusive), timezone-naive dates."""
    s, e = _ts(start), _ts(end)
    if e < s:
        return pd.DatetimeIndex([])
    weekdays = pd.bdate_range(s, e)
    return weekdays.difference(holidays(s, e))


def is_trading_day(d: DateLike) -> bool:
    """True when the NYSE is open for the full or a partial day on `d`."""
    t = _ts(d)
    return len(trading_days(t, t)) == 1


def next_trading_day(d: DateLike) -> pd.Timestamp:
    """First trading day strictly after `d`."""
    t = _ts(d)
    days = trading_days(t + pd.Timedelta(days=1), t + pd.Timedelta(days=14))
    return pd.Timestamp(days[0])


def previous_trading_day(d: DateLike) -> pd.Timestamp:
    """Last trading day strictly before `d`."""
    t = _ts(d)
    days = trading_days(t - pd.Timedelta(days=14), t - pd.Timedelta(days=1))
    if len(days) == 0:
        raise CalendarError(f"no trading day before {t.date()} inside the supported range")
    return pd.Timestamp(days[-1])
