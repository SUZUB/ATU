"""Alpaca client against recorded-format responses (no network)."""

import pandas as pd
import pytest

from atu2.core.errors import DataSourceError, RateLimitError
from atu2.data.alpaca_client import BAR_COLUMNS, last_complete_session
from tests.data.fakes import CREDS, NOW, FakeAlpaca, FakeResponse, RecordedHttp, client, make_bars

NY = "America/New_York"


def test_parses_recorded_pages_and_joins_adjusted_close():
    http = RecordedHttp()
    c = client(http, now=pd.Timestamp("2024-01-10 21:00", tz=NY))
    df = c.fetch("SPY", "2024-01-02", "2024-01-04")
    assert list(df.columns) == list(BAR_COLUMNS)
    assert list(df.index.strftime("%Y-%m-%d")) == ["2024-01-02", "2024-01-03", "2024-01-04"]
    assert df.index.tz is None and df.index.name == "date"
    assert df.loc["2024-01-03", "close"] == pytest.approx(468.79)
    assert df.loc["2024-01-03", "adj_close"] == pytest.approx(468.79 * 0.97, rel=1e-4)
    assert df.attrs["feed"] == "sip"
    raw_calls = [p for p in http.calls if p["adjustment"] == "raw"]
    assert len(raw_calls) == 2 and raw_calls[1]["page_token"]  # followed next_page_token
    assert {p["timeframe"] for p in http.calls} == {"1Day"}


def test_round_trip_through_fake_server_with_pagination_and_chunks():
    bars = make_bars("2016-01-04", 2000, split=("2020-08-31", 4.0), dividends=(("2019-03-15", 0.4),))
    fake = FakeAlpaca({"AAA": bars}, page_size=97)
    c = client(fake, chunk_years=2)
    df = c.fetch("AAA", "2016-01-01")
    expected = bars[bars.index <= last_complete_session(NOW)]
    pd.testing.assert_frame_equal(df, expected, check_freq=False, check_exact=False, rtol=1e-12)
    starts = sorted({p["start"] for p in fake.calls})
    assert len(starts) >= 4  # several date chunks
    assert all(p["limit"] == 10_000 for p in fake.calls)


def test_keys_only_in_headers():
    fake = FakeAlpaca({"AAA": make_bars(n_days=20)})
    client(fake).fetch("AAA", "2023-01-01")
    assert all(h["APCA-API-KEY-ID"] == CREDS.key_id for h in fake.headers_seen)
    assert all(CREDS.secret not in str(v) for p in fake.calls for v in p.values())


def test_rate_limit_backs_off_then_succeeds():
    fake = FakeAlpaca({"AAA": make_bars(n_days=20)}, rate_limited=3)
    c = client(fake)
    df = c.fetch("AAA", "2023-01-01")
    assert len(df) == 20
    assert c.sleeps[:3] == [2.0, 4.0, 8.0]  # type: ignore[attr-defined]


def test_rate_limit_gives_up_with_named_error():
    fake = FakeAlpaca({"AAA": make_bars(n_days=20)}, rate_limited=100)
    with pytest.raises(RateLimitError):
        client(fake, max_retries=2).fetch("AAA", "2023-01-01")


def test_sip_denied_falls_back_to_iex_for_every_row():
    bars = make_bars(n_days=300)
    fake = FakeAlpaca({"AAA": bars}, sip_denied=True, page_size=60)
    c = client(fake, chunk_years=1)
    df = c.fetch("AAA", "2023-01-01")
    assert c.feed == "iex" and df.attrs["feed"] == "iex" and len(df) == 300
    served = [p["feed"] for p in fake.calls]
    assert served[0] == "sip" and served.count("sip") == 1


def test_sip_denied_without_fallback_raises():
    fake = FakeAlpaca({"AAA": make_bars(n_days=20)}, sip_denied=True)
    with pytest.raises(DataSourceError, match="403"):
        client(fake, allow_iex_fallback=False).fetch("AAA", "2023-01-01")


def test_http_error_and_bad_payloads_raise():
    with pytest.raises(DataSourceError, match="500"):
        client(FakeAlpaca({}, fail_status=500)).fetch("AAA", "2023-01-01")

    class Bad:
        def __init__(self, payload):
            self.payload = payload

        def get(self, url, *, headers, params, timeout):
            return FakeResponse(200, self.payload)

    with pytest.raises(DataSourceError, match="unexpected"):
        client(Bad({"bars": {"AAA": "oops"}})).fetch("AAA", "2023-01-02")
    with pytest.raises(DataSourceError, match="missing fields"):
        client(Bad({"bars": {"AAA": [{"t": "2023-01-03T05:00:00Z", "c": 1.0}]}})).fetch("AAA", "2023-01-02")


def test_unknown_symbol_and_empty_range():
    assert client(FakeAlpaca({})).fetch("NONE", "2023-01-01").empty
    assert client(FakeAlpaca({})).fetch("NONE", "2030-01-01").empty  # start after the last session


@pytest.mark.parametrize(
    "now,expected",
    [
        ("2024-06-14 15:00", "2024-06-13"),  # Friday, market open: today's bar unfinished
        ("2024-06-14 19:59", "2024-06-13"),  # after the close but before settle time
        ("2024-06-14 20:00", "2024-06-14"),
        ("2024-06-15 10:00", "2024-06-14"),  # Saturday
        ("2024-06-17 09:00", "2024-06-14"),  # Monday morning
        ("2024-07-04 22:00", "2024-07-03"),  # holiday
    ],
)
def test_last_complete_session(now, expected):
    assert last_complete_session(pd.Timestamp(now, tz=NY)) == pd.Timestamp(expected)
    assert last_complete_session(pd.Timestamp(now)) == pd.Timestamp(expected)  # naive = New York


def test_todays_unfinished_bar_is_never_returned():
    bars = make_bars("2024-06-03", 10)  # through Friday 2024-06-14
    during_session = pd.Timestamp("2024-06-14 14:00", tz=NY)
    df = client(FakeAlpaca({"AAA": bars}), now=during_session).fetch("AAA", "2024-06-01")
    assert df.index[-1] == pd.Timestamp("2024-06-13")
