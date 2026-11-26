"""Second-source cross-check through the adapter interface (recorded fixtures only)."""

import pandas as pd
import pytest

from atu2.core.errors import CrossCheckError, DataSourceError
from atu2.data.cross_check import StooqSource, compare_closes, cross_check
from atu2.data.settings import CrossCheckSettings
from tests.data.fakes import FIXTURES, FakeResponse, FakeStooq, make_bars

S = CrossCheckSettings(
    enabled=True, source="stooq", return_tolerance=0.005, max_mismatch_share=0.02, min_overlap_days=200
)


class _Http:
    def __init__(self, status=200, text=None):
        self.status, self.text, self.params = status, text, None

    def get(self, url, *, headers, params, timeout):
        self.params = params
        text = self.text if self.text is not None else (FIXTURES / "stooq_spy.txt").read_text(encoding="utf-8")
        return FakeResponse(self.status, text=text)


def test_stooq_source_parses_recorded_csv():
    http = _Http()
    s = StooqSource(http=http).daily_closes("SPY", pd.Timestamp("2024-01-02"), pd.Timestamp("2024-01-04"))
    assert list(s.index.strftime("%Y-%m-%d")) == ["2024-01-02", "2024-01-03", "2024-01-04"]
    assert s.iloc[1] == pytest.approx(454.73)
    assert http.params == {"s": "spy.us", "i": "d", "d1": "20240102", "d2": "20240104"}


@pytest.mark.parametrize("status,text,match", [(500, "", "HTTP 500"), (200, "No data", "no CSV")])
def test_stooq_errors(status, text, match):
    with pytest.raises(DataSourceError, match=match):
        StooqSource(http=_Http(status, text)).daily_closes(
            "SPY", pd.Timestamp("2024-01-02"), pd.Timestamp("2024-01-04")
        )


def test_agreeing_sources_pass_even_with_a_different_adjustment_base():
    bars = make_bars(n_days=300, dividends=(("2023-06-15", 0.3),))
    other = bars["adj_close"] * 1.37  # same returns, different level
    r = cross_check("AAA", bars, FakeStooq({"AAA": other}), S)
    assert r.passed and r.overlap_days == 299 and r.mismatch_days == 0


def test_disagreeing_source_fails():
    bars = make_bars(n_days=300)
    noisy = bars["adj_close"] * (1 + pd.Series(range(300), index=bars.index).mod(7).eq(0) * 0.02)
    with pytest.raises(CrossCheckError, match="disagree"):
        cross_check("AAA", bars, FakeStooq({"AAA": noisy}), S)


def test_too_little_overlap_fails():
    bars = make_bars(n_days=300)
    with pytest.raises(CrossCheckError, match="overlapping"):
        cross_check("AAA", bars, FakeStooq({"AAA": bars["adj_close"].iloc[:50]}), S)
    r = compare_closes("AAA", bars["adj_close"], bars["adj_close"].iloc[:0], S, "x")
    assert not r.passed and r.overlap_days == 0 and r.as_dict()["ticker"] == "AAA"
