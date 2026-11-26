"""FR-01 cache: second run downloads nothing new; rewrites trigger a rebuild; hashes per file (NFR-09)."""

import pandas as pd
import pytest

from atu2.core.errors import CacheRewriteError, DataSourceError
from atu2.core.hashing import hash_dataframe, hash_file
from atu2.data.cache import BarCache, history_rewritten, update_ticker
from tests.data.fakes import NOW, FakeAlpaca, client, make_bars

NY = "America/New_York"
KW = {"start": "2016-01-01", "overlap_days": 5, "tolerance": 1e-6}


def _setup(tmp_path, bars, **fake_kw):
    fake = FakeAlpaca({"AAA": bars}, **fake_kw)
    return fake, client(fake), BarCache(tmp_path / "raw" / "sip", "sip")


def test_first_run_full_second_run_downloads_nothing(tmp_path):
    bars = make_bars("2023-01-03", 400)
    fake, c, cache = _setup(tmp_path, bars)
    r1 = update_ticker(c, cache, "AAA", now=NOW, **KW)
    assert r1.action == "full" and r1.rows == r1.new_rows == 365  # through 2024-06-14
    n_calls = len(fake.calls)
    r2 = update_ticker(c, cache, "AAA", now=NOW, **KW)
    assert r2.action == "current" and r2.new_rows == 0
    assert len(fake.calls) == n_calls  # no request at all
    assert r2.data_hash == r1.data_hash


def test_new_days_are_appended_without_touching_old_rows(tmp_path):
    bars = make_bars("2023-01-03", 400)
    fake, c, cache = _setup(tmp_path, bars)
    early = pd.Timestamp("2024-03-01 21:00", tz=NY)
    update_ticker(c, cache, "AAA", now=early, **KW)
    before = cache.read("AAA")
    assert before is not None
    r = update_ticker(c, cache, "AAA", now=NOW, **KW)
    after = cache.read("AAA")
    assert r.action == "incremental" and r.new_rows > 0
    assert after is not None
    pd.testing.assert_frame_equal(after.loc[before.index], before)
    assert after.index[-1] == pd.Timestamp("2024-06-14")  # the last complete session at NOW
    assert not after.index.has_duplicates
    overlap_starts = {p["start"] for p in fake.calls[-4:]}
    assert all(s >= "2024-02-2" for s in overlap_starts)  # only the overlap was re-fetched


def test_dividend_rewrite_triggers_full_rebuild(tmp_path):
    plain = make_bars("2023-01-03", 400)
    fake, c, cache = _setup(tmp_path, plain)
    update_ticker(c, cache, "AAA", now=pd.Timestamp("2024-03-01 21:00", tz=NY), **KW)
    # a dividend going ex on 2024-03-05 rescales every earlier adjusted close
    fake.data["AAA"] = make_bars("2023-01-03", 400, dividends=(("2024-03-05", 1.0),))
    r = update_ticker(c, cache, "AAA", now=NOW, **KW)
    assert r.action == "rebuilt"
    cached = cache.read("AAA")
    assert cached is not None
    expected = fake.data["AAA"]["adj_close"]
    pd.testing.assert_series_equal(cached["adj_close"], expected[expected.index <= "2024-06-14"], check_freq=False)


def test_history_rewritten_detects_close_or_adjusted_changes():
    a = make_bars(n_days=20)
    assert not history_rewritten(a, a.copy(), 1e-6)
    b = a.copy()
    b.iloc[-1, b.columns.get_loc("adj_close")] *= 1.001
    assert history_rewritten(a, b, 1e-6)
    assert not history_rewritten(a, a.iloc[:0], 1e-6)


def test_manifest_records_hashes_and_changes_when_one_price_changes(tmp_path):
    bars = make_bars("2023-01-03", 50)
    cache = BarCache(tmp_path, "sip")
    h1 = cache.write("AAA", bars, "sip")
    m = cache.manifest()["AAA"]
    assert m["data_hash"] == h1 == hash_dataframe(bars)
    assert m["file_hash"] == hash_file(cache.path("AAA"))
    assert (m["rows"], m["first"], m["last"], m["feed"]) == (50, "2023-01-03", str(bars.index[-1].date()), "sip")
    changed = bars.copy()
    changed.iloc[10, 3] += 0.01
    assert cache.write("AAA", changed, "sip") != h1
    assert cache.tickers() == ["AAA"]
    pd.testing.assert_frame_equal(cache.read("AAA"), changed, check_freq=False)


def test_feeds_are_never_mixed(tmp_path):
    cache = BarCache(tmp_path, "sip")
    with pytest.raises(DataSourceError, match="iex"):
        cache.write("AAA", make_bars(n_days=5), "iex")


def test_empty_download_and_empty_rebuild_raise(tmp_path):
    fake, c, cache = _setup(tmp_path, make_bars("2023-01-03", 400))
    with pytest.raises(DataSourceError, match="no bars"):
        update_ticker(c, cache, "MISSING", now=NOW, **KW)
    update_ticker(c, cache, "AAA", now=pd.Timestamp("2024-03-01 21:00", tz=NY), **KW)
    changed = make_bars("2023-01-03", 400, dividends=(("2024-03-05", 1.0),))
    fake.data["AAA"] = changed

    class Vanishing(FakeAlpaca):
        def get(self, url, *, headers, params, timeout):
            if params["start"] < "2020":  # the full rebuild request
                self.data = {}
            return super().get(url, headers=headers, params=params, timeout=timeout)

    c2 = client(Vanishing({"AAA": changed}))
    with pytest.raises(CacheRewriteError):
        update_ticker(c2, cache, "AAA", now=NOW, **KW)


def test_missing_cache_file_reads_none(tmp_path):
    cache = BarCache(tmp_path / "nothing", "sip")
    assert cache.read("AAA") is None and cache.manifest() == {} and cache.tickers() == []
