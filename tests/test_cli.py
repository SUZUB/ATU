"""FR-22 (data commands): each command runs end to end offline; no key ever reaches the output."""

from pathlib import Path

import pandas as pd
import pytest
import yaml

from atu2 import cli
from atu2.core.credentials import CREDENTIAL_ENV_VARS
from tests.data.fakes import CREDS, FakeAlpaca, FakeStooq, client, make_bars

REPO = Path(__file__).resolve().parents[1]
TICKERS = ["SPY", "TLT", "GLD"]


@pytest.fixture
def setup(tmp_path, monkeypatch):
    cfg = yaml.safe_load((REPO / "configs" / "data.yaml").read_text(encoding="utf-8"))
    cfg["download"]["data_dir"] = str(tmp_path / "data")
    cfg["cross_check"]["min_overlap_days"] = 100
    data_cfg = tmp_path / "data.yaml"
    data_cfg.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    bars = {t: make_bars("2023-01-03", 400, seed=i, price=100.0 * (i + 1)) for i, t in enumerate(TICKERS)}
    fake = FakeAlpaca(bars, page_size=120)
    monkeypatch.setattr(cli, "make_client", lambda c, creds: client(fake, chunk_years=c.download.chunk_years))
    monkeypatch.setattr(cli, "make_second_source", lambda c: FakeStooq({t: b["adj_close"] for t, b in bars.items()}))
    for name in CREDENTIAL_ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("APCA_API_KEY_ID", CREDS.key_id)
    monkeypatch.setenv("APCA_API_SECRET_KEY", CREDS.secret)
    common = ["--data-config", str(data_cfg), "--universe", str(REPO / "configs" / "universe.yaml")]
    return fake, common, tmp_path


def run(capsys, *argv):
    code = cli.main(list(argv))
    out = capsys.readouterr()
    assert CREDS.key_id not in out.out + out.err and CREDS.secret not in out.out + out.err
    return code, out.out


def test_download_validate_report_universe_check_end_to_end(setup, capsys):
    fake, common, tmp = setup
    code, out = run(capsys, "data", "download", *common, "--tickers", *TICKERS)
    assert code == 0, out
    assert "3/3 ok" in out and "full" in out
    assert len(list((tmp / "data" / "raw" / "sip").glob("*_1d.parquet"))) == 3
    assert list((tmp / "data" / "reports").glob("download_*.json"))

    n = len(fake.calls)
    code, out = run(capsys, "data", "download", *common, "--tickers", *TICKERS)
    assert code == 0 and out.count("current") == 3
    assert len(fake.calls) == n + 2  # only the feed probe (raw + adjusted); no ticker re-download

    code, out = run(capsys, "data", "validate", *common)
    assert code == 0 and "3 tickers, 0 error(s)" in out

    code, out = run(capsys, "data", "report", *common)
    assert code == 0 and "paste this back" in out and "tickers cached: 3/40" in out
    assert "not cached:" in out and "SPY" in out

    code, out = run(capsys, "data", "universe-check", *common)
    assert "EXCLUDED: no cached data" in out and "proposed universe (3)" in out
    assert code == 1  # most asset classes have no cached data yet


def test_download_reports_problems_and_continues(setup, capsys):
    fake, common, _ = setup
    fake.data["TLT"] = fake.data["TLT"].copy()
    fake.data["TLT"].iloc[100:, fake.data["TLT"].columns.get_loc("adj_close")] *= 1.5  # bad jump
    code, out = run(capsys, "data", "download", *common, "--tickers", "SPY", "TLT", "XYZ", "--no-cross-check")
    assert code == 1
    assert "PROBLEMS: TLT, XYZ" in out and "no bars returned" in out and "skipped" in out


def test_cross_check_failure_is_reported(setup, capsys, monkeypatch):
    fake, common, _ = setup
    monkeypatch.setattr(cli, "make_second_source", lambda c: FakeStooq({"SPY": fake.data["SPY"]["adj_close"] * 0 + 1}))
    code, out = run(capsys, "data", "download", *common, "--tickers", "SPY")
    assert code == 1 and "FAIL" in out


def test_missing_keys_exit_2_with_a_clear_message(setup, capsys, monkeypatch):
    _, common, _ = setup
    monkeypatch.delenv("APCA_API_KEY_ID")
    code, out = run(capsys, "data", "download", *common)
    assert code == 2 and "APCA_API_KEY_ID" in out


def test_conflicting_keys_exit_2(setup, capsys, monkeypatch):
    _, common, _ = setup
    monkeypatch.setenv("ALPACA_API_KEY", "different-value")
    code, out = run(capsys, "data", "download", *common)
    assert code == 2 and "different values" in out


def test_empty_cache_and_bad_config(setup, capsys, tmp_path):
    _, common, _ = setup
    assert run(capsys, "data", "validate", *common)[0] == 2
    assert run(capsys, "data", "report", *common)[0] == 2
    code, out = run(capsys, "data", "universe-check", *common)
    assert code == 0 and "proposed universe (40)" in out and "volume not checked" in out
    bad = tmp_path / "bad.yaml"
    bad.write_text("download: {}\n", encoding="utf-8")
    assert run(capsys, "data", "validate", "--data-config", str(bad))[0] == 2
    assert run(capsys, "data", "validate", "--data-config", str(tmp_path / "missing.yaml"))[0] == 2


def test_validate_flags_a_corrupted_cache(setup, capsys):
    _, common, tmp = setup
    run(capsys, "data", "download", *common, "--tickers", "SPY", "--no-cross-check")
    path = tmp / "data" / "raw" / "sip" / "SPY_1d.parquet"
    df = pd.read_parquet(path)
    df.loc[10, "high"] = df.loc[10, "low"] * 0.5
    df.to_parquet(path, index=False)
    code, out = run(capsys, "data", "validate", *common)
    assert code == 1 and "high below low" in out
    code, out = run(capsys, "data", "validate", *common, "--tickers", "SPY", "QQQ")
    assert "not in cache" in out


def test_iex_fallback_uses_the_iex_cache(setup, capsys, monkeypatch, tmp_path):
    fake, common, tmp = setup
    fake.sip_denied = True
    code, out = run(capsys, "data", "download", *common, "--tickers", "SPY", "--no-cross-check")
    assert code == 0 and "using 'iex'" in out
    assert (tmp / "data" / "raw" / "iex" / "SPY_1d.parquet").exists()
    assert not (tmp / "data" / "raw" / "sip").exists()
    code, out = run(capsys, "data", "validate", *common, "--feed", "iex")
    assert code == 0


def test_real_client_factory_uses_config():
    cfg = cli.load_data_config(REPO / "configs" / "data.yaml")
    c = cli.make_client(cfg, CREDS)
    assert (c.feed, c.chunk_years, c.settle_time) == ("sip", 5, "20:00") and c.http is None
    assert cli.make_second_source(cfg).name == "stooq"
