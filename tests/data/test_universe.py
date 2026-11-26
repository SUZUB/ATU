"""FR-03: written rule, point-in-time warm-up, and the PriceFrame contract."""

from pathlib import Path

import numpy as np
import pytest

from atu2.core.errors import ConfigTypeError, ContractError, UnknownConfigKeyError
from atu2.core.types import PriceFrame
from atu2.data.settings import load_data_config, load_universe
from atu2.data.universe import apply_rule, build_price_frame, missing_asset_classes
from tests.data.fakes import make_bars

REPO = Path(__file__).resolve().parents[2]
UNIVERSE = REPO / "configs" / "universe.yaml"


def test_repository_configs_load():
    u = load_universe(UNIVERSE)
    assert u.status == "proposed" and len(u.candidates) == 40
    assert len(set(u.tickers)) == 40
    assert {c.asset_class for c in u.candidates} == set(u.rule.required_asset_classes)
    assert u.rule.warm_up_days == 252 and u.rule.min_history_years == 12.0
    cfg = load_data_config(REPO / "configs" / "data.yaml")
    assert cfg.raw_dir == Path("data") / "raw" / "sip"
    assert cfg.validation.approved_moves == ()


def _universe_yaml(tmp_path, body: str) -> Path:
    p = tmp_path / "u.yaml"
    p.write_text(
        "status: proposed\nrule:\n  as_of: '2026-11-26'\n  min_history_years: 12.0\n"
        "  min_median_dollar_volume: 20000000.0\n  volume_lookback_days: 252\n  warm_up_days: 252\n"
        "  required_asset_classes: [us_equity, currency]\n" + body,
        encoding="utf-8",
    )
    return p


def test_bad_universe_files_are_rejected(tmp_path):
    good = "candidates:\n  - {ticker: SPY, asset_class: us_equity, inception: '1993-01-22', name: x}\n"
    with pytest.raises(ConfigTypeError, match="currency"):
        load_universe(_universe_yaml(tmp_path, good))
    dup = good + "  - {ticker: SPY, asset_class: currency, inception: '1993-01-22', name: y}\n"
    with pytest.raises(ConfigTypeError, match="duplicate"):
        load_universe(_universe_yaml(tmp_path, dup))
    with pytest.raises(ConfigTypeError):
        load_universe(_universe_yaml(tmp_path, good.replace("SPY", "spy")))
    with pytest.raises(ConfigTypeError):
        load_universe(_universe_yaml(tmp_path, good.replace("1993-01-22", "not-a-date")))
    with pytest.raises(UnknownConfigKeyError):
        load_universe(_universe_yaml(tmp_path, good.replace("name: x", "name: x, sector: tech")))


def test_rule_uses_fund_history_and_cached_volume():
    u = load_universe(UNIVERSE)
    no_data = apply_rule(u)
    assert all(r.eligible for r in no_data)  # every candidate launched 12+ years before as_of
    spy = make_bars("2016-01-04", 300, price=400.0)  # 400 * 1e6 shares = ~$400M a day
    thin = make_bars("2016-01-04", 300, price=10.0).assign(volume=1000.0)  # ~$10k a day
    res = {r.ticker: r for r in apply_rule(u, {"SPY": spy, "FXB": thin})}
    assert res["SPY"].eligible and res["SPY"].median_dollar_volume > 1e8
    assert res["SPY"].data_history_years == pytest.approx(1.19, abs=0.05)
    assert not res["FXB"].eligible and "dollar volume" in res["FXB"].reasons[0]
    assert not res["QQQ"].eligible and res["QQQ"].reasons == ("no cached data",)
    gaps = missing_asset_classes(u, list(res.values()))
    assert "us_equity" not in gaps  # SPY is eligible
    assert "currency" in gaps and "us_sector" in gaps  # no eligible asset left in these classes


def test_young_fund_and_late_data_are_excluded(tmp_path):
    p = _universe_yaml(
        tmp_path,
        "candidates:\n  - {ticker: NEW, asset_class: us_equity, inception: '2020-01-02', name: young}\n"
        "  - {ticker: UUP, asset_class: currency, inception: '2007-02-20', name: dollar}\n",
    )
    u = load_universe(p)
    res = {r.ticker: r for r in apply_rule(u, {"UUP": make_bars("2027-01-04", 10)})}
    assert not res["NEW"].eligible and "fund history" in res["NEW"].reasons[0]
    assert not res["UUP"].eligible and "on or before" in res["UUP"].reasons[0]


def test_asset_with_100_days_is_excluded_until_day_253():
    long = make_bars("2022-01-03", 600)
    short = make_bars(str(long.index[-100].date()), 100, seed=1)
    frame = build_price_frame({"OLD": long, "NEW": short}, warm_up_days=252)
    assert not frame.tradable["NEW"].any()  # only 100 days of history
    new253 = make_bars("2022-01-03", 300, seed=2)
    frame2 = build_price_frame({"OLD": long, "NEW": new253}, warm_up_days=252)
    listed = frame2.tradable["NEW"][frame2.dates >= new253.index[0]]
    assert not listed.iloc[:252].any()  # days 1..252 excluded
    assert listed.iloc[252]  # day 253 included
    assert listed.iloc[252:300].all()
    assert not frame2.tradable["NEW"].iloc[300:].any()  # no data after its last bar


def test_price_frame_contract_fill_rule_and_hash():
    a = make_bars("2023-01-03", 300)
    b = make_bars("2023-01-03", 300, seed=3)
    one_gap = b.drop(b.index[100])
    two_gap = b.drop(b.index[200:202])
    frame = build_price_frame({"A": a, "B1": one_gap, "B2": two_gap}, warm_up_days=10)
    assert isinstance(frame, PriceFrame) and frame.adjusted
    assert frame.tickers == ["A", "B1", "B2"] and frame.dates.is_unique and frame.dates.is_monotonic_increasing
    d1 = b.index[100]
    assert frame.adj_close.loc[d1, "B1"] == b["adj_close"].iloc[99]  # forward-filled once
    assert frame.volume.loc[d1, "B1"] == 0.0 and frame.tradable.loc[d1, "B1"]
    assert frame.adj_close.loc[b.index[200], "B2"] == b["adj_close"].iloc[199]
    assert np.isnan(frame.adj_close.loc[b.index[201], "B2"])  # second missing day stays missing
    assert not frame.tradable.loc[b.index[201], "B2"]
    same = build_price_frame({"A": a, "B1": one_gap, "B2": two_gap}, warm_up_days=10)
    assert same.data_hash == frame.data_hash
    a2 = a.copy()
    a2.iloc[5, a2.columns.get_loc("adj_close")] += 0.01
    assert build_price_frame({"A": a2, "B1": one_gap, "B2": two_gap}, warm_up_days=10).data_hash != frame.data_hash


def test_price_frame_rejects_broken_contracts():
    frame = build_price_frame({"A": make_bars(n_days=30)}, warm_up_days=5)
    fields = {f: getattr(frame, f) for f in ("open", "high", "low", "close", "volume", "adj_close", "tradable")}
    with pytest.raises(ContractError, match="share"):
        PriceFrame(**{**fields, "open": fields["open"].iloc[1:]}, adjusted=True, data_hash="x")
    with pytest.raises(ContractError, match="boolean"):
        PriceFrame(**{**fields, "tradable": fields["tradable"].astype(int)}, adjusted=True, data_hash="x")
    rev = {k: v.iloc[::-1] for k, v in fields.items()}
    with pytest.raises(ContractError, match="sorted"):
        PriceFrame(**rev, adjusted=True, data_hash="x")
    tz = {k: v.tz_localize("UTC") for k, v in fields.items()}
    with pytest.raises(ContractError, match="naive"):
        PriceFrame(**tz, adjusted=True, data_hash="x")
    plain = {k: v.reset_index(drop=True) for k, v in fields.items()}
    with pytest.raises(ContractError, match="DatetimeIndex"):
        PriceFrame(**plain, adjusted=True, data_hash="x")
    with pytest.raises(ContractError):
        build_price_frame({}, warm_up_days=5)
