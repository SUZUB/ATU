"""Typed config loader: unknown keys, missing keys and wrong types are named errors."""

from dataclasses import dataclass, field
from typing import Literal

import pytest

from atu2.core.config import from_mapping, load_config, read_yaml
from atu2.core.errors import ConfigError, ConfigTypeError, MissingConfigKeyError, UnknownConfigKeyError
from atu2.core.hashing import hash_config


@dataclass(frozen=True)
class Costs:
    equity_bps: float
    borrow_bps: float = 0.0


@dataclass(frozen=True)
class Experiment:
    name: str
    seed: int
    model: Literal["lstm", "tft"]
    costs: Costs
    lookbacks: tuple[int, ...] = (1, 21, 63)
    tickers: list[str] = field(default_factory=list)
    band: float | None = None
    weights: dict[str, float] = field(default_factory=dict)
    verbose: bool = False


GOOD = {
    "name": "tft_base",
    "seed": 1,
    "model": "tft",
    "costs": {"equity_bps": 2},
    "lookbacks": [1, 21, 63, 126, 252],
    "tickers": ["SPY", "TLT"],
    "weights": {"turn": 0.1},
}


def test_valid_config_builds_typed_objects():
    cfg = from_mapping(Experiment, GOOD)
    assert cfg.costs == Costs(equity_bps=2.0)
    assert isinstance(cfg.costs.equity_bps, float)
    assert cfg.lookbacks == (1, 21, 63, 126, 252)
    assert cfg.band is None and cfg.verbose is False


def test_unknown_key_is_an_error_with_path():
    with pytest.raises(UnknownConfigKeyError, match=r"config.costs.*equity_bp"):
        from_mapping(Experiment, {**GOOD, "costs": {"equity_bp": 2}})
    with pytest.raises(UnknownConfigKeyError):
        from_mapping(Experiment, {**GOOD, "extra": 1})


def test_missing_key_is_an_error():
    bad = dict(GOOD)
    del bad["seed"]
    with pytest.raises(MissingConfigKeyError, match="seed"):
        from_mapping(Experiment, bad)


@pytest.mark.parametrize(
    "key,value",
    [
        ("seed", "1"),
        ("seed", True),
        ("seed", 1.5),
        ("model", "gru"),
        ("lookbacks", 5),
        ("lookbacks", [1, "x"]),
        ("tickers", "SPY"),
        ("band", "wide"),
        ("verbose", "yes"),
        ("weights", ["turn"]),
        ("costs", 3),
        ("name", 3),
    ],
)
def test_wrong_types_raise(key, value):
    with pytest.raises(ConfigTypeError):
        from_mapping(Experiment, {**GOOD, key: value})


def test_fixed_length_tuple_and_non_dataclass_schema():
    @dataclass(frozen=True)
    class Pair:
        fast_slow: tuple[int, int]

    assert from_mapping(Pair, {"fast_slow": [8, 24]}).fast_slow == (8, 24)
    with pytest.raises(ConfigTypeError):
        from_mapping(Pair, {"fast_slow": [8, 24, 3]})
    with pytest.raises(ConfigTypeError):
        from_mapping(Pair, {"fast_slow": {"a": 1}})
    with pytest.raises(ConfigError):
        from_mapping(dict, {})  # type: ignore[arg-type]
    with pytest.raises(ConfigTypeError):
        from_mapping(Pair, [1, 2])  # type: ignore[arg-type]


def test_unsupported_schema_type():
    @dataclass(frozen=True)
    class Odd:
        x: set[int]

    with pytest.raises(ConfigError, match="unsupported"):
        from_mapping(Odd, {"x": [1]})


def test_load_from_yaml_and_hash_is_format_independent(tmp_path):
    a = tmp_path / "a.yaml"
    b = tmp_path / "b.yaml"
    a.write_text("name: x\nseed: 1\nmodel: lstm\ncosts: {equity_bps: 2.0}\n", encoding="utf-8")
    b.write_text("# comment\r\ncosts:\r\n  equity_bps: 2.0\r\nmodel: lstm\r\nseed: 1\r\nname: x\r\n", encoding="utf-8")
    assert load_config(a, Experiment) == load_config(b, Experiment)
    assert hash_config(read_yaml(a)) == hash_config(read_yaml(b))
    assert hash_config(load_config(a, Experiment)) == hash_config(load_config(b, Experiment))


def test_yaml_top_level_must_be_mapping(tmp_path):
    p = tmp_path / "list.yaml"
    p.write_text("- 1\n- 2\n", encoding="utf-8")
    with pytest.raises(ConfigTypeError):
        read_yaml(p)
    empty = tmp_path / "empty.yaml"
    empty.write_text("", encoding="utf-8")
    with pytest.raises(ConfigTypeError):
        read_yaml(empty)
