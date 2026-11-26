"""Content hashes (NFR-09): stable for equal content, different after any change."""

import dataclasses

import numpy as np
import pandas as pd
import pytest

from atu2.core.errors import HashMismatchError
from atu2.core.hashing import canonical_json, hash_bytes, hash_config, hash_dataframe, hash_file, verify_hash


def _prices() -> pd.DataFrame:
    idx = pd.DatetimeIndex(pd.bdate_range("2024-01-02", periods=50), name="date")
    rng = np.random.default_rng(0)
    return pd.DataFrame(100 + rng.normal(0, 1, (50, 3)).cumsum(axis=0), index=idx, columns=["SPY", "TLT", "GLD"])


def test_dataframe_hash_changes_when_one_price_changes():
    df = _prices()
    h = hash_dataframe(df)
    assert hash_dataframe(df.copy()) == h
    changed = df.copy()
    changed.iloc[17, 1] += 0.01
    assert hash_dataframe(changed) != h


def test_dataframe_hash_covers_schema_and_index():
    df = _prices()
    h = hash_dataframe(df)
    assert hash_dataframe(df.rename(columns={"SPY": "IVV"})) != h
    assert hash_dataframe(df.astype("float32")) != h
    assert hash_dataframe(df.iloc[::-1]) != h
    shifted = df.copy()
    shifted.index = shifted.index + pd.Timedelta(days=1)
    assert hash_dataframe(shifted) != h


def test_config_hash_is_order_independent_and_value_sensitive():
    a = {"seed": 1, "costs": {"equity": 2.0, "borrow": 0.5}, "lookbacks": [1, 21]}
    b = {"lookbacks": [1, 21], "costs": {"borrow": 0.5, "equity": 2.0}, "seed": 1}
    assert hash_config(a) == hash_config(b)
    assert hash_config({**a, "seed": 2}) != hash_config(a)
    assert canonical_json({"b": 1, "a": np.float64(0.5), "t": (1, 2)}) == '{"a":0.5,"b":1,"t":[1,2]}'


def test_config_hash_of_dataclass_equals_its_mapping():
    @dataclasses.dataclass(frozen=True)
    class C:
        seed: int
        band: float

    assert hash_config(C(seed=1, band=0.1)) == hash_config({"band": 0.1, "seed": 1})


@pytest.mark.parametrize("bad", [{"x": float("nan")}, {"x": {1, 2}}, {1: "non-string key"}])
def test_unhashable_configs_raise(bad):
    with pytest.raises((TypeError, ValueError)):
        hash_config(bad)


def test_file_hash_matches_bytes_hash(tmp_path):
    p = tmp_path / "f.bin"
    data = bytes(range(256)) * 9000  # larger than one read chunk
    p.write_bytes(data)
    assert hash_file(p) == hash_bytes(data)
    assert len(hash_file(p)) == 64


def test_verify_hash():
    verify_hash("ABC", "abc ", "x")
    with pytest.raises(HashMismatchError):
        verify_hash("abc", "abd", "x")
