"""Content hashing for files, configs and dataframes (NFR-09).

All hashes are lowercase SHA-256 hex digests. Config hashes are taken over canonical
JSON (sorted keys, no whitespace), so they do not depend on key order, YAML formatting
or line endings. Dataframe hashes cover the schema (column names, dtypes, index) and
every value, so changing a single price changes the hash.
"""

import dataclasses
import hashlib
import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from atu2.core.errors import HashMismatchError

_CHUNK_BYTES = 1 << 20


def hash_bytes(data: bytes) -> str:
    """SHA-256 hex digest of raw bytes."""
    return hashlib.sha256(data).hexdigest()


def hash_file(path: str | Path) -> str:
    """SHA-256 hex digest of a file's bytes, read in 1 MiB chunks."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(_CHUNK_BYTES), b""):
            h.update(chunk)
    return h.hexdigest()


def _to_plain(obj: Any) -> Any:
    """Converts dataclasses, tuples and numpy scalars into JSON-ready plain values."""
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {f.name: _to_plain(getattr(obj, f.name)) for f in dataclasses.fields(obj)}
    if isinstance(obj, Mapping):
        out = {}
        for k, v in obj.items():
            if not isinstance(k, str):
                raise TypeError(f"config keys must be strings, got {k!r}")
            out[k] = _to_plain(v)
        return out
    if isinstance(obj, (list, tuple)):
        return [_to_plain(v) for v in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, float) and not math.isfinite(obj):
        raise ValueError("non-finite floats cannot be hashed in a config")
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    raise TypeError(f"cannot hash value of type {type(obj).__name__}")


def canonical_json(obj: Any) -> str:
    """Canonical JSON text: sorted keys, compact separators, UTF-8, no NaN/inf."""
    return json.dumps(_to_plain(obj), sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def hash_config(obj: Any) -> str:
    """Hash of a config mapping or dataclass, independent of key order and formatting."""
    return hash_bytes(canonical_json(obj).encode("utf-8"))


def hash_dataframe(df: pd.DataFrame) -> str:
    """Hash of a dataframe's schema, index and values (row order matters)."""
    schema = {
        "columns": [str(c) for c in df.columns],
        "dtypes": [str(t) for t in df.dtypes],
        "index_name": str(df.index.name),
        "index_dtype": str(df.index.dtype),
        "shape": list(df.shape),
    }
    h = hashlib.sha256(canonical_json(schema).encode("utf-8"))
    row_hashes = pd.util.hash_pandas_object(df, index=True).to_numpy(dtype="<u8")
    h.update(row_hashes.tobytes())
    return h.hexdigest()


def verify_hash(actual: str, expected: str, what: str) -> None:
    """Raises HashMismatchError when two hashes differ."""
    if actual.strip().lower() != expected.strip().lower():
        raise HashMismatchError(f"{what}: hash {actual} does not match recorded {expected}")
