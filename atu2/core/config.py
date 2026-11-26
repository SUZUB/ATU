"""Typed config loading (YAML -> frozen dataclasses).

Rules from the plan:
- one YAML file per experiment under configs/, validated by a typed schema at load time;
- unknown keys are errors, not warnings (UnknownConfigKeyError);
- every config is hashed (atu2.core.hashing.hash_config) and the hash is stored with the run;
- the gate thresholds live in their own frozen file, configs/gate.yaml, with a recorded hash.

Supported field types: bool, int, float (ints are accepted), str, Literal[...],
Optional[X] / X | None, list[X], tuple[X, ...], dict[str, X] and nested dataclasses.
"""

import dataclasses
import types
import typing
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, TypeVar

import yaml

from atu2.core.errors import (
    ConfigError,
    ConfigTypeError,
    HashMismatchError,
    MissingConfigKeyError,
    UnknownConfigKeyError,
)
from atu2.core.hashing import hash_config

T = TypeVar("T")


def _type_name(tp: Any) -> str:
    return getattr(tp, "__name__", None) or str(tp)


def _convert(value: Any, tp: Any, path: str) -> Any:
    """Checks `value` against type `tp` and converts it (nested dataclasses, tuples)."""
    origin = typing.get_origin(tp)
    args = typing.get_args(tp)

    if tp is Any:
        return value
    if origin is typing.Union or origin is types.UnionType:
        if value is None and type(None) in args:
            return None
        errors = []
        for arg in (a for a in args if a is not type(None)):
            try:
                return _convert(value, arg, path)
            except ConfigTypeError as e:
                errors.append(str(e))
        raise ConfigTypeError(f"{path}: {value!r} matches none of {[_type_name(a) for a in args]}")
    if origin is Literal:
        if value not in args:
            raise ConfigTypeError(f"{path}: {value!r} is not one of {list(args)}")
        return value
    if dataclasses.is_dataclass(tp) and isinstance(tp, type):
        if not isinstance(value, Mapping):
            raise ConfigTypeError(f"{path}: expected a mapping for {tp.__name__}, got {type(value).__name__}")
        return from_mapping(tp, value, path)
    if origin is list:
        if not isinstance(value, list):
            raise ConfigTypeError(f"{path}: expected a list, got {type(value).__name__}")
        return [_convert(v, args[0] if args else Any, f"{path}[{i}]") for i, v in enumerate(value)]
    if origin is tuple:
        if not isinstance(value, (list, tuple)):
            raise ConfigTypeError(f"{path}: expected a list, got {type(value).__name__}")
        if len(args) == 2 and args[1] is Ellipsis:
            return tuple(_convert(v, args[0], f"{path}[{i}]") for i, v in enumerate(value))
        if len(args) != len(value):
            raise ConfigTypeError(f"{path}: expected {len(args)} items, got {len(value)}")
        return tuple(_convert(v, a, f"{path}[{i}]") for i, (v, a) in enumerate(zip(value, args, strict=True)))
    if origin is dict:
        if not isinstance(value, Mapping):
            raise ConfigTypeError(f"{path}: expected a mapping, got {type(value).__name__}")
        kt, vt = args if args else (Any, Any)
        return {_convert(k, kt, f"{path}.<key>"): _convert(v, vt, f"{path}.{k}") for k, v in value.items()}
    if tp is bool:
        if not isinstance(value, bool):
            raise ConfigTypeError(f"{path}: expected bool, got {value!r}")
        return value
    if tp is int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ConfigTypeError(f"{path}: expected int, got {value!r}")
        return value
    if tp is float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ConfigTypeError(f"{path}: expected a number, got {value!r}")
        return float(value)
    if tp is str:
        if not isinstance(value, str):
            raise ConfigTypeError(f"{path}: expected str, got {value!r}")
        return value
    raise ConfigError(f"{path}: unsupported schema type {_type_name(tp)}")


def from_mapping(schema: type[T], data: Mapping[str, Any], path: str = "config") -> T:
    """
    Builds the dataclass `schema` from `data`.
    Raises UnknownConfigKeyError, MissingConfigKeyError or ConfigTypeError with the
    dotted path of the offending key.
    """
    if not (dataclasses.is_dataclass(schema) and isinstance(schema, type)):
        raise ConfigError(f"{_type_name(schema)} is not a dataclass schema")
    if not isinstance(data, Mapping):
        raise ConfigTypeError(f"{path}: expected a mapping, got {type(data).__name__}")
    hints = typing.get_type_hints(schema)
    fields = {f.name: f for f in dataclasses.fields(schema) if f.init}
    unknown = sorted(set(data) - set(fields))
    if unknown:
        raise UnknownConfigKeyError(f"{path}: unknown key(s) {unknown}; allowed: {sorted(fields)}")
    kwargs: dict[str, Any] = {}
    for name, f in fields.items():
        if name in data:
            kwargs[name] = _convert(data[name], hints[name], f"{path}.{name}")
        elif f.default is dataclasses.MISSING and f.default_factory is dataclasses.MISSING:
            raise MissingConfigKeyError(f"{path}: missing required key '{name}'")
    obj = schema(**kwargs)
    validate = getattr(obj, "validate", None)
    if callable(validate):
        validate()
    return obj


def read_yaml(path: str | Path) -> dict[str, Any]:
    """Reads a YAML file whose top level is a mapping (an empty file is an error)."""
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ConfigTypeError(f"{path}: top level must be a mapping")
    return data


def load_config(path: str | Path, schema: type[T]) -> T:
    """Reads a YAML file and validates it against `schema`."""
    return from_mapping(schema, read_yaml(path), path=Path(path).stem)


# --- gate config (configs/gate.yaml, frozen; see ADR 0008) ---------------------------------
@dataclass(frozen=True)
class GateConfig:
    """
    Gate B thresholds (frozen before the holdout is opened; any change needs an ADR and
    resets the trial count).

    central_cost_bps                 assumed cost, basis points per unit of turnover
    min_net_sharpe_at_central_cost   holdout net Sharpe (annualised) must be strictly above this
    min_deflated_sharpe_probability  deflated Sharpe probability, 0..1, must be at least this
    min_breakeven_cost_multiple      break-even cost / central cost must be at least this
    min_placebo_p_value              placebo p-value must be above this (no skill in the placebo)
    min_neighbour_sharpe_retention   neighbouring settings keep at least this share of the Sharpe
    reported_cost_levels_bps         cost levels every report shows
    """

    version: int
    central_cost_bps: float
    min_net_sharpe_at_central_cost: float
    min_deflated_sharpe_probability: float
    min_breakeven_cost_multiple: float
    min_placebo_p_value: float
    min_neighbour_sharpe_retention: float
    reported_cost_levels_bps: tuple[float, ...]

    def validate(self) -> None:
        """Range checks; raises ConfigTypeError on an impossible threshold."""
        for name in ("min_deflated_sharpe_probability", "min_placebo_p_value", "min_neighbour_sharpe_retention"):
            v = getattr(self, name)
            if not 0.0 <= v <= 1.0:
                raise ConfigTypeError(f"gate.{name}: {v} is outside [0, 1]")
        if self.central_cost_bps <= 0 or self.min_breakeven_cost_multiple <= 0:
            raise ConfigTypeError("gate: central_cost_bps and min_breakeven_cost_multiple must be > 0")
        if self.central_cost_bps not in self.reported_cost_levels_bps:
            raise ConfigTypeError("gate: central_cost_bps must be one of reported_cost_levels_bps")


def gate_hash(path: str | Path) -> str:
    """Hash of the gate config's parsed content (independent of formatting and line endings)."""
    return hash_config(read_yaml(path))


def load_gate(path: str | Path, recorded_hash_path: str | Path | None = None) -> GateConfig:
    """
    Loads configs/gate.yaml. With `recorded_hash_path`, also checks the content against
    the recorded hash and raises HashMismatchError if the frozen file was edited.
    """
    if recorded_hash_path is not None:
        expected = Path(recorded_hash_path).read_text(encoding="utf-8").split()[0]
        actual = gate_hash(path)
        if actual != expected:
            raise HashMismatchError(f"{path}: content hash {actual} != recorded {expected}")
    return load_config(path, GateConfig)
