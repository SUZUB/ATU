"""configs/gate.yaml is valid and frozen: its content must match the recorded hash."""

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from atu2.core.config import GateConfig, from_mapping, gate_hash, load_gate, read_yaml
from atu2.core.errors import ConfigTypeError, HashMismatchError, UnknownConfigKeyError

REPO = Path(__file__).resolve().parents[2]
GATE = REPO / "configs" / "gate.yaml"
RECORD = REPO / "configs" / "gate.sha256"


def test_gate_loads_with_plan_thresholds():
    g = load_gate(GATE, RECORD)
    assert g.central_cost_bps == 2.0
    assert g.min_net_sharpe_at_central_cost == 0.0
    assert g.min_deflated_sharpe_probability == 0.95
    assert g.min_breakeven_cost_multiple == 2.0
    assert g.min_placebo_p_value == 0.10
    assert g.min_neighbour_sharpe_retention == 0.5
    assert g.reported_cost_levels_bps == (0.0, 1.0, 2.0, 3.0, 5.0)


def test_recorded_hash_matches_frozen_gate():
    recorded = RECORD.read_text(encoding="utf-8").split()[0]
    assert gate_hash(GATE) == recorded


def test_edited_gate_is_detected(tmp_path):
    edited = tmp_path / "gate.yaml"
    shutil.copy(GATE, edited)
    edited.write_text(
        edited.read_text(encoding="utf-8").replace(
            "min_deflated_sharpe_probability: 0.95", "min_deflated_sharpe_probability: 0.90"
        ),
        encoding="utf-8",
    )
    with pytest.raises(HashMismatchError):
        load_gate(edited, RECORD)


def test_comment_only_change_keeps_the_hash(tmp_path):
    copy = tmp_path / "gate.yaml"
    copy.write_text("# reformatted\n" + GATE.read_text(encoding="utf-8").replace("\n", "\r\n"), encoding="utf-8")
    assert gate_hash(copy) == gate_hash(GATE)


@pytest.mark.parametrize(
    "change",
    [
        {"min_deflated_sharpe_probability": 1.5},
        {"min_placebo_p_value": -0.1},
        {"central_cost_bps": 0.0},
        {"central_cost_bps": 4.0},
    ],
)
def test_impossible_thresholds_rejected(change):
    data = {**read_yaml(GATE), **change}
    with pytest.raises(ConfigTypeError):
        from_mapping(GateConfig, data)


def test_unknown_gate_key_rejected():
    with pytest.raises(UnknownConfigKeyError):
        from_mapping(GateConfig, {**read_yaml(GATE), "min_sharpe": 1.0})


def test_hash_script_reports_match():
    out = subprocess.run(
        [sys.executable, str(REPO / "scripts" / "gate_hash.py")], capture_output=True, text=True, check=False
    )
    assert out.returncode == 0, out.stdout + out.stderr
    assert "matches the recorded hash" in out.stdout
