"""Prints the content hash of configs/gate.yaml and checks it against configs/gate.sha256.

python scripts/gate_hash.py            # print hash, exit 1 if it differs from the record
python scripts/gate_hash.py --record   # write the record (only when an ADR approves a change)
"""

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from atu2.core.config import gate_hash, load_gate  # noqa: E402

GATE = REPO / "configs" / "gate.yaml"
RECORD = REPO / "configs" / "gate.sha256"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--record", action="store_true", help="write configs/gate.sha256")
    args = parser.parse_args()

    load_gate(GATE)  # schema check
    digest = gate_hash(GATE)
    print(f"{digest}  configs/gate.yaml")
    if args.record:
        RECORD.write_text(f"{digest}  configs/gate.yaml\n", encoding="utf-8", newline="\n")
        print("recorded in configs/gate.sha256")
        return 0
    if not RECORD.exists():
        print("no recorded hash (configs/gate.sha256 missing)")
        return 1
    recorded = RECORD.read_text(encoding="utf-8").split()[0]
    if recorded != digest:
        print(f"MISMATCH: recorded {recorded}")
        return 1
    print("matches the recorded hash")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
