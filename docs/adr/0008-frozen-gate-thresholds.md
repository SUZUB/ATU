# ADR 0008: Frozen gate thresholds

Status: Draft (M0), awaiting review · Date: 2026-11-26

## Decision
Gate B thresholds live in `configs/gate.yaml`:

| Condition | Threshold |
|---|---|
| Holdout net Sharpe at the central cost (2 bps) | > 0 |
| Deflated Sharpe probability (registry trial count) | >= 0.95 |
| Break-even cost / central cost | >= 2 |
| Placebo p-value (placebo must show no skill) | > 0.10 |
| Share of Sharpe kept by neighbouring settings | >= 0.5 |
| Cost levels in every report | 0, 1, 2, 3, 5 bps |

Content hash (SHA-256 of the canonical JSON of the parsed YAML, so comments and line
endings do not matter): `ffea711912cad275986166662e4f8030f98e3697207863b804cafe67010fe21b`, recorded in `configs/gate.sha256`.
`python scripts/gate_hash.py` prints the hash and checks the record; the test suite fails
if the file and the record disagree.

## Rules
- Never edit the gate to make a result pass. A change needs a new ADR, resets the trial
  count, and is not allowed once the holdout has been opened.
- This ADR must be accepted before the holdout is touched (D7).
