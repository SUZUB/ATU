# ADR 0003: PyTorch for our models; reference repo isolated

Status: Draft (M0), awaiting review · Date: 2026-11-26

## Context
The reference Momentum Transformer code is several years old and has its own dependencies.

## Decision
- Our model code uses PyTorch (D2) and runs on CPU; a GPU is optional.
- The reference repository runs only in its own virtual environment under `ref/`
  (git-ignored). Nothing in `atu2/` imports it. We exchange data through CSV files of positions.
- No reference code is copied into `atu2/` until its licence is confirmed (see docs/licences.md).

## Alternatives
- TensorFlow to match the reference: two frameworks to maintain and test.

## Consequences
- Model results can differ from the reference implementation; M3 measures that gap.
