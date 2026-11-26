# ADR 0004: Data sources, adjustment policy and cache format

Status: Draft (M0), awaiting review · Date: 2026-11-26

## Context
Free, consistent daily history is needed for about 40 ETFs; the futures set is only for
reproducing the paper.

## Decision
- Primary data: Alpaca daily bars with split and dividend adjustment (D4, total-return
  prices throughout). Reproduction data: the free Nasdaq Data Link continuous futures set (D5).
- Cache: one Parquet file per ticker under `data/raw/`, append-only. Every update re-fetches
  the last few days; a mismatch means a corporate action rewrote history and the file is rebuilt.
- Raw files are never edited after download. Derived files carry the hash of their inputs
  in their name. Timestamps are stored in UTC, dates on the exchange calendar.
- A second free daily-price source cross-checks closes; disagreement above a tolerance fails the build.
- Keys come only from environment variables (D8). Decided 2026-11-26: both naming schemes
  are accepted (`atu2/core/credentials.py`). Primary: `APCA_API_KEY_ID` / `APCA_API_SECRET_KEY`
  (Alpaca's own names, also used by ORB v2 and Colab). Aliases: `ALPACA_API_KEY` /
  `ALPACA_API_SECRET` (the plan's names). If a primary and its alias are both set with
  different values, `ConflictingCredentialError` is raised; nothing is picked silently.
  All four names are always redacted from logs and checked by the repository secret scan.

## Consequences
- Free data has no delisted funds: survivorship bias is a documented limitation (R10).
