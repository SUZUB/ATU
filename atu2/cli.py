"""Command line entry point (FR-22). Commands arrive milestone by milestone; M1 adds `data`.

    atu2 data download        download/update the cache, validate, cross-check (needs keys)
    atu2 data validate        validate every cached ticker (no network)
    atu2 data report          paste-back summary: counts, date ranges, hashes, findings (no network)
    atu2 data universe-check  apply the universe rule; uses cached data when present (no network)

Keys are read from the environment only (APCA_API_KEY_ID / APCA_API_SECRET_KEY, or the
aliases ALPACA_API_KEY / ALPACA_API_SECRET). Nothing printed or logged contains a key.
Exit codes: 0 ok, 1 data problems found, 2 setup error (config, keys, empty cache).
"""

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pandas as pd

from atu2.core.credentials import AlpacaCredentials, alpaca_credentials
from atu2.core.errors import ATUError, ConfigError, ConflictingCredentialError, DataError, MissingSecretError
from atu2.core.logging import configure_logging, get_logger, new_run_id
from atu2.data.alpaca_client import AlpacaDailyClient, last_complete_session
from atu2.data.cache import BarCache, update_ticker
from atu2.data.cross_check import DailyPriceSource, StooqSource, cross_check
from atu2.data.settings import DataConfig, load_data_config, load_universe
from atu2.data.universe import apply_rule, build_price_frame, missing_asset_classes
from atu2.data.validators import findings

DEFAULT_DATA_CONFIG = "configs/data.yaml"
DEFAULT_UNIVERSE = "configs/universe.yaml"
PROBE_TICKER = "SPY"

log = get_logger("cli")


# ----------------------------------------------------------------- factories (replaced in tests)
def make_client(cfg: DataConfig, creds: AlpacaCredentials) -> AlpacaDailyClient:
    """Alpaca client built from configs/data.yaml."""
    d = cfg.download
    return AlpacaDailyClient(
        credentials=creds,
        feed=d.feed,
        allow_iex_fallback=d.allow_iex_fallback,
        chunk_years=d.chunk_years,
        pause_seconds=d.pause_seconds,
        max_retries=d.max_retries,
        settle_time=d.settle_time,
    )


def make_second_source(cfg: DataConfig) -> DailyPriceSource:
    """Second price source named in configs/data.yaml."""
    return StooqSource()


# ----------------------------------------------------------------- helpers
def _out(text: str = "") -> None:
    sys.stdout.write(text + "\n")


def _write_report(cfg: DataConfig, name: str, payload: dict[str, Any]) -> Path:
    folder = Path(cfg.download.data_dir) / "reports"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{name}.json"
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return path


def _cache_for(cfg: DataConfig, feed: str | None) -> BarCache:
    f = feed or cfg.download.feed
    return BarCache(Path(cfg.download.data_dir) / "raw" / f, f)


def _validate_cache(cfg: DataConfig, cache: BarCache, tickers: Sequence[str]) -> tuple[list[dict[str, str]], int]:
    rows: list[dict[str, str]] = []
    n_err = 0
    for t in tickers:
        bars = cache.read(t)
        if bars is None:
            rows.append({"ticker": t, "check": "cache", "severity": "error", "message": "not in cache"})
            n_err += 1
            continue
        for f in findings(t, bars, cfg.validation):
            rows.append(f.as_dict())
            n_err += f.severity == "error"
    return rows, n_err


# ----------------------------------------------------------------- commands
def cmd_download(args: argparse.Namespace, cfg: DataConfig) -> int:
    """Downloads or updates every ticker, then validates and cross-checks it."""
    universe = load_universe(args.universe)
    tickers = args.tickers or universe.tickers
    try:
        creds = alpaca_credentials()
    except (MissingSecretError, ConflictingCredentialError) as e:
        _out(f"ERROR: {e}")
        return 2
    client = make_client(cfg, creds)
    # resolve the feed with one small request, so the cache folder matches the feed actually served
    now = client.clock()
    end = last_complete_session(now, cfg.download.settle_time)
    client.fetch(PROBE_TICKER, end - pd.Timedelta(days=14), end)
    if client.feed != cfg.download.feed:
        _out(f"NOTE: feed '{cfg.download.feed}' is not available on this account; using '{client.feed}'.")
    cache = _cache_for(cfg, client.feed)
    source = make_second_source(cfg) if cfg.cross_check.enabled and not args.no_cross_check else None

    results: list[dict[str, Any]] = []
    _out(f"{'ticker':<7}{'action':<12}{'rows':>6}{'new':>6}  {'first':<11}{'last':<11}{'validation':<11}cross-check")
    for t in tickers:
        row: dict[str, Any] = {"ticker": t}
        try:
            r = update_ticker(
                client,
                cache,
                t,
                start=cfg.download.start,
                overlap_days=cfg.download.overlap_days,
                tolerance=cfg.download.rewrite_tolerance,
                now=now,
            )
            row.update(r.__dict__)
            bars = cache.read(t)
            assert bars is not None
            found = findings(t, bars, cfg.validation)
            errs = [f.message for f in found if f.severity == "error"]
            row["validation"] = "FAIL" if errs else "ok"
            row["findings"] = [f.as_dict() for f in found]
            if source is not None:
                try:
                    row["cross_check"] = cross_check(t, bars, source, cfg.cross_check).as_dict()
                    row["cross_check_status"] = "ok"
                except DataError as e:
                    row["cross_check_status"] = f"FAIL: {e}"
            else:
                row["cross_check_status"] = "skipped"
        except ATUError as e:
            row.update({"action": "error", "validation": "FAIL", "error": str(e), "cross_check_status": "-"})
        results.append(row)
        _out(
            f"{t:<7}{row.get('action', ''):<12}{row.get('rows', 0):>6}{row.get('new_rows', 0):>6}  "
            f"{row.get('first') or '-':<11}{row.get('last') or '-':<11}{row['validation']:<11}"
            f"{str(row['cross_check_status'])[:60]}"
        )
        if "error" in row:
            _out(f"       {row['error'][:200]}")
    failed = [
        r["ticker"] for r in results if r["validation"] != "ok" or str(r["cross_check_status"]).startswith("FAIL")
    ]
    report = _write_report(
        cfg,
        f"download_{new_run_id()}",
        {
            "feed": client.feed,
            "requests": client.requests_made,
            "tickers": results,
            "failed": failed,
        },
    )
    _out(
        f"\nfeed {client.feed}, {client.requests_made} requests, {len(tickers) - len(failed)}/{len(tickers)} ok; "
        f"report: {report}"
    )
    if failed:
        _out(f"PROBLEMS: {', '.join(failed)} (run 'atu2 data validate' for details)")
    return 1 if failed else 0


def cmd_validate(args: argparse.Namespace, cfg: DataConfig) -> int:
    """Validates every cached ticker without any network access."""
    cache = _cache_for(cfg, args.feed)
    tickers = args.tickers or cache.tickers()
    if not tickers:
        _out(f"ERROR: cache {cache.root} is empty; run 'atu2 data download' first")
        return 2
    rows, n_err = _validate_cache(cfg, cache, tickers)
    for r in rows:
        _out(f"{r['severity'].upper():<8}{r['ticker']:<7}{r['check']:<15}{r['message']}")
    path = _write_report(cfg, "validation", {"feed": cache.feed, "findings": rows, "errors": n_err})
    _out(f"\n{len(tickers)} tickers, {n_err} error(s), {len(rows) - n_err} warning(s); report: {path}")
    return 1 if n_err else 0


def cmd_report(args: argparse.Namespace, cfg: DataConfig) -> int:
    """Prints the summary to paste back: counts, date ranges, hashes, findings. Never keys."""
    universe = load_universe(args.universe)
    cache = _cache_for(cfg, args.feed)
    manifest = cache.manifest()
    if not manifest:
        _out(f"ERROR: cache {cache.root} is empty; run 'atu2 data download' first")
        return 2
    rows, n_err = _validate_cache(cfg, cache, sorted(manifest))
    bars = {t: b for t in sorted(manifest) if (b := cache.read(t)) is not None}
    frame = build_price_frame(bars, universe.rule.warm_up_days)
    _out("=== ATU 2 data report (paste this back; it contains no keys) ===")
    _out(
        f"feed: {cache.feed}   tickers cached: {len(manifest)}/{len(universe.tickers)}   "
        f"calendar days: {len(frame.dates)} ({frame.dates[0].date()} -> {frame.dates[-1].date()})"
    )
    _out(f"price frame data_hash: {frame.data_hash[:16]}")
    _out(f"{'ticker':<7}{'rows':>6}  {'first':<11}{'last':<11}{'data_hash':<18}")
    for t in sorted(manifest):
        m = manifest[t]
        _out(f"{t:<7}{m['rows']:>6}  {m['first']:<11}{m['last']:<11}{m['data_hash'][:16]:<18}")
    missing = sorted(set(universe.tickers) - set(manifest))
    _out(f"not cached: {', '.join(missing) if missing else 'none'}")
    _out(f"validation: {n_err} error(s), {len(rows) - n_err} warning(s)")
    for r in rows:
        _out(f"  {r['severity']:<8}{r['ticker']:<7}{r['check']:<15}{r['message'][:150]}")
    return 1 if n_err else 0


def cmd_universe_check(args: argparse.Namespace, cfg: DataConfig) -> int:
    """Applies the written universe rule and proposes the final list."""
    universe = load_universe(args.universe)
    cache = _cache_for(cfg, args.feed)
    cached = {t: b for t in cache.tickers() if (b := cache.read(t)) is not None} or None
    results = apply_rule(universe, cached)
    rule = universe.rule
    _out(
        f"universe status: {universe.status}; rule as of {rule.as_of}: fund history >= {rule.min_history_years:.0f}y, "
        f"median dollar volume >= ${rule.min_median_dollar_volume / 1e6:.0f}M"
        + ("" if cached else "  (no cached data: volume not checked yet)")
    )
    _out(f"{'ticker':<7}{'class':<13}{'fund_y':>7}{'data_y':>8}{'med$vol':>10}  result")
    for r in results:
        dy = f"{r.data_history_years:.1f}" if r.data_history_years is not None else "-"
        mv = f"{r.median_dollar_volume / 1e6:.0f}M" if r.median_dollar_volume is not None else "-"
        _out(
            f"{r.ticker:<7}{r.asset_class:<13}{r.fund_history_years:>7.1f}{dy:>8}{mv:>10}  "
            f"{'eligible' if r.eligible else 'EXCLUDED: ' + '; '.join(r.reasons)}"
        )
    eligible = [r.ticker for r in results if r.eligible]
    gaps = missing_asset_classes(universe, results)
    _out(f"\nproposed universe ({len(eligible)}): {' '.join(eligible)}")
    if gaps:
        _out(f"WARNING: no eligible asset left in required classes: {', '.join(gaps)}")
    if universe.status != "confirmed":
        _out("Status is 'proposed': confirm the list before any backtest (ADR 0005).")
    return 1 if gaps else 0


# ----------------------------------------------------------------- parser
def build_parser() -> argparse.ArgumentParser:
    """Builds the argparse tree for every command."""
    parser = argparse.ArgumentParser(prog="atu2", description="ATU 2 research platform (docs/BUILD_PLAN.md)")
    sub = parser.add_subparsers(dest="command", required=True)
    data = sub.add_parser("data", help="data layer (M1)")
    dsub = data.add_subparsers(dest="action", required=True)
    for name, func, help_text in (
        ("download", cmd_download, "download/update the cache, then validate and cross-check (needs keys)"),
        ("validate", cmd_validate, "validate the cache (no network)"),
        ("report", cmd_report, "summary to paste back (no network, no keys)"),
        ("universe-check", cmd_universe_check, "apply the universe rule (no network)"),
    ):
        p = dsub.add_parser(name, help=help_text)
        p.add_argument("--data-config", default=DEFAULT_DATA_CONFIG, help="path to configs/data.yaml")
        p.add_argument("--universe", default=DEFAULT_UNIVERSE, help="path to configs/universe.yaml")
        p.add_argument("--tickers", nargs="+", help="limit to these tickers")
        p.add_argument("--feed", choices=["sip", "iex"], help="cache feed to read (default: data.yaml)")
        if name == "download":
            p.add_argument("--no-cross-check", action="store_true", help="skip the second-source comparison")
        p.set_defaults(func=func)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point of the `atu2` console script."""
    args = build_parser().parse_args(argv)
    configure_logging(new_run_id())
    try:
        cfg = load_data_config(args.data_config)
        code: int = args.func(args, cfg)
    except (ConfigError, FileNotFoundError) as e:
        _out(f"ERROR: {e}")
        return 2
    return code


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
