"""Point-in-time universe (FR-03) and the PriceFrame builder.

- `apply_rule` applies the written rule in configs/universe.yaml to every candidate, as of
  the rule's fixed date: fund history (inception) and, when cached data is available,
  median daily dollar volume and the history actually present in the data.
- `build_price_frame` aligns cached bars on the single NYSE calendar, forward-fills at most
  one missing day (the stated fill rule), and marks an asset tradable only from its
  (warm_up_days + 1)-th trading day: with 252, an asset is excluded until day 253.
"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from atu2.core.calendars import trading_days
from atu2.core.errors import ContractError
from atu2.core.hashing import hash_bytes, hash_dataframe
from atu2.core.types import PRICE_FIELDS, PriceFrame
from atu2.data.settings import UniverseConfig

DAYS_PER_YEAR = 365.25


@dataclass(frozen=True)
class RuleResult:
    """Rule outcome for one candidate."""

    ticker: str
    asset_class: str
    eligible: bool
    fund_history_years: float
    data_history_years: float | None = None
    median_dollar_volume: float | None = None
    reasons: tuple[str, ...] = field(default_factory=tuple)


def apply_rule(universe: UniverseConfig, bars: dict[str, pd.DataFrame] | None = None) -> list[RuleResult]:
    """Checks every candidate against the rule (data-based checks only for tickers in `bars`)."""
    rule = universe.rule
    as_of = pd.Timestamp(rule.as_of)
    results = []
    for c in universe.candidates:
        reasons: list[str] = []
        fund_years = (as_of - pd.Timestamp(c.inception)).days / DAYS_PER_YEAR
        if fund_years < rule.min_history_years:
            reasons.append(f"fund history {fund_years:.1f}y < {rule.min_history_years:.0f}y")
        data_years = mdv = None
        df = (bars or {}).get(c.ticker)
        if df is not None and len(df):
            upto = df[df.index <= as_of]
            if len(upto):
                data_years = (upto.index[-1] - upto.index[0]).days / DAYS_PER_YEAR
                window = upto.tail(rule.volume_lookback_days)
                mdv = float((window["close"] * window["volume"]).median())
                if mdv < rule.min_median_dollar_volume:
                    reasons.append(
                        f"median dollar volume ${mdv / 1e6:.1f}M < ${rule.min_median_dollar_volume / 1e6:.0f}M"
                    )
            else:
                reasons.append(f"no cached data on or before {as_of.date()}")
        elif bars is not None:
            reasons.append("no cached data")
        results.append(
            RuleResult(
                c.ticker,
                c.asset_class,
                not reasons,
                round(fund_years, 2),
                None if data_years is None else round(data_years, 2),
                mdv,
                tuple(reasons),
            )
        )
    return results


def missing_asset_classes(universe: UniverseConfig, results: list[RuleResult]) -> list[str]:
    """Required asset classes left without any eligible asset."""
    kept = {r.asset_class for r in results if r.eligible}
    return sorted(set(universe.rule.required_asset_classes) - kept)


def build_price_frame(bars: dict[str, pd.DataFrame], warm_up_days: int, adjusted: bool = True) -> PriceFrame:
    """
    Aligns per-ticker bars (date index, BAR_COLUMNS) on the NYSE calendar from the earliest
    to the latest date. Before an asset's first bar and after its last bar every field is
    NaN. A single missing day inside its history is forward-filled (volume 0); longer gaps stay NaN. `tradable` is
    True from the asset's (warm_up_days + 1)-th trading day on, and only on days with data.
    """
    if not bars:
        raise ContractError("build_price_frame: no tickers given")
    tickers = sorted(bars)
    first = min(df.index[0] for df in bars.values() if len(df))
    last = max(df.index[-1] for df in bars.values() if len(df))
    dates = pd.DatetimeIndex(trading_days(first, last), name="date")
    fields: dict[str, dict[str, pd.Series]] = {f: {} for f in PRICE_FIELDS}
    tradable: dict[str, pd.Series] = {}
    for t in tickers:
        df = bars[t].reindex(dates)
        listed = (dates >= bars[t].index[0]) & (dates <= bars[t].index[-1])  # fills only inside its history
        observed = df["adj_close"].notna().to_numpy()
        filled = df.ffill(limit=1)
        filled.loc[~listed] = np.nan
        was_filled = filled["adj_close"].notna().to_numpy() & ~observed
        filled.loc[was_filled, "volume"] = 0.0
        for f in PRICE_FIELDS:
            fields[f][t] = filled[f]
        age = np.cumsum(dates >= bars[t].index[0]) - 1  # 0 on the listing day
        tradable[t] = pd.Series((age >= warm_up_days) & listed & filled["adj_close"].notna().to_numpy(), index=dates)
    frames = {f: pd.DataFrame(fields[f], index=dates, columns=tickers) for f in PRICE_FIELDS}
    trad = pd.DataFrame(tradable, index=dates, columns=tickers).astype(bool)
    combined = "".join(hash_dataframe(frames[f]) for f in PRICE_FIELDS) + hash_dataframe(trad) + str(adjusted)
    return PriceFrame(**frames, tradable=trad, adjusted=adjusted, data_hash=hash_bytes(combined.encode("utf-8")))
