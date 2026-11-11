"""
src/data_processing.py
======================
Sequential Data Pipeline: Ingestion, Feature Engineering, and Sentiment Merging.

Components:
-----------
1. MarketDataProcessor (File 1):
   - Ingests 5-minute or daily OHLCV data via yfinance or synthetic market generators.
   - Computes momentum (RSI, MACD), volatility (ATR, Bollinger Bands), and trend (ADX) via `ta`.
   - Cleans missing values and MinMax-scales OHLCV/MACD/ATR/BB with a scaler fit on the
     first 80% (train slice) only; preserves unscaled prices in `raw_close`.
2. SentimentProcessor (File 2):
   - Ingests financial headlines from archive datasets or live news feeds.
   - Computes continuous sentiment in [-1.0, 1.0] using ProsusAI/finbert with CUDA GPU/CPU support.
   - Groups by calendar date to calculate daily average sentiment scores.
3. final_state_vector Generator:
   - Merges Market Features (File 1) and Sentiment Scores (File 2), lagging sentiment by one day.
"""

from pathlib import Path
from typing import List, Optional, Union
import numpy as np
import pandas as pd
try:
    from sentiment_processor import SentimentProcessor
    from market_data_processor import (
        MarketDataProcessor,
        RAW_PRICE_COLUMN,
        compute_split_index,
        create_final_state_vector,
    )
except ImportError:
    import sys
    sys.path.append(str(Path(__file__).resolve().parent.parent))
    from sentiment_processor import SentimentProcessor
    from market_data_processor import (
        MarketDataProcessor,
        RAW_PRICE_COLUMN,
        compute_split_index,
        create_final_state_vector,
    )


# =====================================================================
# File 1: MarketDataProcessor (single source of truth: root market_data_processor.py)
# =====================================================================
# Re-exported so every pipeline uses the same leak-free scaling (scaler fit on the
# training slice only) and the preserved `raw_close` column for environment P&L.


# =====================================================================
# File 2: SentimentProcessor (single source of truth: root sentiment_processor.py)
# =====================================================================


# =====================================================================
# Archive Data Loader & Intraday Market Generator
# =====================================================================
def load_archive_headlines(
    archive_dir: Optional[Union[str, Path]] = None,
    ticker: Optional[str] = None,
    max_rows: int = 1000,
    allow_synthetic_fallback: bool = False,
) -> pd.DataFrame:
    """
    Loads real headlines from local archive directory or Google Drive dataset storage.
    Searches candidate directories:
      1. Provided archive_dir
      2. /content/drive/MyDrive/ATU_Data (Google Colab persistent Drive storage)
      3. archive/ (Local workspace)
    Checks:
      - analyst_ratings_processed.csv
      - raw_analyst_ratings.csv
      - raw_partner_headlines.csv

    Args:
        archive_dir: Directory path containing raw CSV files.
        ticker: Target stock ticker symbol (e.g. 'AAPL').
        max_rows: Maximum number of headlines to retrieve.
        allow_synthetic_fallback: If False (default), raises an error when real data is missing.
                                  If True, falls back to simulated headlines with an explicit warning.

    Returns:
        pd.DataFrame: Formatted with columns ['date', 'headline'].
    """
    candidate_dirs = []
    if archive_dir is not None:
        candidate_dirs.append(Path(archive_dir))
    candidate_dirs.extend([
        Path("/content/drive/MyDrive/ATU_Workspace/datasets"),
        Path("/content/drive/MyDrive/ATU_Workspace"),
        Path("/content/drive/MyDrive/ATU_Data"),
        Path("archive"),
        Path("../archive"),
    ])

    file_names = [
        "analyst_ratings_processed.csv",
        "raw_analyst_ratings.csv",
        "raw_partner_headlines.csv",
    ]

    target_file = None
    for c_dir in candidate_dirs:
        for f_name in file_names:
            candidate = c_dir / f_name
            if candidate.exists() and candidate.stat().st_size > 0:
                target_file = candidate
                break
        if target_file is not None:
            break

    if target_file is not None:
        try:
            # Memory-efficient chunked reader (does not load full 150MB+ file at once)
            matched_frames = []
            total_matched = 0
            chunksize = 25000

            for chunk in pd.read_csv(target_file, chunksize=chunksize):
                if "title" in chunk.columns and "headline" not in chunk.columns:
                    chunk = chunk.rename(columns={"title": "headline"})

                stock_col = next((c for c in chunk.columns if c.lower() in ["stock", "ticker"]), None)
                if ticker is not None and stock_col is not None:
                    filtered = chunk[chunk[stock_col].astype(str).str.upper() == ticker.upper()].copy()
                else:
                    filtered = chunk.copy()

                filtered = filtered.dropna(subset=["headline", "date"])
                if not filtered.empty:
                    filtered["date"] = pd.to_datetime(filtered["date"], utc=True, errors="coerce").dt.strftime("%Y-%m-%d")
                    filtered = filtered.dropna(subset=["date"])
                    matched_frames.append(filtered[["date", "headline"]])
                    total_matched += len(filtered)
                    if total_matched >= max_rows:
                        break

            if matched_frames:
                df = pd.concat(matched_frames, ignore_index=True).head(max_rows)
                print(f"  - Successfully loaded {len(df):,} real historical headlines from: {target_file}")
                return df
            else:
                print(f"  - Notice: No headlines matching ticker '{ticker}' found in {target_file}")
        except Exception as e:
            print(f"Warning: Failed to parse archive headlines from {target_file}: {e}")

    # If real data is missing or empty, handle fallback explicitly
    if not allow_synthetic_fallback:
        searched_paths = [str(d) for d in candidate_dirs]
        raise FileNotFoundError(
            f"CRITICAL: Real financial headlines for ticker '{ticker}' were not found in {searched_paths}.\n"
            f"Silent synthetic fallback is disabled to prevent data leakage and guarantee real model integrity.\n"
            f"Please ensure 'analyst_ratings_processed.csv' is uploaded to your dataset directory (e.g. /content/drive/MyDrive/ATU_Data/),\n"
            f"or explicitly pass allow_synthetic_fallback=True (or --allow-synthetic-fallback) to opt into synthetic testing."
        )

    print("WARNING: Real headlines dataset not found. Using simulated fallback headlines as explicitly requested.")
    return pd.DataFrame(
        {
            "date": ["2024-01-02", "2024-01-03", "2024-01-04"],
            "headline": [
                "Tech shares surge following positive earnings release.",
                "Central bank signals rate stabilization, boosting market sentiment.",
                "Market consolidates as quarterly revenue growth meets expectations.",
            ],
        }
    )


def generate_intraday_5min_market_data(
    n_rows: int = 1500,
    start_date: str = "2024-01-02",
    initial_price: float = 150.0,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generates realistic 5-minute intraday market sessions (09:15 to 15:25, 75 bars/day, Mon-Fri).
    """
    rng = np.random.default_rng(seed)
    timestamps = []
    current_date = pd.Timestamp(start_date)

    while len(timestamps) < n_rows:
        if current_date.weekday() < 5:
            day_candles = pd.date_range(
                start=current_date.replace(hour=9, minute=15, second=0),
                end=current_date.replace(hour=15, minute=25, second=0),
                freq="5min",
            )
            for ts in day_candles:
                timestamps.append(ts)
                if len(timestamps) == n_rows:
                    break
        current_date += pd.Timedelta(days=1)

    datetime_index = pd.DatetimeIndex(timestamps)
    returns = rng.normal(loc=0.0001, scale=0.0035, size=n_rows)
    close_prices = initial_price * np.cumprod(1.0 + returns)

    raw_df = pd.DataFrame(
        {
            "Open": close_prices * (1.0 + rng.normal(0.0, 0.0008, size=n_rows)),
            "High": close_prices * (1.0 + np.abs(rng.normal(0.0008, 0.0015, size=n_rows))),
            "Low": close_prices * (1.0 - np.abs(rng.normal(0.0008, 0.0015, size=n_rows))),
            "Close": close_prices,
            "Volume": rng.uniform(10000, 500000, size=n_rows),
        },
        index=datetime_index,
    )
    return raw_df


# =====================================================================
# Pipeline Merging Function: final_state_vector
# =====================================================================
# Re-exported from market_data_processor: sentiment is lagged by one day so that
# a day's headlines only affect the following trading session's bars.
