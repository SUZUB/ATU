"""
MarketDataProcessor Module
===========================
Data ingestion, technical-indicator engineering, leak-free feature scaling and
sentiment alignment for the ATU intraday RL trading system.

Look-ahead protections:
-----------------------
1. The MinMaxScaler is fit ONLY on the chronological training slice (first
   `train_ratio` of rows, default 80%). The remaining test rows are transformed
   with that already-fitted scaler, so test-set extremes never influence the
   training features. Test values may therefore fall slightly outside [0, 1];
   that is expected and correct.
2. Daily sentiment is lagged: a headline dated D only becomes visible to bars on
   the first trading day on or after D + `sentiment_lag_days` (default 1), so
   afternoon news cannot leak into same-day morning decisions.

Raw prices:
-----------
Unscaled close/high/low prices are preserved in `raw_close`, `raw_high` and
`raw_low`. The trading environment uses them for P&L, balance and intra-candle
stop-loss fills, and excludes them from the agent's observation.
"""

from typing import List, Optional, Union
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import yfinance as yf

# Technical Analysis library imports
from ta.momentum import RSIIndicator
from ta.trend import ADXIndicator, MACD
from ta.volatility import AverageTrueRange, BollingerBands


# Unscaled price columns consumed by ConstrainedTradingEnv (P&L and stop-loss fills).
RAW_PRICE_COLUMN: str = "raw_close"
RAW_HIGH_COLUMN: str = "raw_high"
RAW_LOW_COLUMN: str = "raw_low"
RAW_PRICE_SOURCES = {RAW_PRICE_COLUMN: "Close", RAW_HIGH_COLUMN: "High", RAW_LOW_COLUMN: "Low"}

# Columns MinMax-scaled to [0, 1] using statistics from the training slice only.
DEFAULT_SCALE_COLUMNS: List[str] = [
    "Open", "High", "Low", "Close", "Volume",
    "MACD", "MACD_Signal", "MACD_Diff",
    "ATR",
    "BB_High", "BB_Mid", "BB_Low",
]

# Oscillators bounded to [0, 100] by definition: divided by 100 (a fixed transform,
# no fitted statistics, so no leakage) to share the same 0-1 range.
BOUNDED_0_100_COLUMNS: List[str] = ["RSI", "ADX"]

DEFAULT_TRAIN_RATIO: float = 0.8


def compute_split_index(n_rows: int, train_ratio: float = DEFAULT_TRAIN_RATIO) -> int:
    """
    Chronological train/test boundary. Training pipelines MUST split with this
    same index so that the scaler's fit window matches the training set exactly.
    """
    if not 0.0 < train_ratio <= 1.0:
        raise ValueError(f"train_ratio must be in (0, 1], got {train_ratio}")
    return int(n_rows * train_ratio)


class MarketDataProcessor:
    """
    Ingests and transforms financial market data into structured state vectors
    for Reinforcement Learning agents.

    Pipeline: fetch OHLCV -> technical indicators -> clean -> leak-free scaling.
    """

    def __init__(
        self,
        scaler: Optional[MinMaxScaler] = None,
        scale_columns: Optional[List[str]] = None,
        train_ratio: float = DEFAULT_TRAIN_RATIO,
    ) -> None:
        """
        Args:
            scaler: Optional pre-configured MinMaxScaler. Defaults to MinMaxScaler((0, 1)).
            scale_columns: Columns to MinMax-scale. Defaults to DEFAULT_SCALE_COLUMNS
                           (OHLCV, MACD family, ATR, Bollinger Bands).
            train_ratio: Fraction of rows (chronological head) the scaler is fit on.
        """
        self.scaler: MinMaxScaler = scaler if scaler is not None else MinMaxScaler(feature_range=(0, 1))
        self.scale_columns: List[str] = (
            list(scale_columns) if scale_columns is not None else list(DEFAULT_SCALE_COLUMNS)
        )
        compute_split_index(1, train_ratio)  # validates the ratio
        self.train_ratio: float = float(train_ratio)
        self._is_fitted: bool = False
        self.fit_end_index: Optional[int] = None

    def fetch_data(
        self,
        ticker: str,
        start: Optional[Union[str, pd.Timestamp]] = None,
        end: Optional[Union[str, pd.Timestamp]] = None,
        period: str = "2y",
        interval: str = "1d",
    ) -> pd.DataFrame:
        """
        Downloads historical OHLCV market data for a given ticker and date range.

        Returns:
            pd.DataFrame: Raw OHLCV DataFrame with columns ['Open', 'High', 'Low', 'Close', 'Volume'].
        """
        download_kwargs = {
            "tickers": ticker,
            "interval": interval,
            "auto_adjust": False,
            "progress": False,
        }

        if start is not None:
            download_kwargs["start"] = start
            if end is not None:
                download_kwargs["end"] = end
        else:
            download_kwargs["period"] = period

        df: pd.DataFrame = yf.download(**download_kwargs)

        if df.empty:
            raise ValueError(
                f"No data returned for ticker '{ticker}' with parameters: {download_kwargs}"
            )

        # Flatten MultiIndex columns introduced in recent yfinance versions
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df.columns = [str(col).strip().title() for col in df.columns]
        df.columns.name = None

        required_cols = ["Open", "High", "Low", "Close", "Volume"]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise KeyError(
                f"Fetched data missing required OHLCV columns: {missing_cols}. "
                f"Available columns: {list(df.columns)}"
            )

        df = df[required_cols].astype(np.float64).sort_index()
        df.index = pd.to_datetime(df.index)
        df.index.name = "Date"
        return df

    def add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Computes causal (backward-looking) indicators with the `ta` library:
        RSI(14), MACD(12, 26, 9) + signal + histogram, ATR(14), Bollinger Bands(20, 2), ADX(14).
        """
        df_feat = df.copy()

        df_feat["RSI"] = RSIIndicator(close=df_feat["Close"], window=14, fillna=False).rsi()

        macd_indicator = MACD(
            close=df_feat["Close"],
            window_slow=26,
            window_fast=12,
            window_sign=9,
            fillna=False,
        )
        df_feat["MACD"] = macd_indicator.macd()
        df_feat["MACD_Signal"] = macd_indicator.macd_signal()
        df_feat["MACD_Diff"] = macd_indicator.macd_diff()

        df_feat["ATR"] = AverageTrueRange(
            high=df_feat["High"],
            low=df_feat["Low"],
            close=df_feat["Close"],
            window=14,
            fillna=False,
        ).average_true_range()

        bb_indicator = BollingerBands(close=df_feat["Close"], window=20, window_dev=2, fillna=False)
        df_feat["BB_High"] = bb_indicator.bollinger_hband()
        df_feat["BB_Mid"] = bb_indicator.bollinger_mavg()
        df_feat["BB_Low"] = bb_indicator.bollinger_lband()

        df_feat["ADX"] = ADXIndicator(
            high=df_feat["High"],
            low=df_feat["Low"],
            close=df_feat["Close"],
            window=14,
            fillna=False,
        ).adx()

        return df_feat

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Forward-fills sporadic gaps (causal) and drops the indicator burn-in rows.
        """
        return df.copy().ffill().dropna()

    def normalize_features(
        self,
        df: pd.DataFrame,
        columns_to_scale: Optional[List[str]] = None,
        fit_scaler: bool = True,
        train_ratio: Optional[float] = None,
    ) -> pd.DataFrame:
        """
        Leak-free normalization.

        - Preserves unscaled close/high/low in `raw_close`/`raw_high`/`raw_low`
          (for environment P&L and stop-loss fills).
        - fit_scaler=True: fits the MinMaxScaler on rows [0, split_index) ONLY,
          then transforms ALL rows (train + test) with that fitted scaler.
        - fit_scaler=False: transforms all rows with the previously fitted scaler
          (e.g. scoring a brand-new out-of-sample period).
        - RSI and ADX are divided by 100.

        Args:
            df: Cleaned feature DataFrame (chronologically sorted).
            columns_to_scale: Columns to MinMax-scale. Defaults to `self.scale_columns`.
            fit_scaler: Whether to fit the scaler on the training slice.
            train_ratio: Override for `self.train_ratio`.
        """
        target_cols = columns_to_scale if columns_to_scale is not None else self.scale_columns
        missing = [col for col in target_cols if col not in df.columns]
        if missing:
            raise KeyError(f"Columns specified for scaling not found in DataFrame: {missing}")

        normalized_df = df.copy()

        # Preserve unscaled prices before any transformation
        if RAW_PRICE_COLUMN not in normalized_df.columns and "Close" not in normalized_df.columns:
            raise KeyError("A 'Close' column is required to preserve raw prices.")
        for raw_col, source_col in RAW_PRICE_SOURCES.items():
            if raw_col not in normalized_df.columns and source_col in normalized_df.columns:
                normalized_df[raw_col] = df[source_col].astype(np.float64)

        if fit_scaler:
            ratio = self.train_ratio if train_ratio is None else float(train_ratio)
            fit_end = compute_split_index(len(normalized_df), ratio)
            if fit_end < 2:
                raise ValueError(
                    f"Training slice too small to fit the scaler ({fit_end} rows from "
                    f"{len(normalized_df)} total at train_ratio={ratio})."
                )
            self.scaler.fit(normalized_df[target_cols].iloc[:fit_end])
            self._is_fitted = True
            self.fit_end_index = fit_end
        elif not self._is_fitted:
            raise RuntimeError(
                "MinMaxScaler is not fitted yet. Call normalize_features with fit_scaler=True "
                "on data whose head is the training period first."
            )

        normalized_df[target_cols] = self.scaler.transform(normalized_df[target_cols])

        for col in BOUNDED_0_100_COLUMNS:
            if col in normalized_df.columns:
                normalized_df[col] = df[col] / 100.0

        return normalized_df

    def process(
        self,
        ticker: str,
        start: Optional[Union[str, pd.Timestamp]] = None,
        end: Optional[Union[str, pd.Timestamp]] = None,
        period: str = "2y",
        interval: str = "1d",
        fit_scaler: bool = True,
        train_ratio: Optional[float] = None,
    ) -> pd.DataFrame:
        """
        Download -> indicators -> clean -> leak-free normalization.
        """
        raw_df = self.fetch_data(ticker=ticker, start=start, end=end, period=period, interval=interval)
        return self.process_dataframe(raw_df, fit_scaler=fit_scaler, train_ratio=train_ratio)

    def process_dataframe(
        self,
        df: pd.DataFrame,
        fit_scaler: bool = True,
        train_ratio: Optional[float] = None,
    ) -> pd.DataFrame:
        """
        Processes an existing OHLCV DataFrame without re-downloading.
        """
        engineered_df = self.add_technical_indicators(df.sort_index())
        cleaned_df = self.clean_data(engineered_df)
        return self.normalize_features(cleaned_df, fit_scaler=fit_scaler, train_ratio=train_ratio)


# =====================================================================
# Sentiment alignment (lagged to prevent same-day look-ahead)
# =====================================================================
def _to_naive_calendar_dates(values) -> pd.Series:
    """Parses dates/timestamps to timezone-naive midnight datetimes (local wall-clock date)."""
    parsed = pd.to_datetime(pd.Series(values))
    if parsed.dt.tz is not None:
        parsed = parsed.dt.tz_localize(None)
    return parsed.dt.normalize().astype("datetime64[ns]")


def create_final_state_vector(
    market_df: pd.DataFrame,
    sentiment_df: Optional[pd.DataFrame] = None,
    sentiment_lag_days: int = 1,
    max_staleness_days: Optional[int] = 4,
) -> pd.DataFrame:
    """
    Merges market features with daily FinBERT sentiment into `final_state_vector`.

    Sentiment dated D becomes available on calendar day D + `sentiment_lag_days`.
    Each bar receives the most recent sentiment that was available on or before its
    own date (so Friday's news reaches Monday's bars). Sentiment older than
    `max_staleness_days` after it became available is dropped, so bars with no
    recent sentiment receive a neutral 0.0 instead of a months-old score. Row count and order of `market_df` are preserved, so the
    chronological train/test split index is unchanged.

    Args:
        market_df: Output of MarketDataProcessor with a DatetimeIndex.
        sentiment_df: DataFrame with ['date', 'daily_sentiment'].
        sentiment_lag_days: Calendar-day delay before a day's sentiment is usable (default 1).
        max_staleness_days: Max calendar days a score is carried forward after it becomes
                            available (default 4, covers long weekends). None = unlimited.
    """
    if sentiment_lag_days < 0:
        raise ValueError("sentiment_lag_days must be >= 0")

    final_df = market_df.copy()

    if isinstance(final_df.index, pd.DatetimeIndex):
        bar_timestamps = final_df.index
    elif "Date" in final_df.columns:
        bar_timestamps = pd.DatetimeIndex(pd.to_datetime(final_df["Date"]))
    else:
        bar_timestamps = pd.DatetimeIndex(pd.to_datetime(final_df.index))

    if sentiment_df is None or sentiment_df.empty or "daily_sentiment" not in sentiment_df.columns:
        final_df["daily_sentiment"] = 0.0
        return final_df

    bars = pd.DataFrame(
        {
            "_bar_date": _to_naive_calendar_dates(bar_timestamps).to_numpy(),
            "_pos": np.arange(len(final_df)),
        }
    ).sort_values("_bar_date", kind="stable")

    sentiment = pd.DataFrame(
        {
            "_available_date": _to_naive_calendar_dates(sentiment_df["date"]).to_numpy()
            + np.timedelta64(int(sentiment_lag_days), "D"),
            "daily_sentiment": pd.to_numeric(sentiment_df["daily_sentiment"], errors="coerce").to_numpy(),
        }
    ).dropna()
    sentiment = (
        sentiment.groupby("_available_date", as_index=False)["daily_sentiment"]
        .mean()
        .sort_values("_available_date")
    )

    aligned = pd.merge_asof(
        bars,
        sentiment,
        left_on="_bar_date",
        right_on="_available_date",
        direction="backward",
        tolerance=None if max_staleness_days is None else pd.Timedelta(days=int(max_staleness_days)),
    ).sort_values("_pos")

    final_df["daily_sentiment"] = aligned["daily_sentiment"].fillna(0.0).to_numpy(dtype=np.float64)
    return final_df


if __name__ == "__main__":
    processor = MarketDataProcessor()

    print("Fetching and processing 2 years of daily data for 'AAPL'...")
    state_vector_df = processor.process(ticker="AAPL", period="2y", interval="1d")
    split_idx = compute_split_index(len(state_vector_df), processor.train_ratio)

    print(f"\nFinal State Vector Shape: {state_vector_df.shape}")
    print(f"Scaler fit on rows [0, {processor.fit_end_index}) | split index: {split_idx}")
    print(f"Feature Columns ({len(state_vector_df.columns)}): {list(state_vector_df.columns)}\n")
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 1000)
    print(state_vector_df.head(5))
