"""
src/data_processing.py
======================
Sequential Data Pipeline: Ingestion, Feature Engineering, and Sentiment Merging.

Components:
-----------
1. MarketDataProcessor (File 1):
   - Ingests 5-minute or daily OHLCV data via yfinance or synthetic market generators.
   - Computes momentum (RSI, MACD), volatility (ATR, Bollinger Bands), and trend (ADX) via `ta`.
   - Cleans missing values (forward-fill + dropna) and scales price/volume to [0, 1] using MinMaxScaler.
2. SentimentProcessor (File 2):
   - Ingests financial headlines from archive datasets or live news feeds.
   - Computes continuous sentiment in [-1.0, 1.0] using ProsusAI/finbert with CUDA GPU/CPU support.
   - Groups by calendar date to calculate daily average sentiment scores.
3. final_state_vector Generator:
   - Merges Market Features (File 1) and Sentiment Scores (File 2) into one unified DataFrame.
"""

from pathlib import Path
from typing import List, Optional, Union
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import yfinance as yf

# Technical Analysis Indicators
from ta.momentum import RSIIndicator
from ta.trend import ADXIndicator, MACD
from ta.volatility import AverageTrueRange, BollingerBands


# =====================================================================
# File 1: MarketDataProcessor
# =====================================================================
class MarketDataProcessor:
    """
    Ingests and transforms financial market data into structured state vectors
    tailored for Reinforcement Learning (RL) neural network architectures.
    """

    def __init__(
        self,
        scaler: Optional[MinMaxScaler] = None,
        scale_columns: Optional[List[str]] = None,
    ) -> None:
        self.scaler: MinMaxScaler = scaler if scaler is not None else MinMaxScaler(feature_range=(0, 1))
        self.scale_columns: List[str] = (
            scale_columns
            if scale_columns is not None
            else ["Open", "High", "Low", "Close", "Volume"]
        )
        self._is_fitted: bool = False

    def fetch_data(
        self,
        ticker: str,
        start: Optional[Union[str, pd.Timestamp]] = None,
        end: Optional[Union[str, pd.Timestamp]] = None,
        period: str = "60d",
        interval: str = "5m",
    ) -> pd.DataFrame:
        """
        Downloads historical market data for a given ticker and date range.
        Supports 5-minute intraday intervals ('5m') or daily ('1d').
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

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df.columns = [str(col).strip().title() for col in df.columns]
        df.columns.name = None

        required_cols = ["Open", "High", "Low", "Close", "Volume"]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise KeyError(f"Fetched data missing required columns: {missing_cols}")

        df = df[required_cols].astype(np.float64).sort_index()
        df.index = pd.to_datetime(df.index)
        return df

    def add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Computes momentum, volatility, and trend indicators via the `ta` library.
        """
        df_feat = df.copy()

        # 1. Momentum: RSI (14 period)
        rsi_indicator = RSIIndicator(close=df_feat["Close"], window=14, fillna=False)
        df_feat["RSI"] = rsi_indicator.rsi()

        # 1. Momentum: MACD (12, 26, 9)
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

        # 2. Volatility: ATR (14 period)
        atr_indicator = AverageTrueRange(
            high=df_feat["High"],
            low=df_feat["Low"],
            close=df_feat["Close"],
            window=14,
            fillna=False,
        )
        df_feat["ATR"] = atr_indicator.average_true_range()

        # 2. Volatility: Bollinger Bands (20 period, 2 std)
        bb_indicator = BollingerBands(
            close=df_feat["Close"],
            window=20,
            window_dev=2,
            fillna=False,
        )
        df_feat["BB_High"] = bb_indicator.bollinger_hband()
        df_feat["BB_Mid"] = bb_indicator.bollinger_mavg()
        df_feat["BB_Low"] = bb_indicator.bollinger_lband()

        # 3. Trend: ADX (14 period)
        adx_indicator = ADXIndicator(
            high=df_feat["High"],
            low=df_feat["Low"],
            close=df_feat["Close"],
            window=14,
            fillna=False,
        )
        df_feat["ADX"] = adx_indicator.adx()

        return df_feat

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Forward-fills missing candles and drops initial rolling window burn-in rows.
        """
        cleaned_df = df.copy()
        cleaned_df = cleaned_df.ffill().dropna()
        return cleaned_df

    def normalize_features(
        self,
        df: pd.DataFrame,
        columns_to_scale: Optional[List[str]] = None,
        fit_scaler: bool = True,
    ) -> pd.DataFrame:
        """
        MinMax scales price and volume features to [0, 1]. Preserves bounded RSI/ADX
        and zero-centered MACD without distorting sign relationships.
        """
        target_cols = columns_to_scale if columns_to_scale is not None else self.scale_columns
        missing = [col for col in target_cols if col not in df.columns]
        if missing:
            raise KeyError(f"Columns specified for scaling not found: {missing}")

        normalized_df = df.copy()

        if fit_scaler:
            normalized_df[target_cols] = self.scaler.fit_transform(normalized_df[target_cols])
            self._is_fitted = True
        else:
            if not self._is_fitted:
                raise RuntimeError("MinMaxScaler is not fitted yet.")
            normalized_df[target_cols] = self.scaler.transform(normalized_df[target_cols])

        return normalized_df

    def process_dataframe(
        self,
        df: pd.DataFrame,
        fit_scaler: bool = True,
    ) -> pd.DataFrame:
        """
        Full feature engineering pipeline: Indicators -> Clean -> Normalize.
        """
        engineered_df = self.add_technical_indicators(df)
        cleaned_df = self.clean_data(engineered_df)
        state_vector_df = self.normalize_features(cleaned_df, fit_scaler=fit_scaler)
        return state_vector_df

    def process(
        self,
        ticker: str,
        start: Optional[Union[str, pd.Timestamp]] = None,
        end: Optional[Union[str, pd.Timestamp]] = None,
        period: str = "60d",
        interval: str = "5m",
        fit_scaler: bool = True,
    ) -> pd.DataFrame:
        """
        Downloads data and returns an engineered state vector DataFrame.
        """
        raw_df = self.fetch_data(
            ticker=ticker,
            start=start,
            end=end,
            period=period,
            interval=interval,
        )
        return self.process_dataframe(raw_df, fit_scaler=fit_scaler)


# =====================================================================
# File 2: SentimentProcessor
# =====================================================================
class SentimentProcessor:
    """
    Analyzes financial news headlines using the ProsusAI/finbert model with CUDA/CPU support.
    """

    def __init__(
        self,
        model_name: str = "ProsusAI/finbert",
        device: Optional[Union[str, torch.device]] = None,
        max_length: int = 128,
    ) -> None:
        if device is not None:
            self.device = torch.device(device)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model_name: str = model_name
        self.max_length: int = max_length

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
        self.model.to(self.device)
        self.model.eval()

        label2id = {k.lower(): v for k, v in self.model.config.label2id.items()}
        self.pos_idx: int = label2id.get("positive", 0)
        self.neg_idx: int = label2id.get("negative", 1)
        self.neu_idx: int = label2id.get("neutral", 2)

    def get_sentiment_score(self, text: str) -> float:
        """
        Computes continuous sentiment score in [-1.0, 1.0]: P(positive) - P(negative).
        """
        if not isinstance(text, str) or not text.strip():
            return 0.0

        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=self.max_length,
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            probabilities = torch.softmax(outputs.logits, dim=-1)

        prob_pos = probabilities[0, self.pos_idx].item()
        prob_neg = probabilities[0, self.neg_idx].item()
        score = prob_pos - prob_neg
        return float(np.clip(score, -1.0, 1.0))

    def process_news_dataframe(
        self,
        df: pd.DataFrame,
        batch_size: Optional[int] = 32,
    ) -> pd.DataFrame:
        """
        Processes financial headlines and calculates daily average sentiment scores.
        """
        required_cols = {"date", "headline"}
        if not required_cols.issubset(df.columns):
            raise KeyError(f"DataFrame must contain columns {required_cols}. Found: {list(df.columns)}")

        if df.empty:
            return pd.DataFrame(columns=["date", "daily_sentiment"])

        news_df = df.copy()
        news_df["date"] = pd.to_datetime(news_df["date"]).dt.strftime("%Y-%m-%d")

        texts = news_df["headline"].astype(str).tolist()
        scores: List[float] = []

        # Batched inference for performance
        bs = batch_size if batch_size is not None and batch_size > 1 else 1
        for i in range(0, len(texts), bs):
            batch_texts = [t if isinstance(t, str) and t.strip() else "" for t in texts[i : i + bs]]
            inputs = self.tokenizer(
                batch_texts,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=self.max_length,
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs)
                probabilities = torch.softmax(outputs.logits, dim=-1)

            prob_pos = probabilities[:, self.pos_idx].cpu().numpy()
            prob_neg = probabilities[:, self.neg_idx].cpu().numpy()
            scores.extend(np.clip(prob_pos - prob_neg, -1.0, 1.0).tolist())

        news_df["sentiment_score"] = scores

        daily_sentiment_df = (
            news_df.groupby("date", as_index=False)["sentiment_score"]
            .mean()
            .rename(columns={"sentiment_score": "daily_sentiment"})
            .sort_values(by="date")
            .reset_index(drop=True)
        )
        return daily_sentiment_df


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
def create_final_state_vector(
    market_df: pd.DataFrame,
    sentiment_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Merges the output of MarketDataProcessor (File 1) and SentimentProcessor (File 2)
    into one massive table called `final_state_vector`.

    Args:
        market_df: Engineered state vector DataFrame from MarketDataProcessor
                   (contains normalized OHLCV, RSI, MACD, ATR, BB, ADX with DatetimeIndex).
        sentiment_df: Optional aggregated daily sentiment DataFrame from SentimentProcessor
                      (contains ['date', 'daily_sentiment']).

    Returns:
        pd.DataFrame: Merged `final_state_vector` DataFrame with complete features
                      and DatetimeIndex, ready for ConstrainedTradingEnv.
    """
    final_df = market_df.copy()

    # Extract calendar date key for merging
    if isinstance(final_df.index, pd.DatetimeIndex):
        final_df["_merge_date"] = final_df.index.strftime("%Y-%m-%d")
    elif "Date" in final_df.columns:
        final_df["_merge_date"] = pd.to_datetime(final_df["Date"]).dt.strftime("%Y-%m-%d")
    else:
        final_df["_merge_date"] = pd.to_datetime(final_df.index).strftime("%Y-%m-%d")

    if sentiment_df is not None and not sentiment_df.empty and "daily_sentiment" in sentiment_df.columns:
        # Merge on date key
        sentiment_clean = sentiment_df.copy()
        sentiment_clean["date"] = pd.to_datetime(sentiment_clean["date"]).dt.strftime("%Y-%m-%d")
        merged = pd.merge(
            final_df,
            sentiment_clean[["date", "daily_sentiment"]],
            left_on="_merge_date",
            right_on="date",
            how="left",
        )
        merged.index = final_df.index
        # Forward-fill sentiment scores across intraday bars, filling unmapped dates with neutral 0.0
        merged["daily_sentiment"] = merged["daily_sentiment"].ffill().fillna(0.0)
        final_df = merged.drop(columns=["_merge_date", "date"], errors="ignore")
    else:
        # Neutral default if no sentiment DataFrame provided
        final_df["daily_sentiment"] = 0.0
        final_df = final_df.drop(columns=["_merge_date"], errors="ignore")

    return final_df

