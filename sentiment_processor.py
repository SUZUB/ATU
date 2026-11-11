"""
SentimentProcessor Module
=========================
A production-grade Python module designed for financial sentiment analysis
within reinforcement learning (RL) trading systems.
Utilizes PyTorch and Hugging Face's Transformers with the 'ProsusAI/finbert' model
to extract continuous sentiment metrics from financial news headlines and aggregate
them into daily state vectors for RL agent consumption.
"""

from typing import List, Optional, Union
import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


class SentimentProcessor:
    """
    Analyzes financial news sentiment using the specialized ProsusAI/finbert model.

    Features:
    ---------
    1. Hardware Acceleration: Automatically detects and utilizes NVIDIA CUDA GPU
       if available, seamlessly falling back to CPU execution.
    2. Continuous Scoring: Computes directional sentiment in [-1.0, 1.0] by taking
       P(positive) - P(negative), treating neutral sentiment as an anchor toward 0.0.
    3. Multi-Headline Aggregation: Processes news DataFrames with multiple daily
       headlines and aggregates them by date into a daily mean sentiment score.
    4. RL Alignment: Outputs standardized DataFrames ready for direct merging
       with OHLCV state vector time series.
    """

    def __init__(
        self,
        model_name: str = "ProsusAI/finbert",
        device: Optional[Union[str, torch.device]] = None,
        max_length: int = 128,
    ) -> None:
        """
        Initializes the SentimentProcessor, loading the tokenizer and model onto
        the appropriate compute device (CUDA GPU if available, else CPU).

        Args:
            model_name: Pretrained Hugging Face model identifier (default: 'ProsusAI/finbert').
            device: Optional explicit torch device ('cuda', 'cpu', etc.). If None, automatically
                    detects CUDA availability.
            max_length: Maximum token sequence length for truncation/padding (default: 128).
        """
        # Requirement 1: Detect and utilize GPU (CUDA) if available, falling back to CPU
        if device is not None:
            self.device = torch.device(device)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model_name: str = model_name
        self.max_length: int = max_length

        # Load tokenizer and sequence classification model
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)

        # Move model to target device and set to evaluation mode (disables dropout layers)
        self.model.to(self.device)
        self.model.eval()

        # Dynamically map label positions from model configuration for robustness
        # In ProsusAI/finbert: 0 -> positive, 1 -> negative, 2 -> neutral
        label2id = {k.lower(): v for k, v in self.model.config.label2id.items()}
        self.pos_idx: int = label2id.get("positive", 0)
        self.neg_idx: int = label2id.get("negative", 1)
        self.neu_idx: int = label2id.get("neutral", 2)

    def get_sentiment_score(self, text: str) -> float:
        """
        Computes a continuous sentiment score for a single headline or text snippet.

        Calculation:
            score = P(positive) - P(negative)
        where probabilities are derived via softmax over model output logits.
        The resulting continuous score is naturally bounded between -1.0 (highly negative)
        and +1.0 (highly positive), while neutral probability naturally pulls the score toward 0.0.

        Args:
            text: A financial news headline or textual narrative.

        Returns:
            float: Continuous sentiment score in the range [-1.0, 1.0].
        """
        # Handle empty, whitespace-only, or invalid inputs safely
        if not isinstance(text, str) or not text.strip():
            return 0.0

        # Tokenize and encode input text
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=self.max_length,
        ).to(self.device)

        # Perform forward inference without gradient computation for optimal performance
        with torch.no_grad():
            outputs = self.model(**inputs)
            probabilities = torch.softmax(outputs.logits, dim=-1)

        # Extract positive and negative probabilities
        prob_pos = probabilities[0, self.pos_idx].item()
        prob_neg = probabilities[0, self.neg_idx].item()

        # Continuous score: P(positive) - P(negative)
        score = prob_pos - prob_neg

        # Numerical stability clamp to strictly ensure [-1.0, 1.0]
        return float(np.clip(score, -1.0, 1.0))

    def process_news_dataframe(
        self,
        df: pd.DataFrame,
        batch_size: Optional[int] = 32,
    ) -> pd.DataFrame:
        """
        Processes a DataFrame of financial headlines, scores each headline's sentiment,
        and aggregates by date to produce a daily average sentiment score.

        Pipeline Steps:
        1. Validates presence of 'date' and 'headline' columns.
        2. Normalizes dates to standard 'YYYY-MM-DD' format.
        3. Applies `get_sentiment_score` to all headlines (supports optional batching
           for high-throughput GPU inference).
        4. Groups results by 'date' and calculates the daily mean sentiment score.

        Args:
            df: Input pandas DataFrame containing 'date' and 'headline' columns.
            batch_size: Mini-batch size for inference (default 32). If None or 1,
                        applies `get_sentiment_score` per headline.

        Returns:
            pd.DataFrame: Clean DataFrame with columns ['date', 'daily_sentiment'],
                          sorted chronologically and ready for merging with OHLCV data.
        """
        # Validate schema
        required_cols = {"date", "headline"}
        if not required_cols.issubset(df.columns):
            raise KeyError(
                f"Input DataFrame must contain columns {required_cols}. "
                f"Found: {list(df.columns)}"
            )

        if df.empty:
            return pd.DataFrame(columns=["date", "daily_sentiment"])

        news_df = df.copy()

        # Standardize date format to YYYY-MM-DD
        news_df["date"] = pd.to_datetime(news_df["date"]).dt.strftime("%Y-%m-%d")

        # Score headlines: apply get_sentiment_score or run batched inference
        if batch_size is not None and batch_size > 1 and len(news_df) > 1:
            news_df["sentiment_score"] = self._compute_batch_scores(
                news_df["headline"].astype(str).tolist(),
                batch_size=batch_size,
            )
        else:
            news_df["sentiment_score"] = news_df["headline"].astype(str).apply(self.get_sentiment_score)

        # Requirement 3 & 4: Group by 'date' and calculate daily average sentiment score
        daily_sentiment_df = (
            news_df.groupby("date", as_index=False)["sentiment_score"]
            .mean()
            .rename(columns={"sentiment_score": "daily_sentiment"})
        )

        # Sort chronologically and reset index
        daily_sentiment_df = daily_sentiment_df.sort_values(by="date").reset_index(drop=True)
        return daily_sentiment_df

    def _compute_batch_scores(self, texts: List[str], batch_size: int = 32) -> List[float]:
        """
        Optimized internal helper to evaluate sentiment in mini-batches on GPU/CPU.

        Args:
            texts: List of text headlines.
            batch_size: Batch size for model forward pass.

        Returns:
            List[float]: Sentiment scores for each text in the list.
        """
        scores: List[float] = []

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]
            cleaned_batch = [t if isinstance(t, str) and t.strip() else "" for t in batch_texts]

            inputs = self.tokenizer(
                cleaned_batch,
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
            batch_scores = np.clip(prob_pos - prob_neg, -1.0, 1.0)
            scores.extend(batch_scores.tolist())

        return scores


if __name__ == "__main__":
    # Create mock DataFrame of 5 financial headlines across 2 days
    mock_data = {
        "date": [
            "2026-11-03",
            "2026-11-03",
            "2026-11-03",
            "2026-11-04",
            "2026-11-04",
        ],
        "headline": [
            "Apple reports record quarterly revenue and surges past Wall Street expectations.",
            "Tech sector rallies as inflation pressures ease and consumer spending rebounds.",
            "Supply chain disruptions threaten holiday iPhone shipment timelines.",
            "Federal Reserve signals prolonged higher interest rates amid stubborn inflation.",
            "Semiconductor analysts downgrade hardware chipmakers citing inventory oversupply.",
        ],
    }
    mock_df = pd.DataFrame(mock_data)

    print("==================================================")
    print("      Financial Sentiment Processor (FinBERT)     ")
    print("==================================================")

    # Initialize the sentiment processor
    processor = SentimentProcessor()
    print(f"Device initialized: {processor.device.type.upper()}")
    print(f"Model loaded: {processor.model_name}")

    print("\n--- Input Financial News Headlines ---")
    for idx, row in mock_df.iterrows():
        print(f"[{row['date']}] {row['headline']}")

    # Process and aggregate sentiment across dates
    print("\nProcessing sentiment and aggregating by date...")
    daily_sentiment_df = processor.process_news_dataframe(mock_df)

    print("\n--- Aggregated Daily Sentiment DataFrame ---")
    print(daily_sentiment_df)

    print("\nDataFrame Info:")
    print(daily_sentiment_df.dtypes)

