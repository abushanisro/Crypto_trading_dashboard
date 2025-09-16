"""Technical indicators calculations."""
import pandas as pd
import numpy as np
from typing import Optional

from utils.logger import get_logger

logger = get_logger("technical_indicators")


class TechnicalIndicators:
    """Technical indicators calculator."""

    @staticmethod
    def calculate_moving_averages(df: pd.DataFrame) -> pd.DataFrame:
        """Calculate moving averages."""
        try:
            df = df.copy()
            df["MA_20"] = df["close"].rolling(window=20, min_periods=1).mean()
            df["MA_50"] = df["close"].rolling(window=50, min_periods=1).mean()
            return df
        except Exception as e:
            logger.error(f"Error calculating moving averages: {e}")
            return df

    @staticmethod
    def calculate_bollinger_bands(df: pd.DataFrame, period: int = 20, std_dev: float = 2) -> pd.DataFrame:
        """Calculate Bollinger Bands."""
        try:
            df = df.copy()
            df["BB_middle"] = df["close"].rolling(window=period, min_periods=1).mean()
            df["BB_std"] = df["close"].rolling(window=period, min_periods=1).std()
            df["BB_upper"] = df["BB_middle"] + (std_dev * df["BB_std"])
            df["BB_lower"] = df["BB_middle"] - (std_dev * df["BB_std"])
            return df
        except Exception as e:
            logger.error(f"Error calculating Bollinger Bands: {e}")
            return df

    @staticmethod
    def calculate_rsi(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """Calculate Relative Strength Index."""
        try:
            df = df.copy()
            delta = df["close"].diff()
            gain = delta.clip(lower=0)
            loss = -1 * delta.clip(upper=0)

            avg_gain = gain.rolling(window=period, min_periods=1).mean()
            avg_loss = loss.rolling(window=period, min_periods=1).mean()

            # Avoid division by zero
            rs = avg_gain / avg_loss.replace(0, np.inf)
            df["RSI"] = 100 - (100 / (1 + rs))

            # Handle edge cases
            df["RSI"] = df["RSI"].fillna(50)  # Neutral RSI for NaN values
            df["RSI"] = np.clip(df["RSI"], 0, 100)

            return df
        except Exception as e:
            logger.error(f"Error calculating RSI: {e}")
            return df

    @staticmethod
    def calculate_macd(df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
        """Calculate MACD (Moving Average Convergence Divergence)."""
        try:
            df = df.copy()
            df["EMA_fast"] = df["close"].ewm(span=fast, min_periods=1).mean()
            df["EMA_slow"] = df["close"].ewm(span=slow, min_periods=1).mean()
            df["MACD"] = df["EMA_fast"] - df["EMA_slow"]
            df["MACD_signal"] = df["MACD"].ewm(span=signal, min_periods=1).mean()
            df["MACD_histogram"] = df["MACD"] - df["MACD_signal"]
            return df
        except Exception as e:
            logger.error(f"Error calculating MACD: {e}")
            return df

    @staticmethod
    def calculate_volume_indicators(df: pd.DataFrame) -> pd.DataFrame:
        """Calculate volume-based indicators."""
        try:
            df = df.copy()
            df["Volume_MA"] = df["volume"].rolling(window=20, min_periods=1).mean()
            df["Volume_ratio"] = df["volume"] / df["Volume_MA"]
            return df
        except Exception as e:
            logger.error(f"Error calculating volume indicators: {e}")
            return df

    @classmethod
    def calculate_all_indicators(cls, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate all technical indicators."""
        if df.empty:
            logger.warning("Empty DataFrame provided to calculate_all_indicators")
            return df

        try:
            logger.debug(f"Calculating indicators for {len(df)} data points")

            # Calculate all indicators
            df = cls.calculate_moving_averages(df)
            df = cls.calculate_bollinger_bands(df)
            df = cls.calculate_rsi(df)
            df = cls.calculate_macd(df)
            df = cls.calculate_volume_indicators(df)

            logger.debug("All technical indicators calculated successfully")
            return df

        except Exception as e:
            logger.error(f"Error calculating technical indicators: {e}")
            return df