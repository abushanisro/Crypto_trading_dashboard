"""Tests for technical indicators."""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from analysis.technical_indicators import TechnicalIndicators


class TestTechnicalIndicators:
    """Test cases for TechnicalIndicators class."""

    def setup_method(self):
        """Set up test fixtures."""
        # Create sample OHLCV data
        dates = [datetime.now() - timedelta(days=i) for i in range(100, 0, -1)]
        np.random.seed(42)  # For reproducible tests

        # Generate realistic price data
        prices = []
        base_price = 100
        for i in range(100):
            change = np.random.normal(0, 0.02)  # 2% volatility
            base_price *= (1 + change)
            prices.append(base_price)

        self.df = pd.DataFrame({
            'timestamp': dates,
            'open': prices,
            'high': [p * (1 + abs(np.random.normal(0, 0.01))) for p in prices],
            'low': [p * (1 - abs(np.random.normal(0, 0.01))) for p in prices],
            'close': prices,
            'volume': [np.random.randint(1000000, 10000000) for _ in range(100)]
        })

    def test_calculate_moving_averages(self):
        """Test moving averages calculation."""
        result = TechnicalIndicators.calculate_moving_averages(self.df)

        assert 'MA_20' in result.columns
        assert 'MA_50' in result.columns
        assert not result['MA_20'].isna().all()
        assert not result['MA_50'].isna().all()

        # MA_20 should be more responsive than MA_50
        assert result['MA_20'].std() >= result['MA_50'].std()

    def test_calculate_bollinger_bands(self):
        """Test Bollinger Bands calculation."""
        result = TechnicalIndicators.calculate_bollinger_bands(self.df)

        assert 'BB_upper' in result.columns
        assert 'BB_lower' in result.columns
        assert 'BB_middle' in result.columns

        # Upper band should be above middle, middle above lower
        last_row = result.iloc[-1]
        assert last_row['BB_upper'] > last_row['BB_middle']
        assert last_row['BB_middle'] > last_row['BB_lower']

    def test_calculate_rsi(self):
        """Test RSI calculation."""
        result = TechnicalIndicators.calculate_rsi(self.df)

        assert 'RSI' in result.columns
        assert result['RSI'].min() >= 0
        assert result['RSI'].max() <= 100
        assert not result['RSI'].isna().all()

    def test_calculate_macd(self):
        """Test MACD calculation."""
        result = TechnicalIndicators.calculate_macd(self.df)

        assert 'MACD' in result.columns
        assert 'MACD_signal' in result.columns
        assert 'MACD_histogram' in result.columns
        assert not result['MACD'].isna().all()

        # Histogram should be MACD - Signal
        np.testing.assert_array_almost_equal(
            result['MACD_histogram'].fillna(0),
            (result['MACD'] - result['MACD_signal']).fillna(0),
            decimal=6
        )

    def test_calculate_volume_indicators(self):
        """Test volume indicators calculation."""
        result = TechnicalIndicators.calculate_volume_indicators(self.df)

        assert 'Volume_MA' in result.columns
        assert 'Volume_ratio' in result.columns
        assert not result['Volume_MA'].isna().all()

    def test_calculate_all_indicators(self):
        """Test calculation of all indicators."""
        result = TechnicalIndicators.calculate_all_indicators(self.df)

        # Check all expected columns are present
        expected_columns = [
            'MA_20', 'MA_50', 'BB_upper', 'BB_lower', 'BB_middle',
            'RSI', 'MACD', 'MACD_signal', 'MACD_histogram',
            'Volume_MA', 'Volume_ratio'
        ]

        for col in expected_columns:
            assert col in result.columns

    def test_empty_dataframe(self):
        """Test handling of empty dataframe."""
        empty_df = pd.DataFrame()
        result = TechnicalIndicators.calculate_all_indicators(empty_df)

        assert result.empty

    def test_insufficient_data(self):
        """Test handling of insufficient data."""
        small_df = self.df.head(5)  # Only 5 rows
        result = TechnicalIndicators.calculate_all_indicators(small_df)

        # Should not crash and should return some results
        assert len(result) == 5
        assert 'MA_20' in result.columns