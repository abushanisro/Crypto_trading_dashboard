"""Tests for signal generator."""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from trading.signal_generator import SignalGenerator
from analysis.technical_indicators import TechnicalIndicators
from analysis.trend_analyzer import TrendLineAnalyzer
from models import TrendDirection, SignalType


class TestSignalGenerator:
    """Test cases for SignalGenerator class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.generator = SignalGenerator()
        self.trend_analyzer = TrendLineAnalyzer()

        # Create sample data with technical indicators
        dates = [datetime.now() - timedelta(minutes=i) for i in range(100, 0, -1)]
        np.random.seed(42)

        # Generate trending price data
        prices = []
        base_price = 100
        for i in range(100):
            trend = i * 0.1  # Slight uptrend
            noise = np.random.normal(0, 1)
            price = base_price + trend + noise
            prices.append(max(price, 50))

        self.df = pd.DataFrame({
            'timestamp': dates,
            'open': prices,
            'high': [p * 1.02 for p in prices],
            'low': [p * 0.98 for p in prices],
            'close': prices,
            'volume': [np.random.randint(1000000, 5000000) for _ in range(100)]
        })

        # Add technical indicators
        self.df = TechnicalIndicators.calculate_all_indicators(self.df)

        # Create trend analysis
        self.trend_analysis = self.trend_analyzer.analyze_trends("TEST/USDT", self.df)

    def test_analyze_technical_conditions(self):
        """Test technical conditions analysis."""
        conditions = self.generator.analyze_technical_conditions(self.df, 50)

        assert isinstance(conditions, dict)
        expected_keys = ['rsi_score', 'macd_score', 'trend_score', 'volume_score', 'bb_score']

        for key in expected_keys:
            assert key in conditions
            assert isinstance(conditions[key], (int, float))

    def test_calculate_trend_bias(self):
        """Test trend bias calculation."""
        bias = self.generator.calculate_trend_bias(self.trend_analysis)

        assert isinstance(bias, float)
        # Should be reasonable range
        assert -10 <= bias <= 10

    def test_analyze_breakouts_empty(self):
        """Test breakout analysis with no breakouts."""
        analysis = self.generator.analyze_breakouts([])

        expected_keys = ['breakout_bullish', 'breakout_bearish', 'breakout_confidence']
        for key in expected_keys:
            assert key in analysis
            assert analysis[key] == 0

    def test_generate_enhanced_signals(self):
        """Test enhanced signal generation."""
        result_df = self.generator.generate_enhanced_signals(
            self.df,
            self.trend_analysis,
            "TEST/USDT"
        )

        assert 'Signal' in result_df.columns
        assert 'SignalReason' in result_df.columns
        assert len(result_df) == len(self.df)

        # Check that signals are valid types
        signals = result_df['Signal'].dropna()
        for signal in signals:
            assert signal in [SignalType.BUY.value, SignalType.SELL.value]

    def test_get_latest_signal(self):
        """Test getting the latest signal."""
        # First generate signals
        df_with_signals = self.generator.generate_enhanced_signals(
            self.df,
            self.trend_analysis,
            "TEST/USDT"
        )

        latest_signal = self.generator.get_latest_signal(df_with_signals)

        if latest_signal:
            assert isinstance(latest_signal.signal, SignalType)
            assert isinstance(latest_signal.price, float)
            assert isinstance(latest_signal.confidence, float)
            assert isinstance(latest_signal.reason, str)
            assert latest_signal.timestamp is not None

    def test_get_latest_signal_no_signals(self):
        """Test getting latest signal when no signals exist."""
        df_no_signals = self.df.copy()
        latest_signal = self.generator.get_latest_signal(df_no_signals)

        assert latest_signal is None

    def test_extract_confidence_from_reason(self):
        """Test confidence extraction from reason string."""
        reason = "BUY: RSI oversold, MACD bullish (Confidence: 75%)"
        confidence = self.generator._extract_confidence_from_reason(reason)

        assert confidence == 75.0

        # Test with no confidence
        reason_no_conf = "BUY: RSI oversold"
        confidence = self.generator._extract_confidence_from_reason(reason_no_conf)

        assert confidence == 50.0  # Default

    def test_empty_dataframe(self):
        """Test handling of empty dataframe."""
        empty_df = pd.DataFrame()
        result = self.generator.generate_enhanced_signals(
            empty_df,
            self.trend_analysis,
            "TEST/USDT"
        )

        assert result.empty

    def test_insufficient_data(self):
        """Test handling of insufficient data."""
        small_df = self.df.head(10)
        result = self.generator.generate_enhanced_signals(
            small_df,
            self.trend_analysis,
            "TEST/USDT"
        )

        # Should handle gracefully
        assert len(result) == 10
        assert 'Signal' in result.columns

    def test_conditions_boundary_values(self):
        """Test technical conditions with boundary RSI values."""
        # Test with extreme RSI values
        test_df = self.df.copy()
        test_df.loc[test_df.index[-1], 'RSI'] = 25  # Oversold
        conditions = self.generator.analyze_technical_conditions(test_df, len(test_df) - 1)

        assert conditions['rsi_score'] > 0  # Should be positive for oversold

        test_df.loc[test_df.index[-1], 'RSI'] = 75  # Overbought
        conditions = self.generator.analyze_technical_conditions(test_df, len(test_df) - 1)

        assert conditions['rsi_score'] < 0  # Should be negative for overbought