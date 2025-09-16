"""Tests for trend analyzer."""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from analysis.trend_analyzer import TrendLineAnalyzer
from models import TrendDirection


class TestTrendLineAnalyzer:
    """Test cases for TrendLineAnalyzer class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.analyzer = TrendLineAnalyzer()

        # Create uptrend data
        dates = [datetime.now() - timedelta(minutes=i) for i in range(100, 0, -1)]
        base_price = 100

        prices = []
        for i in range(100):
            # Create uptrend with some noise
            trend_component = i * 0.5  # Upward trend
            noise = np.random.normal(0, 1)  # Random noise
            price = base_price + trend_component + noise
            prices.append(max(price, 50))  # Ensure positive prices

        self.uptrend_df = pd.DataFrame({
            'timestamp': dates,
            'open': prices,
            'high': [p * 1.02 for p in prices],
            'low': [p * 0.98 for p in prices],
            'close': prices,
            'volume': [1000000] * 100
        })

        # Create downtrend data
        prices_down = []
        for i in range(100):
            trend_component = -i * 0.3  # Downward trend
            noise = np.random.normal(0, 1)
            price = base_price + 50 + trend_component + noise  # Start higher
            prices_down.append(max(price, 20))

        self.downtrend_df = pd.DataFrame({
            'timestamp': dates,
            'open': prices_down,
            'high': [p * 1.02 for p in prices_down],
            'low': [p * 0.98 for p in prices_down],
            'close': prices_down,
            'volume': [1000000] * 100
        })

        # Create sideways data
        sideways_prices = [100 + np.random.normal(0, 2) for _ in range(100)]
        self.sideways_df = pd.DataFrame({
            'timestamp': dates,
            'open': sideways_prices,
            'high': [p * 1.01 for p in sideways_prices],
            'low': [p * 0.99 for p in sideways_prices],
            'close': sideways_prices,
            'volume': [1000000] * 100
        })

    def test_find_pivot_points(self):
        """Test pivot point detection."""
        pivot_highs, pivot_lows = self.analyzer.find_pivot_points(self.uptrend_df)

        assert isinstance(pivot_highs, list)
        assert isinstance(pivot_lows, list)
        assert len(pivot_highs) > 0
        assert len(pivot_lows) > 0

        # Check that pivot points have required attributes
        if pivot_highs:
            pivot = pivot_highs[0]
            assert hasattr(pivot, 'timestamp')
            assert hasattr(pivot, 'price')
            assert hasattr(pivot, 'index')

    def test_find_trend_lines(self):
        """Test trend line detection."""
        trend_lines = self.analyzer.find_trend_lines(self.uptrend_df)

        assert 'support_lines' in trend_lines
        assert 'resistance_lines' in trend_lines
        assert isinstance(trend_lines['support_lines'], list)
        assert isinstance(trend_lines['resistance_lines'], list)

    def test_get_current_trend_uptrend(self):
        """Test trend detection for uptrend data."""
        trend = self.analyzer.get_current_trend(self.uptrend_df)

        assert trend in [
            TrendDirection.UPTREND,
            TrendDirection.STRONG_UPTREND,
            TrendDirection.SHORT_TERM_UPTREND
        ]

    def test_get_current_trend_downtrend(self):
        """Test trend detection for downtrend data."""
        trend = self.analyzer.get_current_trend(self.downtrend_df)

        assert trend in [
            TrendDirection.DOWNTREND,
            TrendDirection.STRONG_DOWNTREND
        ]

    def test_get_current_trend_sideways(self):
        """Test trend detection for sideways data."""
        trend = self.analyzer.get_current_trend(self.sideways_df)

        # Should detect sideways or insufficient data
        assert trend in [TrendDirection.SIDEWAYS, TrendDirection.INSUFFICIENT_DATA]

    def test_get_current_trend_insufficient_data(self):
        """Test trend detection with insufficient data."""
        small_df = self.uptrend_df.head(10)
        trend = self.analyzer.get_current_trend(small_df)

        assert trend == TrendDirection.INSUFFICIENT_DATA

    def test_analyze_trends(self):
        """Test complete trend analysis."""
        analysis = self.analyzer.analyze_trends("TEST/USDT", self.uptrend_df)

        assert analysis.symbol == "TEST/USDT"
        assert isinstance(analysis.current_trend, TrendDirection)
        assert isinstance(analysis.support_lines, list)
        assert isinstance(analysis.resistance_lines, list)
        assert isinstance(analysis.breakouts, list)
        assert analysis.timestamp is not None
        assert analysis.strength_score >= 0

    def test_detect_breakouts(self):
        """Test breakout detection."""
        trend_lines = self.analyzer.find_trend_lines(self.uptrend_df)
        breakouts = self.analyzer.detect_breakouts(self.uptrend_df, trend_lines)

        assert isinstance(breakouts, list)
        # Breakouts may or may not exist depending on the data

    def test_count_touches(self):
        """Test trend line touches counting."""
        trend_lines = self.analyzer.find_trend_lines(self.uptrend_df)

        if trend_lines['support_lines']:
            trend_line = trend_lines['support_lines'][0]
            touches = self.analyzer.count_touches(self.uptrend_df, trend_line, 'low')
            assert isinstance(touches, int)
            assert touches >= 0

    def test_calculate_strength(self):
        """Test trend line strength calculation."""
        trend_lines = self.analyzer.find_trend_lines(self.uptrend_df)

        if trend_lines['support_lines']:
            trend_line = trend_lines['support_lines'][0]
            trend_line.touches = 3  # Set some touches
            strength = self.analyzer.calculate_strength(trend_line, self.uptrend_df)
            assert isinstance(strength, float)
            assert strength >= 0

    def test_empty_dataframe(self):
        """Test handling of empty dataframe."""
        empty_df = pd.DataFrame()
        trend_lines = self.analyzer.find_trend_lines(empty_df)

        assert trend_lines['support_lines'] == []
        assert trend_lines['resistance_lines'] == []

    def test_minimal_data(self):
        """Test handling of minimal data."""
        minimal_df = self.uptrend_df.head(15)  # Just enough data
        trend_lines = self.analyzer.find_trend_lines(minimal_df)

        # Should handle gracefully without crashing
        assert isinstance(trend_lines, dict)
        assert 'support_lines' in trend_lines
        assert 'resistance_lines' in trend_lines