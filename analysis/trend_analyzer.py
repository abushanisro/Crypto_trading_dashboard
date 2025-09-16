"""Professional-grade trend line detection for trading annotations."""
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
from datetime import datetime
import math

from models import (
    TrendLine, PivotPoint, BreakoutSignal, TrendDirection,
    BreakoutType, SignalType, TrendAnalysis
)
from utils.logger import get_logger
from config import settings

logger = get_logger("professional_trend_analyzer")


class ProfessionalTrendAnalyzer:
    """Professional-grade trend line detection suitable for real trading."""

    def __init__(self):
        # Professional trading standards - More adaptive
        self.min_touches = 2  # Reduced for more trend lines
        self.touch_precision = 0.003  # 0.3% precision (more flexible)
        self.min_line_length = 20  # Reduced minimum length
        self.max_lines_per_direction = 3  # Allow more lines
        self.min_significance = 0.85  # Lowered R² requirement for more trend lines
        self.volume_weight = 0.3  # Volume confirmation weight

        # Trend line validation
        self.min_angle = 5  # Lower minimum angle
        self.max_angle = 85  # Higher maximum angle
        self.recent_bias_periods = 50  # Recent data gets higher weight

        # Adaptive thresholds per symbol
        self.adaptive_thresholds = True
        self.fallback_mode = True  # Generate lines even with lower quality

    def find_professional_pivots(self, df: pd.DataFrame) -> Tuple[List[PivotPoint], List[PivotPoint]]:
        """Find high-quality pivot points using professional criteria."""
        if len(df) < 50:
            return [], []

        try:
            highs = df['high'].values
            lows = df['low'].values
            closes = df['close'].values
            volumes = df['volume'].values
            timestamps = df['timestamp'].values

            # Calculate market structure metrics
            atr = self._calculate_atr(df, 14)
            volume_avg = np.mean(volumes[-50:])

            pivot_highs = []
            pivot_lows = []

            # Dynamic window based on volatility
            base_window = 5
            volatility_factor = atr / np.mean(closes[-50:])
            window = max(3, min(8, int(base_window * (1 + volatility_factor * 20))))

            # Find swing highs with professional criteria
            for i in range(window * 2, len(highs) - window):
                # Check if it's a clear swing high
                left_max = np.max(highs[i-window*2:i])
                right_max = np.max(highs[i+1:i+window+1])

                if highs[i] > left_max and highs[i] > right_max:
                    # Additional validation criteria

                    # 1. Significant price level (not noise)
                    local_range = np.max(highs[i-10:i+10]) - np.min(lows[i-10:i+10])
                    if local_range > atr * 0.5:  # Must be significant move

                        # 2. Volume confirmation
                        vol_score = volumes[i] / volume_avg

                        # 3. Price action quality (close near high)
                        wick_ratio = (highs[i] - closes[i]) / (highs[i] - lows[i] + 1e-8)

                        # 4. Time since last pivot (avoid clustering)
                        time_valid = True
                        if pivot_highs:
                            last_pivot_idx = pivot_highs[-1].index
                            if i - last_pivot_idx < window:
                                time_valid = False

                        # Quality score (0-100)
                        quality = 50  # Base score
                        quality += min(25, vol_score * 10)  # Volume component
                        quality += min(15, (1 - wick_ratio) * 30)  # Price action
                        quality += min(10, (local_range / atr) * 2)  # Significance

                        if quality >= 65 and time_valid:
                            pivot_highs.append(PivotPoint(timestamps[i], highs[i], i))

            # Find swing lows with same professional criteria
            for i in range(window * 2, len(lows) - window):
                left_min = np.min(lows[i-window*2:i])
                right_min = np.min(lows[i+1:i+window+1])

                if lows[i] < left_min and lows[i] < right_min:
                    local_range = np.max(highs[i-10:i+10]) - np.min(lows[i-10:i+10])

                    if local_range > atr * 0.5:
                        vol_score = volumes[i] / volume_avg
                        wick_ratio = (closes[i] - lows[i]) / (highs[i] - lows[i] + 1e-8)

                        time_valid = True
                        if pivot_lows:
                            last_pivot_idx = pivot_lows[-1].index
                            if i - last_pivot_idx < window:
                                time_valid = False

                        quality = 50
                        quality += min(25, vol_score * 10)
                        quality += min(15, (1 - wick_ratio) * 30)
                        quality += min(10, (local_range / atr) * 2)

                        if quality >= 65 and time_valid:
                            pivot_lows.append(PivotPoint(timestamps[i], lows[i], i))

            # Keep only most significant pivots
            pivot_highs = sorted(pivot_highs, key=lambda p: p.price, reverse=True)[:10]
            pivot_lows = sorted(pivot_lows, key=lambda p: p.price)[:10]

            logger.debug(f"Found {len(pivot_highs)} professional pivot highs and {len(pivot_lows)} pivot lows")
            return pivot_highs, pivot_lows

        except Exception as e:
            logger.error(f"Error finding professional pivots: {e}")
            return [], []

    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> float:
        """Calculate Average True Range for volatility measurement."""
        try:
            high = df['high'].values
            low = df['low'].values
            close = df['close'].values

            tr_list = []
            for i in range(1, len(high)):
                tr = max(
                    high[i] - low[i],
                    abs(high[i] - close[i-1]),
                    abs(low[i] - close[i-1])
                )
                tr_list.append(tr)

            return np.mean(tr_list[-period:]) if len(tr_list) >= period else np.mean(tr_list)

        except Exception:
            return np.std(df['close'].values[-20:])  # Fallback

    def find_professional_trend_lines(self, df: pd.DataFrame) -> Dict[str, List[TrendLine]]:
        """Find professional-grade trend lines suitable for trading."""
        if len(df) < self.min_line_length:
            return {'support_lines': [], 'resistance_lines': []}

        try:
            pivot_highs, pivot_lows = self.find_professional_pivots(df)

            # Find the best trend lines
            resistance_lines = self._find_best_trend_lines(df, pivot_highs, 'resistance')
            support_lines = self._find_best_trend_lines(df, pivot_lows, 'support')

            # Fallback mode: Generate predictive trend lines if none found
            if not resistance_lines and self.fallback_mode:
                resistance_lines = self._generate_predictive_trend_lines(df, 'resistance')

            if not support_lines and self.fallback_mode:
                support_lines = self._generate_predictive_trend_lines(df, 'support')

            logger.info(f"Generated {len(resistance_lines)} resistance and {len(support_lines)} support lines")

            return {
                'resistance_lines': resistance_lines,
                'support_lines': support_lines
            }

        except Exception as e:
            logger.error(f"Error finding professional trend lines: {e}")
            return {'support_lines': [], 'resistance_lines': []}

    def _find_best_trend_lines(self, df: pd.DataFrame, pivots: List[PivotPoint], line_type: str) -> List[TrendLine]:
        """Find the best trend lines from pivot points."""
        if len(pivots) < 2:
            return []

        valid_lines = []

        # Try all combinations of pivots
        for i in range(len(pivots)):
            for j in range(i + 1, len(pivots)):
                line = self._create_trend_line(df, [pivots[i], pivots[j]], line_type)

                if line and self._validate_trend_line(df, line):
                    # Find additional touches
                    touches = self._count_professional_touches(df, line, line_type)
                    line.touches = touches

                    if touches >= self.min_touches:
                        line.strength = self._calculate_professional_strength(df, line)
                        valid_lines.append(line)

        # Select best lines
        valid_lines = sorted(valid_lines, key=lambda x: x.strength, reverse=True)

        # Remove overlapping lines (keep best one)
        final_lines = []
        for line in valid_lines[:self.max_lines_per_direction]:
            if not self._is_overlapping(line, final_lines):
                final_lines.append(line)

        return final_lines

    def _create_trend_line(self, df: pd.DataFrame, points: List[PivotPoint], line_type: str) -> Optional[TrendLine]:
        """Create a trend line with professional validation."""
        if len(points) < 2:
            return None

        try:
            # Extract data
            indices = [p.index for p in points]
            prices = [p.price for p in points]

            # Check minimum length
            if indices[-1] - indices[0] < self.min_line_length:
                return None

            # Linear regression with weights (recent data more important)
            weights = []
            total_periods = len(df)
            for idx in indices:
                # Weight based on recency
                recency_weight = 1.0 + (idx / total_periods) * 0.5
                weights.append(recency_weight)

            # Weighted linear regression
            weights = np.array(weights)
            x_vals = np.array(indices)
            y_vals = np.array(prices)

            # Calculate weighted regression
            w_sum = np.sum(weights)
            x_mean = np.sum(weights * x_vals) / w_sum
            y_mean = np.sum(weights * y_vals) / w_sum

            numerator = np.sum(weights * (x_vals - x_mean) * (y_vals - y_mean))
            denominator = np.sum(weights * (x_vals - x_mean) ** 2)

            if denominator == 0:
                return None

            slope = numerator / denominator
            intercept = y_mean - slope * x_mean

            # Calculate R-squared
            y_pred = slope * x_vals + intercept
            ss_res = np.sum(weights * (y_vals - y_pred) ** 2)
            ss_tot = np.sum(weights * (y_vals - y_mean) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

            # Validate angle
            angle = abs(math.degrees(math.atan(slope)))
            if angle < self.min_angle or angle > self.max_angle:
                return None

            return TrendLine(
                slope=slope,
                intercept=intercept,
                r_squared=r_squared,
                points=points,
                start_idx=min(indices),
                end_idx=max(indices),
                type=line_type
            )

        except Exception as e:
            logger.error(f"Error creating trend line: {e}")
            return None

    def _validate_trend_line(self, df: pd.DataFrame, line: TrendLine) -> bool:
        """Validate trend line meets professional standards with adaptive thresholds."""
        # Adaptive R-squared requirement
        min_r2 = self.min_significance
        if self.fallback_mode and line.r_squared >= 0.7:  # Allow lower quality in fallback
            min_r2 = 0.7

        if line.r_squared < min_r2:
            return False

        # More flexible line extension requirement
        latest_idx = len(df) - 1
        max_gap = self.recent_bias_periods * 1.5  # More flexible
        if latest_idx - line.end_idx > max_gap:
            return False

        # More lenient breach tolerance for different coins
        recent_data = df.iloc[max(0, line.end_idx):].copy()
        if len(recent_data) > 5:
            breach_tolerance = 0.05  # 5% breach tolerance (more lenient)
            breach_count = 0
            max_breaches = len(recent_data) // 3  # Allow up to 1/3 of points to breach

            for idx, row in recent_data.iterrows():
                expected_price = line.slope * idx + line.intercept
                actual_price = row['high'] if line.type == 'resistance' else row['low']

                # Check for significant breach
                deviation = abs(actual_price - expected_price) / expected_price
                if deviation > breach_tolerance:
                    breach_count += 1
                    if breach_count > max_breaches:
                        return False

        return True

    def _count_professional_touches(self, df: pd.DataFrame, line: TrendLine, line_type: str) -> int:
        """Count high-quality touches of the trend line."""
        touches = 0
        price_col = 'high' if line_type == 'resistance' else 'low'

        for i in range(line.start_idx, min(line.end_idx + 1, len(df))):
            expected_price = line.slope * i + line.intercept
            actual_price = df.iloc[i][price_col]

            # Precise touch calculation
            deviation = abs(actual_price - expected_price) / expected_price
            if deviation <= self.touch_precision:
                touches += 1

        return touches

    def _calculate_professional_strength(self, df: pd.DataFrame, line: TrendLine) -> float:
        """Calculate professional strength score (0-100)."""
        strength = 0

        # R-squared component (40 points max)
        strength += line.r_squared * 40

        # Touch quality (30 points max)
        strength += min(30, line.touches * 8)

        # Length component (20 points max)
        length_factor = (line.end_idx - line.start_idx) / len(df)
        strength += min(20, length_factor * 50)

        # Recency bonus (10 points max)
        latest_idx = len(df) - 1
        recency_factor = 1.0 - ((latest_idx - line.end_idx) / latest_idx)
        strength += recency_factor * 10

        return min(100, strength)

    def _is_overlapping(self, new_line: TrendLine, existing_lines: List[TrendLine]) -> bool:
        """Check if trend line overlaps with existing ones."""
        for existing in existing_lines:
            # Check if lines are too similar (slope and intercept)
            slope_diff = abs(new_line.slope - existing.slope) / abs(existing.slope + 1e-8)

            if slope_diff < 0.1:  # Less than 10% difference in slope
                # Check intercept similarity at midpoint
                midpoint = (new_line.start_idx + new_line.end_idx) / 2
                new_price = new_line.slope * midpoint + new_line.intercept
                existing_price = existing.slope * midpoint + existing.intercept

                price_diff = abs(new_price - existing_price) / existing_price
                if price_diff < 0.02:  # Less than 2% price difference
                    return True

        return False

    def detect_professional_breakouts(self, df: pd.DataFrame, trend_lines: Dict[str, List[TrendLine]]) -> List[BreakoutSignal]:
        """Detect high-confidence breakouts suitable for trading."""
        if len(df) < 2:
            return []

        breakouts = []
        latest_candle = df.iloc[-1]
        prev_candle = df.iloc[-2]
        current_idx = len(df) - 1

        try:
            # Check resistance breakouts
            for line in trend_lines['resistance_lines']:
                expected_price = line.slope * current_idx + line.intercept

                # Professional breakout criteria
                if (prev_candle['close'] <= expected_price * 0.999 and  # Was below
                    latest_candle['close'] > expected_price * 1.002 and  # Broke above
                    latest_candle['volume'] > df['volume'].rolling(20).mean().iloc[-1] * 1.5):  # Volume confirmation

                    # Additional validation
                    body_size = abs(latest_candle['close'] - latest_candle['open'])
                    total_range = latest_candle['high'] - latest_candle['low']

                    if body_size / total_range > 0.6:  # Strong body indicates conviction
                        confidence = min(95, line.strength + 20)  # Base + breakout bonus

                        breakouts.append(BreakoutSignal(
                            type=BreakoutType.RESISTANCE_BREAKOUT,
                            direction='bullish',
                            strength=line.strength,
                            price=expected_price,
                            signal=SignalType.BUY,
                            confidence=confidence,
                            timestamp=datetime.now()
                        ))

            # Check support breakdowns
            for line in trend_lines['support_lines']:
                expected_price = line.slope * current_idx + line.intercept

                if (prev_candle['close'] >= expected_price * 1.001 and  # Was above
                    latest_candle['close'] < expected_price * 0.998 and  # Broke below
                    latest_candle['volume'] > df['volume'].rolling(20).mean().iloc[-1] * 1.5):  # Volume confirmation

                    body_size = abs(latest_candle['close'] - latest_candle['open'])
                    total_range = latest_candle['high'] - latest_candle['low']

                    if body_size / total_range > 0.6:
                        confidence = min(95, line.strength + 20)

                        breakouts.append(BreakoutSignal(
                            type=BreakoutType.SUPPORT_BREAKDOWN,
                            direction='bearish',
                            strength=line.strength,
                            price=expected_price,
                            signal=SignalType.SELL,
                            confidence=confidence,
                            timestamp=datetime.now()
                        ))

            if breakouts:
                logger.info(f"Detected {len(breakouts)} professional breakouts")

        except Exception as e:
            logger.error(f"Error detecting breakouts: {e}")

        return breakouts

    def analyze_professional_trends(self, symbol: str, df: pd.DataFrame) -> TrendAnalysis:
        """Complete professional trend analysis."""
        try:
            trend_lines = self.find_professional_trend_lines(df)
            breakouts = self.detect_professional_breakouts(df, trend_lines)
            current_trend = self._determine_professional_trend(df)

            # Calculate overall market structure strength
            structure_strength = 0
            if trend_lines['support_lines']:
                structure_strength += np.mean([line.strength for line in trend_lines['support_lines']])
            if trend_lines['resistance_lines']:
                structure_strength += np.mean([line.strength for line in trend_lines['resistance_lines']])

            structure_strength = structure_strength / 2 if trend_lines['support_lines'] and trend_lines['resistance_lines'] else structure_strength

            return TrendAnalysis(
                symbol=symbol,
                current_trend=current_trend,
                support_lines=trend_lines['support_lines'],
                resistance_lines=trend_lines['resistance_lines'],
                breakouts=breakouts,
                timestamp=datetime.now(),
                strength_score=structure_strength
            )

        except Exception as e:
            logger.error(f"Error in professional trend analysis for {symbol}: {e}")
            return TrendAnalysis(
                symbol=symbol,
                current_trend=TrendDirection.INSUFFICIENT_DATA,
                support_lines=[],
                resistance_lines=[],
                breakouts=[],
                timestamp=datetime.now(),
                strength_score=0.0
            )

    def _determine_professional_trend(self, df: pd.DataFrame) -> TrendDirection:
        """Determine trend using professional market structure analysis."""
        if len(df) < 100:
            return TrendDirection.INSUFFICIENT_DATA

        try:
            # Multiple timeframe analysis
            closes = df['close'].values

            # Short-term: Last 20 periods
            short_term = closes[-20:]
            short_slope = (short_term[-1] - short_term[0]) / len(short_term)
            short_slope_pct = short_slope / short_term[0]

            # Medium-term: Last 50 periods
            medium_term = closes[-50:]
            medium_slope = (medium_term[-1] - medium_term[0]) / len(medium_term)
            medium_slope_pct = medium_slope / medium_term[0]

            # Long-term: Last 100 periods
            long_term = closes[-100:]
            long_slope = (long_term[-1] - long_term[0]) / len(long_term)
            long_slope_pct = long_slope / long_term[0]

            # Higher highs / lower lows analysis
            recent_highs = df['high'].tail(50)
            recent_lows = df['low'].tail(50)

            higher_highs = recent_highs.iloc[-1] > recent_highs.iloc[-20:].max() * 0.999
            higher_lows = recent_lows.iloc[-1] > recent_lows.iloc[-20:].min() * 1.001
            lower_highs = recent_highs.iloc[-1] < recent_highs.iloc[-20:].max() * 1.001
            lower_lows = recent_lows.iloc[-1] < recent_lows.iloc[-20:].min() * 0.999

            # Trend determination logic
            if short_slope_pct > 0.005 and medium_slope_pct > 0.003 and (higher_highs and higher_lows):
                return TrendDirection.STRONG_UPTREND
            elif short_slope_pct > 0.002 and medium_slope_pct > 0.001:
                return TrendDirection.UPTREND
            elif short_slope_pct > 0 and medium_slope_pct <= 0:
                return TrendDirection.SHORT_TERM_UPTREND
            elif short_slope_pct <= 0 and medium_slope_pct > 0:
                return TrendDirection.PULLBACK_IN_UPTREND
            elif short_slope_pct < -0.005 and medium_slope_pct < -0.003 and (lower_highs and lower_lows):
                return TrendDirection.STRONG_DOWNTREND
            elif short_slope_pct < -0.002 and medium_slope_pct < -0.001:
                return TrendDirection.DOWNTREND
            else:
                return TrendDirection.SIDEWAYS

        except Exception as e:
            logger.error(f"Error determining trend: {e}")
            return TrendDirection.INSUFFICIENT_DATA