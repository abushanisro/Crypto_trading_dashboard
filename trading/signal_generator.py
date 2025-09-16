"""Trading signal generation with trend line analysis."""
from typing import List, Dict, Optional
import pandas as pd
import numpy as np
from datetime import datetime

from models import (
    TradingSignal, SignalType, TrendDirection, TrendAnalysis,
    BreakoutSignal, TechnicalIndicators
)
from utils.logger import get_logger

logger = get_logger("signal_generator")


class SignalGenerator:
    """Advanced trading signal generator with trend line integration."""

    def __init__(self):
        self.min_signal_strength = 3  # Minimum strength to generate signal

    def analyze_technical_conditions(self, df: pd.DataFrame, row_idx: int) -> Dict[str, float]:
        """Analyze technical conditions and return scoring."""
        if row_idx < 1 or row_idx >= len(df):
            return {}

        try:
            row = df.iloc[row_idx]
            prev_row = df.iloc[row_idx - 1]

            conditions = {
                'rsi_score': 0,
                'macd_score': 0,
                'trend_score': 0,
                'volume_score': 0,
                'bb_score': 0
            }

            # RSI analysis
            rsi = row.get("RSI", 50)
            if rsi < 30:
                conditions['rsi_score'] = 2  # Strong oversold
            elif rsi < 45:
                conditions['rsi_score'] = 1  # Mild oversold
            elif rsi > 70:
                conditions['rsi_score'] = -2  # Strong overbought
            elif rsi > 55:
                conditions['rsi_score'] = -1  # Mild overbought

            # MACD analysis
            macd = row.get("MACD", 0)
            macd_signal = row.get("MACD_signal", 0)
            prev_macd = prev_row.get("MACD", 0)
            prev_macd_signal = prev_row.get("MACD_signal", 0)

            if macd > macd_signal and prev_macd <= prev_macd_signal:
                conditions['macd_score'] = 2  # Bullish crossover
            elif macd < macd_signal and prev_macd >= prev_macd_signal:
                conditions['macd_score'] = -2  # Bearish crossover
            elif macd > macd_signal:
                conditions['macd_score'] = 1
            else:
                conditions['macd_score'] = -1

            # Trend analysis
            ma_20 = row.get("MA_20", 0)
            ma_50 = row.get("MA_50", 0)
            if ma_20 > ma_50:
                conditions['trend_score'] = 1
            else:
                conditions['trend_score'] = -1

            # Volume confirmation
            volume = row.get("volume", 0)
            volume_ma = row.get("Volume_MA", 0)
            if volume > volume_ma * 1.2:  # High volume
                conditions['volume_score'] = 1

            # Bollinger Bands
            close = row.get("close", 0)
            bb_upper = row.get("BB_upper", close)
            bb_lower = row.get("BB_lower", close)

            if close <= bb_lower:
                conditions['bb_score'] = 1  # Oversold
            elif close >= bb_upper:
                conditions['bb_score'] = -1  # Overbought

            return conditions

        except Exception as e:
            logger.error(f"Error analyzing technical conditions: {e}")
            return {}

    def calculate_trend_bias(self, trend_analysis: TrendAnalysis) -> float:
        """Calculate trend bias score."""
        try:
            bias_score = 0

            # Current trend bias
            trend_weights = {
                TrendDirection.STRONG_UPTREND: 3,
                TrendDirection.UPTREND: 2,
                TrendDirection.SHORT_TERM_UPTREND: 1,
                TrendDirection.PULLBACK_IN_UPTREND: 0.5,
                TrendDirection.SIDEWAYS: 0,
                TrendDirection.STRONG_DOWNTREND: -3,
                TrendDirection.DOWNTREND: -2,
                TrendDirection.INSUFFICIENT_DATA: 0
            }

            bias_score += trend_weights.get(trend_analysis.current_trend, 0)

            # Support/resistance strength
            support_strength = sum(line.strength for line in trend_analysis.support_lines) / 100
            resistance_strength = sum(line.strength for line in trend_analysis.resistance_lines) / 100

            # Stronger support = bullish bias, stronger resistance = bearish bias
            bias_score += (support_strength - resistance_strength) * 0.5

            return bias_score

        except Exception as e:
            logger.error(f"Error calculating trend bias: {e}")
            return 0

    def analyze_breakouts(self, breakouts: List[BreakoutSignal]) -> Dict[str, float]:
        """Analyze breakout signals."""
        try:
            breakout_analysis = {
                'breakout_bullish': 0,
                'breakout_bearish': 0,
                'breakout_confidence': 0
            }

            if not breakouts:
                return breakout_analysis

            for breakout in breakouts:
                confidence_weight = breakout.confidence / 100

                if breakout.signal == SignalType.BUY:
                    breakout_analysis['breakout_bullish'] += 5 * confidence_weight
                elif breakout.signal == SignalType.SELL:
                    breakout_analysis['breakout_bearish'] += 5 * confidence_weight

                breakout_analysis['breakout_confidence'] = max(
                    breakout_analysis['breakout_confidence'],
                    breakout.confidence
                )

            return breakout_analysis

        except Exception as e:
            logger.error(f"Error analyzing breakouts: {e}")
            return {'breakout_bullish': 0, 'breakout_bearish': 0, 'breakout_confidence': 0}

    def generate_enhanced_signals(
        self,
        df: pd.DataFrame,
        trend_analysis: TrendAnalysis,
        symbol: str
    ) -> pd.DataFrame:
        """Generate enhanced trading signals with trend line analysis."""
        if df.empty:
            return df

        try:
            signals = []
            reasons = []

            # Calculate trend bias once
            trend_bias = self.calculate_trend_bias(trend_analysis)
            breakout_analysis = self.analyze_breakouts(trend_analysis.breakouts)

            logger.debug(f"Trend bias for {symbol}: {trend_bias:.2f}")

            for i in range(len(df)):
                signal = None
                reason = ""

                if i < 50:  # Need enough data for indicators
                    signals.append(signal)
                    reasons.append(reason)
                    continue

                # Analyze technical conditions
                conditions = self.analyze_technical_conditions(df, i)
                if not conditions:
                    signals.append(signal)
                    reasons.append(reason)
                    continue

                # Calculate signal strength
                bullish_score = 0
                bearish_score = 0

                # Technical analysis scores
                for condition, score in conditions.items():
                    if score > 0:
                        bullish_score += score
                    else:
                        bearish_score += abs(score)

                # Add trend bias
                if trend_bias > 0:
                    bullish_score += trend_bias
                else:
                    bearish_score += abs(trend_bias)

                # Add breakout analysis (only for latest candle)
                if i == len(df) - 1:
                    bullish_score += breakout_analysis['breakout_bullish']
                    bearish_score += breakout_analysis['breakout_bearish']

                # Volume confirmation
                if conditions.get('volume_score', 0) > 0:
                    if bullish_score > bearish_score:
                        bullish_score += 1
                    else:
                        bearish_score += 1

                # Generate signal based on strength and bias
                net_score = bullish_score - bearish_score

                if bullish_score >= self.min_signal_strength and net_score > 1:
                    signal = SignalType.BUY.value
                    confidence = min(60 + bullish_score * 5, 95)

                    reason_parts = []
                    if conditions.get('rsi_score', 0) > 0:
                        reason_parts.append(f"RSI oversold")
                    if conditions.get('macd_score', 0) > 0:
                        reason_parts.append("MACD bullish")
                    if trend_bias > 0:
                        reason_parts.append(f"Trend: {trend_analysis.current_trend.value}")
                    if breakout_analysis['breakout_bullish'] > 0:
                        reason_parts.append("Breakout")

                    reason = f"BUY: {', '.join(reason_parts)} (Confidence: {confidence:.0f}%)"

                elif bearish_score >= self.min_signal_strength and net_score < -1:
                    signal = SignalType.SELL.value
                    confidence = min(60 + bearish_score * 5, 95)

                    reason_parts = []
                    if conditions.get('rsi_score', 0) < 0:
                        reason_parts.append("RSI overbought")
                    if conditions.get('macd_score', 0) < 0:
                        reason_parts.append("MACD bearish")
                    if trend_bias < 0:
                        reason_parts.append(f"Trend: {trend_analysis.current_trend.value}")
                    if breakout_analysis['breakout_bearish'] > 0:
                        reason_parts.append("Breakdown")

                    reason = f"SELL: {', '.join(reason_parts)} (Confidence: {confidence:.0f}%)"

                signals.append(signal)
                reasons.append(reason)

            # Add signals to dataframe
            df = df.copy()
            df["Signal"] = signals
            df["SignalReason"] = reasons

            # Log signal summary
            buy_signals = sum(1 for s in signals if s == SignalType.BUY.value)
            sell_signals = sum(1 for s in signals if s == SignalType.SELL.value)

            if buy_signals > 0 or sell_signals > 0:
                logger.info(f"{symbol}: Generated {buy_signals} BUY and {sell_signals} SELL signals")

            return df

        except Exception as e:
            logger.error(f"Error generating signals for {symbol}: {e}")
            return df

    def get_latest_signal(self, df: pd.DataFrame) -> Optional[TradingSignal]:
        """Get the latest trading signal from the dataframe."""
        if df.empty or 'Signal' not in df.columns:
            return None

        try:
            # Look for the most recent non-null signal
            recent_signals = df[df['Signal'].notna()].tail(5)

            if recent_signals.empty:
                return None

            latest = recent_signals.iloc[-1]

            if pd.isna(latest['Signal']):
                return None

            return TradingSignal(
                symbol="",  # Will be set by caller
                signal=SignalType(latest['Signal']),
                price=latest['close'],
                confidence=self._extract_confidence_from_reason(latest.get('SignalReason', '')),
                reason=latest.get('SignalReason', ''),
                timestamp=latest['timestamp']
            )

        except Exception as e:
            logger.error(f"Error getting latest signal: {e}")
            return None

    def _extract_confidence_from_reason(self, reason: str) -> float:
        """Extract confidence percentage from reason string."""
        try:
            if 'Confidence:' in reason:
                confidence_part = reason.split('Confidence:')[1].split('%')[0].strip()
                return float(confidence_part)
        except (IndexError, ValueError):
            pass
        return 50.0  # Default confidence