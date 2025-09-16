"""Core dashboard functionality."""
from typing import Dict, List, Optional
import pandas as pd
from datetime import datetime
import queue
import threading
import time

from models import (
    DashboardState, MarketData, TradingSignal, MLPrediction,
    TrendAnalysis, MLStatus, TrendDirection
)
from data.providers import DataProviderFactory
from analysis.technical_indicators import TechnicalIndicators
from analysis.trend_analyzer import ProfessionalTrendAnalyzer
from trading.signal_generator import SignalGenerator
from ml.predictor import MLPredictionService
from utils.logger import get_logger
from config import settings

logger = get_logger("dashboard_core")


class DashboardCore:
    """Core dashboard business logic."""

    def __init__(self):
        # Initialize components
        self.data_provider = DataProviderFactory.create_provider()
        self.technical_indicators = TechnicalIndicators()
        self.trend_analyzer = ProfessionalTrendAnalyzer()
        self.signal_generator = SignalGenerator()
        self.ml_service = MLPredictionService()

        # State management
        self.current_state = DashboardState(
            market_data={},
            trend_analyses={},
            trading_signals={},
            ml_predictions={},
            ml_status={symbol: MLStatus.NOT_TRAINED for symbol in settings.SYMBOLS},
            last_update=datetime.now()
        )

        # Thread-safe queues for live data
        self.live_output_queue = queue.Queue(maxsize=100)
        self.commentary_queue = queue.Queue(maxsize=50)

        # Journal storage
        self.trading_journal: Dict[str, List[str]] = {
            symbol: [] for symbol in settings.SYMBOLS
        }

        logger.info("Dashboard core initialized successfully")

    def fetch_and_process_data(self, symbol: str) -> Optional[pd.DataFrame]:
        """Fetch and process data for a single symbol."""
        try:
            # Fetch raw data
            df = self.data_provider.fetch_ohlcv(
                symbol=symbol,
                timeframe=settings.TIMEFRAME,
                limit=settings.OHLCV_LIMIT
            )

            if df.empty:
                logger.warning(f"No data received for {symbol}")
                return None

            # Calculate technical indicators
            df = self.technical_indicators.calculate_all_indicators(df)

            logger.debug(f"Processed {len(df)} data points for {symbol}")
            return df

        except Exception as e:
            logger.error(f"Error processing data for {symbol}: {e}")
            return None

    def update_market_data(self, symbol: str, df: pd.DataFrame) -> None:
        """Update market data for a symbol."""
        try:
            if df.empty:
                return

            latest = df.iloc[-1]
            prev = df.iloc[-2] if len(df) > 1 else latest

            # Calculate price change
            price_change = latest['close'] - prev['close']

            # Update market data
            self.current_state.market_data[symbol] = MarketData(
                symbol=symbol,
                price=latest['close'],
                timestamp=latest['timestamp'],
                candles_count=len(df),
                price_change=price_change
            )

        except Exception as e:
            logger.error(f"Error updating market data for {symbol}: {e}")

    def analyze_trends(self, symbol: str, df: pd.DataFrame) -> TrendAnalysis:
        """Analyze trends for a symbol."""
        try:
            trend_analysis = self.trend_analyzer.analyze_professional_trends(symbol, df)
            self.current_state.trend_analyses[symbol] = trend_analysis

            # Add breakout alerts to live output
            if trend_analysis.breakouts:
                for breakout in trend_analysis.breakouts:
                    alert = (
                        f"🚨 {symbol} {breakout.type.value.upper()}: "
                        f"{breakout.signal.value} at ${breakout.price:.2f} "
                        f"(Confidence: {breakout.confidence:.0f}%)"
                    )
                    self._add_to_live_output(alert)

            return trend_analysis

        except Exception as e:
            logger.error(f"Error analyzing trends for {symbol}: {e}")
            return TrendAnalysis(
                symbol=symbol,
                current_trend=TrendDirection.INSUFFICIENT_DATA,
                support_lines=[],
                resistance_lines=[],
                breakouts=[],
                timestamp=datetime.now(),
                strength_score=0.0
            )

    def generate_trading_signals(
        self,
        symbol: str,
        df: pd.DataFrame,
        trend_analysis: TrendAnalysis
    ) -> List[TradingSignal]:
        """Generate trading signals for a symbol."""
        try:
            # Generate signals using trend analysis
            df_with_signals = self.signal_generator.generate_enhanced_signals(
                df, trend_analysis, symbol
            )

            # Extract latest signal
            latest_signal = self.signal_generator.get_latest_signal(df_with_signals)

            signals = []
            if latest_signal:
                latest_signal.symbol = symbol
                signals.append(latest_signal)

                # Update journal
                self._update_trading_journal(symbol, latest_signal)

                # Add to live output
                signal_text = (
                    f"{symbol}: {latest_signal.signal.value} at ${latest_signal.price:.2f} - "
                    f"{latest_signal.reason}"
                )
                self._add_to_live_output(signal_text)

            self.current_state.trading_signals[symbol] = signals
            return signals

        except Exception as e:
            logger.error(f"Error generating signals for {symbol}: {e}")
            return []

    def make_ml_prediction(self, symbol: str, df: pd.DataFrame) -> Optional[MLPrediction]:
        """Make ML prediction for a symbol."""
        try:
            prediction = self.ml_service.make_prediction(symbol, df)

            if prediction:
                self.current_state.ml_predictions[symbol] = prediction

                # Update market data with ML prediction
                if symbol in self.current_state.market_data:
                    market_data = self.current_state.market_data[symbol]
                    market_data.ml_prediction = prediction.predicted_price

                    current_price = df['close'].iloc[-1]
                    market_data.ml_change = (
                        (prediction.predicted_price - current_price) / current_price * 100
                    )

            # Update ML status
            self.current_state.ml_status[symbol] = self.ml_service.get_model_status(symbol)

            return prediction

        except Exception as e:
            logger.error(f"Error making ML prediction for {symbol}: {e}")
            return None

    def update_dashboard_data(self) -> DashboardState:
        """Update all dashboard data for all symbols."""
        try:
            logger.debug("Starting dashboard data update")

            for symbol in settings.SYMBOLS:
                # Fetch and process data
                df = self.fetch_and_process_data(symbol)
                if df is None:
                    continue

                # Update market data
                self.update_market_data(symbol, df)

                # Analyze trends
                trend_analysis = self.analyze_trends(symbol, df)

                # Generate trading signals
                self.generate_trading_signals(symbol, df, trend_analysis)

                # Make ML predictions
                self.make_ml_prediction(symbol, df)

            # Update timestamp
            self.current_state.last_update = datetime.now()

            logger.debug("Dashboard data update completed")
            return self.current_state

        except Exception as e:
            logger.error(f"Error updating dashboard data: {e}")
            return self.current_state

    def _add_to_live_output(self, message: str) -> None:
        """Add message to live output queue."""
        try:
            timestamp = datetime.now().strftime('%H:%M:%S')
            formatted_message = f"[{timestamp}] {message}"

            if not self.live_output_queue.full():
                self.live_output_queue.put(formatted_message)
            else:
                # Remove oldest message to make space
                try:
                    self.live_output_queue.get_nowait()
                    self.live_output_queue.put(formatted_message)
                except queue.Empty:
                    pass

        except Exception as e:
            logger.error(f"Error adding to live output: {e}")

    def _update_trading_journal(self, symbol: str, signal: TradingSignal) -> None:
        """Update trading journal for a symbol."""
        try:
            timestamp_str = signal.timestamp.strftime('%H:%M:%S')
            entry = (
                f"[{timestamp_str}] {symbol}: {signal.signal.value} - "
                f"{signal.reason} (${signal.price:.2f})"
            )

            if symbol not in self.trading_journal:
                self.trading_journal[symbol] = []

            # Avoid duplicates
            if entry not in self.trading_journal[symbol]:
                self.trading_journal[symbol].append(entry)

                # Keep only last N entries
                if len(self.trading_journal[symbol]) > settings.MAX_JOURNAL_LENGTH:
                    self.trading_journal[symbol] = (
                        self.trading_journal[symbol][-settings.MAX_JOURNAL_LENGTH:]
                    )

        except Exception as e:
            logger.error(f"Error updating trading journal for {symbol}: {e}")

    def get_live_output(self, max_items: int = 25) -> List[str]:
        """Get recent live output messages."""
        messages = []
        temp_messages = []

        # Extract all messages from queue
        try:
            while not self.live_output_queue.empty():
                temp_messages.append(self.live_output_queue.get_nowait())
        except queue.Empty:
            pass

        # Return most recent messages and put back the rest
        messages = temp_messages[-max_items:] if temp_messages else []

        # Put messages back in queue
        for msg in temp_messages[-50:]:  # Keep last 50 in queue
            try:
                self.live_output_queue.put_nowait(msg)
            except queue.Full:
                break

        return messages

    def get_trading_journal(self) -> Dict[str, List[str]]:
        """Get current trading journal."""
        return self.trading_journal.copy()

    def get_dashboard_state(self) -> DashboardState:
        """Get current dashboard state."""
        return self.current_state