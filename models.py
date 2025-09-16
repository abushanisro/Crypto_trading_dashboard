"""Data models for the crypto trading dashboard."""
from datetime import datetime
from typing import List, Optional, Dict, Any
from dataclasses import dataclass
from enum import Enum
import pandas as pd


class TrendDirection(Enum):
    """Trend direction enumeration."""
    STRONG_UPTREND = "STRONG_UPTREND"
    UPTREND = "UPTREND"
    SHORT_TERM_UPTREND = "SHORT_TERM_UPTREND"
    PULLBACK_IN_UPTREND = "PULLBACK_IN_UPTREND"
    SIDEWAYS = "SIDEWAYS"
    STRONG_DOWNTREND = "STRONG_DOWNTREND"
    DOWNTREND = "DOWNTREND"
    INSUFFICIENT_DATA = "INSUFFICIENT_DATA"


class SignalType(Enum):
    """Trading signal types."""
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


class BreakoutType(Enum):
    """Breakout types."""
    RESISTANCE_BREAKOUT = "resistance_breakout"
    SUPPORT_BREAKDOWN = "support_breakdown"


@dataclass
class PivotPoint:
    """Represents a pivot point in price data."""
    timestamp: datetime
    price: float
    index: int


@dataclass
class TrendLine:
    """Represents a trend line with its properties."""
    slope: float
    intercept: float
    r_squared: float
    points: List[PivotPoint]
    start_idx: int
    end_idx: int
    touches: int = 0
    strength: float = 0.0
    type: str = ""


@dataclass
class BreakoutSignal:
    """Represents a breakout signal."""
    type: BreakoutType
    direction: str
    strength: float
    price: float
    signal: SignalType
    confidence: float
    timestamp: datetime = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


@dataclass
class TradingSignal:
    """Represents a trading signal with context."""
    symbol: str
    signal: SignalType
    price: float
    confidence: float
    reason: str
    timestamp: datetime
    trend_context: Optional[TrendDirection] = None


@dataclass
class MarketData:
    """Real-time market data structure."""
    symbol: str
    price: float
    timestamp: datetime
    candles_count: int
    price_change: float
    ml_prediction: Optional[float] = None
    ml_change: Optional[float] = None


@dataclass
class TechnicalIndicators:
    """Technical indicators for a symbol."""
    rsi: float
    macd: float
    macd_signal: float
    macd_histogram: float
    ma_20: float
    ma_50: float
    bb_upper: float
    bb_lower: float
    bb_middle: float
    volume_ma: float


@dataclass
class PriceData:
    """OHLCV price data structure."""
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float


@dataclass
class TrendAnalysis:
    """Complete trend analysis for a symbol."""
    symbol: str
    current_trend: TrendDirection
    support_lines: List[TrendLine]
    resistance_lines: List[TrendLine]
    breakouts: List[BreakoutSignal]
    timestamp: datetime
    strength_score: float = 0.0


class MLStatus(Enum):
    """ML model status."""
    NOT_TRAINED = "Not Trained"
    TRAINING = "Training..."
    TRAINED = "Trained & Ready"
    ERROR = "Error"


@dataclass
class MLPrediction:
    """ML prediction result."""
    symbol: str
    predicted_price: float
    confidence: float
    movement_direction: str
    timestamp: datetime
    model_status: MLStatus


@dataclass
class DashboardState:
    """Complete dashboard state."""
    market_data: Dict[str, MarketData]
    trend_analyses: Dict[str, TrendAnalysis]
    trading_signals: Dict[str, List[TradingSignal]]
    ml_predictions: Dict[str, MLPrediction]
    ml_status: Dict[str, MLStatus]
    last_update: datetime