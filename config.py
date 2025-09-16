"""Configuration management for the crypto trading dashboard."""
import os
from typing import List, Optional
from dotenv import load_dotenv

load_dotenv()

try:
    from pydantic_settings import BaseSettings
except ImportError:
    from pydantic import BaseSettings


class Settings(BaseSettings):
    """Application settings with validation."""

    # Application settings
    APP_NAME: str = "Crypto Trading Dashboard"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False
    HOST: str = "0.0.0.0"
    PORT: int = int(os.getenv('PORT', 5050))

    # Trading settings
    SYMBOLS: List[str] = [
        "BTC/USDT", "ETH/USDT", "BNB/USDT", "ADA/USDT", "SOL/USDT"
    ]
    UPDATE_INTERVAL: int = 3000  # milliseconds
    MAX_JOURNAL_LENGTH: int = 10

    # Data settings
    TIMEFRAME: str = "1m"
    OHLCV_LIMIT: int = 500
    SIMULATED_PERIODS: int = 400
    SIMULATED_INTERVAL_SECONDS: int = 15

    # Trend analysis settings
    TREND_MIN_TOUCHES: int = 2
    TREND_TOUCH_THRESHOLD: float = 0.002  # 0.2%
    TREND_LOOKBACK_PERIODS: int = 100
    TREND_PIVOT_WINDOW: int = 5

    # ML settings
    ML_LOOKBACK: int = 30
    ML_MIN_TRAINING_DATA: int = 100
    ML_N_ESTIMATORS: int = 100
    ML_MAX_DEPTH: int = 10
    ML_RANDOM_STATE: int = 42

    # Exchange settings
    EXCHANGE_NAME: str = "binance"
    EXCHANGE_RATE_LIMIT: int = 1200

    # Redis settings (for caching)
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379
    REDIS_DB: int = 0
    REDIS_PASSWORD: Optional[str] = None

    # Logging settings
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    class Config:
        env_file = ".env"
        case_sensitive = True
        extra = "ignore"


# Global settings instance
settings = Settings()