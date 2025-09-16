"""Machine learning prediction module."""
from typing import Optional, Dict, Any
import pandas as pd
import numpy as np
from datetime import datetime

from models import MLPrediction, MLStatus, SignalType
from utils.logger import get_logger
from config import settings

logger = get_logger("ml_predictor")


class CryptoMLPredictor:
    """Machine learning predictor for cryptocurrency prices."""

    def __init__(self):
        self.model = None
        self.scaler = None
        self.lookback = settings.ML_LOOKBACK
        self.is_trained = False
        self._initialize_dependencies()

    def _initialize_dependencies(self) -> None:
        """Initialize ML dependencies if available."""
        try:
            from sklearn.preprocessing import StandardScaler
            from sklearn.ensemble import RandomForestRegressor

            self.scaler = StandardScaler()
            self.model_class = RandomForestRegressor
            self.dependencies_available = True

        except ImportError:
            logger.warning("Scikit-learn not available - ML predictions disabled")
            self.dependencies_available = False
            self.scaler = None
            self.model_class = None

    def prepare_features(self, df: pd.DataFrame) -> Optional[np.ndarray]:
        """Create advanced features for ML prediction."""
        if len(df) < 50 or not self.dependencies_available:
            return None

        try:
            features = []
            latest_idx = len(df) - 1

            # Price features
            close = df['close'].iloc[latest_idx]
            features.extend([
                close,
                df['high'].iloc[latest_idx] - df['low'].iloc[latest_idx],  # Range
                df['close'].pct_change().iloc[latest_idx],  # Return
            ])

            # Moving averages
            if 'MA_20' in df.columns:
                features.append(df['MA_20'].iloc[latest_idx])
            if 'MA_50' in df.columns:
                features.append(df['MA_50'].iloc[latest_idx])

            # Technical indicators
            if 'RSI' in df.columns:
                features.append(df['RSI'].iloc[latest_idx])

            if 'MACD' in df.columns and 'MACD_signal' in df.columns:
                features.extend([
                    df['MACD'].iloc[latest_idx],
                    df['MACD_signal'].iloc[latest_idx]
                ])

            # Volume features
            features.extend([
                df['volume'].iloc[latest_idx],
                df['Volume_MA'].iloc[latest_idx] if 'Volume_MA' in df.columns else df['volume'].iloc[latest_idx]
            ])

            # Volatility features
            volatility = df['close'].rolling(20).std().iloc[latest_idx]
            features.append(volatility if not pd.isna(volatility) else 0)

            # Price momentum features
            if len(df) >= 10:
                short_momentum = (df['close'].iloc[latest_idx] - df['close'].iloc[latest_idx-5]) / df['close'].iloc[latest_idx-5]
                medium_momentum = (df['close'].iloc[latest_idx] - df['close'].iloc[latest_idx-10]) / df['close'].iloc[latest_idx-10]
                features.extend([short_momentum, medium_momentum])

            return np.array(features).reshape(1, -1)

        except Exception as e:
            logger.error(f"Error preparing features: {e}")
            return None

    def build_model(self):
        """Build Random Forest model for price prediction."""
        if not self.dependencies_available:
            return None

        return self.model_class(
            n_estimators=settings.ML_N_ESTIMATORS,
            max_depth=settings.ML_MAX_DEPTH,
            random_state=settings.ML_RANDOM_STATE,
            n_jobs=-1
        )

    def train(self, df: pd.DataFrame) -> bool:
        """Train ML model on historical data."""
        if not self.dependencies_available or len(df) < settings.ML_MIN_TRAINING_DATA:
            return False

        try:
            X, y = [], []

            # Create training data with sliding window
            min_length = max(50, self.lookback)
            for i in range(min_length, len(df) - 1):
                sub_df = df.iloc[i-min_length:i+1].copy()
                features = self.prepare_features(sub_df)

                if features is not None:
                    target = df.iloc[i+1]['close']
                    X.append(features.flatten())
                    y.append(target)

            if len(X) < 20:
                logger.warning("Insufficient training data")
                return False

            X = np.array(X)
            y = np.array(y)

            # Scale features
            X = self.scaler.fit_transform(X)

            # Train model
            self.model = self.build_model()
            if self.model is None:
                return False

            self.model.fit(X, y)
            self.is_trained = True

            # Calculate training accuracy
            train_score = self.model.score(X, y)
            logger.info(f"Model trained successfully with R² score: {train_score:.3f}")

            return True

        except Exception as e:
            logger.error(f"Training error: {e}")
            self.is_trained = False
            return False

    def predict(self, df: pd.DataFrame) -> Optional[float]:
        """Make price prediction using trained ML model."""
        if not self.is_trained or self.model is None or not self.dependencies_available:
            return None

        try:
            features = self.prepare_features(df)
            if features is None:
                return None

            # Scale features
            scaled_features = self.scaler.transform(features)

            # Make prediction
            prediction = self.model.predict(scaled_features)[0]

            # Sanity check - prediction shouldn't be too far from current price
            current_price = df['close'].iloc[-1]
            if abs(prediction - current_price) / current_price > 0.5:  # 50% change limit
                logger.warning(f"Prediction {prediction} too far from current price {current_price}")
                return None

            return float(prediction)

        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return None

    def get_status(self) -> MLStatus:
        """Get current ML model status."""
        if not self.dependencies_available:
            return MLStatus.ERROR
        elif not self.is_trained:
            return MLStatus.NOT_TRAINED
        else:
            return MLStatus.TRAINED


class MLPredictionService:
    """Service for managing ML predictions across multiple symbols."""

    def __init__(self):
        self.predictors: Dict[str, CryptoMLPredictor] = {}
        self.ml_available = self._check_ml_availability()

    def _check_ml_availability(self) -> bool:
        """Check if ML dependencies are available."""
        try:
            import sklearn
            return True
        except ImportError:
            logger.warning("ML dependencies not available")
            return False

    def get_predictor(self, symbol: str) -> CryptoMLPredictor:
        """Get or create predictor for symbol."""
        if symbol not in self.predictors:
            self.predictors[symbol] = CryptoMLPredictor()
        return self.predictors[symbol]

    def train_model(self, symbol: str, df: pd.DataFrame) -> bool:
        """Train model for a specific symbol."""
        if not self.ml_available:
            return False

        predictor = self.get_predictor(symbol)
        success = predictor.train(df)

        if success:
            logger.info(f"Successfully trained ML model for {symbol}")
        else:
            logger.warning(f"Failed to train ML model for {symbol}")

        return success

    def make_prediction(self, symbol: str, df: pd.DataFrame) -> Optional[MLPrediction]:
        """Make ML prediction for a symbol."""
        if not self.ml_available:
            return None

        try:
            predictor = self.get_predictor(symbol)

            # Auto-train if not trained and we have enough data
            if not predictor.is_trained and len(df) >= settings.ML_MIN_TRAINING_DATA:
                logger.info(f"Auto-training model for {symbol}")
                predictor.train(df)

            # Make prediction
            predicted_price = predictor.predict(df)
            if predicted_price is None:
                return None

            current_price = df['close'].iloc[-1]
            change_pct = (predicted_price - current_price) / current_price * 100

            # Determine movement direction and confidence
            if change_pct > 0.5:
                movement = "STRONG_UP"
                confidence = min(85 + abs(change_pct) * 2, 95)
            elif change_pct > 0.1:
                movement = "UP"
                confidence = min(70 + abs(change_pct) * 5, 85)
            elif change_pct < -0.5:
                movement = "STRONG_DOWN"
                confidence = min(85 + abs(change_pct) * 2, 95)
            elif change_pct < -0.1:
                movement = "DOWN"
                confidence = min(70 + abs(change_pct) * 5, 85)
            else:
                movement = "CONSOLIDATION"
                confidence = 60 + abs(change_pct) * 10

            return MLPrediction(
                symbol=symbol,
                predicted_price=predicted_price,
                confidence=confidence,
                movement_direction=movement,
                timestamp=datetime.now(),
                model_status=predictor.get_status()
            )

        except Exception as e:
            logger.error(f"Error making ML prediction for {symbol}: {e}")
            return None

    def get_model_status(self, symbol: str) -> MLStatus:
        """Get model training status for a symbol."""
        if not self.ml_available:
            return MLStatus.ERROR

        predictor = self.get_predictor(symbol)
        return predictor.get_status()