![Crypto](https://github.com/abushanisro/Crypto_trading_dashboard/blob/main/cryptotrade.png?raw=true)
# Crypto Trading Dashboard

A production-ready cryptocurrency trading dashboard with advanced trend line analysis, ML predictions, and real-time breakout detection.

## Features

### Core Functionality
- **Real-time Data**: Live cryptocurrency price feeds via CCXT or simulated high-frequency data
- **Advanced Trend Analysis**: Professional pivot-point based trend line detection
- **Support/Resistance Detection**: Automated identification of key price levels
- **Breakout Alerts**: Real-time detection and notification of trend line breaks
- **ML Predictions**: Machine learning price forecasting using Random Forest models
- **Trading Signals**: AI-enhanced buy/sell signal generation
- **Technical Indicators**: RSI, MACD, Bollinger Bands, Moving Averages
- **Live Dashboard**: Interactive real-time web interface

### Technical Highlights
- **Production Architecture**: Modular, scalable, and maintainable code structure
- **Type Safety**: Full type hints with Pydantic models
- **Error Handling**: Comprehensive logging and graceful error recovery
- **Testing**: Unit tests for critical components
- **Configuration**: Environment-based configuration management
- **Performance**: Optimized for high-frequency data processing

## 📋 Requirements

### System Requirements
- Python 3.8+
- 4GB+ RAM recommended
- Modern web browser

### Dependencies
```
dash==2.14.1
plotly==5.17.0
pandas==2.1.1
numpy==1.24.3
scipy==1.11.3
scikit-learn==1.3.0  # For ML predictions
ccxt==4.1.4          # For real crypto data
pydantic==2.4.2      # Data validation
python-dotenv==1.0.0 # Environment management
```

## 🛠️ Installation

1. **Clone or extract the project**:
   ```bash
   cd /path/to/trendline
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Optional: Create environment file**:
   ```bash
   cp .env.example .env
   # Edit .env with your preferences
   ```

## 🚀 Quick Start

### Basic Usage
```bash
python main.py
```

The dashboard will start at `http://localhost:5050`

### Advanced Configuration
Create a `.env` file to customize settings:

```bash
# Application Settings
DEBUG=False
HOST=0.0.0.0
PORT=5050

# Trading Settings
SYMBOLS=["BTC/USDT", "ETH/USDT", "BNB/USDT"]
UPDATE_INTERVAL=3000
TIMEFRAME=1m

# ML Settings
ML_LOOKBACK=30
ML_MIN_TRAINING_DATA=100

# Logging
LOG_LEVEL=INFO
```

## 📊 Architecture

### Project Structure
```
trendline/
├── analysis/           # Technical analysis modules
│   ├── technical_indicators.py
│   └── trend_analyzer.py
├── dashboard/          # Dashboard components
│   ├── core.py
│   └── chart_utils.py
├── data/              # Data providers
│   └── data_provider.py
├── ml/                # Machine learning
│   └── predictor.py
├── trading/           # Trading logic
│   └── signal_generator.py
├── tests/             # Unit tests
├── utils/             # Utilities
│   └── logger.py
├── config.py          # Configuration
├── models.py          # Data models
├── main.py           # Application entry point
└── requirements.txt
```

### Key Components

#### 1. Data Layer (`data/`)
- **DataProvider**: Abstract interface for data sources
- **CCXTDataProvider**: Real cryptocurrency data via CCXT
- **SimulatedDataProvider**: High-frequency simulated data

#### 2. Analysis Layer (`analysis/`)
- **TechnicalIndicators**: RSI, MACD, Bollinger Bands, Moving Averages
- **TrendLineAnalyzer**: Advanced pivot-point trend line detection

#### 3. ML Layer (`ml/`)
- **CryptoMLPredictor**: Random Forest price prediction
- **MLPredictionService**: ML model management

#### 4. Trading Layer (`trading/`)
- **SignalGenerator**: AI-enhanced trading signal generation

#### 5. Dashboard Layer (`dashboard/`)
- **DashboardCore**: Main business logic
- **ChartBuilder**: Professional chart generation

## 🔧 Configuration Options

### Core Settings
- `SYMBOLS`: List of trading pairs to monitor
- `UPDATE_INTERVAL`: Refresh rate in milliseconds
- `TIMEFRAME`: Data timeframe (1m, 5m, 1h, etc.)

### Trend Analysis
- `TREND_MIN_TOUCHES`: Minimum touches for trend line validation
- `TREND_TOUCH_THRESHOLD`: Price distance threshold for trend line touches
- `TREND_PIVOT_WINDOW`: Window size for pivot point detection

### ML Settings
- `ML_LOOKBACK`: Historical periods for ML training
- `ML_MIN_TRAINING_DATA`: Minimum data points required for training
- `ML_N_ESTIMATORS`: Random Forest estimator count

## 🧪 Testing

Run the test suite:

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=. --cov-report=html

# Run specific test file
pytest tests/test_technical_indicators.py
```

### Test Coverage
- Technical indicators calculation
- Trend line detection algorithms
- Signal generation logic
- Data provider functionality

## 🔍 Usage Examples

### Custom Data Provider
```python
from data.data_provider import DataProvider
import pandas as pd

class CustomDataProvider(DataProvider):
    def fetch_ohlcv(self, symbol: str, timeframe: str, limit: int) -> pd.DataFrame:
        # Your custom data fetching logic
        return df

    def is_available(self) -> bool:
        return True
```

### Manual Trend Analysis
```python
from analysis.trend_analyzer import TrendLineAnalyzer
from analysis.technical_indicators import TechnicalIndicators

# Load your data
df = pd.read_csv('your_data.csv')

# Calculate indicators
df = TechnicalIndicators.calculate_all_indicators(df)

# Analyze trends
analyzer = TrendLineAnalyzer()
trend_analysis = analyzer.analyze_trends("BTC/USDT", df)

print(f"Current trend: {trend_analysis.current_trend}")
print(f"Support lines: {len(trend_analysis.support_lines)}")
print(f"Resistance lines: {len(trend_analysis.resistance_lines)}")
```

### Custom Signal Generation
```python
from trading.signal_generator import SignalGenerator

generator = SignalGenerator()
df_with_signals = generator.generate_enhanced_signals(df, trend_analysis, "BTC/USDT")

# Get latest signal
latest_signal = generator.get_latest_signal(df_with_signals)
if latest_signal:
    print(f"Signal: {latest_signal.signal.value} at ${latest_signal.price:.2f}")
```

## 📈 Performance & Scalability

### Optimization Features
- **Efficient Algorithms**: Optimized trend line detection using scipy
- **Caching**: Trend analysis results cached to avoid recalculation
- **Threading**: Background data updates don't block UI
- **Memory Management**: Circular buffers for live data streams

### Scaling Recommendations
- **Multiple Symbols**: Tested with 5-10 symbols simultaneously
- **Higher Frequency**: Can handle 1-second updates with proper hardware
- **Database Integration**: Add PostgreSQL/Redis for persistence
- **Load Balancing**: Deploy multiple instances behind nginx

## 🔐 Security Considerations

- **No API Keys**: Default configuration doesn't require exchange credentials
- **Input Validation**: All data validated using Pydantic models
- **Error Isolation**: Exceptions contained per symbol/component
- **Logging**: Comprehensive audit trail without sensitive data

## 🚨 Important Notes

### Data Sources
- **Real Data**: Requires internet connection for CCXT
- **Simulated Data**: High-quality simulated data when CCXT unavailable
- **Rate Limits**: CCXT providers have rate limits - monitor logs

### ML Predictions
- **Training Required**: Models need 100+ data points for training
- **Performance**: Predictions improve with more historical data
- **Validation**: Always validate ML predictions against market conditions

### Production Deployment
- **Environment**: Set `DEBUG=False` in production
- **Monitoring**: Monitor logs for errors and performance
- **Resources**: Ensure adequate CPU/memory for chosen symbol count

## 🐛 Troubleshooting

### Common Issues

1. **CCXT Import Error**:
   ```bash
   pip install ccxt
   ```

2. **Scipy Missing**:
   ```bash
   pip install scipy
   ```

3. **Memory Usage High**:
   - Reduce number of symbols
   - Decrease update frequency
   - Lower OHLCV_LIMIT in config

4. **Charts Not Loading**:
   - Check browser console for errors
   - Verify all dependencies installed
   - Check logs for data fetching issues

### Debug Mode
Enable debug logging:
```bash
LOG_LEVEL=DEBUG python main.py
```

## 📝 License

This project is provided as-is for educational and research purposes.

## 🤝 Contributing

1. Follow the existing code structure
2. Add tests for new features
3. Update documentation
4. Use type hints
5. Follow PEP8 standards

## 📞 Support

For issues and questions:
1. Check the troubleshooting section
2. Review logs with `LOG_LEVEL=DEBUG`
3. Test with minimal configuration
4. Verify all dependencies are installed

---
