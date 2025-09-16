#!/bin/bash

# Production-Ready Crypto Trading Dashboard Startup Script

echo "🚀 Starting Crypto Trading Dashboard..."
echo "======================================"

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "❌ Virtual environment not found. Please create one first:"
    echo "python3 -m venv venv"
    echo "source venv/bin/activate"
    echo "pip install -r requirements.txt"
    exit 1
fi

# Activate virtual environment
source venv/bin/activate

# Check if dependencies are installed
if ! python -c "import dash" 2>/dev/null; then
    echo "📦 Installing dependencies..."
    pip install -r requirements.txt
fi

# Check for optional .env file
if [ -f ".env" ]; then
    echo "✅ Using configuration from .env file"
else
    echo "ℹ️  Using default configuration (create .env file to customize)"
fi

# Start the application
PORT=${PORT:-5050}
echo "🌐 Dashboard will be available at: http://localhost:$PORT"
echo "🔄 Press Ctrl+C to stop"
echo "======================================"

PORT=$PORT python app.py