#!/bin/bash

echo "=================================="
echo "Percussion Health Analysis Setup"
echo "=================================="
echo ""

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3.8 or higher."
    exit 1
fi

echo "✅ Python found: $(python3 --version)"
echo ""

# Create virtual environment (optional but recommended)
echo "📦 Creating virtual environment..."
python3 -m venv venv

# Activate virtual environment
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
    # Windows
    source venv/Scripts/activate
else
    # Linux/Mac
    source venv/bin/activate
fi

echo "✅ Virtual environment activated"
echo ""

# Install dependencies
echo "📥 Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

echo ""
echo "✅ Installation complete!"
echo ""
echo "=================================="
echo "Starting Percussion Analysis App"
echo "=================================="
echo ""
echo "The app will open in your browser automatically."
echo "If it doesn't, navigate to: http://localhost:8501"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

# Run the Streamlit app
streamlit run percussion_browser_recording.py
