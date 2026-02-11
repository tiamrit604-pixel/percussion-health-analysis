@echo off
echo ==================================
echo Percussion Health Analysis Setup
echo ==================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python is not installed. Please install Python 3.8 or higher.
    pause
    exit /b 1
)

echo ✅ Python found
echo.

REM Create virtual environment
echo 📦 Creating virtual environment...
python -m venv venv

REM Activate virtual environment
call venv\Scripts\activate.bat

echo ✅ Virtual environment activated
echo.

REM Install dependencies
echo 📥 Installing dependencies...
python -m pip install --upgrade pip
pip install -r requirements.txt

echo.
echo ✅ Installation complete!
echo.
echo ==================================
echo Starting Percussion Analysis App
echo ==================================
echo.
echo The app will open in your browser automatically.
echo If it doesn't, navigate to: http://localhost:8501
echo.
echo Press Ctrl+C to stop the server
echo.

REM Run the Streamlit app
streamlit run percussion_browser_recording.py

pause
