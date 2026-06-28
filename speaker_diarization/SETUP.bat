@echo off
title Speaker Diarization - Setup
color 0E
echo.
echo  ================================================
echo    Speaker Diarization - First-Time Setup
echo  ================================================
echo.
echo  This will install everything needed.
echo  Takes about 2-3 minutes. Do this only once.
echo.
pause

:: Check Python
echo.
echo  [1/3] Checking Python...
python --version >nul 2>&1
if errorlevel 1 (
    echo.
    echo  Python not found! Please install it from:
    echo  https://www.python.org/downloads/
    echo  Make sure to check "Add Python to PATH" during install.
    echo.
    pause
    exit /b 1
)
python --version
echo  Python OK.

:: Install ffmpeg
echo.
echo  [2/3] Installing ffmpeg...
winget install --id Gyan.FFmpeg -e --silent
echo  ffmpeg OK.

:: Install Python packages
echo.
echo  [3/3] Installing Python packages...
pip install resemblyzer librosa scikit-learn soundfile numpy webrtcvad-wheels pyaudio
echo.

echo  ================================================
echo    Setup complete! You can now use:
echo    "RUN MY VIDEO.bat" - drag your video onto it
echo  ================================================
echo.
pause
