@echo off
title Speaker Diarization - My Video
color 0B
echo.
echo  ================================================
echo    Speaker Diarization - Your Video
echo  ================================================
echo.

if "%~1"=="" (
    echo  No video was dropped onto this file.
    echo  Please paste the full path to your video below.
    echo  Supported formats: .mp4  .mkv  .mov  .avi  .webm
    echo.
    python "%~dp0run_my_video.py"
) else (
    echo  Video: %~1
    echo.
    python "%~dp0run_my_video.py" "%~1"
)

if errorlevel 1 (
    echo.
    echo  Something went wrong. See the error above.
)
echo.
pause
