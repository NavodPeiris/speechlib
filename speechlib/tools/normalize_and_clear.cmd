@echo off
REM Normalize and clear (enhance) audio files
REM Uses same output structure as speechlib pipeline
REM Output: .<stem>/processed.wav alongside source file

setlocal enabledelayedexpansion

set "AUDIO_FILE=%~1"

if "%AUDIO_FILE%"=="" (
    echo Usage: normalize_and_clear.cmd "path\to\audio.m4a" [--skip-clear]
    exit /b 1
)

echo ================================================
echo Normalizing and clearing: %AUDIO_FILE%
echo ================================================

python -m speechlib.tools.normalize_and_clear %*

if errorlevel 1 (
    echo ERROR: Processing failed
    exit /b 1
)

echo.
echo Done!
