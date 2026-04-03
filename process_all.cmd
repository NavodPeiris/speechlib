@echo off
REM Batch process all audio files in a directory
REM Usage: process_all.cmd "path\to\directory"

setlocal enabledelayedexpansion

set "INPUT_DIR=%~1"

if "%INPUT_DIR%"=="" (
    echo Usage: process_all.cmd "path\to\directory"
    exit /b 1
)

set "SCRIPT_DIR=%~dp0"
set "OUTPUT_DIR=%INPUT_DIR%\_processed"

echo ================================================
echo Batch Processing: %INPUT_DIR%
echo Output: %OUTPUT_DIR%
echo ================================================

REM Create output directory
if not exist "%OUTPUT_DIR%" mkdir "%OUTPUT_DIR%"

REM Process M4A files
for %%F in ("%INPUT_DIR%\*.m4a") do (
    echo.
    echo Processing: %%~nxF
    python "%SCRIPT_DIR%process_audio.py" "%%F" "%OUTPUT_DIR%"
    if errorlevel 1 (
        echo FAILED: %%~nxF
    ) else (
        echo OK: %%~nxF
    )
)

REM Process MP3 files
for %%F in ("%INPUT_DIR%\*.mp3") do (
    echo.
    echo Processing: %%~nxF
    python "%SCRIPT_DIR%process_audio.py" "%%F" "%OUTPUT_DIR%"
    if errorlevel 1 (
        echo FAILED: %%~nxF
    ) else (
        echo OK: %%~nxF
    )
)

REM Process WAV files
for %%F in ("%INPUT_DIR%\*.wav") do (
    echo.
    echo Processing: %%~nxF
    python "%SCRIPT_DIR%process_audio.py" "%%F" "%OUTPUT_DIR%"
    if errorlevel 1 (
        echo FAILED: %%~nxF
    ) else (
        echo OK: %%~nxF
    )
)

echo.
echo ================================================
echo Batch processing complete!
echo Output directory: %OUTPUT_DIR%
echo ================================================
