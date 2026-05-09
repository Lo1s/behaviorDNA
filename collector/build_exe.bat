@echo off
REM build_exe.bat
REM Run this on Windows to compile recorder_gui.py into a standalone .exe
REM Requirements: pip install pyinstaller pynput

echo.
echo === BehaviorDNA Recorder - Build ===
echo.

where python >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found. Install Python from https://python.org
    pause
    exit /b 1
)

echo [1/3] Installing dependencies...
pip install pyinstaller pynput --quiet

echo [2/3] Building .exe...
pyinstaller ^
    --onefile ^
    --windowed ^
    --name BehaviorDNA_Recorder ^
    --add-data "README.md;." ^
    recorder_gui.py

echo [3/3] Done!
echo.
echo Your .exe is at:  dist\BehaviorDNA_Recorder.exe
echo Share that single file with your friends - no Python needed.
echo.
pause
