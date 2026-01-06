@echo off
title JUNKGAME Server
echo ==========================================
echo      JUNKGAME Server Launcher
echo ==========================================

echo [1/3] Installing/Verifying Dependencies...
python -m pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo [ERROR] Pip install failed.
    pause
    exit /b
)

echo.
echo [2/3] Testing Model Loading...
python test_model.py
if %errorlevel% neq 0 (
    echo [ERROR] Model test failed.
    pause
    exit /b
)

echo.
echo [3/3] Starting Server...
echo API will be available at http://127.0.0.1:8000
python -m uvicorn main:app --reload --host 0.0.0.0 --port 8000

pause
