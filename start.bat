@echo off
REM AUTHENTICA - Quick Start Script for Windows

setlocal enabledelayedexpansion

title AUTHENTICA - Signature Verification System

echo.
echo ============================================================
echo       [AUTHENTICA] Signature Verification System
echo ============================================================
echo.

set CONDA_ENV=authentica-cpu
set BACKEND_PORT=5000
set FRONTEND_PORT=8000

REM Check conda
echo [1/4] Checking conda environment...
conda info --envs | find /I "%CONDA_ENV%" >nul
if !errorlevel! neq 0 (
    echo Creating environment '%CONDA_ENV%'...
    call conda create -n %CONDA_ENV% python=3.10 -y
) else (
    echo [OK] Environment '%CONDA_ENV%' found
)

REM Install dependencies
echo.
echo [2/4] Installing dependencies...
call conda run -n %CONDA_ENV% pip install -q -r requirements.txt
call conda run -n %CONDA_ENV% pip install -q Flask Flask-CORS
echo [OK] Dependencies ready

REM Check model
echo.
echo [3/4] Verifying model checkpoint...
if exist "checkpoints\best_model.pth" (
    echo [OK] Model checkpoint found
) else (
    echo [WARNING] Model checkpoint not found
    echo To train: conda run -n %CONDA_ENV% python train.py
)

REM Start backend
echo.
echo [4/4] Starting services...
echo.
echo ============================================================
echo [BACKEND] Starting Flask server on http://localhost:%BACKEND_PORT%
echo ============================================================
echo.

start "AUTHENTICA Backend" cmd /k conda run -n %CONDA_ENV% python app.py

timeout /t 3 /nobreak

REM Start frontend
echo.
echo ============================================================
echo [FRONTEND] Starting HTTP server on http://localhost:%FRONTEND_PORT%
echo ============================================================
echo.

start "AUTHENTICA Frontend" cmd /k python -m http.server %FRONTEND_PORT%

timeout /t 2 /nobreak

REM Open in browser
start http://localhost:%FRONTEND_PORT%

echo.
echo [OK] Services started!
echo.
echo URLs:
echo   Frontend: http://localhost:%FRONTEND_PORT%
echo   Backend:  http://localhost:%BACKEND_PORT%
echo   Health:   http://localhost:%BACKEND_PORT%/health
echo.
echo Press any key to exit...
pause >nul

REM Kill processes
taskkill /F /IM python.exe >nul 2>&1

echo Services stopped.
endlocal
