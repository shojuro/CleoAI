@echo off
REM Quick installation script for CleoAI full setup

echo ========================================
echo CleoAI Full Setup Implementation
echo ========================================
echo This will install all dependencies and set up the complete environment
echo.

REM Check if virtual environment is activated
if "%VIRTUAL_ENV%"=="" (
    echo Activating virtual environment...
    call venv\Scripts\activate.bat
    if errorlevel 1 (
        echo Error: Cannot activate virtual environment
        echo Please run this from the CleoAI directory with venv folder
        pause
        exit /b 1
    )
)

echo Virtual environment active: %VIRTUAL_ENV%
echo.

REM Upgrade pip first
echo Upgrading pip...
python -m pip install --upgrade pip setuptools wheel

REM Run the full setup implementation
echo Running full setup implementation...
python implement_full_setup.py

if errorlevel 1 (
    echo.
    echo Setup encountered some issues, but may still be partially functional
) else (
    echo.
    echo ========================================
    echo Setup completed successfully!
    echo ========================================
)

echo.
echo Quick start commands:
echo   python main_api_minimal.py    - Start minimal API
echo   start_cleoai.bat             - Start with batch file
echo   python test_api_simple.py    - Test the API
echo.

pause