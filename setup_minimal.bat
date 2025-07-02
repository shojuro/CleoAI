@echo off
REM Batch script to set up minimal CleoAI API
echo Setting up CleoAI Minimal API...

REM Check if virtual environment is activated
if "%VIRTUAL_ENV%"=="" (
    echo Virtual environment not activated. Activating...
    call venv\Scripts\activate.bat
)

REM Install minimal requirements
echo.
echo Installing minimal requirements...
pip install -r requirements-minimal.txt

if %errorlevel% equ 0 (
    echo.
    echo Installation successful!
    echo.
    echo Starting minimal API server...
    python main_api_minimal.py --debug
) else (
    echo.
    echo Installation failed. Please check the error messages above.
    pause
)