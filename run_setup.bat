@echo off
REM Batch file to run the setup steps

echo Running CleoAI Development Setup...
echo.

echo Step 1: Setting up development environment...
python setup_dev_environment.py

echo.
echo Step 2: Creating test data...
python create_test_data.py

echo.
echo Step 3: Testing memory system...
python test_memory_simple.py

echo.
echo Setup complete! Next steps:
echo - Run the API: python main_api_minimal.py
echo - Test endpoints: Run test_api.ps1 or use curl/browser
pause