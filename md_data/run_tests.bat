@echo off
REM Script to run tests for the MobileDepth model on Windows

echo Starting tests for the MobileDepth model...
python -m unittest discover -s . -p "test_depth_model.py" -v

REM Check the return code
IF %ERRORLEVEL% EQU 0 (
    echo All tests passed successfully!
) ELSE (
    echo Some tests finished with errors!
    exit /b 1
)

