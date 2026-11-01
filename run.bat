@echo off
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo Project setup is required. Running setup.bat...
    call setup.bat
    if errorlevel 1 exit /b 1
)

if not exist "models\hand_landmarker.task" (
    echo The hand landmark model is missing. Running setup.bat...
    call setup.bat
    if errorlevel 1 exit /b 1
)

".venv\Scripts\python.exe" main.py
exit /b %ERRORLEVEL%
