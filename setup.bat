@echo off
setlocal
cd /d "%~dp0"

py -3.11 --version >nul 2>&1
if errorlevel 1 (
    echo Python 3.11 was not found. Install Python 3.11 and enable the Python launcher.
    exit /b 1
)

if not exist ".venv\Scripts\python.exe" (
    echo Creating Python 3.11 virtual environment...
    py -3.11 -m venv .venv
    if errorlevel 1 exit /b 1
)

echo Installing project dependencies...
".venv\Scripts\python.exe" -m pip install --upgrade pip
if errorlevel 1 exit /b 1
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 exit /b 1

if not exist "models" mkdir "models"
if not exist "models\hand_landmarker.task" (
    echo Downloading the MediaPipe hand landmark model...
    curl.exe -fL "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task" -o "models\hand_landmarker.task"
    if errorlevel 1 (
        echo Model download failed. Check your internet connection and run setup.bat again.
        exit /b 1
    )
)

echo Setup complete. Start the app with run.bat.
exit /b 0
