@echo off
REM GestureBind Installation Script for Windows

echo ===== GestureBind Installation =====
echo Installing dependencies for GestureBind

REM Check if Python is installed
python --version >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo Python is not installed or not in PATH
    echo Please install Python 3.7 or higher from https://www.python.org/downloads/
    pause
    exit /b 1
)

REM Check Python version
for /f "tokens=2" %%I in ('python --version') do set PYTHON_VERSION=%%I
for /f "tokens=1,2 delims=." %%A in ("%PYTHON_VERSION%") do (
    set MAJOR=%%A
    set MINOR=%%B
)

if %MAJOR% LSS 3 (
    echo Python 3.7 or higher is required. You have %PYTHON_VERSION%
    pause
    exit /b 1
) else (
    if %MAJOR% EQU 3 (
        if %MINOR% LSS 7 (
            echo Python 3.7 or higher is required. You have %PYTHON_VERSION%
            pause
            exit /b 1
        )
    )
)

REM Install required packages
echo Installing Python dependencies...
pip install -r ..\requirements.txt

REM Create desktop shortcut
echo Creating desktop shortcut...
set SCRIPT_DIR=%~dp0
set MAIN_DIR=%SCRIPT_DIR%..

echo Set oWS = WScript.CreateObject("WScript.Shell") > %TEMP%\shortcut.vbs
echo sLinkFile = "%USERPROFILE%\Desktop\GestureBind.lnk" >> %TEMP%\shortcut.vbs
echo Set oLink = oWS.CreateShortcut(sLinkFile) >> %TEMP%\shortcut.vbs
echo oLink.TargetPath = "pythonw" >> %TEMP%\shortcut.vbs
echo oLink.Arguments = "%MAIN_DIR%\main.py" >> %TEMP%\shortcut.vbs
echo oLink.WorkingDirectory = "%MAIN_DIR%" >> %TEMP%\shortcut.vbs
echo oLink.Description = "Gesture-Based Shortcut Mapper" >> %TEMP%\shortcut.vbs
echo oLink.IconLocation = "%MAIN_DIR%\resources\icon.ico, 0" >> %TEMP%\shortcut.vbs
echo oLink.Save >> %TEMP%\shortcut.vbs
cscript /nologo %TEMP%\shortcut.vbs
del %TEMP%\shortcut.vbs

echo Installation complete!
echo You can start GestureBind using the desktop shortcut or by running 'python main.py' from the main directory.
pause