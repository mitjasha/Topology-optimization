@echo off
setlocal
cd /d "%~dp0"
set "PYTHON=python"
if exist ".venv\Scripts\python.exe" set "PYTHON=.venv\Scripts\python.exe"
%PYTHON% -c "import streamlit" >nul 2>&1
if errorlevel 1 (
    echo.
    echo Install GUI requirements first:
    echo    %PYTHON% -m pip install -r requirements-gui.txt
    echo.
    pause
    exit /b 1
)
%PYTHON% -m streamlit run topopt\mlnext\gui.py --server.address 127.0.0.1
if errorlevel 1 pause
