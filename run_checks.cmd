@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Create .venv and install requirements-reproduce.txt first.
  echo See RUN_WINDOWS_RU.md.
  exit /b 1
)
".venv\Scripts\python.exe" -m topopt.evidence --output my_results
if errorlevel 1 exit /b 1
".venv\Scripts\python.exe" -m topopt.mesh_study --output my_mesh_study
if errorlevel 1 exit /b 1
echo Completed. Open my_results\report.html and my_mesh_study\report.html.
exit /b 0
