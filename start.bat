@echo off
setlocal
cd /d "%~dp0"

rem --- Dependency check: venv present and all imports work ---
set "DEPS_OK=0"
if exist .venv\Scripts\python.exe (
  .venv\Scripts\python.exe -c "import brickgen, cadquery, matplotlib, typer, pydantic" >nul 2>&1 && set "DEPS_OK=1"
)

rem --- Install on demand ---
if "%DEPS_OK%"=="0" (
  echo Dependencies missing - running setup...
  powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\setup.ps1"
  if errorlevel 1 (
    echo.
    echo Setup failed. Fix the errors above and retry.
    echo %cmdcmdline% | find /i "%~f0" >nul && pause
    exit /b 1
  )
)

echo Starting BrickGen GUI...
if exist .venv\Scripts\pythonw.exe (
  start "" .venv\Scripts\pythonw.exe -m brickgen.gui.app
) else (
  .venv\Scripts\python.exe -m brickgen.gui.app
)
exit /b 0
