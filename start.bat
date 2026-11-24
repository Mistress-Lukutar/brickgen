@echo off
setlocal
cd /d "%~dp0"

rem --- Optional skip flag: start.bat /skipupdate ---
set "SKIP_UPDATE=0"
if /i "%~1"=="/skipupdate" set "SKIP_UPDATE=1"

rem --- Update from git (skipped with /skipupdate) ---
if not "%SKIP_UPDATE%"=="1" (
  echo Updating repository...
  git pull --ff-only 2>nul
  if errorlevel 1 (
    echo Note: git pull failed or repository is not a git repo — continuing anyway.
  )
)

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
