@echo off
setlocal
cd /d "%~dp0"
if not exist .venv\Scripts\python.exe (
  echo .venv not found. Run setup.bat first.
  exit /b 2
)
echo Starting BrickGen GUI...
if exist .venv\Scripts\pythonw.exe (
  start "" .venv\Scripts\pythonw.exe -m brickgen.gui.app
) else (
  .venv\Scripts\python.exe -m brickgen.gui.app
)
