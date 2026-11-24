# One-time BrickGen installer: venv and Python dependencies.
$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $root

Write-Host "=== BrickGen setup ===" -ForegroundColor Cyan

# --- Python check (py launcher, 3.11+) ---
$python = $null
foreach ($candidate in @('py -V:3.13', 'py -V:3.12', 'py -V:3.11')) {
  $name, $ver = $candidate.Split(' ')
  & $name $ver -c "1" 2>$null
  if ($LASTEXITCODE -eq 0) { $python = $candidate; break }
}
if (-not $python) {
  Write-Error "Python 3.11+ not found via py launcher. Install Python first."
}
Write-Host "Using: $python"

# --- venv ---
if (-not (Test-Path "$root\.venv\Scripts\python.exe")) {
  Write-Host "Creating .venv..."
  $parts = $python.Split(' ')
  & $parts[0] $parts[1] -m venv "$root\.venv"
  if ($LASTEXITCODE -ne 0) { Write-Error "venv creation failed" }
}

Write-Host ""
Write-Host "Installing Python dependencies (cadquery, pydantic, numpy, matplotlib, typer)..." -ForegroundColor Cyan

Write-Host "  [1/2] Upgrading pip..."
& "$root\.venv\Scripts\python.exe" -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) { Write-Error "pip upgrade failed" }

Write-Host "  [2/2] Installing project dependencies (this may take several minutes)..."
Write-Host ""
& "$root\.venv\Scripts\python.exe" -m pip install -e "$root[dev]"
if ($LASTEXITCODE -ne 0) { Write-Error "pip install failed" }

Write-Host ""
Write-Host "  Dependencies installed successfully." -ForegroundColor Green

Write-Host ""
Write-Host "Setup complete. Start with start.bat." -ForegroundColor Green
