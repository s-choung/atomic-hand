# Atomic Hand - one-click installer for Windows (PowerShell).
#
# !!! UNTESTED on Windows (authored on macOS). Treat as a starting point;
#     a Windows machine is needed to verify. Report issues / adjust as needed.
#
# Easiest run (no download, no execution-policy fuss):
#   irm https://raw.githubusercontent.com/s-choung/atomic-hand/main/install.ps1 | iex
# Or from a downloaded copy:
#   powershell -ExecutionPolicy Bypass -File install.ps1
#
# It will: install Miniconda if needed -> create the atomic-hand env ->
# install deps (OVITO, MediaPipe, ...) -> create 'Atomic Hand.bat' launcher.

$ErrorActionPreference = "Stop"
$EnvName = "atomic-hand"
$PyVer   = "3.11"
$RepoUrl = "https://github.com/s-choung/atomic-hand.git"

function Say($m)  { Write-Host $m -ForegroundColor Cyan }
function Ok($m)   { Write-Host "OK  $m" -ForegroundColor Green }
function Warn($m) { Write-Host "!   $m" -ForegroundColor Yellow }

Say "Atomic Hand installer (Windows)"
Say "Note: this is an untested draft. Tell us if something breaks."

# 0. locate the project (this script's folder, or clone it)
if ($PSScriptRoot -and (Test-Path (Join-Path $PSScriptRoot "src\app_window.py"))) {
    $Repo = $PSScriptRoot
} else {
    $Repo = Join-Path $env:USERPROFILE "atomic-hand"
    if (-not (Test-Path (Join-Path $Repo "src\app_window.py"))) {
        if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
            throw "git not found. Install Git for Windows (https://git-scm.com), then re-run."
        }
        Say "Downloading Atomic Hand..."
        git clone $RepoUrl $Repo
    }
}
Ok "Project at $Repo"

# 1. find or install conda
function Find-CondaBase {
    if (Get-Command conda -ErrorAction SilentlyContinue) { return (& conda info --base).Trim() }
    foreach ($p in @("$env:USERPROFILE\miniconda3","$env:USERPROFILE\anaconda3",
                     "$env:USERPROFILE\miniforge3","C:\ProgramData\miniconda3")) {
        if (Test-Path "$p\Scripts\conda.exe") { return $p }
    }
    return $null
}

Say "Step 1/4 - checking for conda..."
$CondaBase = Find-CondaBase
if (-not $CondaBase) {
    Warn "conda not found. Installing Miniconda (self-contained, no admin needed)."
    $mc = Join-Path $env:TEMP "miniconda.exe"
    Invoke-WebRequest "https://repo.anaconda.com/miniconda/Miniconda3-latest-Windows-x86_64.exe" -OutFile $mc
    $dest = "$env:USERPROFILE\miniconda3"
    Start-Process $mc -ArgumentList "/S","/InstallationType=JustMe","/AddToPath=0","/RegisterPython=0","/D=$dest" -Wait
    Remove-Item $mc -ErrorAction SilentlyContinue
    $CondaBase = $dest
    Ok "Miniconda installed at $CondaBase"
} else {
    Ok "conda at $CondaBase"
}
$Conda = "$CondaBase\Scripts\conda.exe"

# 2. create the environment
Say "Step 2/4 - creating the '$EnvName' environment (Python $PyVer)..."
$envs = & $Conda env list
if ($envs -match "(?m)^\s*$EnvName\s") {
    Ok "Environment '$EnvName' already exists - reusing it."
} else {
    & $Conda create -y -n $EnvName "python=$PyVer"
    Ok "Environment created."
}

# 3. install dependencies (OVITO first, to avoid Qt conflicts)
Say "Step 3/4 - installing dependencies (this is the long part)..."
& $Conda run -n $EnvName python -m pip install --upgrade pip | Out-Null
& $Conda run -n $EnvName pip install ovito
& $Conda run -n $EnvName pip install "mediapipe==0.10.14" opencv-python PySide6 numpy
Ok "All dependencies installed."

# 4. create a double-clickable launcher
Say "Step 4/4 - creating the launcher..."
$bat = Join-Path $Repo "Atomic Hand.bat"
@"
@echo off
cd /d "$Repo"
"$Conda" run --no-capture-output -n $EnvName python "src\app_window.py" "structures"
"@ | Set-Content -Encoding ASCII -Path $bat
Ok "Launcher created: $bat"

# also drop a copy on the Desktop
$desktop = [Environment]::GetFolderPath("Desktop")
if (Test-Path $desktop) { Copy-Item $bat (Join-Path $desktop "Atomic Hand.bat") -Force }

Write-Host ""
Say "Done!  Double-click 'Atomic Hand.bat' (on your Desktop) to launch."
Say "Allow camera access when Windows asks."
