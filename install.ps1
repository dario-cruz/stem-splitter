<#
.SYNOPSIS
    Installer for Stem Splitter (stems / stem-splitter) on Windows.

.DESCRIPTION
    Checks/installs prerequisites (uv, FFmpeg), installs stem-splitter either
    globally (default) or locally (.venv only), and optionally pre-downloads
    local Demucs v4 neural network weights for 100% offline use.

.PARAMETER Mode
    'global' (default: installs `stems` and `stem-splitter` into user PATH)
    or 'local' (creates `.venv` inside the repo only).

.PARAMETER SkipModels
    Skip pre-downloading the local Demucs v4 model weights during setup.

.EXAMPLE
    .\install.ps1
    .\install.ps1 -Mode local
#>
param(
    [ValidateSet("global", "local")]
    [string]$Mode = "global",
    [switch]$SkipModels
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $RepoRoot

Write-Host "`n======================================================" -ForegroundColor Cyan
Write-Host "  Stem Splitter Installer ($Mode mode)" -ForegroundColor Cyan
Write-Host "======================================================`n" -ForegroundColor Cyan

function Refresh-UserPath {
    $env:Path = [System.Environment]::GetEnvironmentVariable("Path", "Machine") + ";" +
                [System.Environment]::GetEnvironmentVariable("Path", "User")
}

# 1. Check / Install uv
Refresh-UserPath
if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    Write-Host "[1/4] 'uv' not found. Installing Astral uv..." -ForegroundColor Yellow
    if (Get-Command winget -ErrorAction SilentlyContinue) {
        winget install --id=astral-sh.uv -e --accept-source-agreements --accept-package-agreements
    } else {
        irm https://astral.sh/uv/install.ps1 | iex
    }
    Refresh-UserPath
} else {
    Write-Host "[1/4] Found uv: $(uv --version)" -ForegroundColor Green
}

# 2. Check / Install FFmpeg
if (-not (Get-Command ffmpeg -ErrorAction SilentlyContinue)) {
    Write-Host "[2/4] 'ffmpeg' not found. Installing FFmpeg via winget..." -ForegroundColor Yellow
    if (Get-Command winget -ErrorAction SilentlyContinue) {
        winget install --id=Gyan.FFmpeg -e --accept-source-agreements --accept-package-agreements
        Refresh-UserPath
    } else {
        Write-Warning "FFmpeg is not installed and winget is unavailable. Please install FFmpeg and add it to PATH."
    }
} else {
    Write-Host "[2/4] Found FFmpeg on PATH" -ForegroundColor Green
}

# 3. Install stem-splitter (Global or Local)
if ($Mode -eq "global") {
    Write-Host "[3/4] Installing 'stems' & 'stem-splitter' globally via uv..." -ForegroundColor Cyan
    uv sync
    uv tool install --editable . --force
    uv tool update-shell
    Refresh-UserPath
} else {
    Write-Host "[3/4] Installing local virtual environment (.venv) via uv..." -ForegroundColor Cyan
    uv sync
}

# 4. Pre-cache offline Demucs weights into ./models
if (-not $SkipModels) {
    Write-Host "[4/4] Ensuring local offline Demucs v4 weights exist in .\models..." -ForegroundColor Cyan
    uv run stems --download-models
} else {
    Write-Host "[4/4] Skipping model weight download (-SkipModels specified)." -ForegroundColor DarkGray
}

Write-Host "`n======================================================" -ForegroundColor Green
Write-Host "  Setup Complete!" -ForegroundColor Green
Write-Host "======================================================" -ForegroundColor Green
if ($Mode -eq "global") {
    Write-Host "  Open a new terminal window and run from anywhere:"
    Write-Host "    stems song.mp3" -ForegroundColor Yellow
    Write-Host "    stems song.mp3 -6" -ForegroundColor Yellow
    Write-Host "    stems song.mp3 -v" -ForegroundColor Yellow
} else {
    Write-Host "  Run from inside this repository:"
    Write-Host "    uv run stems song.mp3" -ForegroundColor Yellow
}
Write-Host ""
