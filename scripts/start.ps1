# Start Production-RAG (API + frontend). Qdrant must already be running on :6333.
# Usage:  .\scripts\start.ps1
# Optional: $env:RAG_PYTHON = "F:\miniconda3\envs\copernicus\python.exe"
#           $env:RAG_CONDA_ENV = "copernicus"

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

function Get-RagPython {
    if ($env:RAG_PYTHON -and (Test-Path $env:RAG_PYTHON)) {
        return $env:RAG_PYTHON
    }
    $envName = if ($env:RAG_CONDA_ENV) { $env:RAG_CONDA_ENV } else { "copernicus" }
    $candidates = @(
        "F:\miniconda3\envs\$envName\python.exe",
        "$env:USERPROFILE\miniconda3\envs\$envName\python.exe",
        "$env:USERPROFILE\anaconda3\envs\$envName\python.exe",
        "$env:LOCALAPPDATA\miniconda3\envs\$envName\python.exe"
    )
    foreach ($p in $candidates) {
        if (Test-Path $p) { return $p }
    }
    $cmd = Get-Command python -ErrorAction SilentlyContinue
    if ($cmd) { return $cmd.Source }
    throw "Python not found. Set RAG_PYTHON or create conda env '$envName'."
}

function Test-PortListening([int]$Port) {
    return [bool](Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue)
}

if (-not (Test-Path (Join-Path $Root ".env"))) {
    throw ".env missing. Copy .env.example to .env and fill in keys first."
}

Write-Host "Checking Qdrant (http://127.0.0.1:6333)..." -ForegroundColor Cyan
try {
    $null = Invoke-WebRequest "http://127.0.0.1:6333/readyz" -UseBasicParsing -TimeoutSec 3
    Write-Host "  Qdrant OK" -ForegroundColor Green
} catch {
    Write-Host "  Qdrant is not ready. Start it first, then re-run this script." -ForegroundColor Yellow
    Write-Host "  Example: docker run -d --name qdrant -p 6333:6333 qdrant/qdrant" -ForegroundColor DarkYellow
    exit 1
}

$Python = Get-RagPython
Write-Host "Using Python: $Python" -ForegroundColor Cyan

if (Test-PortListening 8000) {
    Write-Host "API already listening on :8000" -ForegroundColor Yellow
} else {
    Write-Host "Starting API on http://127.0.0.1:8000 ..." -ForegroundColor Cyan
    Start-Process powershell -WorkingDirectory $Root -ArgumentList @(
        "-NoExit",
        "-Command",
        "& '$Python' -m uvicorn app.main:app --host 127.0.0.1 --port 8000"
    )
}

if (Test-PortListening 5173) {
    Write-Host "Frontend already listening on :5173" -ForegroundColor Yellow
} else {
    $fe = Join-Path $Root "frontend"
    if (-not (Test-Path (Join-Path $fe "node_modules"))) {
        Write-Host "frontend/node_modules missing; run npm install in frontend first." -ForegroundColor Red
        exit 1
    }
    Write-Host "Starting frontend on http://127.0.0.1:5173 ..." -ForegroundColor Cyan
    Start-Process powershell -WorkingDirectory $fe -ArgumentList @(
        "-NoExit",
        "-Command",
        "npm run dev -- --host 127.0.0.1 --port 5173"
    )
}

Write-Host ""
Write-Host "Started. Open http://127.0.0.1:5173" -ForegroundColor Green
Write-Host "Stop with:  .\scripts\stop.ps1" -ForegroundColor DarkGray
