# Stop Production-RAG API (:8000) and frontend (:5173). Does not stop Qdrant.
# Usage:  .\scripts\stop.ps1

$ErrorActionPreference = "Continue"

function Stop-PortListeners([int]$Port) {
    $conns = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue
    if (-not $conns) {
        Write-Host "Port ${Port}: nothing listening" -ForegroundColor DarkGray
        return
    }
    $pids = $conns | Select-Object -ExpandProperty OwningProcess -Unique | Where-Object { $_ -and $_ -gt 0 }
    foreach ($procId in $pids) {
        $p = Get-Process -Id $procId -ErrorAction SilentlyContinue
        $name = if ($p) { $p.ProcessName } else { "?" }
        Write-Host "Stopping PID $procId ($name) on port $Port" -ForegroundColor Cyan
        Stop-Process -Id $procId -Force -ErrorAction SilentlyContinue
    }
}

Write-Host "Stopping Production-RAG..." -ForegroundColor Cyan
Stop-PortListeners 8000
Stop-PortListeners 5173
Start-Sleep -Seconds 1

$apiUp = [bool](Get-NetTCPConnection -LocalPort 8000 -State Listen -ErrorAction SilentlyContinue)
$feUp = [bool](Get-NetTCPConnection -LocalPort 5173 -State Listen -ErrorAction SilentlyContinue)
Write-Host "API: $(if ($apiUp) { 'still up' } else { 'stopped' })" -ForegroundColor $(if ($apiUp) { 'Yellow' } else { 'Green' })
Write-Host "Frontend: $(if ($feUp) { 'still up' } else { 'stopped' })" -ForegroundColor $(if ($feUp) { 'Yellow' } else { 'Green' })
Write-Host "Qdrant was left running (stop that process separately if needed)." -ForegroundColor DarkGray
