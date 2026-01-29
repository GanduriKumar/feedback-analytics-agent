param(
    [int]$BackendPort = 8000,
    [int]$FrontendPort = 5173
)

$root = Split-Path -Parent $PSScriptRoot
$backendPath = Join-Path $root "backend"
$frontendPath = Join-Path $root "frontend"

Write-Host "Starting backend in a new PowerShell window..." -ForegroundColor Cyan
Start-Process -FilePath "powershell.exe" -WorkingDirectory $backendPath -ArgumentList @(
    "-NoExit",
    "-Command",
    "`$host.UI.RawUI.WindowTitle='FAA Backend'; Set-Location '$backendPath'; python -m uvicorn app.main:app --reload --port $BackendPort"
)

Write-Host "Starting frontend in a new PowerShell window..." -ForegroundColor Cyan
Start-Process -FilePath "powershell.exe" -WorkingDirectory $frontendPath -ArgumentList @(
    "-NoExit",
    "-Command",
    "`$host.UI.RawUI.WindowTitle='FAA Frontend'; Set-Location '$frontendPath'; npm run dev -- --port $FrontendPort"
)

Write-Host "Done. Two windows should be running (Backend + Frontend)." -ForegroundColor Green