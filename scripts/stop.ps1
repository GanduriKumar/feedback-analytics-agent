param(
    [int]$BackendPort = 8000,
    [int[]]$FrontendPorts = @(5173, 5174, 5175)
)

function Stop-ByWindowTitle {
    param([string]$Title)
    $procs = Get-Process | Where-Object { $_.MainWindowTitle -like "*$Title*" }
    foreach ($p in $procs) {
        Write-Host "Stopping window: $($p.MainWindowTitle) (PID $($p.Id))" -ForegroundColor Yellow
        Stop-Process -Id $p.Id -Force -ErrorAction SilentlyContinue
    }
}

function Stop-ByPort {
    param([int]$Port)
    try {
        $connections = Get-NetTCPConnection -LocalPort $Port -ErrorAction SilentlyContinue
        foreach ($c in $connections) {
            if ($c.OwningProcess) {
                Write-Host "Stopping process on port $Port (PID $($c.OwningProcess))" -ForegroundColor Yellow
                Stop-Process -Id $c.OwningProcess -Force -ErrorAction SilentlyContinue
            }
        }
    } catch {
        Write-Host "Unable to query port ${Port}: $($_.Exception.Message)" -ForegroundColor DarkYellow
    }
}

function Stop-ByCommandLine {
    param([string]$Match)
    $procs = Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -and $_.CommandLine -like "*$Match*" }
    foreach ($p in $procs) {
        Write-Host "Stopping process: $($p.Name) (PID $($p.ProcessId))" -ForegroundColor Yellow
        Stop-Process -Id $p.ProcessId -Force -ErrorAction SilentlyContinue
    }
}

Write-Host "Stopping backend/frontend service windows..." -ForegroundColor Cyan
Stop-ByWindowTitle -Title "FAA Backend"
Stop-ByWindowTitle -Title "FAA Frontend"

Write-Host "Stopping uvicorn reload processes..." -ForegroundColor Cyan
Stop-ByCommandLine -Match "uvicorn app.main:app"
Stop-ByCommandLine -Match "feedback-analytics-agent\\backend"

Write-Host "Ensuring ports are freed..." -ForegroundColor Cyan
Stop-ByPort -Port $BackendPort
foreach ($p in $FrontendPorts) {
    Stop-ByPort -Port $p
}

Write-Host "Done." -ForegroundColor Green