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
                taskkill /F /T /PID $($c.OwningProcess) 2>$null | Out-Null
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
        # Use taskkill with /F /T to force kill the process tree
        taskkill /F /T /PID $($p.ProcessId) 2>$null | Out-Null
        # Fallback to Stop-Process
        Stop-Process -Id $p.ProcessId -Force -ErrorAction SilentlyContinue
    }
}

function Stop-ByNameAndPath {
    param([string]$Name, [string]$PathMatch)
    $procs = Get-CimInstance Win32_Process | Where-Object {
        $_.Name -eq $Name -and $_.CommandLine -and $_.CommandLine -like "*$PathMatch*"
    }
    foreach ($p in $procs) {
        Write-Host "Stopping $Name tied to $PathMatch (PID $($p.ProcessId))" -ForegroundColor Yellow
        Stop-Process -Id $p.ProcessId -Force -ErrorAction SilentlyContinue
    }
}

function Stop-ByPortFallback {
    param([int]$Port)
    try {
        $lines = netstat -ano | Select-String -Pattern ":$Port\s"
        foreach ($line in $lines) {
            $parts = ($line -replace "\s+", " ").Trim().Split(" ")
            $procId = $parts[-1]
            if ($procId -match "^\d+$") {
                Write-Host "Stopping process on port $Port (PID $procId)" -ForegroundColor Yellow
                Stop-Process -Id $procId -Force -ErrorAction SilentlyContinue
            }
        }
    } catch {
        Write-Host "Fallback port scan failed for ${Port}: $($_.Exception.Message)" -ForegroundColor DarkYellow
    }
}

Write-Host "Stopping backend/frontend service windows..." -ForegroundColor Cyan
Stop-ByWindowTitle -Title "FAA Backend"
Stop-ByWindowTitle -Title "FAA Frontend"

# Also stop Windows Terminal running these services
$terminalProcs = Get-Process -Name "WindowsTerminal" -ErrorAction SilentlyContinue
foreach ($term in $terminalProcs) {
    if ($term.MainWindowTitle -like "*FAA Backend*" -or $term.MainWindowTitle -like "*FAA Frontend*") {
        Write-Host "Stopping Windows Terminal: $($term.MainWindowTitle) (PID $($term.Id))" -ForegroundColor Yellow
        Stop-Process -Id $term.Id -Force -ErrorAction SilentlyContinue
    }
}


Write-Host "Stopping uvicorn/backend processes (including reloaders/children)..." -ForegroundColor Cyan
# Stop any uvicorn process (reloader, server, etc.)
Stop-ByCommandLine -Match "uvicorn"
# Stop any python process running the backend app
Stop-ByCommandLine -Match "app.main:app"
# Stop any python or uvicorn process with backend path
Stop-ByNameAndPath -Name "python.exe" -PathMatch "feedback-analytics-agent\\backend"
Stop-ByNameAndPath -Name "uvicorn.exe" -PathMatch "feedback-analytics-agent\\backend"
# Extra: Stop any python process with 'reload' in command line (uvicorn reload)
Stop-ByCommandLine -Match "reload"
# Extra: Stop any python process with 'WatchFiles' (uvicorn's reloader)
Stop-ByCommandLine -Match "WatchFiles"

Write-Host "Stopping frontend launcher/window processes..." -ForegroundColor Cyan
# Kill the PowerShell window that launched the frontend (it uses -NoExit, so it stays open unless we terminate it)
Stop-ByCommandLine -Match "FAA Frontend"
Stop-ByCommandLine -Match "npm run dev"
Stop-ByCommandLine -Match "feedback-analytics-agent\\frontend"
# Kill Node/Vite processes tied to the frontend folder (in case they outlive the shell)
Stop-ByNameAndPath -Name "node.exe" -PathMatch "feedback-analytics-agent\\frontend"

Write-Host "Ensuring ports are freed..." -ForegroundColor Cyan
Stop-ByPort -Port $BackendPort
Stop-ByPortFallback -Port $BackendPort
foreach ($p in $FrontendPorts) {
    Stop-ByPort -Port $p
    Stop-ByPortFallback -Port $p
}

Write-Host "Cleaning up orphaned port bindings..." -ForegroundColor Cyan
Start-Sleep -Seconds 2
# Force close any remaining TCP connections on the ports
try {
    $tcpConns = Get-NetTCPConnection -LocalPort $BackendPort -ErrorAction SilentlyContinue
    foreach ($conn in $tcpConns) {
        if ($conn.OwningProcess) {
            $proc = Get-Process -Id $conn.OwningProcess -ErrorAction SilentlyContinue
            if (-not $proc) {
                Write-Host "Found orphaned port binding on port $BackendPort (zombie PID $($conn.OwningProcess))" -ForegroundColor Yellow
                Write-Host "Port will be released when the TCP stack times out (may take 30-60 seconds)" -ForegroundColor DarkYellow
            } else {
                Write-Host "Stopping remaining process on port $BackendPort (PID $($conn.OwningProcess))" -ForegroundColor Yellow
                Stop-Process -Id $conn.OwningProcess -Force -ErrorAction SilentlyContinue
            }
        }
    }
} catch {
    # Ignore errors
}

Write-Host "Done. If ports are still in use, wait 30-60 seconds for TCP cleanup." -ForegroundColor Green