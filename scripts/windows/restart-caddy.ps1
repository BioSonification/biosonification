# Restart Caddy Server
# Run as Administrator

$ServiceName = "CaddyServer"

Write-Host "=== Restarting Caddy Server ===" -ForegroundColor Cyan

# Check Administrator
$currentPrincipal = New-Object Security.Principal.WindowsPrincipal([Security.Principal.WindowsIdentity]::GetCurrent())
if (-not $currentPrincipal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
    Write-Host "ERROR: Run as Administrator" -ForegroundColor Red
    exit 1
}

# Check if task exists
$task = Get-ScheduledTask -TaskName $ServiceName -ErrorAction SilentlyContinue
if (-not $task) {
    Write-Host "ERROR: Task '$ServiceName' not found" -ForegroundColor Red
    Write-Host "Install first: .\install-caddy-service.ps1" -ForegroundColor Yellow
    exit 1
}

# Stop task
Write-Host "Stopping $ServiceName..." -ForegroundColor Yellow
Stop-ScheduledTask -TaskName $ServiceName -ErrorAction SilentlyContinue
Start-Sleep -Seconds 2

# Start task
Write-Host "Starting $ServiceName..." -ForegroundColor Green
Start-ScheduledTask -TaskName $ServiceName
Start-Sleep -Seconds 3

# Check status
$task = Get-ScheduledTask -TaskName $ServiceName
Write-Host ""
Write-Host "Status: $($task.State)" -ForegroundColor $(if ($task.State -eq "Running") { "Green" } else { "Red" })

# Check logs
Write-Host ""
Write-Host "=== Recent Logs ===" -ForegroundColor Cyan
$logFile = "C:\Tools\caddy\logs\stderr.log"
if (Test-Path $logFile) {
    Get-Content $logFile -Tail 10
} else {
    Write-Host "No logs found" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "Done! Check https://biosonification.ddns.net" -ForegroundColor Green
