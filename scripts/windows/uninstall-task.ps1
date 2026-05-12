# Uninstall BioSonification Task Scheduler task
# Run this script as Administrator

$ServiceName = "BioSonification"

Write-Host "=== BioSonification Task Uninstallation ===" -ForegroundColor Cyan
Write-Host ""

# Check if running as Administrator
$currentPrincipal = New-Object Security.Principal.WindowsPrincipal([Security.Principal.WindowsIdentity]::GetCurrent())
if (-not $currentPrincipal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
    Write-Host "ERROR: This script must be run as Administrator" -ForegroundColor Red
    exit 1
}

# Check if task exists
$task = Get-ScheduledTask -TaskName $ServiceName -ErrorAction SilentlyContinue
if (-not $task) {
    Write-Host "Task '$ServiceName' not found" -ForegroundColor Yellow
    exit 0
}

# Stop task if running
Write-Host "Stopping task..."
Stop-ScheduledTask -TaskName $ServiceName -ErrorAction SilentlyContinue
Start-Sleep -Seconds 2

# Remove task
Write-Host "Removing task..."
Unregister-ScheduledTask -TaskName $ServiceName -Confirm:$false

Write-Host ""
Write-Host "Task removed successfully!" -ForegroundColor Green
Write-Host ""
