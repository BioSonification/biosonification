# Stop BioSonification task

$ServiceName = "BioSonification"

Write-Host "Stopping $ServiceName task..." -ForegroundColor Cyan

# Check if task exists
$task = Get-ScheduledTask -TaskName $ServiceName -ErrorAction SilentlyContinue
if (-not $task) {
    Write-Host "ERROR: Task '$ServiceName' not found" -ForegroundColor Red
    exit 1
}

# Stop the task
Stop-ScheduledTask -TaskName $ServiceName

# Wait for task to stop
Start-Sleep -Seconds 3

# Check status
$taskInfo = Get-ScheduledTask -TaskName $ServiceName
Write-Host "Task status: $($taskInfo.State)" -ForegroundColor $(if ($taskInfo.State -eq "Ready") { "Green" } else { "Yellow" })
Write-Host ""
