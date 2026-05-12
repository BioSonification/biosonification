# Restart BioSonification task

$ServiceName = "BioSonification"

Write-Host "Restarting $ServiceName task..." -ForegroundColor Cyan

# Check if task exists
$task = Get-ScheduledTask -TaskName $ServiceName -ErrorAction SilentlyContinue
if (-not $task) {
    Write-Host "ERROR: Task '$ServiceName' not found" -ForegroundColor Red
    exit 1
}

# Stop the task
Write-Host "Stopping task..."
Stop-ScheduledTask -TaskName $ServiceName
Start-Sleep -Seconds 3

# Start the task
Write-Host "Starting task..."
Start-ScheduledTask -TaskName $ServiceName
Start-Sleep -Seconds 5

# Check status
$taskInfo = Get-ScheduledTask -TaskName $ServiceName
Write-Host "Task status: $($taskInfo.State)" -ForegroundColor $(if ($taskInfo.State -eq "Running") { "Green" } else { "Yellow" })

# Check health endpoint
Write-Host ""
Write-Host "Checking health endpoint..." -ForegroundColor Cyan
try {
    $response = Invoke-WebRequest -Uri "http://localhost:5001/health" -UseBasicParsing -TimeoutSec 10
    Write-Host "Health check: OK" -ForegroundColor Green
    $health = $response.Content | ConvertFrom-Json
    Write-Host "  Status: $($health.status)"
    Write-Host "  Generator Ready: $($health.generator_ready)"
    Write-Host "  GPU Available: $($health.gpu_available)"
} catch {
    Write-Host "Health check: FAILED" -ForegroundColor Red
    Write-Host "  Error: $($_.Exception.Message)" -ForegroundColor Yellow
}

Write-Host ""
