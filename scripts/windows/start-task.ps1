# Start BioSonification task

$ServiceName = "BioSonification"

Write-Host "Starting $ServiceName task..." -ForegroundColor Cyan

# Check if task exists
$task = Get-ScheduledTask -TaskName $ServiceName -ErrorAction SilentlyContinue
if (-not $task) {
    Write-Host "ERROR: Task '$ServiceName' not found" -ForegroundColor Red
    Write-Host "Please run .\scripts\install-task.ps1 first" -ForegroundColor Yellow
    exit 1
}

# Start the task
Start-ScheduledTask -TaskName $ServiceName

# Wait for task to start
Start-Sleep -Seconds 3

# Check status
$taskInfo = Get-ScheduledTask -TaskName $ServiceName
Write-Host "Task status: $($taskInfo.State)" -ForegroundColor $(if ($taskInfo.State -eq "Running") { "Green" } else { "Yellow" })

# Check health endpoint
Write-Host ""
Write-Host "Checking health endpoint..." -ForegroundColor Cyan
Start-Sleep -Seconds 2  # Give app time to start

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
    Write-Host ""
    Write-Host "Check logs for details:"
    Write-Host "  Get-Content logs\service.log -Tail 20"
    Write-Host "  Get-Content logs\stderr.log -Tail 20"
}

Write-Host ""
Write-Host "Web interface: http://localhost:5001" -ForegroundColor Cyan
Write-Host ""
