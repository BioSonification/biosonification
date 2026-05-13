# Monitor BioSonification Task Scheduler task

$ServiceName = "BioSonification"
$ProjectRoot = $PSScriptRoot | Split-Path -Parent
$LogFile = "$ProjectRoot\logs\service.log"
$StdoutLog = "$ProjectRoot\logs\stdout.log"
$StderrLog = "$ProjectRoot\logs\stderr.log"

Write-Host "=== BioSonification Monitor ===" -ForegroundColor Cyan
Write-Host ""

# Check if task exists
$task = Get-ScheduledTask -TaskName $ServiceName -ErrorAction SilentlyContinue
if (-not $task) {
    Write-Host "ERROR: Task '$ServiceName' not found" -ForegroundColor Red
    Write-Host "Please run .\scripts\install-task.ps1 first" -ForegroundColor Yellow
    exit 1
}

# Task status
$taskInfo = Get-ScheduledTask -TaskName $ServiceName
Write-Host "Task Status: $($taskInfo.State)" -ForegroundColor $(if ($taskInfo.State -eq "Running") { "Green" } else { "Yellow" })

# Last run time
$taskInfo = Get-ScheduledTaskInfo -TaskName $ServiceName
Write-Host "Last Run: $($taskInfo.LastRunTime)"
Write-Host "Last Result: $($taskInfo.LastTaskResult) $(if ($taskInfo.LastTaskResult -eq 0) { '(Success)' } else { '(Error)' })"

Write-Host ""

# Health check
Write-Host "=== Health Check ===" -ForegroundColor Cyan
try {
    $response = Invoke-WebRequest -Uri "http://localhost:5001/health" -UseBasicParsing -TimeoutSec 10
    $health = $response.Content | ConvertFrom-Json

    Write-Host "Status: $($health.status)" -ForegroundColor $(if ($health.status -eq "healthy") { "Green" } else { "Yellow" })
    Write-Host "Generator Ready: $($health.generator_ready)"
    Write-Host "GPU Available: $($health.gpu_available)"
    Write-Host "Timestamp: $($health.timestamp)"
} catch {
    Write-Host "Health Check: FAILED" -ForegroundColor Red
    Write-Host "Error: $($_.Exception.Message)" -ForegroundColor Yellow
}

Write-Host ""

# Recent logs
if (Test-Path $LogFile) {
    Write-Host "=== Service Log (last 10 lines) ===" -ForegroundColor Cyan
    Get-Content $LogFile -Tail 10 -ErrorAction SilentlyContinue
    Write-Host ""
}

if (Test-Path $StderrLog) {
    $stderrContent = Get-Content $StderrLog -Tail 5 -ErrorAction SilentlyContinue
    if ($stderrContent) {
        Write-Host "=== Recent Errors (last 5 lines) ===" -ForegroundColor Yellow
        $stderrContent
        Write-Host ""
    }
}

# GPU status (if available)
if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) {
    Write-Host "=== GPU Status ===" -ForegroundColor Cyan
    try {
        $gpuInfo = nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader,nounits
        $gpuInfo | ForEach-Object {
            $parts = $_ -split ','
            Write-Host "GPU: $($parts[0].Trim())"
            Write-Host "  Utilization: $($parts[1].Trim())%"
            Write-Host "  Memory: $($parts[2].Trim()) MB / $($parts[3].Trim()) MB"
            Write-Host "  Temperature: $($parts[4].Trim())°C"
        }
    } catch {
        Write-Host "Could not query GPU status" -ForegroundColor Yellow
    }
    Write-Host ""
}

# Process info
Write-Host "=== Process Info ===" -ForegroundColor Cyan
$pythonProcesses = Get-Process python -ErrorAction SilentlyContinue | Where-Object { $_.Path -like "*biosonification*" }
if ($pythonProcesses) {
    foreach ($proc in $pythonProcesses) {
        Write-Host "PID: $($proc.Id)"
        Write-Host "  CPU: $([math]::Round($proc.CPU, 2))s"
        Write-Host "  Memory: $([math]::Round($proc.WorkingSet64 / 1MB, 2)) MB"
        Write-Host "  Threads: $($proc.Threads.Count)"
    }
} else {
    Write-Host "No Python processes found" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "Web interface: http://localhost:5001" -ForegroundColor Cyan
Write-Host ""
