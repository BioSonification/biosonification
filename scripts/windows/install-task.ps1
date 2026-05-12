# Install BioSonification using Windows Task Scheduler
# Alternative to NSSM - uses built-in Windows features
# Run this script as Administrator

$ServiceName = "BioSonification"
$ProjectRoot = $PSScriptRoot | Split-Path -Parent
$PythonExe = "$ProjectRoot\.venv\Scripts\python.exe"
$AppScript = "$ProjectRoot\web\wsgi.py"
$LogDir = "$ProjectRoot\logs"

Write-Host "=== BioSonification Task Scheduler Installation ===" -ForegroundColor Cyan
Write-Host ""
Write-Host "Project Root: $ProjectRoot"
Write-Host "Python: $PythonExe"
Write-Host "App Script: $AppScript"
Write-Host ""

# Check if running as Administrator
$currentPrincipal = New-Object Security.Principal.WindowsPrincipal([Security.Principal.WindowsIdentity]::GetCurrent())
if (-not $currentPrincipal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
    Write-Host "ERROR: This script must be run as Administrator" -ForegroundColor Red
    Write-Host "Right-click PowerShell and select 'Run as Administrator'" -ForegroundColor Yellow
    exit 1
}

# Check if Python exists
if (-not (Test-Path $PythonExe)) {
    Write-Host "ERROR: Python not found at $PythonExe" -ForegroundColor Red
    Write-Host "Please ensure virtual environment is created" -ForegroundColor Yellow
    exit 1
}

# Create logs directory
New-Item -ItemType Directory -Path $LogDir -Force | Out-Null

# Create wrapper script that handles restart logic
$WrapperScript = "$ProjectRoot\scripts\run-service.ps1"
$WrapperContent = @"
# BioSonification Service Wrapper
# This script runs the application and restarts it on failure

`$ProjectRoot = "$ProjectRoot"
`$PythonExe = "$PythonExe"
`$AppScript = "$AppScript"
`$LogFile = "$LogDir\service.log"

function Write-Log {
    param([string]`$Message)
    `$timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    "`$timestamp - `$Message" | Out-File -FilePath `$LogFile -Append
    Write-Host `$Message
}

Write-Log "=== BioSonification Service Starting ==="
Write-Log "Python: `$PythonExe"
Write-Log "Script: `$AppScript"

# Infinite loop with restart logic
while (`$true) {
    try {
        Write-Log "Starting application..."

        # Start the application
        `$process = Start-Process -FilePath `$PythonExe ``
            -ArgumentList `$AppScript ``
            -WorkingDirectory `$ProjectRoot ``
            -NoNewWindow ``
            -PassThru ``
            -RedirectStandardOutput "$LogDir\stdout.log" ``
            -RedirectStandardError "$LogDir\stderr.log"

        Write-Log "Application started (PID: `$(`$process.Id))"

        # Wait for process to exit
        `$process.WaitForExit()

        `$exitCode = `$process.ExitCode
        Write-Log "Application exited with code: `$exitCode"

        # If exit code is 0, it was intentional shutdown
        if (`$exitCode -eq 0) {
            Write-Log "Clean shutdown detected, exiting service"
            break
        }

        # Otherwise, restart after delay
        Write-Log "Unexpected exit, restarting in 5 seconds..."
        Start-Sleep -Seconds 5

    } catch {
        Write-Log "ERROR: `$(`$_.Exception.Message)"
        Write-Log "Restarting in 5 seconds..."
        Start-Sleep -Seconds 5
    }
}

Write-Log "=== BioSonification Service Stopped ==="
"@

Set-Content -Path $WrapperScript -Value $WrapperContent
Write-Host "Created wrapper script: $WrapperScript" -ForegroundColor Green

# Remove existing task if it exists
$existingTask = Get-ScheduledTask -TaskName $ServiceName -ErrorAction SilentlyContinue
if ($existingTask) {
    Write-Host "WARNING: Task '$ServiceName' already exists" -ForegroundColor Yellow
    $response = Read-Host "Do you want to reinstall? (y/N)"
    if ($response -ne 'y' -and $response -ne 'Y') {
        Write-Host "Installation cancelled"
        exit 0
    }
    Write-Host "Removing existing task..."
    Unregister-ScheduledTask -TaskName $ServiceName -Confirm:$false
}

# Create scheduled task
Write-Host "Creating scheduled task..." -ForegroundColor Cyan

$action = New-ScheduledTaskAction `
    -Execute "PowerShell.exe" `
    -Argument "-NoProfile -ExecutionPolicy Bypass -File `"$WrapperScript`"" `
    -WorkingDirectory $ProjectRoot

$trigger = New-ScheduledTaskTrigger -AtStartup

$settings = New-ScheduledTaskSettingsSet `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -StartWhenAvailable `
    -RestartCount 3 `
    -RestartInterval (New-TimeSpan -Minutes 1) `
    -ExecutionTimeLimit (New-TimeSpan -Days 365)

$principal = New-ScheduledTaskPrincipal `
    -UserId "SYSTEM" `
    -LogonType ServiceAccount `
    -RunLevel Highest

Register-ScheduledTask `
    -TaskName $ServiceName `
    -Action $action `
    -Trigger $trigger `
    -Settings $settings `
    -Principal $principal `
    -Description "BioSonification Web Service - Generates music from biological sequences"

Write-Host ""
Write-Host "=== Task Installed Successfully ===" -ForegroundColor Green
Write-Host ""
Write-Host "Management commands:"
Write-Host "  Start:   .\scripts\start-task.ps1"
Write-Host "  Stop:    .\scripts\stop-task.ps1"
Write-Host "  Restart: .\scripts\restart-task.ps1"
Write-Host "  Monitor: .\scripts\monitor-task.ps1"
Write-Host ""
Write-Host "Or use Task Scheduler directly:"
Write-Host "  Start-ScheduledTask -TaskName $ServiceName"
Write-Host "  Stop-ScheduledTask -TaskName $ServiceName"
Write-Host "  Get-ScheduledTask -TaskName $ServiceName"
Write-Host ""
Write-Host "To uninstall: .\scripts\uninstall-task.ps1"
Write-Host ""
Write-Host "The task will start automatically on next system boot."
Write-Host "To start now, run: .\scripts\start-task.ps1"
Write-Host ""
