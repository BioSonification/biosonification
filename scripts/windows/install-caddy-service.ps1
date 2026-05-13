# Install Caddy as Windows Service using Task Scheduler
# Run as Administrator

$ServiceName = "CaddyServer"
$CaddyExe = "C:\Tools\caddy\caddy.exe"
$CaddyFile = "C:\Tools\caddy\Caddyfile"
$WorkingDir = "C:\Tools\caddy"

Write-Host "=== Caddy Service Installation ===" -ForegroundColor Cyan
Write-Host ""

# Check Administrator
$currentPrincipal = New-Object Security.Principal.WindowsPrincipal([Security.Principal.WindowsIdentity]::GetCurrent())
if (-not $currentPrincipal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
    Write-Host "ERROR: This script must be run as Administrator" -ForegroundColor Red
    Write-Host "Right-click PowerShell and select 'Run as Administrator'" -ForegroundColor Yellow
    exit 1
}

# Check Caddy
if (-not (Test-Path $CaddyExe)) {
    Write-Host "ERROR: Caddy not found at $CaddyExe" -ForegroundColor Red
    Write-Host ""
    Write-Host "Please download Caddy:" -ForegroundColor Yellow
    Write-Host "  1. Go to https://caddyserver.com/download" -ForegroundColor Yellow
    Write-Host "  2. Select Platform: Windows, Architecture: amd64" -ForegroundColor Yellow
    Write-Host "  3. Download caddy_windows_amd64.zip" -ForegroundColor Yellow
    Write-Host "  4. Extract caddy.exe to C:\Tools\caddy\" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "Or use PowerShell to download:" -ForegroundColor Cyan
    Write-Host '  Invoke-WebRequest -Uri "https://caddyserver.com/api/download?os=windows&arch=amd64" -OutFile "$env:USERPROFILE\Downloads\caddy.exe"' -ForegroundColor Cyan
    Write-Host '  Move-Item "$env:USERPROFILE\Downloads\caddy.exe" "C:\Tools\caddy\caddy.exe"' -ForegroundColor Cyan
    exit 1
}

# Check Caddyfile
if (-not (Test-Path $CaddyFile)) {
    Write-Host "ERROR: Caddyfile not found at $CaddyFile" -ForegroundColor Red
    exit 1
}

# Test Caddy version
Write-Host "Caddy version:" -ForegroundColor Cyan
& $CaddyExe version

# Validate Caddyfile
Write-Host ""
Write-Host "Validating Caddyfile..." -ForegroundColor Cyan
$validateResult = & $CaddyExe validate --config $CaddyFile 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Caddyfile validation failed" -ForegroundColor Red
    Write-Host $validateResult
    exit 1
}
Write-Host "Caddyfile is valid" -ForegroundColor Green

# Create wrapper script
$WrapperScript = "$WorkingDir\run-caddy.ps1"
$WrapperContent = @"
# Caddy Service Wrapper
`$CaddyExe = "$CaddyExe"
`$CaddyFile = "$CaddyFile"
`$WorkingDir = "$WorkingDir"
`$LogFile = "`$WorkingDir\logs\service.log"

function Write-Log {
    param([string]`$Message)
    `$timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    "`$timestamp - `$Message" | Out-File -FilePath `$LogFile -Append
    Write-Host `$Message
}

Write-Log "=== Caddy Server Starting ==="

while (`$true) {
    try {
        Write-Log "Starting Caddy..."

        `$process = Start-Process -FilePath `$CaddyExe ``
            -ArgumentList "run", "--config", `$CaddyFile ``
            -WorkingDirectory `$WorkingDir ``
            -NoNewWindow ``
            -PassThru ``
            -RedirectStandardOutput "`$WorkingDir\logs\stdout.log" ``
            -RedirectStandardError "`$WorkingDir\logs\stderr.log"

        Write-Log "Caddy started (PID: `$(`$process.Id))"

        `$process.WaitForExit()

        `$exitCode = `$process.ExitCode
        Write-Log "Caddy exited with code: `$exitCode"

        if (`$exitCode -eq 0) {
            Write-Log "Clean shutdown, exiting"
            break
        }

        Write-Log "Unexpected exit, restarting in 5 seconds..."
        Start-Sleep -Seconds 5

    } catch {
        Write-Log "ERROR: `$(`$_.Exception.Message)"
        Write-Log "Restarting in 5 seconds..."
        Start-Sleep -Seconds 5
    }
}

Write-Log "=== Caddy Server Stopped ==="
"@

Set-Content -Path $WrapperScript -Value $WrapperContent
Write-Host ""
Write-Host "Created wrapper script: $WrapperScript" -ForegroundColor Green

# Remove existing task
$existingTask = Get-ScheduledTask -TaskName $ServiceName -ErrorAction SilentlyContinue
if ($existingTask) {
    Write-Host ""
    Write-Host "WARNING: Task '$ServiceName' already exists" -ForegroundColor Yellow
    $response = Read-Host "Do you want to reinstall? (y/N)"
    if ($response -ne 'y' -and $response -ne 'Y') {
        Write-Host "Installation cancelled"
        exit 0
    }
    Write-Host "Removing existing task..." -ForegroundColor Yellow
    Unregister-ScheduledTask -TaskName $ServiceName -Confirm:$false
}

# Create scheduled task
Write-Host ""
Write-Host "Creating scheduled task..." -ForegroundColor Cyan

$action = New-ScheduledTaskAction `
    -Execute "PowerShell.exe" `
    -Argument "-NoProfile -ExecutionPolicy Bypass -File `"$WrapperScript`"" `
    -WorkingDirectory $WorkingDir

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
    -Description "Caddy reverse proxy for BioSonification with automatic HTTPS" | Out-Null

Write-Host ""
Write-Host "=== Task Installed Successfully ===" -ForegroundColor Green
Write-Host ""
Write-Host "Management commands:"
Write-Host "  Start:   Start-ScheduledTask -TaskName $ServiceName" -ForegroundColor Cyan
Write-Host "  Stop:    Stop-ScheduledTask -TaskName $ServiceName" -ForegroundColor Cyan
Write-Host "  Status:  Get-ScheduledTask -TaskName $ServiceName" -ForegroundColor Cyan
Write-Host "  Logs:    Get-Content C:\Tools\caddy\logs\service.log -Tail 20" -ForegroundColor Cyan
Write-Host ""
Write-Host "IMPORTANT: Before starting Caddy, make sure:" -ForegroundColor Yellow
Write-Host "  1. Port forwarding is configured on your router (ports 80 and 443)" -ForegroundColor Yellow
Write-Host "  2. Windows Firewall rules are set (run .\scripts\setup-firewall.ps1)" -ForegroundColor Yellow
Write-Host "  3. BioSonification is running on localhost:5001" -ForegroundColor Yellow
Write-Host ""
Write-Host "To start now: Start-ScheduledTask -TaskName $ServiceName" -ForegroundColor Cyan
Write-Host ""
