# Monitor both Caddy and BioSonification

Write-Host "=== Production Monitoring ===" -ForegroundColor Cyan
Write-Host ""

# Caddy status
Write-Host "=== Caddy Server ===" -ForegroundColor Cyan
$caddyTask = Get-ScheduledTask -TaskName "CaddyServer" -ErrorAction SilentlyContinue
if ($caddyTask) {
    Write-Host "Status: $($caddyTask.State)" -ForegroundColor $(if ($caddyTask.State -eq "Running") { "Green" } else { "Red" })
} else {
    Write-Host "Status: NOT INSTALLED" -ForegroundColor Red
}

# BioSonification status
Write-Host ""
Write-Host "=== BioSonification ===" -ForegroundColor Cyan
$bioTask = Get-ScheduledTask -TaskName "BioSonification" -ErrorAction SilentlyContinue
if ($bioTask) {
    Write-Host "Status: $($bioTask.State)" -ForegroundColor $(if ($bioTask.State -eq "Running") { "Green" } else { "Red" })
} else {
    Write-Host "Status: NOT INSTALLED" -ForegroundColor Red
}

# Health check - local
Write-Host ""
Write-Host "=== Health Check (Local) ===" -ForegroundColor Cyan
try {
    $response = Invoke-WebRequest -Uri "http://localhost:5001/health" -UseBasicParsing -TimeoutSec 5
    $health = $response.Content | ConvertFrom-Json
    Write-Host "Status: OK" -ForegroundColor Green
    Write-Host "  Generator Ready: $($health.generator_ready)"
    Write-Host "  GPU Available: $($health.gpu_available)"
} catch {
    Write-Host "Status: FAILED" -ForegroundColor Red
    Write-Host "  Error: $($_.Exception.Message)" -ForegroundColor Yellow
}

# Health check - public HTTPS
Write-Host ""
Write-Host "=== Health Check (Public HTTPS) ===" -ForegroundColor Cyan
try {
    $response = Invoke-WebRequest -Uri "https://biosonification.ddns.net/health" -UseBasicParsing -TimeoutSec 10
    Write-Host "Status: OK" -ForegroundColor Green
} catch {
    Write-Host "Status: FAILED" -ForegroundColor Red
    Write-Host "  Error: $($_.Exception.Message)" -ForegroundColor Yellow
}

# SSL Certificate
Write-Host ""
Write-Host "=== SSL Certificate ===" -ForegroundColor Cyan
try {
    $req = [Net.HttpWebRequest]::Create("https://biosonification.ddns.net")
    $req.GetResponse() | Out-Null
    $cert = $req.ServicePoint.Certificate
    $expiryDate = [DateTime]::Parse($cert.GetExpirationDateString())
    $daysLeft = ($expiryDate - (Get-Date)).Days
    Write-Host "Expires: $expiryDate" -ForegroundColor $(if ($daysLeft -lt 30) { "Yellow" } else { "Green" })
    Write-Host "Days left: $daysLeft" -ForegroundColor $(if ($daysLeft -lt 30) { "Yellow" } else { "Green" })

    if ($daysLeft -lt 7) {
        Write-Host "WARNING: Certificate expires soon!" -ForegroundColor Red
    }
} catch {
    Write-Host "Could not check certificate" -ForegroundColor Yellow
    Write-Host "  (This is normal if Caddy hasn't obtained a certificate yet)" -ForegroundColor Gray
}

# Recent errors
Write-Host ""
Write-Host "=== Recent Errors ===" -ForegroundColor Cyan
$hasErrors = $false

if (Test-Path "C:\Tools\caddy\logs\stderr.log") {
    $caddyErrors = Get-Content "C:\Tools\caddy\logs\stderr.log" -Tail 3 -ErrorAction SilentlyContinue
    if ($caddyErrors) {
        Write-Host "Caddy:" -ForegroundColor Yellow
        $caddyErrors
        $hasErrors = $true
    }
}

if (Test-Path "C:\Users\vlasi\Documents\biosonification\logs\stderr.log") {
    $bioErrors = Get-Content "C:\Users\vlasi\Documents\biosonification\logs\stderr.log" -Tail 3 -ErrorAction SilentlyContinue
    if ($bioErrors) {
        Write-Host "BioSonification:" -ForegroundColor Yellow
        $bioErrors
        $hasErrors = $true
    }
}

if (-not $hasErrors) {
    Write-Host "No recent errors" -ForegroundColor Green
}

# GPU status (if available)
if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) {
    Write-Host ""
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
}

Write-Host ""
Write-Host "Public URL: https://biosonification.ddns.net" -ForegroundColor Cyan
Write-Host "Local URL: http://localhost:5001" -ForegroundColor Cyan
Write-Host ""
