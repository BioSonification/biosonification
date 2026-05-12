# Setup Windows Firewall for BioSonification HTTPS
# Run as Administrator

Write-Host "=== Windows Firewall Setup ===" -ForegroundColor Cyan
Write-Host ""

# Check Administrator
$currentPrincipal = New-Object Security.Principal.WindowsPrincipal([Security.Principal.WindowsIdentity]::GetCurrent())
if (-not $currentPrincipal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
    Write-Host "ERROR: This script must be run as Administrator" -ForegroundColor Red
    Write-Host "Right-click PowerShell and select 'Run as Administrator'" -ForegroundColor Yellow
    exit 1
}

Write-Host "Configuring firewall rules..." -ForegroundColor Cyan

# Remove existing rules if they exist
$existingRules = @("Block External 5001", "Allow Localhost 5001", "Caddy HTTP", "Caddy HTTPS")
foreach ($ruleName in $existingRules) {
    $rule = Get-NetFirewallRule -DisplayName $ruleName -ErrorAction SilentlyContinue
    if ($rule) {
        Write-Host "  Removing existing rule: $ruleName" -ForegroundColor Yellow
        Remove-NetFirewallRule -DisplayName $ruleName
    }
}

# Block external access to port 5001 (Waitress)
Write-Host "  Creating rule: Block External 5001" -ForegroundColor Cyan
New-NetFirewallRule -DisplayName "Block External 5001" `
    -Direction Inbound `
    -Protocol TCP `
    -LocalPort 5001 `
    -RemoteAddress Any `
    -Action Block `
    -Profile Any | Out-Null

# Allow localhost access to port 5001
Write-Host "  Creating rule: Allow Localhost 5001" -ForegroundColor Cyan
New-NetFirewallRule -DisplayName "Allow Localhost 5001" `
    -Direction Inbound `
    -Protocol TCP `
    -LocalPort 5001 `
    -RemoteAddress 127.0.0.1 `
    -Action Allow `
    -Profile Any | Out-Null

# Allow HTTP (port 80) for ACME challenge
Write-Host "  Creating rule: Caddy HTTP" -ForegroundColor Cyan
New-NetFirewallRule -DisplayName "Caddy HTTP" `
    -Direction Inbound `
    -Protocol TCP `
    -LocalPort 80 `
    -Action Allow `
    -Profile Any | Out-Null

# Allow HTTPS (port 443)
Write-Host "  Creating rule: Caddy HTTPS" -ForegroundColor Cyan
New-NetFirewallRule -DisplayName "Caddy HTTPS" `
    -Direction Inbound `
    -Protocol TCP `
    -LocalPort 443 `
    -Action Allow `
    -Profile Any | Out-Null

Write-Host ""
Write-Host "=== Firewall Configuration Complete ===" -ForegroundColor Green
Write-Host ""
Write-Host "Rules created:"
Get-NetFirewallRule -DisplayName "*5001*", "Caddy*" | Format-Table -Property DisplayName, Enabled, Direction, Action

Write-Host ""
Write-Host "IMPORTANT: You also need to configure Port Forwarding on your router:" -ForegroundColor Yellow
Write-Host "  1. Login to your router (usually 192.168.1.1 or 192.168.0.1)" -ForegroundColor Yellow
Write-Host "  2. Find Port Forwarding / Virtual Server / NAT section" -ForegroundColor Yellow
Write-Host "  3. Add these rules:" -ForegroundColor Yellow
Write-Host "     - External Port 80 -> Internal IP: <your-local-ip> -> Internal Port 80" -ForegroundColor Yellow
Write-Host "     - External Port 443 -> Internal IP: <your-local-ip> -> Internal Port 443" -ForegroundColor Yellow
Write-Host ""
Write-Host "Your local IP addresses:" -ForegroundColor Cyan
Get-NetIPAddress -AddressFamily IPv4 | Where-Object { $_.InterfaceAlias -notlike "*Loopback*" } | Select-Object IPAddress, InterfaceAlias
Write-Host ""
