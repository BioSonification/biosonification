# HTTPS Setup Guide for BioSonification

This guide explains how to set up HTTPS access to BioSonification using Caddy reverse proxy with automatic SSL certificates from Let's Encrypt.

## Prerequisites

- Windows 11
- BioSonification installed and running (Task Scheduler service)
- Domain name pointing to your public IP: `biosonification.ddns.net` → `84.237.53.139`
- Router with port forwarding capability
- Administrator access to Windows

## Architecture

```
Internet (HTTPS/443)
    ↓
Router Port Forwarding (80, 443)
    ↓
Windows Firewall (allow 80, 443; block external 5001)
    ↓
Caddy Server (Task Scheduler)
  - Reverse proxy
  - SSL termination (Let's Encrypt)
  - Auto-renewal
  - Security headers
    ↓ HTTP/localhost:5001
Waitress WSGI Server (Task Scheduler)
  - Production mode (DEBUG=False)
  - Multi-threaded
    ↓
Flask Application
```

## Step 1: Download Caddy

1. Go to https://caddyserver.com/download
2. Select:
   - Platform: **Windows**
   - Architecture: **amd64**
3. Download `caddy_windows_amd64.zip`
4. Extract `caddy.exe` to `C:\Tools\caddy\`

**Or use PowerShell:**
```powershell
Invoke-WebRequest -Uri "https://caddyserver.com/api/download?os=windows&arch=amd64" -OutFile "$env:USERPROFILE\Downloads\caddy.exe"
Move-Item "$env:USERPROFILE\Downloads\caddy.exe" "C:\Tools\caddy\caddy.exe"
```

**Verify installation:**
```powershell
C:\Tools\caddy\caddy.exe version
```

## Step 2: Configure Windows Firewall

Run as Administrator:
```powershell
cd C:\Users\vlasi\Documents\biosonification
.\scripts\setup-firewall.ps1
```

This script will:
- Block external access to port 5001 (Waitress)
- Allow localhost access to port 5001
- Open port 80 for HTTP (ACME challenge)
- Open port 443 for HTTPS

**Verify firewall rules:**
```powershell
Get-NetFirewallRule -DisplayName "*5001*", "Caddy*" | Format-Table -Property DisplayName, Enabled, Direction, Action
```

## Step 3: Configure Router Port Forwarding

You need to forward ports 80 and 443 from your router to your computer.

1. Find your local IP address:
   ```powershell
   Get-NetIPAddress -AddressFamily IPv4 | Where-Object { $_.InterfaceAlias -notlike "*Loopback*" }
   ```

2. Login to your router (usually `192.168.1.1` or `192.168.0.1`)

3. Find the Port Forwarding / Virtual Server / NAT section

4. Add these rules:
   - **Rule 1:**
     - Protocol: TCP
     - External Port: 80
     - Internal IP: `<your-local-ip>`
     - Internal Port: 80
   
   - **Rule 2:**
     - Protocol: TCP
     - External Port: 443
     - Internal IP: `<your-local-ip>`
     - Internal Port: 443

5. Save and apply settings

**Test port forwarding:**
From another network (mobile data or friend's network):
```bash
curl -I http://biosonification.ddns.net
```

## Step 4: Install Caddy Service

Run as Administrator:
```powershell
cd C:\Users\vlasi\Documents\biosonification
.\scripts\install-caddy-service.ps1
```

This will:
- Validate Caddyfile configuration
- Create wrapper script with auto-restart logic
- Register Caddy as Task Scheduler task
- Configure auto-start on system boot

## Step 5: Start Services

**Start BioSonification (if not running):**
```powershell
.\scripts\start-task.ps1
```

**Start Caddy:**
```powershell
Start-ScheduledTask -TaskName CaddyServer
```

**Wait 30-60 seconds** for Caddy to obtain SSL certificate from Let's Encrypt.

## Step 6: Verify HTTPS

**Check Caddy logs:**
```powershell
Get-Content C:\Tools\caddy\logs\service.log -Tail 20
Get-Content C:\Tools\caddy\logs\stderr.log -Tail 20
```

**Test locally:**
```powershell
curl http://localhost:5001/health
```

**Test HTTPS (from another network):**
```bash
curl https://biosonification.ddns.net/health
```

**Test in browser:**
1. Open https://biosonification.ddns.net
2. Check for green padlock (valid SSL)
3. Click padlock → View certificate
4. Verify issued by Let's Encrypt

## Monitoring

**Check status of both services:**
```powershell
.\scripts\monitor-production.ps1
```

This shows:
- Caddy status
- BioSonification status
- Local health check
- Public HTTPS health check
- SSL certificate expiry
- Recent errors
- GPU status

**Manual checks:**
```powershell
# Caddy status
Get-ScheduledTask -TaskName CaddyServer

# BioSonification status
Get-ScheduledTask -TaskName BioSonification

# Caddy logs
Get-Content C:\Tools\caddy\logs\service.log -Tail 50

# BioSonification logs
Get-Content logs\biosonification.log -Tail 50
```

## Management Commands

**Caddy:**
```powershell
# Start
Start-ScheduledTask -TaskName CaddyServer

# Stop
Stop-ScheduledTask -TaskName CaddyServer

# Restart
Stop-ScheduledTask -TaskName CaddyServer
Start-Sleep -Seconds 3
Start-ScheduledTask -TaskName CaddyServer

# Status
Get-ScheduledTask -TaskName CaddyServer
```

**BioSonification:**
```powershell
# Start
.\scripts\start-task.ps1

# Stop
.\scripts\stop-task.ps1

# Restart
.\scripts\restart-task.ps1

# Monitor
.\scripts\monitor-task.ps1
```

## SSL Certificate Auto-Renewal

Caddy automatically renews SSL certificates **30 days before expiration**. No manual intervention needed.

**Check certificate expiry:**
```powershell
.\scripts\monitor-production.ps1
```

**Manual renewal (if needed):**
```powershell
Stop-ScheduledTask -TaskName CaddyServer
Start-ScheduledTask -TaskName CaddyServer
```

## Troubleshooting

### Problem 1: Caddy can't obtain certificate

**Symptoms:**
- Error "acme: error: 403"
- "challenge failed"

**Solutions:**
1. Check DNS:
   ```powershell
   nslookup biosonification.ddns.net
   # Should return: 84.237.53.139
   ```

2. Check port 80 is accessible:
   ```bash
   # From another network
   curl -I http://biosonification.ddns.net
   ```

3. Check firewall rules:
   ```powershell
   Get-NetFirewallRule -DisplayName "Caddy*"
   ```

4. Check port forwarding on router

5. Check Caddy logs:
   ```powershell
   Get-Content C:\Tools\caddy\logs\stderr.log -Tail 50
   ```

6. Temporarily disable antivirus and test

### Problem 2: Timeout during generation

**Symptoms:**
- 504 Gateway Timeout
- Generation stops mid-process

**Solutions:**
1. Increase timeouts in `C:\Tools\caddy\Caddyfile`:
   ```caddyfile
   transport http {
       response_header_timeout 600s  # 10 minutes
       read_timeout 610s
   }
   ```

2. Restart Caddy:
   ```powershell
   Stop-ScheduledTask -TaskName CaddyServer
   Start-ScheduledTask -TaskName CaddyServer
   ```

### Problem 3: Certificate not renewing

**Symptoms:**
- Certificate expires
- Browser shows certificate error

**Solutions:**
1. Check Caddy logs:
   ```powershell
   Get-Content C:\Tools\caddy\logs\stderr.log | Select-String "certificate"
   ```

2. Restart Caddy:
   ```powershell
   Stop-ScheduledTask -TaskName CaddyServer
   Start-ScheduledTask -TaskName CaddyServer
   ```

3. Ensure port 80 is accessible (needed for ACME challenge)

### Problem 4: "Connection refused" from outside

**Symptoms:**
- Works locally but not from internet
- `curl https://biosonification.ddns.net` fails

**Solutions:**
1. Check port forwarding on router
2. Check public IP matches DNS:
   ```powershell
   # Your public IP
   Invoke-RestMethod -Uri "https://api.ipify.org"
   
   # DNS resolution
   nslookup biosonification.ddns.net
   ```

3. Test from mobile data (different network)

### Problem 5: Mixed content warnings

**Symptoms:**
- Browser shows "Not secure" warnings
- Some resources not loading

**Solutions:**
1. Ensure all links in HTML use relative paths or HTTPS
2. Check Flask is receiving correct headers:
   ```python
   # In web/app.py
   @app.before_request
   def log_request_info():
       app.logger.debug('Headers: %s', request.headers)
   ```

## Security Checklist

- [x] `.env` not in git (`.gitignore`)
- [x] Port 5001 blocked from external access
- [x] HTTPS with valid SSL certificate
- [x] Security headers enabled (HSTS, X-Frame-Options, etc.)
- [x] Logs with rotation
- [x] Auto-restart on failure
- [x] Production mode (DEBUG=0)

## Updating the Application

```powershell
# 1. Stop services
Stop-ScheduledTask -TaskName CaddyServer
.\scripts\stop-task.ps1

# 2. Update code
git pull

# 3. Update dependencies (if needed)
.\.venv\Scripts\pip install -r requirements.txt

# 4. Start services
.\scripts\start-task.ps1
Start-ScheduledTask -TaskName CaddyServer

# 5. Verify
.\scripts\monitor-production.ps1
```

## Files Created

- `C:\Tools\caddy\caddy.exe` - Caddy binary
- `C:\Tools\caddy\Caddyfile` - Caddy configuration
- `C:\Tools\caddy\run-caddy.ps1` - Wrapper script
- `scripts\setup-firewall.ps1` - Firewall configuration
- `scripts\install-caddy-service.ps1` - Service installation
- `scripts\monitor-production.ps1` - Monitoring script
- `.env` - Production configuration

## URLs

- **Public HTTPS:** https://biosonification.ddns.net
- **Local:** http://localhost:5001
- **Health check:** https://biosonification.ddns.net/health

## Support

For issues:
1. Check logs: `.\scripts\monitor-production.ps1`
2. Review this guide's Troubleshooting section
3. Check Caddy documentation: https://caddyserver.com/docs/
