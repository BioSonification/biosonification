# BioSonification Service Wrapper
# This script runs the application and restarts it on failure

$ProjectRoot = "C:\Users\vlasi\Documents\biosonification"
$PythonExe = "C:\Users\vlasi\Documents\biosonification\.venv\Scripts\python.exe"
$AppScript = "C:\Users\vlasi\Documents\biosonification\web\wsgi.py"
$LogFile = "C:\Users\vlasi\Documents\biosonification\logs\service.log"

function Write-Log {
    param([string]$Message)
    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    "$timestamp - $Message" | Out-File -FilePath $LogFile -Append
    Write-Host $Message
}

Write-Log "=== BioSonification Service Starting ==="
Write-Log "Python: $PythonExe"
Write-Log "Script: $AppScript"

# Infinite loop with restart logic
while ($true) {
    try {
        Write-Log "Starting application..."

        # Start the application
        $process = Start-Process -FilePath $PythonExe `
            -ArgumentList $AppScript `
            -WorkingDirectory $ProjectRoot `
            -NoNewWindow `
            -PassThru `
            -RedirectStandardOutput "C:\Users\vlasi\Documents\biosonification\logs\stdout.log" `
            -RedirectStandardError "C:\Users\vlasi\Documents\biosonification\logs\stderr.log"

        Write-Log "Application started (PID: $($process.Id))"

        # Wait for process to exit
        $process.WaitForExit()

        $exitCode = $process.ExitCode
        Write-Log "Application exited with code: $exitCode"

        # If exit code is 0, it was intentional shutdown
        if ($exitCode -eq 0) {
            Write-Log "Clean shutdown detected, exiting service"
            break
        }

        # Otherwise, restart after delay
        Write-Log "Unexpected exit, restarting in 5 seconds..."
        Start-Sleep -Seconds 5

    } catch {
        Write-Log "ERROR: $($_.Exception.Message)"
        Write-Log "Restarting in 5 seconds..."
        Start-Sleep -Seconds 5
    }
}

Write-Log "=== BioSonification Service Stopped ==="
