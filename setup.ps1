# ============================================================
#  DSP Hand Gesture Controller - Setup & Launcher
#  Usage: irm https://raw.githubusercontent.com/vk566/dsp-gesture-controller/main/setup.ps1 | iex
# ============================================================

$REPO_RAW    = "https://raw.githubusercontent.com/vk566/dsp-gesture-controller/main"
$INSTALL_DIR = "$env:USERPROFILE\DSPGestureController"
$LAUNCHER    = "$INSTALL_DIR\launcher.ps1"

# ── Colours ─────────────────────────────────────────────────
function Write-Green  { param($m) Write-Host $m -ForegroundColor Green }
function Write-Cyan   { param($m) Write-Host $m -ForegroundColor Cyan }
function Write-Yellow { param($m) Write-Host $m -ForegroundColor Yellow }
function Write-Red    { param($m) Write-Host $m -ForegroundColor Red }

# ── Menu Function ────────────────────────────────────────────
function Show-Menu {
    Clear-Host
    Write-Cyan "============================================"
    Write-Cyan "         DSP GESTURE CONTROLLER"
    Write-Cyan "============================================"
    Write-Host "  [1]  Record Gestures"
    Write-Host "  [2]  Run Controller (PPT)"
    Write-Host "  [3]  Exit"
    Write-Cyan "============================================"
    Write-Host ""
}

# ── Already installed? Skip setup ───────────────────────────
$alreadyInstalled = (Test-Path "$INSTALL_DIR\collect_gestures.py") -and
                    (Test-Path "$INSTALL_DIR\dsp_hand_gesture_ppt.py") -and
                    (Test-Path "$INSTALL_DIR\hand_landmarker.task")

if (-not $alreadyInstalled) {

    Clear-Host
    Write-Cyan "============================================"
    Write-Cyan "   DSP Hand Gesture Controller - Setup"
    Write-Cyan "============================================"
    Write-Host ""

    # ── 1. Create install folder ─────────────────────────────
    if (-not (Test-Path $INSTALL_DIR)) {
        New-Item -ItemType Directory -Path $INSTALL_DIR | Out-Null
        Write-Green "[+] Created folder: $INSTALL_DIR"
    }
    Set-Location $INSTALL_DIR

    # ── 2. Check Python & Auto Install if missing ────────────
    Write-Host ""
    Write-Cyan "[*] Checking Python..."

    $pythonCmd = $null
    foreach ($cmd in @("python","python3","py")) {
        try {
            $ver = & $cmd --version 2>&1
            if ($ver -match "Python") {
                $pythonCmd = $cmd
                Write-Green "[+] Found: $ver (command: $cmd)"
                break
            }
        } catch {}
    }

    if (-not $pythonCmd) {
        Write-Yellow "[!] Python not found. Auto-installing Python 3.12..."
        Write-Host ""

        $pyInstaller = "$env:TEMP\python-3.12.0-amd64.exe"
        $pyURL       = "https://www.python.org/ftp/python/3.12.0/python-3.12.0-amd64.exe"

        Write-Cyan "[*] Downloading Python 3.12.0 installer..."
        Invoke-WebRequest -Uri $pyURL -OutFile $pyInstaller
        Write-Green "[+] Download complete."

        Write-Cyan "[*] Installing Python 3.12.0 silently..."
        Start-Process -FilePath $pyInstaller -ArgumentList `
            "/quiet InstallAllUsers=0 PrependPath=1 Include_pip=1 Include_launcher=1" `
            -Wait
        Write-Green "[+] Python installed!"

        # ── Refresh PATH so python is found immediately ──────
        $userPath    = [System.Environment]::GetEnvironmentVariable("PATH","User")
        $machinePath = [System.Environment]::GetEnvironmentVariable("PATH","Machine")
        $env:PATH    = "$userPath;$machinePath"

        # ── Also add Python paths manually just in case ──────
        $pyPaths = @(
            "$env:LOCALAPPDATA\Programs\Python\Python312",
            "$env:LOCALAPPDATA\Programs\Python\Python312\Scripts"
        )
        foreach ($p in $pyPaths) {
            if (Test-Path $p) {
                $env:PATH = "$p;$env:PATH"
                Write-Green "[+] Added to PATH: $p"
            }
        }

        # ── Verify install ───────────────────────────────────
        try {
            $ver = python --version 2>&1
            Write-Green "[+] Verified: $ver"
            $pythonCmd = "python"
        } catch {
            Write-Red "[!] Python install failed. Please install manually from https://python.org"
            pause; exit 1
        }

        # Cleanup installer
        Remove-Item $pyInstaller -Force -ErrorAction SilentlyContinue
    }

    # ── 3. Download project files ────────────────────────────
    Write-Host ""
    Write-Cyan "[*] Downloading project files from GitHub..."
    foreach ($file in @("collect_gestures.py","dsp_hand_gesture_ppt.py","hand_landmarker.task")) {
        try {
            Invoke-WebRequest -Uri "$REPO_RAW/$file" -OutFile "$INSTALL_DIR\$file" -ErrorAction Stop
            Write-Green "[+] Downloaded: $file"
        } catch {
            Write-Red "[!] Failed to download: $file"
        }
    }

    # ── 4. Install Python dependencies ───────────────────────
    Write-Host ""
    Write-Cyan "[*] Installing Python dependencies..."
    foreach ($dep in @("opencv-python","mediapipe","numpy","imageio","pyautogui")) {
        Write-Host "    Installing $dep ..."
        & $pythonCmd -m pip install $dep --quiet
    }
    Write-Green "[+] All dependencies installed."

    # ── 5. Create launcher.ps1 ───────────────────────────────
    $launcherCode = @'
$INSTALL_DIR = "$env:USERPROFILE\DSPGestureController"
Set-Location $INSTALL_DIR
# Ensure Python is in PATH
$pyPaths = @(
    "$env:LOCALAPPDATA\Programs\Python\Python312",
    "$env:LOCALAPPDATA\Programs\Python\Python312\Scripts"
)
foreach ($p in $pyPaths) {
    if (Test-Path $p) { $env:PATH = "$p;$env:PATH" }
}
function Write-Green  { param($m) Write-Host $m -ForegroundColor Green }
function Write-Cyan   { param($m) Write-Host $m -ForegroundColor Cyan }
function Write-Yellow { param($m) Write-Host $m -ForegroundColor Yellow }
function Write-Red    { param($m) Write-Host $m -ForegroundColor Red }
while ($true) {
    Clear-Host
    Write-Cyan "============================================"
    Write-Cyan "         DSP GESTURE CONTROLLER"
    Write-Cyan "============================================"
    Write-Host "  [1]  Record Gestures"
    Write-Host "  [2]  Run Controller (PPT)"
    Write-Host "  [3]  Exit"
    Write-Cyan "============================================"
    Write-Host ""
    $choice = Read-Host "Enter your choice (1/2/3)"
    switch ($choice) {
        "1" { Write-Green "`n[>] Launching Recorder..."; Write-Yellow "    R=Record | S=Save | Q=Quit"; python "$INSTALL_DIR\collect_gestures.py" }
        "2" { Write-Green "`n[>] Launching PPT Controller..."; Write-Yellow "    Press Q to quit."; python "$INSTALL_DIR\dsp_hand_gesture_ppt.py" }
        "3" { Write-Cyan "`nGoodbye!"; exit 0 }
        default { Write-Red "Invalid! Enter 1, 2 or 3." }
    }
    Write-Host ""
    Read-Host "Press Enter to return to menu..."
}
'@
    $launcherCode | Out-File -FilePath $LAUNCHER -Encoding UTF8
    Write-Green "[+] Launcher created."

    # ── 6. Create Desktop Shortcut ───────────────────────────
    Write-Host ""
    Write-Cyan "[*] Creating desktop shortcut..."
    $desktopPath  = [System.Environment]::GetFolderPath("Desktop")
    $shortcutPath = "$desktopPath\DSP Gesture Controller.lnk"
    $shell        = New-Object -ComObject WScript.Shell
    $shortcut     = $shell.CreateShortcut($shortcutPath)
    $shortcut.TargetPath      = "powershell.exe"
    $shortcut.Arguments       = "-ExecutionPolicy Bypass -File `"$LAUNCHER`""
    $shortcut.WorkingDirectory = $INSTALL_DIR
    $shortcut.WindowStyle     = 1
    $shortcut.Description     = "DSP Hand Gesture Controller"
    $shortcut.Save()
    Write-Green "[+] Desktop shortcut created!"

    Write-Host ""
    Write-Green "============================================"
    Write-Green "   Setup Complete! Shortcut on Desktop!"
    Write-Green "============================================"
    Write-Host ""
    Start-Sleep -Seconds 2
}

Set-Location $INSTALL_DIR

# ── Run Menu ─────────────────────────────────────────────────
while ($true) {
    Show-Menu
    $choice = Read-Host "Enter your choice (1/2/3)"
    switch ($choice) {
        "1" {
            Write-Green "`n[>] Launching Recorder..."
            Write-Yellow "    R=Record | S=Save | Q=Quit"
            python "$INSTALL_DIR\collect_gestures.py"
        }
        "2" {
            Write-Green "`n[>] Launching PPT Controller..."
            Write-Yellow "    Press Q to quit."
            python "$INSTALL_DIR\dsp_hand_gesture_ppt.py"
        }
        "3" { Write-Cyan "`nGoodbye!"; exit 0 }
        default { Write-Red "Invalid! Enter 1, 2 or 3." }
    }
    Write-Host ""
    Read-Host "Press Enter to return to menu..."
}