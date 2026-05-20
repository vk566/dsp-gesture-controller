# ============================================================
#  DSP Hand Gesture Controller - Setup & Launcher
#  Usage: irm https://raw.githubusercontent.com/vk566/dsp-gesture-controller/main/setup.ps1 | iex
# ============================================================

$REPO_RAW    = "https://raw.githubusercontent.com/vk566/dsp-gesture-controller/main"
$INSTALL_DIR = "$env:USERPROFILE\DSPGestureController"
$LAUNCHER    = "$INSTALL_DIR\launcher.ps1"
$PY312_PATH  = "$env:LOCALAPPDATA\Programs\Python\Python312\python.exe"
$PY312_URL   = "https://www.python.org/ftp/python/3.12.0/python-3.12.0-amd64.exe"

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

    # ── 2. Check if Python 3.12 specifically is installed ────
    Write-Host ""
    Write-Cyan "[*] Checking for Python 3.12..."

    if (Test-Path $PY312_PATH) {
        $ver = & $PY312_PATH --version 2>&1
        Write-Green "[+] Python 3.12 already installed: $ver"
    } else {
        Write-Yellow "[!] Python 3.12 not found. Installing now..."
        Write-Host ""

        $pyInstaller = "$env:TEMP\python-3.12.0-amd64.exe"

        Write-Cyan "[*] Downloading Python 3.12.0..."
        Invoke-WebRequest -Uri $PY312_URL -OutFile $pyInstaller
        Write-Green "[+] Download complete."

        Write-Cyan "[*] Installing Python 3.12.0 silently..."
        Start-Process -FilePath $pyInstaller -ArgumentList `
            "/quiet InstallAllUsers=0 PrependPath=0 Include_pip=1 Include_launcher=1" `
            -Wait
        Write-Green "[+] Python 3.12 installed!"

        # Cleanup installer
        Remove-Item $pyInstaller -Force -ErrorAction SilentlyContinue

        # Verify
        if (Test-Path $PY312_PATH) {
            $ver = & $PY312_PATH --version 2>&1
            Write-Green "[+] Verified: $ver"
        } else {
            Write-Red "[!] Python 3.12 install failed. Please install manually from https://python.org"
            pause; exit 1
        }
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

    # ── 4. Install Python dependencies using Python 3.12 ─────
    Write-Host ""
    Write-Cyan "[*] Installing Python dependencies with Python 3.12..."
    foreach ($dep in @("opencv-python","mediapipe","numpy","imageio","pyautogui")) {
        Write-Host "    Installing $dep ..."
        & $PY312_PATH -m pip install $dep --quiet
    }
    Write-Green "[+] All dependencies installed."

    # ── 5. Create launcher.ps1 ───────────────────────────────
    $launcherCode = @'
$INSTALL_DIR = "$env:USERPROFILE\DSPGestureController"
$PY312 = "$env:LOCALAPPDATA\Programs\Python\Python312\python.exe"
Set-Location $INSTALL_DIR
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
        "1" { Write-Green "`n[>] Launching Recorder..."; Write-Yellow "    R=Record | S=Save | Q=Quit"; & $PY312 "$INSTALL_DIR\collect_gestures.py" }
        "2" { Write-Green "`n[>] Launching PPT Controller..."; Write-Yellow "    Press Q to quit."; & $PY312 "$INSTALL_DIR\dsp_hand_gesture_ppt.py" }
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
    $shortcut.TargetPath       = "powershell.exe"
    $shortcut.Arguments        = "-ExecutionPolicy Bypass -File `"$LAUNCHER`""
    $shortcut.WorkingDirectory = $INSTALL_DIR
    $shortcut.WindowStyle      = 1
    $shortcut.Description      = "DSP Hand Gesture Controller"
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
    $PY312 = "$env:LOCALAPPDATA\Programs\Python\Python312\python.exe"
    $choice = Read-Host "Enter your choice (1/2/3)"
    switch ($choice) {
        "1" {
            Write-Green "`n[>] Launching Recorder..."
            Write-Yellow "    R=Record | S=Save | Q=Quit"
            & $PY312 "$INSTALL_DIR\collect_gestures.py"
        }
        "2" {
            Write-Green "`n[>] Launching PPT Controller..."
            Write-Yellow "    Press Q to quit."
            & $PY312 "$INSTALL_DIR\dsp_hand_gesture_ppt.py"
        }
        "3" { Write-Cyan "`nGoodbye!"; exit 0 }
        default { Write-Red "Invalid! Enter 1, 2 or 3." }
    }
    Write-Host ""
    Read-Host "Press Enter to return to menu..."
}