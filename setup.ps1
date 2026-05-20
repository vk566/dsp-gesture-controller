# ============================================================
#  DSP Hand Gesture Controller - Setup & Launcher
#  Usage: irm https://raw.githubusercontent.com/vk566/dsp-gesture-controller/main/setup.ps1 | iex
# ============================================================

$REPO_RAW  = "https://raw.githubusercontent.com/vk566/dsp-gesture-controller/main"
$INSTALL_DIR = "$env:USERPROFILE\DSPGestureController"
$LAUNCHER  = "$INSTALL_DIR\launcher.ps1"

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

# ── Already installed? Skip setup, go straight to menu ──────
$alreadyInstalled = (Test-Path "$INSTALL_DIR\collect_gestures.py") -and
                    (Test-Path "$INSTALL_DIR\dsp_hand_gesture_ppt.py") -and
                    (Test-Path "$INSTALL_DIR\hand_landmarker.task")

if ($alreadyInstalled) {
    Write-Green "Already installed! Launching menu..."
    Set-Location $INSTALL_DIR
} else {

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

    # ── 2. Check Python ──────────────────────────────────────
    Write-Host ""
    Write-Cyan "[*] Checking Python..."
    try {
        $pyver = python --version 2>&1
        Write-Green "[+] Found: $pyver"
    } catch {
        Write-Red "[!] Python not found. Please install Python and re-run."
        pause; exit 1
    }

    # ── 3. Download files ────────────────────────────────────
    Write-Host ""
    Write-Cyan "[*] Downloading project files..."
    foreach ($file in @("collect_gestures.py","dsp_hand_gesture_ppt.py","hand_landmarker.task")) {
        try {
            Invoke-WebRequest -Uri "$REPO_RAW/$file" -OutFile "$INSTALL_DIR\$file" -ErrorAction Stop
            Write-Green "[+] Downloaded: $file"
        } catch {
            Write-Red "[!] Failed: $file"
        }
    }

    # ── 4. Install dependencies ──────────────────────────────
    Write-Host ""
    Write-Cyan "[*] Installing Python dependencies..."
    foreach ($dep in @("opencv-python","mediapipe","numpy","imageio","pyautogui")) {
        Write-Host "    Installing $dep ..."
        python -m pip install $dep --quiet
    }
    Write-Green "[+] All dependencies installed."

    # ── 5. Create launcher.ps1 ───────────────────────────────
    $launcherCode = @'
$INSTALL_DIR = "$env:USERPROFILE\DSPGestureController"
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
        "1" { Write-Green "`n[>] Launching Recorder..."; Write-Yellow "    R=Record | S=Save | Q=Quit"; python "$INSTALL_DIR\collect_gestures.py" }
        "2" { Write-Green "`n[>] Launching PPT Controller..."; Write-Yellow "    Press Q to quit."; python "$INSTALL_DIR\dsp_hand_gesture_ppt.py" }
        "3" { Write-Cyan "`nGoodbye!"; exit 0 }
        default { Write-Red "Invalid! Enter 1, 2 or 3." }
    }
    Write-Host ""; Read-Host "Press Enter to return to menu..."
}
'@
    $launcherCode | Out-File -FilePath $LAUNCHER -Encoding UTF8
    Write-Green "[+] Launcher created."

    # ── 6. Create Desktop Shortcut ───────────────────────────
    Write-Host ""
    Write-Cyan "[*] Creating desktop shortcut..."
    $desktopPath = [System.Environment]::GetFolderPath("Desktop")
    $shortcutPath = "$desktopPath\DSP Gesture Controller.lnk"
    $shell = New-Object -ComObject WScript.Shell
    $shortcut = $shell.CreateShortcut($shortcutPath)
    $shortcut.TargetPath = "powershell.exe"
    $shortcut.Arguments = "-ExecutionPolicy Bypass -File `"$LAUNCHER`""
    $shortcut.WorkingDirectory = $INSTALL_DIR
    $shortcut.WindowStyle = 1
    $shortcut.Description = "DSP Hand Gesture Controller"
    $shortcut.Save()
    Write-Green "[+] Desktop shortcut created: 'DSP Gesture Controller'"

    Write-Host ""
    Write-Green "============================================"
    Write-Green "   Setup Complete! Shortcut on Desktop!"
    Write-Green "============================================"
    Write-Host ""
    Start-Sleep -Seconds 2
}

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