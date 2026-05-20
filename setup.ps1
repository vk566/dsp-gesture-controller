# ============================================================
#  DSP Hand Gesture Controller - Setup & Launcher
#  Usage: irm https://raw.githubusercontent.com/vk566/dsp-gesture-controller/main/setup.ps1 | iex
# ============================================================

$REPO_RAW     = "https://raw.githubusercontent.com/vk566/dsp-gesture-controller/main"
$INSTALL_DIR  = "$env:USERPROFILE\DSPGestureController"
$LAUNCHER     = "$INSTALL_DIR\launcher.ps1"
$PY312_PATH   = "$env:LOCALAPPDATA\Programs\Python\Python312\python.exe"
$PY312_URL    = "https://www.python.org/ftp/python/3.12.0/python-3.12.0-amd64.exe"
$VERSION_FILE = "$INSTALL_DIR\version.txt"
$VERSION_URL  = "$REPO_RAW/version.txt"

# ── Colours ─────────────────────────────────────────────────
function Write-Green  { param($m) Write-Host $m -ForegroundColor Green }
function Write-Cyan   { param($m) Write-Host $m -ForegroundColor Cyan }
function Write-Yellow { param($m) Write-Host $m -ForegroundColor Yellow }
function Write-Red    { param($m) Write-Host $m -ForegroundColor Red }

# ── Auto Update Check ────────────────────────────────────────
function Check-ForUpdates {
    try {
        $remoteVersion = (Invoke-WebRequest -Uri $VERSION_URL -UseBasicParsing -ErrorAction Stop).Content.Trim()
        $localVersion  = if (Test-Path $VERSION_FILE) { (Get-Content $VERSION_FILE).Trim() } else { "0.0" }
        if ($remoteVersion -ne $localVersion) {
            Write-Yellow "============================================"
            Write-Yellow "  UPDATE AVAILABLE! ($localVersion -> $remoteVersion)"
            Write-Yellow "  Downloading latest files..."
            Write-Yellow "============================================"
            foreach ($file in @("collect_gestures.py","dsp_hand_gesture_ppt.py")) {
                try {
                    Invoke-WebRequest -Uri "$REPO_RAW/$file" -OutFile "$INSTALL_DIR\$file" -ErrorAction Stop
                    Write-Green "[+] Updated: $file"
                } catch { Write-Red "[!] Failed to update: $file" }
            }
            $remoteVersion | Out-File -FilePath $VERSION_FILE -Encoding UTF8 -Force
            Write-Green "[+] Updated to version $remoteVersion"
            Write-Host ""
            Start-Sleep -Seconds 1
        } else {
            Write-Green "[+] Already up to date! (v$localVersion)"
            Start-Sleep -Seconds 1
        }
    } catch {
        Write-Yellow "[~] No internet — skipping update check."
        Start-Sleep -Seconds 1
    }
}

# ── Menu ─────────────────────────────────────────────────────
function Show-Menu {
    Clear-Host
    Write-Cyan "============================================"
    Write-Cyan "         DSP GESTURE CONTROLLER"
    Write-Cyan "============================================"
    Write-Host "  [1]  Record Gestures"
    Write-Host "  [2]  Run Controller (PPT)"
    Write-Host "  [3]  View & Delete Gestures"
    Write-Host "  [4]  Exit"
    Write-Cyan "============================================"
    Write-Host ""
}

# ── View & Delete Gestures ───────────────────────────────────
function Manage-Gestures {
    $pkl = "$INSTALL_DIR\clean_gestures.pkl"

    while ($true) {
        Clear-Host
        Write-Cyan "============================================"
        Write-Cyan "         TRAINED GESTURES"
        Write-Cyan "============================================"

        if (-not (Test-Path $pkl)) {
            Write-Yellow "  No gestures file found!"
            Write-Yellow "  Please record gestures first (Option 1)"
            Write-Cyan "============================================"
            Write-Host ""
            Read-Host "Press Enter to go back"
            return
        }

        # Get gesture list from Python
        $listScript = @"
import pickle
with open(r'$pkl', 'rb') as f:
    g = pickle.load(f)
if not g:
    print('EMPTY')
else:
    for i,(name,samples) in enumerate(g.items(),1):
        print(f'{i}|{name}|{len(samples)}')
"@
        $output = & $PY312_PATH -c $listScript 2>&1

        if ($output -contains "EMPTY" -or -not $output) {
            Write-Yellow "  No gestures trained yet!"
            Write-Yellow "  Please record gestures first (Option 1)"
            Write-Cyan "============================================"
            Write-Host ""
            Read-Host "Press Enter to go back"
            return
        }

        # Show gesture list
        $gestures = @()
        foreach ($line in $output) {
            if ($line -match "^(\d+)\|(.+)\|(\d+)$") {
                $num     = $matches[1]
                $gName   = $matches[2]
                $samples = $matches[3]
                Write-Host "  [$num]  " -NoNewline
                Write-Host "$gName" -ForegroundColor White -NoNewline
                Write-Host "  →  " -NoNewline
                Write-Host "$samples sample(s)" -ForegroundColor Yellow
                $gestures += $gName
            }
        }

        Write-Cyan "--------------------------------------------"
        Write-Host "  Total gestures: " -NoNewline
        Write-Host "$($gestures.Count)" -ForegroundColor Green
        Write-Cyan "--------------------------------------------"
        Write-Host "  [D]  Delete a gesture"
        Write-Host "  [B]  Back to main menu"
        Write-Cyan "============================================"
        Write-Host ""

        $choice = Read-Host "Enter your choice"

        if ($choice -eq "B" -or $choice -eq "b") { return }

        if ($choice -eq "D" -or $choice -eq "d") {
            Write-Host ""
            Write-Yellow "Available gestures: $($gestures -join ', ')"
            $delName = (Read-Host "Enter gesture name to delete").Trim().ToLower()

            if ($gestures -contains $delName) {
                $delScript = @"
import pickle
with open(r'$pkl', 'rb') as f:
    g = pickle.load(f)
g.pop('$delName', None)
with open(r'$pkl', 'wb') as f:
    pickle.dump(g, f)
print('DELETED')
"@
                $res = & $PY312_PATH -c $delScript 2>&1
                if ($res -contains "DELETED") {
                    Write-Green "✓ Deleted gesture: '$delName'"
                } else {
                    Write-Red "Error deleting gesture!"
                }
            } else {
                Write-Red "Gesture '$delName' not found!"
            }
            Start-Sleep -Seconds 1
        }
    }
}

# ── First Install ────────────────────────────────────────────
$alreadyInstalled = (Test-Path "$INSTALL_DIR\collect_gestures.py") -and
                    (Test-Path "$INSTALL_DIR\dsp_hand_gesture_ppt.py") -and
                    (Test-Path "$INSTALL_DIR\hand_landmarker.task")

if (-not $alreadyInstalled) {
    Clear-Host
    Write-Cyan "============================================"
    Write-Cyan "   DSP Hand Gesture Controller - Setup"
    Write-Cyan "============================================"
    Write-Host ""

    if (-not (Test-Path $INSTALL_DIR)) {
        New-Item -ItemType Directory -Path $INSTALL_DIR | Out-Null
        Write-Green "[+] Created folder: $INSTALL_DIR"
    }
    Set-Location $INSTALL_DIR

    # Python 3.12
    Write-Host ""
    Write-Cyan "[*] Checking for Python 3.12..."
    if (Test-Path $PY312_PATH) {
        $ver = & $PY312_PATH --version 2>&1
        Write-Green "[+] Found: $ver"
    } else {
        Write-Yellow "[!] Installing Python 3.12..."
        $pyInstaller = "$env:TEMP\python-3.12.0-amd64.exe"
        Write-Cyan "[*] Downloading Python 3.12.0..."
        Invoke-WebRequest -Uri $PY312_URL -OutFile $pyInstaller
        Write-Green "[+] Download complete."
        Start-Process -FilePath $pyInstaller -ArgumentList `
            "/quiet InstallAllUsers=0 PrependPath=0 Include_pip=1 Include_launcher=1" -Wait
        Remove-Item $pyInstaller -Force -ErrorAction SilentlyContinue
        Write-Green "[+] Python 3.12 installed!"
        if (-not (Test-Path $PY312_PATH)) {
            Write-Red "[!] Install failed. Please install from https://python.org"
            pause; exit 1
        }
    }

    # Download files
    Write-Host ""
    Write-Cyan "[*] Downloading project files..."
    foreach ($file in @("collect_gestures.py","dsp_hand_gesture_ppt.py","hand_landmarker.task")) {
        try {
            Invoke-WebRequest -Uri "$REPO_RAW/$file" -OutFile "$INSTALL_DIR\$file" -ErrorAction Stop
            Write-Green "[+] Downloaded: $file"
        } catch { Write-Red "[!] Failed: $file" }
    }

    # Save version
    try {
        $v = (Invoke-WebRequest -Uri $VERSION_URL -UseBasicParsing -ErrorAction Stop).Content.Trim()
        $v | Out-File -FilePath $VERSION_FILE -Encoding UTF8 -Force
    } catch {
        "1.0" | Out-File -FilePath $VERSION_FILE -Encoding UTF8 -Force
    }

    # Install pip packages
    Write-Host ""
    Write-Cyan "[*] Installing Python dependencies..."
    foreach ($dep in @("opencv-python","mediapipe","numpy","imageio","pyautogui")) {
        Write-Host "    Installing $dep ..."
        & $PY312_PATH -m pip install $dep --quiet
    }
    Write-Green "[+] All dependencies installed."

    # Desktop Shortcut — points back to this script on GitHub
    Write-Host ""
    Write-Cyan "[*] Creating desktop shortcut..."
    $launcherCmd  = "-ExecutionPolicy Bypass -Command `". { iwr -useb $REPO_RAW/setup.ps1 } | iex`""
    $desktopPath  = [System.Environment]::GetFolderPath("Desktop")
    $shortcutPath = "$desktopPath\DSP Gesture Controller.lnk"
    $shell        = New-Object -ComObject WScript.Shell
    $shortcut     = $shell.CreateShortcut($shortcutPath)
    $shortcut.TargetPath       = "powershell.exe"
    $shortcut.Arguments        = $launcherCmd
    $shortcut.WorkingDirectory = $INSTALL_DIR
    $shortcut.WindowStyle      = 1
    $shortcut.Description      = "DSP Hand Gesture Controller"
    $shortcut.Save()
    Write-Green "[+] Desktop shortcut created!"

    Write-Host ""
    Write-Green "============================================"
    Write-Green "   Setup Complete! Shortcut on Desktop!"
    Write-Green "============================================"
    Start-Sleep -Seconds 2
}

Set-Location $INSTALL_DIR

# ── Check updates every launch ───────────────────────────────
Clear-Host
Write-Cyan "============================================"
Write-Cyan "   DSP GESTURE CONTROLLER"
Write-Cyan "============================================"
Write-Host ""
Write-Cyan "[*] Checking for updates..."
Check-ForUpdates

# ── Main Menu Loop ───────────────────────────────────────────
while ($true) {
    Show-Menu
    $choice = Read-Host "Enter your choice (1/2/3/4)"
    switch ($choice) {
        "1" {
            Write-Green "`n[>] Launching Recorder..."
            Write-Yellow "    R=Record | S=Save | Q=Quit"
            Write-Host ""
            & $PY312_PATH "$INSTALL_DIR\collect_gestures.py"
        }
        "2" {
            Write-Green "`n[>] Launching PPT Controller (Background)..."
            Write-Yellow "    Camera runs hidden — gesture away!"
            Write-Yellow "    Press CTRL+C here to stop."
            Write-Host ""
            & $PY312_PATH "$INSTALL_DIR\dsp_hand_gesture_ppt.py"
        }
        "3" {
            Manage-Gestures
        }
        "4" {
            Write-Cyan "`nGoodbye!"
            exit 0
        }
        default {
            Write-Red "Invalid! Enter 1, 2, 3 or 4."
        }
    }
    if ($choice -ne "3") {
        Write-Host ""
        Read-Host "Press Enter to return to menu..."
    }
}