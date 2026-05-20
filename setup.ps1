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
    Write-Host "  [3]  View & Delete Gestures"
    Write-Host "  [4]  Exit"
    Write-Cyan "============================================"
    Write-Host ""
}

# ── View & Delete Gestures ───────────────────────────────────
function Manage-Gestures {
    $pkl = "$INSTALL_DIR\clean_gestures.pkl"
    if (-not (Test-Path $pkl)) {
        Write-Yellow "No gestures file found!"
        return
    }

    # Use Python to list gestures
    $listScript = @"
import pickle
with open(r'$pkl', 'rb') as f:
    g = pickle.load(f)
if not g:
    print('NO_GESTURES')
else:
    for name, samples in g.items():
        print(f'{name}|{len(samples)}')
"@
    $output = & $PY312_PATH -c $listScript 2>&1

    if ($output -contains "NO_GESTURES" -or -not $output) {
        Write-Yellow "No gestures trained yet!"
        return
    }

    while ($true) {
        Clear-Host
        Write-Cyan "============================================"
        Write-Cyan "         TRAINED GESTURES"
        Write-Cyan "============================================"

        $gestures = @()
        $i = 1
        foreach ($line in $output) {
            if ($line -match "^(.+)\|(\d+)$") {
                $gName    = $matches[1]
                $gSamples = $matches[2]
                Write-Host "  [$i]  $gName  ($gSamples samples)"
                $gestures += $gName
                $i++
            }
        }

        Write-Cyan "--------------------------------------------"
        Write-Host "  [D]  Delete a gesture"
        Write-Host "  [B]  Back to main menu"
        Write-Cyan "============================================"
        Write-Host ""

        $choice = Read-Host "Enter your choice"

        if ($choice -eq "B" -or $choice -eq "b") { return }

        if ($choice -eq "D" -or $choice -eq "d") {
            $delName = Read-Host "Enter gesture name to delete"
            $delName = $delName.Trim().ToLower()

            if ($gestures -contains $delName) {
                $deleteScript = @"
import pickle
with open(r'$pkl', 'rb') as f:
    g = pickle.load(f)
if '$delName' in g:
    del g['$delName']
    with open(r'$pkl', 'wb') as f:
        pickle.dump(g, f)
    print('DELETED')
else:
    print('NOT_FOUND')
"@
                $result = & $PY312_PATH -c $deleteScript 2>&1
                if ($result -contains "DELETED") {
                    Write-Green "✓ Deleted: $delName"
                    # Refresh list
                    $output = & $PY312_PATH -c $listScript 2>&1
                } else {
                    Write-Red "Could not find gesture: $delName"
                }
            } else {
                Write-Red "Gesture '$delName' not found!"
            }
            Start-Sleep -Seconds 1
        }
    }
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

    # ── 2. Check Python 3.12 ─────────────────────────────────
    Write-Host ""
    Write-Cyan "[*] Checking for Python 3.12..."

    if (Test-Path $PY312_PATH) {
        $ver = & $PY312_PATH --version 2>&1
        Write-Green "[+] Python 3.12 already installed: $ver"
    } else {
        Write-Yellow "[!] Python 3.12 not found. Installing now..."
        $pyInstaller = "$env:TEMP\python-3.12.0-amd64.exe"
        Write-Cyan "[*] Downloading Python 3.12.0..."
        Invoke-WebRequest -Uri $PY312_URL -OutFile $pyInstaller
        Write-Green "[+] Download complete."
        Write-Cyan "[*] Installing silently..."
        Start-Process -FilePath $pyInstaller -ArgumentList `
            "/quiet InstallAllUsers=0 PrependPath=0 Include_pip=1 Include_launcher=1" -Wait
        Write-Green "[+] Python 3.12 installed!"
        Remove-Item $pyInstaller -Force -ErrorAction SilentlyContinue
        if (-not (Test-Path $PY312_PATH)) {
            Write-Red "[!] Install failed. Please install from https://python.org"
            pause; exit 1
        }
    }

    # ── 3. Download project files ────────────────────────────
    Write-Host ""
    Write-Cyan "[*] Downloading project files..."
    foreach ($file in @("collect_gestures.py","dsp_hand_gesture_ppt.py","hand_landmarker.task")) {
        try {
            Invoke-WebRequest -Uri "$REPO_RAW/$file" -OutFile "$INSTALL_DIR\$file" -ErrorAction Stop
            Write-Green "[+] Downloaded: $file"
        } catch { Write-Red "[!] Failed: $file" }
    }

    # ── 4. Install dependencies ──────────────────────────────
    Write-Host ""
    Write-Cyan "[*] Installing Python dependencies..."
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
function Manage-Gestures {
    $pkl = "$INSTALL_DIR\clean_gestures.pkl"
    if (-not (Test-Path $pkl)) { Write-Yellow "No gestures file found!"; return }
    $listScript = "import pickle`nwith open(r'$pkl'.replace('`$env:USERPROFILE', [System.Environment]::GetFolderPath('UserProfile')), 'rb') as f:`n    g = pickle.load(f)`nprint('NO_GESTURES') if not g else [print(f'{n}|{len(s)}') for n,s in g.items()]"
    $pkl2 = $pkl.Replace('$env:USERPROFILE', $env:USERPROFILE)
    $listScript2 = "import pickle`nwith open(r'$pkl2', 'rb') as f:`n    g = pickle.load(f)`n" + 'print(chr(10).join([f"{n}|{len(s)}" for n,s in g.items()])) if g else print("NO_GESTURES")'
    $output = & $PY312 -c $listScript2 2>&1
    if ($output -contains "NO_GESTURES" -or -not $output) { Write-Yellow "No gestures trained yet!"; return }
    while ($true) {
        Clear-Host
        Write-Cyan "============================================"
        Write-Cyan "         TRAINED GESTURES"
        Write-Cyan "============================================"
        $gestures = @(); $i = 1
        foreach ($line in $output) {
            if ($line -match "^(.+)\|(\d+)$") {
                Write-Host "  [$i]  $($matches[1])  ($($matches[2]) samples)"
                $gestures += $matches[1]; $i++
            }
        }
        Write-Cyan "--------------------------------------------"
        Write-Host "  [D]  Delete a gesture"
        Write-Host "  [B]  Back to main menu"
        Write-Cyan "============================================"
        Write-Host ""
        $choice = Read-Host "Enter your choice"
        if ($choice -eq "B" -or $choice -eq "b") { return }
        if ($choice -eq "D" -or $choice -eq "d") {
            $delName = (Read-Host "Enter gesture name to delete").Trim().ToLower()
            if ($gestures -contains $delName) {
                $delScript = "import pickle`nwith open(r'$pkl2', 'rb') as f:`n    g = pickle.load(f)`ng.pop('$delName', None)`n" + "open(r'$pkl2', 'wb').__class__" 
                $delScript2 = "import pickle`nf=open(r'$pkl2','rb');g=pickle.load(f);f.close();g.pop('$delName',None);f=open(r'$pkl2','wb');pickle.dump(g,f);f.close();print('DELETED')"
                $res = & $PY312 -c $delScript2 2>&1
                if ($res -contains "DELETED") { Write-Green "Deleted: $delName"; $output = & $PY312 -c $listScript2 2>&1 }
                else { Write-Red "Error deleting!" }
            } else { Write-Red "Gesture not found!" }
            Start-Sleep -Seconds 1
        }
    }
}
while ($true) {
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
    $choice = Read-Host "Enter your choice (1/2/3/4)"
    switch ($choice) {
        "1" { Write-Green "`n[>] Launching Recorder..."; Write-Yellow "    R=Record | S=Save | Q=Quit"; & $PY312 "$INSTALL_DIR\collect_gestures.py" }
        "2" { Write-Green "`n[>] Launching PPT Controller..."; Write-Yellow "    Press Q to quit."; & $PY312 "$INSTALL_DIR\dsp_hand_gesture_ppt.py" }
        "3" { Manage-Gestures }
        "4" { Write-Cyan "`nGoodbye!"; exit 0 }
        default { Write-Red "Invalid! Enter 1, 2, 3 or 4." }
    }
    if ($choice -ne "3") { Write-Host ""; Read-Host "Press Enter to return to menu..." }
}
'@
    $launcherCode | Out-File -FilePath $LAUNCHER -Encoding UTF8
    Write-Green "[+] Launcher created."

    # ── 6. Desktop Shortcut ──────────────────────────────────
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
    $choice = Read-Host "Enter your choice (1/2/3/4)"
    switch ($choice) {
        "1" { Write-Green "`n[>] Launching Recorder..."; Write-Yellow "    R=Record | S=Save | Q=Quit"; & $PY312_PATH "$INSTALL_DIR\collect_gestures.py" }
        "2" { Write-Green "`n[>] Launching PPT Controller..."; Write-Yellow "    Press Q to quit."; & $PY312_PATH "$INSTALL_DIR\dsp_hand_gesture_ppt.py" }
        "3" { Manage-Gestures }
        "4" { Write-Cyan "`nGoodbye!"; exit 0 }
        default { Write-Red "Invalid! Enter 1, 2, 3 or 4." }
    }
    if ($choice -ne "3") { Write-Host ""; Read-Host "Press Enter to return to menu..." }
}