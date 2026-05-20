# ============================================================
#  DSP Hand Gesture Controller - Setup & Launcher
#  Usage: irm https://raw.githubusercontent.com/YOUR_USERNAME/YOUR_REPO/main/setup.ps1 | iex
# ============================================================

$REPO_RAW = "https://raw.githubusercontent.com/vk566/dsp-gesture-controller/main"
$INSTALL_DIR = "$env:USERPROFILE\DSPGestureController"

# ── Colours ─────────────────────────────────────────────────
function Write-Green  { param($m) Write-Host $m -ForegroundColor Green }
function Write-Cyan   { param($m) Write-Host $m -ForegroundColor Cyan }
function Write-Yellow { param($m) Write-Host $m -ForegroundColor Yellow }
function Write-Red    { param($m) Write-Host $m -ForegroundColor Red }

Clear-Host
Write-Cyan "============================================"
Write-Cyan "   DSP Hand Gesture Controller - Setup"
Write-Cyan "============================================"
Write-Host ""

# ── 1. Create install folder ────────────────────────────────
if (-not (Test-Path $INSTALL_DIR)) {
    New-Item -ItemType Directory -Path $INSTALL_DIR | Out-Null
    Write-Green "[+] Created folder: $INSTALL_DIR"
} else {
    Write-Yellow "[~] Folder already exists: $INSTALL_DIR"
}
Set-Location $INSTALL_DIR

# ── 2. Check Python ─────────────────────────────────────────
Write-Host ""
Write-Cyan "[*] Checking Python..."
try {
    $pyver = python --version 2>&1
    Write-Green "[+] Found: $pyver"
} catch {
    Write-Red "[!] Python not found. Please install Python 3.10+ and re-run."
    pause
    exit 1
}

# ── 3. Download project files ───────────────────────────────
Write-Host ""
Write-Cyan "[*] Downloading project files from GitHub..."

$files = @(
    "collect_gestures.py",
    "dsp_hand_gesture_ppt.py",
    "hand_landmarker.task"
)

foreach ($file in $files) {
    $url  = "$REPO_RAW/$file"
    $dest = "$INSTALL_DIR\$file"
    try {
        Invoke-WebRequest -Uri $url -OutFile $dest -ErrorAction Stop
        Write-Green "[+] Downloaded: $file"
    } catch {
        Write-Red "[!] Failed to download: $file"
        Write-Red "    Make sure it exists in your GitHub repo."
    }
}

# ── 4. Install Python dependencies ──────────────────────────
Write-Host ""
Write-Cyan "[*] Installing Python dependencies..."

$deps = @(
    "opencv-python",
    "mediapipe",
    "numpy",
    "imageio",
    "pyautogui"
)

foreach ($dep in $deps) {
    Write-Host "    Installing $dep ..."
    python -m pip install $dep --quiet
}

Write-Green "[+] All dependencies installed."

# ── 5. Menu ─────────────────────────────────────────────────
function Show-Menu {
    Write-Host ""
    Write-Cyan "============================================"
    Write-Cyan "         DSP GESTURE CONTROLLER"
    Write-Cyan "============================================"
    Write-Host "  [1]  Record Gestures   (collect_gestures.py)"
    Write-Host "  [2]  Run Controller    (dsp_hand_gesture_ppt.py)"
    Write-Host "  [3]  Exit"
    Write-Cyan "============================================"
    Write-Host ""
}

while ($true) {
    Show-Menu
    $choice = Read-Host "Enter your choice (1/2/3)"

    switch ($choice) {
        "1" {
            Write-Green "`n[>] Launching Gesture Recorder..."
            Write-Yellow "    Controls: R = Record | S = Save | D = Delete | Q = Quit"
            Write-Host ""
            python "$INSTALL_DIR\collect_gestures.py"
        }
        "2" {
            Write-Green "`n[>] Launching PPT Controller..."
            Write-Yellow "    Use recorded gestures to control your slideshow."
            Write-Yellow "    Press Q in the camera window to quit."
            Write-Host ""
            python "$INSTALL_DIR\dsp_hand_gesture_ppt.py"
        }
        "3" {
            Write-Cyan "`nGoodbye!"
            exit 0
        }
        default {
            Write-Red "Invalid choice. Enter 1, 2, or 3."
        }
    }

    Write-Host ""
    Read-Host "Press Enter to return to menu..."
}