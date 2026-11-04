param(
    [string]$PythonPath,
    [switch]$SkipDeviceCheck
)

$ErrorActionPreference = 'Stop'
$taskPython = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
Push-Location $PSScriptRoot
try {
    if (-not (Test-Path -LiteralPath $taskPython)) {
        if ($PythonPath) {
            $taskBasePython = $PythonPath
        } elseif (Get-Command py.exe -ErrorAction SilentlyContinue) {
            $taskBasePython = & py.exe -3 -c 'import sys; print(sys.executable)'
            if ($LASTEXITCODE -ne 0) { throw 'Install 64-bit Python 3.11+ or pass -PythonPath.' }
        } elseif (Get-Command python.exe -ErrorAction SilentlyContinue) {
            $taskBasePython = (Get-Command python.exe).Source
        } else {
            throw 'Install 64-bit Python 3.11+ (python.org), then run setup.ps1 again.'
        }
        & $taskBasePython -c 'import sys, struct; sys.exit(0 if sys.version_info >= (3, 11) and struct.calcsize(chr(80)) == 8 else 1)'
        if ($LASTEXITCODE -ne 0) { throw '64-bit Python 3.11+ is required.' }
        & $taskBasePython -m venv .venv
        if ($LASTEXITCODE -ne 0) { throw 'Failed to create .venv.' }
    }
    & $taskPython -c 'import sys, struct; sys.exit(0 if sys.version_info >= (3, 11) and struct.calcsize(chr(80)) == 8 else 1)'
    if ($LASTEXITCODE -ne 0) { throw 'Existing .venv requires 64-bit Python 3.11+.' }
    & $taskPython -m pip install -r requirements-vision.txt
    if ($LASTEXITCODE -ne 0) { throw 'Dependency installation failed. Check network access and retry setup.ps1.' }
    if (-not $SkipDeviceCheck) {
        & $taskPython -m pet_alliance doctor
        if ($LASTEXITCODE -ne 0) {
            throw 'Dependencies installed. Start LDPlayer, enable local ADB, use 1920x1080 landscape, then run .\.venv\Scripts\python.exe -m pet_alliance doctor. See README.md for custom ADB paths.'
        }
    }
    Write-Host 'Setup complete. Open the game at Tianxing Town, then run .\run_daily_hero.ps1 -Preview or .\run_daily_hero.ps1.'
} finally {
    Pop-Location
}
