param(
    [switch]$Preview,
    [string]$AdbPath,
    [string]$Serial
)

$ErrorActionPreference = 'Stop'
$taskPython = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $taskPython)) {
    throw 'Run: powershell -NoProfile -ExecutionPolicy Bypass -File .\setup.ps1'
}
Push-Location $PSScriptRoot
try {
    $taskArguments = @('-m', 'pet_alliance')
    if ($AdbPath) { $taskArguments += @('--adb', $AdbPath) }
    if ($Serial) { $taskArguments += @('--serial', $Serial) }
    if ($Preview) { $taskArguments += '--dry-run' }
    $taskArguments += @('run', 'daily_hero_reset')
    & $taskPython @taskArguments
    $taskExitCode = $LASTEXITCODE
} finally {
    Pop-Location
}
exit $taskExitCode
