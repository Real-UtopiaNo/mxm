param(
    [string]$Config,
    [string]$AdbPath,
    [string]$Serial
)

$ErrorActionPreference = 'Stop'
$taskPython = Join-Path $PSScriptRoot '.venv\Scripts\pythonw.exe'
$taskUseConsolePython = -not (Test-Path -LiteralPath $taskPython)
if ($taskUseConsolePython) {
    $taskPython = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
}
if (-not (Test-Path -LiteralPath $taskPython)) {
    throw 'Run: powershell -NoProfile -ExecutionPolicy Bypass -File .\setup.ps1'
}

# Start-Process joins its argument array; quote using Windows argv rules so
# configuration/ADB paths containing spaces and trailing slashes stay intact.
function ConvertTo-TaskWindowsArgument([string]$Value) {
    $taskEscaped = [regex]::Replace($Value, '(\\*)"', '$1$1\"')
    $taskEscaped = [regex]::Replace($taskEscaped, '(\\+)$', '$1$1')
    return '"' + $taskEscaped + '"'
}

$taskArguments = @('-m', 'pet_alliance.scheduler_ui')
if ($Config) { $taskArguments += @('--config', $Config) }
if ($AdbPath) { $taskArguments += @('--adb', $AdbPath) }
if ($Serial) { $taskArguments += @('--serial', $Serial) }
$taskArgumentLine = ($taskArguments | ForEach-Object { ConvertTo-TaskWindowsArgument $_ }) -join ' '
$taskStartOptions = @{
    FilePath = $taskPython
    WorkingDirectory = $PSScriptRoot
    ArgumentList = $taskArgumentLine
    WindowStyle = 'Hidden'
}
Start-Process @taskStartOptions
