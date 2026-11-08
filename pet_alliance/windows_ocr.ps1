param(
    [Parameter(Mandatory = $true)][string]$ImagePath,
    [string]$Language = 'zh-Hans-CN'
)

$ErrorActionPreference = 'Stop'
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new($false)
$null = Add-Type -AssemblyName System.Runtime.WindowsRuntime
$asTask = [System.WindowsRuntimeSystemExtensions].GetMethods() | Where-Object {
    $_.Name -eq 'AsTask' -and $_.IsGenericMethod -and
    $_.GetGenericArguments().Count -eq 1 -and $_.GetParameters().Count -eq 1
} | Select-Object -First 1

function Wait-WinRT($Operation, [Type]$ResultType) {
    $task = $asTask.MakeGenericMethod($ResultType).Invoke($null, @($Operation))
    $task.GetAwaiter().GetResult()
}

$fileType = [Windows.Storage.StorageFile, Windows.Storage, ContentType = WindowsRuntime]
$streamType = [Windows.Storage.Streams.IRandomAccessStream, Windows.Storage.Streams, ContentType = WindowsRuntime]
$decoderType = [Windows.Graphics.Imaging.BitmapDecoder, Windows.Graphics.Imaging, ContentType = WindowsRuntime]
$bitmapType = [Windows.Graphics.Imaging.SoftwareBitmap, Windows.Graphics.Imaging, ContentType = WindowsRuntime]
$resultType = [Windows.Media.Ocr.OcrResult, Windows.Foundation, ContentType = WindowsRuntime]
$engineType = [Windows.Media.Ocr.OcrEngine, Windows.Foundation, ContentType = WindowsRuntime]
$languageType = [Windows.Globalization.Language, Windows.Foundation, ContentType = WindowsRuntime]
$engine = $engineType::TryCreateFromLanguage($languageType::new($Language))
if ($null -eq $engine) {
    throw "Windows OCR language unavailable: $Language"
}

$stream = $null
$bitmap = $null
try {
    $file = Wait-WinRT ($fileType::GetFileFromPathAsync($ImagePath)) $fileType
    $stream = Wait-WinRT ($file.OpenAsync([Windows.Storage.FileAccessMode, Windows.Storage, ContentType = WindowsRuntime]::Read)) $streamType
    $decoder = Wait-WinRT ($decoderType::CreateAsync($stream)) $decoderType
    $bitmap = Wait-WinRT ($decoder.GetSoftwareBitmapAsync()) $bitmapType
    $result = Wait-WinRT ($engine.RecognizeAsync($bitmap)) $resultType
    @{ text = $result.Text; language = $engine.RecognizerLanguage.LanguageTag } | ConvertTo-Json -Compress
}
finally {
    if ($null -ne $bitmap) { $bitmap.Dispose() }
    if ($null -ne $stream) { $stream.Dispose() }
}
