param(
    [string]$Dataset = "",
    [string]$Bundle = "",
    [int]$Port = 8765
)
$ErrorActionPreference = 'Stop'
$AsrRoot = $PSScriptRoot
$ServerPath = Join-Path $AsrRoot 'review_web/server.py'
if (-not $Dataset) { $Dataset = Join-Path $AsrRoot 'data/train+public_test.zip' }
if (-not (Test-Path -LiteralPath $Dataset -PathType Leaf)) {
    throw 'Dataset ZIP not found. Run with -Dataset <path-to-train+public_test.zip>.'
}
$PythonArgs = @('-u', $ServerPath, '--dataset', $Dataset, '--port', "$Port")
if ($Bundle) {
    if (-not (Test-Path -LiteralPath $Bundle -PathType Leaf)) { throw 'Review bundle ZIP not found.' }
    $PythonArgs += @('--bundle', $Bundle)
}
Write-Host "Open http://127.0.0.1:$Port after the server starts. Ctrl+C to stop."
& python @PythonArgs
exit $LASTEXITCODE
