param(
    [string] $PythonPath = $env:MULTIMODAL_PYTHON,
    [string] $HostAddress = "127.0.0.1",
    [int] $Port = 8001
)

$ErrorActionPreference = "Stop"

if ([string]::IsNullOrWhiteSpace($PythonPath)) {
    throw "Set MULTIMODAL_PYTHON or pass -PythonPath with the Camera/Radar algorithm Python executable."
}

if (-not (Test-Path -LiteralPath $PythonPath -PathType Leaf)) {
    throw "Multimodal Python executable was not found: $PythonPath"
}

$repositoryRoot = Split-Path -Parent $PSScriptRoot
$backendDirectory = Join-Path $repositoryRoot "backend"

Push-Location $backendDirectory
try {
    & $PythonPath -m uvicorn `
        app.modules.fall.multimodal_engine.main:app `
        --host $HostAddress `
        --port $Port
}
finally {
    Pop-Location
}
