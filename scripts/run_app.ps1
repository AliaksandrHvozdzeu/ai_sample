# Runs app.py with the project's .venv Python (no global site-packages).
$ErrorActionPreference = "Stop"
$Root = Resolve-Path (Join-Path $PSScriptRoot "..")
$Py = Join-Path $Root ".venv\Scripts\python.exe"
if (-not (Test-Path $Py)) {
    Write-Error "Missing $Py - run .\scripts\setup_venv.ps1 first."
}
$EnvLocal = Join-Path $Root "env_local.ps1"
if (Test-Path $EnvLocal) {
    . $EnvLocal
}
Set-Location $Root
$App = Join-Path $Root 'app.py'
& $Py $App @args
