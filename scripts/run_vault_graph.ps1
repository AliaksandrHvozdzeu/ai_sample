# Lightweight server: vault graph API + same web UI as main app (no Chroma / LLM). Port 8001 vs full stack on 8000.
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
& $Py -m uvicorn web.vault_server:app --host 127.0.0.1 --port 8001 @args
