# Web UI on port 8000: chat + 3D graph at /rag/chat (`/` redirects there); APIs under /api/rag/* and /api/vault/*.
# For graph-only on port 8001 without loading the LLM, use .\scripts\run_vault_graph.ps1
# Extra args go to uvicorn, e.g. .\scripts\run_web.ps1 --reload
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
& $Py -m uvicorn web.server:app --host 127.0.0.1 --port 8000 @args
