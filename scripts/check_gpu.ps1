# Quick check: PyTorch sees CUDA (run after setup_venv.ps1).
$ErrorActionPreference = "Stop"
$Root = Resolve-Path (Join-Path $PSScriptRoot "..")
$Py = Join-Path $Root ".venv\Scripts\python.exe"
if (-not (Test-Path $Py)) {
    Write-Error ('Missing {0} - run .\scripts\setup_venv.ps1 first.' -f $Py)
}
& $Py -c 'import torch; print(torch.__version__, torch.cuda.is_available())'
