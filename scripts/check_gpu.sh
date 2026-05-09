#!/usr/bin/env bash
# Quick check: PyTorch build and CUDA / MPS availability (run after setup_venv.sh).

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="$ROOT/.venv/bin/python"

if [[ ! -x "$PY" ]]; then
  echo "ERROR: Missing $PY — run ./scripts/setup_venv.sh first." >&2
  exit 1
fi

exec "$PY" -c "
import torch
print('torch', torch.__version__)
print('cuda_available', torch.cuda.is_available())
if getattr(torch.backends, 'mps', None) is not None and torch.backends.mps.is_built():
    print('mps_available', torch.backends.mps.is_available())
else:
    print('mps_available', False)
"
