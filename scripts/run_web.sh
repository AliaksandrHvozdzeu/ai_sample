#!/usr/bin/env bash
# Web UI on port 8000 (uvicorn). Extra args go to uvicorn, e.g. ./scripts/run_web.sh --reload

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="$ROOT/.venv/bin/python"

if [[ ! -x "$PY" ]]; then
  echo "ERROR: Missing $PY — run ./scripts/setup_venv.sh first." >&2
  exit 1
fi

if [[ -f "$ROOT/env_local.sh" ]]; then
  # shellcheck disable=SC1091
  source "$ROOT/env_local.sh"
fi

cd "$ROOT"
exec "$PY" -m uvicorn web.server:app --host 127.0.0.1 --port 8000 "$@"
