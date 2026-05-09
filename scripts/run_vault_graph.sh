#!/usr/bin/env bash
# Lightweight server: vault graph API + web UI without full Chroma / LLM load. Port 8001.

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
exec "$PY" -m uvicorn web.vault_server:app --host 127.0.0.1 --port 8001 "$@"
