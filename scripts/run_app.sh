#!/usr/bin/env bash
# Runs app.py with the project's .venv Python (no global site-packages).

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
exec "$PY" "$ROOT/app.py" "$@"
