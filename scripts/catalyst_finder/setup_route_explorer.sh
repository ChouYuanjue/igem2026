#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="${PYTHON:-$ROOT/.venv/bin/python}"
PIP="${PIP:-$ROOT/.venv/bin/pip}"
SITE="$ROOT/results/catalyst_finder_runtime/route_design/pickaxe_site"
VENDOR="$ROOT/external_repos/route_design/MINE-Database"
COMMIT_FILE="$ROOT/external_repos/route_design/MINE-Database.UPSTREAM_COMMIT"

if [[ ! -d "$VENDOR" || ! -f "$COMMIT_FILE" ]]; then
  echo "Pinned MINE-Database source is missing under external_repos/route_design." >&2
  exit 2
fi
mkdir -p "$SITE"
# These are import-time dependencies of the pinned upstream Pickaxe source that are
# not needed by Catalyst Finder itself. Install only into the worker site.
"$PIP" install --disable-pip-version-check --target "$SITE" \
  'python-libsbml==5.21.1' 'lxml==5.4.0'
PYTHONPATH="$SITE:$VENDOR" "$PYTHON" - <<'PY'
import libsbml, lxml
print("route explorer runtime ready")
print("libSBML", libsbml.getLibSBMLDottedVersion())
print("lxml", lxml.__version__)
PY
