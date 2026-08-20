#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="${PYTHON:-$ROOT/.venv/bin/python}"
PIP="${PIP:-$ROOT/.venv/bin/pip}"
BASE="$ROOT/results/catalyst_finder_runtime/route_feasibility"
THERMO_SITE="$BASE/thermo_site"
THERMO_CACHE="$BASE/thermo_cache"
FBA_SITE="$BASE/fba_site"
MODEL="$BASE/iML1515.json"
POOL="$BASE/iML1515_cytosol_chebi.txt"
mkdir -p "$THERMO_SITE" "$THERMO_CACHE" "$FBA_SITE"

if ! PYTHONPATH="$THERMO_SITE" "$PYTHON" - <<'PY' >/dev/null 2>&1
from importlib.metadata import version
import equilibrator_api, equilibrator_pathway
assert version('equilibrator-api') == '0.7.0'
assert version('equilibrator-pathway') == '0.7.1'
PY
then
  "$PIP" install --disable-pip-version-check --upgrade --target "$THERMO_SITE" \
    'equilibrator-api==0.7.0' 'equilibrator-pathway==0.7.1'
fi

if ! PYTHONPATH="$FBA_SITE" "$PYTHON" - <<'PY' >/dev/null 2>&1
import cobra
assert cobra.__version__ == '0.32.1'
PY
then
  "$PIP" install --disable-pip-version-check --upgrade --target "$FBA_SITE" 'cobra==0.32.1'
fi

if [[ ! -s "$THERMO_CACHE/equilibrator/compounds.sqlite" || ! -s "$THERMO_CACHE/equilibrator/cc_params.npz" ]]; then
  echo "Initializing official eQuilibrator thermodynamic cache (~1.4 GB)..."
  XDG_CACHE_HOME="$THERMO_CACHE" PYTHONPATH="$THERMO_SITE" "$PYTHON" - <<'PY'
from equilibrator_api import ComponentContribution
cc = ComponentContribution()
print('eQuilibrator cache ready:', cc.ccache.water)
PY
fi

PYTHONPATH="$FBA_SITE" "$PYTHON" - <<PY
from pathlib import Path
from cobra.io import load_json_model, load_model, save_json_model
model_path = Path(r"$MODEL")
if model_path.exists():
    model = load_json_model(str(model_path))
else:
    model = load_model('iML1515')
    save_json_model(model, str(model_path))
solution = model.optimize()
if str(solution.status) != 'optimal':
    raise SystemExit('iML1515 baseline optimization is not optimal')
ids = set()
for metabolite in model.metabolites:
    if metabolite.compartment != 'c':
        continue
    raw = (metabolite.annotation or {}).get('chebi')
    values = raw if isinstance(raw, list) else ([raw] if raw else [])
    for value in values:
        text = str(value or '').upper().strip()
        if text and not text.startswith('CHEBI:'):
            text = 'CHEBI:' + text.replace('CHEBI', '').lstrip(':')
        if text:
            ids.add(text)
Path(r"$POOL").write_text('\n'.join(sorted(ids)) + '\n', encoding='utf-8')
print('iML1515 ready:', len(model.reactions), 'reactions,', len(model.metabolites), 'metabolites, baseline growth', float(solution.objective_value or 0.0))
print('cytosolic ChEBI annotations:', len(ids))
PY

XDG_CACHE_HOME="$THERMO_CACHE" PYTHONPATH="$THERMO_SITE" "$PYTHON" - <<'PY'
from importlib.metadata import version
from equilibrator_api import ComponentContribution
cc = ComponentContribution()
print('equilibrator-api', version('equilibrator-api'))
print('equilibrator-pathway', version('equilibrator-pathway'))
print('conditions:', 'pH', cc.p_h, 'pMg', cc.p_mg, 'I', cc.ionic_strength, 'T', cc.temperature)
PY
PYTHONPATH="$FBA_SITE" "$PYTHON" - <<'PY'
import cobra
print('COBRApy', cobra.__version__)
PY
