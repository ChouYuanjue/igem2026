#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="${STARASE_NAVIGATOR_PYTHON:-${ROOT_DIR}/.venv/bin/python}"
ENV_FILE="${STARASE_NAVIGATOR_ENV_FILE:-${ROOT_DIR}/results/starase_navigator_runtime/deepseek.env}"
EVENTS="${STARASE_AGENT_EVAL_EVENTS:-${ROOT_DIR}/results/starase_navigator_runtime/run_events.jsonl}"
OUTPUT_DIR="${1:-${ROOT_DIR}/results/starase_navigator_agent_eval_current}"
HISTORY_LIMIT="${STARASE_AGENT_HISTORY_LIMIT:-20}"
METAMORPHIC_LIMIT="${STARASE_AGENT_METAMORPHIC_LIMIT:-20}"

if [[ ! -x "${PYTHON}" ]]; then
  echo "[error] Python environment not found: ${PYTHON}" >&2
  exit 1
fi
if [[ ! -f "${ENV_FILE}" ]]; then
  echo "[error] Starase runtime environment not found: ${ENV_FILE}" >&2
  exit 1
fi
if [[ ! -f "${EVENTS}" ]]; then
  echo "[error] Starase run-event history not found: ${EVENTS}" >&2
  exit 1
fi

mkdir -p "${OUTPUT_DIR}"
chmod 700 "${OUTPUT_DIR}" 2>/dev/null || true
set -a
# shellcheck disable=SC1090
source "${ENV_FILE}"
set +a

cd "${ROOT_DIR}"
"${PYTHON}" -m scripts.starase_navigator.evaluate_agent_history   --events "${EVENTS}"   --limit "${HISTORY_LIMIT}"   --output "${OUTPUT_DIR}/history.json"

"${PYTHON}" -m scripts.starase_navigator.evaluate_agent_metamorphic   --events "${EVENTS}"   --limit "${METAMORPHIC_LIMIT}"   --output "${OUTPUT_DIR}/metamorphic.json"

"${PYTHON}" - "${OUTPUT_DIR}" <<'PY'
from __future__ import annotations
import json
import sys
from pathlib import Path

out = Path(sys.argv[1])
history = json.loads((out / "history.json").read_text(encoding="utf-8"))
metamorphic = json.loads((out / "metamorphic.json").read_text(encoding="utf-8"))
summary = {
    "schema": "starase-agent-evaluation-suite-v1",
    "history": {
        "case_count": int(history.get("case_count") or 0),
        "success_count": int(history.get("success_count") or 0),
        "failure_count": int(history.get("failure_count") or 0),
    },
    "metamorphic": {
        "case_count": int(metamorphic.get("case_count") or 0),
        "variant_count": int(metamorphic.get("variant_count") or 0),
        "non_original_variant_count": int(metamorphic.get("non_original_variant_count") or 0),
        "success_count": int(metamorphic.get("matching_variant_count") or 0),
        "failure_count": max(
            0,
            int(metamorphic.get("non_original_variant_count") or 0)
            - int(metamorphic.get("matching_variant_count") or 0),
        ),
        "error_variant_count": int(metamorphic.get("error_variant_count") or 0),
    },
}
summary["passed"] = (
    summary["history"]["failure_count"] == 0
    and summary["metamorphic"]["failure_count"] == 0
)
(out / "summary.json").write_text(
    json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
    encoding="utf-8",
)
print(json.dumps(summary, ensure_ascii=False, indent=2))
raise SystemExit(0 if summary["passed"] else 1)
PY
