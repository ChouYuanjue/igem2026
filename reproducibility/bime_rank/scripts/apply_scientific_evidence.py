from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from projects.active.fibre.runtime.scientific_evidence import (
    TabularEvidenceModule,
    fuse_admitted_evidence,
    load_admitted_evidence,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Apply one or more admitted scientific-evidence tables to a frozen "
            "FIBRE/BiME candidate score table."
        )
    )
    parser.add_argument("--core-csv", type=Path, required=True)
    parser.add_argument("--direction", choices=("r2e", "e2r"), required=True)
    parser.add_argument("--query-id", required=True)
    parser.add_argument(
        "--evidence-csv",
        type=Path,
        action="append",
        default=[],
        help="Repeat once per admitted evidence module.",
    )
    parser.add_argument(
        "--admission-json",
        type=Path,
        action="append",
        default=[],
        help="Cross-fit admission output paired with --evidence-csv.",
    )
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if len(args.evidence_csv) != len(args.admission_json):
        raise ValueError("--evidence-csv and --admission-json counts must match")
    if args.top_k <= 0:
        raise ValueError("--top-k must be positive")

    core = pd.read_csv(args.core_csv, dtype={"query_id": str, "candidate_id": str})
    required = {"query_id", "candidate_id", "core_score"}
    missing = sorted(required - set(core.columns))
    if missing:
        raise ValueError(f"core CSV missing columns: {missing}")
    core = core[core["query_id"].astype(str).eq(str(args.query_id))].copy()
    if core.empty:
        raise ValueError(f"query {args.query_id!r} is absent from core CSV")
    if core["candidate_id"].duplicated().any():
        raise ValueError("core CSV contains duplicate candidate IDs for the query")
    core["core_score"] = pd.to_numeric(core["core_score"], errors="raise").astype(float)
    if not np.isfinite(core["core_score"]).all():
        raise ValueError("core_score must be finite")

    candidate_ids = core["candidate_id"].astype(str).tolist()
    core_score = core["core_score"].to_numpy(np.float64)

    outputs = []
    admitted = []
    for evidence_path, admission_path in zip(
        args.evidence_csv,
        args.admission_json,
        strict=True,
    ):
        registration = load_admitted_evidence(admission_path)
        if args.direction not in registration.descriptor.directions:
            raise ValueError(
                f"{registration.descriptor.name} does not support {args.direction}"
            )
        module = TabularEvidenceModule.from_csv(
            evidence_path,
            registration.descriptor,
        )
        outputs.append(
            module.score(
                direction=args.direction,
                query_id=str(args.query_id),
                candidate_ids=candidate_ids,
            )
        )
        admitted.append(registration)

    fused, contributions = fuse_admitted_evidence(
        core_score,
        outputs,
        admitted,
    )
    result = core[["query_id", "candidate_id", "core_score"]].copy()
    result["fused_score"] = fused
    for name, values in contributions.items():
        result[f"evidence:{name}"] = values
    result = result.sort_values(
        ["fused_score", "candidate_id"],
        ascending=[False, True],
        kind="stable",
    ).reset_index(drop=True)
    result.insert(0, "rank", np.arange(1, len(result) + 1, dtype=np.int64))
    result = result.head(args.top_k)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    audit = {
        "schema": "fibre-scientific-evidence-query-v1",
        "query_id": str(args.query_id),
        "direction": args.direction,
        "core_csv": str(args.core_csv),
        "modules": [
            {
                "name": registration.descriptor.name,
                "kind": registration.descriptor.kind,
                "strength": registration.strength,
                "quality_slope": registration.quality_slope,
                "evidence_csv": str(evidence_path),
                "admission_json": str(admission_path),
            }
            for evidence_path, admission_path, registration in zip(
                args.evidence_csv,
                args.admission_json,
                admitted,
                strict=True,
            )
        ],
        "candidate_count": int(len(core)),
        "reported_top_k": int(len(result)),
    }
    args.output.with_suffix(args.output.suffix + ".audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n"
    )
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
