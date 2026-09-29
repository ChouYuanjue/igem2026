from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SUMMARY = ROOT / "results/fibre_heterogeneous_general_v1_seed20260723/summary.json"
DEFAULT_OUTPUT = (
    ROOT
    / "reproducibility/bime_rank/records/FIBRE_HETEROGENEOUS_CONDITIONAL_MODES_V1_RESULT.json"
)
BIME_SCORECARD = (
    ROOT
    / "reproducibility/bime_rank/records/BIME_RANK_RETRIEVAL_CAPABILITY_SCORECARD_V2.json"
)
CONFIG = (
    ROOT
    / "reproducibility/bime_rank/configs/fibre_heterogeneous_conditional_modes_v1.yaml"
)
TRAINER = (
    ROOT
    / "reproducibility/bime_rank/support/train_fibre_heterogeneous_general_v1.py"
)
KERNEL = ROOT / "projects/active/fibre/kernel/heterogeneous_modes.py"
RUNTIME = ROOT / "projects/active/fibre/runtime/heterogeneous_fibre.py"
QUERY_CLI = ROOT / "reproducibility/bime_rank/scripts/query_fibre_hcm_v1.py"
PROTOCOL_LOCK = (
    ROOT / "reproducibility/bime_rank/records/FIBRE_HCM_V1_PROTOCOL_LOCK.json"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def metric_delta(candidate: dict[str, float], baseline: dict[str, float]) -> dict[str, float]:
    keys = ("mrr", "map", "ndcg_at_10", "hit_at_10", "hit_at_20", "hit_at_50")
    return {
        key: float(candidate[key]) - float(baseline[key])
        for key in keys
        if key in candidate and key in baseline
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    summary = json.loads(args.summary.read_text())
    protocol_lock = json.loads(PROTOCOL_LOCK.read_text())
    locked = protocol_lock["implementation_lock"]
    current_hashes = {
        "config": sha256(CONFIG),
        "trainer": sha256(TRAINER),
        "kernel": sha256(KERNEL),
    }
    for name, current in current_hashes.items():
        expected = str(locked[name]["sha256"])
        if current != expected:
            raise RuntimeError(
                f"formal FIBRE-HCM implementation drift after protocol lock: "
                f"{name} expected {expected}, got {current}"
            )
    bime = json.loads(BIME_SCORECARD.read_text())
    strict = bime["zero_shot_external"]["clipzyme_strict_temporal_same_support"]
    bime_r2e = strict["r2e"]["bime_rank"]
    bime_e2r = strict["e2r"]["bime_rank"]
    fibre_r2e = summary["strict_temporal"]["r2e"]
    fibre_e2r = summary["strict_temporal"]["e2r"]

    record = {
        "schema": "fibre-heterogeneous-conditional-modes-v1-result",
        "method": "FIBRE-HCM",
        "method_expansion": "FIBRE Heterogeneous Conditional Modes",
        "status": "frozen_candidate_evaluated_once",
        "predecessor": {
            "method": "FIBRE-Conditional Modes",
            "record": "reproducibility/bime_rank/records/FIBRE_CONDITIONAL_MODES_SCORECARD_V2.json",
        },
        "motivation": (
            "Bring heterogeneous evidence that previously lived in BiME-Rank routing "
            "and FIBRE side experiments into one availability-aware, query-conditioned "
            "FIBRE computation graph, then train it directly on the general sparse relation set."
        ),
        "architecture": summary["architecture"],
        "training": summary["training"],
        "strict_temporal": {
            "protocol": (
                "Rhea release128->141 strict double-cold. Training and architecture were "
                "fixed before this label set was evaluated."
            ),
            "fibre_hcm": summary["strict_temporal"],
            "retained_bime_rank_reference": {
                "r2e": bime_r2e,
                "e2r": bime_e2r,
            },
            "delta_fibre_hcm_minus_bime_rank": {
                "r2e": metric_delta(fibre_r2e, bime_r2e),
                "e2r": metric_delta(fibre_e2r, bime_e2r),
            },
        },
        "checkpoint": {
            "path": summary["checkpoint"],
            "sha256": summary["checkpoint_sha256"],
        },
        "provenance": {
            "summary": str(args.summary),
            "summary_sha256": sha256(args.summary),
            "config": str(CONFIG.relative_to(ROOT)),
            "config_sha256": sha256(CONFIG),
            "trainer": str(TRAINER.relative_to(ROOT)),
            "trainer_sha256": sha256(TRAINER),
            "kernel": str(KERNEL.relative_to(ROOT)),
            "kernel_sha256": sha256(KERNEL),
            "runtime": str(RUNTIME.relative_to(ROOT)),
            "runtime_sha256": sha256(RUNTIME),
            "query_cli": str(QUERY_CLI.relative_to(ROOT)),
            "query_cli_sha256": sha256(QUERY_CLI),
            "protocol_lock": str(PROTOCOL_LOCK.relative_to(ROOT)),
            "protocol_lock_sha256": sha256(PROTOCOL_LOCK),
            "bime_reference": str(BIME_SCORECARD.relative_to(ROOT)),
            "bime_reference_sha256": sha256(BIME_SCORECARD),
        },
        "decision": {
            "default_production_route_changed": False,
            "automatic_promotion": False,
            "note": (
                "This record freezes the first native general FIBRE-HCM result. "
                "The result is recorded regardless of direction or magnitude; it is not "
                "used to retune this evaluated run."
            ),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(record, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
