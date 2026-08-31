from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Mapping

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from projects.active.terpene_screening.benchmark_baseline_provenance import payload as baseline_payload

DEFAULT_FREEZE = ROOT / "projects/active/terpene_screening/ENZYME405_CLEANROOM_FREEZE_V1.json"
DEFAULT_RESULT = ROOT / "results/enzyme405_cleanroom_selected_confirmatory_v1/full_official/summary.json"
DEFAULT_BOOTSTRAP = ROOT / "results/enzyme405_cleanroom_selected_confirmatory_v1/full_official/neural_bootstrap.json"
DEFAULT_SEQUENCE_AUDIT = ROOT / "results/enzymecage_405_sequence_consistency_v1/summary.json"
DEFAULT_LOCAL_REPRODUCTION = ROOT / "projects/active/terpene_screening/ENZYMECAGE_LOCAL_REPRODUCTION_BASELINE_V1.json"
DEFAULT_OUTPUT = ROOT / "results/enzyme405_evidence_chain_audit_v1/summary.json"


def _load(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _require_equal(errors: list[str], label: str, observed: object, expected: object) -> None:
    if observed != expected:
        errors.append(f"{label}: observed={observed!r}, expected={expected!r}")


def _require_close(errors: list[str], label: str, observed: object, expected: object, atol: float = 1e-12) -> None:
    if observed is None or expected is None or not math.isclose(float(observed), float(expected), rel_tol=0.0, abs_tol=atol):
        errors.append(f"{label}: observed={observed!r}, expected={expected!r}")


def audit_evidence_chain(
    freeze: Mapping[str, object],
    result: Mapping[str, object],
    bootstrap: Mapping[str, object],
    sequence_audit: Mapping[str, object],
    baseline: Mapping[str, object],
    local_reproduction: Mapping[str, object],
) -> dict[str, object]:
    errors: list[str] = []

    _require_equal(errors, "freeze.selected_before_enzyme405_reveal", freeze.get("selected_before_enzyme405_reveal"), True)
    _require_equal(errors, "result.reservoir_mode", result.get("reservoir_mode"), "full_official")
    _require_equal(errors, "result.target_labels_used_for_routing", result.get("target_labels_used_for_routing"), False)

    selected_candidate = str(freeze.get("selected_candidate", ""))
    model_dir = Path(str(result.get("model_dir", "")))
    _require_equal(errors, "selected candidate/model directory", model_dir.name, selected_candidate)

    model_summary = result.get("model_summary")
    if not isinstance(model_summary, Mapping):
        errors.append("result.model_summary missing or invalid")
        model_summary = {}
    _require_equal(errors, "model target labels read", model_summary.get("target_benchmark_labels_read"), False)
    _require_equal(errors, "model target metadata used", model_summary.get("target_benchmark_metadata_used_for_training"), False)
    _require_equal(errors, "full clean retraining dev fold", model_summary.get("dev_fold"), -1)
    _require_equal(errors, "full clean retraining source pairs", model_summary.get("n_source_pairs"), 218537)
    _require_equal(errors, "full clean retraining train pairs", model_summary.get("n_train_pairs"), 218537)

    recipe = freeze.get("selected_recipe")
    if not isinstance(recipe, Mapping):
        errors.append("freeze.selected_recipe missing or invalid")
        recipe = {}
    model_cfg = model_summary.get("model_config") if isinstance(model_summary.get("model_config"), Mapping) else {}
    training = model_summary.get("training") if isinstance(model_summary.get("training"), Mapping) else {}

    for key in ("hidden_dim", "embedding_dim", "dropout"):
        _require_equal(errors, f"recipe.model_config.{key}", model_cfg.get(key), recipe.get(key))
    for key in (
        "epochs",
        "steps_per_epoch",
        "reaction_batch_size",
        "protein_batch_size",
        "neighbor_k",
        "hard_negatives",
        "random_negatives",
        "temperature",
        "topk",
        "topk_weight",
        "margin",
        "r2e_weight",
        "learning_rate",
        "weight_decay",
    ):
        _require_equal(errors, f"recipe.training.{key}", training.get(key), recipe.get(key))
    novelty = training.get("reaction_novelty_replay") if isinstance(training.get("reaction_novelty_replay"), Mapping) else {}
    _require_equal(errors, "recipe.reaction_novelty_repeat", novelty.get("repeat"), recipe.get("reaction_novelty_repeat"))

    neural = result.get("neural_metrics")
    protocol_aware = result.get("protocol_aware_metrics")
    if not isinstance(neural, Mapping) or not isinstance(neural.get("enzymecage_native_r2e"), Mapping):
        errors.append("result.neural_metrics.enzymecage_native_r2e missing")
        native = {}
    else:
        native = neural["enzymecage_native_r2e"]
    if not isinstance(protocol_aware, Mapping):
        errors.append("result.protocol_aware_metrics missing")

    _require_equal(errors, "bootstrap.score", bootstrap.get("score"), "neural_score_only")
    _require_equal(errors, "bootstrap.queries", bootstrap.get("queries"), result.get("queries"))
    bootstrap_metrics = bootstrap.get("metrics") if isinstance(bootstrap.get("metrics"), Mapping) else {}
    for boot_name, native_name in {
        "native_sr1": "top1_sr",
        "native_sr3": "top3_sr",
        "native_sr5": "top5_sr",
        "native_sr10": "top10_sr",
        "native_dcg10": "top10_dcg",
        "native_ef1": "top1_percent_ef",
    }.items():
        entry = bootstrap_metrics.get(boot_name)
        if not isinstance(entry, Mapping):
            errors.append(f"bootstrap metric missing: {boot_name}")
            continue
        _require_close(errors, f"bootstrap {boot_name} estimate", entry.get("estimate"), native.get(native_name))

    _require_equal(errors, "sequence audit automatic correction", sequence_audit.get("automatic_sequence_correction_performed"), False)
    _require_equal(errors, "sequence audit benchmark_uids", sequence_audit.get("benchmark_uids"), result.get("candidate_uids"))

    baseline_records = baseline.get("records") if isinstance(baseline.get("records"), list) else []
    cage = next((row for row in baseline_records if row.get("scenario_id") == "enzyme405" and row.get("model") == "EnzymeCAGE"), None)
    paper_context: dict[str, object] = {}
    if not isinstance(cage, Mapping):
        errors.append("EnzymeCAGE enzyme405 provenance row missing")
    else:
        _require_equal(errors, "EnzymeCAGE source type", cage.get("source_type"), "paper_reported")
        _require_equal(
            errors,
            "EnzymeCAGE paper comparison role",
            cage.get("comparison_role"),
            "context_only_author_report_not_primary_reproducible_baseline",
        )
        cage_metrics = cage.get("metrics") if isinstance(cage.get("metrics"), Mapping) else {}
        paper_context = {
            "role": "context_only_not_used_for_reproducible_delta",
            "metrics": dict(cage_metrics),
        }

    _require_equal(errors, "local reproduction model selection", local_reproduction.get("model_selection_allowed"), False)
    _require_equal(
        errors,
        "paper metrics role",
        local_reproduction.get("paper_reported_metrics_role"),
        "context_only_not_used_for_reproducible_delta",
    )
    local = local_reproduction.get("enzyme405_100_local_reconstruction")
    reproducible_comparison: dict[str, object] = {}
    if not isinstance(local, Mapping):
        errors.append("enzyme405_100 local reproduction evidence missing")
    else:
        support = local.get("support") if isinstance(local.get("support"), Mapping) else {}
        local_cage = local.get("enzymecage") if isinstance(local.get("enzymecage"), Mapping) else {}
        local_catalyst = local.get("catalyst_frozen_same_support") if isinstance(local.get("catalyst_frozen_same_support"), Mapping) else {}
        _require_equal(errors, "local reconstruction valid reactions", support.get("valid_reactions"), 99)
        _require_equal(errors, "same-support Catalyst model", local_catalyst.get("model"), selected_candidate)
        _require_equal(errors, "same-support Catalyst role", local_catalyst.get("evaluation_role"), "post_reveal_descriptive_only")
        for metric in ("top5_sr", "top10_sr"):
            ours = local_catalyst.get(metric)
            theirs = local_cage.get(metric)
            if ours is None or theirs is None:
                errors.append(f"local reproduction comparison metric missing: {metric}")
                continue
            delta = float(ours) - float(theirs)
            reproducible_comparison[metric] = {
                "catalyst_frozen_same_support": float(ours),
                "enzymecage_local_reconstruction": float(theirs),
                "catalyst_minus_enzymecage": delta,
                "leader": "Catalyst" if delta > 0 else "EnzymeCAGE" if delta < 0 else "tie",
                "denominator_reactions": int(support.get("valid_reactions", 0)),
            }

    return {
        "status": "pass" if not errors else "fail",
        "errors": errors,
        "audit_role": "post_reveal_evidence_chain_only",
        "model_selection_allowed": False,
        "reranking_or_hyperparameter_changes_allowed": False,
        "raw_benchmark_labels_read_by_this_audit": False,
        "primary_score_family": "neural_score_only",
        "protocol_aware_shortcut_excluded_from_comparative_claims": True,
        "selected_candidate": selected_candidate,
        "full_clean_train_pairs": model_summary.get("n_train_pairs"),
        "queries": result.get("queries"),
        "candidate_uids": result.get("candidate_uids"),
        "sequence_provenance": {
            "reference_covered_uids": sequence_audit.get("reference_covered_uids"),
            "exact_match_uids": sequence_audit.get("exact_match_uids"),
            "mismatch_uids": sequence_audit.get("mismatch_uids"),
            "mismatched_positive_uids": sequence_audit.get("mismatched_positive_uids"),
            "reference_missing_uids": sequence_audit.get("reference_missing_uids"),
            "automatic_sequence_correction_performed": sequence_audit.get("automatic_sequence_correction_performed"),
        },
        "reproducible_same_support_comparison": reproducible_comparison,
        "paper_metric_context": paper_context,
        "interpretation": (
            "A passing audit establishes lineage/protocol consistency of the already revealed clean Enzyme-405 result. "
            "It does not authorize post-reveal model selection. Sequence mismatches remain observational provenance findings, "
            "not grounds for silently modifying the benchmark."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit the frozen Enzyme-405 cleanroom evidence chain without rereading raw labels.")
    parser.add_argument("--freeze", type=Path, default=DEFAULT_FREEZE)
    parser.add_argument("--result", type=Path, default=DEFAULT_RESULT)
    parser.add_argument("--bootstrap", type=Path, default=DEFAULT_BOOTSTRAP)
    parser.add_argument("--sequence-audit", type=Path, default=DEFAULT_SEQUENCE_AUDIT)
    parser.add_argument("--local-reproduction", type=Path, default=DEFAULT_LOCAL_REPRODUCTION)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    report = audit_evidence_chain(
        _load(args.freeze),
        _load(args.result),
        _load(args.bootstrap),
        _load(args.sequence_audit),
        baseline_payload(),
        _load(args.local_reproduction),
    )
    report["sources"] = {
        "freeze": str(args.freeze.resolve()),
        "result": str(args.result.resolve()),
        "bootstrap": str(args.bootstrap.resolve()),
        "sequence_audit": str(args.sequence_audit.resolve()),
        "local_reproduction": str(args.local_reproduction.resolve()),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if report["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
