from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Mapping

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from projects.active.terpene_screening.model_capability_registry import DEFAULT_SCENARIOS

ENZYMECAGE_PAPER_DOI = "10.1038/s41929-026-01478-y"
LOCAL_REPRODUCTION_PATH = ROOT / "projects/active/terpene_screening/ENZYMECAGE_LOCAL_REPRODUCTION_BASELINE_V1.json"
LOCAL_REPRODUCTION = json.loads(LOCAL_REPRODUCTION_PATH.read_text(encoding="utf-8"))

PAPER_REPORTED_BASELINES: tuple[dict[str, object], ...] = (
    {
        "scenario_id": "enzyme405",
        "model": "EnzymeCAGE",
        "status": "complete",
        "source_type": "paper_reported",
        "source": {
            "doi": ENZYMECAGE_PAPER_DOI,
            "location": "Supplementary Table 4",
            "repository": "https://github.com/GENTEL-lab/EnzymeCAGE",
        },
        "protocol": "author Enzyme-405 evaluation",
        "metrics": {
            "top10_sr": 0.5797,
            "top1_percent_ef": 36.6031,
            "top10_dcg": 0.4523,
        },
        "common_ir_metrics": None,
        "comparison_role": "context_only_author_report_not_primary_reproducible_baseline",
        "local_reproduction_evidence": LOCAL_REPRODUCTION["enzyme405_100_local_reconstruction"],
        "rerun_status": "best_available_100_reaction_local_reconstruction_complete_full_author_equivalent_pending",
        "note": (
            "These metrics are author-reported context, not a local rerun and not the primary reproducible comparison. "
            "The attached local_reproduction_evidence is the primary directly reproducible EnzymeCAGE evidence where support matches. "
            "Common IR metrics for EnzymeCAGE remain unavailable unless raw local prediction rows are retained."
        ),
    },
)


def _na_reason(scenario_id: str, candidate_scope: str, directions: tuple[str, ...]) -> str:
    if scenario_id == "enzymecage_official":
        return (
            "Exact author prediction-level outputs or a complete author-equivalent local rerun are not "
            "available for this registry row; missing author-native/common-IR values must not be imputed."
        )
    direction_note = ""
    if "enzyme_to_reaction" in directions:
        direction_note = " The registered scenario also includes E2R, which is not an author-native EnzymeCAGE retrieval protocol."
    if candidate_scope == "full_general_universe":
        return (
            "EnzymeCAGE does not provide complete author-supported pocket/model inputs for the registered "
            "185,918-protein / 11,081-reaction full candidate universe, so a same-universe score would require "
            "changing candidate support rather than reproducing the baseline." + direction_note
        )
    if candidate_scope == "frozen_tps_candidate_pool":
        return (
            "The frozen TPS candidate pool/task is not the EnzymeCAGE author evaluation reservoir; exact "
            "same-support author-model predictions are unavailable, so no score is fabricated." + direction_note
        )
    if candidate_scope == "official_pair_reservoir":
        return (
            "Exact author prediction-level outputs are unavailable for this registered reservoir and a complete "
            "author-equivalent rerun has not been established; no score is imputed."
        )
    return (
        "This scenario is outside the author-supported EnzymeCAGE evaluation/candidate protocol; exact same-support "
        "predictions are unavailable, so the mandatory baseline is reported as N/A rather than silently omitted."
        + direction_note
    )


def unavailable_baselines() -> tuple[dict[str, object], ...]:
    existing = {str(record["scenario_id"]) for record in PAPER_REPORTED_BASELINES}
    records: list[dict[str, object]] = []
    for scenario in DEFAULT_SCENARIOS:
        if scenario.scenario_id in existing:
            continue
        records.append(
            {
                "scenario_id": scenario.scenario_id,
                "model": "EnzymeCAGE",
                "status": "na",
                "source_type": "not_applicable",
                "protocol": scenario.role,
                "metrics": None,
                "common_ir_metrics": None,
                "incompatibility_reason": _na_reason(
                    scenario.scenario_id, scenario.candidate_scope, scenario.directions
                ),
            }
        )
    return tuple(records)


def validate_record(record: Mapping[str, object]) -> list[str]:
    errors: list[str] = []
    for key in ("scenario_id", "model", "status", "source_type", "protocol", "metrics"):
        if key not in record:
            errors.append(f"missing {key}")
    status = str(record.get("status", ""))
    source_type = str(record.get("source_type", ""))
    if status not in {"complete", "na"}:
        errors.append(f"invalid status={status!r}")
    if source_type not in {"paper_reported", "author_prediction", "local_rerun", "not_applicable"}:
        errors.append(f"invalid source_type={source_type!r}")
    metrics = record.get("metrics")
    if status == "complete" and (not isinstance(metrics, Mapping) or not metrics):
        errors.append("complete metrics must be a non-empty object")
    if status == "na":
        if metrics not in (None, {}):
            errors.append("N/A record cannot contain fabricated metrics")
        if not str(record.get("incompatibility_reason", "")).strip():
            errors.append("N/A record requires incompatibility_reason")
    if source_type == "paper_reported" and record.get("common_ir_metrics") not in (None, {}):
        errors.append("paper-reported record cannot invent common_ir_metrics absent from the source")
    return errors


def payload() -> dict[str, object]:
    records = list(PAPER_REPORTED_BASELINES) + list(unavailable_baselines())
    errors = [error for record in records for error in validate_record(record)]
    scenario_ids = [str(record["scenario_id"]) for record in records]
    registered = [scenario.scenario_id for scenario in DEFAULT_SCENARIOS]
    if sorted(scenario_ids) != sorted(registered):
        errors.append("baseline provenance must contain exactly one EnzymeCAGE row for every registered scenario")
    if len(scenario_ids) != len(set(scenario_ids)):
        errors.append("duplicate scenario_id in baseline provenance")
    if errors:
        raise ValueError("; ".join(errors))
    return {
        "evidence_policy": {
            "enzymecage_row_for_every_registered_scenario": True,
            "paper_reported_is_not_local_rerun": True,
            "common_metrics_require_prediction_level_data_or_exact_rerun": True,
            "missing_metrics_are_never_imputed": True,
            "incompatible_or_unsupported_protocol_is_explicit_na": True,
        },
        "records": records,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Emit provenance-safe external baseline records.")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    data = payload()
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(data, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
