from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Mapping

ENZYMECAGE_PAPER_DOI = "10.1038/s41929-026-01478-y"

PAPER_REPORTED_BASELINES: tuple[dict[str, object], ...] = (
    {
        "scenario_id": "enzyme405",
        "model": "EnzymeCAGE",
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
        "rerun_status": "author_prediction_or_full_feature_rerun_pending",
        "note": (
            "These are author-reported values, not a local rerun. Common IR metrics "
            "must remain unavailable until author predictions or an exact full rerun are obtained."
        ),
    },
)


def validate_record(record: Mapping[str, object]) -> list[str]:
    errors: list[str] = []
    for key in ("scenario_id", "model", "source_type", "protocol", "metrics"):
        if key not in record:
            errors.append(f"missing {key}")
    source_type = str(record.get("source_type", ""))
    if source_type not in {"paper_reported", "author_prediction", "local_rerun"}:
        errors.append(f"invalid source_type={source_type!r}")
    metrics = record.get("metrics")
    if not isinstance(metrics, Mapping) or not metrics:
        errors.append("metrics must be a non-empty object")
    if source_type == "paper_reported" and record.get("common_ir_metrics") not in (None, {}):
        errors.append("paper-reported record cannot invent common_ir_metrics absent from the source")
    return errors


def payload() -> dict[str, object]:
    records = list(PAPER_REPORTED_BASELINES)
    errors = [error for record in records for error in validate_record(record)]
    if errors:
        raise ValueError("; ".join(errors))
    return {
        "evidence_policy": {
            "paper_reported_is_not_local_rerun": True,
            "common_metrics_require_prediction_level_data_or_exact_rerun": True,
            "missing_metrics_are_never_imputed": True,
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
