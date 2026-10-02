from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import yaml

from projects.active.fibre.model.assets import ROOT

CAGE = ROOT / "external_repos/EnzymeCAGE"
if str(CAGE) not in sys.path:
    sys.path.insert(0, str(CAGE))

from reproducibility.bime_rank.scripts.extract_enzymecage_family_response_v1 import (
    build_dataset,
    resolve_config,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--config",
        default=str(
            ROOT
            / "results/enzymecage_reaction_family_response_v1/full_outer_assets/probe.yaml"
        ),
    )
    ap.add_argument(
        "--output",
        default=str(
            ROOT
            / "results/enzymecage_reaction_family_response_v1/full_outer_assets/probe_strict.csv"
        ),
    )
    args = ap.parse_args()

    config_path = Path(args.config)
    output_path = Path(args.output)
    conf = resolve_config(config_path)
    _, dataset, audit = build_dataset(conf)
    frame = getattr(dataset, "df_data").copy().reset_index(drop=True)

    failures: list[dict[str, object]] = []
    valid_rows: list[int] = []
    for i in range(len(dataset)):
        try:
            _ = dataset[i]
            valid_rows.append(i)
        except BaseException as exc:
            row = frame.iloc[i]
            failures.append(
                {
                    "index": int(i),
                    "reaction_id": str(row.get("reaction_id", "")),
                    "reaction": str(row.get("CANO_RXN_SMILES", "")),
                    "reason": f"{type(exc).__name__}:{exc}",
                }
            )

    strict = frame.iloc[valid_rows].copy()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    strict.to_csv(output_path, index=False)

    raw = yaml.safe_load(config_path.read_text())
    raw["data_path"] = str(output_path.resolve())
    config_path.write_text(yaml.safe_dump(raw, sort_keys=False))

    result = {
        "schema": "enzymecage-reaction-probe-dataset-preflight-v1",
        "status": "completed",
        "dataset_rows_before": int(len(dataset)),
        "strict_rows": int(len(strict)),
        "failed_rows": int(len(failures)),
        "failures": failures,
        "dataset_build_audit": audit,
        "policy": (
            "A failed optional EnzymeCAGE reaction probe becomes unavailable; "
            "the corresponding Broad/FIBRE benchmark query remains in the test set."
        ),
        "labels_used": False,
    }
    result_path = output_path.parent / "dataset_preflight.json"
    result_path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
