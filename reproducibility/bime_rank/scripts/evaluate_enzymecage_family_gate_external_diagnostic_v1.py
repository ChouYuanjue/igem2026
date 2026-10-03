from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from projects.active.bridge.model.assets import ROOT

FAMILY_ROOT = ROOT / "results/enzymecage_family_response_v1"
CAGE_ROOT = ROOT / "results/fibre_vs_enzymecage_external_families_v1/cage"
OUT = FAMILY_ROOT / "external_high_confidence_diagnostic.json"

FILES = {
    "p450": (
        FAMILY_ROOT / "p450_broad_top64_probe/query_features_calibrated.csv",
        CAGE_ROOT / "p450/pretrain/test_P450_epoch_19.csv",
        CAGE_ROOT / "p450/finetune/test_P450_epoch_9.csv",
    ),
    "phosphatase": (
        FAMILY_ROOT / "phosphatase_broad_top64_probe/query_features_calibrated.csv",
        CAGE_ROOT / "phosphatase/pretrain/test_Phosphatase_epoch_19.csv",
        CAGE_ROOT / "phosphatase/finetune/test_Phosphatase_epoch_9.csv",
    ),
    "terpene": (
        FAMILY_ROOT / "terpene_broad_top64_probe/query_features_calibrated.csv",
        CAGE_ROOT / "terpene/pretrain/test_Terpene_epoch_19.csv",
        CAGE_ROOT / "terpene/finetune/test_Terpene_epoch_9.csv",
    ),
}


def query_metrics(frame: pd.DataFrame, score: str) -> pd.DataFrame:
    rows = []
    for query, group in frame.groupby("CANO_RXN_SMILES", sort=True):
        ordered = group.sort_values(
            [score, "UniprotID"],
            ascending=[False, True],
            kind="stable",
        )
        y = pd.to_numeric(
            ordered["Label"], errors="coerce"
        ).fillna(0).astype(int).to_numpy()
        pos = np.flatnonzero(y == 1) + 1
        rank = int(pos[0]) if len(pos) else 0
        rows.append(
            {
                "query": str(query),
                "rr": 0.0 if rank == 0 else 1.0 / rank,
                "hit10": int(rank > 0 and rank <= 10),
                "rank": rank,
            }
        )
    return pd.DataFrame(rows, columns=["query", "rr", "hit10", "rank"])


def main() -> None:
    result = {
        "schema": "enzymecage-high-confidence-family-direct-rank-diagnostic-v1",
        "status": "completed",
        "semantic_gate": (
            "calibrated cross-family winner and corresponding "
            "response_top10pct_mean_percentile >= 0.95"
        ),
        "purpose": (
            "external evaluation only: test whether a high family-adaptation "
            "response justifies replacing generic CAGE ranking with the "
            "family-finetuned CAGE ranking"
        ),
        "threshold_or_model_selection_from_external_labels": False,
        "families": {},
    }

    for family, (query_path, pretrain_path, finetune_path) in FILES.items():
        qf = pd.read_csv(query_path)
        active = qf[
            qf["family_calibrated_winner"].eq(family)
            & (
                qf[f"{family}_response_top10pct_mean_percentile"]
                >= 0.95
            )
        ]
        active_queries = set(active["CANO_RXN_SMILES"].astype(str))

        pre = pd.read_csv(
            pretrain_path,
            dtype={"CANO_RXN_SMILES": str, "UniprotID": str},
        ).fillna("")
        fine = pd.read_csv(
            finetune_path,
            dtype={"CANO_RXN_SMILES": str, "UniprotID": str},
        ).fillna("")
        pre_score = "pred_logit" if "pred_logit" in pre.columns else "pred"
        fine_score = "pred_logit" if "pred_logit" in fine.columns else "pred"

        common = pre[
            ["CANO_RXN_SMILES", "UniprotID", "Label", pre_score]
        ].merge(
            fine[
                ["CANO_RXN_SMILES", "UniprotID", fine_score]
            ],
            on=["CANO_RXN_SMILES", "UniprotID"],
            how="inner",
            suffixes=("_generic", "_finetune"),
        )
        label_col = (
            "Label_generic"
            if "Label_generic" in common.columns
            else "Label"
        )
        common["Label"] = common[label_col]
        common = common[
            common["CANO_RXN_SMILES"].isin(active_queries)
        ].copy()

        generic_col = (
            pre_score + "_generic"
            if pre_score + "_generic" in common.columns
            else pre_score
        )
        finetune_col = (
            fine_score + "_finetune"
            if fine_score + "_finetune" in common.columns
            else fine_score
        )
        generic = query_metrics(common, generic_col)
        finetune = query_metrics(common, finetune_col)
        merged = generic.merge(
            finetune, on="query", suffixes=("_generic", "_finetune")
        )

        result["families"][family] = {
            "external_queries_total": int(
                qf["CANO_RXN_SMILES"].nunique()
            ),
            "gate_active_queries": int(len(active_queries)),
            "gate_active_fraction": float(
                len(active_queries)
                / max(qf["CANO_RXN_SMILES"].nunique(), 1)
            ),
            "evaluable_active_queries": int(len(merged)),
            "generic_mrr": (
                float(merged["rr_generic"].mean())
                if len(merged)
                else None
            ),
            "finetune_mrr": (
                float(merged["rr_finetune"].mean())
                if len(merged)
                else None
            ),
            "delta_mrr": (
                float(
                    (
                        merged["rr_finetune"]
                        - merged["rr_generic"]
                    ).mean()
                )
                if len(merged)
                else None
            ),
            "generic_hit10": (
                float(merged["hit10_generic"].mean())
                if len(merged)
                else None
            ),
            "finetune_hit10": (
                float(merged["hit10_finetune"].mean())
                if len(merged)
                else None
            ),
            "delta_hit10": (
                float(
                    (
                        merged["hit10_finetune"]
                        - merged["hit10_generic"]
                    ).mean()
                )
                if len(merged)
                else None
            ),
            "improved": int(
                (
                    merged["rr_finetune"]
                    > merged["rr_generic"]
                ).sum()
            ),
            "worsened": int(
                (
                    merged["rr_finetune"]
                    < merged["rr_generic"]
                ).sum()
            ),
            "tied": int(
                (
                    merged["rr_finetune"]
                    == merged["rr_generic"]
                ).sum()
            ),
        }

    OUT.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n"
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
