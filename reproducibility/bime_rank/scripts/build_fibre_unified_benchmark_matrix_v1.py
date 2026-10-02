from __future__ import annotations

import argparse
import json
import math
import sys
from hashlib import blake2b
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from drfp import DrfpEncoder

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from projects.active.fibre.model.index import DEFAULT_INDEX, FibreCandidateIndex
from projects.active.fibre.runtime.cli import encode_reaction_with_audit, load_feature_schema

OUT = ROOT / "results/fibre_unified_benchmark_matrix_v1"
PREP = OUT / "prep"
MISSING_ESMC = PREP / "missing_external_esmc"
SCHEMA_SOURCE = ROOT / "results/terpene_production_models/marts_adapted_drfp_pu"
GENERAL_SEQS = ROOT / "data/catalyst_candidate_universes/general_merged/protein_sequences.tsv"

E405_PAIRS = ROOT / "results/bime_rank_unified_v1/enzyme405_complete226_augmented_v1/pair_scores.csv"
E405_CAGE_QM = ROOT / "results/bime_rank_unified_v1/enzyme405_complete226_augmented_v1/enzymecage_seed_query_metrics.csv"
ORPHAN_PAIRS = ROOT / "results/orphan335_fixed_pool_v1/pair_scores.csv"
ORPHAN_SUMMARY = ROOT / "results/orphan335_fixed_pool_v1/summary.json"
EXTERNAL_RECORD = ROOT / "reproducibility/bime_rank/records/FIBRE_CAGE_EXTERNAL_FAMILIES_V1_RESULT.json"
BROAD_SUITE_RECORD = ROOT / "reproducibility/bime_rank/records/FIBRE_BROAD_SUITE_ALL_CELLS_V1_RESULT.json"
CARE_RECORD = ROOT / "reproducibility/bime_rank/records/FIBRE_CARE_TASK2_V1_RESULT.json"


def legacy_hash(shingling):
    values = [int(blake2b(token, digest_size=4).hexdigest(), 16) for token in shingling]
    return np.asarray(values, dtype=np.uint32).view(np.int32)


def rank_metrics(frame: pd.DataFrame, score_col: str) -> dict[str, float]:
    rows = []
    for query, group in frame.groupby("query_id", sort=True):
        ordered = group.sort_values([score_col, "candidate_id"], ascending=[False, True], kind="stable")
        y = ordered["label"].astype(int).to_numpy()
        ranks = np.flatnonzero(y == 1) + 1
        positive_count = int(y.sum())
        best = int(ranks[0]) if len(ranks) else 0
        ap = 0.0
        if positive_count:
            ap = float(sum(y[:rank].sum() / rank for rank in ranks) / positive_count)
        ideal10 = sum(1.0 / math.log2(i + 2) for i in range(min(positive_count, 10)))
        dcg10 = sum(1.0 / math.log2(int(rank) + 1) for rank in ranks if rank <= 10)
        row = {
            "query_id": str(query),
            "candidate_count": int(len(group)),
            "positive_count": positive_count,
            "reciprocal_rank": 0.0 if best == 0 else 1.0 / best,
            "average_precision": ap,
            "ndcg_at_10": 0.0 if ideal10 == 0 else float(dcg10 / ideal10),
            "best_positive_rank": best,
        }
        for k in (1, 3, 5, 10, 20, 50, 100, 1000, 5000):
            row[f"hit_at_{k}"] = float(best > 0 and best <= k)
        rows.append(row)
    q = pd.DataFrame(rows)
    result = {
        "queries": int(len(q)),
        "candidate_rows": int(len(frame)),
        "positive_rows": int(frame["label"].sum()),
        "mrr": float(q["reciprocal_rank"].mean()),
        "map": float(q["average_precision"].mean()),
        "ndcg_at_10": float(q["ndcg_at_10"].mean()),
        "median_best_positive_rank": float(q["best_positive_rank"].replace(0, np.nan).median()),
    }
    for k in (1, 3, 5, 10, 20, 50, 100, 1000, 5000):
        result[f"hit_at_{k}"] = float(q[f"hit_at_{k}"].mean())
    return result


def encode_reactions(smiles: list[str], index: FibreCandidateIndex) -> torch.Tensor:
    DrfpEncoder.hash = staticmethod(legacy_hash)
    schema = load_feature_schema(SCHEMA_SOURCE)
    features = []
    for value in smiles:
        feature, audit = encode_reaction_with_audit(
            value,
            schema,
            failure_policy="strict",
            cache_dir=None,
        )
        if audit.drfp_status != "encoded" or bool(audit.fallback_used):
            raise RuntimeError(f"reaction encoding fallback: {audit}")
        features.append(feature)
    x = torch.as_tensor(np.stack(features).astype(np.float32), device=index.device)
    with torch.no_grad():
        return index.model.encode_reactions(x)


def build_missing_latents(index: FibreCandidateIndex) -> dict[str, torch.Tensor]:
    entries = pd.read_csv(MISSING_ESMC / "entries.csv", dtype=str).fillna("")
    entries["row"] = pd.to_numeric(entries["row"]).astype(int)
    raw = np.load(MISSING_ESMC / "embeddings.npy").astype(np.float32)
    with torch.no_grad():
        z = index.model.encode_proteins(torch.as_tensor(raw, device=index.device))
    return {
        str(entry): z[int(row)]
        for entry, row in entries[["Entry", "row"]].itertuples(index=False)
    }


def score_general_broad(
    source: Path,
    *,
    benchmark: str,
    index: FibreCandidateIndex,
    seq_to_pid: dict[str, str],
    missing_latent: dict[str, torch.Tensor],
) -> tuple[pd.DataFrame, dict[str, float]]:
    src = pd.read_csv(source, dtype=str).fillna("")
    if benchmark == "enzyme405_complete226":
        base = src[["CANO_RXN_SMILES", "UniprotID", "sequence", "Label"]].copy()
    elif benchmark == "orphan335":
        base = src[["CANO_RXN_SMILES", "UniprotID", "sequence", "label"]].copy()
        base = base.rename(columns={"label": "Label"})
    else:
        raise ValueError(benchmark)

    base["Label"] = pd.to_numeric(base["Label"], errors="raise").astype(int)
    base = base.drop_duplicates(["CANO_RXN_SMILES", "UniprotID"], keep="first").reset_index(drop=True)
    reactions = sorted(base["CANO_RXN_SMILES"].unique().tolist())
    reaction_z = encode_reactions(reactions, index)
    reaction_index = {value: i for i, value in enumerate(reactions)}

    unique_proteins = base[["UniprotID", "sequence"]].drop_duplicates("UniprotID")
    protein_vectors: list[torch.Tensor] = []
    protein_ids: list[str] = []
    missing = []
    for uid, seq in unique_proteins.itertuples(index=False):
        pid = seq_to_pid.get(str(seq))
        if pid is not None and pid in index.protein_index:
            protein_vectors.append(index.protein_embeddings[index.protein_index[pid]])
            protein_ids.append(str(uid))
        elif str(uid) in missing_latent:
            protein_vectors.append(missing_latent[str(uid)])
            protein_ids.append(str(uid))
        else:
            missing.append(str(uid))
    if missing:
        raise RuntimeError(f"{benchmark}: missing {len(missing)} protein embeddings, first={missing[:5]}")
    pz = torch.stack(protein_vectors, dim=0)
    protein_index = {uid: i for i, uid in enumerate(protein_ids)}

    ri = torch.as_tensor(
        [reaction_index[x] for x in base["CANO_RXN_SMILES"].astype(str)],
        device=index.device,
        dtype=torch.long,
    )
    pi = torch.as_tensor(
        [protein_index[x] for x in base["UniprotID"].astype(str)],
        device=index.device,
        dtype=torch.long,
    )
    with torch.no_grad():
        scores = (reaction_z.index_select(0, ri) * pz.index_select(0, pi)).sum(dim=1).float().cpu().numpy()

    scored = pd.DataFrame(
        {
            "query_id": base["CANO_RXN_SMILES"].astype(str),
            "candidate_id": base["UniprotID"].astype(str),
            "label": base["Label"].astype(int),
            "broad_score": scores,
        }
    )
    scored.to_csv(OUT / f"{benchmark}_broad_pair_scores.csv.gz", index=False, compression="gzip")
    metrics = rank_metrics(scored, "broad_score")
    return scored, metrics


def general_row(
    *,
    benchmark_group: str,
    benchmark: str,
    task: str,
    model: str,
    source_class: str,
    metrics: dict[str, object],
    candidate_scope: str,
    notes: str = "",
) -> dict[str, object]:
    row = {
        "benchmark_group": benchmark_group,
        "benchmark": benchmark,
        "task": task,
        "model": model,
        "model_class": "general",
        "source_class": source_class,
        "candidate_scope": candidate_scope,
        "notes": notes,
    }
    for key in (
        "queries",
        "candidate_rows",
        "positive_rows",
        "candidate_ecs",
        "mrr",
        "map",
        "ndcg_at_10",
        "median_best_positive_rank",
        "hit_at_1",
        "hit_at_3",
        "hit_at_5",
        "hit_at_10",
        "hit_at_20",
        "hit_at_50",
        "hit_at_100",
        "hit_at_1000",
        "hit_at_5000",
    ):
        row[key] = metrics.get(key)
    return row


def build_matrix(index: FibreCandidateIndex) -> tuple[pd.DataFrame, dict[str, object]]:
    general_seq = pd.read_csv(GENERAL_SEQS, sep="\t", dtype=str).fillna("")
    seq_to_pid = dict(zip(general_seq["sequence"].astype(str), general_seq["protein_id"].astype(str)))
    missing_latent = build_missing_latents(index)

    _, e405_broad = score_general_broad(
        E405_PAIRS,
        benchmark="enzyme405_complete226",
        index=index,
        seq_to_pid=seq_to_pid,
        missing_latent=missing_latent,
    )
    _, orphan_broad = score_general_broad(
        ORPHAN_PAIRS,
        benchmark="orphan335",
        index=index,
        seq_to_pid=seq_to_pid,
        missing_latent=missing_latent,
    )

    rows: list[dict[str, object]] = []
    rows.append(general_row(
        benchmark_group="internal_author_pool",
        benchmark="Enzyme-405 complete226",
        task="reaction_to_protein",
        model="FIBRE Broad Core",
        source_class="ours_general",
        metrics=e405_broad,
        candidate_scope="immutable Enzyme-405 complete226 candidate pool",
        notes="pure frozen Broad Core; reaction-center/EnzGFM specialists excluded",
    ))

    cage_q = pd.read_csv(E405_CAGE_QM)
    cage_q = cage_q[cage_q["direction"].eq("reaction_to_enzyme")].copy()
    cage_metrics = {
        "queries": int(cage_q["query_id"].nunique()),
        "candidate_rows": int(len(pd.read_csv(E405_PAIRS))),
        "positive_rows": int(pd.read_csv(E405_PAIRS)["label"].astype(int).sum()),
        "mrr": float(cage_q["reciprocal_rank"].mean()),
        "map": float(cage_q["average_precision"].mean()),
        "ndcg_at_10": float(cage_q["ndcg_at_10"].mean()),
        "median_best_positive_rank": float(cage_q.groupby("query_id")["best_positive_rank"].mean().median()),
    }
    for k in (1, 3, 5, 10, 20, 50):
        cage_metrics[f"hit_at_{k}"] = float(cage_q[f"hit_at_{k}"].mean())
    rows.append(general_row(
        benchmark_group="internal_author_pool",
        benchmark="Enzyme-405 complete226",
        task="reaction_to_protein",
        model="EnzymeCAGE generic pretrain",
        source_class="external_general",
        metrics=cage_metrics,
        candidate_scope="immutable Enzyme-405 complete226 candidate pool",
        notes="mean across fixed official seeds 40-44; no family fine-tuning",
    ))

    rows.append(general_row(
        benchmark_group="internal_author_pool",
        benchmark="Orphan-335",
        task="reaction_to_protein",
        model="FIBRE Broad Core",
        source_class="ours_general",
        metrics=orphan_broad,
        candidate_scope="immutable author Orphan-335 candidate pool",
        notes="pure frozen Broad Core; reaction-center specialist excluded",
    ))
    orphan_summary = json.loads(ORPHAN_SUMMARY.read_text())
    selenzyme = dict(orphan_summary["selenzyme_metrics"]["reaction_to_enzyme"])
    rows.append(general_row(
        benchmark_group="internal_author_pool",
        benchmark="Orphan-335",
        task="reaction_to_protein",
        model="Selenzyme",
        source_class="external_general",
        metrics={
            "queries": selenzyme.get("query_count"),
            "candidate_rows": selenzyme.get("candidate_rows"),
            "positive_rows": selenzyme.get("positive_rows"),
            "mrr": selenzyme.get("mrr"),
            "map": selenzyme.get("map"),
            "ndcg_at_10": selenzyme.get("ndcg_at_10"),
            "median_best_positive_rank": selenzyme.get("median_best_positive_rank"),
            **{f"hit_at_{k}": selenzyme.get(f"hit_at_{k}") for k in (1, 3, 5, 10, 20, 50)},
        },
        candidate_scope="immutable author Orphan-335 candidate pool",
        notes="author Selenzyme score; general comparator",
    ))

    external = json.loads(EXTERNAL_RECORD.read_text())
    for family, label in (("p450", "P450 external"), ("phosphatase", "Phosphatase external"), ("terpene", "Terpene external")):
        block = external["families"][family]
        rows.append(general_row(
            benchmark_group="external_family",
            benchmark=label,
            task="reaction_to_protein",
            model="FIBRE Broad Core",
            source_class="ours_general",
            metrics=block["broad"],
            candidate_scope="official EnzymeCAGE external-test candidate pool",
            notes="same-pool general model comparison",
        ))
        generic = block["cage_pretrain"]["enzymecage"]
        rows.append(general_row(
            benchmark_group="external_family",
            benchmark=label,
            task="reaction_to_protein",
            model="EnzymeCAGE generic pretrain",
            source_class="external_general",
            metrics=generic,
            candidate_scope="official EnzymeCAGE external-test candidate pool",
            notes=f"pair coverage={block['cage_pretrain']['pair_coverage']:.6f}; family fine-tune excluded from baseline",
        ))

    broad_suite = json.loads(BROAD_SUITE_RECORD.read_text())
    for cell, metrics in broad_suite["per_cell"].items():
        rows.append(general_row(
            benchmark_group="broad_rhea_suite",
            benchmark=cell,
            task="reaction_to_protein_full_universe",
            model="FIBRE Broad Core",
            source_class="ours_general",
            metrics={
                "queries": metrics["query_instances"],
                "positive_rows": metrics["positive_pairs"],
                **{f"hit_at_{k}": metrics.get(f"broad_hit_at_{k}") for k in (10, 20, 50, 100, 1000, 5000)},
            },
            candidate_scope=f"full {broad_suite['current_broad_candidate_universe']}-protein Broad universe",
            notes=f"joint-clean queries against CAGE2023={metrics['joint_clean_query_count']}",
        ))

    care = json.loads(CARE_RECORD.read_text())
    for split, metrics in care["splits"].items():
        rows.append(general_row(
            benchmark_group="external_task_transfer",
            benchmark=f"CARE Task2 {split}",
            task="reaction_to_EC",
            model="FIBRE Broad Core",
            source_class="ours_general",
            metrics=metrics,
            candidate_scope=f"official CARE {metrics['candidate_ecs']}-EC candidate set",
            notes="EC-cluster transfer task; numerically separate from protein retrieval",
        ))

    matrix = pd.DataFrame(rows)
    matrix.to_csv(OUT / "benchmark_matrix.csv", index=False)
    display_cols = [
        "benchmark_group", "benchmark", "model", "queries", "candidate_rows",
        "mrr", "map", "ndcg_at_10", "hit_at_1", "hit_at_10", "hit_at_50",
        "hit_at_100", "hit_at_1000", "hit_at_5000", "notes",
    ]
    view = matrix[display_cols].copy()
    def fmt(value):
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return ""
        if isinstance(value, (float, np.floating)):
            return f"{float(value):.4f}"
        return str(value).replace("|", "\\|").replace("\n", " ")
    header = "| " + " | ".join(display_cols) + " |"
    sep = "| " + " | ".join(["---"] * len(display_cols)) + " |"
    lines = [header, sep]
    for row in view.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(fmt(v) for v in row) + " |")
    (OUT / "benchmark_matrix.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    summary = {
        "schema": "fibre-unified-general-benchmark-matrix-v1",
        "status": "completed",
        "policy": {
            "main_matrix": "general-purpose models only",
            "broad_core_is_expert": False,
            "excluded_from_general_comparison": [
                "EnzymeCAGE family-finetuned P450",
                "EnzymeCAGE family-finetuned phosphatase",
                "EnzymeCAGE family-finetuned terpene",
                "reaction-center augmented Catalyst checkpoints",
                "EnzGFM routed/center specialists",
                "TPS application coordinate",
            ],
            "specialized_methods_destination": "expert registry / expert-type ablation",
        },
        "pure_general_rescore": {
            "enzyme405_complete226": e405_broad,
            "orphan335": orphan_broad,
            "missing_external_proteins_encoded": int(len(missing_latent)),
            "reaction_feature_fallbacks": 0,
        },
        "matrix_rows": int(len(matrix)),
        "benchmark_names": sorted(matrix["benchmark"].unique().tolist()),
        "models_in_main_matrix": sorted(matrix["model"].unique().tolist()),
        "artifacts": {
            "csv": "results/fibre_unified_benchmark_matrix_v1/benchmark_matrix.csv",
            "markdown": "results/fibre_unified_benchmark_matrix_v1/benchmark_matrix.md",
        },
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    return matrix, summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    index = FibreCandidateIndex(device=args.device)
    matrix, summary = build_matrix(index)
    print(matrix[["benchmark", "model", "queries", "mrr", "hit_at_10"]].to_string(index=False))
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
