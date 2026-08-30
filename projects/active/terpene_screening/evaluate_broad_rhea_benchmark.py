from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from projects.active.terpene_screening.broad_rhea_metrics import (  # noqa: E402
    candidate_ranking_context,
    evaluate_full_candidate_scores,
    positive_rank_map,
    summarize_query_metrics,
)
from projects.active.terpene_screening.fair_benchmark import (  # noqa: E402
    DEFAULT_BUDGETS,
    DEFAULT_TOP_PERCENTS,
    audit_exact_overlap,
    sha256_file,
)
from projects.active.terpene_screening.rank_open_world import (  # noqa: E402
    encode_model_reactions,
    load_feature_schema,
    load_models,
    load_protein_library,
    load_registered_reaction_feature_library,
)

DEFAULT_UNIVERSE = ROOT / "data/catalyst_candidate_universes/general_merged"
DEFAULT_BASE_MODEL = ROOT / "results/terpene_production_models/marts_adapted_drfp_pu"
DEFAULT_BENCHMARK_ROOT = ROOT / "results/broad_rhea_fair_benchmarks_v1"
DEFAULT_OUTPUT = ROOT / "results/broad_rhea_full_candidate_eval"


def parse_ints(value: str) -> tuple[int, ...]:
    result = tuple(sorted({int(part.strip()) for part in value.split(",") if part.strip()}))
    if not result or result[0] <= 0:
        raise ValueError("expected comma-separated positive integers")
    return result


def parse_floats(value: str) -> tuple[float, ...]:
    result = tuple(sorted({float(part.strip()) for part in value.split(",") if part.strip()}))
    if not result or any(item <= 0 or item > 1 for item in result):
        raise ValueError("expected comma-separated fractions in (0, 1]")
    return result


def encode_chunks(
    model: torch.nn.Module,
    values: np.ndarray,
    *,
    kind: str,
    device: torch.device,
    chunk_size: int,
) -> torch.Tensor:
    chunks: list[torch.Tensor] = []
    with torch.no_grad():
        for start in range(0, len(values), chunk_size):
            batch = torch.as_tensor(
                values[start : start + chunk_size], dtype=torch.float32, device=device
            )
            if kind == "protein":
                encoded = model.encode_proteins(batch)
            elif kind == "reaction":
                encoded = encode_model_reactions(model, batch, None)
            else:
                raise ValueError(kind)
            chunks.append(encoded.detach())
    return torch.cat(chunks, dim=0)


def load_ensemble_embeddings(
    model_dir: Path,
    protein_features: np.ndarray,
    reaction_features: np.ndarray,
    *,
    device: torch.device,
    feature_chunk_size: int,
) -> tuple[list[torch.Tensor], list[torch.Tensor], int]:
    models = load_models(model_dir.resolve() / "models", "production", device)
    if not models:
        raise ValueError(f"No production checkpoints under {model_dir / 'models'}")
    proteins: list[torch.Tensor] = []
    reactions: list[torch.Tensor] = []
    for model in models:
        proteins.append(
            encode_chunks(
                model,
                protein_features,
                kind="protein",
                device=device,
                chunk_size=feature_chunk_size,
            )
        )
        reactions.append(
            encode_chunks(
                model,
                reaction_features,
                kind="reaction",
                device=device,
                chunk_size=feature_chunk_size,
            )
        )
    return proteins, reactions, len(models)


def positive_maps(test_pairs: pd.DataFrame) -> tuple[dict[str, set[str]], dict[str, set[str]]]:
    by_reaction: dict[str, set[str]] = defaultdict(set)
    by_protein: dict[str, set[str]] = defaultdict(set)
    for protein_id, reaction_id in test_pairs[["protein_id", "reaction_id"]].itertuples(index=False):
        by_reaction[str(reaction_id)].add(str(protein_id))
        by_protein[str(protein_id)].add(str(reaction_id))
    return dict(by_reaction), dict(by_protein)


def score_queries(
    query_ids: list[str],
    *,
    query_index: dict[str, int],
    candidate_ids: list[str],
    positives_by_query: dict[str, set[str]],
    query_member_embeddings: list[torch.Tensor],
    candidate_member_embeddings: list[torch.Tensor],
    direction: str,
    budgets: tuple[int, ...],
    top_percents: tuple[float, ...],
    batch_size: int,
    device: torch.device,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    candidate_index, lexical_order = candidate_ranking_context(candidate_ids)
    records: list[dict[str, object]] = []
    positive_records: list[dict[str, object]] = []
    for start in range(0, len(query_ids), batch_size):
        batch_ids = query_ids[start : start + batch_size]
        rows = torch.as_tensor(
            [query_index[value] for value in batch_ids], dtype=torch.long, device=device
        )
        scores = None
        with torch.no_grad():
            for query_emb, candidate_emb in zip(
                query_member_embeddings, candidate_member_embeddings, strict=True
            ):
                member = query_emb[rows] @ candidate_emb.T
                scores = member if scores is None else scores + member
            values = (scores / len(query_member_embeddings)).cpu().numpy()
        for local_i, query_id in enumerate(batch_ids):
            positives = positives_by_query[query_id]
            metrics = evaluate_full_candidate_scores(
                values[local_i],
                candidate_ids,
                positives,
                budgets=budgets,
                top_percents=top_percents,
                candidate_index=candidate_index,
                lexical_order=lexical_order,
            )
            records.append({"direction": direction, "query_id": query_id, **metrics})
            per_positive = positive_rank_map(
                values[local_i],
                candidate_ids,
                positives,
                candidate_index=candidate_index,
                lexical_order=lexical_order,
            )
            for positive_id, rank in per_positive.items():
                positive_records.append({
                    "direction": direction,
                    "query_id": query_id,
                    "positive_id": positive_id,
                    "positive_rank": int(rank),
                    "candidate_count": int(len(candidate_ids)),
                    "positive_rank_fraction": float(rank / len(candidate_ids)),
                    "positive_reciprocal_rank": float(1.0 / rank),
                })
        print(
            f"{direction} {min(start + len(batch_ids), len(query_ids))}/{len(query_ids)}",
            flush=True,
        )
    return pd.DataFrame(records), pd.DataFrame(positive_records)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate one leakage-controlled Rhea benchmark cell over the full candidate universe."
    )
    parser.add_argument("--cell", required=True, help="Benchmark cell name under --benchmark-root")
    parser.add_argument("--benchmark-root", type=Path, default=DEFAULT_BENCHMARK_ROOT)
    parser.add_argument("--universe-dir", type=Path, default=DEFAULT_UNIVERSE)
    parser.add_argument("--r2e-model-dir", type=Path, default=DEFAULT_BASE_MODEL)
    parser.add_argument("--e2r-model-dir", type=Path, default=DEFAULT_BASE_MODEL)
    parser.add_argument("--r2e-reaction-feature-dir", type=Path, default=None)
    parser.add_argument("--e2r-reaction-feature-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--budgets", default=",".join(map(str, DEFAULT_BUDGETS)))
    parser.add_argument("--top-percents", default=",".join(map(str, DEFAULT_TOP_PERCENTS)))
    parser.add_argument("--query-batch-size", type=int, default=32)
    parser.add_argument("--feature-chunk-size", type=int, default=8192)
    parser.add_argument("--max-r2e-queries", type=int, default=0)
    parser.add_argument("--max-e2r-queries", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    budgets = parse_ints(args.budgets)
    top_percents = parse_floats(args.top_percents)
    if args.query_batch_size <= 0 or args.feature_chunk_size <= 0:
        raise ValueError("batch sizes must be positive")

    benchmark_root = args.benchmark_root.resolve()
    cell_dir = benchmark_root / args.cell
    manifest_path = cell_dir / "manifest.json"
    train_path = cell_dir / "train_pairs.csv"
    test_path = cell_dir / "test_pairs.csv"
    for path in (manifest_path, train_path, test_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not bool(manifest.get("valid")):
        raise RuntimeError(f"Benchmark cell is not valid: {manifest.get('violations')}")

    train_pairs = pd.read_csv(train_path, dtype=str).fillna("")
    test_pairs = pd.read_csv(test_path, dtype=str).fillna("")
    _, overlap = audit_exact_overlap(
        train_pairs.rename(columns={"reaction_id": "query_id", "protein_id": "candidate_id"}),
        test_pairs.rename(columns={"reaction_id": "query_id", "protein_id": "candidate_id"}),
    )
    if not overlap["generalization_claim_safe_exact_pair"]:
        raise RuntimeError("Exact train-test pair overlap detected at evaluation time")

    universe = args.universe_dir.resolve()
    device = torch.device(args.device)
    protein_features, protein_ids = load_protein_library(universe / "proteins")
    protein_index = {value: i for i, value in enumerate(protein_ids)}

    r2e_schema = load_feature_schema(args.r2e_model_dir.resolve())
    e2r_schema = load_feature_schema(args.e2r_model_dir.resolve())
    if int(r2e_schema.get("protein_feature_dimension") or protein_features.shape[1]) != protein_features.shape[1]:
        raise ValueError("R2E protein feature dimension differs from candidate universe")
    if int(e2r_schema.get("protein_feature_dimension") or protein_features.shape[1]) != protein_features.shape[1]:
        raise ValueError("E2R protein feature dimension differs from candidate universe")
    default_reaction_dir = universe / "reaction_features" / "drfp_categorical_v1"
    r2e_reaction_feature_dir = (
        args.r2e_reaction_feature_dir.resolve()
        if args.r2e_reaction_feature_dir is not None
        else default_reaction_dir
    )
    e2r_reaction_feature_dir = (
        args.e2r_reaction_feature_dir.resolve()
        if args.e2r_reaction_feature_dir is not None
        else default_reaction_dir
    )
    r2e_reaction_features, r2e_reaction_ids = load_registered_reaction_feature_library(
        r2e_reaction_feature_dir, r2e_schema
    )
    e2r_reaction_features, e2r_reaction_ids = load_registered_reaction_feature_library(
        e2r_reaction_feature_dir, e2r_schema
    )
    if r2e_reaction_ids != e2r_reaction_ids:
        raise ValueError("R2E and E2R reaction feature libraries have different candidate IDs/order")
    reaction_ids = r2e_reaction_ids
    reaction_index = {value: i for i, value in enumerate(reaction_ids)}

    protein_set = set(protein_ids)
    reaction_set = set(reaction_ids)
    missing_proteins = sorted(set(test_pairs["protein_id"]) - protein_set)
    missing_reactions = sorted(set(test_pairs["reaction_id"]) - reaction_set)
    if missing_proteins or missing_reactions:
        raise ValueError(
            f"Test cell is outside candidate universe: proteins={missing_proteins[:5]}, "
            f"reactions={missing_reactions[:5]}"
        )

    r2e_proteins, r2e_reactions, r2e_members = load_ensemble_embeddings(
        args.r2e_model_dir.resolve(),
        protein_features,
        r2e_reaction_features,
        device=device,
        feature_chunk_size=args.feature_chunk_size,
    )
    if (
        args.e2r_model_dir.resolve() == args.r2e_model_dir.resolve()
        and e2r_reaction_feature_dir == r2e_reaction_feature_dir
    ):
        e2r_proteins, e2r_reactions, e2r_members = r2e_proteins, r2e_reactions, r2e_members
    else:
        e2r_proteins, e2r_reactions, e2r_members = load_ensemble_embeddings(
            args.e2r_model_dir.resolve(),
            protein_features,
            e2r_reaction_features,
            device=device,
            feature_chunk_size=args.feature_chunk_size,
        )

    by_reaction, by_protein = positive_maps(test_pairs)
    r2e_queries = sorted(by_reaction)
    e2r_queries = sorted(by_protein)
    if args.max_r2e_queries > 0:
        r2e_queries = r2e_queries[: args.max_r2e_queries]
    if args.max_e2r_queries > 0:
        e2r_queries = e2r_queries[: args.max_e2r_queries]

    r2e_frame, r2e_positive_ranks = score_queries(
        r2e_queries,
        query_index=reaction_index,
        candidate_ids=protein_ids,
        positives_by_query=by_reaction,
        query_member_embeddings=r2e_reactions,
        candidate_member_embeddings=r2e_proteins,
        direction="reaction_to_enzyme",
        budgets=budgets,
        top_percents=top_percents,
        batch_size=args.query_batch_size,
        device=device,
    )
    e2r_frame, e2r_positive_ranks = score_queries(
        e2r_queries,
        query_index=protein_index,
        candidate_ids=reaction_ids,
        positives_by_query=by_protein,
        query_member_embeddings=e2r_proteins,
        candidate_member_embeddings=e2r_reactions,
        direction="enzyme_to_reaction",
        budgets=budgets,
        top_percents=top_percents,
        batch_size=args.query_batch_size,
        device=device,
    )

    output = args.output_dir.resolve() / args.cell
    output.mkdir(parents=True, exist_ok=True)
    query_metrics = pd.concat([r2e_frame, e2r_frame], ignore_index=True)
    query_metrics.to_csv(output / "query_metrics.csv", index=False)
    positive_ranks = pd.concat([r2e_positive_ranks, e2r_positive_ranks], ignore_index=True)
    positive_ranks.to_csv(output / "positive_ranks.csv", index=False)
    summaries = {
        "reaction_to_enzyme": summarize_query_metrics(
            r2e_frame, budgets=budgets, top_percents=top_percents
        ),
        "enzyme_to_reaction": summarize_query_metrics(
            e2r_frame, budgets=budgets, top_percents=top_percents
        ),
    }
    payload = {
        "cell": args.cell,
        "cell_manifest": manifest,
        "cell_manifest_sha256": sha256_file(manifest_path),
        "train_pairs_sha256": sha256_file(train_path),
        "test_pairs_sha256": sha256_file(test_path),
        "evaluation_overlap_audit": overlap,
        "candidate_universe": str(universe),
        "candidate_proteins": len(protein_ids),
        "candidate_reactions": len(reaction_ids),
        "r2e_model_dir": str(args.r2e_model_dir.resolve()),
        "e2r_model_dir": str(args.e2r_model_dir.resolve()),
        "r2e_reaction_feature_dir": str(r2e_reaction_feature_dir),
        "e2r_reaction_feature_dir": str(e2r_reaction_feature_dir),
        "r2e_ensemble_members": r2e_members,
        "e2r_ensemble_members": e2r_members,
        "budgets": budgets,
        "top_percents": top_percents,
        "positive_rank_rows": int(len(positive_ranks)),
        "positive_rank_output": str((output / "positive_ranks.csv").resolve()),
        "metrics": summaries,
    }
    (output / "summary.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
