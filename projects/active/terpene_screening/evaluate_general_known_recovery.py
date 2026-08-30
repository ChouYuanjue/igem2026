from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from projects.active.terpene_screening.rank_open_world import (  # noqa: E402
    encode_model_reactions,
    load_feature_schema,
    load_models,
    load_protein_library,
    load_registered_reaction_feature_library,
)

DEFAULT_UNIVERSE = ROOT / "data/catalyst_candidate_universes/general_merged"
DEFAULT_MODEL = ROOT / "results/terpene_production_models/marts_adapted_drfp_pu"
DEFAULT_TRAINING = DEFAULT_MODEL / "training_pairs.csv"
DEFAULT_OUTPUT = ROOT / "results/terpene_general_known_recovery"
DEFAULT_BUDGETS = (1, 3, 5, 10, 20)


def parse_budgets(value: str) -> tuple[int, ...]:
    budgets = tuple(sorted({int(part) for part in value.split(",") if part.strip()}))
    if not budgets or budgets[0] <= 0:
        raise ValueError("budgets must contain positive integers")
    return budgets


def _metrics_from_scores(
    scores: np.ndarray,
    candidate_ids: list[str],
    positive_ids: set[str],
    budgets: tuple[int, ...],
) -> dict[str, float | int]:
    """Compute retrieval metrics without materializing a full sorted candidate list."""
    if not positive_ids:
        raise ValueError("positive_ids must be non-empty")
    index = {value: i for i, value in enumerate(candidate_ids)}
    positive_rows = np.asarray([index[value] for value in positive_ids if value in index], dtype=np.int64)
    if len(positive_rows) == 0:
        raise ValueError("none of the positives are in candidate_ids")
    positive_scores = scores[positive_rows]
    # Deterministic tie break matches the project's normal rank semantics: score desc, ID asc.
    best_local = min(
        range(len(positive_rows)),
        key=lambda i: (-float(positive_scores[i]), candidate_ids[int(positive_rows[i])]),
    )
    best_row = int(positive_rows[best_local])
    best_score = float(scores[best_row])
    best_id = candidate_ids[best_row]
    better = int(np.count_nonzero(scores > best_score))
    tied_before = sum(
        1 for i, value in enumerate(scores)
        if i != best_row and float(value) == best_score and candidate_ids[i] < best_id
    )
    best_rank = better + tied_before + 1

    max_budget = min(max(budgets), len(candidate_ids))
    if max_budget == len(candidate_ids):
        top_rows = np.lexsort((np.asarray(candidate_ids, dtype=object), -scores))[:max_budget]
    else:
        rough = np.argpartition(-scores, max_budget - 1)[:max_budget]
        top_rows = rough[np.lexsort((np.asarray([candidate_ids[i] for i in rough], dtype=object), -scores[rough]))]
    top_ids = [candidate_ids[int(i)] for i in top_rows]
    out: dict[str, float | int] = {
        "n_positives": int(len(positive_rows)),
        "best_positive_rank": int(best_rank),
        "reciprocal_rank": float(1.0 / best_rank),
    }
    for budget in budgets:
        panel = set(top_ids[: min(budget, len(top_ids))])
        hits = len(panel & positive_ids)
        out[f"hit_at_{budget}"] = int(hits > 0)
        out[f"positive_recall_at_{budget}"] = float(hits / len(positive_rows))
    return out


def _encode_in_chunks(model: torch.nn.Module, values: np.ndarray, *, kind: str, device: torch.device, chunk_size: int) -> torch.Tensor:
    chunks: list[torch.Tensor] = []
    with torch.no_grad():
        for start in range(0, len(values), chunk_size):
            batch = torch.as_tensor(values[start : start + chunk_size], dtype=torch.float32, device=device)
            if kind == "protein":
                encoded = model.encode_proteins(batch)
            elif kind == "reaction":
                encoded = encode_model_reactions(model, batch, None)
            else:
                raise ValueError(kind)
            chunks.append(encoded.detach())
    return torch.cat(chunks, dim=0)


def _aggregate(frame: pd.DataFrame, budgets: tuple[int, ...]) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    aggregations: dict[str, tuple[str, str]] = {
        "n_queries": ("query_id", "size"),
        "mean_positive_count": ("n_positives", "mean"),
        "mrr": ("reciprocal_rank", "mean"),
        "median_best_positive_rank": ("best_positive_rank", "median"),
    }
    for k in budgets:
        aggregations[f"hit_at_{k}"] = (f"hit_at_{k}", "mean")
        aggregations[f"positive_recall_at_{k}"] = (f"positive_recall_at_{k}", "mean")
    return frame.groupby(["direction", "stratum"], as_index=False).agg(**aggregations)


def _positive_maps(
    associations: pd.DataFrame,
    training_pairs: set[tuple[str, str]],
) -> dict[str, tuple[dict[str, set[str]], dict[str, set[str]]]]:
    strata: dict[str, set[tuple[str, str]]] = {
        "all_known": set(),
        "project_catalog": set(),
        "uniprot_rhea_cached": set(),
        "historical_training_pair": set(),
        "unseen_to_historical_training": set(),
    }
    for row in associations[["protein_id", "reaction_id", "source"]].itertuples(index=False):
        pair = (str(row.protein_id), str(row.reaction_id))
        strata["all_known"].add(pair)
        source = str(row.source)
        if source in strata:
            strata[source].add(pair)
        if pair in training_pairs:
            strata["historical_training_pair"].add(pair)
        else:
            strata["unseen_to_historical_training"].add(pair)
    result = {}
    for name, pairs in strata.items():
        by_reaction: dict[str, set[str]] = defaultdict(set)
        by_protein: dict[str, set[str]] = defaultdict(set)
        for protein_id, reaction_id in pairs:
            by_reaction[reaction_id].add(protein_id)
            by_protein[protein_id].add(reaction_id)
        result[name] = (dict(by_reaction), dict(by_protein))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Retrospective zero-shot recovery of recorded associations in a broad candidate universe.")
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--universe-dir", type=Path, default=DEFAULT_UNIVERSE)
    parser.add_argument("--training-pairs", type=Path, default=DEFAULT_TRAINING)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--budgets", default=",".join(map(str, DEFAULT_BUDGETS)))
    parser.add_argument("--query-batch-size", type=int, default=64)
    parser.add_argument("--feature-chunk-size", type=int, default=8192)
    parser.add_argument("--max-r2e-queries", type=int, default=0)
    parser.add_argument("--max-e2r-queries", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    budgets = parse_budgets(args.budgets)
    device = torch.device(args.device)
    universe = args.universe_dir.resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)

    protein_features, protein_ids = load_protein_library(universe / "proteins")
    schema = load_feature_schema(args.model_dir.resolve())
    reaction_features, reaction_ids = load_registered_reaction_feature_library(
        universe / "reaction_features" / "drfp_categorical_v1", schema
    )
    protein_set, reaction_set = set(protein_ids), set(reaction_ids)
    associations = pd.read_csv(universe / "associations.csv", dtype=str).fillna("")
    associations = associations[
        associations["protein_id"].isin(protein_set) & associations["reaction_id"].isin(reaction_set)
    ].drop_duplicates(["protein_id", "reaction_id"]).copy()
    training_pairs: set[tuple[str, str]] = set()
    if args.training_pairs.is_file():
        training = pd.read_csv(args.training_pairs, dtype=str).fillna("")
        training_pairs = set(zip(training["Entry"].astype(str), training["rhea_id"].astype(str)))
    strata = _positive_maps(associations, training_pairs)

    models = load_models(args.model_dir.resolve() / "models", "production", device)
    if any(model.__class__.__name__ != "TerpeneDualTower" for model in models):
        raise ValueError("This direct known-recovery evaluator currently expects standard dual-tower checkpoints")

    protein_member_embeddings: list[torch.Tensor] = []
    reaction_member_embeddings: list[torch.Tensor] = []
    for model in models:
        protein_member_embeddings.append(
            _encode_in_chunks(model, protein_features, kind="protein", device=device, chunk_size=args.feature_chunk_size)
        )
        reaction_member_embeddings.append(
            _encode_in_chunks(model, reaction_features, kind="reaction", device=device, chunk_size=args.feature_chunk_size)
        )

    records: list[dict[str, object]] = []
    r2e_queries = sorted(strata["all_known"][0])
    if args.max_r2e_queries > 0:
        r2e_queries = r2e_queries[: args.max_r2e_queries]
    reaction_row = {value: i for i, value in enumerate(reaction_ids)}
    for start in range(0, len(r2e_queries), args.query_batch_size):
        batch_ids = r2e_queries[start : start + args.query_batch_size]
        rows = torch.as_tensor([reaction_row[value] for value in batch_ids], dtype=torch.long, device=device)
        scores = None
        with torch.no_grad():
            for r_emb, p_emb in zip(reaction_member_embeddings, protein_member_embeddings, strict=True):
                member = r_emb[rows] @ p_emb.T
                scores = member if scores is None else scores + member
            scores = (scores / len(models)).cpu().numpy()
        for local_i, query_id in enumerate(batch_ids):
            values = scores[local_i]
            for stratum, (by_reaction, _by_protein) in strata.items():
                positives = by_reaction.get(query_id, set())
                if positives:
                    records.append({
                        "direction": "reaction_to_enzyme", "stratum": stratum, "query_id": query_id,
                        **_metrics_from_scores(values, protein_ids, positives, budgets),
                    })
        print(f"R2E {min(start + len(batch_ids), len(r2e_queries))}/{len(r2e_queries)}", flush=True)

    e2r_queries = sorted(strata["all_known"][1])
    if args.max_e2r_queries > 0:
        e2r_queries = e2r_queries[: args.max_e2r_queries]
    protein_row = {value: i for i, value in enumerate(protein_ids)}
    for start in range(0, len(e2r_queries), args.query_batch_size):
        batch_ids = e2r_queries[start : start + args.query_batch_size]
        rows = torch.as_tensor([protein_row[value] for value in batch_ids], dtype=torch.long, device=device)
        scores = None
        with torch.no_grad():
            for r_emb, p_emb in zip(reaction_member_embeddings, protein_member_embeddings, strict=True):
                member = p_emb[rows] @ r_emb.T
                scores = member if scores is None else scores + member
            scores = (scores / len(models)).cpu().numpy()
        for local_i, query_id in enumerate(batch_ids):
            values = scores[local_i]
            for stratum, (_by_reaction, by_protein) in strata.items():
                positives = by_protein.get(query_id, set())
                if positives:
                    records.append({
                        "direction": "enzyme_to_reaction", "stratum": stratum, "query_id": query_id,
                        **_metrics_from_scores(values, reaction_ids, positives, budgets),
                    })
        print(f"E2R {min(start + len(batch_ids), len(e2r_queries))}/{len(e2r_queries)}", flush=True)

    frame = pd.DataFrame(records)
    metrics = _aggregate(frame, budgets)
    frame.to_csv(output / "query_metrics.csv", index=False)
    metrics.to_csv(output / "metrics.csv", index=False)
    summary = {
        "model_dir": str(args.model_dir.resolve()),
        "universe_dir": str(universe),
        "n_proteins": len(protein_ids),
        "n_reactions": len(reaction_ids),
        "n_recorded_pairs": int(len(associations)),
        "n_historical_training_pairs": len(training_pairs),
        "budgets": budgets,
        "evaluation": "direct ensemble zero-shot; recorded associations are labels only and are not used in scoring",
        "outputs": {"metrics": str(output / "metrics.csv"), "query_metrics": str(output / "query_metrics.csv")},
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()
