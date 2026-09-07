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
    evaluate_full_candidate_ranks,
    summarize_query_metrics,
)
from projects.active.terpene_screening.fair_benchmark import (  # noqa: E402
    DEFAULT_BUDGETS,
    DEFAULT_TOP_PERCENTS,
    sha256_file,
)
from projects.active.terpene_screening.rank_open_world import (  # noqa: E402
    load_feature_schema,
    load_models,
    load_protein_library,
    load_registered_reaction_feature_library,
)
from projects.active.terpene_screening.run_internal_top2000_pair_reranker_v1 import (  # noqa: E402
    DEFAULT_BENCH,
    DEFAULT_COARSE_EVAL,
    DEFAULT_DIFFICULTY,
    DEFAULT_MODEL_ROOT,
    DEFAULT_PROTEINS,
    DEFAULT_REACTIONS,
    encode_library,
    query_metrics_from_positive_rank_frame,
    reconstruct_positive_ranks,
)

DEFAULT_PROTOCOL = ROOT / "projects/active/terpene_screening/CLEANROOM_R2E_FUNCTIONAL_PROTOTYPE_RESIDUAL_V1.json"
DEFAULT_EC_METADATA = ROOT / "data/external/reactzyme/cleaned_uniprot_rhea.tsv"
DEFAULT_OUTPUT = ROOT / "results/cleanroom_internal_functional_prototype_residual_v1"


def parse_ec_prefixes(value: str, level: int = 2) -> set[str]:
    if level <= 0:
        raise ValueError("level must be positive")
    out: set[str] = set()
    for raw in str(value).split(";"):
        parts = [part.strip() for part in raw.strip().split(".")]
        if len(parts) >= level and all(part.isdigit() for part in parts[:level]):
            out.add(".".join(parts[:level]))
    return out


def collect_train_only_ec_labels(
    metadata_path: Path,
    *,
    train_proteins: set[str],
    train_reactions: set[str],
    level: int,
) -> tuple[dict[str, set[str]], dict[str, set[str]], dict[str, object]]:
    protein_labels: dict[str, set[str]] = defaultdict(set)
    reaction_labels: dict[str, set[str]] = defaultdict(set)
    frame = pd.read_csv(
        metadata_path,
        sep="\t",
        dtype=str,
        usecols=["Entry", "EC number", "Rhea ID"],
    ).fillna("")
    for entry, ec_value, rhea_value in frame.itertuples(index=False):
        labels = parse_ec_prefixes(ec_value, level=level)
        if not labels:
            continue
        entry = str(entry)
        if entry in train_proteins:
            protein_labels[entry].update(labels)
        for rid in str(rhea_value).split(";"):
            rid = rid.strip()
            if rid in train_reactions:
                reaction_labels[rid].update(labels)
    audit = {
        "metadata_rows_scanned": int(len(frame)),
        "train_proteins_requested": int(len(train_proteins)),
        "train_reactions_requested": int(len(train_reactions)),
        "train_proteins_with_labels": int(len(protein_labels)),
        "train_reactions_with_labels": int(len(reaction_labels)),
        "retained_label_ids_are_train_only": True,
        "test_or_dev_ec_labels_used": False,
    }
    return dict(protein_labels), dict(reaction_labels), audit


def eligible_classes(
    protein_labels: dict[str, set[str]],
    reaction_labels: dict[str, set[str]],
    *,
    min_proteins: int,
    min_reactions: int,
) -> tuple[list[str], dict[str, dict[str, int]]]:
    pcounts: dict[str, int] = defaultdict(int)
    rcounts: dict[str, int] = defaultdict(int)
    for labels in protein_labels.values():
        for label in labels:
            pcounts[label] += 1
    for labels in reaction_labels.values():
        for label in labels:
            rcounts[label] += 1
    classes = sorted(
        label
        for label in set(pcounts) & set(rcounts)
        if pcounts[label] >= int(min_proteins) and rcounts[label] >= int(min_reactions)
    )
    support = {
        label: {
            "train_proteins": int(pcounts[label]),
            "train_reactions": int(rcounts[label]),
        }
        for label in sorted(set(pcounts) | set(rcounts))
    }
    return classes, support


def build_class_centroids(
    embeddings: np.ndarray,
    identifiers: list[str],
    labels_by_id: dict[str, set[str]],
    classes: list[str],
) -> np.ndarray:
    index = {value: row for row, value in enumerate(identifiers)}
    rows: list[np.ndarray] = []
    for label in classes:
        members = sorted(
            identifier
            for identifier, labels in labels_by_id.items()
            if label in labels and identifier in index
        )
        if not members:
            raise ValueError(f"No feature members for class {label}")
        centroid = np.asarray(
            embeddings[[index[identifier] for identifier in members]], dtype=np.float32
        ).mean(axis=0)
        norm = float(np.linalg.norm(centroid))
        if norm <= 0:
            raise ValueError(f"Zero centroid for class {label}")
        rows.append((centroid / norm).astype(np.float32))
    return np.stack(rows, axis=0)


def combo_slug(scale: float, margin: float) -> str:
    return f"scale_{scale:g}__margin_{margin:g}".replace(".", "p")


def summarize_with_slices(frame: pd.DataFrame, slices: pd.DataFrame) -> dict[str, object]:
    joined = frame.merge(
        slices[["query_id", "reaction_similarity_bucket"]],
        on="query_id",
        how="left",
        validate="one_to_one",
    )
    by_slice: dict[str, object] = {}
    for bucket in sorted(joined["reaction_similarity_bucket"].dropna().astype(str).unique()):
        part = joined[joined["reaction_similarity_bucket"].astype(str) == bucket]
        by_slice[bucket] = summarize_query_metrics(
            part, budgets=DEFAULT_BUDGETS, top_percents=DEFAULT_TOP_PERCENTS
        )
    return {
        "all": summarize_query_metrics(
            frame, budgets=DEFAULT_BUDGETS, top_percents=DEFAULT_TOP_PERCENTS
        ),
        "reaction_similarity_slices": by_slice,
    }


def evaluate_fold(
    fold: int,
    *,
    protocol: dict[str, object],
    output_root: Path,
    device: torch.device,
) -> dict[str, object]:
    cell = f"clean2023_internal_double_cold_fold{fold}"
    bench = DEFAULT_BENCH / cell
    model_dir = DEFAULT_MODEL_ROOT / f"fold{fold}"
    schema = load_feature_schema(model_dir)
    protein_features, protein_ids = load_protein_library(DEFAULT_PROTEINS)
    reaction_features, reaction_ids = load_registered_reaction_feature_library(DEFAULT_REACTIONS, schema)
    models = load_models(model_dir / "models", "production", device)
    if len(models) != 1:
        raise ValueError(f"Expected one coarse model for fold{fold}, got {len(models)}")
    p_emb = encode_library(models[0], protein_features, kind="protein", device=device)
    r_emb = encode_library(models[0], reaction_features, kind="reaction", device=device)

    train_pairs = pd.read_csv(bench / "train_pairs.csv", dtype=str).fillna("")
    test_pairs = pd.read_csv(bench / "test_pairs.csv", dtype=str).fillna("")
    train_p, test_p = set(train_pairs.protein_id), set(test_pairs.protein_id)
    train_r, test_r = set(train_pairs.reaction_id), set(test_pairs.reaction_id)
    if train_p & test_p or train_r & test_r:
        raise RuntimeError("Functional-prototype development requires strict entity-disjoint folds")

    ec = dict(protocol["ec_metadata"])
    p_labels, r_labels, ec_audit = collect_train_only_ec_labels(
        DEFAULT_EC_METADATA,
        train_proteins=train_p,
        train_reactions=train_r,
        level=int(ec["prefix_level"]),
    )
    classes, class_support = eligible_classes(
        p_labels,
        r_labels,
        min_proteins=int(ec["min_train_proteins_per_class"]),
        min_reactions=int(ec["min_train_reactions_per_class"]),
    )
    if len(classes) < 2:
        raise RuntimeError(f"Need at least two eligible EC classes; got {classes}")
    p_cent = build_class_centroids(p_emb, protein_ids, p_labels, classes)
    r_cent = build_class_centroids(r_emb, reaction_ids, r_labels, classes)

    coarse_pos = pd.read_csv(
        DEFAULT_COARSE_EVAL / cell / "positive_ranks.csv",
        dtype={"query_id": str, "positive_id": str},
    )
    coarse_pos = coarse_pos[coarse_pos.direction == "reaction_to_enzyme"].copy()
    coarse_query = query_metrics_from_positive_rank_frame(coarse_pos)
    coarse_rank_map = {
        (str(row.query_id), str(row.positive_id)): int(row.positive_rank)
        for row in coarse_pos.itertuples(index=False)
    }
    slices = pd.read_csv(DEFAULT_DIFFICULTY / cell / "reaction_slices.csv", dtype={"reaction_id": str})
    slices = slices.rename(columns={"reaction_id": "query_id"})
    positives_by_reaction = {
        str(rid): set(group.protein_id.astype(str))
        for rid, group in test_pairs.groupby("reaction_id", sort=True)
    }

    grid = dict(protocol["grid"])
    combos = [
        (float(scale), float(margin))
        for scale in grid["residual_scales"]
        for margin in grid["confidence_margins"]
    ]
    records: dict[tuple[float, float], list[dict[str, object]]] = {combo: [] for combo in combos}
    selected = {combo: 0 for combo in combos}
    confidences: list[float] = []
    r_index = {value: row for row, value in enumerate(reaction_ids)}
    candidate_ids = np.asarray(protein_ids, dtype=object)
    p_all = torch.as_tensor(p_emb, dtype=torch.float32, device=device)
    r_all = torch.as_tensor(r_emb, dtype=torch.float32, device=device)
    p_cent_t = torch.as_tensor(p_cent, dtype=torch.float32, device=device)
    r_cent_t = torch.as_tensor(r_cent, dtype=torch.float32, device=device)
    queries = sorted(positives_by_reaction)
    k = min(int(protocol["expert"]["shortlist_size"]), len(protein_ids))

    with torch.no_grad():
        for start in range(0, len(queries), 16):
            batch = queries[start : start + 16]
            rr = torch.as_tensor([r_index[q] for q in batch], dtype=torch.long, device=device)
            coarse = r_all[rr] @ p_all.T
            top_values, top_rows = torch.topk(coarse, k=k, dim=1, largest=True, sorted=False)
            proto_logits = r_all[rr] @ r_cent_t.T
            proto_top2, proto_classes = torch.topk(proto_logits, k=2, dim=1, largest=True, sorted=True)
            for i, qid in enumerate(batch):
                rows = top_rows[i].cpu().numpy().astype(np.int64)
                values = top_values[i].cpu().numpy().astype(np.float64)
                order = np.lexsort((candidate_ids[rows], -values))
                rows, values = rows[order], values[order]
                class_idx = int(proto_classes[i, 0].item())
                confidence = float((proto_top2[i, 0] - proto_top2[i, 1]).item())
                confidences.append(confidence)
                proto_score = (
                    p_all[torch.as_tensor(rows, dtype=torch.long, device=device)] @ p_cent_t[class_idx]
                ).cpu().numpy().astype(np.float64)
                positives = positives_by_reaction[qid]
                old = {pid: coarse_rank_map[(qid, pid)] for pid in positives}
                for combo in combos:
                    scale, margin = combo
                    use_expert = confidence >= margin
                    final = values + scale * proto_score if use_expert else values
                    if use_expert:
                        selected[combo] += 1
                    rerank_order = np.lexsort((candidate_ids[rows], -final))
                    ranks = reconstruct_positive_ranks(
                        positives=positives,
                        reranked_top_ids=candidate_ids[rows[rerank_order]].astype(str).tolist(),
                        coarse_positive_ranks=old,
                    )
                    metrics = evaluate_full_candidate_ranks(
                        ranks,
                        len(protein_ids),
                        budgets=DEFAULT_BUDGETS,
                        top_percents=DEFAULT_TOP_PERCENTS,
                    )
                    records[combo].append(
                        {"direction": "reaction_to_enzyme", "query_id": qid, **metrics}
                    )

    out = output_root / f"fold{fold}"
    out.mkdir(parents=True, exist_ok=True)
    coarse_query.to_csv(out / "coarse_query_metrics.csv", index=False)
    candidate_summaries: dict[str, object] = {}
    for combo in combos:
        frame = pd.DataFrame(records[combo])
        slug = combo_slug(*combo)
        frame.to_csv(out / f"{slug}_query_metrics.csv", index=False)
        candidate_summaries[slug] = {
            "residual_scale": combo[0],
            "confidence_margin": combo[1],
            "selected_fraction": float(selected[combo] / max(len(queries), 1)),
            "metrics": summarize_with_slices(frame, slices),
        }
    payload = {
        "fold": fold,
        "cell": cell,
        "protocol": str(DEFAULT_PROTOCOL.resolve()),
        "protocol_sha256": sha256_file(DEFAULT_PROTOCOL),
        "outer_labels_used": False,
        "target_benchmark_labels_used": False,
        "test_ec_metadata_used": False,
        "strict_entity_disjoint": True,
        "eligible_ec_classes": classes,
        "eligible_ec_class_count": len(classes),
        "class_support": class_support,
        "ec_audit": ec_audit,
        "query_confidence": {
            "min": float(np.min(confidences)),
            "median": float(np.median(confidences)),
            "max": float(np.max(confidences)),
        },
        "baseline": summarize_with_slices(coarse_query, slices),
        "candidates": candidate_summaries,
    }
    (out / "summary.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"fold": fold, "eligible_ec_classes": len(classes), "confidence": payload["query_confidence"]}), flush=True)
    return payload


def main() -> None:
    ap = argparse.ArgumentParser(description="Internal-only EC prototype Top-2000 R2E residual screening.")
    ap.add_argument("--folds", default="0,1,2")
    ap.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    protocol = json.loads(DEFAULT_PROTOCOL.read_text(encoding="utf-8"))
    folds = [int(x) for x in args.folds.split(",") if x.strip()]
    for fold in folds:
        summary = args.output_root.resolve() / f"fold{fold}" / "summary.json"
        if summary.is_file():
            print(f"skip fold{fold}: {summary} exists", flush=True)
            continue
        evaluate_fold(
            fold,
            protocol=protocol,
            output_root=args.output_root.resolve(),
            device=torch.device(args.device),
        )


if __name__ == "__main__":
    main()
