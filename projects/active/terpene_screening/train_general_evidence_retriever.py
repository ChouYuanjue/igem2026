from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from projects.active.terpene_screening.rank_open_world import (  # noqa: E402
    load_feature_schema,
    load_protein_library,
    load_registered_reaction_feature_library,
)
from projects.active.terpene_screening.third_party.mammoth_lwf import (  # noqa: E402
    bidirectional_distillation as lwf_bidirectional_distillation,
    distillation as lwf_distillation,
)
from projects.active.terpene_screening.third_party.margin_mse_loss import MarginMSELoss  # noqa: E402
from projects.active.terpene_screening.third_party.recadam import RecAdam  # noqa: E402
from projects.active.terpene_screening.train_dual_tower_cold import (  # noqa: E402
    ModelConfig,
    TerpeneDualTower,
    seed_everything,
)

DEFAULT_UNIVERSE = ROOT / "data/catalyst_candidate_universes/general_merged"
DEFAULT_BASE = ROOT / "results/terpene_production_models/marts_adapted_drfp_pu"
DEFAULT_OUTPUT = ROOT / "results/terpene_production_models/general_evidence_continuation"
DEFAULT_CURRENT_PROTEINS = ROOT / "data/terpene_embeddings/esmc600m_mean"
DEFAULT_CURRENT_POSITIVES = ROOT / "data/terpene/enzyme_terpene_synthase.tsv"


def _query_positive_rows(
    associations: pd.DataFrame,
    *,
    direction: str,
    query_index: dict[str, int],
    candidate_index: dict[str, int],
) -> tuple[list[str], list[np.ndarray]]:
    if direction == "r2e":
        qcol, ccol = "reaction_id", "protein_id"
    elif direction == "e2r":
        qcol, ccol = "protein_id", "reaction_id"
    else:
        raise ValueError(direction)
    rows: list[tuple[str, np.ndarray]] = []
    for query_id, group in associations.groupby(qcol, sort=True):
        q = str(query_id)
        if q not in query_index:
            continue
        candidates = sorted({candidate_index[str(value)] for value in group[ccol] if str(value) in candidate_index})
        if candidates:
            rows.append((q, np.asarray(candidates, dtype=np.int64)))
    return [value for value, _ in rows], [indices for _, indices in rows]


def _directional_full_candidate_loss(
    query_embeddings: torch.Tensor,
    candidate_embeddings: torch.Tensor,
    positive_rows: list[np.ndarray],
    *,
    temperature: float,
    topk_k: int,
    topk_weight: float,
    topk_margin: float,
    all_positive_weight: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if topk_k <= 0:
        raise ValueError("topk_k must be positive")
    logits = query_embeddings @ candidate_embeddings.T / temperature
    row_losses: list[torch.Tensor] = []
    topk_losses: list[torch.Tensor] = []
    positive_alignment: list[torch.Tensor] = []
    for i, indices_np in enumerate(positive_rows):
        indices = torch.as_tensor(indices_np, dtype=torch.long, device=logits.device)
        positive_logits = logits[i, indices]
        row_losses.append(torch.logsumexp(logits[i], dim=0) - torch.logsumexp(positive_logits, dim=0))
        if all_positive_weight > 0:
            # Cosine alignment before temperature scaling; encourages recall of more than one known positive.
            positive_alignment.append(1.0 - (query_embeddings[i] * candidate_embeddings[indices]).sum(dim=1).mean())
        if topk_weight > 0:
            negatives = logits[i].clone()
            negatives[indices] = torch.finfo(negatives.dtype).min
            k = min(int(topk_k), max(1, len(negatives) - len(indices)))
            kth_negative = torch.topk(negatives, k=k).values[-1]
            best_positive = positive_logits.max()
            topk_losses.append(F.softplus(kth_negative - best_positive + topk_margin))
    contrastive = torch.stack(row_losses).mean()
    topk = torch.stack(topk_losses).mean() if topk_losses else torch.zeros((), device=logits.device)
    alignment = (
        torch.stack(positive_alignment).mean() if positive_alignment else torch.zeros((), device=logits.device)
    )
    total = contrastive + float(topk_weight) * topk + float(all_positive_weight) * alignment
    return total, {
        "contrastive_loss": float(contrastive.detach().cpu()),
        "topk_loss": float(topk.detach().cpu()),
        "positive_alignment_loss": float(alignment.detach().cpu()),
    }


def _encode_chunks(
    model: TerpeneDualTower,
    features: np.ndarray,
    *,
    kind: str,
    device: torch.device,
    chunk_size: int,
) -> torch.Tensor:
    values: list[torch.Tensor] = []
    with torch.no_grad():
        for start in range(0, len(features), chunk_size):
            batch = torch.as_tensor(features[start : start + chunk_size], dtype=torch.float32, device=device)
            encoded = model.encode_proteins(batch) if kind == "protein" else model.encode_reactions(batch)
            values.append(encoded.detach())
    return torch.cat(values, dim=0)


def _historical_query_ids(base_dir: Path, direction: str, available: set[str]) -> list[str]:
    filename = "reaction_registry.csv" if direction == "r2e" else "protein_registry.csv"
    column = "reaction_id" if direction == "r2e" else "protein_id"
    table = pd.read_csv(base_dir / filename, dtype=str).fillna("")
    return sorted(set(table[column].astype(str)) & available)


def _current_retention_ids(
    base_dir: Path,
    direction: str,
    *,
    available_queries: set[str],
    available_candidates: set[str],
) -> tuple[list[str], list[str]]:
    current_proteins = set(
        pd.read_csv(DEFAULT_CURRENT_PROTEINS / "entries.csv", dtype={"Entry": str})["Entry"].astype(str)
    )
    current_reactions = set(str(value) for value in load_feature_schema(base_dir)["reaction_ids"])
    if direction == "r2e":
        query_ids, candidate_ids = current_reactions, current_proteins
    elif direction == "e2r":
        query_ids, candidate_ids = current_proteins, current_reactions
    else:
        raise ValueError(direction)
    return (
        sorted(query_ids & available_queries),
        sorted(candidate_ids & available_candidates),
    )


def _current_positive_index_pairs(
    direction: str,
    query_ids: list[str],
    candidate_ids: list[str],
) -> list[tuple[int, int]]:
    positives = pd.read_csv(DEFAULT_CURRENT_POSITIVES, sep="\t", dtype=str).fillna("")
    qindex = {value: index for index, value in enumerate(query_ids)}
    cindex = {value: index for index, value in enumerate(candidate_ids)}
    pairs: set[tuple[int, int]] = set()
    for row in positives[["rhea_id", "Entry"]].drop_duplicates().itertuples(index=False):
        reaction_id, protein_id = str(row.rhea_id), str(row.Entry)
        query_id, candidate_id = (reaction_id, protein_id) if direction == "r2e" else (protein_id, reaction_id)
        if query_id in qindex and candidate_id in cindex:
            pairs.add((qindex[query_id], cindex[candidate_id]))
    return sorted(pairs)


def _build_bidirectional_margin_pairs(
    teacher_scores: torch.Tensor,
    positive_pairs: list[tuple[int, int]],
    *,
    topk: int,
) -> dict[str, torch.Tensor]:
    if topk <= 0:
        raise ValueError("margin distillation topk must be positive")
    if teacher_scores.ndim != 2:
        raise ValueError("teacher_scores must be a matrix")
    row_positives: dict[int, set[int]] = {}
    col_positives: dict[int, set[int]] = {}
    for row, col in positive_pairs:
        row_positives.setdefault(row, set()).add(col)
        col_positives.setdefault(col, set()).add(row)

    row_q: list[int] = []
    row_pos: list[int] = []
    row_neg: list[int] = []
    col_q: list[int] = []
    col_pos: list[int] = []
    col_neg: list[int] = []
    with torch.no_grad():
        for row, positives in sorted(row_positives.items()):
            values = teacher_scores[row].clone()
            values[list(positives)] = torch.finfo(values.dtype).min
            k = min(topk, values.numel() - len(positives))
            negatives = torch.topk(values, k=max(1, k)).indices.tolist()
            for positive in sorted(positives):
                for negative in negatives:
                    row_q.append(row); row_pos.append(positive); row_neg.append(int(negative))
        for col, positives in sorted(col_positives.items()):
            values = teacher_scores[:, col].clone()
            values[list(positives)] = torch.finfo(values.dtype).min
            k = min(topk, values.numel() - len(positives))
            negatives = torch.topk(values, k=max(1, k)).indices.tolist()
            for positive in sorted(positives):
                for negative in negatives:
                    col_q.append(col); col_pos.append(positive); col_neg.append(int(negative))
    device = teacher_scores.device
    return {
        "row_query": torch.as_tensor(row_q, dtype=torch.long, device=device),
        "row_positive": torch.as_tensor(row_pos, dtype=torch.long, device=device),
        "row_negative": torch.as_tensor(row_neg, dtype=torch.long, device=device),
        "col_query": torch.as_tensor(col_q, dtype=torch.long, device=device),
        "col_positive": torch.as_tensor(col_pos, dtype=torch.long, device=device),
        "col_negative": torch.as_tensor(col_neg, dtype=torch.long, device=device),
    }


def _bidirectional_margin_mse(
    student_scores: torch.Tensor,
    teacher_scores: torch.Tensor,
    pairs: dict[str, torch.Tensor],
    loss_fn: MarginMSELoss,
) -> torch.Tensor:
    losses: list[torch.Tensor] = []
    if len(pairs["row_query"]):
        q, p, n = pairs["row_query"], pairs["row_positive"], pairs["row_negative"]
        losses.append(loss_fn(student_scores[q, p], student_scores[q, n], teacher_scores[q, p], teacher_scores[q, n]))
    if len(pairs["col_query"]):
        q, p, n = pairs["col_query"], pairs["col_positive"], pairs["col_negative"]
        losses.append(loss_fn(student_scores[p, q], student_scores[n, q], teacher_scores[p, q], teacher_scores[n, q]))
    if not losses:
        return torch.zeros((), dtype=student_scores.dtype, device=student_scores.device)
    return torch.stack(losses).mean()


def _build_optimizer(
    model: TerpeneDualTower,
    base_model: TerpeneDualTower,
    *,
    optimizer_name: str,
    learning_rate: float,
    weight_decay: float,
    total_training_steps: int,
    recadam_anneal_fun: str,
    recadam_anneal_k: float,
    recadam_anneal_t0: int,
    recadam_pretrain_cof: float,
) -> torch.optim.Optimizer:
    trainable = [(name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad]
    if not trainable:
        raise ValueError("No trainable parameters")
    parameters = [parameter for _, parameter in trainable]
    if optimizer_name == "adamw":
        return torch.optim.AdamW(parameters, lr=learning_rate, weight_decay=weight_decay)
    if optimizer_name != "recadam":
        raise ValueError(f"Unknown optimizer: {optimizer_name}")

    base_parameters = dict(base_model.named_parameters())
    references = [base_parameters[name] for name, _ in trainable]
    t0 = int(recadam_anneal_t0)
    if t0 <= 0:
        t0 = max(1, int(round(total_training_steps * 0.5)))
    return RecAdam(
        parameters,
        lr=learning_rate,
        weight_decay=weight_decay,
        correct_bias=True,
        anneal_fun=recadam_anneal_fun,
        anneal_k=float(recadam_anneal_k),
        anneal_t0=t0,
        anneal_w=1.0,
        pretrain_cof=float(recadam_pretrain_cof),
        pretrain_params=references,
    )


def train_one(
    checkpoint: Path,
    *,
    direction: str,
    protein_features: np.ndarray,
    protein_ids: list[str],
    reaction_features: np.ndarray,
    reaction_ids: list[str],
    associations: pd.DataFrame,
    base_dir: Path,
    output_dir: Path,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    temperature: float,
    batch_size: int,
    topk_k: int,
    topk_weight: float,
    topk_margin: float,
    all_positive_weight: float,
    anchor_weight: float,
    anchor_batch_size: int,
    score_distill_weight: float,
    score_distill_temperature: float,
    score_distill_batch_size: int,
    score_distill_bidirectional: bool,
    margin_distill_weight: float,
    margin_distill_topk: int,
    historical_query_repeat: int,
    feature_chunk_size: int,
    optimizer_name: str,
    recadam_anneal_fun: str,
    recadam_anneal_k: float,
    recadam_anneal_t0: int,
    recadam_pretrain_cof: float,
    device: torch.device,
) -> tuple[Path, list[dict[str, float]]]:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    seed = int(payload.get("seed", 20260723))
    seed_everything(seed)
    config = ModelConfig(**payload["model_config"])
    model = TerpeneDualTower(config).to(device)
    model.load_state_dict(payload["model_state_dict"])
    base_model = TerpeneDualTower(config).to(device)
    base_model.load_state_dict(payload["model_state_dict"])
    base_model.eval()

    protein_index = {value: i for i, value in enumerate(protein_ids)}
    reaction_index = {value: i for i, value in enumerate(reaction_ids)}
    query_index = reaction_index if direction == "r2e" else protein_index
    candidate_index = protein_index if direction == "r2e" else reaction_index
    query_ids, positives = _query_positive_rows(
        associations, direction=direction, query_index=query_index, candidate_index=candidate_index
    )
    positive_by_query = dict(zip(query_ids, positives, strict=True))

    if direction == "r2e":
        for parameter in model.protein_tower.parameters():
            parameter.requires_grad = False
        candidate_embeddings = _encode_chunks(
            base_model, protein_features, kind="protein", device=device, chunk_size=feature_chunk_size
        )
        query_features = reaction_features
        query_kind = "reaction"
    else:
        for parameter in model.reaction_tower.parameters():
            parameter.requires_grad = False
        candidate_embeddings = _encode_chunks(
            base_model, reaction_features, kind="reaction", device=device, chunk_size=feature_chunk_size
        )
        query_features = protein_features
        query_kind = "protein"
    candidate_embeddings = candidate_embeddings.detach()

    historical_ids = _historical_query_ids(base_dir, direction, set(query_ids))
    historical_rows = np.asarray([query_index[value] for value in historical_ids], dtype=np.int64)
    historical_features = query_features[historical_rows] if len(historical_rows) else np.empty((0, query_features.shape[1]), np.float32)
    historical_targets = (
        _encode_chunks(base_model, historical_features, kind=query_kind, device=device, chunk_size=feature_chunk_size)
        if len(historical_features) else torch.empty((0, config.embedding_dim), device=device)
    )

    retention_query_ids, retention_candidate_ids = _current_retention_ids(
        base_dir, direction,
        available_queries=set(query_index),
        available_candidates=set(candidate_index),
    )
    retention_query_rows = np.asarray([query_index[value] for value in retention_query_ids], dtype=np.int64)
    retention_candidate_rows = torch.as_tensor(
        [candidate_index[value] for value in retention_candidate_ids], dtype=torch.long, device=device
    )
    retention_features = (
        query_features[retention_query_rows]
        if len(retention_query_rows)
        else np.empty((0, query_features.shape[1]), np.float32)
    )
    retention_teacher_queries = (
        _encode_chunks(base_model, retention_features, kind=query_kind, device=device, chunk_size=feature_chunk_size)
        if len(retention_features)
        else torch.empty((0, config.embedding_dim), device=device)
    )
    retention_candidate_embeddings = (
        candidate_embeddings[retention_candidate_rows]
        if len(retention_candidate_rows)
        else torch.empty((0, config.embedding_dim), device=device)
    )
    retention_teacher_scores = (
        retention_teacher_queries @ retention_candidate_embeddings.T
        if len(retention_query_rows) and len(retention_candidate_rows)
        else torch.empty((0, 0), device=device)
    )
    margin_positive_pairs = _current_positive_index_pairs(
        direction, retention_query_ids, retention_candidate_ids
    )
    margin_pairs = (
        _build_bidirectional_margin_pairs(
            retention_teacher_scores, margin_positive_pairs, topk=margin_distill_topk
        )
        if margin_distill_weight > 0 and len(margin_positive_pairs)
        else {}
    )
    margin_loss_fn = MarginMSELoss().to(device)

    rng = random.Random(seed)
    history: list[dict[str, float]] = []
    base_queries = list(query_ids)
    repeated_historical = historical_ids * max(0, int(historical_query_repeat))
    steps_per_epoch = max(1, int(np.ceil((len(base_queries) + len(repeated_historical)) / batch_size)))
    total_training_steps = max(1, int(epochs) * steps_per_epoch)
    optimizer = _build_optimizer(
        model,
        base_model,
        optimizer_name=optimizer_name,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        total_training_steps=total_training_steps,
        recadam_anneal_fun=recadam_anneal_fun,
        recadam_anneal_k=recadam_anneal_k,
        recadam_anneal_t0=recadam_anneal_t0,
        recadam_pretrain_cof=recadam_pretrain_cof,
    )
    for epoch in range(1, epochs + 1):
        schedule = base_queries + repeated_historical
        rng.shuffle(schedule)
        model.train()
        totals: list[float] = []
        contrastives: list[float] = []
        topks: list[float] = []
        aligns: list[float] = []
        anchors: list[float] = []
        score_distills: list[float] = []
        margin_distills: list[float] = []
        for start in range(0, len(schedule), batch_size):
            batch_ids = schedule[start : start + batch_size]
            rows = np.asarray([query_index[value] for value in batch_ids], dtype=np.int64)
            batch_features = torch.as_tensor(query_features[rows], dtype=torch.float32, device=device)
            query_embeddings = model.encode_reactions(batch_features) if direction == "r2e" else model.encode_proteins(batch_features)
            batch_positives = [positive_by_query[value] for value in batch_ids]
            loss, components = _directional_full_candidate_loss(
                query_embeddings, candidate_embeddings, batch_positives,
                temperature=temperature, topk_k=topk_k, topk_weight=topk_weight,
                topk_margin=topk_margin, all_positive_weight=all_positive_weight,
            )
            anchor_loss = torch.zeros((), device=device)
            if anchor_weight > 0 and len(historical_rows):
                n = min(anchor_batch_size, len(historical_rows))
                local = np.asarray(rng.sample(range(len(historical_rows)), n), dtype=np.int64)
                anchor_features = torch.as_tensor(historical_features[local], dtype=torch.float32, device=device)
                current = model.encode_reactions(anchor_features) if direction == "r2e" else model.encode_proteins(anchor_features)
                anchor_loss = (1.0 - (current * historical_targets[torch.as_tensor(local, device=device)]).sum(dim=1)).mean()
                loss = loss + float(anchor_weight) * anchor_loss
            score_distill_loss = torch.zeros((), device=device)
            if score_distill_weight > 0 and len(retention_query_rows) and len(retention_candidate_rows):
                n = (
                    len(retention_query_rows)
                    if score_distill_bidirectional
                    else min(score_distill_batch_size, len(retention_query_rows))
                )
                local = (
                    np.arange(len(retention_query_rows), dtype=np.int64)
                    if n == len(retention_query_rows)
                    else np.asarray(rng.sample(range(len(retention_query_rows)), n), dtype=np.int64)
                )
                retention_batch = torch.as_tensor(retention_features[local], dtype=torch.float32, device=device)
                student_queries = (
                    model.encode_reactions(retention_batch)
                    if direction == "r2e"
                    else model.encode_proteins(retention_batch)
                )
                teacher_queries = retention_teacher_queries[torch.as_tensor(local, device=device)]
                teacher_logits = teacher_queries @ retention_candidate_embeddings.T / temperature
                student_logits = student_queries @ retention_candidate_embeddings.T / temperature
                score_distill_loss = (
                    lwf_bidirectional_distillation(
                        teacher_logits, student_logits, temperature=score_distill_temperature
                    )
                    if score_distill_bidirectional
                    else lwf_distillation(
                        teacher_logits, student_logits, temperature=score_distill_temperature
                    )
                )
                loss = loss + float(score_distill_weight) * score_distill_loss
            margin_distill_loss = torch.zeros((), device=device)
            if margin_distill_weight > 0 and margin_pairs:
                retention_batch = torch.as_tensor(retention_features, dtype=torch.float32, device=device)
                student_retention_queries = (
                    model.encode_reactions(retention_batch)
                    if direction == "r2e"
                    else model.encode_proteins(retention_batch)
                )
                student_retention_scores = student_retention_queries @ retention_candidate_embeddings.T
                margin_distill_loss = _bidirectional_margin_mse(
                    student_retention_scores, retention_teacher_scores, margin_pairs, margin_loss_fn
                )
                loss = loss + float(margin_distill_weight) * margin_distill_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            totals.append(float(loss.detach().cpu()))
            contrastives.append(components["contrastive_loss"])
            topks.append(components["topk_loss"])
            aligns.append(components["positive_alignment_loss"])
            anchors.append(float(anchor_loss.detach().cpu()))
            score_distills.append(float(score_distill_loss.detach().cpu()))
            margin_distills.append(float(margin_distill_loss.detach().cpu()))
        row = {
            "seed": float(seed), "epoch": float(epoch), "loss": float(np.mean(totals)),
            "contrastive_loss": float(np.mean(contrastives)), "topk_loss": float(np.mean(topks)),
            "positive_alignment_loss": float(np.mean(aligns)), "anchor_loss": float(np.mean(anchors)),
            "score_distill_loss": float(np.mean(score_distills)),
            "margin_distill_loss": float(np.mean(margin_distills)),
        }
        history.append(row)
        print(json.dumps(row), flush=True)

    model.eval()
    output_model_dir = output_dir / "models"
    output_model_dir.mkdir(parents=True, exist_ok=True)
    target = output_model_dir / f"production_seed{seed}.pt"
    torch.save(
        {
            **{k: v for k, v in payload.items() if k not in {"model_state_dict", "training_sources", "n_training_pairs"}},
            "model_state_dict": model.state_dict(),
            "model_config": asdict(config),
            "seed": seed,
            "model_type": "dual_tower",
            "base_checkpoint": str(checkpoint.resolve()),
            "training_sources": [str((DEFAULT_UNIVERSE / "associations.csv").resolve())],
            "n_training_pairs": int(len(associations)),
            "general_evidence_direction": direction,
        },
        target,
    )
    return target, history


def main() -> None:
    parser = argparse.ArgumentParser(description="Continue a production dual tower on the broad recorded-association graph.")
    parser.add_argument("--direction", choices=["r2e", "e2r"], required=True)
    parser.add_argument("--base-dir", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--universe-dir", type=Path, default=DEFAULT_UNIVERSE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=3e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--topk-k", type=int, default=10)
    parser.add_argument("--topk-weight", type=float, default=0.10)
    parser.add_argument("--topk-margin", type=float, default=0.0)
    parser.add_argument("--all-positive-weight", type=float, default=0.05)
    parser.add_argument("--anchor-weight", type=float, default=0.10)
    parser.add_argument("--anchor-batch-size", type=int, default=256)
    parser.add_argument("--score-distill-weight", type=float, default=0.0, help="LwF-style current-ranking distillation weight; 0 disables it.")
    parser.add_argument("--score-distill-temperature", type=float, default=2.0)
    parser.add_argument("--score-distill-batch-size", type=int, default=128)
    parser.add_argument("--score-distill-bidirectional", action=argparse.BooleanOptionalAction, default=False, help="Distill the full current score matrix in both retrieval directions, as in Mammoth ZSCL.")
    parser.add_argument("--margin-distill-weight", type=float, default=0.0, help="Bidirectional Margin-MSE weight on teacher hard-negative ranking gaps; 0 disables it.")
    parser.add_argument("--margin-distill-topk", type=int, default=20, help="Teacher hard negatives per current positive for Margin-MSE.")
    parser.add_argument("--historical-query-repeat", type=int, default=2)
    parser.add_argument("--feature-chunk-size", type=int, default=8192)
    parser.add_argument("--optimizer", choices=["adamw", "recadam"], default="adamw")
    parser.add_argument("--recadam-anneal-fun", choices=["sigmoid", "linear", "constant"], default="sigmoid")
    parser.add_argument("--recadam-anneal-k", type=float, default=0.1)
    parser.add_argument("--recadam-anneal-t0", type=int, default=0, help="RecAdam objective-shift midpoint in optimizer steps; <=0 uses half of total continuation steps.")
    parser.add_argument("--recadam-pretrain-cof", type=float, default=5000.0)
    parser.add_argument("--seeds", default="", help="Optional comma-separated checkpoint seeds; empty means every production seed.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if args.epochs <= 0 or args.batch_size <= 0 or args.learning_rate <= 0:
        raise ValueError("epochs, batch-size and learning-rate must be positive")
    if args.score_distill_weight < 0 or args.score_distill_temperature <= 0 or args.score_distill_batch_size <= 0:
        raise ValueError("score distillation weight must be non-negative and temperature/batch-size positive")
    if args.margin_distill_weight < 0 or args.margin_distill_topk <= 0:
        raise ValueError("margin distillation weight must be non-negative and topk positive")

    device = torch.device(args.device)
    universe = args.universe_dir.resolve()
    base_dir = args.base_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    schema = load_feature_schema(base_dir)
    protein_features, protein_ids = load_protein_library(universe / "proteins")
    reaction_features, reaction_ids = load_registered_reaction_feature_library(
        universe / "reaction_features" / "drfp_categorical_v1", schema
    )
    associations = pd.read_csv(universe / "associations.csv", dtype=str).fillna("")
    associations = associations[
        associations["protein_id"].isin(set(protein_ids)) & associations["reaction_id"].isin(set(reaction_ids))
    ].drop_duplicates(["protein_id", "reaction_id"]).copy()

    requested = {int(value) for value in args.seeds.split(",") if value.strip()}
    checkpoints = sorted((base_dir / "models").glob("production_seed*.pt"))
    if requested:
        checkpoints = [path for path in checkpoints if int(path.stem.replace("production_seed", "")) in requested]
    if not checkpoints:
        raise FileNotFoundError("No matching production checkpoints")

    all_history: list[dict[str, float]] = []
    outputs: list[str] = []
    for checkpoint in checkpoints:
        target, history = train_one(
            checkpoint,
            direction=args.direction,
            protein_features=protein_features,
            protein_ids=protein_ids,
            reaction_features=reaction_features,
            reaction_ids=reaction_ids,
            associations=associations,
            base_dir=base_dir,
            output_dir=output_dir,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            temperature=args.temperature,
            batch_size=args.batch_size,
            topk_k=args.topk_k,
            topk_weight=args.topk_weight,
            topk_margin=args.topk_margin,
            all_positive_weight=args.all_positive_weight,
            anchor_weight=args.anchor_weight,
            anchor_batch_size=args.anchor_batch_size,
            score_distill_weight=args.score_distill_weight,
            score_distill_temperature=args.score_distill_temperature,
            score_distill_batch_size=args.score_distill_batch_size,
            score_distill_bidirectional=args.score_distill_bidirectional,
            margin_distill_weight=args.margin_distill_weight,
            margin_distill_topk=args.margin_distill_topk,
            historical_query_repeat=args.historical_query_repeat,
            feature_chunk_size=args.feature_chunk_size,
            optimizer_name=args.optimizer,
            recadam_anneal_fun=args.recadam_anneal_fun,
            recadam_anneal_k=args.recadam_anneal_k,
            recadam_anneal_t0=args.recadam_anneal_t0,
            recadam_pretrain_cof=args.recadam_pretrain_cof,
            device=device,
        )
        outputs.append(str(target))
        all_history.extend(history)

    pd.DataFrame(all_history).to_csv(output_dir / "training_history.csv", index=False)
    shutil.copy2(base_dir / "feature_schema.json", output_dir / "feature_schema.json")
    # Keep current-domain assets beside the checkpoint so existing retention tools can score it directly.
    for filename in ["reaction_feature_matrix.npy", "reaction_features.csv", "protein_registry.csv", "reaction_registry.csv"]:
        source = base_dir / filename
        if source.exists():
            shutil.copy2(source, output_dir / filename)
    summary = {
        "model_type": "general_evidence_directional_continuation",
        "direction": args.direction,
        "base_dir": str(base_dir),
        "universe_dir": str(universe),
        "n_training_pairs": int(len(associations)),
        "n_proteins": len(protein_ids),
        "n_reactions": len(reaction_ids),
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "optimizer": args.optimizer,
        "recadam": {
            "upstream_repository": "https://github.com/Sanyuan-Chen/RecAdam",
            "upstream_commit": "505ba3c265d5b6b90996dddd254f3eb38adaabae",
            "anneal_fun": args.recadam_anneal_fun,
            "anneal_k": args.recadam_anneal_k,
            "anneal_t0": args.recadam_anneal_t0,
            "pretrain_cof": args.recadam_pretrain_cof,
        } if args.optimizer == "recadam" else None,
        "temperature": args.temperature,
        "topk_k": args.topk_k,
        "topk_weight": args.topk_weight,
        "all_positive_weight": args.all_positive_weight,
        "anchor_weight": args.anchor_weight,
        "score_distillation": {
            "method": "learning_without_forgetting_retrieval_logits",
            "weight": args.score_distill_weight,
            "temperature": args.score_distill_temperature,
            "batch_size": args.score_distill_batch_size,
            "bidirectional": args.score_distill_bidirectional,
            "upstream_repository": "https://github.com/aimagelab/mammoth",
            "upstream_commit": "e75a491c69fd729edeb01431afb753d9157d9a81",
        },
        "margin_distillation": {
            "method": "bidirectional_teacher_hard_negative_margin_mse",
            "weight": args.margin_distill_weight,
            "topk": args.margin_distill_topk,
            "upstream_repository": "https://github.com/sebastian-hofstaetter/neural-ranking-kd",
            "upstream_commit": "aafcc73d6b78ee9849c3d8f5ccf084051fcae2e9",
        },
        "historical_query_repeat": args.historical_query_repeat,
        "checkpoints": outputs,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
