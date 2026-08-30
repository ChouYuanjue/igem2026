from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
import sys
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from projects.active.terpene_screening.evaluate_enzymecage_official_aligned import (  # noqa: E402
    evaluate_enzymecage_native_r2e,
)
from projects.active.terpene_screening.fair_benchmark import (  # noqa: E402
    DEFAULT_BUDGETS,
    DEFAULT_TOP_PERCENTS,
    evaluate_ranking_frame,
    sha256_file,
)
from projects.active.terpene_screening.rank_open_world import (  # noqa: E402
    load_feature_schema,
    load_protein_library,
    load_registered_reaction_feature_library,
)
from projects.active.terpene_screening.train_general_evidence_retriever import (  # noqa: E402
    _train_reaction_novelty,
)
from projects.active.terpene_screening.train_dual_tower_cold import (  # noqa: E402
    ModelConfig,
    TerpeneDualTower,
    seed_everything,
)

DEFAULT_UNIVERSE = ROOT / "data/catalyst_candidate_universes/general_merged"
DEFAULT_ASSOCIATIONS = (
    ROOT / "data/external/enzymecage_current/catalyst_features/clean2023/training_pairs.csv"
)
DEFAULT_SCHEMA_DIR = ROOT / "results/terpene_production_models/marts_adapted_drfp_pu"
DEFAULT_OUTPUT = ROOT / "results/enzymecage_cleanroom_2023"


def stable_entity_fold(identifier: str, folds: int) -> int:
    if folds < 2:
        raise ValueError("folds must be >= 2")
    digest = hashlib.blake2b(str(identifier).encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") % folds


def split_double_cold(
    pairs: pd.DataFrame,
    *,
    dev_fold: int,
    folds: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not 0 <= dev_fold < folds:
        raise ValueError("dev_fold must be within [0, folds)")
    data = pairs[["protein_id", "reaction_id"]].drop_duplicates().copy()
    p_fold = data["protein_id"].map(lambda value: stable_entity_fold(str(value), folds))
    r_fold = data["reaction_id"].map(lambda value: stable_entity_fold(str(value), folds))
    train = data[(p_fold != dev_fold) & (r_fold != dev_fold)].copy()
    dev = data[(p_fold == dev_fold) & (r_fold == dev_fold)].copy()
    if set(train["protein_id"]) & set(dev["protein_id"]):
        raise AssertionError("double-cold split leaked proteins")
    if set(train["reaction_id"]) & set(dev["reaction_id"]):
        raise AssertionError("double-cold split leaked reactions")
    return train.reset_index(drop=True), dev.reset_index(drop=True)


def positive_maps(
    pairs: pd.DataFrame,
) -> tuple[dict[str, set[str]], dict[str, set[str]]]:
    by_reaction: dict[str, set[str]] = defaultdict(set)
    by_protein: dict[str, set[str]] = defaultdict(set)
    for protein_id, reaction_id in pairs[["protein_id", "reaction_id"]].itertuples(index=False):
        by_reaction[str(reaction_id)].add(str(protein_id))
        by_protein[str(protein_id)].add(str(reaction_id))
    return dict(by_reaction), dict(by_protein)


def _sample_ids(rng: random.Random, values: list[str], count: int) -> list[str]:
    if not values or count <= 0:
        return []
    if len(values) >= count:
        return rng.sample(values, count)
    return [rng.choice(values) for _ in range(count)]


def _negative_curriculum_counts(
    *,
    epoch: int,
    target_hard: int,
    target_random: int,
    start_hard: int,
    ramp_epochs: int,
) -> tuple[int, int]:
    """Linearly ramp hard negatives while preserving total negative budget.

    ``ramp_epochs <= 0`` disables the curriculum and exactly preserves the old
    fixed hard/random counts.  When enabled, epoch 1 starts at ``start_hard``
    and the requested target mix is reached at ``ramp_epochs``.
    """
    if epoch <= 0 or target_hard < 0 or target_random < 0:
        raise ValueError("epoch must be positive and negative counts non-negative")
    if ramp_epochs <= 0:
        return int(target_hard), int(target_random)
    if not 0 <= start_hard <= target_hard:
        raise ValueError("start_hard must be in [0,target_hard]")
    if ramp_epochs == 1:
        hard = int(target_hard)
    else:
        progress = min(1.0, max(0.0, (int(epoch) - 1) / float(ramp_epochs - 1)))
        hard = int(round(start_hard + progress * (target_hard - start_hard)))
    total = int(target_hard) + int(target_random)
    random_count = max(0, total - hard)
    return hard, random_count


def _reaction_replay_pool(
    train_reactions: list[str], novel_reactions: list[str], *, repeat: int
) -> list[str]:
    if repeat < 0:
        raise ValueError("reaction novelty repeat must be non-negative")
    train_set = set(train_reactions)
    unknown = set(novel_reactions) - train_set
    if unknown:
        raise ValueError(f"novel replay contains non-training reactions: {sorted(unknown)[:5]}")
    return list(train_reactions) + list(novel_reactions) * int(repeat)


def sample_random_excluding(
    rng: random.Random,
    values: list[str],
    forbidden: set[str],
    count: int,
) -> list[str]:
    """Sample distinct train-only candidates without scanning the full universe."""
    target = min(max(0, count), max(0, len(values) - len(forbidden)))
    selected: set[str] = set()
    attempts = 0
    max_attempts = max(100, target * 50)
    while len(selected) < target and attempts < max_attempts:
        value = rng.choice(values)
        attempts += 1
        if value not in forbidden:
            selected.add(value)
    if len(selected) < target:
        # Only used when the forbidden set is unusually dense.
        eligible = [value for value in values if value not in forbidden and value not in selected]
        selected.update(rng.sample(eligible, min(target - len(selected), len(eligible))))
    return sorted(selected)


def build_reaction_neighbors(
    reaction_features: np.ndarray,
    reaction_ids: list[str],
    *,
    train_reactions: set[str],
    query_reactions: set[str],
    topk: int,
    device: torch.device,
    batch_size: int = 128,
) -> dict[str, list[tuple[str, float]]]:
    """Top train-reaction neighbors using a Tanimoto kernel over the DRFP block.

    The benchmark's exact negative generator uses a molecular Morgan reaction
    similarity.  DRFP-Tanimoto is deliberately used here as a fast training-only
    proxy.  No target-benchmark metadata enters this graph.
    """
    if topk <= 0:
        return {value: [] for value in query_reactions}
    index = {value: row for row, value in enumerate(reaction_ids)}
    train_ids = sorted(set(train_reactions) & set(index))
    query_ids = sorted(set(query_reactions) & set(index))
    if not train_ids or not query_ids:
        raise ValueError("reaction-neighbor graph has empty train/query support")
    train_rows = np.asarray([index[value] for value in train_ids], dtype=np.int64)
    query_rows = np.asarray([index[value] for value in query_ids], dtype=np.int64)
    # DRFP occupies the first 2048 dimensions in the registered schema.
    train_np = np.asarray(reaction_features[train_rows, :2048], dtype=np.float32)
    train = torch.as_tensor(train_np, device=device)
    train_norm2 = (train * train).sum(dim=1).clamp_min(1e-8)
    train_local = {value: i for i, value in enumerate(train_ids)}
    output: dict[str, list[tuple[str, float]]] = {}
    k = min(topk + 1, len(train_ids))
    with torch.no_grad():
        for start in range(0, len(query_ids), batch_size):
            local_ids = query_ids[start : start + batch_size]
            rows = query_rows[start : start + len(local_ids)]
            query = torch.as_tensor(
                np.asarray(reaction_features[rows, :2048], dtype=np.float32),
                device=device,
            )
            dot = query @ train.T
            query_norm2 = (query * query).sum(dim=1, keepdim=True).clamp_min(1e-8)
            similarity = dot / (query_norm2 + train_norm2.unsqueeze(0) - dot).clamp_min(1e-8)
            values, cols = torch.topk(similarity, k=k, dim=1)
            for row_i, query_id in enumerate(local_ids):
                records: list[tuple[str, float]] = []
                for score, col in zip(values[row_i].tolist(), cols[row_i].tolist()):
                    candidate = train_ids[int(col)]
                    if candidate == query_id:
                        continue
                    records.append((candidate, float(score)))
                    if len(records) >= topk:
                        break
                output[query_id] = records
    return output


def hard_proteins_for_reaction(
    query_id: str,
    *,
    positives: set[str],
    neighbors: dict[str, list[tuple[str, float]]],
    proteins_by_reaction: dict[str, set[str]],
    limit: int,
) -> list[str]:
    hardness: dict[str, float] = {}
    for reaction_id, similarity in neighbors.get(query_id, []):
        for protein_id in proteins_by_reaction.get(reaction_id, set()):
            if protein_id in positives:
                continue
            hardness[protein_id] = max(hardness.get(protein_id, -np.inf), similarity)
    return [
        protein_id
        for protein_id, _ in sorted(hardness.items(), key=lambda item: (-item[1], item[0]))[:limit]
    ]


def hard_reactions_for_protein(
    protein_id: str,
    *,
    positives: set[str],
    neighbors: dict[str, list[tuple[str, float]]],
    limit: int,
) -> list[str]:
    hardness: dict[str, float] = {}
    for positive_reaction in positives:
        for reaction_id, similarity in neighbors.get(positive_reaction, []):
            if reaction_id in positives:
                continue
            hardness[reaction_id] = max(hardness.get(reaction_id, -np.inf), similarity)
    return [
        reaction_id
        for reaction_id, _ in sorted(hardness.items(), key=lambda item: (-item[1], item[0]))[:limit]
    ]


def build_candidate_batch(
    query_ids: list[str],
    *,
    direction: str,
    positives_by_query: dict[str, set[str]],
    neighbors: dict[str, list[tuple[str, float]]],
    proteins_by_reaction: dict[str, set[str]],
    candidate_universe: list[str],
    hard_negatives: int,
    random_negatives: int,
    rng: random.Random,
) -> tuple[list[str], np.ndarray]:
    selected: set[str] = set()
    hard_by_query: dict[str, list[str]] = {}
    for query_id in query_ids:
        positives = positives_by_query[query_id]
        selected.update(positives)
        if direction == "r2e":
            hard = hard_proteins_for_reaction(
                query_id,
                positives=positives,
                neighbors=neighbors,
                proteins_by_reaction=proteins_by_reaction,
                limit=hard_negatives,
            )
        elif direction == "e2r":
            hard = hard_reactions_for_protein(
                query_id,
                positives=positives,
                neighbors=neighbors,
                limit=hard_negatives,
            )
        else:
            raise ValueError(direction)
        hard_by_query[query_id] = hard
        selected.update(hard)
        forbidden = positives | set(hard)
        selected.update(
            sample_random_excluding(
                rng,
                candidate_universe,
                forbidden,
                random_negatives,
            )
        )
    candidate_ids = sorted(selected)
    local = {value: i for i, value in enumerate(candidate_ids)}
    positive_mask = np.zeros((len(query_ids), len(candidate_ids)), dtype=bool)
    for row, query_id in enumerate(query_ids):
        # Mark *all* known positives that happen to enter the union, including
        # positives pulled in by another query. They must never become negatives.
        for candidate in positives_by_query[query_id]:
            col = local.get(candidate)
            if col is not None:
                positive_mask[row, col] = True
    if not positive_mask.any(axis=1).all():
        raise AssertionError("every sampled query must retain at least one positive")
    return candidate_ids, positive_mask


def multi_positive_topk_loss(
    query_embeddings: torch.Tensor,
    candidate_embeddings: torch.Tensor,
    positive_mask: torch.Tensor,
    *,
    temperature: float,
    topk: int,
    topk_weight: float,
    margin: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    if temperature <= 0 or topk <= 0 or topk_weight < 0 or margin < 0:
        raise ValueError("invalid loss hyperparameters")
    logits = query_embeddings @ candidate_embeddings.T / temperature
    negative_inf = torch.finfo(logits.dtype).min
    positive_logits = logits.masked_fill(~positive_mask, negative_inf)
    contrastive = (
        torch.logsumexp(logits, dim=1) - torch.logsumexp(positive_logits, dim=1)
    ).mean()
    negative_logits = logits.masked_fill(positive_mask, negative_inf)
    eligible = (~positive_mask).sum(dim=1)
    k = min(int(topk), int(logits.shape[1]))
    kth = torch.topk(negative_logits, k=k, dim=1).values[:, -1]
    best_positive = positive_logits.max(dim=1).values
    valid = eligible >= k
    topk_loss = (
        F.softplus(kth[valid] - best_positive[valid] + margin).mean()
        if bool(valid.any())
        else torch.zeros((), dtype=logits.dtype, device=logits.device)
    )
    total = contrastive + float(topk_weight) * topk_loss
    return total, {
        "contrastive": float(contrastive.detach().cpu()),
        "topk": float(topk_loss.detach().cpu()),
    }


def _tensor_rows(
    values: np.ndarray,
    identifiers: list[str],
    index: dict[str, int],
    device: torch.device,
) -> torch.Tensor:
    rows = np.asarray([index[value] for value in identifiers], dtype=np.int64)
    return torch.as_tensor(values[rows], dtype=torch.float32, device=device)


def train_cleanroom(
    protein_features: np.ndarray,
    protein_ids: list[str],
    reaction_features: np.ndarray,
    reaction_ids: list[str],
    train_pairs: pd.DataFrame,
    *,
    config: ModelConfig,
    epochs: int,
    steps_per_epoch: int,
    reaction_batch_size: int,
    protein_batch_size: int,
    hard_negatives: int,
    random_negatives: int,
    hard_negative_start: int,
    hard_negative_ramp_epochs: int,
    neighbor_k: int,
    learning_rate: float,
    weight_decay: float,
    temperature: float,
    topk: int,
    topk_weight: float,
    margin: float,
    r2e_weight: float,
    reaction_novelty_threshold: float,
    reaction_novelty_repeat: int,
    seed: int,
    device: torch.device,
    neighbor_queries: set[str] | None = None,
) -> tuple[
    TerpeneDualTower,
    list[dict[str, float]],
    dict[str, list[tuple[str, float]]],
    dict[str, object],
]:
    if not 0 <= r2e_weight <= 1:
        raise ValueError("r2e_weight must be in [0,1]")
    seed_everything(seed)
    rng = random.Random(seed)
    model = TerpeneDualTower(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    pindex = {value: i for i, value in enumerate(protein_ids)}
    rindex = {value: i for i, value in enumerate(reaction_ids)}
    pairs = train_pairs[
        train_pairs["protein_id"].isin(pindex) & train_pairs["reaction_id"].isin(rindex)
    ].drop_duplicates(["protein_id", "reaction_id"]).copy()
    if pairs.empty:
        raise ValueError("no clean training pairs remain")
    proteins_by_reaction, reactions_by_protein = positive_maps(pairs)
    train_reactions = sorted(proteins_by_reaction)
    train_proteins = sorted(reactions_by_protein)
    novelty_stats: dict[str, object] = {
        "enabled": False,
        "threshold": float(reaction_novelty_threshold),
        "repeat": int(reaction_novelty_repeat),
        "novel_query_count": 0,
    }
    reaction_sampling_pool = list(train_reactions)
    if reaction_novelty_repeat > 0:
        novel_reactions, measured = _train_reaction_novelty(
            reaction_features, reaction_ids, train_reactions,
            threshold=reaction_novelty_threshold, device=device,
        )
        reaction_sampling_pool = _reaction_replay_pool(
            train_reactions, novel_reactions, repeat=int(reaction_novelty_repeat)
        )
        novelty_stats = {
            **measured,
            "enabled": True,
            "repeat": int(reaction_novelty_repeat),
            "sampling_pool_size": int(len(reaction_sampling_pool)),
        }
    query_reactions = set(train_reactions) | set(neighbor_queries or set())
    neighbors = build_reaction_neighbors(
        reaction_features,
        reaction_ids,
        train_reactions=set(train_reactions),
        query_reactions=query_reactions,
        topk=neighbor_k,
        device=device,
    )
    history: list[dict[str, float]] = []
    for epoch in range(1, epochs + 1):
        epoch_hard_negatives, epoch_random_negatives = _negative_curriculum_counts(
            epoch=epoch,
            target_hard=hard_negatives,
            target_random=random_negatives,
            start_hard=hard_negative_start,
            ramp_epochs=hard_negative_ramp_epochs,
        )
        model.train()
        total_values: list[float] = []
        r2e_values: list[float] = []
        e2r_values: list[float] = []
        r2e_topk_values: list[float] = []
        e2r_topk_values: list[float] = []
        for _ in range(steps_per_epoch):
            rq = _sample_ids(rng, reaction_sampling_pool, reaction_batch_size)
            r_candidates, r_mask_np = build_candidate_batch(
                rq,
                direction="r2e",
                positives_by_query=proteins_by_reaction,
                neighbors=neighbors,
                proteins_by_reaction=proteins_by_reaction,
                candidate_universe=train_proteins,
                hard_negatives=epoch_hard_negatives,
                random_negatives=epoch_random_negatives,
                rng=rng,
            )
            pq = _sample_ids(rng, train_proteins, protein_batch_size)
            p_candidates, p_mask_np = build_candidate_batch(
                pq,
                direction="e2r",
                positives_by_query=reactions_by_protein,
                neighbors=neighbors,
                proteins_by_reaction=proteins_by_reaction,
                candidate_universe=train_reactions,
                hard_negatives=epoch_hard_negatives,
                random_negatives=epoch_random_negatives,
                rng=rng,
            )
            rq_emb = model.encode_reactions(_tensor_rows(reaction_features, rq, rindex, device))
            rc_emb = model.encode_proteins(_tensor_rows(protein_features, r_candidates, pindex, device))
            pq_emb = model.encode_proteins(_tensor_rows(protein_features, pq, pindex, device))
            pc_emb = model.encode_reactions(_tensor_rows(reaction_features, p_candidates, rindex, device))
            rmask = torch.as_tensor(r_mask_np, dtype=torch.bool, device=device)
            pmask = torch.as_tensor(p_mask_np, dtype=torch.bool, device=device)
            r_loss, r_parts = multi_positive_topk_loss(
                rq_emb,
                rc_emb,
                rmask,
                temperature=temperature,
                topk=topk,
                topk_weight=topk_weight,
                margin=margin,
            )
            p_loss, p_parts = multi_positive_topk_loss(
                pq_emb,
                pc_emb,
                pmask,
                temperature=temperature,
                topk=topk,
                topk_weight=topk_weight,
                margin=margin,
            )
            loss = float(r2e_weight) * r_loss + (1.0 - float(r2e_weight)) * p_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total_values.append(float(loss.detach().cpu()))
            r2e_values.append(float(r_parts["contrastive"]))
            e2r_values.append(float(p_parts["contrastive"]))
            r2e_topk_values.append(float(r_parts["topk"]))
            e2r_topk_values.append(float(p_parts["topk"]))
        row = {
            "epoch": epoch,
            "hard_negatives": int(epoch_hard_negatives),
            "random_negatives": int(epoch_random_negatives),
            "loss": float(np.mean(total_values)),
            "r2e_contrastive": float(np.mean(r2e_values)),
            "e2r_contrastive": float(np.mean(e2r_values)),
            "r2e_topk": float(np.mean(r2e_topk_values)),
            "e2r_topk": float(np.mean(e2r_topk_values)),
        }
        history.append(row)
        print(json.dumps(row), flush=True)
    return model, history, neighbors, novelty_stats


def build_author_like_dev_reservoir(
    dev_pairs: pd.DataFrame,
    *,
    neighbors: dict[str, list[tuple[str, float]]],
    train_proteins_by_reaction: dict[str, set[str]],
    neighbor_reactions: int,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    dev_by_reaction, _ = positive_maps(dev_pairs)
    for reaction_id, positives in sorted(dev_by_reaction.items()):
        candidates = set(positives)
        for neighbor_id, _ in neighbors.get(reaction_id, [])[:neighbor_reactions]:
            candidates.update(train_proteins_by_reaction.get(neighbor_id, set()))
        for protein_id in sorted(candidates):
            rows.append(
                {
                    "reaction_id": reaction_id,
                    "protein_id": protein_id,
                    "label": int(protein_id in positives),
                }
            )
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise ValueError("empty author-like development reservoir")
    return frame.drop_duplicates(["reaction_id", "protein_id"], keep="first")


def score_reservoir(
    model: TerpeneDualTower,
    reservoir: pd.DataFrame,
    protein_features: np.ndarray,
    protein_ids: list[str],
    reaction_features: np.ndarray,
    reaction_ids: list[str],
    *,
    device: torch.device,
) -> pd.DataFrame:
    pindex = {value: i for i, value in enumerate(protein_ids)}
    rindex = {value: i for i, value in enumerate(reaction_ids)}
    data = reservoir[
        reservoir["protein_id"].isin(pindex) & reservoir["reaction_id"].isin(rindex)
    ].copy()
    unique_p = sorted(data["protein_id"].unique())
    unique_r = sorted(data["reaction_id"].unique())
    local_p = {value: i for i, value in enumerate(unique_p)}
    local_r = {value: i for i, value in enumerate(unique_r)}
    model.eval()
    with torch.no_grad():
        p_emb = model.encode_proteins(_tensor_rows(protein_features, unique_p, pindex, device))
        r_emb = model.encode_reactions(_tensor_rows(reaction_features, unique_r, rindex, device))
        p_rows = torch.as_tensor([local_p[value] for value in data["protein_id"]], device=device)
        r_rows = torch.as_tensor([local_r[value] for value in data["reaction_id"]], device=device)
        data["score"] = (p_emb[p_rows] * r_emb[r_rows]).sum(dim=1).cpu().numpy()
    return data


def evaluate_dev(frame: pd.DataFrame) -> tuple[dict[str, object], pd.DataFrame]:
    ranking = frame.rename(
        columns={"reaction_id": "query_id", "protein_id": "candidate_id"}
    )[["query_id", "candidate_id", "score", "label"]]
    query, common = evaluate_ranking_frame(
        ranking, budgets=DEFAULT_BUDGETS, top_percents=DEFAULT_TOP_PERCENTS
    )
    native = evaluate_enzymecage_native_r2e(frame, "score")
    return {"author_native_r2e": native, "common_ir_r2e": common}, query


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Random-init 2023-only dual-tower training with EnzymeCAGE-like reaction-neighborhood "
            "hard negatives. Target benchmark labels are never read."
        )
    )
    parser.add_argument("--associations-csv", type=Path, default=DEFAULT_ASSOCIATIONS)
    parser.add_argument("--universe-dir", type=Path, default=DEFAULT_UNIVERSE)
    parser.add_argument("--schema-dir", type=Path, default=DEFAULT_SCHEMA_DIR)
    parser.add_argument(
        "--reaction-feature-dir", type=Path, default=None,
        help="Registered reaction feature library; defaults to <universe>/reaction_features/drfp_categorical_v1.",
    )
    parser.add_argument(
        "--protein-feature-dir", type=Path, default=None,
        help="Protein embedding library with entries.csv + embeddings.npy; defaults to <universe>/proteins.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--dev-fold", type=int, default=-1, help="-1 trains on all 2023 pairs; otherwise train/dev use strict entity-disjoint hash fold")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--steps-per-epoch", type=int, default=100)
    parser.add_argument("--reaction-batch-size", type=int, default=48)
    parser.add_argument("--protein-batch-size", type=int, default=48)
    parser.add_argument("--neighbor-k", type=int, default=20)
    parser.add_argument("--dev-neighbor-reactions", type=int, default=10)
    parser.add_argument("--hard-negatives", type=int, default=32)
    parser.add_argument("--random-negatives", type=int, default=16)
    parser.add_argument(
        "--hard-negative-start", type=int, default=0,
        help="Hard negatives in epoch 1 when curriculum is enabled.",
    )
    parser.add_argument(
        "--hard-negative-ramp-epochs", type=int, default=0,
        help="Reach --hard-negatives by this epoch while preserving total negative budget; 0 disables curriculum.",
    )
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--embedding-dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--temperature", type=float, default=0.05)
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--topk-weight", type=float, default=0.35)
    parser.add_argument("--margin", type=float, default=0.05)
    parser.add_argument("--r2e-weight", type=float, default=0.70)
    parser.add_argument(
        "--reaction-novelty-threshold", type=float, default=0.7,
        help="Train-only nearest-other binary-DRFP threshold used to identify pseudo-novel reaction queries.",
    )
    parser.add_argument(
        "--reaction-novelty-repeat", type=int, default=0,
        help="Extra copies of each pseudo-novel reaction in the R2E query sampling pool; 0 disables replay.",
    )
    parser.add_argument("--seed", type=int, default=20260723)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if args.epochs <= 0 or args.steps_per_epoch <= 0:
        raise ValueError("epochs and steps-per-epoch must be positive")
    if args.reaction_batch_size <= 0 or args.protein_batch_size <= 0:
        raise ValueError("batch sizes must be positive")
    if args.hard_negatives < 0 or args.random_negatives < 0 or args.hard_negative_ramp_epochs < 0:
        raise ValueError("negative counts and hard-negative ramp epochs must be non-negative")
    if args.hard_negative_ramp_epochs > 0 and not 0 <= args.hard_negative_start <= args.hard_negatives:
        raise ValueError("hard-negative-start must be in [0, hard-negatives]")
    if args.reaction_novelty_repeat < 0:
        raise ValueError("reaction novelty repeat must be non-negative")
    if args.reaction_novelty_repeat > 0 and not 0.0 < args.reaction_novelty_threshold <= 1.0:
        raise ValueError("reaction novelty threshold must be in (0,1] when replay is enabled")

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    universe = args.universe_dir.resolve()
    association_source = args.associations_csv.resolve()
    schema_dir = args.schema_dir.resolve()
    schema = load_feature_schema(schema_dir)
    protein_feature_dir = (
        args.protein_feature_dir.resolve()
        if args.protein_feature_dir is not None
        else universe / "proteins"
    )
    protein_features, protein_ids = load_protein_library(protein_feature_dir)
    reaction_feature_dir = (
        args.reaction_feature_dir.resolve()
        if args.reaction_feature_dir is not None
        else universe / "reaction_features/drfp_categorical_v1"
    )
    reaction_features, reaction_ids = load_registered_reaction_feature_library(
        reaction_feature_dir, schema
    )
    pset, rset = set(protein_ids), set(reaction_ids)
    pairs = pd.read_csv(association_source, dtype=str).fillna("")
    required = {"protein_id", "reaction_id"}
    if not required <= set(pairs.columns):
        raise ValueError(f"association source needs {sorted(required)}")
    pairs = pairs[pairs["protein_id"].isin(pset) & pairs["reaction_id"].isin(rset)]
    pairs = pairs[["protein_id", "reaction_id"]].drop_duplicates().reset_index(drop=True)
    if args.dev_fold >= 0:
        train_pairs, dev_pairs = split_double_cold(
            pairs, dev_fold=args.dev_fold, folds=args.folds
        )
    else:
        train_pairs, dev_pairs = pairs.copy(), pd.DataFrame(columns=pairs.columns)
    config = ModelConfig(
        protein_input_dim=int(protein_features.shape[1]),
        reaction_input_dim=int(reaction_features.shape[1]),
        hidden_dim=args.hidden_dim,
        embedding_dim=args.embedding_dim,
        dropout=args.dropout,
    )
    neighbor_queries = set(dev_pairs["reaction_id"].astype(str)) if len(dev_pairs) else set()
    model, history, neighbors, novelty_stats = train_cleanroom(
        protein_features,
        protein_ids,
        reaction_features,
        reaction_ids,
        train_pairs,
        config=config,
        epochs=args.epochs,
        steps_per_epoch=args.steps_per_epoch,
        reaction_batch_size=args.reaction_batch_size,
        protein_batch_size=args.protein_batch_size,
        hard_negatives=args.hard_negatives,
        random_negatives=args.random_negatives,
        hard_negative_start=args.hard_negative_start,
        hard_negative_ramp_epochs=args.hard_negative_ramp_epochs,
        neighbor_k=max(args.neighbor_k, args.dev_neighbor_reactions),
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        temperature=args.temperature,
        topk=args.topk,
        topk_weight=args.topk_weight,
        margin=args.margin,
        r2e_weight=args.r2e_weight,
        reaction_novelty_threshold=args.reaction_novelty_threshold,
        reaction_novelty_repeat=args.reaction_novelty_repeat,
        seed=args.seed,
        device=device,
        neighbor_queries=neighbor_queries,
    )
    model_dir = output / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = model_dir / f"production_seed{args.seed}.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_config": asdict(config),
            "model_type": "dual_tower",
            "seed": args.seed,
            "cleanroom_random_init": True,
            "base_checkpoint": None,
            "training_source": str(association_source),
            "training_source_sha256": sha256_file(association_source),
            "dev_fold": args.dev_fold,
            "folds": args.folds,
            "reaction_novelty_replay": novelty_stats,
        },
        checkpoint_path,
    )
    output_schema = dict(schema)
    output_schema["protein_feature_dimension"] = int(protein_features.shape[1])
    output_schema["reaction_feature_dimension"] = int(reaction_features.shape[1])
    (output / "feature_schema.json").write_text(
        json.dumps(output_schema, indent=2), encoding="utf-8"
    )
    train_pairs.to_csv(output / "training_pairs.csv", index=False)
    pd.DataFrame(history).to_csv(output / "training_history.csv", index=False)

    dev_metrics: dict[str, object] | None = None
    if len(dev_pairs):
        train_by_reaction, _ = positive_maps(train_pairs)
        reservoir = build_author_like_dev_reservoir(
            dev_pairs,
            neighbors=neighbors,
            train_proteins_by_reaction=train_by_reaction,
            neighbor_reactions=args.dev_neighbor_reactions,
        )
        scored = score_reservoir(
            model,
            reservoir,
            protein_features,
            protein_ids,
            reaction_features,
            reaction_ids,
            device=device,
        )
        dev_metrics, query_metrics = evaluate_dev(scored)
        scored.to_csv(output / "dev_pair_scores.csv", index=False)
        query_metrics.to_csv(output / "dev_query_metrics.csv", index=False)
        dev_pairs.to_csv(output / "dev_pairs.csv", index=False)

    summary = {
        "method": "cleanroom_random_init_bidirectional_hard_reservoir_dual_tower",
        "target_benchmark_labels_read": False,
        "target_benchmark_metadata_used_for_training": False,
        "association_source": str(association_source),
        "association_source_sha256": sha256_file(association_source),
        "protein_feature_dir": str(protein_feature_dir),
        "reaction_feature_dir": str(reaction_feature_dir),
        "reaction_feature_manifest_sha256": sha256_file(reaction_feature_dir / "manifest.json"),
        "random_initialization": True,
        "base_checkpoint": None,
        "dev_protocol": (
            "strict protein+reaction hash double-cold inside 2023 snapshot"
            if args.dev_fold >= 0
            else "none; full 2023 retraining"
        ),
        "dev_fold": args.dev_fold,
        "folds": args.folds,
        "n_source_pairs": int(len(pairs)),
        "n_train_pairs": int(len(train_pairs)),
        "n_dev_pairs": int(len(dev_pairs)),
        "n_train_proteins": int(train_pairs["protein_id"].nunique()),
        "n_train_reactions": int(train_pairs["reaction_id"].nunique()),
        "model_config": asdict(config),
        "training": {
            "epochs": args.epochs,
            "steps_per_epoch": args.steps_per_epoch,
            "reaction_batch_size": args.reaction_batch_size,
            "protein_batch_size": args.protein_batch_size,
            "neighbor_similarity": "Tanimoto over the 2048-d DRFP block, train entities only",
            "neighbor_k": args.neighbor_k,
            "hard_negatives": args.hard_negatives,
            "random_negatives": args.random_negatives,
            "hard_negative_curriculum": {
                "start_hard": args.hard_negative_start,
                "ramp_epochs": args.hard_negative_ramp_epochs,
                "preserve_total_negative_budget": True,
            },
            "temperature": args.temperature,
            "topk": args.topk,
            "topk_weight": args.topk_weight,
            "margin": args.margin,
            "r2e_weight": args.r2e_weight,
            "reaction_novelty_replay": novelty_stats,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
        },
        "dev_metrics": dev_metrics,
        "checkpoint": str(checkpoint_path),
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
