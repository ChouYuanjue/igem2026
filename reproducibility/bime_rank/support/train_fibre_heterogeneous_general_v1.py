from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from projects.active.bridge.kernel.heterogeneous_modes import (
    HeterogeneousConditionalFibre,
    HeterogeneousFibreConfig,
)
from reproducibility.bime_rank.support.evaluate_multi_expert_protocol_comparison import (
    MultiExpertConfig,
    gate_regularization,
)


DEFAULT_ASSET_ROOT = Path("/home/s241850073/igem2026")
DEFAULT_OUT = Path("results/fibre_heterogeneous_general_v1_seed20260723")
DEFAULT_PAIRS = Path("data/external/enzymecage_current/catalyst_features/clean2023/training_pairs.csv")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _id_column(frame: pd.DataFrame, kind: str) -> str:
    candidates = (
        ("Entry", "protein_id", "query_id", "id")
        if kind == "protein"
        else ("reaction_id", "rhea_id", "id")
    )
    for value in candidates:
        if value in frame.columns:
            return value
    raise ValueError(f"cannot infer {kind} id column from {frame.columns.tolist()}")


def load_entries(path: Path, kind: str) -> tuple[list[str], dict[str, int]]:
    frame = pd.read_csv(path, dtype=str).fillna("")
    id_col = _id_column(frame, kind)
    if "row" in frame:
        frame["row"] = pd.to_numeric(frame["row"]).astype(int)
        frame = frame.sort_values("row")
    ids = frame[id_col].astype(str).tolist()
    return ids, {value: index for index, value in enumerate(ids)}


def align_rows(base_ids: list[str], other_map: dict[str, int]) -> np.ndarray:
    return np.asarray([other_map.get(value, -1) for value in base_ids], dtype=np.int64)


def gather(
    matrix: np.ndarray,
    rows: np.ndarray,
    *,
    fill_dim: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    rows = np.asarray(rows, dtype=np.int64)
    available = rows >= 0
    dim = int(fill_dim or matrix.shape[1])
    out = np.zeros((len(rows), dim), dtype=np.float32)
    if available.any():
        out[available] = np.asarray(matrix[rows[available]], dtype=np.float32)
    return out, available


def cosine_pairs(
    left_matrix: np.ndarray,
    left_rows: np.ndarray,
    right_matrix: np.ndarray,
    right_rows: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    left, left_ok = gather(left_matrix, left_rows)
    right, right_ok = gather(right_matrix, right_rows)
    ok = left_ok & right_ok
    score = np.zeros(len(left_rows), dtype=np.float32)
    if ok.any():
        lv = left[ok]
        rv = right[ok]
        denom = np.linalg.norm(lv, axis=1) * np.linalg.norm(rv, axis=1)
        valid = denom > 0
        local = np.zeros(len(lv), dtype=np.float32)
        local[valid] = (lv[valid] * rv[valid]).sum(axis=1) / denom[valid]
        score[ok] = local
        ok_idx = np.flatnonzero(ok)
        ok[ok_idx[~valid]] = False
    return score, ok


class Assets:
    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        base = self.root / "data/catalyst_candidate_universes/general_merged"
        self.protein_dir = base / "proteins"
        self.reaction_dir = base / "reaction_features/drfp_categorical_rdkitplus_v1"
        self.center_dir = base / "reaction_features/drfp_categorical_rdkitplus_center_v1"
        self.enzgfm_dir = self.root / "data/external/enzgfm_current/general_merged_650m_mean_v1"
        self.clip_protein_dir = self.root / "results/bime_rank_unified_v1/clipzyme_r2e_candidate_asset_v1"
        self.clip_reaction_dir = (
            self.root
            / "results/clipzyme_native_extension_v1/full_hplus_candidate_reactions/clipzyme_embeddings_gpu_v1"
        )

        self.protein_ids, self.protein_map = load_entries(
            self.protein_dir / "entries.csv", "protein"
        )
        self.reaction_ids, self.reaction_map = load_entries(
            self.reaction_dir / "entries.csv", "reaction"
        )
        # The general assets are only a few GiB in total on nju-server-06,
        # while sparse negative sampling performs highly non-contiguous row
        # access. Keep them resident in RAM instead of forcing thousands of
        # random mmap reads per epoch.
        self.protein = np.load(self.protein_dir / "embeddings.npy").astype(
            np.float32, copy=False
        )
        self.reaction = np.load(
            self.reaction_dir / "reaction_feature_matrix.npy"
        ).astype(np.float32, copy=False)
        center_full = np.load(
            self.center_dir / "reaction_feature_matrix.npy"
        ).astype(np.float32, copy=False)
        if center_full.shape[0] != self.reaction.shape[0]:
            raise ValueError("reaction-center and base reaction rows differ")
        if center_full.shape[1] <= self.reaction.shape[1]:
            raise ValueError("reaction-center matrix has no appended center coordinates")
        self.center = center_full[:, self.reaction.shape[1] :]

        enz_ids, enz_map = load_entries(self.enzgfm_dir / "entries.csv", "protein")
        self.enzgfm = np.load(self.enzgfm_dir / "embeddings.npy").astype(
            np.float32, copy=False
        )
        self.enz_rows = align_rows(self.protein_ids, enz_map)

        cp_ids, cp_map = load_entries(
            self.clip_protein_dir / "entries.csv", "protein"
        )
        self.clip_protein = np.load(
            self.clip_protein_dir / "embeddings.npy"
        ).astype(np.float32, copy=False)
        self.clip_protein_rows = align_rows(self.protein_ids, cp_map)

        cr_frame = pd.read_csv(self.clip_reaction_dir / "entries.csv", dtype=str).fillna("")
        cr_col = _id_column(cr_frame, "reaction")
        if "row" in cr_frame:
            cr_frame["row"] = pd.to_numeric(cr_frame["row"]).astype(int)
        if "clipzyme_supported" in cr_frame:
            supported = cr_frame["clipzyme_supported"].astype(str).str.lower().eq("true")
            cr_frame = cr_frame[supported]
        cr_map = dict(
            zip(
                cr_frame[cr_col].astype(str),
                (
                    cr_frame["row"].astype(int)
                    if "row" in cr_frame
                    else np.arange(len(cr_frame), dtype=int)
                ),
            )
        )
        self.clip_reaction = np.load(
            self.clip_reaction_dir / "embeddings.npy"
        ).astype(np.float32, copy=False)
        self.clip_reaction_rows = align_rows(self.reaction_ids, cr_map)

        self.center_available = np.linalg.norm(
            np.asarray(self.center, dtype=np.float32), axis=1
        ) > 0

    @property
    def n_proteins(self) -> int:
        return len(self.protein_ids)

    @property
    def n_reactions(self) -> int:
        return len(self.reaction_ids)

    def tensor_inputs(
        self,
        reaction_rows: np.ndarray,
        protein_rows: np.ndarray,
        device: torch.device,
    ) -> dict[str, torch.Tensor]:
        reaction_rows = np.asarray(reaction_rows, dtype=np.int64)
        protein_rows = np.asarray(protein_rows, dtype=np.int64)
        protein = np.asarray(self.protein[protein_rows], dtype=np.float32)
        reaction = np.asarray(self.reaction[reaction_rows], dtype=np.float32)
        center = np.asarray(self.center[reaction_rows], dtype=np.float32)
        enz, enz_ok = gather(self.enzgfm, self.enz_rows[protein_rows])
        clip, clip_ok = cosine_pairs(
            self.clip_reaction,
            self.clip_reaction_rows[reaction_rows],
            self.clip_protein,
            self.clip_protein_rows[protein_rows],
        )
        return {
            "protein_values": torch.as_tensor(protein, device=device),
            "reaction_values": torch.as_tensor(reaction, device=device),
            "enzgfm_values": torch.as_tensor(enz, device=device),
            "reaction_center_values": torch.as_tensor(center, device=device),
            "enzgfm_available": torch.as_tensor(enz_ok, device=device),
            "reaction_center_available": torch.as_tensor(
                self.center_available[reaction_rows], device=device
            ),
            "clipzyme_scores": torch.as_tensor(clip, device=device),
            "clipzyme_available": torch.as_tensor(clip_ok, device=device),
            "seed_context_available": torch.zeros(
                len(reaction_rows), dtype=torch.bool, device=device
            ),
        }

    def seed_similarity(
        self,
        *,
        side: str,
        candidate_rows: np.ndarray,
        seed_rows: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Cosine evidence from another known positive of the same query."""
        candidate_rows = np.asarray(candidate_rows, dtype=np.int64)
        seed_rows = np.asarray(seed_rows, dtype=np.int64)
        if candidate_rows.shape != seed_rows.shape:
            raise ValueError("candidate_rows and seed_rows must align")
        available = seed_rows >= 0
        scores = np.zeros(len(candidate_rows), dtype=np.float32)
        if not available.any():
            return scores, available
        if side == "protein":
            matrix = self.protein
        elif side == "reaction":
            # The first 2048 coordinates are the DRFP block and have a
            # coherent cosine geometry; categorical/descriptor scales are
            # intentionally excluded from the context similarity.
            matrix = self.reaction[:, :2048]
        else:
            raise ValueError(side)
        candidate = np.asarray(matrix[candidate_rows[available]], dtype=np.float32)
        seed = np.asarray(matrix[seed_rows[available]], dtype=np.float32)
        denominator = np.linalg.norm(candidate, axis=1) * np.linalg.norm(seed, axis=1)
        good = denominator > 0
        local = np.zeros(len(candidate), dtype=np.float32)
        local[good] = (
            (candidate[good] * seed[good]).sum(axis=1) / denominator[good]
        )
        scores[available] = local
        available_indices = np.flatnonzero(available)
        available[available_indices[~good]] = False
        return scores, available


def sampled_topk_loss(logits: torch.Tensor) -> torch.Tensor:
    positive = logits[:, 0]
    negatives = logits[:, 1:]
    terms = ((3, 0.10), (10, 0.05), (20, 0.025))
    pieces: list[torch.Tensor] = []
    total = 0.0
    for k, weight in terms:
        local_k = min(k, negatives.shape[1])
        kth = torch.topk(negatives, k=local_k, dim=1).values[:, -1]
        pieces.append(weight * F.softplus(kth - positive).mean())
        total += weight
    return sum(pieces) / total


def sample_negative_matrix(
    rng: np.random.Generator,
    query_rows: np.ndarray,
    positive_sets: dict[int, set[int]],
    candidate_count: int,
    k: int,
) -> np.ndarray:
    result = rng.integers(0, candidate_count, size=(len(query_rows), k), dtype=np.int64)
    for i, query in enumerate(query_rows):
        positives = positive_sets[int(query)]
        bad = np.fromiter(
            (int(value) in positives for value in result[i]),
            dtype=bool,
            count=k,
        )
        while bad.any():
            result[i, bad] = rng.integers(0, candidate_count, size=int(bad.sum()))
            bad = np.fromiter(
                (int(value) in positives for value in result[i]),
                dtype=bool,
                count=k,
            )
    return result


def sample_context_seed(
    rng: np.random.Generator,
    query_rows: np.ndarray,
    target_rows: np.ndarray,
    positive_sets: dict[int, set[int]],
) -> np.ndarray:
    """Choose one other known positive for the same query when available."""
    result = np.full(len(query_rows), -1, dtype=np.int64)
    for index, (query, target) in enumerate(zip(query_rows, target_rows)):
        options = sorted(positive_sets[int(query)] - {int(target)})
        if options:
            result[index] = options[int(rng.integers(0, len(options)))]
    return result


def load_training_pairs(assets: Assets, pair_path: Path) -> pd.DataFrame:
    pairs = pd.read_csv(pair_path, dtype=str).fillna("")
    pcol = "protein_id" if "protein_id" in pairs else "Entry"
    rcol = "reaction_id" if "reaction_id" in pairs else "rhea_id"
    pairs = pairs[[pcol, rcol]].rename(columns={pcol: "protein_id", rcol: "reaction_id"})
    pairs = pairs.drop_duplicates()
    pairs["protein_row"] = pairs["protein_id"].map(assets.protein_map)
    pairs["reaction_row"] = pairs["reaction_id"].map(assets.reaction_map)
    pairs = pairs.dropna(subset=["protein_row", "reaction_row"]).copy()
    pairs["protein_row"] = pairs["protein_row"].astype(int)
    pairs["reaction_row"] = pairs["reaction_row"].astype(int)
    return pairs.reset_index(drop=True)


def train(
    *,
    assets: Assets,
    pairs: pd.DataFrame,
    seed: int,
    epochs: int,
    batch_size: int,
    negative_count: int,
    learning_rate: float,
    weight_decay: float,
    temperature: float,
    device: torch.device,
) -> tuple[HeterogeneousConditionalFibre, list[dict[str, float]]]:
    seed_everything(seed)
    base_cfg = MultiExpertConfig(
        protein_input_dim=int(assets.protein.shape[1]),
        reaction_input_dim=int(assets.reaction.shape[1]),
        hidden_dim=512,
        global_dim=128,
        n_experts=8,
        expert_dim=32,
        dropout=0.1,
        gate_temperature=1.0,
        expert_mix_init=0.5,
    )
    model = HeterogeneousConditionalFibre(
        HeterogeneousFibreConfig(
            base=base_cfg,
            enzgfm_input_dim=int(assets.enzgfm.shape[1]),
            reaction_center_input_dim=int(assets.center.shape[1]),
            specialist_hidden_dim=256,
            specialist_dim=64,
        )
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )

    reaction_positive: dict[int, set[int]] = {}
    protein_positive: dict[int, set[int]] = {}
    for row in pairs.itertuples(index=False):
        reaction_positive.setdefault(int(row.reaction_row), set()).add(int(row.protein_row))
        protein_positive.setdefault(int(row.protein_row), set()).add(int(row.reaction_row))

    order = np.arange(len(pairs), dtype=np.int64)
    history: list[dict[str, float]] = []
    best = math.inf
    best_state: dict[str, torch.Tensor] | None = None

    for epoch in range(1, epochs + 1):
        rng = np.random.default_rng(seed + epoch)
        rng.shuffle(order)
        totals = {"loss": 0.0, "r2e": 0.0, "e2r": 0.0, "topk": 0.0}
        seen = 0
        model.train()
        for start in range(0, len(order), batch_size):
            idx = order[start : start + batch_size]
            batch = pairs.iloc[idx]
            r = batch["reaction_row"].to_numpy(np.int64)
            p = batch["protein_row"].to_numpy(np.int64)
            b = len(batch)

            neg_p = sample_negative_matrix(
                rng, r, reaction_positive, assets.n_proteins, negative_count
            )
            r2e_r = np.repeat(r[:, None], negative_count + 1, axis=1)
            r2e_p = np.concatenate([p[:, None], neg_p], axis=1)
            r2e_inputs = assets.tensor_inputs(
                r2e_r.reshape(-1), r2e_p.reshape(-1), device
            )
            r2e_seed = sample_context_seed(
                rng,
                r,
                p,
                reaction_positive,
            )
            r2e_seed_rows = np.repeat(
                r2e_seed[:, None],
                negative_count + 1,
                axis=1,
            ).reshape(-1)
            r2e_seed_scores, r2e_seed_ok = assets.seed_similarity(
                side="protein",
                candidate_rows=r2e_p.reshape(-1),
                seed_rows=r2e_seed_rows,
            )
            r2e_inputs["seed_context_scores"] = torch.as_tensor(
                r2e_seed_scores,
                device=device,
            )
            r2e_inputs["seed_context_available"] = torch.as_tensor(
                r2e_seed_ok,
                device=device,
            )
            r2e_score, _, rdiag = model.score_pairs(**r2e_inputs)
            r2e_logits = r2e_score.reshape(b, negative_count + 1) / temperature
            target = torch.zeros(b, dtype=torch.long, device=device)
            r2e_loss = F.cross_entropy(r2e_logits, target)
            r2e_topk = sampled_topk_loss(r2e_logits)

            neg_r = sample_negative_matrix(
                rng, p, protein_positive, assets.n_reactions, negative_count
            )
            e2r_p = np.repeat(p[:, None], negative_count + 1, axis=1)
            e2r_r = np.concatenate([r[:, None], neg_r], axis=1)
            e2r_inputs = assets.tensor_inputs(
                e2r_r.reshape(-1), e2r_p.reshape(-1), device
            )
            e2r_seed = sample_context_seed(
                rng,
                p,
                r,
                protein_positive,
            )
            e2r_seed_rows = np.repeat(
                e2r_seed[:, None],
                negative_count + 1,
                axis=1,
            ).reshape(-1)
            e2r_seed_scores, e2r_seed_ok = assets.seed_similarity(
                side="reaction",
                candidate_rows=e2r_r.reshape(-1),
                seed_rows=e2r_seed_rows,
            )
            e2r_inputs["seed_context_scores"] = torch.as_tensor(
                e2r_seed_scores,
                device=device,
            )
            e2r_inputs["seed_context_available"] = torch.as_tensor(
                e2r_seed_ok,
                device=device,
            )
            _, e2r_score, ediag = model.score_pairs(**e2r_inputs)
            e2r_logits = e2r_score.reshape(b, negative_count + 1) / temperature
            e2r_loss = F.cross_entropy(e2r_logits, target)
            e2r_topk = sampled_topk_loss(e2r_logits)

            rb, re, rd = gate_regularization(
                rdiag["reaction_gates"], rdiag["reaction_experts"]
            )
            pb, pe, pdv = gate_regularization(
                ediag["protein_gates"], ediag["protein_experts"]
            )
            balance = 0.5 * (rb + pb)
            entropy = 0.5 * (re + pe)
            diversity = 0.5 * (rd + pdv)

            contrastive = 0.5 * (r2e_loss + e2r_loss)
            topk = 0.5 * (r2e_topk + e2r_topk)
            loss = (
                contrastive
                + topk
                + 0.05 * balance
                + 0.005 * entropy
                + 0.01 * diversity
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            totals["loss"] += float(loss.detach()) * b
            totals["r2e"] += float(r2e_loss.detach()) * b
            totals["e2r"] += float(e2r_loss.detach()) * b
            totals["topk"] += float(topk.detach()) * b
            seen += b

        row = {
            "epoch": float(epoch),
            **{key: value / max(seen, 1) for key, value in totals.items()},
        }
        history.append(row)
        print(json.dumps(row), flush=True)
        if row["loss"] < best:
            best = row["loss"]
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history


def encode_in_chunks(
    model: HeterogeneousConditionalFibre,
    assets: Assets,
    *,
    side: str,
    rows: np.ndarray,
    device: torch.device,
    chunk: int = 4096,
) -> dict[str, torch.Tensor]:
    parts: dict[str, list[torch.Tensor]] = {}
    model.eval()
    with torch.no_grad():
        for start in range(0, len(rows), chunk):
            local = rows[start : start + chunk]
            if side == "protein":
                protein = torch.as_tensor(
                    np.asarray(assets.protein[local], dtype=np.float32), device=device
                )
                enz_np, _ = gather(assets.enzgfm, assets.enz_rows[local])
                enz = torch.as_tensor(enz_np, device=device)
                encoded = model.encode_protein_side(protein, enz)
            elif side == "reaction":
                reaction = torch.as_tensor(
                    np.asarray(assets.reaction[local], dtype=np.float32), device=device
                )
                center = torch.as_tensor(
                    np.asarray(assets.center[local], dtype=np.float32), device=device
                )
                encoded = model.encode_reaction_side(reaction, center)
            else:
                raise ValueError(side)
            for key, value in encoded.items():
                parts.setdefault(key, []).append(value.detach())
    return {key: torch.cat(values, dim=0) for key, values in parts.items()}


def query_metrics(order: list[str], positives: set[str]) -> dict[str, float]:
    rank = {value: i + 1 for i, value in enumerate(order)}
    ranks = sorted(rank[value] for value in positives if value in rank)
    if not ranks:
        raise ValueError("query has no positive in candidate support")
    p = len(ranks)
    ap = float(np.mean(np.arange(1, p + 1, dtype=float) / np.asarray(ranks, dtype=float)))
    gains = np.asarray([1.0 if value in positives else 0.0 for value in order[:10]])
    discounts = 1.0 / np.log2(np.arange(2, 12, dtype=float))
    ideal = float(discounts[: min(p, 10)].sum())
    return {
        "positive_count": float(p),
        "candidate_count": float(len(order)),
        "mrr": 1.0 / ranks[0],
        "map": ap,
        "ndcg_at_10": float((gains * discounts).sum() / ideal if ideal else 0.0),
        "best_positive_rank": float(ranks[0]),
        "mean_positive_rank": float(np.mean(ranks)),
        **{f"hit_at_{k}": float(ranks[0] <= k) for k in (1, 3, 5, 10, 20, 50)},
    }


def summarize(rows: list[dict[str, float]]) -> dict[str, float]:
    frame = pd.DataFrame(rows)
    out = {
        "queries": int(len(frame)),
        "candidate_count": int(frame["candidate_count"].iloc[0]),
        "positive_pairs": int(frame["positive_count"].sum()),
        "mrr": float(frame["mrr"].mean()),
        "map": float(frame["map"].mean()),
        "ndcg_at_10": float(frame["ndcg_at_10"].mean()),
        "median_best_positive_rank": float(frame["best_positive_rank"].median()),
    }
    for k in (1, 3, 5, 10, 20, 50):
        out[f"hit_at_{k}"] = float(frame[f"hit_at_{k}"].mean())
    return out


def clip_matrix_r2e(
    assets: Assets,
    reaction_rows: np.ndarray,
    protein_rows: np.ndarray,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    rr = assets.clip_reaction_rows[reaction_rows]
    pr = assets.clip_protein_rows[protein_rows]
    r, rok = gather(assets.clip_reaction, rr)
    p, pok = gather(assets.clip_protein, pr)
    rn = np.linalg.norm(r, axis=1, keepdims=True)
    pn = np.linalg.norm(p, axis=1, keepdims=True)
    rn[rn == 0] = 1.0
    pn[pn == 0] = 1.0
    score = torch.as_tensor((r / rn) @ (p / pn).T, device=device)
    avail = torch.as_tensor(rok[:, None] & pok[None, :], device=device)
    return score, avail


def evaluate_strict_temporal(
    *,
    model: HeterogeneousConditionalFibre,
    assets: Assets,
    asset_root: Path,
    device: torch.device,
    output_dir: Path,
) -> dict[str, object]:
    model.eval()
    result: dict[str, object] = {}

    # R2E: exact frozen 144-query x 166,202 common support.
    r2e_root = asset_root / "results/clipzyme_native_extension_v1"
    query_ids = [
        line.strip()
        for line in (
            r2e_root / "r2e_strict650_same_support_v1/mutual_cold_query_ids.txt"
        ).read_text().splitlines()
        if line.strip()
    ]
    candidate_ids = [
        line.strip()
        for line in (r2e_root / "r2e_strict650_candidate_ids.txt").read_text().splitlines()
        if line.strip()
    ]
    pair_frame = pd.read_csv(
        r2e_root / "r2e_strict650_same_support_v1/mutual_cold_test_pairs.csv",
        dtype=str,
    ).fillna("")
    positives = (
        pair_frame.groupby("reaction_id")["protein_id"]
        .agg(lambda values: set(values.astype(str)))
        .to_dict()
    )
    rrows = np.asarray([assets.reaction_map[q] for q in query_ids], dtype=np.int64)
    prows = np.asarray([assets.protein_map[p] for p in candidate_ids], dtype=np.int64)
    encoded_r = encode_in_chunks(model, assets, side="reaction", rows=rrows, device=device)
    encoded_p = encode_in_chunks(model, assets, side="protein", rows=prows, device=device)
    clip, clip_ok = clip_matrix_r2e(assets, rrows, prows, device)
    center_ok = torch.as_tensor(assets.center_available[rrows], device=device)
    enz_ok = torch.as_tensor(assets.enz_rows[prows] >= 0, device=device)
    with torch.no_grad():
        scores, _, diag = model.score_encoded_cross(
            proteins=encoded_p,
            reactions=encoded_r,
            protein_enzgfm_available=enz_ok,
            reaction_center_available=center_ok,
            clipzyme_scores=clip,
            clipzyme_available=clip_ok,
            seed_context_available=torch.zeros_like(clip_ok),
        )
    rows: list[dict[str, float]] = []
    for i, query in enumerate(query_ids):
        order = np.argsort(-scores[i].detach().cpu().numpy(), kind="stable")
        ordered = [candidate_ids[j] for j in order]
        row = {"query_id": query, **query_metrics(ordered, positives[query])}
        rows.append(row)
    pd.DataFrame(rows).to_csv(output_dir / "strict_temporal_r2e_query_metrics.csv", index=False)
    result["r2e"] = summarize(rows)
    result["r2e"]["mean_channel_weights"] = {
        name: float(diag["r2e_channel_weights"][..., index].mean().detach().cpu())
        for index, name in enumerate(("base", "enzgfm", "reaction_center", "clipzyme", "seed_context"))
    }

    # E2R: exact frozen 248-query x 10,131 CLIP-supported reaction support.
    clip_q = pd.read_csv(
        r2e_root / "e2r_strict650_mutual_cold_10131_v2_lexical/official_clipzyme_query_metrics.csv",
        dtype={"query_id": str},
    )
    e_query_ids = clip_q["query_id"].astype(str).tolist()
    cr = pd.read_csv(
        assets.clip_reaction_dir / "entries.csv", dtype=str
    ).fillna("")
    if "clipzyme_supported" in cr:
        cr = cr[cr["clipzyme_supported"].astype(str).str.lower().eq("true")]
    e_candidate_ids = sorted(cr[_id_column(cr, "reaction")].astype(str).tolist())
    strict_pairs = pd.read_csv(
        asset_root / "results/rhea128_to141_external_v2/rhea128_to141_sprot_strict_double_cold_v2/test_pairs.csv",
        dtype=str,
    ).fillna("")
    e_set = set(e_candidate_ids)
    e_pos = (
        strict_pairs[
            strict_pairs["protein_id"].isin(e_query_ids)
            & strict_pairs["reaction_id"].isin(e_set)
        ]
        .groupby("protein_id")["reaction_id"]
        .agg(lambda values: set(values.astype(str)))
        .to_dict()
    )
    e_prows = np.asarray([assets.protein_map[p] for p in e_query_ids], dtype=np.int64)
    e_rrows = np.asarray([assets.reaction_map[r] for r in e_candidate_ids], dtype=np.int64)
    encoded_ep = encode_in_chunks(model, assets, side="protein", rows=e_prows, device=device)
    encoded_er = encode_in_chunks(model, assets, side="reaction", rows=e_rrows, device=device)
    clip_e, clip_e_ok = clip_matrix_r2e(assets, e_rrows, e_prows, device)
    center_e_ok = torch.as_tensor(assets.center_available[e_rrows], device=device)
    enz_e_ok = torch.as_tensor(assets.enz_rows[e_prows] >= 0, device=device)
    with torch.no_grad():
        _, e_scores, ediag = model.score_encoded_cross(
            proteins=encoded_ep,
            reactions=encoded_er,
            protein_enzgfm_available=enz_e_ok,
            reaction_center_available=center_e_ok,
            clipzyme_scores=clip_e,
            clipzyme_available=clip_e_ok,
            seed_context_available=torch.zeros_like(clip_e_ok),
        )
    erows: list[dict[str, float]] = []
    for j, query in enumerate(e_query_ids):
        order = np.argsort(-e_scores[:, j].detach().cpu().numpy(), kind="stable")
        ordered = [e_candidate_ids[i] for i in order]
        erows.append({"query_id": query, **query_metrics(ordered, e_pos[query])})
    pd.DataFrame(erows).to_csv(output_dir / "strict_temporal_e2r_query_metrics.csv", index=False)
    result["e2r"] = summarize(erows)
    result["e2r"]["mean_channel_weights"] = {
        name: float(ediag["e2r_channel_weights"][..., index].mean().detach().cpu())
        for index, name in enumerate(("base", "enzgfm", "reaction_center", "clipzyme", "seed_context"))
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train FIBRE-HCM on the sparse general universe and evaluate the frozen Rhea128->141 strict-temporal protocol."
    )
    parser.add_argument("--asset-root", type=Path, default=DEFAULT_ASSET_ROOT)
    parser.add_argument("--pairs", type=Path, default=DEFAULT_PAIRS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--seed", type=int, default=20260723)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--negative-count", type=int, default=24)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--train-limit", type=int, default=0)
    parser.add_argument("--skip-evaluation", action="store_true")
    args = parser.parse_args()

    root = args.asset_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    pair_path = args.pairs if args.pairs.is_absolute() else root / args.pairs
    device = torch.device(args.device)
    assets = Assets(root)
    pairs = load_training_pairs(assets, pair_path)
    if args.train_limit > 0:
        pairs = pairs.iloc[: args.train_limit].copy()
    print(
        json.dumps(
            {
                "method": "FIBRE-HCM",
                "proteins": assets.n_proteins,
                "reactions": assets.n_reactions,
                "training_pairs": len(pairs),
                "protein_dim": int(assets.protein.shape[1]),
                "reaction_dim": int(assets.reaction.shape[1]),
                "enzgfm_dim": int(assets.enzgfm.shape[1]),
                "reaction_center_dim": int(assets.center.shape[1]),
                "device": str(device),
            },
            indent=2,
        ),
        flush=True,
    )
    model, history = train(
        assets=assets,
        pairs=pairs,
        seed=args.seed,
        epochs=args.epochs,
        batch_size=args.batch_size,
        negative_count=args.negative_count,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        temperature=args.temperature,
        device=device,
    )
    checkpoint = output_dir / "fibre_hcm.pt"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "config": {
                "protein_input_dim": int(assets.protein.shape[1]),
                "reaction_input_dim": int(assets.reaction.shape[1]),
                "enzgfm_input_dim": int(assets.enzgfm.shape[1]),
                "reaction_center_input_dim": int(assets.center.shape[1]),
                "hidden_dim": 512,
                "global_dim": 128,
                "n_experts": 8,
                "expert_dim": 32,
                "specialist_hidden_dim": 256,
                "specialist_dim": 64,
            },
            "seed": args.seed,
        },
        checkpoint,
    )
    pd.DataFrame(history).to_csv(output_dir / "training_history.csv", index=False)

    evaluation = {}
    if not args.skip_evaluation:
        evaluation = evaluate_strict_temporal(
            model=model,
            assets=assets,
            asset_root=root,
            device=device,
            output_dir=output_dir,
        )
    summary = {
        "schema": "fibre-heterogeneous-conditional-modes-general-v1",
        "method": "FIBRE-HCM",
        "status": "general_sparse_training_complete",
        "asset_root": str(root),
        "training": {
            "pairs": len(pairs),
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "negative_count": args.negative_count,
            "seed": args.seed,
            "temperature": args.temperature,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "loss_history": history,
        },
        "architecture": {
            "base": "FIBRE conditional modes: 1 universal + 8 query-conditioned local modes",
            "heterogeneous_channels": [
                "EnzGFM protein view",
                "atom-mapped reaction-center view",
                "CLIPZyme structure score when available",
                "known-positive seed-context score when supplied",
            ],
            "missing_policy": "unavailable channels receive exact zero mixture mass and remaining channels renormalise",
            "seed_context_training": (
                "leave-one-known-positive context within clean2023 training queries "
                "when another positive exists"
            ),
            "strict_temporal_seed_context_available": False,
        },
        "strict_temporal": evaluation,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
