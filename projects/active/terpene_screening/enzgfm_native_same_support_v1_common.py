from __future__ import annotations

import hashlib
import math
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[3]
RESULT_ROOT = ROOT / "results/enzgfm_native_same_support_catalyst_v1"
PREPARED = RESULT_ROOT / "prepared"
PROTEIN_FEATURE_ROOT = ROOT / "data/external/enzgfm_current/general_merged_650m_mean_v1"
PROTEIN_SEQUENCE_TSV = ROOT / "data/catalyst_candidate_universes/general_merged/protein_sequences.tsv"
TRAIN_POS = ROOT / "data/external/reactzyme/enzyme_smi_split/positive_train_val_seq_smi.pt"
TEST_POS = ROOT / "data/external/reactzyme/enzyme_smi_split/positive_test_seq_smi.pt"
DEV_SALT = "enzgfm_native_same_support_v1_dev_20260901"
SELECTION_SALT = "enzgfm_native_same_support_v1_selection_support_20260901"
SEED = 20260901
MORGAN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048, includeChirality=False)


def seed_all(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def normalize_sequence(seq: str) -> str:
    return "".join(str(seq).split()).upper().rstrip("*")


def normalize_reaction_bag(bag: str) -> str:
    # EnzGFM/ReactZyme get_samples replaces wildcard '*' with carbon before reaction embedding.
    parts = [x.strip().replace("*", "C") for x in str(bag).split(".") if x.strip()]
    return ".".join(sorted(parts))


def split_is_dev(seq: str) -> bool:
    key = f"{DEV_SALT}|{normalize_sequence(seq)}".encode()
    return int(hashlib.sha256(key).hexdigest()[:16], 16) % 20 == 0


def support_priority(kind: str, value: str) -> str:
    return hashlib.sha256(f"{SELECTION_SALT}|{kind}|{value}".encode()).hexdigest()


def bag_feature(bag: str) -> tuple[np.ndarray, int, int]:
    rows: list[np.ndarray] = []
    invalid = 0
    for token in normalize_reaction_bag(bag).split("."):
        if not token:
            continue
        mol = Chem.MolFromSmiles(token)
        if mol is None:
            invalid += 1
            continue
        rows.append(np.asarray(MORGAN.GetFingerprintAsNumPy(mol), dtype=np.float32))
    if not rows:
        return np.zeros(4096, dtype=np.float32), 0, invalid
    x = np.stack(rows)
    return np.concatenate([x.mean(0), x.max(0)]).astype(np.float32), len(rows), invalid


class DualTowerNative(nn.Module):
    def __init__(self, reaction_dim: int = 4096, protein_dim: int = 2048, hidden: int = 512, embed: int = 256, dropout: float = 0.1):
        super().__init__()
        self.reaction = nn.Sequential(
            nn.LayerNorm(reaction_dim), nn.Linear(reaction_dim, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, embed)
        )
        self.protein = nn.Sequential(
            nn.LayerNorm(protein_dim), nn.Linear(protein_dim, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, embed)
        )

    def encode_reactions(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.reaction(x), p=2, dim=-1)

    def encode_proteins(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.protein(x), p=2, dim=-1)

    def forward(self, r: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        return (self.encode_reactions(r) * self.encode_proteins(p)).sum(-1)


class CrossAttention(nn.Module):
    def __init__(self, query_input_dim: int, key_input_dim: int, output_dim: int):
        super().__init__()
        self.q = nn.Linear(query_input_dim, output_dim)
        self.k = nn.Linear(key_input_dim, output_dim)
        self.v = nn.Linear(key_input_dim, output_dim)
        self.scale = math.sqrt(output_dim)

    def forward(self, query_input: torch.Tensor, key_input: torch.Tensor, value_input: torch.Tensor) -> torch.Tensor:
        q, k, v = self.q(query_input), self.k(key_input), self.v(value_input)
        w = torch.softmax(torch.matmul(q, k.transpose(1, 2)) / self.scale, dim=-1)
        return torch.matmul(w, v)


class AuthorPairwiseNative(nn.Module):
    """Author EnzGFM/ReactZyme retrieval-head topology with only input dimensions adapted.

    This is an independent author-architecture reproduction candidate, not an author checkpoint.
    """

    def __init__(self, reaction_dim: int = 4096, protein_dim: int = 2048, hidden: int = 128, dropout: float = 0.0):
        super().__init__()
        self.lin_mol_embed = nn.Sequential(
            nn.Linear(reaction_dim, 256, bias=False), nn.Dropout(dropout), nn.BatchNorm1d(256), nn.SiLU(),
            nn.Linear(256, 256, bias=False), nn.Dropout(dropout), nn.BatchNorm1d(256), nn.SiLU(),
            nn.Linear(256, hidden, bias=False),
        )
        self.lin_seq_embed = nn.Sequential(
            nn.Linear(protein_dim, 512, bias=False), nn.Dropout(dropout), nn.BatchNorm1d(512), nn.SiLU(),
            nn.Linear(512, 256, bias=False), nn.Dropout(dropout), nn.BatchNorm1d(256), nn.SiLU(),
            nn.Linear(256, hidden, bias=False),
        )
        self.cross_attn_seq = CrossAttention(hidden, hidden, hidden)
        self.cross_attn_mol = CrossAttention(hidden, hidden, hidden)
        self.transformer = nn.Transformer(
            d_model=hidden, nhead=8, num_encoder_layers=4, num_decoder_layers=4,
            dim_feedforward=hidden, batch_first=True, dropout=dropout,
        )
        self.lin_out = nn.Sequential(
            nn.Linear(hidden, hidden, bias=False), nn.Dropout(dropout), nn.LayerNorm(hidden), nn.ReLU(),
            nn.Linear(hidden, 64, bias=False), nn.Dropout(dropout), nn.LayerNorm(64), nn.ReLU(),
            nn.Linear(64, 16, bias=False), nn.Linear(16, 1, bias=False),
        )

    def forward(self, reaction: torch.Tensor, protein: torch.Tensor) -> torch.Tensor:
        b = reaction.shape[0]
        mol = self.lin_mol_embed(reaction).reshape(b, 1, -1)
        seq = self.lin_seq_embed(protein).reshape(b, 1, -1)
        mol2 = self.cross_attn_mol(mol, seq, seq)
        seq2 = self.cross_attn_seq(seq, mol, mol)
        x = torch.cat([mol2, seq2], dim=1)
        x = self.transformer(x, x)
        return self.lin_out(x.sum(1)).reshape(-1)


def load_protein_memmap() -> np.ndarray:
    return np.load(PROTEIN_FEATURE_ROOT / "embeddings.npy", mmap_mode="r")


def build_label_matrix(pairs: pd.DataFrame, protein_entries: pd.DataFrame, reaction_entries: pd.DataFrame) -> np.ndarray:
    pmap = {int(r.protein_row): i for i, r in protein_entries.reset_index(drop=True).iterrows()}
    rmap = {int(r.reaction_idx): i for i, r in reaction_entries.reset_index(drop=True).iterrows()}
    labels = np.zeros((len(protein_entries), len(reaction_entries)), dtype=bool)
    for x in pairs.itertuples(index=False):
        pi, ri = pmap.get(int(x.protein_row)), rmap.get(int(x.reaction_idx))
        if pi is not None and ri is not None:
            labels[pi, ri] = True
    return labels


def query_metrics(scores: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    if scores.shape != labels.shape or scores.ndim != 2:
        raise ValueError("scores and labels must be same 2D shape")
    valid = labels.sum(1) > 0
    scores, labels = scores[valid], labels[valid]
    if len(scores) == 0:
        raise ValueError("no queries with positives")
    rows: list[dict[str, float]] = []
    for s, y in zip(scores, labels):
        order = np.argsort(-s, kind="stable")
        ranks = np.empty(len(s), dtype=np.int64)
        ranks[order] = np.arange(1, len(s) + 1)
        pr = np.sort(ranks[y])
        rel_at_order = y[order].astype(np.float64)
        csum = np.cumsum(rel_at_order)
        ap = float(np.sum((csum / np.arange(1, len(s) + 1)) * rel_at_order) / len(pr))
        dcg10 = float(sum(1.0 / np.log2(r + 1) for r in pr if r <= 10))
        ideal10 = float(sum(1.0 / np.log2(i + 2) for i in range(min(len(pr), 10))))
        d = {
            "map": ap,
            "best_mrr": float(1.0 / pr[0]),
            "author_avg_positive_rr": float(np.mean(1.0 / pr)),
            "mean_positive_rank": float(np.mean(pr)),
            "best_positive_rank": float(pr[0]),
            "ndcg_at_10": dcg10 / ideal10 if ideal10 else 0.0,
        }
        for k in (1, 5, 10, 20, 50):
            d[f"hit_at_{k}"] = float(np.any(pr <= k))
            d[f"author_acc_n_at_{k}"] = float(np.sum(pr <= k) / k)
        rows.append(d)
    out: dict[str, float] = {"n_queries": float(len(rows))}
    keys = [k for k in rows[0] if k != "best_positive_rank"]
    for k in keys:
        out[k] = float(np.mean([r[k] for r in rows]))
    out["median_best_positive_rank"] = float(np.median([r["best_positive_rank"] for r in rows]))
    return out


def balanced_selection_score(e2r: dict[str, float], r2e: dict[str, float]) -> float:
    a, b = float(e2r["map"]), float(r2e["map"])
    return 0.0 if a <= 0 or b <= 0 else float(2 * a * b / (a + b))
