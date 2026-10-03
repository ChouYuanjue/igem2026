from __future__ import annotations

import json
import pickle
import warnings
from functools import lru_cache
from pathlib import Path

import lmdb
import numpy as np
import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence

from .fibre import ExpertEvidence


ROOT = Path(__file__).resolve().parents[4]
GENERAL = ROOT / "data/catalyst_candidate_universes/general_merged"


def _entries(path: Path, names: tuple[str, ...]) -> tuple[list[str], dict[str, int]]:
    frame = pd.read_csv(path, dtype=str).fillna("")
    if "row" in frame.columns:
        frame["row"] = pd.to_numeric(frame["row"]).astype(int)
        frame = frame.sort_values("row")
    col = next(name for name in names if name in frame.columns)
    values = frame[col].astype(str).tolist()
    return values, {value: i for i, value in enumerate(values)}


def _normalise_rows(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    norm = np.linalg.norm(values, axis=1, keepdims=True)
    return values / np.maximum(norm, 1e-8)


class ReactionTokenStore:
    def __init__(
        self,
        root: Path = GENERAL / "reaction_features/eram_unimol_v1",
        *,
        max_tokens_per_side: int = 384,
    ) -> None:
        self.max_tokens_per_side = int(max_tokens_per_side)
        if self.max_tokens_per_side < 32:
            raise ValueError("max_tokens_per_side is too small")
        frame = pd.read_csv(root / "reactions.csv", dtype=str).fillna("")
        self.records = {
            str(row.reaction_id): (
                tuple(json.loads(row.reactant_ids)),
                tuple(json.loads(row.product_ids)),
                str(row.eram_available).lower() == "true",
            )
            for row in frame.itertuples(index=False)
        }
        self.env = lmdb.open(
            str(root / "molecules.lmdb"),
            readonly=True,
            lock=False,
            readahead=False,
            max_readers=512,
        )

    def available(self, reaction_id: str) -> bool:
        rec = self.records.get(str(reaction_id))
        return bool(rec and rec[2])

    @lru_cache(maxsize=12000)
    def _compound(self, compound_id: int) -> torch.Tensor:
        with self.env.begin() as txn:
            raw = txn.get(str(int(compound_id)).encode())
        if raw is None:
            raise KeyError(f"missing UniMol compound {compound_id}")
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning, module="torch.storage")
            value = pickle.loads(raw)
        return torch.as_tensor(value, dtype=torch.float32)

    def _pack_side(self, compound_ids: tuple[int, ...]) -> torch.Tensor:
        parts = [self._compound(i) for i in compound_ids]
        total = sum(len(part) for part in parts)
        if total <= self.max_tokens_per_side:
            return torch.cat(parts, dim=0)

        # Broad Rhea reactions occasionally contain very large multi-component
        # states. Preserve every molecular CLS token, then distribute the
        # remaining atom-token budget proportionally across components.
        cls_budget = len(parts)
        atom_budget = max(0, self.max_tokens_per_side - cls_budget)
        atom_counts = [max(0, len(part) - 1) for part in parts]
        total_atoms = sum(atom_counts)
        allocation = [
            min(count, int(atom_budget * count / max(total_atoms, 1)))
            for count in atom_counts
        ]
        remainder = atom_budget - sum(allocation)
        while remainder > 0:
            changed = False
            for i, count in enumerate(atom_counts):
                if allocation[i] < count:
                    allocation[i] += 1
                    remainder -= 1
                    changed = True
                    if remainder == 0:
                        break
            if not changed:
                break

        packed: list[torch.Tensor] = []
        for part, keep in zip(parts, allocation, strict=True):
            if keep <= 0:
                packed.append(part[:1])
                continue
            if keep >= len(part) - 1:
                packed.append(part)
                continue
            idx = torch.linspace(1, len(part) - 1, steps=keep).round().long()
            packed.append(torch.cat([part[:1], part[idx]], dim=0))
        return torch.cat(packed, dim=0)

    @lru_cache(maxsize=4096)
    def reaction(self, reaction_id: str) -> tuple[torch.Tensor, torch.Tensor]:
        reactants, products, available = self.records[str(reaction_id)]
        if not available:
            raise KeyError(f"ERAM tokens unavailable for {reaction_id}")
        return self._pack_side(reactants), self._pack_side(products)


class FibreAssetStore:
    """Read-only access to the existing broad and specialist BRIDGE assets."""

    def __init__(self) -> None:
        self.protein_ids, self.protein_index = _entries(
            GENERAL / "proteins/entries.csv", ("Entry", "protein_id")
        )
        self.protein = np.load(GENERAL / "proteins/embeddings.npy", mmap_mode="r")

        self.reaction_ids, self.reaction_index = _entries(
            GENERAL / "reaction_features/drfp_categorical_v1/entries.csv",
            ("reaction_id", "rhea_id"),
        )
        self.reaction_features = np.load(
            GENERAL / "reaction_features/drfp_categorical_v1/reaction_feature_matrix.npy",
            mmap_mode="r",
        )

        _, center_index = _entries(
            GENERAL / "reaction_features/drfp_categorical_rdkitplus_center_v1/entries.csv",
            ("reaction_id", "rhea_id"),
        )
        center_full = np.load(
            GENERAL
            / "reaction_features/drfp_categorical_rdkitplus_center_v1/reaction_feature_matrix.npy",
            mmap_mode="r",
        )
        self.center = center_full[:, 3139:]
        self.center_index = center_index

        enzgfm_dir = ROOT / "data/external/enzgfm_current/general_merged_650m_mean_v1"
        _, self.enzgfm_index = _entries(
            enzgfm_dir / "entries.csv", ("Entry", "protein_id")
        )
        self.enzgfm = np.load(enzgfm_dir / "embeddings.npy", mmap_mode="r")

        clip_p_dir = ROOT / "results/bime_rank_unified_v1/clipzyme_r2e_candidate_asset_v1"
        _, self.clip_p_index = _entries(
            clip_p_dir / "entries.csv", ("protein_id", "Entry")
        )
        self.clip_p = np.load(clip_p_dir / "embeddings.npy", mmap_mode="r")

        clip_r_dir = (
            ROOT
            / "results/clipzyme_native_extension_v1/full_hplus_candidate_reactions/clipzyme_embeddings_gpu_v1"
        )
        clip_r_frame = pd.read_csv(clip_r_dir / "entries.csv", dtype=str).fillna("")
        self.clip_r_index = {
            str(row.reaction_id): int(row.row)
            for row in clip_r_frame.itertuples(index=False)
            if str(row.clipzyme_supported).lower() == "true"
        }
        self.clip_r = np.load(clip_r_dir / "embeddings.npy", mmap_mode="r")

        tps_dir = ROOT / "results/fibre_application/tps_adapted_coordinate"
        self.tps_p, self.tps_p_index = self._coordinate(
            tps_dir / "protein_tps_adapted.csv", "protein_id"
        )
        self.tps_r, self.tps_r_index = self._coordinate(
            tps_dir / "reaction_tps_adapted.csv", "reaction_id"
        )

        self.tokens = ReactionTokenStore()

    @staticmethod
    def _coordinate(path: Path, id_col: str) -> tuple[np.ndarray, dict[str, int]]:
        frame = pd.read_csv(path, dtype=str).fillna("")
        ids = frame[id_col].astype(str).tolist()
        matrix = frame.drop(columns=[id_col]).astype(np.float32).to_numpy(copy=True)
        return matrix, {value: i for i, value in enumerate(ids)}

    @staticmethod
    def _gather(
        matrix: np.ndarray,
        index: dict[str, int],
        ids: list[str],
    ) -> tuple[np.ndarray, np.ndarray]:
        width = int(matrix.shape[1])
        out = np.zeros((len(ids), width), dtype=np.float32)
        available = np.zeros(len(ids), dtype=bool)
        rows = [index.get(str(value), -1) for value in ids]
        valid = np.asarray(rows, dtype=np.int64) >= 0
        if valid.any():
            rr = np.asarray(rows, dtype=np.int64)[valid]
            out[valid] = np.asarray(matrix[rr], dtype=np.float32)
            available[valid] = np.linalg.norm(out[valid], axis=1) > 1e-8
        return out, available

    def supported_pairs(self, frame: pd.DataFrame) -> pd.DataFrame:
        mask = (
            frame["protein_id"].astype(str).isin(self.protein_index)
            & frame["reaction_id"].astype(str).isin(self.reaction_index)
            & frame["reaction_id"].astype(str).map(self.tokens.available)
        )
        return frame.loc[mask, ["protein_id", "reaction_id"]].drop_duplicates().reset_index(drop=True)

    def batch(
        self,
        protein_ids: list[str],
        reaction_ids: list[str],
        device: torch.device,
        *,
        expert_names: set[str] | None = None,
    ) -> dict[str, object]:
        if len(protein_ids) != len(reaction_ids):
            raise ValueError("protein/reaction pair lengths differ")
        p_rows = np.asarray([self.protein_index[x] for x in protein_ids], dtype=np.int64)
        base = torch.as_tensor(
            np.asarray(self.protein[p_rows], dtype=np.float32).copy(),
            device=device,
        )

        reactants: list[torch.Tensor] = []
        products: list[torch.Tensor] = []
        for rid in reaction_ids:
            r, p = self.tokens.reaction(rid)
            reactants.append(r)
            products.append(p)
        r_len = [len(x) for x in reactants]
        p_len = [len(x) for x in products]
        reactant = pad_sequence(reactants, batch_first=True).to(device)
        product = pad_sequence(products, batch_first=True).to(device)
        r_mask = torch.ones((len(reactants), reactant.shape[1]), dtype=torch.bool, device=device)
        p_mask = torch.ones((len(products), product.shape[1]), dtype=torch.bool, device=device)
        for i, n in enumerate(r_len):
            r_mask[i, :n] = False
        for i, n in enumerate(p_len):
            p_mask[i, :n] = False

        wanted = expert_names or {"enzgfm", "reaction_center", "clipzyme", "tps"}
        evidence: dict[str, ExpertEvidence] = {}
        if "enzgfm" in wanted:
            enzgfm, enzgfm_ok = self._gather(self.enzgfm, self.enzgfm_index, protein_ids)
            evidence["enzgfm"] = ExpertEvidence(
                torch.as_tensor(enzgfm, device=device),
                torch.as_tensor(enzgfm_ok, device=device),
            )
        if "reaction_center" in wanted:
            center, center_ok = self._gather(self.center, self.center_index, reaction_ids)
            evidence["reaction_center"] = ExpertEvidence(
                torch.as_tensor(center, device=device),
                torch.as_tensor(center_ok, device=device),
            )
        if "clipzyme" in wanted:
            clip_p, clip_p_ok = self._gather(self.clip_p, self.clip_p_index, protein_ids)
            clip_r, clip_r_ok = self._gather(self.clip_r, self.clip_r_index, reaction_ids)
            clip_ok = clip_p_ok & clip_r_ok
            clip_tokens = np.stack([_normalise_rows(clip_p), _normalise_rows(clip_r)], axis=1)
            clip_tokens[~clip_ok] = 0.0
            evidence["clipzyme"] = ExpertEvidence(
                torch.as_tensor(clip_tokens, device=device),
                torch.as_tensor(clip_ok, device=device),
            )
        if "tps" in wanted:
            tps_p, tps_p_ok = self._gather(self.tps_p, self.tps_p_index, protein_ids)
            tps_r, tps_r_ok = self._gather(self.tps_r, self.tps_r_index, reaction_ids)
            tps_ok = tps_p_ok & tps_r_ok
            tps_tokens = np.stack([_normalise_rows(tps_p), _normalise_rows(tps_r)], axis=1)
            tps_tokens[~tps_ok] = 0.0
            evidence["tps"] = ExpertEvidence(
                torch.as_tensor(tps_tokens, device=device),
                torch.as_tensor(tps_ok, device=device),
            )
        return {
            "protein": base,
            "reactant": reactant,
            "product": product,
            "reactant_padding_mask": r_mask,
            "product_padding_mask": p_mask,
            "evidence": evidence,
        }
