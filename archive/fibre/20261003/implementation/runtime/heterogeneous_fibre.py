from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch

from projects.active.fibre.kernel.heterogeneous_modes import (
    HeterogeneousConditionalFibre,
    HeterogeneousFibreConfig,
)
from reproducibility.bime_rank.support.evaluate_multi_expert_protocol_comparison import (
    MultiExpertConfig,
)


def _entry_map(path: Path, candidates: tuple[str, ...]) -> tuple[list[str], dict[str, int]]:
    frame = pd.read_csv(path, dtype=str).fillna("")
    column = next((name for name in candidates if name in frame.columns), None)
    if column is None:
        raise ValueError(f"cannot infer identifier column from {path}")
    if "row" in frame.columns:
        frame["row"] = pd.to_numeric(frame["row"]).astype(int)
        frame = frame.sort_values("row")
    values = frame[column].astype(str).tolist()
    return values, {value: index for index, value in enumerate(values)}


def load_fibre_hcm_checkpoint(
    checkpoint: Path,
    *,
    device: torch.device | str = "cpu",
) -> HeterogeneousConditionalFibre:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    config = payload["config"]
    base = MultiExpertConfig(
        protein_input_dim=int(config["protein_input_dim"]),
        reaction_input_dim=int(config["reaction_input_dim"]),
        hidden_dim=int(config["hidden_dim"]),
        global_dim=int(config["global_dim"]),
        n_experts=int(config["n_experts"]),
        expert_dim=int(config["expert_dim"]),
        dropout=float(config.get("dropout", 0.1)),
        gate_temperature=float(config.get("gate_temperature", 1.0)),
        expert_mix_init=float(config.get("expert_mix_init", 0.5)),
    )
    model = HeterogeneousConditionalFibre(
        HeterogeneousFibreConfig(
            base=base,
            enzgfm_input_dim=int(config["enzgfm_input_dim"]),
            reaction_center_input_dim=int(config["reaction_center_input_dim"]),
            specialist_hidden_dim=int(config["specialist_hidden_dim"]),
            specialist_dim=int(config["specialist_dim"]),
        )
    )
    model.load_state_dict(payload["state_dict"])
    model.to(torch.device(device))
    model.eval()
    return model


@dataclass(frozen=True)
class FibreHCMAssetPaths:
    protein_dir: Path
    reaction_dir: Path
    reaction_center_dir: Path
    enzgfm_dir: Path
    clip_protein_dir: Path
    clip_reaction_dir: Path

    @classmethod
    def from_root(cls, root: Path) -> "FibreHCMAssetPaths":
        root = root.resolve()
        general = root / "data/catalyst_candidate_universes/general_merged"
        return cls(
            protein_dir=general / "proteins",
            reaction_dir=general / "reaction_features/drfp_categorical_rdkitplus_v1",
            reaction_center_dir=general
            / "reaction_features/drfp_categorical_rdkitplus_center_v1",
            enzgfm_dir=root / "data/external/enzgfm_current/general_merged_650m_mean_v1",
            clip_protein_dir=root
            / "results/bime_rank_unified_v1/clipzyme_r2e_candidate_asset_v1",
            clip_reaction_dir=root
            / "results/clipzyme_native_extension_v1/full_hplus_candidate_reactions/clipzyme_embeddings_gpu_v1",
        )


class FibreHCMRuntime:
    """Read-only registered-universe inference for a frozen FIBRE-HCM checkpoint."""

    def __init__(
        self,
        *,
        checkpoint: Path,
        asset_root: Path,
        device: torch.device | str = "cuda",
    ) -> None:
        self.device = torch.device(device)
        self.model = load_fibre_hcm_checkpoint(checkpoint, device=self.device)
        paths = FibreHCMAssetPaths.from_root(asset_root)
        self.paths = paths

        self.protein_ids, self.protein_map = _entry_map(
            paths.protein_dir / "entries.csv",
            ("Entry", "protein_id"),
        )
        self.reaction_ids, self.reaction_map = _entry_map(
            paths.reaction_dir / "entries.csv",
            ("reaction_id", "rhea_id"),
        )
        self.protein = np.load(paths.protein_dir / "embeddings.npy", mmap_mode="r")
        self.reaction = np.load(
            paths.reaction_dir / "reaction_feature_matrix.npy",
            mmap_mode="r",
        )
        center_full = np.load(
            paths.reaction_center_dir / "reaction_feature_matrix.npy",
            mmap_mode="r",
        )
        self.center = center_full[:, self.reaction.shape[1] :]
        self.center_available = np.linalg.norm(
            np.asarray(self.center, dtype=np.float32), axis=1
        ) > 0

        _, enz_map = _entry_map(paths.enzgfm_dir / "entries.csv", ("Entry", "protein_id"))
        self.enzgfm = np.load(paths.enzgfm_dir / "embeddings.npy", mmap_mode="r")
        self.enz_rows = np.asarray(
            [enz_map.get(value, -1) for value in self.protein_ids], dtype=np.int64
        )

        _, cp_map = _entry_map(
            paths.clip_protein_dir / "entries.csv",
            ("Entry", "protein_id"),
        )
        self.clip_protein = np.load(
            paths.clip_protein_dir / "embeddings.npy",
            mmap_mode="r",
        )
        self.clip_protein_rows = np.asarray(
            [cp_map.get(value, -1) for value in self.protein_ids], dtype=np.int64
        )

        cr_frame = pd.read_csv(paths.clip_reaction_dir / "entries.csv", dtype=str).fillna("")
        reaction_column = next(
            name for name in ("reaction_id", "rhea_id") if name in cr_frame.columns
        )
        if "clipzyme_supported" in cr_frame.columns:
            cr_frame = cr_frame[
                cr_frame["clipzyme_supported"].astype(str).str.lower().eq("true")
            ]
        if "row" in cr_frame.columns:
            cr_rows = pd.to_numeric(cr_frame["row"]).astype(int)
        else:
            cr_rows = pd.Series(np.arange(len(cr_frame), dtype=int), index=cr_frame.index)
        cr_map = dict(zip(cr_frame[reaction_column].astype(str), cr_rows.astype(int)))
        self.clip_reaction = np.load(
            paths.clip_reaction_dir / "embeddings.npy",
            mmap_mode="r",
        )
        self.clip_reaction_rows = np.asarray(
            [cr_map.get(value, -1) for value in self.reaction_ids], dtype=np.int64
        )

    def _protein_side(self, rows: np.ndarray) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        base = torch.as_tensor(
            np.asarray(self.protein[rows], dtype=np.float32),
            device=self.device,
        )
        enz_rows = self.enz_rows[rows]
        enz = np.zeros((len(rows), self.enzgfm.shape[1]), dtype=np.float32)
        available = enz_rows >= 0
        if available.any():
            enz[available] = np.asarray(
                self.enzgfm[enz_rows[available]],
                dtype=np.float32,
            )
        with torch.no_grad():
            encoded = self.model.encode_protein_side(
                base,
                torch.as_tensor(enz, device=self.device),
            )
        return encoded, torch.as_tensor(available, device=self.device)

    def _reaction_side(
        self,
        rows: np.ndarray,
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        base = torch.as_tensor(
            np.asarray(self.reaction[rows], dtype=np.float32),
            device=self.device,
        )
        center = torch.as_tensor(
            np.asarray(self.center[rows], dtype=np.float32),
            device=self.device,
        )
        with torch.no_grad():
            encoded = self.model.encode_reaction_side(base, center)
        return encoded, torch.as_tensor(
            self.center_available[rows],
            device=self.device,
        )

    def _clip_scores(
        self,
        reaction_rows: np.ndarray,
        protein_rows: np.ndarray,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        rr = self.clip_reaction_rows[reaction_rows]
        pr = self.clip_protein_rows[protein_rows]
        r_ok = rr >= 0
        p_ok = pr >= 0
        r = np.zeros((len(rr), self.clip_reaction.shape[1]), dtype=np.float32)
        p = np.zeros((len(pr), self.clip_protein.shape[1]), dtype=np.float32)
        if r_ok.any():
            r[r_ok] = np.asarray(self.clip_reaction[rr[r_ok]], dtype=np.float32)
        if p_ok.any():
            p[p_ok] = np.asarray(self.clip_protein[pr[p_ok]], dtype=np.float32)
        rn = np.linalg.norm(r, axis=1, keepdims=True)
        pn = np.linalg.norm(p, axis=1, keepdims=True)
        rn[rn == 0] = 1.0
        pn[pn == 0] = 1.0
        score = (r / rn) @ (p / pn).T
        available = r_ok[:, None] & p_ok[None, :]
        return (
            torch.as_tensor(score, device=self.device),
            torch.as_tensor(available, device=self.device),
        )

    @staticmethod
    def _max_seed_cosine(
        matrix: np.ndarray,
        candidate_rows: np.ndarray,
        seed_rows: np.ndarray,
    ) -> np.ndarray:
        if len(seed_rows) == 0:
            return np.zeros(len(candidate_rows), dtype=np.float32)
        candidate = np.asarray(matrix[candidate_rows], dtype=np.float32)
        seed = np.asarray(matrix[seed_rows], dtype=np.float32)
        candidate_norm = np.linalg.norm(candidate, axis=1, keepdims=True)
        seed_norm = np.linalg.norm(seed, axis=1, keepdims=True)
        candidate_norm[candidate_norm == 0] = 1.0
        seed_norm[seed_norm == 0] = 1.0
        return ((candidate / candidate_norm) @ (seed / seed_norm).T).max(
            axis=1
        ).astype(np.float32)

    def rank_reaction_to_enzyme(
        self,
        reaction_id: str,
        *,
        candidate_ids: Iterable[str] | None = None,
        known_enzyme_ids: Iterable[str] | None = None,
        top_k: int = 50,
        chunk_size: int = 8192,
    ) -> pd.DataFrame:
        if reaction_id not in self.reaction_map:
            raise KeyError(f"unknown registered reaction: {reaction_id}")
        candidates = list(candidate_ids) if candidate_ids is not None else self.protein_ids
        missing = [value for value in candidates if value not in self.protein_map]
        if missing:
            raise KeyError(f"unknown registered proteins: {missing[:5]}")
        seed_ids = [
            value
            for value in map(str, known_enzyme_ids or ())
            if value in self.protein_map
        ]
        seed_rows = np.asarray(
            [self.protein_map[value] for value in seed_ids],
            dtype=np.int64,
        )
        reaction_rows = np.asarray([self.reaction_map[reaction_id]], dtype=np.int64)
        reactions, center_ok = self._reaction_side(reaction_rows)
        scores: list[np.ndarray] = []
        ids: list[str] = []
        for start in range(0, len(candidates), chunk_size):
            local_ids = candidates[start : start + chunk_size]
            protein_rows = np.asarray(
                [self.protein_map[value] for value in local_ids],
                dtype=np.int64,
            )
            proteins, enz_ok = self._protein_side(protein_rows)
            clip, clip_ok = self._clip_scores(reaction_rows, protein_rows)
            if len(seed_rows):
                seed_scores_np = self._max_seed_cosine(
                    self.protein,
                    protein_rows,
                    seed_rows,
                )
                seed_scores = torch.as_tensor(
                    seed_scores_np[None, :],
                    device=self.device,
                )
                seed_available = torch.ones_like(clip_ok)
            else:
                seed_scores = None
                seed_available = torch.zeros_like(clip_ok)
            with torch.no_grad():
                r2e, _, _ = self.model.score_encoded_cross(
                    proteins=proteins,
                    reactions=reactions,
                    protein_enzgfm_available=enz_ok,
                    reaction_center_available=center_ok,
                    clipzyme_scores=clip,
                    clipzyme_available=clip_ok,
                    seed_context_scores=seed_scores,
                    seed_context_available=seed_available,
                )
            scores.append(r2e[0].detach().cpu().numpy())
            ids.extend(local_ids)
        values = np.concatenate(scores)
        if seed_ids:
            masked = set(seed_ids)
            values = values.copy()
            for index, value in enumerate(ids):
                if value in masked:
                    values[index] = -np.inf
        order = np.argsort(-values, kind="stable")[:top_k]
        return pd.DataFrame(
            {
                "rank": np.arange(1, len(order) + 1),
                "protein_id": np.asarray(ids, dtype=object)[order],
                "score": values[order],
            }
        )

    def rank_enzyme_to_reaction(
        self,
        protein_id: str,
        *,
        candidate_ids: Iterable[str] | None = None,
        known_reaction_ids: Iterable[str] | None = None,
        top_k: int = 50,
        chunk_size: int = 4096,
    ) -> pd.DataFrame:
        if protein_id not in self.protein_map:
            raise KeyError(f"unknown registered protein: {protein_id}")
        candidates = list(candidate_ids) if candidate_ids is not None else self.reaction_ids
        missing = [value for value in candidates if value not in self.reaction_map]
        if missing:
            raise KeyError(f"unknown registered reactions: {missing[:5]}")
        seed_ids = [
            value
            for value in map(str, known_reaction_ids or ())
            if value in self.reaction_map
        ]
        seed_rows = np.asarray(
            [self.reaction_map[value] for value in seed_ids],
            dtype=np.int64,
        )
        protein_rows = np.asarray([self.protein_map[protein_id]], dtype=np.int64)
        proteins, enz_ok = self._protein_side(protein_rows)
        scores: list[np.ndarray] = []
        ids: list[str] = []
        for start in range(0, len(candidates), chunk_size):
            local_ids = candidates[start : start + chunk_size]
            reaction_rows = np.asarray(
                [self.reaction_map[value] for value in local_ids],
                dtype=np.int64,
            )
            reactions, center_ok = self._reaction_side(reaction_rows)
            clip, clip_ok = self._clip_scores(reaction_rows, protein_rows)
            if len(seed_rows):
                seed_scores_np = self._max_seed_cosine(
                    self.reaction[:, :2048],
                    reaction_rows,
                    seed_rows,
                )
                seed_scores = torch.as_tensor(
                    seed_scores_np[:, None],
                    device=self.device,
                )
                seed_available = torch.ones_like(clip_ok)
            else:
                seed_scores = None
                seed_available = torch.zeros_like(clip_ok)
            with torch.no_grad():
                _, e2r, _ = self.model.score_encoded_cross(
                    proteins=proteins,
                    reactions=reactions,
                    protein_enzgfm_available=enz_ok,
                    reaction_center_available=center_ok,
                    clipzyme_scores=clip,
                    clipzyme_available=clip_ok,
                    seed_context_scores=seed_scores,
                    seed_context_available=seed_available,
                )
            scores.append(e2r[:, 0].detach().cpu().numpy())
            ids.extend(local_ids)
        values = np.concatenate(scores)
        if seed_ids:
            masked = set(seed_ids)
            values = values.copy()
            for index, value in enumerate(ids):
                if value in masked:
                    values[index] = -np.inf
        order = np.argsort(-values, kind="stable")[:top_k]
        return pd.DataFrame(
            {
                "rank": np.arange(1, len(order) + 1),
                "reaction_id": np.asarray(ids, dtype=object)[order],
                "score": values[order],
            }
        )
