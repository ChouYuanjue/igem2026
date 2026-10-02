from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from projects.active.fibre.model.assets import ROOT
from projects.active.fibre.model.index import FibreCandidateIndex
from projects.active.fibre.runtime.cli import load_models
from projects.active.fibre.runtime.scientific_evidence import Direction, EvidenceOutput


class ReactionCenterPairEvidence:
    """Frozen clean2023 reaction-centre ranker exposed as mechanism evidence.

    The selected bounded residual model is kept intact.  We only cache its
    protein/reaction latent vectors so the expert can be used as a pair scorer.
    """

    name = "reaction_center_mechanism"
    kind = "mechanistic"
    role = "rerank"
    directions = ("r2e",)

    def __init__(
        self,
        *,
        device: str | torch.device = "cuda",
        cache_dir: Path | None = None,
    ) -> None:
        self.device = torch.device(device)
        self.model_root = ROOT / "results/catalyst_clean_mainline_v1/r2e_center_bounded_cap0p1"
        summary = json.loads((self.model_root / "summary.json").read_text())
        if summary.get("model_type") != "rdkitplus_bounded_identity_hidden_residual":
            raise ValueError("unexpected reaction-centre expert model type")
        if bool(summary.get("target_benchmark_labels_read")):
            raise ValueError("reaction-centre expert is target-label contaminated")
        if abs(float(summary.get("max_residual_ratio")) - 0.1) > 1e-12:
            raise ValueError("reaction-centre expert does not match frozen selected cap")

        base = json.loads((Path(summary["base_dir"]) / "summary.json").read_text())
        self.protein_feature_dir = Path(base["protein_feature_dir"])
        self.reaction_feature_dir = Path(summary["reaction_feature_dir"])
        self.cache_dir = (
            ROOT / "results/fibre_expert_assets_v1/reaction_center"
            if cache_dir is None
            else Path(cache_dir)
        )
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._prepare_cache()
        self.p_ids, self.p_index = self._load_entries(
            self.protein_feature_dir / "entries.csv", ("Entry", "protein_id")
        )
        self.r_ids, self.r_index = self._load_entries(
            self.reaction_feature_dir / "entries.csv", ("reaction_id", "rhea_id")
        )
        self.p = torch.from_numpy(
            np.asarray(np.load(self.cache_dir / "protein_latents.npy", mmap_mode="r")).copy()
        ).to(self.device)
        self.r = torch.from_numpy(
            np.asarray(np.load(self.cache_dir / "reaction_latents.npy", mmap_mode="r")).copy()
        ).to(self.device)

    @staticmethod
    def _load_entries(path: Path, columns: tuple[str, ...]) -> tuple[list[str], dict[str, int]]:
        frame = pd.read_csv(path, dtype=str).fillna("")
        if "row" in frame.columns:
            frame["row"] = pd.to_numeric(frame["row"]).astype(int)
            frame = frame.sort_values("row", kind="stable")
        column = next(name for name in columns if name in frame.columns)
        ids = frame[column].astype(str).tolist()
        return ids, {value: i for i, value in enumerate(ids)}

    def _prepare_cache(self) -> None:
        p_out = self.cache_dir / "protein_latents.npy"
        r_out = self.cache_dir / "reaction_latents.npy"
        manifest = self.cache_dir / "manifest.json"
        if p_out.exists() and r_out.exists() and manifest.exists():
            return

        models = load_models(self.model_root / "models", "production", self.device)
        if len(models) != 1:
            raise ValueError("reaction-centre expert expects one frozen production model")
        model = models[0].eval()
        p_raw = np.load(self.protein_feature_dir / "embeddings.npy", mmap_mode="r")
        r_raw = np.load(self.reaction_feature_dir / "reaction_feature_matrix.npy", mmap_mode="r")

        def encode(matrix: np.ndarray, fn, batch: int) -> np.ndarray:
            parts: list[np.ndarray] = []
            with torch.no_grad():
                for start in range(0, len(matrix), batch):
                    x = torch.as_tensor(
                        np.asarray(matrix[start:start + batch], dtype=np.float32).copy(),
                        device=self.device,
                    )
                    parts.append(fn(x).float().cpu().numpy())
            return np.concatenate(parts, axis=0).astype(np.float32)

        p = encode(p_raw, model.encode_proteins, 4096)
        r = encode(r_raw, model.encode_reactions, 512)
        np.save(p_out, p)
        np.save(r_out, r)
        manifest.write_text(
            json.dumps(
                {
                    "schema": "fibre-reaction-center-pair-evidence-cache-v1",
                    "model_root": str(self.model_root.relative_to(ROOT)),
                    "protein_count": int(len(p)),
                    "reaction_count": int(len(r)),
                    "latent_dim": int(p.shape[1]),
                    "labels_used_for_cache": False,
                },
                indent=2,
            )
            + "\n"
        )

    def score(
        self,
        *,
        direction: Direction,
        query_id: str,
        candidate_ids: list[str],
    ) -> EvidenceOutput:
        if direction != "r2e":
            raise ValueError("reaction-centre expert is R2E only")
        q = self.r_index.get(str(query_id), -1)
        rows = np.asarray([self.p_index.get(str(x), -1) for x in candidate_ids], dtype=np.int64)
        available = rows >= 0
        values = np.zeros(len(candidate_ids), dtype=np.float64)
        if q < 0:
            available[:] = False
        elif available.any():
            idx = torch.as_tensor(rows[available], dtype=torch.long, device=self.device)
            values[available] = (
                self.p.index_select(0, idx) @ self.r[q]
            ).float().cpu().numpy()
        return EvidenceOutput(values, available)


class CageFamilyPairEvidence:
    """Family-finetuned EnzymeCAGE score with strict native support semantics.

    This expert never fabricates scores outside the domain assets on which the
    family model was actually run.  That makes it a local expert rather than a
    general baseline.
    """

    kind = "structural"
    role = "rerank"
    directions = ("r2e",)

    FAMILY_FILES = {
        "p450": ROOT
        / "results/fibre_vs_enzymecage_external_families_v1/cage/p450/finetune/test_P450_epoch_9.csv",
        "phosphatase": ROOT
        / "results/fibre_vs_enzymecage_external_families_v1/cage/phosphatase/finetune/test_Phosphatase_epoch_9.csv",
        "terpene": ROOT
        / "results/fibre_vs_enzymecage_external_families_v1/cage/terpene/finetune/test_Terpene_epoch_9.csv",
    }

    def __init__(self, family: str) -> None:
        family = str(family).lower()
        if family not in self.FAMILY_FILES:
            raise ValueError(f"unknown CAGE family expert: {family}")
        self.family = family
        self.name = f"enzymecage_{family}_family"
        frame = pd.read_csv(
            self.FAMILY_FILES[family],
            dtype={"CANO_RXN_SMILES": str, "UniprotID": str},
        ).fillna("")
        score_col = "pred_logit" if "pred_logit" in frame.columns else "pred"
        frame[score_col] = pd.to_numeric(frame[score_col], errors="raise")
        frame = frame.drop_duplicates(["CANO_RXN_SMILES", "UniprotID"], keep="first")
        self._groups = {
            str(query): group.set_index("UniprotID")[score_col].to_dict()
            for query, group in frame.groupby("CANO_RXN_SMILES", sort=False)
        }

        reactions = pd.read_csv(
            ROOT / "data/catalyst_candidate_universes/general_merged/reactions.csv",
            dtype=str,
        ).fillna("")
        self._reaction_smiles = dict(
            zip(reactions["reaction_id"].astype(str), reactions["reaction_smiles"].astype(str))
        )

        metadata = pd.read_csv(
            ROOT / "data/catalyst_candidate_universes/general_merged/protein_metadata.csv",
            dtype=str,
        ).fillna("")
        self._candidate_aliases: dict[str, tuple[str, ...]] = {}
        for record in metadata.to_dict("records"):
            canonical = str(record["protein_id"])
            aliases = [canonical, str(record.get("canonical_accession", ""))]
            aliases.extend(str(record.get("aliases", "")).split(";"))
            self._candidate_aliases[canonical] = tuple(
                dict.fromkeys(x.strip() for x in aliases if x.strip())
            )

    def _query_key(self, query_id: str) -> str:
        value = str(query_id)
        return self._reaction_smiles.get(value, value)

    def score(
        self,
        *,
        direction: Direction,
        query_id: str,
        candidate_ids: list[str],
    ) -> EvidenceOutput:
        if direction != "r2e":
            raise ValueError("family experts are R2E only")
        group = self._groups.get(self._query_key(query_id))
        values = np.zeros(len(candidate_ids), dtype=np.float64)
        available = np.zeros(len(candidate_ids), dtype=bool)
        if group is None:
            return EvidenceOutput(values, available)
        for i, candidate in enumerate(map(str, candidate_ids)):
            aliases = self._candidate_aliases.get(candidate, (candidate,))
            for alias in aliases:
                if alias in group:
                    values[i] = float(group[alias])
                    available[i] = True
                    break
        return EvidenceOutput(values, available)


class SeedHomologyPairEvidence:
    """Query-local seed/homology evidence in the frozen Broad protein space."""

    name = "seed_homology_context"
    kind = "experimental_context"
    role = "rerank"
    directions = ("r2e",)

    def __init__(
        self,
        seed_ids: list[str] | tuple[str, ...],
        *,
        index: FibreCandidateIndex | None = None,
        device: str | torch.device = "cuda",
    ) -> None:
        self.index = index if index is not None else FibreCandidateIndex(device=device)
        self.seed_ids = tuple(
            str(value) for value in seed_ids if str(value) in self.index.protein_index
        )
        if not self.seed_ids:
            raise ValueError("seed/homology expert requires at least one Broad protein seed")
        rows = torch.as_tensor(
            [self.index.protein_index[value] for value in self.seed_ids],
            dtype=torch.long,
            device=self.index.device,
        )
        self.seed_z = self.index.protein_embeddings.index_select(0, rows)

    def score(
        self,
        *,
        direction: Direction,
        query_id: str,
        candidate_ids: list[str],
    ) -> EvidenceOutput:
        if direction != "r2e":
            raise ValueError("seed/homology expert is R2E only")
        rows = np.asarray(
            [self.index.protein_index.get(str(x), -1) for x in candidate_ids],
            dtype=np.int64,
        )
        available = rows >= 0
        values = np.zeros(len(candidate_ids), dtype=np.float64)
        if available.any():
            idx = torch.as_tensor(rows[available], dtype=torch.long, device=self.index.device)
            with torch.no_grad():
                score = (
                    self.index.protein_embeddings.index_select(0, idx) @ self.seed_z.T
                ).max(dim=1).values
            values[available] = score.float().cpu().numpy()
        return EvidenceOutput(values, available)
