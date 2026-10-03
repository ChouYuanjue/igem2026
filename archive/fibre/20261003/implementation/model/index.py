from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from projects.active.fibre.runtime.base_model import ModelConfig, TerpeneDualTower
from .assets import GENERAL, ROOT


DEFAULT_INDEX = (
    ROOT
    / "results/enzymecage_cleanroom_selected_full_v1/rankstrong_r2e98/models/production_seed20260723.pt"
)


def _ids(path: Path, candidates: tuple[str, ...]) -> list[str]:
    frame = pd.read_csv(path, dtype=str).fillna("")
    if "row" in frame.columns:
        frame["row"] = pd.to_numeric(frame["row"]).astype(int)
        frame = frame.sort_values("row")
    column = next(name for name in candidates if name in frame.columns)
    return frame[column].astype(str).tolist()


class FibreCandidateIndex:
    """Frozen full-universe dual-encoder prior used by canonical FIBRE.

    The broad score owns the candidate universe and remains present with coefficient
    one. Admitted scientific experts can add calibrated pair-level evidence without
    changing this embedding space or removing unsupported candidates.
    """

    def __init__(
        self,
        checkpoint: str | Path = DEFAULT_INDEX,
        *,
        device: str | torch.device = "cuda",
        encode_batch: int = 4096,
    ) -> None:
        self.device = torch.device(device)
        payload = torch.load(Path(checkpoint), map_location="cpu", weights_only=False)
        config = ModelConfig(**dict(payload["model_config"]))
        self.model = TerpeneDualTower(config)
        self.model.load_state_dict(payload["model_state_dict"])
        self.model.to(self.device).eval()

        self.protein_ids = _ids(GENERAL / "proteins/entries.csv", ("Entry", "protein_id"))
        self.reaction_ids = _ids(
            GENERAL / "reaction_features/drfp_categorical_v1/entries.csv",
            ("reaction_id", "rhea_id"),
        )
        self.protein_index = {value: i for i, value in enumerate(self.protein_ids)}
        self.reaction_index = {value: i for i, value in enumerate(self.reaction_ids)}
        self.protein_features = np.load(
            GENERAL / "proteins/embeddings.npy", mmap_mode="r"
        )
        self.reaction_features = np.load(
            GENERAL / "reaction_features/drfp_categorical_v1/reaction_feature_matrix.npy",
            mmap_mode="r",
        )
        self.protein_embeddings = self._encode_all(
            self.protein_features,
            self.model.encode_proteins,
            encode_batch,
        )
        self.reaction_embeddings = self._encode_all(
            self.reaction_features,
            self.model.encode_reactions,
            encode_batch,
        )

    def _encode_all(self, matrix: np.ndarray, fn, batch: int) -> torch.Tensor:
        rows = []
        with torch.no_grad():
            for start in range(0, len(matrix), batch):
                x = torch.as_tensor(
                    np.asarray(matrix[start : start + batch], dtype=np.float32).copy(),
                    device=self.device,
                )
                rows.append(fn(x))
        return torch.cat(rows, dim=0)

    @torch.no_grad()
    def proteins_for_reaction(
        self,
        reaction_id: str,
        *,
        k: int = 4096,
    ) -> tuple[list[str], torch.Tensor]:
        row = self.reaction_index[str(reaction_id)]
        query = self.reaction_embeddings[row]
        scores = self.protein_embeddings @ query
        values, cols = torch.topk(scores, k=min(int(k), len(self.protein_ids)))
        return [self.protein_ids[int(i)] for i in cols.cpu()], values.cpu()

    @torch.no_grad()
    def reactions_for_protein(
        self,
        protein_id: str,
        *,
        k: int = 4096,
    ) -> tuple[list[str], torch.Tensor]:
        row = self.protein_index[str(protein_id)]
        query = self.protein_embeddings[row]
        scores = self.reaction_embeddings @ query
        values, cols = torch.topk(scores, k=min(int(k), len(self.reaction_ids)))
        return [self.reaction_ids[int(i)] for i in cols.cpu()], values.cpu()
