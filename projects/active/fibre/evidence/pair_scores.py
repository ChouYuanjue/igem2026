from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from projects.active.fibre.model.assets import GENERAL, ROOT
from projects.active.fibre.runtime.base_model import ModelConfig, TerpeneDualTower
from projects.active.fibre.runtime.scientific_evidence import (
    Direction,
    EvidenceOutput,
)


def _id_rows(path: Path, candidates: tuple[str, ...]) -> tuple[list[str], dict[str, int], pd.DataFrame]:
    frame = pd.read_csv(path, dtype=str).fillna("")
    if "row" in frame.columns:
        frame["row"] = pd.to_numeric(frame["row"]).astype(int)
        frame = frame.sort_values("row", kind="stable")
    column = next(name for name in candidates if name in frame.columns)
    ids = frame[column].astype(str).tolist()
    return ids, {value: i for i, value in enumerate(ids)}, frame


def _normalise(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    norm = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.maximum(norm, 1e-8)


@dataclass(frozen=True)
class PairEvidenceMetadata:
    name: str
    kind: str
    role: str
    directions: tuple[Direction, ...]
    benchmark_scope: str


class ClipzymePairEvidence:
    """CLIPZyme cosine compatibility as structural pair evidence."""

    name = "clipzyme_structure"
    kind = "structural"
    role = "rerank"
    directions = ("r2e", "e2r")

    def __init__(self, *, device: str | torch.device | None = None) -> None:
        self.device = None if device is None else torch.device(device)
        p_root = ROOT / "results/bime_rank_unified_v1/clipzyme_r2e_candidate_asset_v1"
        r_root = ROOT / "results/clipzyme_native_extension_v1/full_hplus_candidate_reactions/clipzyme_embeddings_gpu_v1"
        _, self.p_index, p_frame = _id_rows(p_root / "entries.csv", ("protein_id", "Entry"))
        _, self.r_index, r_frame = _id_rows(r_root / "entries.csv", ("reaction_id", "rhea_id"))
        self.p = np.load(p_root / "embeddings.npy", mmap_mode="r")
        self.r = np.load(r_root / "embeddings.npy", mmap_mode="r")
        self.p_device = None if self.device is None else torch.tensor(np.asarray(self.p), device=self.device)
        self.r_device = None if self.device is None else torch.tensor(np.asarray(self.r), device=self.device)
        if "supported" in p_frame.columns:
            self.p_supported = p_frame["supported"].astype(str).str.lower().eq("true").to_numpy(bool)
        elif "clipzyme_supported" in p_frame.columns:
            self.p_supported = p_frame["clipzyme_supported"].astype(str).str.lower().eq("true").to_numpy(bool)
        else:
            self.p_supported = np.isfinite(self.p).all(axis=1)
        if "clipzyme_supported" in r_frame.columns:
            self.r_supported = r_frame["clipzyme_supported"].astype(str).str.lower().eq("true").to_numpy(bool)
        else:
            self.r_supported = np.isfinite(self.r).all(axis=1)

    def score(self, *, direction: Direction, query_id: str, candidate_ids: list[str]) -> EvidenceOutput:
        if direction not in self.directions:
            raise ValueError(direction)
        if direction == "r2e":
            rr = self.r_index.get(str(query_id), -1)
            q_ok = rr >= 0 and bool(self.r_supported[rr])
            rows = np.asarray([self.p_index.get(str(x), -1) for x in candidate_ids], dtype=np.int64)
            ok = rows >= 0
            if q_ok:
                ok &= self.p_supported[np.maximum(rows, 0)]
            else:
                ok[:] = False
            values = np.zeros(len(candidate_ids), dtype=np.float64)
            if ok.any():
                if self.device is None:
                    values[ok] = np.asarray(self.p[rows[ok]], dtype=np.float32) @ np.asarray(self.r[rr], dtype=np.float32)
                else:
                    idx = torch.as_tensor(rows[ok], dtype=torch.long, device=self.device)
                    values[ok] = (self.p_device.index_select(0, idx) @ self.r_device[rr]).float().cpu().numpy()
        else:
            pp = self.p_index.get(str(query_id), -1)
            q_ok = pp >= 0 and bool(self.p_supported[pp])
            rows = np.asarray([self.r_index.get(str(x), -1) for x in candidate_ids], dtype=np.int64)
            ok = rows >= 0
            if q_ok:
                ok &= self.r_supported[np.maximum(rows, 0)]
            else:
                ok[:] = False
            values = np.zeros(len(candidate_ids), dtype=np.float64)
            if ok.any():
                if self.device is None:
                    values[ok] = np.asarray(self.r[rows[ok]], dtype=np.float32) @ np.asarray(self.p[pp], dtype=np.float32)
                else:
                    idx = torch.as_tensor(rows[ok], dtype=torch.long, device=self.device)
                    values[ok] = (self.r_device.index_select(0, idx) @ self.p_device[pp]).float().cpu().numpy()
        return EvidenceOutput(values, ok)


class TpsPairEvidence:
    """Application-only TPS-family compatibility evidence."""

    name = "tps_family"
    kind = "mechanistic"
    role = "rerank"
    directions = ("r2e", "e2r")
    benchmark_scope = "application_only"

    def __init__(self) -> None:
        root = ROOT / "results/fibre_application/tps_adapted_coordinate"
        p = pd.read_csv(root / "protein_tps_adapted.csv", dtype={"protein_id": str})
        r = pd.read_csv(root / "reaction_tps_adapted.csv", dtype={"reaction_id": str})
        self.p_ids = p.pop("protein_id").astype(str).tolist()
        self.r_ids = r.pop("reaction_id").astype(str).tolist()
        self.p_index = {x: i for i, x in enumerate(self.p_ids)}
        self.r_index = {x: i for i, x in enumerate(self.r_ids)}
        self.p = _normalise(p.to_numpy(np.float32))
        self.r = _normalise(r.to_numpy(np.float32))

    def score(self, *, direction: Direction, query_id: str, candidate_ids: list[str]) -> EvidenceOutput:
        if direction == "r2e":
            q = self.r_index.get(str(query_id), -1)
            rows = np.asarray([self.p_index.get(str(x), -1) for x in candidate_ids], dtype=np.int64)
            ok = (q >= 0) & (rows >= 0)
            values = np.zeros(len(candidate_ids), dtype=np.float64)
            if q >= 0 and np.any(rows >= 0):
                mask = rows >= 0
                values[mask] = self.p[rows[mask]] @ self.r[q]
                ok = mask
        elif direction == "e2r":
            q = self.p_index.get(str(query_id), -1)
            rows = np.asarray([self.r_index.get(str(x), -1) for x in candidate_ids], dtype=np.int64)
            ok = (q >= 0) & (rows >= 0)
            values = np.zeros(len(candidate_ids), dtype=np.float64)
            if q >= 0 and np.any(rows >= 0):
                mask = rows >= 0
                values[mask] = self.r[rows[mask]] @ self.p[q]
                ok = mask
        else:
            raise ValueError(direction)
        return EvidenceOutput(values, np.asarray(ok, dtype=bool))


class DualTowerPairEvidence:
    """Frozen dual-tower checkpoint exposed only as pair-level evidence."""

    kind = "molecular_view"
    role = "rerank"
    directions = ("r2e", "e2r")

    def __init__(
        self,
        *,
        name: str,
        checkpoint: Path,
        protein_feature_dir: Path,
        reaction_feature_dir: Path,
        device: str | torch.device = "cuda",
        encode_batch: int = 4096,
    ) -> None:
        self.name = str(name)
        self.device = torch.device(device)
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        self.model = TerpeneDualTower(ModelConfig(**dict(payload["model_config"])))
        self.model.load_state_dict(payload["model_state_dict"])
        self.model.to(self.device).eval()
        self.p_ids, self.p_index, _ = _id_rows(protein_feature_dir / "entries.csv", ("Entry", "protein_id"))
        self.r_ids, self.r_index, _ = _id_rows(reaction_feature_dir / "entries.csv", ("reaction_id", "rhea_id"))
        p = np.load(protein_feature_dir / "embeddings.npy", mmap_mode="r")
        r = np.load(reaction_feature_dir / "reaction_feature_matrix.npy", mmap_mode="r")
        self.p = self._encode(p, self.model.encode_proteins, encode_batch)
        self.r = self._encode(r, self.model.encode_reactions, encode_batch)

    def _encode(self, matrix: np.ndarray, fn, batch: int) -> torch.Tensor:
        out: list[torch.Tensor] = []
        with torch.no_grad():
            for start in range(0, len(matrix), batch):
                x = torch.as_tensor(np.asarray(matrix[start:start + batch], dtype=np.float32).copy(), device=self.device)
                out.append(fn(x))
        return torch.cat(out, dim=0)

    def score(self, *, direction: Direction, query_id: str, candidate_ids: list[str]) -> EvidenceOutput:
        if direction == "r2e":
            q = self.r_index.get(str(query_id), -1)
            rows = np.asarray([self.p_index.get(str(x), -1) for x in candidate_ids], dtype=np.int64)
            ok = rows >= 0
            values = np.zeros(len(candidate_ids), dtype=np.float64)
            if q < 0:
                ok[:] = False
            elif ok.any():
                idx = torch.as_tensor(rows[ok], dtype=torch.long, device=self.device)
                values[ok] = (self.p.index_select(0, idx) @ self.r[q]).float().cpu().numpy()
        elif direction == "e2r":
            q = self.p_index.get(str(query_id), -1)
            rows = np.asarray([self.r_index.get(str(x), -1) for x in candidate_ids], dtype=np.int64)
            ok = rows >= 0
            values = np.zeros(len(candidate_ids), dtype=np.float64)
            if q < 0:
                ok[:] = False
            elif ok.any():
                idx = torch.as_tensor(rows[ok], dtype=torch.long, device=self.device)
                values[ok] = (self.r.index_select(0, idx) @ self.p[q]).float().cpu().numpy()
        else:
            raise ValueError(direction)
        return EvidenceOutput(values, ok)


def enzgfm_pair_evidence(
    direction: Direction,
    *,
    device: str | torch.device = "cuda",
    fold: int | None = None,
) -> DualTowerPairEvidence:
    if direction not in ("r2e", "e2r"):
        raise ValueError(direction)
    if fold is None:
        root = (
            ROOT / "results/catalyst_clean_mainline_v1/r2e_enzgfm_base_router_v1"
            if direction == "r2e"
            else ROOT / "results/catalyst_clean_mainline_v1/e2r_anchored_lambdamart_v3/experts/enzgfm"
        )
    else:
        root = (
            ROOT / f"results/comprehensive_enzgfm_center_top1_v1/dev/candidate_base/fold{int(fold)}"
            if direction == "r2e"
            else ROOT / f"results/unified_safe_system_v1/e2r_anchored_lambdamart_v3_dev/experts/enzgfm/fold{int(fold)}"
        )
    summary = __import__("json").loads((root / "summary.json").read_text())
    source = str(summary.get("training_source") or summary.get("association_source") or "")
    if "clean2023" not in source:
        raise ValueError(f"EnzGFM expert is not clean2023-bound: {source}")
    return DualTowerPairEvidence(
        name=f"enzgfm_{direction}",
        checkpoint=Path(summary["checkpoint"]),
        protein_feature_dir=Path(summary["protein_feature_dir"]),
        reaction_feature_dir=Path(summary["reaction_feature_dir"]),
        device=device,
    )
