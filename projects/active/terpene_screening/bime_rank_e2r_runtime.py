from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import xgboost as xgb

from projects.active.terpene_screening.e2r_anchored_lambdamart_runtime import (
    AnchoredE2RResult,
    AnchoredE2RRuntime,
)
from projects.active.terpene_screening.run_e2r_clipzyme_anchored_lambdamart_v4 import anchored_order_v4

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_V4 = ROOT / "results/unified_safe_system_v1/e2r_clipzyme_anchored_lambdamart_v4_dev/selected"
DEFAULT_CLIP_PROTEINS = ROOT / "results/bime_rank_unified_v1/clipzyme_e2r_query_asset_v1"
DEFAULT_CLIP_REACTIONS = ROOT / "results/clipzyme_native_extension_v1/full_hplus_candidate_reactions/clipzyme_embeddings_gpu_v1"


@dataclass(frozen=True)
class BiMEE2RResult:
    candidate_ids: tuple[str, ...]
    order: np.ndarray
    baseline_order: np.ndarray
    selected_rows: np.ndarray
    expert_scores: np.ndarray
    ranker_predictions: np.ndarray
    structure_expert_applied: bool
    structure_expert_name: str | None
    structure_query_supported: bool
    structure_supported_candidates: int

    def top_ids(self, k: int) -> list[str]:
        return [self.candidate_ids[int(row)] for row in self.order[: int(k)]]


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _resolve_asset(value: object, *, relative_to: Path = ROOT) -> Path:
    path = Path(str(value))
    return path.resolve() if path.is_absolute() else (relative_to / path).resolve()


@dataclass(frozen=True)
class _ProteinEmbeddingLayer:
    name: str
    root: Path
    row_by_id: dict[str, int]
    matrix: np.ndarray


def _open_single_protein_layer(root: Path, *, name: str) -> _ProteinEmbeddingLayer:
    entries = pd.read_csv(root / "entries.csv", dtype=str).fillna("")
    if "protein_id" not in entries:
        raise RuntimeError(f"CLIPZyme protein layer has no protein_id: {root}")
    matrix = np.load(root / "embeddings.npy", mmap_mode="r")
    if len(entries) != len(matrix) or matrix.ndim != 2 or matrix.shape[1] != 1280:
        raise RuntimeError(f"CLIPZyme protein layer shape mismatch: {root}")
    if "row" in entries:
        rows = entries["row"].astype(int).to_numpy(np.int64)
        if not np.array_equal(rows, np.arange(len(rows), dtype=np.int64)):
            raise RuntimeError(f"CLIPZyme protein layer rows are not contiguous: {root}")
    ids = entries["protein_id"].astype(str).tolist()
    if len(ids) != len(set(ids)):
        raise RuntimeError(f"duplicate CLIPZyme protein IDs in layer: {root}")
    return _ProteinEmbeddingLayer(name=name, root=root, row_by_id={p: i for i, p in enumerate(ids)}, matrix=matrix)


def _open_layered_query_asset(
    root: Path,
    *,
    expected_manifest_sha256: str | None,
) -> tuple[tuple[_ProteinEmbeddingLayer, ...], dict[str, tuple[int, int]], str]:
    manifest_path = root / "manifest.json"
    actual_manifest_sha = _sha(manifest_path)
    if expected_manifest_sha256 and actual_manifest_sha != str(expected_manifest_sha256):
        raise RuntimeError(
            f"CLIPZyme E2R query asset manifest hash mismatch: {actual_manifest_sha} != {expected_manifest_sha256}"
        )
    manifest = json.loads(manifest_path.read_text())
    if str(manifest.get("asset_type")) != "layered_protein_query_embeddings":
        raise RuntimeError(f"expected layered E2R query asset: {root}")
    if bool(manifest.get("selection_uses_labels", True)) or bool(manifest.get("selection_uses_model_scores", True)):
        raise RuntimeError("E2R structural support registry must be label- and score-independent")

    layer_specs = [
        ("base", manifest["base_asset"], manifest["base_manifest_sha256"]),
        ("pdb_extension", manifest["extension_asset"], manifest["extension_manifest_sha256"]),
    ]
    layers: list[_ProteinEmbeddingLayer] = []
    lookup: dict[str, tuple[int, int]] = {}
    for layer_name, value, expected_sha in layer_specs:
        layer_root = _resolve_asset(value)
        actual_sha = _sha(layer_root / "manifest.json")
        if actual_sha != str(expected_sha):
            raise RuntimeError(f"{layer_name} CLIPZyme protein manifest hash mismatch")
        layer = _open_single_protein_layer(layer_root, name=layer_name)
        layer_index = len(layers)
        for protein_id, row in layer.row_by_id.items():
            if protein_id in lookup:
                raise RuntimeError(f"CLIPZyme E2R query layers overlap at {protein_id}")
            lookup[protein_id] = (layer_index, row)
        layers.append(layer)

    expected_total = int(manifest.get("total_supported_count", -1))
    if expected_total != len(lookup):
        raise RuntimeError(f"layered E2R query support count mismatch: {len(lookup)} != {expected_total}")
    return tuple(layers), lookup, actual_manifest_sha


class BiMEE2RRuntime:
    """Availability-aware E2R wrapper over the frozen Anchored LambdaMART V3.

    The runtime is intentionally configuration-verifiable. A registered protein activates
    the frozen five-expert V4 only when the layered CLIPZyme query registry has an embedding.
    Missing structural context returns the exact V3 order. Reaction candidates outside
    CLIPZyme graph support remain in the 11,081-candidate universe through the four universal
    experts and an explicit support mask.
    """

    def __init__(
        self,
        *,
        v4_root: Path = DEFAULT_V4,
        clip_protein_root: Path = DEFAULT_CLIP_PROTEINS,
        clip_reaction_root: Path = DEFAULT_CLIP_REACTIONS,
        expected_ranker_sha256: str | None = None,
        expected_protein_manifest_sha256: str | None = None,
        expected_reaction_manifest_sha256: str | None = None,
        device: str | torch.device = "cuda" if torch.cuda.is_available() else "cpu",
    ) -> None:
        self.device = torch.device(device)
        self.base = AnchoredE2RRuntime(device=self.device)
        self.candidate_ids = self.base.candidate_ids

        v4_root = v4_root.resolve()
        cfg = json.loads((v4_root / "config.json").read_text())
        config_ranker_sha = str(cfg["ranker_sha256"])
        if expected_ranker_sha256 and config_ranker_sha != str(expected_ranker_sha256):
            raise RuntimeError("candidate E2R V4 config hash does not match route manifest")
        ranker_path = v4_root / "ranker.json"
        actual = _sha(ranker_path)
        expected = str(expected_ranker_sha256 or config_ranker_sha)
        if actual != expected:
            raise RuntimeError(f"E2R V4 ranker hash mismatch: {actual} != {expected}")
        self.ranker = xgb.Booster()
        self.ranker.load_model(ranker_path)
        selected = dict(cfg["selected_config"])
        self.protected_prefix = int(selected["protected_prefix"])
        self.pool_k = int(selected["pool_k"])
        self.prefix_k = int(selected["prefix_k"])
        self.v4_ranker_sha256 = actual

        self._clip_protein_layers, self._clip_protein_lookup, self.protein_manifest_sha256 = _open_layered_query_asset(
            clip_protein_root.resolve(), expected_manifest_sha256=expected_protein_manifest_sha256
        )
        self.structure_supported_queries = len(self._clip_protein_lookup)

        reaction_root = clip_reaction_root.resolve()
        reaction_manifest_path = reaction_root / "manifest.json"
        reaction_manifest_sha = _sha(reaction_manifest_path)
        if expected_reaction_manifest_sha256 and reaction_manifest_sha != str(expected_reaction_manifest_sha256):
            raise RuntimeError("CLIPZyme E2R reaction asset manifest hash mismatch")
        self.reaction_manifest_sha256 = reaction_manifest_sha
        re = pd.read_csv(reaction_root / "entries.csv", dtype=str).fillna("")
        rmat = np.load(reaction_root / "embeddings.npy", mmap_mode="r")
        if rmat.ndim != 2 or rmat.shape[1] != 1280:
            raise RuntimeError("CLIPZyme reaction asset dimension mismatch")
        supported = re[re["clipzyme_supported"].str.lower().eq("true")]
        row_by_id = {str(r): int(row) for r, row in supported[["reaction_id", "row"]].itertuples(index=False)}
        support = np.asarray([rid in row_by_id for rid in self.candidate_ids], dtype=bool)
        dense = np.zeros((len(self.candidate_ids), 1280), dtype=np.float32)
        ids = [rid for rid in self.candidate_ids if rid in row_by_id]
        dense[support] = np.asarray(rmat[[row_by_id[rid] for rid in ids]], dtype=np.float32)
        if not np.isfinite(dense[support]).all():
            raise RuntimeError("supported CLIPZyme reaction embeddings are non-finite")
        self.clip_candidate_support = support
        self.structure_supported_candidates = int(support.sum())
        self._clip_reaction_tensor = torch.as_tensor(dense, dtype=torch.float32, device=self.device)

    @staticmethod
    def _wrap_base(result: AnchoredE2RResult, supported_candidates: int) -> BiMEE2RResult:
        return BiMEE2RResult(
            candidate_ids=result.candidate_ids,
            order=result.order,
            baseline_order=result.baseline_order,
            selected_rows=result.selected_rows,
            expert_scores=result.expert_scores,
            ranker_predictions=result.ranker_predictions,
            structure_expert_applied=False,
            structure_expert_name=None,
            structure_query_supported=False,
            structure_supported_candidates=int(supported_candidates),
        )

    def query_supported(self, protein_id: str) -> bool:
        return str(protein_id) in self._clip_protein_lookup

    def _query_embedding(self, protein_id: str) -> np.ndarray:
        layer_index, row = self._clip_protein_lookup[str(protein_id)]
        value = np.array(self._clip_protein_layers[layer_index].matrix[row], dtype=np.float32, copy=True)
        if value.shape != (1280,) or not np.isfinite(value).all():
            raise RuntimeError("supported CLIPZyme protein query embedding is invalid")
        norm = float(np.linalg.norm(value))
        if norm <= 0:
            raise RuntimeError("supported CLIPZyme protein query embedding has zero norm")
        return value / norm

    def seed_similarity_scores(self, seed_ids: list[str]) -> np.ndarray | None:
        """Mean cosine across the four frozen non-structural BiME reaction experts."""
        index = {rid: i for i, rid in enumerate(self.candidate_ids)}
        rows = [index[str(rid)] for rid in seed_ids if str(rid) in index]
        if not rows:
            return None
        total = np.zeros(len(self.candidate_ids), dtype=np.float32)
        for name in self.base.expert_names:
            emb = self.base._reaction_embeddings[name]
            with torch.no_grad():
                values = emb @ emb[torch.as_tensor(rows, dtype=torch.long, device=emb.device)].T
                total += values.max(dim=1).values.detach().cpu().numpy().astype(np.float32, copy=False)
        return total / float(len(self.base.expert_names))

    def rank_registered(self, protein_id: str) -> BiMEE2RResult:
        protein_id = str(protein_id)
        if not self.query_supported(protein_id):
            return self._wrap_base(self.base.rank_registered(protein_id), self.structure_supported_candidates)

        features = self.base.registered_query_features(protein_id)
        S = self.base._expert_scores(features)
        q = self._query_embedding(protein_id)
        qt = torch.as_tensor(q, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            cscore = (self._clip_reaction_tensor @ qt).detach().cpu().numpy().astype(np.float32, copy=False)
        order, audit = anchored_order_v4(
            S,
            cscore,
            self.clip_candidate_support,
            True,
            self.ranker,
            protected_prefix=self.protected_prefix,
            pool_k=self.pool_k,
            prefix_k=self.prefix_k,
        )
        baseline_order = np.argsort(-S[0], kind="stable").astype(np.int32)
        selected_size = int(audit["selected_size"])
        selected = order[self.protected_prefix : self.protected_prefix + selected_size].astype(np.int32, copy=False)
        return BiMEE2RResult(
            candidate_ids=self.candidate_ids,
            order=order,
            baseline_order=baseline_order,
            selected_rows=selected,
            expert_scores=S,
            ranker_predictions=np.empty(0, dtype=np.float32),
            structure_expert_applied=True,
            structure_expert_name="CLIPZyme",
            structure_query_supported=True,
            structure_supported_candidates=self.structure_supported_candidates,
        )
