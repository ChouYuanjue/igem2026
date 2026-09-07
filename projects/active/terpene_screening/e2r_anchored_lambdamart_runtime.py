from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import xgboost as xgb

from projects.active.terpene_screening import run_unified_safe_system_e2r_anchored_lambdamart_v3 as v3
from projects.active.terpene_screening.rank_open_world import (
    load_feature_schema,
    load_models_runtime,
    load_registered_reaction_feature_library,
    normalize_rows,
)

ROOT = v3.ROOT
DEFAULT_PROTOCOL = ROOT / "projects/active/terpene_screening/CATALYST_E2R_ANCHORED_LAMBDAMART_V3_PRODUCTION.json"
DEFAULT_BUNDLE = ROOT / "results/catalyst_clean_mainline_v1/e2r_anchored_lambdamart_v3"


@dataclass(frozen=True)
class ProteinRowLibrary:
    root: Path
    ids: tuple[str, ...]
    row_by_id: dict[str, int]
    matrix: np.ndarray

    @classmethod
    def open(cls, root: Path) -> "ProteinRowLibrary":
        root = root.resolve()
        entries = pd.read_csv(root / "entries.csv", dtype={"Entry": str}).sort_values("row")
        ids = tuple(entries["Entry"].astype(str))
        rows = entries["row"].to_numpy(dtype=np.int64)
        if not np.array_equal(rows, np.arange(len(rows), dtype=np.int64)):
            raise ValueError(f"protein library rows are not contiguous: {root}")
        matrix = np.load(root / "embeddings.npy", mmap_mode="r")
        if len(matrix) != len(ids):
            raise ValueError(f"protein library row count mismatch: {root}")
        return cls(root=root, ids=ids, row_by_id={value: i for i, value in enumerate(ids)}, matrix=matrix)

    def normalized_row(self, protein_id: str) -> np.ndarray:
        row = self.row_by_id.get(str(protein_id))
        if row is None:
            raise KeyError(str(protein_id))
        value = np.asarray(self.matrix[row], dtype=np.float32).reshape(1, -1)
        return normalize_rows(value)[0]


@dataclass(frozen=True)
class AnchoredE2RResult:
    candidate_ids: tuple[str, ...]
    order: np.ndarray
    baseline_order: np.ndarray
    selected_rows: np.ndarray
    expert_scores: np.ndarray
    ranker_predictions: np.ndarray

    def top_ids(self, k: int) -> list[str]:
        return [self.candidate_ids[int(row)] for row in self.order[: int(k)]]


class AnchoredE2RRuntime:
    def __init__(
        self,
        *,
        protocol_path: Path = DEFAULT_PROTOCOL,
        bundle_root: Path = DEFAULT_BUNDLE,
        ranker_path: Path | None = None,
        device: str | torch.device = "cuda" if torch.cuda.is_available() else "cpu",
    ) -> None:
        self.protocol_path = protocol_path.resolve()
        self.protocol = json.loads(self.protocol_path.read_text())
        self.bundle_root = bundle_root.resolve()
        self.device = torch.device(device)
        selected = dict(self.protocol["selected_config"])
        self.protected_prefix = int(selected["protected_prefix"])
        self.pool_k = int(selected["pool_k"])
        self.prefix_k = int(selected["prefix_k"])
        self.expert_names = tuple(v3.NAMES)
        if self.expert_names != ("enzgfm", "esmc", "equalblock", "rdkitplus"):
            raise AssertionError(self.expert_names)
        source_ranker = ranker_path or (ROOT / self.protocol["ranker"]["source"])
        self.ranker_path = source_ranker.resolve()
        actual_ranker_sha = hashlib.sha256(self.ranker_path.read_bytes()).hexdigest()
        expected_ranker_sha = str(self.protocol["ranker"]["sha256"])
        if actual_ranker_sha != expected_ranker_sha:
            raise ValueError(f"E2R ranker hash mismatch: {actual_ranker_sha} != {expected_ranker_sha}")
        self.ranker = xgb.Booster()
        self.ranker.load_model(self.ranker_path)
        self._models: dict[str, torch.nn.Module] = {}
        self._protein_libraries: dict[str, ProteinRowLibrary] = {}
        self._reaction_embeddings: dict[str, torch.Tensor] = {}
        reaction_ids: tuple[str, ...] | None = None
        for name in self.expert_names:
            expert_dir = self.bundle_root / "experts" / name
            summary = json.loads((expert_dir / "summary.json").read_text())
            if int(summary.get("dev_fold", 0)) != -1 or int(summary.get("n_train_pairs", 0)) != 218537:
                raise ValueError(f"not a full-clean E2R expert: {expert_dir}")
            models = load_models_runtime(expert_dir / "models", "production", self.device)
            if len(models) != 1:
                raise ValueError(f"expected one frozen production model for {name}, got {len(models)}")
            model = models[0]
            self._models[name] = model
            protein_dir = Path(summary["protein_feature_dir"])
            self._protein_libraries[name] = ProteinRowLibrary.open(protein_dir)
            schema = load_feature_schema(expert_dir)
            reaction_features, local_ids = load_registered_reaction_feature_library(
                Path(summary["reaction_feature_dir"]), schema
            )
            local_tuple = tuple(map(str, local_ids))
            local_index = {value: i for i, value in enumerate(local_tuple)}
            if len(local_index) != len(local_tuple):
                raise ValueError(f"duplicate reaction IDs for expert {name}")
            if reaction_ids is None:
                reaction_ids = tuple(sorted(local_tuple))
            elif set(reaction_ids) != set(local_tuple):
                raise ValueError(f"reaction ID support differs for expert {name}")
            reorder = np.asarray([local_index[value] for value in reaction_ids], dtype=np.int64)
            with torch.no_grad():
                rows: list[torch.Tensor] = []
                for start in range(0, len(reorder), 4096):
                    tensor = torch.as_tensor(
                        reaction_features[reorder[start : start + 4096]], dtype=torch.float32, device=self.device
                    )
                    rows.append(model.encode_reactions(tensor))
                self._reaction_embeddings[name] = torch.cat(rows, dim=0)
        assert reaction_ids is not None
        if len(reaction_ids) != v3.N_CANDIDATES:
            raise ValueError(f"expected {v3.N_CANDIDATES} reactions, got {len(reaction_ids)}")
        if list(reaction_ids) != sorted(reaction_ids):
            raise ValueError("runtime reaction IDs must be lexical, matching frozen V3 tie semantics")
        self.candidate_ids = reaction_ids

    def registered_query_features(self, protein_id: str) -> dict[str, np.ndarray]:
        values = {name: self._protein_libraries[name].normalized_row(protein_id) for name in self.expert_names}
        return values

    def _expert_scores(self, features: dict[str, np.ndarray]) -> np.ndarray:
        scores: list[np.ndarray] = []
        with torch.no_grad():
            for name in self.expert_names:
                value = np.asarray(features[name], dtype=np.float32).reshape(1, -1)
                q = self._models[name].encode_proteins(
                    torch.as_tensor(value, dtype=torch.float32, device=self.device)
                )[0]
                score = (self._reaction_embeddings[name] @ q).detach().cpu().numpy().astype(np.float32)
                scores.append(score)
        return np.stack(scores, axis=0)

    @staticmethod
    def anchored_order(
        expert_scores: np.ndarray,
        ranker_predictions: np.ndarray,
        union_rows: np.ndarray,
        *,
        protected_prefix: int,
        pool_k: int,
        prefix_k: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        S = np.asarray(expert_scores, dtype=np.float32)
        union_rows = np.asarray(union_rows, dtype=np.int32)
        ranks_full = np.stack([v3.full_ranks(S[e]) for e in range(S.shape[0])], axis=1)
        baseline_order = np.argsort(-S[0], kind="stable").astype(np.int32)
        union_ranks = ranks_full[union_rows]
        pool = (union_ranks.min(1) <= int(pool_k)) & (union_ranks[:, 0] > int(protected_prefix))
        local = np.flatnonzero(pool)
        local = local[np.lexsort((union_rows[local], -np.asarray(ranker_predictions)[local]))]
        learned_slots = max(0, int(prefix_k) - int(protected_prefix))
        selected_rows = union_rows[local[:learned_slots]].astype(np.int32, copy=False)
        protected = baseline_order[: int(protected_prefix)]
        blocked = set(map(int, protected)) | set(map(int, selected_rows))
        tail = np.fromiter((int(x) for x in baseline_order if int(x) not in blocked), dtype=np.int32)
        order = np.concatenate([protected, selected_rows, tail]).astype(np.int32, copy=False)
        if len(order) != S.shape[1] or len(np.unique(order)) != len(order):
            raise AssertionError("anchored E2R runtime did not produce a full permutation")
        return order, baseline_order, selected_rows

    def rank_features(self, features: dict[str, np.ndarray]) -> AnchoredE2RResult:
        S = self._expert_scores(features)
        ranks_full = np.stack([v3.full_ranks(S[e]) for e in range(4)], axis=1)
        union = np.asarray(
            sorted(set().union(*(set(map(int, v3.top_rows(S[e], v3.MAX_POOL))) for e in range(4)))),
            dtype=np.int32,
        )
        X = v3.feature_matrix(S, union, ranks_full)
        pred = self.ranker.predict(xgb.DMatrix(X))
        order, baseline_order, selected_rows = self.anchored_order(
            S,
            pred,
            union,
            protected_prefix=self.protected_prefix,
            pool_k=self.pool_k,
            prefix_k=self.prefix_k,
        )
        return AnchoredE2RResult(
            candidate_ids=self.candidate_ids,
            order=order,
            baseline_order=baseline_order,
            selected_rows=selected_rows,
            expert_scores=S,
            ranker_predictions=np.asarray(pred, dtype=np.float32),
        )

    def rank_registered(self, protein_id: str) -> AnchoredE2RResult:
        return self.rank_features(self.registered_query_features(protein_id))
