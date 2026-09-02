from __future__ import annotations

import argparse
import copy
import json
import sys
import threading
from dataclasses import replace
from functools import lru_cache
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
from drfp import DrfpEncoder
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from projects.active.terpene_screening.core.cache import (  # noqa: E402
    DEFAULT_FEATURE_CACHE,
    FeatureCache,
    stable_digest,
)
from projects.active.terpene_screening.core.input_audit import (  # noqa: E402
    ProteinInputAudit,
    ReactionInputAudit,
    audit_protein_sequence,
    clean_protein_sequence,
    initial_reaction_audit,
)
from projects.active.terpene_screening.core.evidence import (  # noqa: E402
    apply_evidence_passport,
)
from projects.active.terpene_screening.core.conformal import (  # noqa: E402
    DEFAULT_CONFORMAL_CALIBRATORS,
    SUPPORTED_CONFORMAL_MODES,
    apply_conformal_retrieval_set,
)
from projects.active.terpene_screening.core.provenance import (  # noqa: E402
    apply_route_provenance,
    identifier_set_hash,
    write_query_audit,
)
from projects.active.terpene_screening.core.routing import (  # noqa: E402
    DEFAULT_ROUTE_MANIFEST,
    resolve_route,
)
from projects.active.terpene_screening.core.taxonomy_scope import (  # noqa: E402
    DEFAULT_TAXONOMY_SCOPE_REGISTRY,
    SUPPORTED_ENZYME_TAXONOMY_SCOPES,
    TAXONOMY_SCOPE_VERSION,
    filter_candidate_ids,
    taxonomy_record,
    validate_scope,
    validate_seed_scope,
)
from projects.active.terpene_screening.core.registry_snapshots import (  # noqa: E402
    registry_version,
    resolve_protein_dir,
    resolve_reaction_path,
)
from projects.active.terpene_screening.evaluate_zero_shot_retrieval_cold import (  # noqa: E402
    reaction_features as zero_shot_reaction_features,
    reaction_similarity as zero_shot_reaction_similarity,
)
from projects.active.terpene_screening.extract_esmc_embeddings import mean_embedding  # noqa: E402
from projects.active.terpene_screening.dual_kernel_runtime import (  # noqa: E402
    DualKernelAssets,
    align_reaction_scores as align_dual_kernel_reaction_scores,
    load_assets as load_dual_kernel_assets,
    score_query as dual_kernel_score_query,
)
from projects.active.terpene_screening.gate_matrix import (  # noqa: E402
    canonical_or_raw_reaction,
    precursor_class_from_reaction,
    product_skeleton_class,
)
from projects.active.terpene_screening.train_dual_tower_cold import (  # noqa: E402
    ModelConfig,
    ProjectionTower,
    TerpeneDualTower,
    reaction_multiview_features,
)

DEFAULT_POSITIVES = ROOT / "data/terpene/enzyme_terpene_synthase.tsv"
DEFAULT_CAGE_SCORES = ROOT / "results/terpene_cage_screen/all_rhea_gate/all_pair_scores.csv"
DEFAULT_BASE_R2E_DUAL_TOWER_DIR = ROOT / "results/terpene_production_models/drfp_categorical"
DEFAULT_BASE_E2R_DUAL_TOWER_DIR = ROOT / "results/terpene_production_models/multiview"
DEFAULT_R2E_DUAL_TOWER_DIR = ROOT / "results/terpene_production_models/marts_adapted_drfp_pu"
DEFAULT_R2E_TOP3_10_DUAL_TOWER_DIR = ROOT / "results/terpene_production_models/marts_adapted_drfp_pu_r2e075"
# Backward-compatible alias: batch scripts still use this as their short-list model.
DEFAULT_R2E_TOP3_DUAL_TOWER_DIR = DEFAULT_R2E_TOP3_10_DUAL_TOWER_DIR
DEFAULT_R2E_TOP10_20_DUAL_TOWER_DIR = ROOT / "results/terpene_production_models/marts_adapted_drfp_pu_r2e_exact_residual"
DEFAULT_E2R_DUAL_TOWER_DIR = ROOT / "results/terpene_production_models/marts_adapted_drfp_pu_e2r"
DEFAULT_E2R_HARDNEG_DUAL_TOWER_DIR = (
    ROOT / "results/terpene_production_models/marts_adapted_drfp_pu_e2r_hardneg128"
)
DEFAULT_E2R_TOP20_DUAL_KERNEL_DIR = (
    ROOT / "results/terpene_production_models/marts_dual_kernel_e2r_top20"
)
E2R_TOP10_RRF_PRIMARY_WEIGHT = 0.35
E2R_TOP10_RRF_CONSTANT = 60.0
E2R_TOP10_PRIMARY_NEIGHBOR_K = 5
E2R_TOP10_PRIMARY_DIRECT_WEIGHT = 0.5
E2R_TOP10_SECONDARY_NEIGHBOR_K = 3
E2R_TOP10_SECONDARY_DIRECT_WEIGHT = 0.9
E2R_TOP20_RRF_PRIMARY_WEIGHT = 0.70
E2R_TOP20_RRF_CONSTANT = 60.0
DEFAULT_PROTEIN_DIR = ROOT / "data/terpene_embeddings/esmc600m_mean"
DEFAULT_REGISTERED_PROTEIN_DIR = ROOT / "data/terpene_open_world_registry/proteins"
DEFAULT_REGISTERED_REACTIONS = ROOT / "data/terpene_open_world_registry/reactions.csv"
DEFAULT_OUTPUT = ROOT / "results/terpene_open_world"
DEFAULT_UNCERTAINTY_CALIBRATORS = ROOT / "results/terpene_open_world_uncertainty_rrf_routing/calibrators.json"
UNCERTAINTY_FEATURE_COLUMNS = [
    "query_nearest_library_similarity",
    "ensemble_top1_vote_fraction",
    "ensemble_top1_rank_std",
    "ensemble_top1_score_std",
    "ensemble_top1_margin_z",
    "ensemble_topk_jaccard",
    "ensemble_topk_vote_mean",
    "ensemble_boundary_margin_z",
]


def normalize_rows(matrix: np.ndarray) -> np.ndarray:
    denominator = np.linalg.norm(matrix, axis=1, keepdims=True)
    denominator[denominator == 0] = 1
    return matrix / denominator


@lru_cache(maxsize=4)
def load_dual_kernel_assets_cached(path: str) -> DualKernelAssets:
    return load_dual_kernel_assets(Path(path))


def clean_sequence(value: object) -> str:
    """Backward-compatible sequence cleaner used by existing scripts."""
    return clean_protein_sequence(value)


def should_use_e2r_top20_dual_kernel(
    *,
    ranking_objective: str,
    is_current_enzyme: bool,
    has_seed_reactions: bool,
    requested_retrieval_mode: str,
    model_dir_override: Path | None,
    dual_tower_dir: Path,
    has_temporary_external_reactions: bool,
    registered_reactions_csv: Path | None,
) -> bool:
    return (
        ranking_objective == "top20"
        and not is_current_enzyme
        and not has_seed_reactions
        and requested_retrieval_mode == "auto"
        and model_dir_override is None
        and dual_tower_dir.resolve() == DEFAULT_E2R_DUAL_TOWER_DIR.resolve()
        and not has_temporary_external_reactions
        and registered_reactions_csv is not None
        and registered_reactions_csv.resolve() == DEFAULT_REGISTERED_REACTIONS.resolve()
    )


class ReactionDistillationResidualBlock(nn.Module):
    def __init__(self, dimension: int, dropout: float) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.LayerNorm(dimension),
            nn.Linear(dimension, dimension * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dimension * 2, dimension),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return values + self.network(values)


class ReactionFeatureDistillerInference(nn.Module):
    def __init__(self, config: dict[str, object]) -> None:
        super().__init__()
        input_dim = int(config["input_dim"])
        hidden_dim = int(config["hidden_dim"])
        output_dim = int(config.get("output_dim", 512))
        dropout = float(config.get("dropout", 0.1))
        residual_blocks = int(config.get("residual_blocks", 2))
        layers: list[nn.Module] = [
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        ]
        layers.extend(
            ReactionDistillationResidualBlock(hidden_dim, dropout)
            for _ in range(residual_blocks)
        )
        layers.extend([nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, output_dim)])
        self.network = nn.Sequential(*layers)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.network(values), dim=-1)


class SelfContainedResidualReactionDualTower(nn.Module):
    def __init__(
        self,
        base_config: ModelConfig,
        aux_input_dim: int,
        aux_hidden_dim: int,
        gate_init: float,
        vector_gate: bool,
        distiller_config: dict[str, object],
    ) -> None:
        super().__init__()
        self.config = base_config
        self.protein_tower = ProjectionTower(
            base_config.protein_input_dim,
            base_config.hidden_dim,
            base_config.embedding_dim,
            base_config.dropout,
        )
        self.base_reaction_tower = ProjectionTower(
            base_config.reaction_input_dim,
            base_config.hidden_dim,
            base_config.embedding_dim,
            base_config.dropout,
        )
        self.aux_reaction_tower = ProjectionTower(
            aux_input_dim,
            aux_hidden_dim,
            base_config.embedding_dim,
            base_config.dropout,
        )
        gate_shape = (base_config.embedding_dim,) if vector_gate else ()
        self.gate_logit = nn.Parameter(torch.full(gate_shape, float(gate_init)))
        self.reaction_distiller = ReactionFeatureDistillerInference(distiller_config)

    def encode_proteins(self, values: torch.Tensor) -> torch.Tensor:
        return self.protein_tower(values)

    def encode_reactions(self, values: torch.Tensor) -> torch.Tensor:
        base = self.base_reaction_tower(values)
        auxiliary_features = self.reaction_distiller(values)
        auxiliary = self.aux_reaction_tower(auxiliary_features)
        gate = torch.sigmoid(self.gate_logit)
        return F.normalize(base + gate * auxiliary, dim=-1)


class IdentityHiddenResidualReactionDualTower(nn.Module):
    """Add auxiliary reaction features before the base GELU with exact zero-init identity.

    The original reaction LayerNorm and Linear operate on exactly the original
    base feature block.  A bias-free auxiliary projection is added to the first
    hidden pre-activation and is initialized to zero, so an expanded checkpoint
    exactly reproduces the source dual tower before continuation.
    """

    def __init__(self, base_config: ModelConfig, aux_input_dim: int) -> None:
        super().__init__()
        if aux_input_dim <= 0:
            raise ValueError("aux_input_dim must be positive")
        self.config = base_config
        self.aux_input_dim = int(aux_input_dim)
        self.protein_tower = ProjectionTower(
            base_config.protein_input_dim,
            base_config.hidden_dim,
            base_config.embedding_dim,
            base_config.dropout,
        )
        self.base_reaction_tower = ProjectionTower(
            base_config.reaction_input_dim,
            base_config.hidden_dim,
            base_config.embedding_dim,
            base_config.dropout,
        )
        self.aux_to_hidden = nn.Linear(self.aux_input_dim, base_config.hidden_dim, bias=False)
        nn.init.zeros_(self.aux_to_hidden.weight)

    @property
    def total_reaction_input_dim(self) -> int:
        return int(self.config.reaction_input_dim + self.aux_input_dim)

    def load_base_state(self, state_dict: dict[str, torch.Tensor]) -> None:
        own = self.state_dict()
        seen: set[str] = set()
        for key, value in state_dict.items():
            if key.startswith("protein_tower."):
                target = key
            elif key.startswith("reaction_tower."):
                target = "base_reaction_tower." + key[len("reaction_tower.") :]
            else:
                continue
            if target not in own or own[target].shape != value.shape:
                raise ValueError(f"Cannot map production parameter {key} -> {target}")
            own[target].copy_(value)
            seen.add(target)
        expected = {
            name for name in own
            if name.startswith("protein_tower.") or name.startswith("base_reaction_tower.")
        }
        if seen != expected:
            raise ValueError(f"Base state mapping incomplete: missing={sorted(expected - seen)}")
        own["aux_to_hidden.weight"].zero_()
        self.load_state_dict(own)

    def encode_proteins(self, values: torch.Tensor) -> torch.Tensor:
        return self.protein_tower(values)

    def encode_reactions(self, values: torch.Tensor) -> torch.Tensor:
        if values.shape[-1] != self.total_reaction_input_dim:
            raise ValueError(
                f"Expected {self.total_reaction_input_dim} reaction features, got {values.shape[-1]}"
            )
        base_values = values[..., : self.config.reaction_input_dim]
        auxiliary_values = values[..., self.config.reaction_input_dim :]
        network = self.base_reaction_tower.network
        hidden = network[1](network[0](base_values)) + self.aux_to_hidden(auxiliary_values)
        hidden = network[2](hidden)
        hidden = network[3](hidden)
        output = network[4](hidden)
        return F.normalize(output, p=2, dim=-1)


class BoundedIdentityHiddenResidualReactionDualTower(IdentityHiddenResidualReactionDualTower):
    """Identity-preserving hidden residual with a fixed per-row geometry cap."""

    def __init__(self, base_config: ModelConfig, aux_input_dim: int, max_residual_ratio: float) -> None:
        if not 0.0 < float(max_residual_ratio) <= 1.0:
            raise ValueError("max_residual_ratio must be in (0, 1]")
        super().__init__(base_config, aux_input_dim)
        self.max_residual_ratio = float(max_residual_ratio)

    def encode_reactions(self, values: torch.Tensor) -> torch.Tensor:
        if values.shape[-1] != self.total_reaction_input_dim:
            raise ValueError(
                f"Expected {self.total_reaction_input_dim} reaction features, got {values.shape[-1]}"
            )
        base_values = values[..., : self.config.reaction_input_dim]
        auxiliary_values = values[..., self.config.reaction_input_dim :]
        network = self.base_reaction_tower.network
        base_hidden = network[1](network[0](base_values))
        residual = self.aux_to_hidden(auxiliary_values)
        residual_norm = residual.norm(p=2, dim=-1, keepdim=True)
        cap = self.max_residual_ratio * base_hidden.norm(p=2, dim=-1, keepdim=True)
        scale = torch.clamp(cap / residual_norm.clamp_min(1e-12), max=1.0)
        hidden = base_hidden + residual * scale
        hidden = network[2](hidden)
        hidden = network[3](hidden)
        output = network[4](hidden)
        return F.normalize(output, p=2, dim=-1)


class ExactResidualReactionDualTower(nn.Module):
    requires_auxiliary_reaction_features = True

    def __init__(
        self,
        base_config: ModelConfig,
        aux_input_dim: int,
        aux_hidden_dim: int,
        gate_init: float,
        vector_gate: bool,
    ) -> None:
        super().__init__()
        self.config = base_config
        self.protein_tower = ProjectionTower(
            base_config.protein_input_dim,
            base_config.hidden_dim,
            base_config.embedding_dim,
            base_config.dropout,
        )
        self.base_reaction_tower = ProjectionTower(
            base_config.reaction_input_dim,
            base_config.hidden_dim,
            base_config.embedding_dim,
            base_config.dropout,
        )
        self.aux_reaction_tower = ProjectionTower(
            aux_input_dim,
            aux_hidden_dim,
            base_config.embedding_dim,
            base_config.dropout,
        )
        gate_shape = (base_config.embedding_dim,) if vector_gate else ()
        self.gate_logit = nn.Parameter(torch.full(gate_shape, float(gate_init)))

    def encode_proteins(self, values: torch.Tensor) -> torch.Tensor:
        return self.protein_tower(values)

    def encode_reactions(
        self,
        values: torch.Tensor,
        auxiliary_values: torch.Tensor,
    ) -> torch.Tensor:
        base = self.base_reaction_tower(values)
        auxiliary = self.aux_reaction_tower(auxiliary_values)
        gate = torch.sigmoid(self.gate_logit)
        return F.normalize(base + gate * auxiliary, dim=-1)


def models_require_auxiliary_reaction_features(models: list[nn.Module]) -> bool:
    flags = [bool(getattr(model, "requires_auxiliary_reaction_features", False)) for model in models]
    if any(flags) and not all(flags):
        raise ValueError("Cannot ensemble models with mixed auxiliary-reaction requirements")
    return bool(flags and flags[0])


def encode_model_reactions(
    model: nn.Module,
    reaction_tensor: torch.Tensor,
    auxiliary_reaction_tensor: torch.Tensor | None,
) -> torch.Tensor:
    if bool(getattr(model, "requires_auxiliary_reaction_features", False)):
        if auxiliary_reaction_tensor is None:
            raise ValueError("Exact residual model requires auxiliary reaction features")
        return model.encode_reactions(reaction_tensor, auxiliary_reaction_tensor)
    return model.encode_reactions(reaction_tensor)


def load_models(model_dir: Path, scope: str, device: torch.device) -> list[nn.Module]:
    pattern = "production_seed*.pt" if scope == "production" else f"{scope}_fold*.pt"
    checkpoints = sorted(model_dir.glob(pattern))
    if not checkpoints:
        raise FileNotFoundError(f"No {scope} checkpoints found under {model_dir}")
    models: list[nn.Module] = []
    distiller_payload: dict[str, object] | None = None
    for path in checkpoints:
        payload = torch.load(path, map_location=device, weights_only=False)
        model_type = str(payload.get("model_type", "dual_tower"))
        if model_type == "rdkitplus_bounded_identity_hidden_residual":
            base_config = ModelConfig(**payload["base_model_config"])
            model = BoundedIdentityHiddenResidualReactionDualTower(
                base_config, int(payload["aux_input_dim"]), float(payload["max_residual_ratio"])
            ).to(device)
            model.load_state_dict(payload["model_state_dict"])
        elif model_type == "rdkitplus_identity_hidden_residual":
            base_config = ModelConfig(**payload["base_model_config"])
            model = IdentityHiddenResidualReactionDualTower(
                base_config, int(payload["aux_input_dim"])
            ).to(device)
            model.load_state_dict(payload["model_state_dict"])
        elif model_type == "horizyn_reaction_residual_exact":
            base_config = ModelConfig(**payload["base_model_config"])
            model = ExactResidualReactionDualTower(
                base_config,
                int(payload["aux_input_dim"]),
                int(payload["aux_hidden_dim"]),
                float(payload["gate_init"]),
                bool(payload.get("vector_gate", False)),
            ).to(device)
            model.load_state_dict(payload["model_state_dict"])
        elif model_type == "horizyn_reaction_residual":
            local_distiller = model_dir.parent / "reaction_feature_distiller.pt"
            if not local_distiller.exists():
                raise FileNotFoundError(
                    f"Residual checkpoint requires packaged reaction distiller: {local_distiller}"
                )
            if distiller_payload is None:
                distiller_payload = torch.load(
                    local_distiller, map_location="cpu", weights_only=False
                )
            base_config = ModelConfig(**payload["base_model_config"])
            model = SelfContainedResidualReactionDualTower(
                base_config,
                int(payload["aux_input_dim"]),
                int(payload["aux_hidden_dim"]),
                float(payload["gate_init"]),
                bool(payload.get("vector_gate", False)),
                dict(distiller_payload["model_config"]),
            ).to(device)
            missing, unexpected = model.load_state_dict(
                payload["model_state_dict"], strict=False
            )
            expected_missing = {
                f"reaction_distiller.{name}"
                for name in distiller_payload["model_state_dict"]
            }
            if set(missing) != expected_missing or unexpected:
                raise ValueError(
                    f"Residual checkpoint mismatch for {path}: missing={missing}, unexpected={unexpected}"
                )
            model.reaction_distiller.load_state_dict(
                distiller_payload["model_state_dict"]
            )
        else:
            config = ModelConfig(**payload["model_config"])
            model = TerpeneDualTower(config).to(device)
            model.load_state_dict(payload["model_state_dict"])
        model.eval()
        models.append(model)
    return models


@lru_cache(maxsize=16)
def _load_models_runtime_cached(
    model_dir: str,
    scope: str,
    device: str,
) -> tuple[nn.Module, ...]:
    return tuple(load_models(Path(model_dir), scope, torch.device(device)))


def load_models_runtime(
    model_dir: Path,
    scope: str,
    device: torch.device,
) -> list[nn.Module]:
    """Return shared eval-mode models for a long-lived inference process."""
    return list(
        _load_models_runtime_cached(
            str(model_dir.resolve()),
            scope,
            str(device),
        )
    )


@lru_cache(maxsize=16)
def _load_feature_schema_cached(path: str) -> dict[str, object]:
    schema_path = Path(path)
    if not schema_path.exists():
        raise FileNotFoundError(schema_path)
    return json.loads(schema_path.read_text(encoding="utf-8"))


def load_feature_schema(dual_tower_dir: Path) -> dict[str, object]:
    return dict(
        _load_feature_schema_cached(
            str((dual_tower_dir / "feature_schema.json").resolve())
        )
    )


@lru_cache(maxsize=512)
def _encode_runtime_rdkitplus_extension(reaction_smiles: str) -> np.ndarray:
    """Encode the exact 1024-d Horizyn RDKit+ block used by registered RDKit+ assets."""
    import tempfile

    horizyn_root = ROOT / "external/horizyn"
    if str(horizyn_root) not in sys.path:
        sys.path.insert(0, str(horizyn_root))
    from horizyn.datasets.csv import CSVDataset
    from horizyn.datasets.fingerprints.rdkit_plus import RDKitPlusFingerprintDataset

    key = "runtime_query"
    with tempfile.TemporaryDirectory(prefix="catalyst_rdkitplus_runtime_") as temp_dir:
        csv_path = Path(temp_dir) / "query.csv"
        pd.DataFrame({"reaction_id": [key], "reaction_smiles": [str(reaction_smiles)]}).to_csv(
            csv_path, index=False
        )
        dataset = CSVDataset(
            str(csv_path), key_column="reaction_id", columns=["reaction_smiles"]
        )
        fingerprint = RDKitPlusFingerprintDataset(
            reaction_dataset=dataset,
            vec_dim=1024,
            mol_fp_type="morgan",
            rxn_fp_type="struct",
            use_chirality=False,
            standardize=True,
            standardize_hypervalent=True,
            standardize_remove_hs=True,
            standardize_kekulize=False,
            standardize_uncharge=True,
            standardize_metals=True,
        )[key]
    values = fingerprint.detach().cpu().numpy().astype(np.float32, copy=False)
    if values.shape != (1024,):
        raise ValueError(f"Runtime RDKit+ extension has unexpected shape {values.shape}")
    return values


@lru_cache(maxsize=1)
def _runtime_rxnmapper():
    """Load the exact local RXNMapper preprocessing model once per retrieval process."""
    from projects.active.terpene_screening.build_rxnmapper_reaction_mapping import load_mapper

    runtime = ROOT / "external_runtime/rxnmapper"
    return load_mapper(runtime, batch_size=1, allow_cuda=True)


@lru_cache(maxsize=512)
def _encode_runtime_reaction_center_extension(
    reaction_smiles: str,
    center_fp_size: int,
    token_dim: int,
    radius: int,
) -> np.ndarray:
    """Map one raw reaction and reproduce the deterministic registered center block."""
    from projects.active.terpene_screening.build_reaction_center_augmented_features import (
        reaction_center_features,
    )

    info = list(_runtime_rxnmapper().map_reactions_with_info([str(reaction_smiles)]))[0]
    mapped = str(info.get("mapped_rxn", "")) if info else ""
    if not mapped or mapped == ">>":
        raise ValueError("RXNMapper did not return a valid mapped reaction")
    values, _ = reaction_center_features(
        mapped,
        center_fp_size=int(center_fp_size),
        token_dim=int(token_dim),
        radius=int(radius),
    )
    expected = 2 * int(center_fp_size) + int(token_dim)
    if values.shape != (expected,):
        raise ValueError(
            f"Runtime reaction-center extension has unexpected shape {values.shape}; expected {(expected,)}"
        )
    return values.astype(np.float32, copy=False)


def encode_reaction_with_audit(
    reaction_smiles: str,
    schema: dict[str, object],
    *,
    failure_policy: str = "warn",
    cache_dir: Path | None = DEFAULT_FEATURE_CACHE,
) -> tuple[np.ndarray, ReactionInputAudit]:
    if failure_policy not in {"strict", "warn", "fallback"}:
        raise ValueError(f"Unsupported reaction feature policy: {failure_policy}")
    raw_reaction = str(reaction_smiles)
    canonical = canonical_or_raw_reaction(raw_reaction)
    audit = initial_reaction_audit(raw_reaction, canonical)
    schema_signature = {
        "drfp_dimension": schema["drfp_dimension"],
        "feature_mode": schema.get("feature_mode", "drfp_categorical"),
        "precursor_classes": schema["precursor_classes"],
        "product_skeleton_classes": schema["product_skeleton_classes"],
        "reaction_feature_dimension": schema.get("reaction_feature_dimension"),
        "reaction_feature_mode_extension": schema.get("reaction_feature_mode_extension"),
        "reaction_center_extension": schema.get("reaction_center_extension"),
    }
    digest = stable_digest(
        "reaction-runtime-v2",
        {"reaction_raw": raw_reaction, "reaction_canonical": canonical, "schema": schema_signature},
    )
    cache = FeatureCache(cache_dir) if cache_dir is not None else None
    expected_dimension = int(schema.get("reaction_feature_dimension") or 0)
    if cache is not None:
        cached = cache.get("reaction_runtime_v2", digest)
        if cached is not None and (expected_dimension <= 0 or len(cached) == expected_dimension):
            return cached, replace(audit, drfp_status="cached")

    drfp_dimension = int(schema["drfp_dimension"])
    drfp_succeeded = False
    drfp_error = ""
    if ">>" in canonical:
        try:
            drfp = DrfpEncoder.encode([canonical])[0].astype(np.float32, copy=False)
            drfp_succeeded = True
        except Exception as exc:
            drfp_error = f"{type(exc).__name__}:{exc}"
            if failure_policy == "strict":
                raise ValueError(f"DRFP encoding failed for reaction: {canonical}") from exc
            drfp = np.zeros(drfp_dimension, dtype=np.float32)
    else:
        drfp_error = "missing_reaction_arrow"
        if failure_policy == "strict":
            raise ValueError(f"Reaction input lacks a reaction arrow: {canonical}")
        drfp = np.zeros(drfp_dimension, dtype=np.float32)

    feature_blocks = [drfp]
    feature_mode = str(schema.get("feature_mode", "drfp_categorical"))
    if feature_mode == "multiview":
        substrate_fp, product_fp, signed_difference, descriptors = reaction_multiview_features(canonical)
        feature_blocks.extend([substrate_fp, product_fp, signed_difference, descriptors])
    elif feature_mode != "drfp_categorical":
        raise ValueError(f"Unsupported reaction feature mode: {feature_mode}")

    precursor_values = [str(value) for value in schema["precursor_classes"]]
    skeleton_values = [str(value) for value in schema["product_skeleton_classes"]]
    categorical = np.zeros(len(precursor_values) + len(skeleton_values), dtype=np.float32)
    precursor = precursor_class_from_reaction(canonical)
    skeleton = product_skeleton_class(canonical)
    if precursor in precursor_values:
        categorical[precursor_values.index(precursor)] = 1.0
    elif "unknown" in precursor_values:
        categorical[precursor_values.index("unknown")] = 1.0
    if skeleton in skeleton_values:
        categorical[len(precursor_values) + skeleton_values.index(skeleton)] = 1.0
    elif "unknown" in skeleton_values:
        categorical[len(precursor_values) + skeleton_values.index("unknown")] = 1.0
    feature_blocks.append(categorical)
    values = np.concatenate(feature_blocks).astype(np.float32)

    extension_errors: list[str] = []
    extension_fallback = False
    center_spec = dict(schema.get("reaction_center_extension") or {})
    center_dim = int(center_spec.get("dimension") or 0)
    expected_dimension = int(schema.get("reaction_feature_dimension") or len(values))
    base_target_dimension = expected_dimension - center_dim
    if base_target_dimension < len(values):
        raise ValueError(
            f"Reaction schema base target {base_target_dimension} is smaller than encoded base {len(values)}"
        )
    if len(values) < base_target_dimension:
        missing = base_target_dimension - len(values)
        extension_mode = str(schema.get("reaction_feature_mode_extension") or "")
        if missing != 1024 or (
            extension_mode != "append_horizyn_rdkitplus_struct_morgan1024" and not center_spec
        ):
            raise ValueError(
                f"Unsupported reaction feature extension: need {missing} dimensions before center"
            )
        try:
            rdkitplus = _encode_runtime_rdkitplus_extension(raw_reaction).copy()
        except Exception as exc:
            if failure_policy == "strict":
                raise ValueError("Runtime RDKit+ feature encoding failed") from exc
            rdkitplus = np.zeros(1024, dtype=np.float32)
            extension_fallback = True
            extension_errors.append(f"rdkitplus_zero_fallback:{type(exc).__name__}:{exc}")
        values = np.concatenate([values, rdkitplus]).astype(np.float32, copy=False)

    if center_dim:
        center_fp_size = int(center_spec.get("center_fp_size_each_side") or 0)
        token_dim = int(center_spec.get("token_dim") or 0)
        radius = int(center_spec.get("radius") or 0)
        if 2 * center_fp_size + token_dim != center_dim or min(center_fp_size, token_dim, radius) <= 0:
            raise ValueError("Invalid reaction-center extension contract in feature schema")
        try:
            center = _encode_runtime_reaction_center_extension(
                raw_reaction, center_fp_size, token_dim, radius
            ).copy()
        except Exception as exc:
            if failure_policy == "strict":
                raise ValueError("Runtime reaction-center feature encoding failed") from exc
            center = np.zeros(center_dim, dtype=np.float32)
            extension_fallback = True
            extension_errors.append(f"reaction_center_zero_fallback:{type(exc).__name__}:{exc}")
        values = np.concatenate([values, center]).astype(np.float32, copy=False)

    if len(values) != expected_dimension:
        raise ValueError(
            f"Runtime reaction feature width mismatch: encoded {len(values)} != schema {expected_dimension}"
        )
    warning_parts = [value for value in [audit.warning, drfp_error, *extension_errors] if value]
    audited = replace(
        audit,
        status="valid" if drfp_succeeded and not audit.warning and not extension_errors else "warning",
        drfp_status="encoded" if drfp_succeeded else "failed",
        fallback_used=(not drfp_succeeded) or extension_fallback,
        warning=";".join(warning_parts),
    )
    if cache is not None and drfp_succeeded and not extension_fallback:
        cache.put("reaction_runtime_v2", digest, values)
    return values, audited


def encode_reaction(reaction_smiles: str, schema: dict[str, object]) -> np.ndarray:
    """Backward-compatible encoder; production paths use encode_reaction_with_audit."""
    values, _ = encode_reaction_with_audit(
        reaction_smiles,
        schema,
        failure_policy="fallback",
    )
    return values


@lru_cache(maxsize=8)
def _load_protein_library_cached(protein_dir: str) -> tuple[np.ndarray, tuple[str, ...]]:
    protein_dir = resolve_protein_dir(Path(protein_dir))
    entries = pd.read_csv(protein_dir / "entries.csv", dtype={"Entry": str}).sort_values("row")
    matrix = np.load(protein_dir / "embeddings.npy").astype(np.float32)
    if len(entries) != len(matrix):
        raise ValueError("Protein feature matrix and entries file have different lengths.")
    normalized = normalize_rows(matrix)
    return normalized, tuple(entries["Entry"].astype(str))


def load_protein_library(protein_dir: Path) -> tuple[np.ndarray, list[str]]:
    """Load one immutable normalized base library, copying only the ID container.

    Production requests previously re-read and normalized the full protein matrix on
    every request. That behavior was tolerable for the historical ~2k TPS universe
    but dominates latency for the merged ~186k universe. The matrix itself is never
    mutated by ranking code; callers receive a fresh list for IDs because temporary
    open-world candidates may be appended to that list.
    """

    resolved = resolve_protein_dir(protein_dir).resolve()
    matrix, entries = _load_protein_library_cached(str(resolved))
    return matrix, list(entries)


def load_reaction_library(dual_tower_dir: Path, schema: dict[str, object]) -> tuple[np.ndarray, list[str]]:
    matrix = np.load(dual_tower_dir / "reaction_feature_matrix.npy").astype(np.float32)
    reaction_ids = [str(value) for value in schema["reaction_ids"]]
    if len(reaction_ids) != len(matrix):
        raise ValueError("Reaction feature matrix and reaction ID schema have different lengths.")
    return matrix, reaction_ids


@lru_cache(maxsize=4)
def _load_registered_reaction_feature_library_cached(
    feature_dir: str,
) -> tuple[np.ndarray, tuple[str, ...], dict[str, object]]:
    directory = Path(feature_dir)
    entries = pd.read_csv(directory / "entries.csv", dtype={"reaction_id": str}).sort_values("row")
    matrix = np.load(directory / "reaction_feature_matrix.npy").astype(np.float32)
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    if len(entries) != len(matrix):
        raise ValueError("Registered reaction feature matrix and entries file differ in length")
    if entries["reaction_id"].duplicated().any():
        raise ValueError("Registered reaction feature entries contain duplicate reaction IDs")
    return matrix, tuple(entries["reaction_id"].astype(str)), manifest


def load_registered_reaction_feature_library(
    feature_dir: Path,
    schema: dict[str, object],
) -> tuple[np.ndarray, list[str]]:
    """Load a persistent expanded reaction library compatible with the active schema."""

    matrix, ids, manifest = _load_registered_reaction_feature_library_cached(
        str(feature_dir.resolve())
    )
    contract = dict(manifest.get("contract") or {})
    # A registered reaction library may be reused with a different protein
    # representation.  Older manifests were built from a full dual-tower schema
    # and therefore carried protein-only metadata such as
    # ``protein_feature_dimension``.  That field is not part of the reaction
    # feature contract and must not make an otherwise identical reaction library
    # incompatible with a new protein encoder.  All reaction-side (and any
    # unknown non-protein) contract keys remain fail-closed.
    protein_only_contract_keys = {"protein_feature_dimension"}
    for key, expected in contract.items():
        if key in protein_only_contract_keys:
            continue
        if schema.get(key) != expected:
            raise ValueError(
                f"Registered reaction feature schema mismatch for {key}: "
                f"{schema.get(key)!r} != {expected!r}"
            )
    expected_dim = int(schema.get("reaction_feature_dimension") or matrix.shape[1])
    if matrix.ndim != 2 or matrix.shape[1] != expected_dim:
        raise ValueError(
            f"Registered reaction feature width mismatch: {matrix.shape} vs {expected_dim}"
        )
    return matrix, list(ids)


def load_auxiliary_reaction_library(
    dual_tower_dir: Path,
    reaction_ids: list[str],
) -> np.ndarray:
    path = dual_tower_dir / "auxiliary_reaction_feature_matrix.npy"
    if not path.exists():
        raise FileNotFoundError(
            f"Exact residual deployment requires auxiliary reaction matrix: {path}"
        )
    matrix = np.load(path).astype(np.float32)
    if len(matrix) != len(reaction_ids):
        raise ValueError("Auxiliary reaction matrix and reaction IDs differ in length")
    return matrix


def encode_packaged_distilled_reactions(
    base_reaction_features: np.ndarray,
    deployment_dir: Path,
    device: torch.device,
) -> np.ndarray:
    checkpoint = deployment_dir / "reaction_feature_distiller.pt"
    if not checkpoint.exists():
        raise FileNotFoundError(
            f"Exact residual fallback requires packaged reaction distiller: {checkpoint}"
        )
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model = ReactionFeatureDistillerInference(dict(payload["model_config"])).to(device)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    tensor = torch.as_tensor(base_reaction_features, dtype=torch.float32, device=device)
    with torch.no_grad():
        return model(tensor).cpu().numpy().astype(np.float32)


def encode_exact_horizyn_reactions(
    reaction_smiles: list[str],
    deployment_dir: Path,
    device: torch.device,
    base_reaction_features: np.ndarray | None = None,
) -> np.ndarray:
    if not reaction_smiles:
        return np.empty((0, 512), dtype=np.float32)
    import tempfile

    from projects.active.terpene_screening.train_horizyn_reaction_adapter_double_cold import (
        build_horizyn_fingerprints,
        encode_horizyn_reactions,
    )

    checkpoint = deployment_dir / "horizyn_v1_0_dev.ckpt"
    config = deployment_dir / "horizyn_sota.yaml"
    if not checkpoint.exists() or not config.exists():
        raise FileNotFoundError(
            f"Exact residual deployment lacks Horizyn runtime assets: {checkpoint}, {config}"
        )
    frame = pd.DataFrame(
        {
            "reaction_id": [f"runtime_{index}" for index in range(len(reaction_smiles))],
            "reaction_smiles": reaction_smiles,
        }
    )
    with tempfile.TemporaryDirectory(prefix="tps_horizyn_runtime_") as temp_dir:
        fingerprints, audit = build_horizyn_fingerprints(
            frame,
            config,
            Path(temp_dir),
        )
        exact = normalize_rows(
            encode_horizyn_reactions(
                fingerprints,
                checkpoint,
                device,
                batch_size=128,
            )
        )
    success = audit["success"].astype(str).str.lower().eq("true").to_numpy()
    if not success.all():
        if base_reaction_features is None:
            failures = audit.loc[~success].head().to_dict("records")
            raise ValueError(
                f"Horizyn reaction fingerprint failures without fallback base features: {failures}"
            )
        base_reaction_features = np.asarray(base_reaction_features, dtype=np.float32)
        if len(base_reaction_features) != len(reaction_smiles):
            raise ValueError("Fallback base reaction features differ in row count")
        fallback = encode_packaged_distilled_reactions(
            base_reaction_features,
            deployment_dir,
            device,
        )
        exact[~success] = fallback[~success]
    return exact.astype(np.float32, copy=False)

def load_external_enzyme_rows(path: Path | None) -> pd.DataFrame:
    if path is None:
        return pd.DataFrame(columns=["enzyme_id", "sequence"])
    frame = pd.read_csv(path, dtype=str).fillna("")
    id_column = "enzyme_id" if "enzyme_id" in frame.columns else "Entry" if "Entry" in frame.columns else None
    sequence_column = "sequence" if "sequence" in frame.columns else "Sequence" if "Sequence" in frame.columns else None
    if id_column is None or sequence_column is None:
        raise ValueError("External enzyme CSV requires enzyme_id/Entry and sequence/Sequence columns.")
    frame = frame[[id_column, sequence_column]].rename(columns={id_column: "enzyme_id", sequence_column: "sequence"})
    frame["enzyme_id"] = frame["enzyme_id"].astype(str).str.strip()
    frame["sequence"] = frame["sequence"].map(clean_sequence)
    return frame[(frame["enzyme_id"] != "") & (frame["sequence"] != "")].drop_duplicates("enzyme_id")


def load_external_reaction_rows(path: Path | None) -> pd.DataFrame:
    if path is None:
        return pd.DataFrame(columns=["reaction_id", "reaction_smiles"])
    path = resolve_reaction_path(path)
    frame = pd.read_csv(path, dtype=str).fillna("")
    id_column = "reaction_id" if "reaction_id" in frame.columns else "rhea_id" if "rhea_id" in frame.columns else None
    smiles_column = "reaction_smiles" if "reaction_smiles" in frame.columns else "smiles_seq" if "smiles_seq" in frame.columns else None
    if id_column is None or smiles_column is None:
        raise ValueError("External reaction CSV requires reaction_id/rhea_id and reaction_smiles/smiles_seq columns.")
    frame = frame[[id_column, smiles_column]].rename(columns={id_column: "reaction_id", smiles_column: "reaction_smiles"})
    frame["reaction_id"] = frame["reaction_id"].astype(str).str.strip()
    frame["reaction_smiles"] = frame["reaction_smiles"].astype(str).str.strip()
    return frame[(frame["reaction_id"] != "") & (frame["reaction_smiles"] != "")].drop_duplicates("reaction_id")


_ESMC_MODEL_CACHE: dict[tuple[str, str], object] = {}
_ESMC_MODEL_SOURCE: dict[tuple[str, str], str] = {}
_ESMC_MODEL_LOAD_LOCK = threading.RLock()
_ESMC_LOCAL_SPECS: dict[str, dict[str, object]] = {
    "esmc_300m": {
        "repo_id": "EvolutionaryScale/esmc-300m-2024-12",
        "weights": "data/weights/esmc_300m_2024_12_v0.pth",
        "d_model": 960,
        "n_heads": 15,
        "n_layers": 30,
    },
    "esmc_600m": {
        "repo_id": "EvolutionaryScale/esmc-600m-2024-12",
        "weights": "data/weights/esmc_600m_2024_12_v0.pth",
        "d_model": 1152,
        "n_heads": 18,
        "n_layers": 36,
    },
}


def _resolve_cached_esmc_assets(model_name: str) -> tuple[Path, dict[str, object]] | None:
    """Resolve a fixed ESM-C checkpoint from the local HF cache without networking.

    The upstream ESM loader calls ``snapshot_download`` on every fresh process, even
    when the complete multi-gigabyte checkpoint is already cached. On production
    hosts that remote repository check can dominate cold-start latency. We therefore
    probe the exact supported cache entry with ``local_files_only=True`` first and
    fall back to the upstream loader only when the checkpoint is genuinely absent.
    """
    spec = _ESMC_LOCAL_SPECS.get(str(model_name))
    if spec is None:
        return None
    try:
        from huggingface_hub import snapshot_download

        snapshot = Path(snapshot_download(
            repo_id=str(spec["repo_id"]),
            local_files_only=True,
        ))
    except Exception:
        return None
    weights = snapshot / str(spec["weights"])
    if not weights.is_file():
        return None
    return weights, spec


def _load_esmc_from_local_cache(model_name: str, device: str):
    resolved = _resolve_cached_esmc_assets(model_name)
    if resolved is None:
        return None
    weights, spec = resolved
    from esm.models.esmc import ESMC
    from esm.tokenization import get_esmc_model_tokenizers

    target = torch.device(device)
    with torch.device(target):
        model = ESMC(
            d_model=int(spec["d_model"]),
            n_heads=int(spec["n_heads"]),
            n_layers=int(spec["n_layers"]),
            tokenizer=get_esmc_model_tokenizers(),
        ).eval()
    state_dict = torch.load(weights, map_location=target, weights_only=True)
    model.load_state_dict(state_dict)
    if target.type != "cpu":
        model = model.to(torch.bfloat16)
    return model


def load_esmc_model_cached(model_name: str, device: str):
    """Load one ESM-C instance per model/device with local-first single-flight semantics.

    The scientific model and checkpoint are unchanged. The local-cache path merely
    avoids an unnecessary Hugging Face network resolution on every web-service
    restart. If the fixed checkpoint is not cached, the official loader remains the
    fallback so first-time installation still works.
    """
    key = (str(model_name), str(device))
    with _ESMC_MODEL_LOAD_LOCK:
        cached = _ESMC_MODEL_CACHE.get(key)
        if cached is not None:
            return cached
        model = _load_esmc_from_local_cache(str(model_name), str(device))
        source = "local_huggingface_cache"
        if model is None:
            from esm.models.esmc import ESMC

            model = ESMC.from_pretrained(model_name).eval().to(device)
            source = "upstream_huggingface_fallback"
        _ESMC_MODEL_CACHE[key] = model
        _ESMC_MODEL_SOURCE[key] = source
        return model


def prewarm_esmc_model(model_name: str = "esmc_600m", device: str | None = None) -> dict[str, str]:
    target = str(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    load_esmc_model_cached(model_name, target)
    return {
        "model": str(model_name),
        "device": target,
        "status": "ready",
        "source": _ESMC_MODEL_SOURCE.get((str(model_name), target), "unknown"),
    }


def encode_external_enzymes_with_audit(
    frame: pd.DataFrame,
    device: str,
    model_name: str,
    *,
    input_policy: str = "warn",
    cache_dir: Path | None = DEFAULT_FEATURE_CACHE,
) -> tuple[np.ndarray, list[ProteinInputAudit]]:
    if frame.empty:
        return np.empty((0, 1152), dtype=np.float32), []
    cleaned: list[str] = []
    audits: list[ProteinInputAudit] = []
    for sequence in frame["sequence"].astype(str):
        value, audit = audit_protein_sequence(sequence, policy=input_policy)
        cleaned.append(value)
        audits.append(audit)
    cache = FeatureCache(cache_dir) if cache_dir is not None else None
    vectors: list[np.ndarray | None] = []
    missing: list[int] = []
    for index, sequence in enumerate(cleaned):
        digest = stable_digest("esmc-mean-v1", {"model": model_name, "sequence": sequence})
        cached = cache.get("esmc_mean_v1", digest) if cache is not None else None
        vectors.append(cached)
        if cached is None:
            missing.append(index)
    if missing:
        model = load_esmc_model_cached(model_name, device)
        for index in missing:
            vector = mean_embedding(model, cleaned[index], max_residues=1000, overlap=100, device=device)
            vectors[index] = np.asarray(vector, dtype=np.float32)
            if cache is not None:
                digest = stable_digest("esmc-mean-v1", {"model": model_name, "sequence": cleaned[index]})
                cache.put("esmc_mean_v1", digest, vectors[index])
    matrix = np.stack([np.asarray(value, dtype=np.float32) for value in vectors])
    return normalize_rows(matrix), audits


def encode_external_enzymes(frame: pd.DataFrame, device: str, model_name: str) -> np.ndarray:
    values, _ = encode_external_enzymes_with_audit(frame, device, model_name)
    return values


def ensemble_similarity(
    models: list[nn.Module],
    protein_features: np.ndarray,
    reaction_features: np.ndarray,
    device: torch.device,
    auxiliary_reaction_features: np.ndarray | None = None,
) -> np.ndarray:
    return ensemble_similarity_members(
        models,
        protein_features,
        reaction_features,
        device,
        auxiliary_reaction_features,
    ).mean(axis=0)


def ensemble_similarity_members(
    models: list[nn.Module],
    protein_features: np.ndarray,
    reaction_features: np.ndarray,
    device: torch.device,
    auxiliary_reaction_features: np.ndarray | None = None,
) -> np.ndarray:
    protein_tensor = torch.as_tensor(protein_features, dtype=torch.float32, device=device)
    reaction_tensor = torch.as_tensor(reaction_features, dtype=torch.float32, device=device)
    auxiliary_tensor = (
        torch.as_tensor(auxiliary_reaction_features, dtype=torch.float32, device=device)
        if auxiliary_reaction_features is not None
        else None
    )
    if auxiliary_tensor is not None and len(auxiliary_tensor) != len(reaction_tensor):
        raise ValueError("Base and auxiliary reaction feature matrices differ in row count")
    requires_auxiliary = models_require_auxiliary_reaction_features(models)
    if requires_auxiliary and auxiliary_tensor is None:
        raise ValueError("Exact residual ensemble requires auxiliary reaction features")
    member_scores: list[np.ndarray] = []
    with torch.no_grad():
        for model in models:
            protein_embeddings = model.encode_proteins(protein_tensor)
            reaction_embeddings = encode_model_reactions(
                model,
                reaction_tensor,
                auxiliary_tensor,
            )
            member_scores.append((reaction_embeddings @ protein_embeddings.T).cpu().numpy())
    if not member_scores:
        raise ValueError("No models were supplied.")
    return np.stack(member_scores).astype(np.float32, copy=False)


@lru_cache(maxsize=4)
def _cached_base_protein_model_embeddings(
    model_dir: str,
    scope: str,
    device: str,
    protein_dir: str,
) -> tuple[np.ndarray, ...]:
    """Project one immutable base protein universe once per model ensemble.

    The returned arrays live on CPU to keep long-lived GPU memory bounded. Query-time
    reaction projection remains on the configured device, while the final dense dot
    product uses NumPy against the cached normalized protein-tower output. This is
    mathematically identical to the original dense dual-tower score.
    """

    models = _load_models_runtime_cached(model_dir, scope, device)
    protein_features, _ = _load_protein_library_cached(protein_dir)
    protein_tensor = torch.as_tensor(
        protein_features,
        dtype=torch.float32,
        device=torch.device(device),
    )
    projected: list[np.ndarray] = []
    with torch.no_grad():
        for model in models:
            values = model.encode_proteins(protein_tensor).cpu().numpy().astype(np.float32, copy=False)
            values.setflags(write=False)
            projected.append(values)
    return tuple(projected)


def ensemble_similarity_members_cached_base_proteins(
    *,
    model_dir: Path,
    scope: str,
    device: torch.device,
    protein_dir: Path,
    reaction_features: np.ndarray,
    auxiliary_reaction_features: np.ndarray | None = None,
) -> np.ndarray:
    """Score reactions against a cached, unmodified base protein universe."""

    model_dir_key = str(model_dir.resolve())
    protein_dir_key = str(resolve_protein_dir(protein_dir).resolve())
    device_key = str(device)
    models = _load_models_runtime_cached(model_dir_key, scope, device_key)
    protein_embeddings = _cached_base_protein_model_embeddings(
        model_dir_key,
        scope,
        device_key,
        protein_dir_key,
    )
    reaction_tensor = torch.as_tensor(reaction_features, dtype=torch.float32, device=device)
    auxiliary_tensor = (
        torch.as_tensor(auxiliary_reaction_features, dtype=torch.float32, device=device)
        if auxiliary_reaction_features is not None
        else None
    )
    if auxiliary_tensor is not None and len(auxiliary_tensor) != len(reaction_tensor):
        raise ValueError("Base and auxiliary reaction feature matrices differ in row count")
    requires_auxiliary = models_require_auxiliary_reaction_features(list(models))
    if requires_auxiliary and auxiliary_tensor is None:
        raise ValueError("Exact residual ensemble requires auxiliary reaction features")
    member_scores: list[np.ndarray] = []
    with torch.no_grad():
        for model, protein_embedding in zip(models, protein_embeddings, strict=True):
            reaction_embedding = encode_model_reactions(model, reaction_tensor, auxiliary_tensor)
            query = reaction_embedding.cpu().numpy().astype(np.float32, copy=False)
            member_scores.append(query @ protein_embedding.T)
    if not member_scores:
        raise ValueError("No models were supplied.")
    return np.stack(member_scores).astype(np.float32, copy=False)

def route_member_scores(
    direct_member_scores: np.ndarray,
    seed_scores: np.ndarray | None,
    candidate_ids: list[str],
    mode: str,
    neighbor_scores: np.ndarray | None,
    hybrid_direct_weight: float,
) -> np.ndarray:
    routed = []
    for direct_scores in direct_member_scores:
        scores, _ = choose_retrieval_scores(
            direct_scores,
            seed_scores,
            candidate_ids,
            mode,
            neighbor_scores=neighbor_scores,
            hybrid_direct_weight=hybrid_direct_weight,
        )
        routed.append(scores)
    return np.stack(routed).astype(np.float32, copy=False)


def candidate_subset_indices(candidate_ids: list[str], requested_ids: list[str] | None) -> tuple[list[int], dict[str, int | bool]]:
    """Resolve an exact candidate-side include subset without interpreting user intent.

    Ordering follows the deployed candidate library, not request order, so scoring and
    provenance remain deterministic. Unknown requested IDs are audited and ignored; an
    entirely unmatched non-empty request is rejected instead of silently widening scope.
    """
    requested = list(dict.fromkeys(str(value).strip() for value in (requested_ids or []) if str(value).strip()))
    if not requested:
        return list(range(len(candidate_ids))), {
            "applied": False, "requested_count": 0, "effective_count": len(candidate_ids), "missing_count": 0,
        }
    allowed = set(requested)
    keep = [index for index, value in enumerate(candidate_ids) if value in allowed]
    if not keep:
        raise ValueError("None of the requested candidate IDs exist in the selected candidate universe")
    effective = {candidate_ids[index] for index in keep}
    return keep, {
        "applied": True,
        "requested_count": len(requested),
        "effective_count": len(keep),
        "missing_count": len(allowed - effective),
    }


def apply_candidate_subset_metadata(result: pd.DataFrame, audit: dict[str, int | bool]) -> pd.DataFrame:
    result["candidate_subset_applied"] = bool(audit.get("applied"))
    result["candidate_subset_requested_count"] = int(audit.get("requested_count", 0))
    result["candidate_subset_effective_count"] = int(audit.get("effective_count", 0))
    result["candidate_subset_missing_count"] = int(audit.get("missing_count", 0))
    return result


def rank_positions(scores: np.ndarray, candidate_ids: list[str], masked_ids: set[str]) -> np.ndarray:
    adjusted = scores.astype(np.float64, copy=True)
    id_to_index = {value: index for index, value in enumerate(candidate_ids)}
    for value in masked_ids:
        index = id_to_index.get(value)
        if index is not None:
            adjusted[index] = -np.inf
    order = np.lexsort((np.asarray(candidate_ids), -adjusted))
    ranks = np.full(len(candidate_ids), np.nan, dtype=np.float32)
    valid_order = [int(index) for index in order if np.isfinite(adjusted[index])]
    ranks[np.asarray(valid_order, dtype=np.int64)] = np.arange(1, len(valid_order) + 1, dtype=np.float32)
    return ranks


def reciprocal_rank_fusion_scores(
    primary_scores: np.ndarray,
    secondary_scores: np.ndarray,
    candidate_ids: list[str],
    primary_weight: float = E2R_TOP10_RRF_PRIMARY_WEIGHT,
    constant: float = E2R_TOP10_RRF_CONSTANT,
    masked_ids: set[str] | None = None,
) -> np.ndarray:
    if not 0 <= primary_weight <= 1:
        raise ValueError("RRF primary weight must be within [0, 1]")
    if constant < 0:
        raise ValueError("RRF constant must be non-negative")
    if primary_scores.shape != secondary_scores.shape:
        raise ValueError("RRF score vectors must have matching shapes")
    if len(primary_scores) != len(candidate_ids):
        raise ValueError("RRF score vectors and candidate IDs differ in length")
    masked = masked_ids or set()
    primary_ranks = rank_positions(primary_scores, candidate_ids, masked)
    secondary_ranks = rank_positions(secondary_scores, candidate_ids, masked)
    fused = np.zeros(len(candidate_ids), dtype=np.float32)
    valid = np.isfinite(primary_ranks) & np.isfinite(secondary_ranks)
    fused[valid] = (
        primary_weight / (constant + primary_ranks[valid])
        + (1.0 - primary_weight) / (constant + secondary_ranks[valid])
    )
    fused[~valid] = -np.inf
    return fused


def reciprocal_rank_fusion_members(
    primary_member_scores: np.ndarray,
    secondary_member_scores: np.ndarray,
    candidate_ids: list[str],
    primary_weight: float = E2R_TOP10_RRF_PRIMARY_WEIGHT,
    constant: float = E2R_TOP10_RRF_CONSTANT,
    masked_ids: set[str] | None = None,
) -> np.ndarray:
    if primary_member_scores.shape != secondary_member_scores.shape:
        raise ValueError("RRF member score tensors must have matching shapes")
    if primary_member_scores.ndim != 2:
        raise ValueError("RRF member score tensors must be two-dimensional")
    return np.stack(
        [
            reciprocal_rank_fusion_scores(
                primary_member_scores[index],
                secondary_member_scores[index],
                candidate_ids,
                primary_weight,
                constant,
                masked_ids,
            )
            for index in range(len(primary_member_scores))
        ]
    ).astype(np.float32, copy=False)


def ensemble_query_diagnostics(
    member_scores: np.ndarray,
    candidate_ids: list[str],
    masked_ids: set[str],
    top_k: int,
    consensus_scores: np.ndarray | None = None,
) -> dict[str, float]:
    mean_scores = (
        np.asarray(consensus_scores, dtype=np.float64).copy()
        if consensus_scores is not None
        else member_scores.mean(axis=0).astype(np.float64)
    )
    if mean_scores.shape != (len(candidate_ids),):
        raise ValueError("Consensus scores and candidate IDs differ in length")
    id_to_index = {value: index for index, value in enumerate(candidate_ids)}
    for value in masked_ids:
        index = id_to_index.get(value)
        if index is not None:
            mean_scores[index] = -np.inf
    order = [
        int(index)
        for index in np.lexsort((np.asarray(candidate_ids), -mean_scores))
        if np.isfinite(mean_scores[index])
    ]
    if not order:
        return {
            "ensemble_top1_vote_fraction": 0.0,
            "ensemble_top1_rank_std": float("nan"),
            "ensemble_top1_score_std": float("nan"),
            "ensemble_top1_margin_z": float("nan"),
            "ensemble_topk_jaccard": float("nan"),
            "ensemble_topk_vote_mean": float("nan"),
            "ensemble_boundary_margin_z": float("nan"),
        }
    top1_index = order[0]
    member_ranks = np.stack(
        [rank_positions(scores, candidate_ids, masked_ids) for scores in member_scores]
    )
    member_top_sets: list[set[int]] = []
    member_top1 = []
    effective_k = min(max(top_k, 1), len(order))
    for scores in member_scores:
        adjusted = scores.astype(np.float64, copy=True)
        for value in masked_ids:
            index = id_to_index.get(value)
            if index is not None:
                adjusted[index] = -np.inf
        member_order = [
            int(index)
            for index in np.lexsort((np.asarray(candidate_ids), -adjusted))
            if np.isfinite(adjusted[index])
        ]
        member_top1.append(member_order[0])
        member_top_sets.append(set(member_order[:effective_k]))
    pairwise_jaccard = []
    for left in range(len(member_top_sets)):
        for right in range(left + 1, len(member_top_sets)):
            union = member_top_sets[left] | member_top_sets[right]
            pairwise_jaccard.append(
                len(member_top_sets[left] & member_top_sets[right]) / len(union) if union else 1.0
            )
    finite_scores = mean_scores[np.isfinite(mean_scores)]
    score_scale = float(np.std(finite_scores))
    second_score = mean_scores[order[1]] if len(order) > 1 else mean_scores[top1_index]
    margin_z = (
        float((mean_scores[top1_index] - second_score) / score_scale)
        if score_scale > 0
        else 0.0
    )
    selected = np.asarray(order[:effective_k], dtype=np.int64)
    topk_vote_mean = (
        float(np.mean(member_ranks[:, selected] <= effective_k)) if len(selected) else 0.0
    )
    boundary_margin_z = 0.0
    if effective_k < len(order) and score_scale > 0:
        boundary_margin_z = float(
            (mean_scores[order[effective_k - 1]] - mean_scores[order[effective_k]]) / score_scale
        )
    return {
        "ensemble_top1_vote_fraction": float(np.mean(np.asarray(member_top1) == top1_index)),
        "ensemble_top1_rank_std": float(np.nanstd(member_ranks[:, top1_index])),
        "ensemble_top1_score_std": float(np.std(member_scores[:, top1_index])),
        "ensemble_top1_margin_z": margin_z,
        "ensemble_topk_jaccard": float(np.mean(pairwise_jaccard)) if pairwise_jaccard else 1.0,
        "ensemble_topk_vote_mean": topk_vote_mean,
        "ensemble_boundary_margin_z": boundary_margin_z,
    }


@lru_cache(maxsize=8)
def load_calibrators_cached(path: str) -> dict[str, object]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def apply_empirical_reliability(
    result: pd.DataFrame,
    direction: str,
    ranking_objective: str,
    calibrators_path: Path,
    applicable: bool,
    not_applicable_reason: str,
) -> pd.DataFrame:
    key = f"{direction}_{ranking_objective}"
    result["empirical_reliability_score"] = np.nan
    result["empirical_reliability_tier"] = "uncalibrated"
    result["empirical_reliability_calibrator"] = key
    result["empirical_reliability_binding_status"] = "not_checked"
    result["reliability_recommendation"] = "manual_review_required"
    if not applicable:
        result["empirical_reliability_status"] = not_applicable_reason
        return result
    if not calibrators_path.exists():
        result["empirical_reliability_status"] = "calibrator_missing"
        return result
    calibrators = load_calibrators_cached(str(calibrators_path.resolve()))
    calibrator = calibrators.get(key)
    if not calibrator:
        result["empirical_reliability_status"] = "calibrator_unavailable"
        return result
    compatibility = calibrator.get("compatibility")
    if compatibility:
        row = result.iloc[0]
        mismatches = []
        for field in ["route_id", "candidate_universe_hash", "model_bundle_version"]:
            expected = str(compatibility.get(field, ""))
            actual = str(row.get(field, ""))
            if expected and expected != actual:
                mismatches.append(f"{field}:{actual}!={expected}")
        if mismatches:
            result["empirical_reliability_binding_status"] = ";".join(mismatches)
            result["empirical_reliability_status"] = "incompatible_calibrator"
            return result
        result["empirical_reliability_binding_status"] = "compatible"
    else:
        result["empirical_reliability_binding_status"] = "legacy_unbound"
    if not bool(calibrator.get("deployable")):
        result["empirical_reliability_status"] = "failed_double_cold_validation"
        return result
    row = result.iloc[0]
    feature_columns = [str(value) for value in calibrator["feature_columns"]]
    production_column = {
        "query_nearest_train_similarity": "query_nearest_library_similarity",
    }
    values = np.asarray(
        [
            float(row.get(production_column.get(column, column), np.nan))
            for column in feature_columns
        ],
        dtype=np.float64,
    )
    imputer = np.asarray(calibrator["imputer_statistics"], dtype=np.float64)
    values = np.where(np.isfinite(values), values, imputer)
    mean = np.asarray(calibrator["scaler_mean"], dtype=np.float64)
    scale = np.asarray(calibrator["scaler_scale"], dtype=np.float64)
    scale[scale == 0] = 1.0
    coefficient = np.asarray(calibrator["coefficient"], dtype=np.float64)
    logit = float(np.dot((values - mean) / scale, coefficient) + float(calibrator["intercept"]))
    score = float(1.0 / (1.0 + np.exp(-np.clip(logit, -30.0, 30.0))))
    thresholds = calibrator["thresholds"]
    if score >= float(thresholds["high"]):
        tier = "higher_evidence"
    elif score < float(thresholds["low"]):
        tier = "lower_evidence"
    else:
        tier = "intermediate"
    result["empirical_reliability_score"] = score
    result["empirical_reliability_tier"] = tier
    result["empirical_reliability_status"] = "validated_external_double_cold"
    result["reliability_recommendation"] = {
        "higher_evidence": "use_ranked_shortlist",
        "intermediate": "use_with_manual_review",
        "lower_evidence": "expand_search_or_add_known_seed",
    }[tier]
    return result


def enforce_reliability_policy(result: pd.DataFrame, policy: str) -> None:
    if policy == "annotate":
        return
    row = result.iloc[0]
    status = str(row["empirical_reliability_status"])
    tier = str(row["empirical_reliability_tier"])
    score = row["empirical_reliability_score"]
    calibrated = status == "validated_external_double_cold"
    accepted = {
        "require_calibrated": calibrated,
        "require_intermediate": calibrated and tier in {"intermediate", "higher_evidence"},
        "require_higher": calibrated and tier == "higher_evidence",
    }[policy]
    if not accepted:
        raise RuntimeError(
            f"Reliability policy {policy!r} rejected this query: "
            f"status={status}, tier={tier}, score={score}."
        )


def annotate_candidate_uncertainty(
    result: pd.DataFrame,
    candidate_ids: list[str],
    member_scores: np.ndarray,
    masked_ids: set[str],
    top_k: int,
    consensus_scores: np.ndarray | None = None,
) -> pd.DataFrame:
    candidate_to_index = {value: index for index, value in enumerate(candidate_ids)}
    member_ranks = np.stack(
        [rank_positions(scores, candidate_ids, masked_ids) for scores in member_scores]
    )
    effective_k = min(max(top_k, 1), len(candidate_ids) - len(masked_ids))
    rows = [candidate_to_index[value] for value in result["candidate_id"].astype(str)]
    result["ensemble_score_mean"] = [float(member_scores[:, index].mean()) for index in rows]
    result["ensemble_score_std"] = [float(member_scores[:, index].std()) for index in rows]
    result["ensemble_rank_mean"] = [float(np.nanmean(member_ranks[:, index])) for index in rows]
    result["ensemble_rank_std"] = [float(np.nanstd(member_ranks[:, index])) for index in rows]
    result["ensemble_topk_vote_fraction"] = [
        float(np.nanmean(member_ranks[:, index] <= effective_k)) for index in rows
    ]
    diagnostics = ensemble_query_diagnostics(
        member_scores,
        candidate_ids,
        masked_ids,
        top_k,
        consensus_scores=consensus_scores,
    )
    for key, value in diagnostics.items():
        result[key] = value
    return result


def nearest_reaction_similarity(
    query_smiles: str,
    positives_path: Path,
    exclude_reaction_id: str | None = None,
) -> tuple[str | None, float]:
    reaction_ids, features, _ = prepare_reaction_neighbor_index(positives_path)
    query = zero_shot_reaction_features(query_smiles)
    candidates = [
        (reaction_id, float(zero_shot_reaction_similarity(query, features[reaction_id])))
        for reaction_id in reaction_ids
        if reaction_id != exclude_reaction_id
    ]
    if not candidates:
        return None, float("nan")
    candidates.sort(key=lambda item: (-item[1], item[0]))
    return candidates[0]


def nearest_protein_similarity(
    query_feature: np.ndarray,
    current_protein_features: np.ndarray,
    current_protein_ids: list[str],
    exclude_protein_id: str | None = None,
) -> tuple[str | None, float]:
    keep = [index for index, value in enumerate(current_protein_ids) if value != exclude_protein_id]
    if not keep:
        return None, float("nan")
    similarities = current_protein_features[np.asarray(keep, dtype=np.int64)] @ query_feature
    local_order = np.lexsort(
        (np.asarray([current_protein_ids[index] for index in keep]), -similarities)
    )
    best_local = int(local_order[0])
    best_index = keep[best_local]
    return current_protein_ids[best_index], float(similarities[best_local])


def reaction_embedding_ensemble(
    models: list[nn.Module],
    reaction_features: np.ndarray,
    device: torch.device,
    auxiliary_reaction_features: np.ndarray | None = None,
) -> list[np.ndarray]:
    reaction_tensor = torch.as_tensor(reaction_features, dtype=torch.float32, device=device)
    auxiliary_tensor = (
        torch.as_tensor(auxiliary_reaction_features, dtype=torch.float32, device=device)
        if auxiliary_reaction_features is not None
        else None
    )
    if auxiliary_tensor is not None and len(auxiliary_tensor) != len(reaction_tensor):
        raise ValueError("Base and auxiliary reaction feature matrices differ in row count")
    embeddings: list[np.ndarray] = []
    with torch.no_grad():
        for model in models:
            embeddings.append(
                encode_model_reactions(model, reaction_tensor, auxiliary_tensor).cpu().numpy()
            )
    return embeddings

def tied_rank_percentile(scores: np.ndarray, ids: list[str]) -> np.ndarray:
    order = np.lexsort((np.asarray(ids), -scores))
    sorted_scores = scores[order]
    result = np.empty(len(scores), dtype=np.float32)
    if len(scores) == 1:
        result[0] = 1.0
        return result
    start = 0
    while start < len(scores):
        end = start + 1
        while end < len(scores) and sorted_scores[end] == sorted_scores[start]:
            end += 1
        average_position = (start + end - 1) / 2
        result[order[start:end]] = 1.0 - average_position / (len(scores) - 1)
        start = end
    return result


def prepare_reaction_neighbor_index(
    positives_path: Path,
) -> tuple[list[str], dict[str, dict[str, object]], dict[str, list[str]]]:
    positives = pd.read_csv(positives_path, sep="\t", dtype=str).fillna("")
    positives = positives[["rhea_id", "Entry", "smiles_seq"]].drop_duplicates(["rhea_id", "Entry"])
    reaction_rows = positives.groupby("rhea_id", as_index=False)["smiles_seq"].first().sort_values("rhea_id")
    reaction_ids = reaction_rows["rhea_id"].astype(str).tolist()
    features = {
        str(row.rhea_id): zero_shot_reaction_features(str(row.smiles_seq))
        for row in reaction_rows.itertuples(index=False)
    }
    enzymes_by_reaction = (
        positives.groupby("rhea_id")["Entry"]
        .apply(lambda values: sorted(set(values.astype(str))))
        .to_dict()
    )
    return reaction_ids, features, enzymes_by_reaction


def prepare_protein_reaction_index(positives_path: Path) -> dict[str, list[str]]:
    positives = pd.read_csv(positives_path, sep="\t", dtype=str).fillna("")
    positives = positives[["Entry", "rhea_id"]].drop_duplicates(["Entry", "rhea_id"])
    return (
        positives.groupby("Entry")["rhea_id"]
        .apply(lambda values: sorted(set(values.astype(str))))
        .to_dict()
    )


def reaction_neighbor_transfer_scores(
    reaction_smiles: str,
    protein_features: np.ndarray,
    protein_ids: list[str],
    positives_path: Path,
    topk_reactions: int,
    neighbor_index: tuple[list[str], dict[str, dict[str, object]], dict[str, list[str]]] | None = None,
) -> np.ndarray | None:
    reaction_ids, features, enzymes_by_reaction = (
        neighbor_index if neighbor_index is not None else prepare_reaction_neighbor_index(positives_path)
    )
    protein_to_row = {value: index for index, value in enumerate(protein_ids)}
    query = zero_shot_reaction_features(reaction_smiles)
    query_canonical = str(query["canonical"])
    neighbors: list[tuple[str, float]] = []
    for reaction_id in reaction_ids:
        candidate = features[reaction_id]
        candidate_canonical = str(candidate["canonical"])
        if query_canonical and candidate_canonical and query_canonical == candidate_canonical:
            continue
        neighbors.append((reaction_id, float(zero_shot_reaction_similarity(query, candidate))))
    neighbors.sort(key=lambda item: (-item[1], item[0]))
    selected = neighbors[:topk_reactions]
    if not selected:
        return None
    weights: dict[str, float] = {}
    for reaction_id, weight in selected:
        for entry in enzymes_by_reaction.get(reaction_id, []):
            if entry in protein_to_row:
                weights[entry] = max(weights.get(entry, 0.0), weight)
    if not weights:
        return None
    seed_ids = sorted(weights)
    seed_rows = np.asarray([protein_to_row[value] for value in seed_ids], dtype=np.int64)
    seed_weights = np.asarray([weights[value] for value in seed_ids], dtype=np.float32)
    return (protein_features @ protein_features[seed_rows].T * seed_weights[None, :]).max(axis=1)


def protein_neighbor_reaction_transfer_scores(
    query_protein_feature: np.ndarray,
    current_protein_features: np.ndarray,
    current_protein_ids: list[str],
    candidate_reaction_ids: list[str],
    reaction_embedding_sets: list[np.ndarray],
    positives_path: Path | None,
    topk_proteins: int,
    exclude_protein_id: str | None = None,
    protein_reaction_index: dict[str, list[str]] | None = None,
) -> np.ndarray | None:
    if protein_reaction_index is not None:
        reactions_by_protein = protein_reaction_index
    elif positives_path is not None:
        reactions_by_protein = prepare_protein_reaction_index(positives_path)
    else:
        raise ValueError("positives_path is required when protein_reaction_index is not supplied")
    protein_to_row = {value: index for index, value in enumerate(current_protein_ids)}
    reaction_to_row = {value: index for index, value in enumerate(candidate_reaction_ids)}
    annotated = [
        value
        for value in reactions_by_protein
        if value in protein_to_row and value != exclude_protein_id
    ]
    if not annotated:
        return None

    query = np.asarray(query_protein_feature, dtype=np.float32)
    query_norm = float(np.linalg.norm(query))
    if query_norm == 0:
        return None
    query = query / query_norm
    annotated_rows = np.asarray([protein_to_row[value] for value in annotated], dtype=np.int64)
    similarities = current_protein_features[annotated_rows] @ query
    order = np.lexsort((np.asarray(annotated), -similarities))

    reaction_weights: dict[str, float] = {}
    selected = 0
    for local_index in order:
        protein_id = annotated[int(local_index)]
        weight = float(similarities[int(local_index)])
        if weight <= 0:
            continue
        selected += 1
        for reaction_id in reactions_by_protein.get(protein_id, []):
            if reaction_id in reaction_to_row:
                reaction_weights[reaction_id] = max(reaction_weights.get(reaction_id, 0.0), weight)
        if selected >= topk_proteins:
            break
    if not reaction_weights:
        return None

    seed_ids = sorted(reaction_weights)
    seed_rows = np.asarray([reaction_to_row[value] for value in seed_ids], dtype=np.int64)
    seed_weights = np.asarray([reaction_weights[value] for value in seed_ids], dtype=np.float32)
    total = np.zeros(len(candidate_reaction_ids), dtype=np.float32)
    for embeddings in reaction_embedding_sets:
        total += (embeddings @ embeddings[seed_rows].T * seed_weights[None, :]).max(axis=1)
    return total / len(reaction_embedding_sets)


def apply_automatic_few_shot_policy(
    mode: str,
    seed_scores: np.ndarray | None,
    route_settings: dict[str, object] | None,
    hybrid_direct_weight: float,
) -> tuple[str, float]:
    """Resolve production ``auto`` few-shot without changing explicit retrieval modes.

    The policy is carried by the production route manifest so experiments that call
    ``choose_retrieval_scores(..., mode="auto")`` retain their historical semantics.
    Explicit ``seed``/``hybrid``/``direct`` requests are never rewritten here.
    """
    if mode != "auto" or seed_scores is None:
        return mode, hybrid_direct_weight
    policy = dict((route_settings or {}).get("few_shot") or {})
    retrieval = str(policy.get("retrieval") or "seed")
    if retrieval == "seed":
        return mode, hybrid_direct_weight
    if retrieval != "hybrid":
        raise ValueError(f"Unsupported automatic few-shot retrieval policy: {retrieval!r}")
    direct_weight = float(policy.get("direct_weight", hybrid_direct_weight))
    if not 0.0 < direct_weight < 1.0:
        raise ValueError("Automatic few-shot hybrid direct weight must be strictly within (0, 1)")
    return "hybrid", direct_weight


def choose_retrieval_scores(
    direct_scores: np.ndarray,
    seed_scores: np.ndarray | None,
    candidate_ids: list[str],
    mode: str,
    neighbor_scores: np.ndarray | None = None,
    hybrid_direct_weight: float = 0.5,
) -> tuple[np.ndarray, str]:
    if not 0 <= hybrid_direct_weight <= 1:
        raise ValueError("hybrid_direct_weight must be within [0, 1]")
    if mode == "auto":
        if seed_scores is not None:
            return seed_scores, "seed"
        if neighbor_scores is not None:
            score = (
                hybrid_direct_weight * tied_rank_percentile(direct_scores, candidate_ids)
                + (1 - hybrid_direct_weight) * tied_rank_percentile(neighbor_scores, candidate_ids)
            )
            return score, f"neighbor_hybrid_direct_{hybrid_direct_weight:g}"
        return direct_scores, "direct"
    if mode == "direct":
        return direct_scores, "direct"
    if mode == "seed":
        if seed_scores is None:
            raise ValueError("seed retrieval requires known associations present in the candidate library")
        return seed_scores, "seed"
    if mode == "hybrid":
        if seed_scores is None:
            raise ValueError("hybrid retrieval requires known associations present in the candidate library")
        score = (
            hybrid_direct_weight * tied_rank_percentile(direct_scores, candidate_ids)
            + (1 - hybrid_direct_weight) * tied_rank_percentile(seed_scores, candidate_ids)
        )
        return score, f"hybrid_direct_{hybrid_direct_weight:g}"
    if mode == "neighbor":
        if neighbor_scores is None:
            raise ValueError("neighbor retrieval is unavailable for this query")
        return neighbor_scores, "neighbor"
    if mode == "neighbor_hybrid":
        if neighbor_scores is None:
            raise ValueError("neighbor_hybrid retrieval is unavailable for this query")
        score = (
            hybrid_direct_weight * tied_rank_percentile(direct_scores, candidate_ids)
            + (1 - hybrid_direct_weight) * tied_rank_percentile(neighbor_scores, candidate_ids)
        )
        return score, f"neighbor_hybrid_direct_{hybrid_direct_weight:g}"
    raise ValueError(f"Unsupported retrieval mode: {mode}")


def resolve_ranking_objective(top_k: int, objective: str) -> str:
    if objective != "auto":
        return objective
    if top_k <= 3:
        return "top3"
    if top_k <= 10:
        return "top10"
    return "top20"


def sort_scores(ids: list[str], scores: np.ndarray, masked_ids: set[str], top_k: int) -> pd.DataFrame:
    adjusted = scores.astype(np.float64, copy=True)
    id_to_index = {value: index for index, value in enumerate(ids)}
    for value in masked_ids:
        index = id_to_index.get(value)
        if index is not None:
            adjusted[index] = -np.inf
    order = np.lexsort((np.asarray(ids), -adjusted))
    rows = []
    for index in order:
        if not np.isfinite(adjusted[index]):
            continue
        rows.append(
            {
                "rank": len(rows) + 1,
                "candidate_id": ids[index],
                "score": float(adjusted[index]),
                "selection_source": "primary",
            }
        )
        if len(rows) >= top_k:
            break
    return pd.DataFrame(rows)


def sort_scores_with_cage_rescue(
    ids: list[str],
    scores: np.ndarray,
    masked_ids: set[str],
    top_k: int,
    reaction_id: str | None,
    cage_scores_path: Path,
    rescue_slots: int,
) -> pd.DataFrame:
    if not reaction_id or rescue_slots <= 0 or top_k < 20 or not cage_scores_path.exists():
        return sort_scores(ids, scores, masked_ids, top_k)

    cage = pd.read_csv(cage_scores_path, dtype=str).fillna("")
    cage = cage[cage["reaction_id"].astype(str).eq(str(reaction_id))].copy()
    if cage.empty:
        return sort_scores(ids, scores, masked_ids, top_k)
    cage["cage_score"] = pd.to_numeric(cage["cage_score"], errors="coerce")
    cage = cage[np.isfinite(cage["cage_score"])].copy()
    if cage.empty:
        return sort_scores(ids, scores, masked_ids, top_k)

    adjusted = scores.astype(np.float64, copy=True)
    id_to_index = {value: index for index, value in enumerate(ids)}
    for value in masked_ids:
        index = id_to_index.get(value)
        if index is not None:
            adjusted[index] = -np.inf
    base_order = [
        int(index)
        for index in np.lexsort((np.asarray(ids), -adjusted))
        if np.isfinite(adjusted[index])
    ]
    rescue_slots = min(rescue_slots, top_k)
    primary_count = top_k - rescue_slots
    selected = base_order[:primary_count]
    selected_set = set(selected)
    source = {index: "primary" for index in selected}

    rescue_candidates: list[tuple[float, float, str, int]] = []
    for row in cage.itertuples(index=False):
        candidate_id = str(row.uniprot_id)
        index = id_to_index.get(candidate_id)
        if index is None or index in selected_set or not np.isfinite(adjusted[index]):
            continue
        rescue_candidates.append(
            (-float(row.cage_score), -float(adjusted[index]), candidate_id, int(index))
        )
    rescue_candidates.sort()
    for _, _, _, index in rescue_candidates:
        if len(selected) >= top_k:
            break
        if index in selected_set:
            continue
        selected.append(index)
        selected_set.add(index)
        source[index] = "cage_rescue"

    for index in base_order:
        if len(selected) >= top_k:
            break
        if index in selected_set:
            continue
        selected.append(index)
        selected_set.add(index)
        source[index] = "primary_fill"

    return pd.DataFrame(
        [
            {
                "rank": rank,
                "candidate_id": ids[index],
                "score": float(adjusted[index]),
                "selection_source": source[index],
            }
            for rank, index in enumerate(selected, start=1)
        ]
    )


@lru_cache(maxsize=4)
def _r2e_binary_drfp_router_assets(
    feature_dir_text: str,
    training_pairs_text: str,
) -> tuple[dict[str, object], np.ndarray, list[str], dict[str, int], np.ndarray, np.ndarray, list[str]]:
    """Load the exact train-only binary-DRFP router index used by clean evaluations."""
    feature_dir = Path(feature_dir_text).resolve()
    training_pairs = Path(training_pairs_text).resolve()
    schema = load_feature_schema(feature_dir)
    features, reaction_ids = load_registered_reaction_feature_library(feature_dir, schema)
    drfp_dim = int(schema.get("drfp_dimension", 2048))
    if drfp_dim <= 0 or features.shape[1] < drfp_dim:
        raise ValueError("Invalid DRFP block for R2E similarity router")
    pairs = pd.read_csv(training_pairs, dtype=str).fillna("")
    if "reaction_id" not in pairs.columns:
        raise ValueError("R2E similarity-router training pairs require reaction_id")
    train_ids = sorted(set(pairs["reaction_id"].astype(str)))
    index = {value: row for row, value in enumerate(reaction_ids)}
    missing = sorted(set(train_ids) - set(index))
    if missing:
        raise ValueError(f"R2E similarity-router feature library misses training reactions: {missing[:5]}")
    train_rows = np.asarray([index[value] for value in train_ids], dtype=np.int64)
    train_binary = (features[train_rows, :drfp_dim] > 0).astype(np.float32, copy=False)
    train_counts = train_binary.sum(axis=1, dtype=np.float32)
    return schema, features, reaction_ids, index, train_binary, train_counts, train_ids


def exact_max_train_binary_drfp_tanimoto(
    *,
    reaction_id: str | None,
    reaction_smiles: str | None,
    feature_dir: Path,
    training_pairs: Path,
    registered_reactions_csv: Path | None = None,
    feature_cache_dir: Path | None = DEFAULT_FEATURE_CACHE,
    failure_policy: str = "warn",
) -> tuple[str | None, float]:
    """Return the exact max binary-DRFP Tanimoto against full-clean training reactions.

    This intentionally matches ``prepare_broad_rhea_difficulty_slices`` rather than
    the older heuristic ``query_nearest_library_similarity`` diagnostic.
    """
    schema, features, reaction_ids, index, train_binary, train_counts, train_ids = (
        _r2e_binary_drfp_router_assets(str(feature_dir.resolve()), str(training_pairs.resolve()))
    )
    drfp_dim = int(schema.get("drfp_dimension", 2048))
    query_feature: np.ndarray | None = None
    if reaction_id and str(reaction_id) in index:
        query_feature = np.asarray(features[index[str(reaction_id)]], dtype=np.float32)
    query_smiles = str(reaction_smiles or "").strip()
    if query_feature is None and not query_smiles and reaction_id and registered_reactions_csv:
        path = registered_reactions_csv.resolve()
        if path.is_file():
            registered = pd.read_csv(path, dtype=str).fillna("")
            if {"reaction_id", "reaction_smiles"} <= set(registered.columns):
                match = registered[registered["reaction_id"].astype(str).eq(str(reaction_id))]
                if not match.empty:
                    query_smiles = str(match.iloc[0]["reaction_smiles"])
    if query_feature is None:
        if not query_smiles:
            raise ValueError("R2E similarity router cannot resolve a reaction feature")
        query_feature, _ = encode_reaction_with_audit(
            query_smiles,
            schema,
            failure_policy=failure_policy,
            cache_dir=feature_cache_dir,
        )
    query_binary = (np.asarray(query_feature[:drfp_dim], dtype=np.float32) > 0).astype(np.float32)
    query_count = float(query_binary.sum())
    intersections = train_binary @ query_binary
    unions = train_counts + query_count - intersections
    similarities = np.divide(
        intersections,
        unions,
        out=np.zeros_like(intersections, dtype=np.float32),
        where=unions > 0,
    )
    best = float(np.max(similarities)) if len(similarities) else 0.0
    tied = np.flatnonzero(np.isclose(similarities, best, rtol=0.0, atol=1e-7))
    nearest = min((train_ids[int(row)] for row in tied), default=None)
    return nearest, best


def _resolve_r2e_similarity_model_route(
    args: argparse.Namespace,
    deployment_route,
) -> tuple[object, Path, dict[str, object]]:
    """Apply the confirmed low-similarity EnzGFM model route when strictly eligible."""
    settings = dict(deployment_route.settings or {})
    spec = dict(settings.get("similarity_model_router") or {})
    audit: dict[str, object] = {
        "status": "not_configured" if not spec else "ineligible",
        "selected": "primary",
        "max_train_drfp_tanimoto": None,
        "nearest_train_reaction_id": None,
    }
    if not spec or deployment_route.secondary_deployment is None:
        return deployment_route, args.protein_dir.resolve(), audit
    primary_protein_dir = (ROOT / str(spec["primary_protein_dir"])).resolve()
    secondary_protein_dir = (ROOT / str(spec["secondary_protein_dir"])).resolve()
    eligible = (
        args.protein_dir.resolve() == primary_protein_dir
        and args.model_dir is None
        and args.dual_tower_dir is None
        and not args.internal_expert_override
        and not (args.known_enzyme_ids or [])
        and not (args.mask_enzyme_ids or [])
        and not (args.candidate_ids or [])
        and args.external_enzymes_csv is None
        and str(args.enzyme_taxonomy_scope) == "all"
        and str(args.retrieval_mode) == "auto"
    )
    if not eligible:
        fallback = spec.get("ineligible_fallback_deployment")
        if fallback:
            deployment_route = replace(
                deployment_route,
                route_id=f"{deployment_route.route_id}+legacy-scope-fallback",
                deployment=(ROOT / str(fallback)).resolve(),
                secondary_deployment=None,
            )
            audit["selected"] = "legacy_scope_fallback"
        return deployment_route, args.protein_dir.resolve(), audit
    nearest, similarity = exact_max_train_binary_drfp_tanimoto(
        reaction_id=args.reaction_id,
        reaction_smiles=args.reaction_smiles,
        feature_dir=(ROOT / str(spec["feature_dir"])).resolve(),
        training_pairs=(ROOT / str(spec["training_pairs"])).resolve(),
        registered_reactions_csv=args.registered_reactions_csv,
        feature_cache_dir=args.feature_cache_dir,
        failure_policy=args.reaction_feature_policy,
    )
    threshold = float(spec["threshold"])
    use_secondary = similarity < threshold
    audit = {
        "status": "applied",
        "threshold": threshold,
        "selected": "secondary" if use_secondary else "primary",
        "max_train_drfp_tanimoto": similarity,
        "nearest_train_reaction_id": nearest,
        "feature": "max_train_binary_drfp_tanimoto",
        "labels_used": False,
    }
    if not use_secondary:
        return replace(
            deployment_route,
            route_id=f"{deployment_route.route_id}+sim-ge-{threshold:g}-primary",
        ), primary_protein_dir, audit
    candidate_bundle_version = str(
        spec.get("secondary_model_bundle_version") or deployment_route.model_bundle_version
    )
    return replace(
        deployment_route,
        route_id=f"{deployment_route.route_id}+sim-lt-{threshold:g}-enzgfm",
        deployment=deployment_route.secondary_deployment,
        model_bundle_version=candidate_bundle_version,
    ), secondary_protein_dir, audit


def _r2e_lambdarank_fusion_spec(
    args: argparse.Namespace,
    deployment_route,
) -> dict[str, object] | None:
    """Return the frozen learned-fusion spec only for its confirmed production scope."""
    settings = dict(deployment_route.settings or {})
    spec = dict(settings.get("lambdarank_fusion") or {})
    if not spec or deployment_route.secondary_deployment is None:
        return None
    primary_protein_dir = (ROOT / str(spec["primary_protein_dir"])).resolve()
    registered_dir = args.registered_protein_dir.resolve() if args.registered_protein_dir else None
    eligible = (
        args.protein_dir.resolve() == primary_protein_dir
        and registered_dir == primary_protein_dir
        and args.model_dir is None
        and args.dual_tower_dir is None
        and not args.internal_expert_override
        and not (args.known_enzyme_ids or [])
        and not (args.mask_enzyme_ids or [])
        and not (args.candidate_ids or [])
        and args.external_enzymes_csv is None
        and str(args.enzyme_taxonomy_scope) == "all"
        and str(args.retrieval_mode) == "auto"
        and str(deployment_route.retrieval) == "direct"
    )
    return spec if eligible else None


def model_bundle_root(model_dir: Path) -> Path:
    """Return the human/audit-facing bundle root for a checkpoint directory."""
    resolved = model_dir.resolve()
    return resolved.parent if resolved.name == "models" else resolved


def rank_enzymes(args: argparse.Namespace) -> pd.DataFrame:
    device = torch.device(args.device)
    ranking_objective = resolve_ranking_objective(args.top_k, args.ranking_objective)
    current_reaction_ids_from_labels = set(
        pd.read_csv(args.positives, sep="\t", usecols=["rhea_id"], dtype=str)["rhea_id"].astype(str)
    )
    query_is_current_reaction = bool(
        args.reaction_id and args.reaction_id in current_reaction_ids_from_labels
    )
    deployment_route = resolve_route(
        direction="reaction_to_enzyme",
        objective=ranking_objective,
        is_current=query_is_current_reaction,
        has_seed=bool(args.known_enzyme_ids),
        manual_override=(
            args.dual_tower_dir is not None
            or (args.model_dir is not None and not args.internal_expert_override)
        ),
        temporary_candidate_extension=bool(args.external_enzymes_csv),
        enzyme_taxonomy_scope=args.enzyme_taxonomy_scope,
        manifest_path=args.route_manifest,
    )
    routed_protein_dir = args.protein_dir.resolve()
    lambdarank_spec = (
        _r2e_lambdarank_fusion_spec(args, deployment_route)
        if args.dual_tower_dir is None and args.model_dir is None
        else None
    )
    model_router_audit: dict[str, object] = {
        "status": "pending" if lambdarank_spec is not None else ("manual_override" if args.dual_tower_dir is not None else "not_configured"),
        "selected": "lambdarank_fusion" if lambdarank_spec is not None else "primary",
        "max_train_drfp_tanimoto": None,
        "nearest_train_reaction_id": None,
    }
    if args.dual_tower_dir is None and args.model_dir is None and lambdarank_spec is None:
        deployment_route, routed_protein_dir, model_router_audit = _resolve_r2e_similarity_model_route(
            args, deployment_route
        )
    dual_tower_dir = (
        args.dual_tower_dir.resolve()
        if args.dual_tower_dir is not None
        else deployment_route.deployment
    )
    schema = load_feature_schema(dual_tower_dir)
    model_dir = args.model_dir.resolve() if args.model_dir else dual_tower_dir / "models"
    models = load_models_runtime(model_dir, args.scope, device)
    protein_features, protein_ids = load_protein_library(routed_protein_dir)
    base_protein_universe_unchanged = True
    registered_protein_ids: set[str] = set()
    if args.registered_protein_dir and args.registered_protein_dir.exists():
        registered_features, registered_ids = load_protein_library(args.registered_protein_dir.resolve())
        existing = set(protein_ids)
        keep = [index for index, value in enumerate(registered_ids) if value not in existing]
        if keep:
            base_protein_universe_unchanged = False
            protein_features = np.concatenate([protein_features, registered_features[keep]], axis=0)
            appended = [registered_ids[index] for index in keep]
            protein_ids.extend(appended)
            registered_protein_ids.update(appended)

    external = load_external_enzyme_rows(args.external_enzymes_csv)
    if not external.empty:
        external_features = encode_external_enzymes(external, args.device, args.esmc_model)
        existing = set(protein_ids)
        keep = [index for index, value in enumerate(external["enzyme_id"].astype(str)) if value not in existing]
        if keep:
            base_protein_universe_unchanged = False
            protein_features = np.concatenate([protein_features, external_features[keep]], axis=0)
            protein_ids.extend(external.iloc[keep]["enzyme_id"].astype(str).tolist())

    enzyme_taxonomy_scope = validate_scope(args.enzyme_taxonomy_scope)
    validate_seed_scope(
        args.known_enzyme_ids or [],
        enzyme_taxonomy_scope,
        registry_path=args.taxonomy_scope_registry.resolve(),
    )
    taxonomy_keep, taxonomy_audit = filter_candidate_ids(
        protein_ids,
        enzyme_taxonomy_scope,
        registry_path=args.taxonomy_scope_registry.resolve(),
    )
    if enzyme_taxonomy_scope != "all":
        base_protein_universe_unchanged = False
        if not taxonomy_keep:
            raise ValueError(f"No enzyme candidates remain for taxonomy scope {enzyme_taxonomy_scope!r}")
        protein_features = protein_features[np.asarray(taxonomy_keep, dtype=np.int64)]
        protein_ids = [protein_ids[index] for index in taxonomy_keep]
        registered_protein_ids.intersection_update(protein_ids)

    candidate_keep, candidate_subset_audit = candidate_subset_indices(protein_ids, args.candidate_ids)
    if candidate_subset_audit["applied"]:
        base_protein_universe_unchanged = False
        protein_features = protein_features[np.asarray(candidate_keep, dtype=np.int64)]
        protein_ids = [protein_ids[index] for index in candidate_keep]
        registered_protein_ids.intersection_update(protein_ids)

    reaction_library, reaction_ids = load_reaction_library(dual_tower_dir, schema)
    requires_auxiliary = models_require_auxiliary_reaction_features(models)
    auxiliary_reaction_library = (
        load_auxiliary_reaction_library(dual_tower_dir, reaction_ids)
        if requires_auxiliary
        else None
    )
    current_reaction_ids = set(reaction_ids)
    registered_reactions = (
        load_external_reaction_rows(args.registered_reactions_csv)
        if args.registered_reactions_csv and args.registered_reactions_csv.exists()
        else pd.DataFrame(columns=["reaction_id", "reaction_smiles"])
    )
    if args.reaction_id:
        if args.reaction_id in reaction_ids:
            query_row = reaction_ids.index(args.reaction_id)
            query_feature = reaction_library[query_row]
            query_auxiliary_feature = (
                auxiliary_reaction_library[query_row]
                if auxiliary_reaction_library is not None
                else None
            )
            positives = pd.read_csv(args.positives, sep="\t", dtype=str).fillna("")
            matched = positives[positives["rhea_id"].astype(str).eq(args.reaction_id)]
            if not matched.empty:
                query_smiles = str(matched.iloc[0]["smiles_seq"])
            else:
                registered_match = registered_reactions[
                    registered_reactions["reaction_id"].astype(str).eq(args.reaction_id)
                ]
                if registered_match.empty:
                    raise ValueError(f"No reaction SMILES found for {args.reaction_id}")
                query_smiles = str(registered_match.iloc[0]["reaction_smiles"])
            reaction_input_audit = replace(
                initial_reaction_audit(query_smiles, canonical_or_raw_reaction(query_smiles)),
                status="precomputed",
                drfp_status="precomputed",
            )
        else:
            # In R2E the registered reaction table is a query-resolution source, not
            # a candidate universe. Encoding every registered reaction here made a
            # broad database catastrophically expensive while contributing nothing
            # to enzyme ranking. Encode only the requested external reaction.
            registered_match = registered_reactions[
                registered_reactions["reaction_id"].astype(str).eq(args.reaction_id)
            ]
            if registered_match.empty:
                raise ValueError(
                    f"Unknown reaction ID: {args.reaction_id}; provide --reaction-smiles for an external query."
                )
            query_smiles = str(registered_match.iloc[0]["reaction_smiles"])
            query_feature, reaction_input_audit = encode_reaction_with_audit(
                query_smiles,
                schema,
                failure_policy=args.reaction_feature_policy,
                cache_dir=args.feature_cache_dir,
            )
            query_auxiliary_feature = (
                encode_exact_horizyn_reactions(
                    [query_smiles],
                    dual_tower_dir,
                    device,
                    query_feature[None, :],
                )[0]
                if requires_auxiliary
                else None
            )
        query_id = args.reaction_id
    else:
        if not args.reaction_smiles:
            raise ValueError("Provide --reaction-id or --reaction-smiles.")
        query_smiles = args.reaction_smiles
        query_feature, reaction_input_audit = encode_reaction_with_audit(
            query_smiles,
            schema,
            failure_policy=args.reaction_feature_policy,
            cache_dir=args.feature_cache_dir,
        )
        query_auxiliary_feature = (
            encode_exact_horizyn_reactions(
                [query_smiles],
                dual_tower_dir,
                device,
                query_feature[None, :],
            )[0]
            if requires_auxiliary
            else None
        )
        query_id = args.query_id or "external_reaction_query"

    if base_protein_universe_unchanged:
        direct_member_scores = ensemble_similarity_members_cached_base_proteins(
            model_dir=model_dir,
            scope=args.scope,
            device=device,
            protein_dir=routed_protein_dir,
            reaction_features=query_feature[None, :],
            auxiliary_reaction_features=(
                query_auxiliary_feature[None, :]
                if query_auxiliary_feature is not None
                else None
            ),
        )[:, 0, :]
    else:
        direct_member_scores = ensemble_similarity_members(
            models,
            protein_features,
            query_feature[None, :],
            device,
            query_auxiliary_feature[None, :] if query_auxiliary_feature is not None else None,
        )[:, 0, :]
    direct_scores = direct_member_scores.mean(axis=0)
    lambdarank_runtime = None
    secondary_member_scores = None
    secondary_bundle_for_audit: Path | None = None
    if lambdarank_spec is not None:
        from projects.active.terpene_screening.r2e_lambdarank_runtime import fuse_r2e_scores

        secondary_protein_dir = (ROOT / str(lambdarank_spec["secondary_protein_dir"])).resolve()
        _secondary_features, secondary_ids = load_protein_library(secondary_protein_dir)
        if secondary_ids != protein_ids:
            raise RuntimeError("Confirmed R2E LambdaRank source candidate IDs/order differ at runtime")
        secondary_bundle = deployment_route.secondary_deployment
        if secondary_bundle is None:
            raise RuntimeError("Confirmed R2E LambdaRank route has no secondary deployment")
        secondary_bundle_for_audit = secondary_bundle.resolve()
        secondary_member_scores = ensemble_similarity_members_cached_base_proteins(
            model_dir=secondary_bundle / "models",
            scope=args.scope,
            device=device,
            protein_dir=secondary_protein_dir,
            reaction_features=query_feature[None, :],
            auxiliary_reaction_features=None,
        )[:, 0, :]
        nearest_train, max_train_similarity = exact_max_train_binary_drfp_tanimoto(
            reaction_id=args.reaction_id,
            reaction_smiles=args.reaction_smiles,
            feature_dir=(ROOT / str(lambdarank_spec["feature_dir"])).resolve(),
            training_pairs=(ROOT / str(lambdarank_spec["training_pairs"])).resolve(),
            registered_reactions_csv=args.registered_reactions_csv,
            feature_cache_dir=args.feature_cache_dir,
            failure_policy=args.reaction_feature_policy,
        )
        lambdarank_runtime = fuse_r2e_scores(
            direct_scores,
            secondary_member_scores.mean(axis=0),
            protein_ids,
            similarity=max_train_similarity,
            threshold=float(lambdarank_spec["threshold"]),
            ranker_bundle=(ROOT / str(lambdarank_spec["ranker_bundle"])).resolve(),
            ranker_sha256=str(lambdarank_spec["ranker_sha256"]),
            expected_pool_k=int(lambdarank_spec["pool_k"]),
            expected_prefix_k=int(lambdarank_spec["prefix_k"]),
        )
        model_router_audit = {
            "status": "applied",
            "selected": "lambdarank_fusion",
            "threshold": float(lambdarank_spec["threshold"]),
            "max_train_drfp_tanimoto": float(max_train_similarity),
            "nearest_train_reaction_id": nearest_train,
            "feature": "max_train_binary_drfp_tanimoto",
            "labels_used": False,
            "fusion_config_id": str(lambdarank_spec["config_id"]),
            "fusion_pool_k": int(lambdarank_spec["pool_k"]),
            "fusion_prefix_k": int(lambdarank_spec["prefix_k"]),
            "fusion_union_size": int(lambdarank_runtime.union_size),
            "fusion_fallback": "secondary" if lambdarank_runtime.fallback_is_secondary else "primary",
        }
    protein_to_row = {value: index for index, value in enumerate(protein_ids)}
    seed_ids = [value for value in (args.known_enzyme_ids or []) if value in protein_to_row]
    seed_scores = None
    if seed_ids:
        seed_rows = np.asarray([protein_to_row[value] for value in seed_ids], dtype=np.int64)
        seed_scores = (protein_features @ protein_features[seed_rows].T).max(axis=1)
    retrieval_mode = args.retrieval_mode
    hybrid_direct_weight = args.hybrid_direct_weight
    if retrieval_mode == "auto" and seed_scores is None:
        retrieval_mode = "direct"
    retrieval_mode, hybrid_direct_weight = apply_automatic_few_shot_policy(
        retrieval_mode,
        seed_scores,
        deployment_route.settings,
        hybrid_direct_weight,
    )
    neighbor_scores = None
    if retrieval_mode in {"neighbor", "neighbor_hybrid"}:
        neighbor_scores = reaction_neighbor_transfer_scores(
            query_smiles,
            protein_features,
            protein_ids,
            args.positives.resolve(),
            args.topk_neighbor_reactions,
        )
    if lambdarank_runtime is not None:
        scores = lambdarank_runtime.priority_scores
        score_source = "r2e_lambdarank_fusion_v1"
        routed_member_scores = np.concatenate(
            [direct_member_scores, secondary_member_scores], axis=0
        ).astype(np.float32, copy=False)
    else:
        scores, score_source = choose_retrieval_scores(
            direct_scores,
            seed_scores,
            protein_ids,
            retrieval_mode,
            neighbor_scores=neighbor_scores,
            hybrid_direct_weight=hybrid_direct_weight,
        )
        routed_member_scores = route_member_scores(
            direct_member_scores,
            seed_scores,
            protein_ids,
            retrieval_mode,
            neighbor_scores,
            hybrid_direct_weight,
        )
    masked_enzyme_ids = set(args.known_enzyme_ids or []) | set(args.mask_enzyme_ids or [])
    if lambdarank_runtime is not None:
        # The frozen confirmation covers the learned prefix followed by exact router
        # fallback. Do not inject the historical CAGE-rescue slots into this scope.
        result = sort_scores(protein_ids, scores, masked_enzyme_ids, args.top_k)
        result["selection_source"] = "r2e_lambdarank_fusion_v1"
    elif args.known_enzyme_ids:
        result = sort_scores(protein_ids, scores, masked_enzyme_ids, args.top_k)
    else:
        result = sort_scores_with_cage_rescue(
            protein_ids,
            scores,
            masked_enzyme_ids,
            args.top_k,
            args.reaction_id,
            args.cage_scores.resolve(),
            args.cage_rescue_slots,
        )
    result = annotate_candidate_uncertainty(
        result, protein_ids, routed_member_scores, masked_enzyme_ids, args.top_k,
        consensus_scores=scores if lambdarank_runtime is not None else None,
    )
    if lambdarank_runtime is not None:
        index = {value: row for row, value in enumerate(protein_ids)}
        result_rows = np.asarray([index[value] for value in result["candidate_id"].astype(str)], dtype=np.int64)
        secondary_scores = secondary_member_scores.mean(axis=0)
        learned_map = {int(row): float(score) for row, score in zip(lambdarank_runtime.learned_rows, lambdarank_runtime.learned_scores, strict=True)}
        result["fusion_primary_score"] = [float(direct_scores[row]) for row in result_rows]
        result["fusion_secondary_score"] = [float(secondary_scores[row]) for row in result_rows]
        result["fusion_lambdarank_score"] = [learned_map.get(int(row), float("nan")) for row in result_rows]
        result["fusion_primary_rank"] = [int(lambdarank_runtime.primary_ranks[row]) for row in result_rows]
        result["fusion_secondary_rank"] = [int(lambdarank_runtime.secondary_ranks[row]) for row in result_rows]
        result["fusion_fallback_rank"] = [int(lambdarank_runtime.fallback_ranks[row]) for row in result_rows]
        result["fusion_fallback_model"] = "secondary" if lambdarank_runtime.fallback_is_secondary else "primary"
        result["fusion_union_size"] = int(lambdarank_runtime.union_size)
        result["fusion_prefix_size"] = int(lambdarank_runtime.prefix_size)
    nearest_id, nearest_similarity = nearest_reaction_similarity(
        query_smiles,
        args.positives.resolve(),
        exclude_reaction_id=args.reaction_id if query_is_current_reaction else None,
    )
    result.insert(0, "query_id", query_id)
    result.insert(1, "direction", "reaction_to_enzyme")
    result.insert(2, "score_source", score_source)
    result.insert(3, "ranking_objective", ranking_objective)
    result.insert(4, "model_directory", str(model_bundle_root(model_dir)))
    result.insert(5, "model_feature_directory", str(dual_tower_dir))
    result["secondary_model_directory"] = (
        str(secondary_bundle_for_audit) if secondary_bundle_for_audit is not None else None
    )
    result.insert(6, "query_nearest_library_id", nearest_id)
    result.insert(6, "query_nearest_library_similarity", nearest_similarity)
    result.insert(7, "model_router_status", str(model_router_audit.get("status", "not_configured")))
    result.insert(8, "model_router_selected", str(model_router_audit.get("selected", "primary")))
    result.insert(9, "model_router_max_train_drfp_tanimoto", model_router_audit.get("max_train_drfp_tanimoto"))
    result.insert(10, "model_router_nearest_train_reaction_id", model_router_audit.get("nearest_train_reaction_id"))
    result.insert(11, "query_is_current_entity", query_is_current_reaction)
    result["taxonomy_scope_version"] = TAXONOMY_SCOPE_VERSION
    result["enzyme_taxonomy_scope"] = enzyme_taxonomy_scope
    result["taxonomy_scope_mode"] = "candidate_filter" if enzyme_taxonomy_scope != "all" else "unrestricted"
    result["candidate_universe_pre_taxonomy_size"] = int(taxonomy_audit["pre_filter_size"])
    result["candidate_universe_post_taxonomy_size"] = int(taxonomy_audit["post_filter_size"])
    result["taxonomy_eukaryote_count"] = int(taxonomy_audit["eukaryote_count"])
    result["taxonomy_prokaryote_count"] = int(taxonomy_audit["prokaryote_count"])
    result["taxonomy_other_count"] = int(taxonomy_audit["other_count"])
    result["taxonomy_unknown_count"] = int(taxonomy_audit["unknown_count"])
    result["taxonomy_excluded_count"] = int(taxonomy_audit["excluded_count"])
    candidate_taxonomy = {
        candidate_id: taxonomy_record(candidate_id, registry_path=args.taxonomy_scope_registry.resolve())
        for candidate_id in result["candidate_id"].astype(str)
    }
    result["candidate_taxonomy_scope"] = result["candidate_id"].astype(str).map(
        lambda value: candidate_taxonomy[value].taxonomy_scope
    )
    result["candidate_kingdom"] = result["candidate_id"].astype(str).map(
        lambda value: candidate_taxonomy[value].kingdom
    )
    result["candidate_taxonomy_source"] = result["candidate_id"].astype(str).map(
        lambda value: candidate_taxonomy[value].taxonomy_source
    )
    external_candidate_ids = registered_protein_ids | set(external["enzyme_id"].astype(str))
    result["is_external_candidate"] = result["candidate_id"].isin(external_candidate_ids)
    reliability_applicable = (
        (not query_is_current_reaction)
        and (not seed_ids)
        and (not args.mask_enzyme_ids)
        and enzyme_taxonomy_scope == "all"
        and not args.candidate_ids
        and args.retrieval_mode == "auto"
        and args.model_dir is None
        and args.dual_tower_dir is None
        and str(model_router_audit.get("status")) != "applied"
    )
    if query_is_current_reaction:
        reliability_reason = "not_applicable_current_entity"
    elif seed_ids:
        reliability_reason = "not_applicable_few_shot"
    elif args.mask_enzyme_ids:
        reliability_reason = (
            "not_applicable_known_associations_masked"
            if args.mask_semantics == "novelty_filter"
            else "not_applicable_output_separation_mask"
        )
    elif enzyme_taxonomy_scope != "all":
        reliability_reason = "not_applicable_taxonomy_restricted"
    elif args.candidate_ids:
        reliability_reason = "not_applicable_candidate_subset"
    elif str(model_router_audit.get("status")) == "applied":
        reliability_reason = "not_applicable_similarity_model_router"
    elif args.retrieval_mode != "auto" or args.model_dir is not None or args.dual_tower_dir is not None:
        reliability_reason = "not_applicable_manual_override"
    else:
        reliability_reason = "not_applicable"
    route = resolve_route(
        direction="reaction_to_enzyme",
        objective=ranking_objective,
        is_current=query_is_current_reaction,
        has_seed=bool(seed_ids),
        manual_override=(
            args.retrieval_mode != "auto"
            or (args.model_dir is not None and not args.internal_expert_override)
            or args.dual_tower_dir is not None
        ),
        temporary_candidate_extension=not external.empty,
        enzyme_taxonomy_scope=enzyme_taxonomy_scope,
        manifest_path=args.route_manifest,
    )
    result = apply_candidate_subset_metadata(result, candidate_subset_audit)
    result = apply_route_provenance(
        result,
        route,
        candidate_ids=protein_ids,
        registry_version=registry_version(args.registered_protein_dir.resolve().parent),
    )
    for column, value in reaction_input_audit.as_columns().items():
        result[column] = value
    return apply_empirical_reliability(
        result,
        "reaction_to_enzyme",
        ranking_objective,
        args.calibrators.resolve(),
        reliability_applicable,
        reliability_reason,
    )


def rank_reactions(args: argparse.Namespace) -> pd.DataFrame:
    device = torch.device(args.device)
    dual_tower_dir = args.dual_tower_dir.resolve()
    schema = load_feature_schema(dual_tower_dir)
    model_dir = args.model_dir.resolve() if args.model_dir else dual_tower_dir / "models"
    models = load_models_runtime(model_dir, args.scope, device)
    # Query coverage and model-training coverage are intentionally different concepts.
    # `args.protein_dir` may be the broad general candidate universe, while the model
    # schema records the protein library used by the locked deployment. The former is
    # used to resolve a precomputed query embedding; the latter defines current/in-domain
    # status, neighbor transfer, and applicability diagnostics.
    query_protein_library, query_protein_ids = load_protein_library(args.protein_dir.resolve())
    training_entries = Path(str(schema.get("protein_ids_file") or ""))
    if not training_entries.is_absolute():
        training_entries = (ROOT / training_entries).resolve()
    if not training_entries.is_file():
        raise FileNotFoundError(
            f"E2R deployment schema references a missing training protein registry: {training_entries}"
        )
    training_protein_library, training_protein_ids = load_protein_library(training_entries.parent)
    protein_library = query_protein_library
    protein_ids = list(query_protein_ids)
    registered_protein_ids: set[str] = set()
    if args.registered_protein_dir and args.registered_protein_dir.exists():
        registered_features, registered_ids = load_protein_library(args.registered_protein_dir.resolve())
        existing = set(protein_ids)
        keep = [index for index, value in enumerate(registered_ids) if value not in existing]
        if keep:
            protein_library = np.concatenate([protein_library, registered_features[keep]], axis=0)
            appended = [registered_ids[index] for index in keep]
            protein_ids.extend(appended)
            registered_protein_ids.update(appended)
    protein_to_row = {value: index for index, value in enumerate(protein_ids)}

    if args.enzyme_id:
        if args.enzyme_id not in protein_to_row:
            raise ValueError(f"Unknown enzyme ID: {args.enzyme_id}; provide --enzyme-sequence for an external query.")
        query_feature = protein_library[protein_to_row[args.enzyme_id]]
        query_id = args.enzyme_id
        protein_input_audit = ProteinInputAudit(
            status="precomputed",
            sequence_sha256="",
            sequence_length=0,
            invalid_characters="",
            ambiguous_fraction=0.0,
            low_complexity_fraction=0.0,
            warning="",
        )
    else:
        if not args.enzyme_sequence:
            raise ValueError("Provide --enzyme-id or --enzyme-sequence.")
        frame = pd.DataFrame(
            {
                "enzyme_id": [args.query_id or "external_enzyme_query"],
                "sequence": [clean_sequence(args.enzyme_sequence)],
            }
        )
        encoded, audits = encode_external_enzymes_with_audit(
            frame,
            args.device,
            args.esmc_model,
            input_policy=args.protein_input_policy,
            cache_dir=args.feature_cache_dir,
        )
        query_feature = encoded[0]
        protein_input_audit = audits[0]
        query_id = frame.iloc[0]["enzyme_id"]

    base_reaction_features, base_reaction_ids = load_reaction_library(dual_tower_dir, schema)
    base_reaction_id_set = set(base_reaction_ids)
    requires_auxiliary = models_require_auxiliary_reaction_features(models)
    registered_candidate_ids: set[str] = set()
    if args.registered_reaction_feature_dir is not None:
        if requires_auxiliary:
            raise ValueError(
                "Expanded registered reaction features do not provide the auxiliary "
                "reaction modality required by this deployment"
            )
        reaction_features, reaction_ids = load_registered_reaction_feature_library(
            args.registered_reaction_feature_dir.resolve(), schema
        )
        missing_base = sorted(base_reaction_id_set - set(reaction_ids))
        if missing_base:
            raise ValueError(
                f"Expanded reaction library is missing base model reactions: {missing_base[:10]}"
            )
        registered_candidate_ids = set(reaction_ids) - base_reaction_id_set
        auxiliary_reaction_features = None
    else:
        reaction_features = base_reaction_features
        reaction_ids = list(base_reaction_ids)
        auxiliary_reaction_features = (
            load_auxiliary_reaction_library(dual_tower_dir, reaction_ids)
            if requires_auxiliary
            else None
        )
    external_frames: list[pd.DataFrame] = []
    if (
        args.registered_reaction_feature_dir is None
        and args.registered_reactions_csv
        and args.registered_reactions_csv.exists()
    ):
        external_frames.append(load_external_reaction_rows(args.registered_reactions_csv))
    temporary_external = load_external_reaction_rows(args.external_reactions_csv)
    if not temporary_external.empty:
        external_frames.append(temporary_external)
    external = (
        pd.concat(external_frames, ignore_index=True).drop_duplicates("reaction_id")
        if external_frames
        else pd.DataFrame(columns=["reaction_id", "reaction_smiles"])
    )
    if not external.empty:
        existing = set(reaction_ids)
        external_rows = [
            (
                row.reaction_id,
                row.reaction_smiles,
                encode_reaction(row.reaction_smiles, schema),
            )
            for row in external.itertuples(index=False)
            if row.reaction_id not in existing
        ]
        if external_rows:
            reaction_ids.extend([value[0] for value in external_rows])
            reaction_features = np.concatenate(
                [reaction_features, np.stack([value[2] for value in external_rows])], axis=0
            )
            if requires_auxiliary:
                external_auxiliary = encode_exact_horizyn_reactions(
                    [value[1] for value in external_rows],
                    dual_tower_dir,
                    device,
                    np.stack([value[2] for value in external_rows]),
                )
                assert auxiliary_reaction_features is not None
                auxiliary_reaction_features = np.concatenate(
                    [auxiliary_reaction_features, external_auxiliary], axis=0
                )

    candidate_keep, candidate_subset_audit = candidate_subset_indices(reaction_ids, args.candidate_ids)
    if candidate_subset_audit["applied"]:
        reaction_features = reaction_features[np.asarray(candidate_keep, dtype=np.int64)]
        reaction_ids = [reaction_ids[index] for index in candidate_keep]
        if auxiliary_reaction_features is not None:
            auxiliary_reaction_features = auxiliary_reaction_features[np.asarray(candidate_keep, dtype=np.int64)]
        registered_candidate_ids.intersection_update(reaction_ids)

    direct_member_scores = ensemble_similarity_members(
        models,
        query_feature[None, :],
        reaction_features,
        device,
        auxiliary_reaction_features,
    )[:, :, 0]
    direct_scores = direct_member_scores.mean(axis=0)
    reaction_embedding_sets = reaction_embedding_ensemble(
        models,
        reaction_features,
        device,
        auxiliary_reaction_features,
    )
    reaction_to_row = {value: index for index, value in enumerate(reaction_ids)}
    seed_ids = [value for value in (args.known_reaction_ids or []) if value in reaction_to_row]
    seed_scores = None
    if seed_ids:
        seed_rows = np.asarray([reaction_to_row[value] for value in seed_ids], dtype=np.int64)
        accumulated = np.zeros(len(reaction_ids), dtype=np.float32)
        for embeddings in reaction_embedding_sets:
            accumulated += (embeddings @ embeddings[seed_rows].T).max(axis=1)
        seed_scores = accumulated / len(reaction_embedding_sets)
    neighbor_scores = protein_neighbor_reaction_transfer_scores(
        query_feature,
        training_protein_library,
        training_protein_ids,
        reaction_ids,
        reaction_embedding_sets,
        args.positives.resolve(),
        args.topk_neighbor_proteins,
        exclude_protein_id=args.enzyme_id,
    )
    ranking_objective = resolve_ranking_objective(args.top_k, args.ranking_objective)
    retrieval_mode = args.retrieval_mode
    hybrid_direct_weight = args.hybrid_direct_weight
    current_protein_id_set = set(training_protein_ids)
    is_current_enzyme = args.enzyme_id in current_protein_id_set
    expected_default_model = dual_tower_dir == DEFAULT_E2R_DUAL_TOWER_DIR.resolve()
    route = resolve_route(
        direction="enzyme_to_reaction",
        objective=ranking_objective,
        is_current=is_current_enzyme,
        has_seed=bool(seed_ids),
        manual_override=(
            args.retrieval_mode != "auto"
            or (args.model_dir is not None and not args.internal_expert_override)
            or not expected_default_model
        ),
        temporary_candidate_extension=not temporary_external.empty,
        masked_discovery=bool(args.mask_reaction_ids) and args.mask_semantics == "novelty_filter",
        manifest_path=args.route_manifest,
    )
    retrieval_mode, hybrid_direct_weight = apply_automatic_few_shot_policy(
        retrieval_mode,
        seed_scores,
        route.settings,
        hybrid_direct_weight,
    )
    use_top10_rrf = (
        ranking_objective == "top10"
        and not is_current_enzyme
        and not seed_ids
        and args.retrieval_mode == "auto"
        and args.model_dir is None
        and expected_default_model
    )
    use_top20_dual_kernel = should_use_e2r_top20_dual_kernel(
        ranking_objective=ranking_objective,
        is_current_enzyme=is_current_enzyme,
        has_seed_reactions=bool(seed_ids),
        requested_retrieval_mode=args.retrieval_mode,
        model_dir_override=args.model_dir,
        dual_tower_dir=dual_tower_dir,
        has_temporary_external_reactions=not temporary_external.empty,
        registered_reactions_csv=args.registered_reactions_csv,
    )
    if retrieval_mode == "auto" and not seed_ids:
        if is_current_enzyme or neighbor_scores is None:
            retrieval_mode = "direct"
        else:
            retrieval_mode = "neighbor_hybrid"
            hybrid_direct_weight = {
                "top3": 0.75,
                "top10": E2R_TOP10_PRIMARY_DIRECT_WEIGHT,
                "top20": 0.75,
            }[ranking_objective]
    scores, score_source = choose_retrieval_scores(
        direct_scores,
        seed_scores,
        reaction_ids,
        retrieval_mode,
        neighbor_scores=neighbor_scores,
        hybrid_direct_weight=hybrid_direct_weight,
    )
    routed_member_scores = route_member_scores(
        direct_member_scores,
        seed_scores,
        reaction_ids,
        retrieval_mode,
        neighbor_scores,
        hybrid_direct_weight,
    )
    masked_reaction_ids = set(args.known_reaction_ids or []) | set(
        args.mask_reaction_ids or []
    )
    secondary_model_directory = ""
    auxiliary_score_directory = ""
    uncertainty_consensus_scores: np.ndarray | None = None
    if use_top10_rrf:
        secondary_dual_tower_dir = DEFAULT_E2R_HARDNEG_DUAL_TOWER_DIR.resolve()
        secondary_schema = load_feature_schema(secondary_dual_tower_dir)
        if [str(value) for value in secondary_schema.get("reaction_ids", [])] != [
            str(value) for value in schema.get("reaction_ids", [])
        ]:
            raise ValueError("Primary and hard-negative E2R deployments use different reaction IDs")
        secondary_models = load_models_runtime(
            secondary_dual_tower_dir / "models", args.scope, device
        )
        if models_require_auxiliary_reaction_features(secondary_models):
            raise ValueError("Hard-negative E2R RRF deployment must use base reaction features")
        if len(secondary_models) != len(models):
            raise ValueError("Primary and hard-negative E2R ensembles differ in size")
        secondary_direct_members = ensemble_similarity_members(
            secondary_models,
            query_feature[None, :],
            reaction_features,
            device,
        )[:, :, 0]
        secondary_direct_scores = secondary_direct_members.mean(axis=0)
        secondary_reaction_embeddings = reaction_embedding_ensemble(
            secondary_models, reaction_features, device
        )
        secondary_neighbor_scores = protein_neighbor_reaction_transfer_scores(
            query_feature,
            training_protein_library,
            training_protein_ids,
            reaction_ids,
            secondary_reaction_embeddings,
            args.positives.resolve(),
            E2R_TOP10_SECONDARY_NEIGHBOR_K,
            exclude_protein_id=args.enzyme_id,
        )
        secondary_mode = (
            "neighbor_hybrid" if secondary_neighbor_scores is not None else "direct"
        )
        secondary_scores, _ = choose_retrieval_scores(
            secondary_direct_scores,
            None,
            reaction_ids,
            secondary_mode,
            neighbor_scores=secondary_neighbor_scores,
            hybrid_direct_weight=E2R_TOP10_SECONDARY_DIRECT_WEIGHT,
        )
        secondary_member_scores = route_member_scores(
            secondary_direct_members,
            None,
            reaction_ids,
            secondary_mode,
            secondary_neighbor_scores,
            E2R_TOP10_SECONDARY_DIRECT_WEIGHT,
        )
        scores = reciprocal_rank_fusion_scores(
            scores,
            secondary_scores,
            reaction_ids,
            E2R_TOP10_RRF_PRIMARY_WEIGHT,
            E2R_TOP10_RRF_CONSTANT,
            masked_reaction_ids,
        )
        routed_member_scores = reciprocal_rank_fusion_members(
            routed_member_scores,
            secondary_member_scores,
            reaction_ids,
            E2R_TOP10_RRF_PRIMARY_WEIGHT,
            E2R_TOP10_RRF_CONSTANT,
            masked_reaction_ids,
        )
        uncertainty_consensus_scores = scores
        secondary_model_directory = str(secondary_dual_tower_dir)
        score_source = "rrf_e2r_top10_primary0.35_secondary0.65_c60"
    elif use_top20_dual_kernel:
        dual_kernel_dir = args.dual_kernel_dir.resolve()
        dual_kernel_assets = load_dual_kernel_assets_cached(str(dual_kernel_dir))
        dual_kernel_scores = align_dual_kernel_reaction_scores(
            dual_kernel_score_query(
                query_feature,
                dual_kernel_assets,
                query_id=args.enzyme_id,
            ),
            dual_kernel_assets,
            reaction_ids,
        )
        dual_kernel_member_scores = np.broadcast_to(
            dual_kernel_scores,
            routed_member_scores.shape,
        ).copy()
        scores = reciprocal_rank_fusion_scores(
            scores,
            dual_kernel_scores,
            reaction_ids,
            E2R_TOP20_RRF_PRIMARY_WEIGHT,
            E2R_TOP20_RRF_CONSTANT,
            masked_reaction_ids,
        )
        routed_member_scores = reciprocal_rank_fusion_members(
            routed_member_scores,
            dual_kernel_member_scores,
            reaction_ids,
            E2R_TOP20_RRF_PRIMARY_WEIGHT,
            E2R_TOP20_RRF_CONSTANT,
            masked_reaction_ids,
        )
        uncertainty_consensus_scores = scores
        auxiliary_score_directory = str(dual_kernel_dir)
        score_source = "rrf_e2r_top20_primary0.7_dual_kernel0.3_c60"
    result = sort_scores(reaction_ids, scores, masked_reaction_ids, args.top_k)
    result = annotate_candidate_uncertainty(
        result,
        reaction_ids,
        routed_member_scores,
        masked_reaction_ids,
        args.top_k,
        consensus_scores=uncertainty_consensus_scores,
    )
    nearest_id, nearest_similarity = nearest_protein_similarity(
        query_feature,
        training_protein_library,
        training_protein_ids,
        exclude_protein_id=args.enzyme_id if args.enzyme_id in set(training_protein_ids) else None,
    )
    result.insert(0, "query_id", query_id)
    result.insert(1, "direction", "enzyme_to_reaction")
    result.insert(2, "score_source", score_source)
    result.insert(3, "ranking_objective", ranking_objective)
    result.insert(4, "model_directory", str(model_bundle_root(model_dir)))
    result.insert(5, "model_feature_directory", str(dual_tower_dir))
    result.insert(6, "secondary_model_directory", secondary_model_directory)
    result.insert(7, "auxiliary_score_directory", auxiliary_score_directory)
    result.insert(8, "query_nearest_library_id", nearest_id)
    result.insert(8, "query_nearest_library_similarity", nearest_similarity)
    result.insert(9, "query_is_current_entity", is_current_enzyme)
    external_candidate_ids = registered_candidate_ids | set(external["reaction_id"].astype(str))
    result["is_external_candidate"] = result["candidate_id"].isin(external_candidate_ids)
    reliability_applicable = (
        (not is_current_enzyme)
        and (not seed_ids)
        and (not args.mask_reaction_ids)
        and not args.candidate_ids
        and args.retrieval_mode == "auto"
        and args.model_dir is None
        and expected_default_model
    )
    if is_current_enzyme:
        reliability_reason = "not_applicable_current_entity"
    elif seed_ids:
        reliability_reason = "not_applicable_few_shot"
    elif args.mask_reaction_ids:
        reliability_reason = (
            "not_applicable_known_associations_masked"
            if args.mask_semantics == "novelty_filter"
            else "not_applicable_output_separation_mask"
        )
    elif args.candidate_ids:
        reliability_reason = "not_applicable_candidate_subset"
    elif args.retrieval_mode != "auto" or args.model_dir is not None or not expected_default_model:
        reliability_reason = "not_applicable_manual_override"
    else:
        reliability_reason = "not_applicable"
    result = apply_candidate_subset_metadata(result, candidate_subset_audit)
    result = apply_route_provenance(
        result,
        route,
        candidate_ids=reaction_ids,
        registry_version=registry_version(args.registered_protein_dir.resolve().parent),
    )
    for column, value in protein_input_audit.as_columns().items():
        result[column] = value
    return apply_empirical_reliability(
        result,
        "enzyme_to_reaction",
        ranking_objective,
        args.calibrators.resolve(),
        reliability_applicable,
        reliability_reason,
    )


def add_common_arguments(parser: argparse.ArgumentParser, default_dual_tower_dir: Path | None) -> None:
    parser.add_argument("--model-dir", type=Path, default=None)
    parser.add_argument(
        "--internal-expert-override",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--dual-tower-dir", type=Path, default=default_dual_tower_dir)
    parser.add_argument("--protein-dir", type=Path, default=DEFAULT_PROTEIN_DIR)
    parser.add_argument("--registered-protein-dir", type=Path, default=DEFAULT_REGISTERED_PROTEIN_DIR)
    parser.add_argument("--registered-reactions-csv", type=Path, default=DEFAULT_REGISTERED_REACTIONS)
    parser.add_argument(
        "--dual-kernel-dir",
        type=Path,
        default=DEFAULT_E2R_TOP20_DUAL_KERNEL_DIR,
        help="Locked sparse dual-kernel assets used only by eligible external E2R Top-20 auto routing.",
    )
    parser.add_argument("--calibrators", type=Path, default=DEFAULT_UNCERTAINTY_CALIBRATORS)
    parser.add_argument("--route-manifest", type=Path, default=DEFAULT_ROUTE_MANIFEST)
    parser.add_argument("--feature-cache-dir", type=Path, default=DEFAULT_FEATURE_CACHE)
    parser.add_argument(
        "--reaction-feature-policy",
        choices=["strict", "warn", "fallback"],
        default="warn",
        help="Control behavior when an external reaction cannot be encoded by DRFP.",
    )
    parser.add_argument(
        "--protein-input-policy",
        choices=["strict", "warn", "fallback"],
        default="warn",
        help="Control validation behavior for an external protein sequence.",
    )
    parser.add_argument(
        "--reliability-policy",
        choices=["annotate", "require_calibrated", "require_intermediate", "require_higher"],
        default="annotate",
        help="Optionally reject queries whose external double-cold reliability evidence is insufficient.",
    )
    parser.add_argument(
        "--conformal-mode",
        choices=sorted(SUPPORTED_CONFORMAL_MODES),
        default="annotate",
        help=(
            "Annotate the route-bound conformal retrieval set, disable it, or expand "
            "the returned prefix to the calibrated set size."
        ),
    )
    parser.add_argument(
        "--conformal-alpha",
        type=float,
        default=0.10,
        help="Requested marginal miscoverage level for conformal retrieval sets.",
    )
    parser.add_argument(
        "--conformal-calibrators",
        type=Path,
        default=DEFAULT_CONFORMAL_CALIBRATORS,
    )
    parser.add_argument("--positives", type=Path, default=DEFAULT_POSITIVES)
    parser.add_argument("--topk-neighbor-reactions", type=int, default=5)
    parser.add_argument("--topk-neighbor-proteins", type=int, default=5)
    parser.add_argument(
        "--scope",
        choices=["production", "protein_cold", "reaction_cold", "double_cold"],
        default="production",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--esmc-model", default="esmc_600m")
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument(
        "--candidate-ids", nargs="*", default=[],
        help="Optional exact include-only candidate subset, intersected with the selected candidate universe before scoring.",
    )
    parser.add_argument(
        "--mask-semantics",
        choices=["output_separation", "novelty_filter"],
        default="novelty_filter",
        help="Describe whether mask IDs only separate already-known evidence from output or represent an explicit novelty filter.",
    )
    parser.add_argument(
        "--ranking-objective",
        choices=["auto", "top3", "top10", "top20"],
        default="auto",
        help="Optimization target for automatic routing; auto follows top-k.",
    )
    parser.add_argument("--hybrid-direct-weight", type=float, default=0.5)
    parser.add_argument(
        "--retrieval-mode",
        choices=["auto", "direct", "seed", "hybrid", "neighbor", "neighbor_hybrid"],
        default="auto",
        help="auto uses supplied seeds when available; otherwise it combines direct and neighbor-transfer scores when possible.",
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--audit-output", type=Path, default=None)
    parser.add_argument("--query-id", default=None)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Open-world bidirectional TPS retrieval.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    enzyme_parser = subparsers.add_parser(
        "rank-enzymes",
        help="Rank enzyme candidates for an existing or external reaction.",
    )
    add_common_arguments(enzyme_parser, None)
    enzyme_parser.add_argument("--reaction-id", default=None)
    enzyme_parser.add_argument("--reaction-smiles", default=None)
    enzyme_parser.add_argument("--external-enzymes-csv", type=Path, default=None)
    enzyme_parser.add_argument("--known-enzyme-ids", nargs="*", default=[])
    enzyme_parser.add_argument(
        "--mask-enzyme-ids",
        nargs="*",
        default=[],
        help="Known enzymes to exclude without using them as few-shot seeds.",
    )
    enzyme_parser.add_argument(
        "--enzyme-taxonomy-scope",
        choices=sorted(SUPPORTED_ENZYME_TAXONOMY_SCOPES),
        default="all",
        help="Restrict the R2E enzyme candidate universe to all, eukaryotic, or prokaryotic proteins before scoring.",
    )
    enzyme_parser.add_argument(
        "--taxonomy-scope-registry",
        type=Path,
        default=DEFAULT_TAXONOMY_SCOPE_REGISTRY,
        help="Local audited protein taxonomy registry used by the R2E candidate-universe filter.",
    )
    enzyme_parser.add_argument("--cage-scores", type=Path, default=DEFAULT_CAGE_SCORES)
    enzyme_parser.add_argument("--cage-rescue-slots", type=int, default=5)

    reaction_parser = subparsers.add_parser(
        "rank-reactions",
        help="Rank reaction candidates for an existing or external enzyme.",
    )
    add_common_arguments(reaction_parser, DEFAULT_E2R_DUAL_TOWER_DIR)
    reaction_parser.add_argument("--enzyme-id", default=None)
    reaction_parser.add_argument("--enzyme-sequence", default=None)
    reaction_parser.add_argument("--external-reactions-csv", type=Path, default=None)
    reaction_parser.add_argument("--registered-reaction-feature-dir", type=Path, default=None)
    reaction_parser.add_argument(
        "--known-reaction-ids",
        nargs="*",
        default=[],
        help="Few-shot reaction seeds; these are also masked from the output ranking.",
    )
    reaction_parser.add_argument(
        "--mask-reaction-ids",
        nargs="*",
        default=[],
        help="Known reactions to exclude without using them as few-shot seeds.",
    )
    return parser


def execute_ranking(args: argparse.Namespace) -> pd.DataFrame:
    if args.top_k <= 0:
        raise ValueError("top-k must be positive")
    if not 0.0 < args.conformal_alpha < 1.0:
        raise ValueError("conformal alpha must be strictly between 0 and 1")

    requested_top_k = int(args.top_k)

    def execute_once(local_args: argparse.Namespace) -> pd.DataFrame:
        local_result = (
            rank_enzymes(local_args)
            if local_args.command == "rank-enzymes"
            else rank_reactions(local_args)
        )
        local_result = apply_evidence_passport(local_result)
        return apply_conformal_retrieval_set(
            local_result,
            calibrators_path=local_args.conformal_calibrators,
            alpha=local_args.conformal_alpha,
            mode=local_args.conformal_mode,
        )

    result = execute_once(args)
    expanded = False
    if args.conformal_mode == "expand" and not result.empty:
        row = result.iloc[0]
        set_size = row.get("conformal_set_size")
        if (
            str(row.get("conformal_status", ""))
            == "validated_external_double_cold_transport"
            and pd.notna(set_size)
            and int(set_size) > len(result)
        ):
            conformal_metadata = {
                column: row[column]
                for column in result.columns
                if column.startswith("conformal_")
                and column not in {"conformal_set_member", "conformal_expanded_output"}
            }
            expanded_args = copy.copy(args)
            expanded_args.top_k = int(set_size)
            expanded_args.ranking_objective = str(row["ranking_objective"])
            expanded_args.conformal_mode = "disabled"
            result = execute_once(expanded_args)
            for column, value in conformal_metadata.items():
                result[column] = value
            result["conformal_mode"] = "expand"
            result["conformal_set_member"] = result["rank"].astype(int).le(int(set_size))
            result["conformal_set_truncated"] = bool(len(result) < int(set_size))
            result["conformal_recommendation"] = "review_conformal_set"
            expanded = True
    result["requested_top_k"] = requested_top_k
    result["conformal_expanded_output"] = expanded
    enforce_reliability_policy(result, args.reliability_policy)
    return result


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    result = execute_ranking(args)
    output = args.output or (DEFAULT_OUTPUT / f"{args.command}.csv")
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False)
    audit_output = args.audit_output or output.with_suffix(output.suffix + ".audit.json")
    write_query_audit(result, audit_output)
    print(result.to_string(index=False))
    print(
        json.dumps(
            {
                "output": str(output.resolve()),
                "audit_output": str(audit_output.resolve()),
                "n_results": len(result),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
