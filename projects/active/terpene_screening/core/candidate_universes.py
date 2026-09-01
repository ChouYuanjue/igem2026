from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path


DEFAULT_CANDIDATE_UNIVERSE = "general_merged"
TPS_SPECIALIZED_UNIVERSE = "tps_specialized"
SUPPORTED_CANDIDATE_UNIVERSES = frozenset(
    {DEFAULT_CANDIDATE_UNIVERSE, TPS_SPECIALIZED_UNIVERSE}
)
TPS_VERSION_UNAVAILABLE = "tps-specialized-assets-unavailable"


@dataclass(frozen=True)
class CandidateUniverseSpec:
    key: str
    protein_dir: Path
    registered_reactions_csv: Path
    association_csv: Path | None
    protein_metadata_csv: Path | None
    description: str
    version: str
    specialized: bool = False
    reaction_feature_dir: Path | None = None

    def validate(self) -> None:
        required = [
            self.protein_dir / "embeddings.npy",
            self.protein_dir / "entries.csv",
            self.registered_reactions_csv,
        ]
        if self.association_csv is not None:
            required.append(self.association_csv)
        if self.protein_metadata_csv is not None:
            required.append(self.protein_metadata_csv)
        if self.reaction_feature_dir is not None:
            required.extend(
                [
                    self.reaction_feature_dir / "reaction_feature_matrix.npy",
                    self.reaction_feature_dir / "entries.csv",
                    self.reaction_feature_dir / "manifest.json",
                ]
            )
        missing = [str(path) for path in required if not path.is_file()]
        if missing:
            raise FileNotFoundError(
                f"Candidate universe {self.key!r} is incomplete: {', '.join(missing)}"
            )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _general_version(merged: Path) -> str:
    manifest = merged / "manifest.json"
    if manifest.is_file():
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        version = str(payload.get("version") or "").strip()
        if version:
            return version
    # A content-derived fallback keeps provenance correct even for a rebuilt local
    # universe whose optional human-readable manifest is absent. If the assets are
    # absent too (for example in a portable CI checkout), the resulting placeholder
    # is metadata only: resolve_candidate_universe(validate=True) still rejects it.
    assets = [merged / "proteins/entries.csv", merged / "reactions.csv"]
    token = "|".join(_sha256_file(path) for path in assets if path.is_file())
    return "general-merged-" + hashlib.sha256(token.encode()).hexdigest()[:12]


def _tps_version(root: Path) -> str:
    assets = [
        root / "data/terpene_embeddings/esmc600m_mean/entries.csv",
        root / "data/terpene_open_world_registry/proteins/entries.csv",
        root / "data/terpene_open_world_registry/reactions.csv",
        root / "results/terpene_production_models/marts_adapted_drfp_pu/reaction_registry.csv",
    ]
    missing = [str(path) for path in assets if not path.is_file()]
    if missing:
        raise FileNotFoundError("TPS candidate-universe version assets missing: " + ", ".join(missing))
    token = "|".join(_sha256_file(path) for path in assets)
    return "tps-specialized-" + hashlib.sha256(token.encode()).hexdigest()[:12]


def _tps_version_if_available(root: Path) -> str:
    try:
        return _tps_version(root)
    except FileNotFoundError:
        return TPS_VERSION_UNAVAILABLE


def universe_specs(root: Path) -> dict[str, CandidateUniverseSpec]:
    """Return registry metadata without requiring every optional universe asset.

    This function is used for introspection and request construction. Strict asset
    validation belongs to ``resolve_candidate_universe(..., validate=True)`` so a
    missing TPS-specialist checkout cannot break an otherwise valid general-universe
    request before execution even starts.
    """
    root = root.resolve()
    merged = root / "data/catalyst_candidate_universes/general_merged"
    return {
        DEFAULT_CANDIDATE_UNIVERSE: CandidateUniverseSpec(
            key=DEFAULT_CANDIDATE_UNIVERSE,
            protein_dir=merged / "proteins",
            registered_reactions_csv=merged / "reactions.csv",
            association_csv=merged / "associations.csv",
            protein_metadata_csv=merged / "protein_metadata.csv",
            description=(
                "General enzyme universe merged with project TPS assets and UniProt TPS "
                "expansion representatives, deduplicated by exact protein sequence."
            ),
            version=_general_version(merged),
            specialized=False,
            reaction_feature_dir=merged / "reaction_features/drfp_categorical_v1",
        ),
        TPS_SPECIALIZED_UNIVERSE: CandidateUniverseSpec(
            key=TPS_SPECIALIZED_UNIVERSE,
            protein_dir=root / "data/terpene_embeddings/esmc600m_mean",
            registered_reactions_csv=root / "data/terpene_open_world_registry/reactions.csv",
            association_csv=None,
            protein_metadata_csv=None,
            description=(
                "Project TPS-specialized universe used with TPS-domain-trained and TPS-evaluated retrieval assets. "
                "It is an explicit specialist scope for terpene-synthase questions, not the general default; "
                "scores from this scope are not compared directly with general_merged scores."
            ),
            version=_tps_version_if_available(root),
            specialized=True,
        ),
    }


def resolve_candidate_universe(
    root: Path,
    key: str | None,
    *,
    validate: bool = True,
) -> CandidateUniverseSpec:
    normalized = str(key or DEFAULT_CANDIDATE_UNIVERSE).strip().lower()
    aliases = {
        "general": DEFAULT_CANDIDATE_UNIVERSE,
        "merged": DEFAULT_CANDIDATE_UNIVERSE,
        "default": DEFAULT_CANDIDATE_UNIVERSE,
        "tps": TPS_SPECIALIZED_UNIVERSE,
        "terpene": TPS_SPECIALIZED_UNIVERSE,
        "specialized": TPS_SPECIALIZED_UNIVERSE,
    }
    normalized = aliases.get(normalized, normalized)
    if normalized not in SUPPORTED_CANDIDATE_UNIVERSES:
        raise ValueError(
            f"Unsupported candidate universe {key!r}; expected one of "
            f"{sorted(SUPPORTED_CANDIDATE_UNIVERSES)}"
        )
    spec = universe_specs(root)[normalized]
    if validate:
        # Version provenance for TPS depends on more than the three minimum runtime
        # files checked by CandidateUniverseSpec.validate(). Re-run the strict version
        # audit only when TPS is actually selected for execution.
        if normalized == TPS_SPECIALIZED_UNIVERSE:
            strict_version = _tps_version(root.resolve())
            if strict_version != spec.version:
                spec = CandidateUniverseSpec(
                    key=spec.key,
                    protein_dir=spec.protein_dir,
                    registered_reactions_csv=spec.registered_reactions_csv,
                    association_csv=spec.association_csv,
                    protein_metadata_csv=spec.protein_metadata_csv,
                    description=spec.description,
                    version=strict_version,
                    specialized=spec.specialized,
                    reaction_feature_dir=spec.reaction_feature_dir,
                )
        spec.validate()
    return spec
