from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


DEFAULT_CANDIDATE_UNIVERSE = "general_merged"
TPS_SPECIALIZED_UNIVERSE = "tps_specialized"
SUPPORTED_CANDIDATE_UNIVERSES = frozenset(
    {DEFAULT_CANDIDATE_UNIVERSE, TPS_SPECIALIZED_UNIVERSE}
)


@dataclass(frozen=True)
class CandidateUniverseSpec:
    key: str
    protein_dir: Path
    registered_reactions_csv: Path
    association_csv: Path | None
    protein_metadata_csv: Path | None
    description: str
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


def universe_specs(root: Path) -> dict[str, CandidateUniverseSpec]:
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
            specialized=True,
        ),
    }


def resolve_candidate_universe(root: Path, key: str | None) -> CandidateUniverseSpec:
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
    specs = universe_specs(root)
    if normalized not in specs:
        raise ValueError(
            f"Unsupported candidate universe {key!r}; expected one of "
            f"{sorted(SUPPORTED_CANDIDATE_UNIVERSES)}"
        )
    spec = specs[normalized]
    spec.validate()
    return spec
