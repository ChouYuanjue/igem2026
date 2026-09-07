from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from projects.active.terpene_screening import build_general_candidate_universe as builder_module
from projects.active.terpene_screening.build_general_candidate_universe import (
    ProteinRow,
    UniverseBuilder,
    load_full_general_reactions,
)
from projects.active.terpene_screening.core import engine
from projects.active.terpene_screening.core.candidate_universes import (
    CandidateUniverseSpec,
    DEFAULT_CANDIDATE_UNIVERSE,
    TPS_SPECIALIZED_UNIVERSE,
    resolve_candidate_universe,
    universe_specs,
)
from projects.active.terpene_screening.rank_open_world import build_parser
from projects.active.terpene_screening.core.taxonomy_scope import (
    filter_candidate_ids,
    taxonomy_record,
    taxonomy_summary,
    validate_seed_scope,
)


def _embedding_dir(root: Path, name: str, vectors: list[list[float]]) -> Path:
    path = root / name
    path.mkdir(parents=True)
    np.save(path / "embeddings.npy", np.asarray(vectors, dtype=np.float32))
    pd.DataFrame({"row": range(len(vectors)), "Entry": [f"ROW{i}" for i in range(len(vectors))]}).to_csv(
        path / "entries.csv", index=False
    )
    return path


def test_full_general_reaction_inventory_ignores_overlap_filter_labels(tmp_path: Path):
    source = tmp_path / "reaction_overlap_audit.csv"
    pd.DataFrame(
        [
            {
                "rhea_id": "RHEA:10001",
                "smiles_seq": "CC>>CO",
                "tps_related": "False",
                "exact_tps_rhea": "False",
            },
            {
                "rhea_id": "RHEA:10002",
                "smiles_seq": "CO>>C=O",
                "tps_related": "True",
                "exact_tps_rhea": "True",
            },
        ]
    ).to_csv(source, index=False)
    reactions = load_full_general_reactions(source)
    assert reactions.columns.tolist() == ["rhea_id", "smiles_seq"]
    assert reactions["rhea_id"].tolist() == ["RHEA:10001", "RHEA:10002"]


def test_exact_sequence_dedup_preserves_all_stable_aliases(tmp_path: Path):
    embeddings = _embedding_dir(tmp_path, "emb", [[1.0, 2.0]])
    builder = UniverseBuilder(tmp_path / "out")
    builder._offer(
        ProteinRow(
            protein_id="P_HIGH",
            sequence="MPEPTIDE",
            embedding_dir=embeddings,
            embedding_row=0,
            source_layer="project",
            priority=100,
            aliases=("P_HIGH",),
        )
    )
    builder._offer(
        ProteinRow(
            protein_id="P_LOW",
            sequence="MPEPTIDE",
            embedding_dir=embeddings,
            embedding_row=0,
            source_layer="general",
            priority=10,
            aliases=("P_LOW", "P_ALIAS"),
        )
    )
    assert len(builder.proteins) == 1
    row = next(iter(builder.proteins.values()))
    assert row.protein_id == "P_HIGH"
    assert set(row.aliases) >= {"P_HIGH", "P_LOW", "P_ALIAS"}


def test_stable_accession_sequence_conflict_selects_priority_and_maps_all_versions(tmp_path: Path):
    high = _embedding_dir(tmp_path, "high", [[1.0, 0.0]])
    low = _embedding_dir(tmp_path, "low", [[0.0, 1.0]])
    builder = UniverseBuilder(tmp_path / "out")
    builder._offer(
        ProteinRow(
            protein_id="P12345",
            sequence="MPEPTIDEA",
            embedding_dir=high,
            embedding_row=0,
            source_layer="project",
            priority=100,
            aliases=("P12345",),
        )
    )
    builder._offer(
        ProteinRow(
            protein_id="P12345",
            sequence="MPEPTIDEB",
            embedding_dir=low,
            embedding_row=0,
            source_layer="old_snapshot",
            priority=10,
            aliases=("P12345",),
        )
    )
    alias_to_canonical, sha_to_canonical = builder.finalize_proteins()
    entries = pd.read_csv(tmp_path / "out/proteins/entries.csv", dtype=str)
    conflicts = pd.read_csv(tmp_path / "out/sequence_version_conflicts.csv", dtype=str)
    assert entries["Entry"].tolist() == ["P12345"]
    assert alias_to_canonical["P12345"] == "P12345"
    assert set(sha_to_canonical.values()) == {"P12345"}
    assert len(conflicts) == 2
    assert conflicts["selected"].astype(str).str.lower().eq("true").sum() == 1


def test_universe_registry_has_general_default_and_explicit_tps_specialization(tmp_path: Path):
    specs = universe_specs(tmp_path)
    assert set(specs) == {DEFAULT_CANDIDATE_UNIVERSE, TPS_SPECIALIZED_UNIVERSE}
    assert specs[DEFAULT_CANDIDATE_UNIVERSE].specialized is False
    assert specs[TPS_SPECIALIZED_UNIVERSE].specialized is True
    with pytest.raises(ValueError):
        resolve_candidate_universe(tmp_path, "not-a-real-universe")


def test_engine_injects_only_registered_general_universe_paths(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    protein_dir = _embedding_dir(tmp_path, "general/proteins", [[1.0, 2.0]])
    reactions = tmp_path / "general/reactions.csv"
    reactions.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{"reaction_id": "RHEA:12345", "reaction_smiles": "CC>>CO"}]).to_csv(
        reactions, index=False
    )
    spec = CandidateUniverseSpec(
        key=DEFAULT_CANDIDATE_UNIVERSE,
        protein_dir=protein_dir,
        registered_reactions_csv=reactions,
        association_csv=None,
        protein_metadata_csv=None,
        description="test general universe",
        version="test-general-v1",
        specialized=False,
    )
    monkeypatch.setattr(engine, "resolve_candidate_universe", lambda root, key, **_kwargs: spec)
    argv = engine.payload_to_argv(
        "rank-enzymes",
        {
            "reaction_smiles": "CC>>CO",
            "candidate_universe": DEFAULT_CANDIDATE_UNIVERSE,
            "top_k": 3,
        },
    )
    args = build_parser().parse_args(argv)
    assert Path(args.protein_dir) == protein_dir
    assert Path(args.registered_reactions_csv) == reactions
    assert args.reaction_smiles == "CC>>CO"
    assert args.top_k == 3
    assert "--registered-reaction-feature-dir" not in argv


def test_direct_core_call_retains_historical_tps_universe(monkeypatch: pytest.MonkeyPatch):
    requested: list[str] = []

    def fake_resolve(root: Path, key: str, **_kwargs) -> CandidateUniverseSpec:
        requested.append(key)
        return CandidateUniverseSpec(
            key=TPS_SPECIALIZED_UNIVERSE,
            protein_dir=Path("unused"),
            registered_reactions_csv=Path("unused.csv"),
            association_csv=None,
            protein_metadata_csv=None,
            description="TPS",
            version="test-tps-v1",
            specialized=True,
        )

    monkeypatch.setattr(engine, "resolve_candidate_universe", fake_resolve)
    argv = engine.payload_to_argv("rank-enzymes", {"reaction_id": "RHEA:12345", "top_k": 1})
    assert requested == [TPS_SPECIALIZED_UNIVERSE]
    assert "--protein-dir" not in argv


def test_taxonomy_scope_registry_contract_uses_explicit_fixture(tmp_path: Path):
    registry = tmp_path / "taxonomy.csv"
    pd.DataFrame([
        {"protein_id": "P_EUK", "taxonomy_scope": "eukaryote", "kingdom": "Plantae", "species": "Plant", "taxonomy_source": "fixture", "taxonomy_confidence": "high"},
        {"protein_id": "P_PROK", "taxonomy_scope": "prokaryote", "kingdom": "Bacteria", "species": "Bacterium", "taxonomy_source": "fixture", "taxonomy_confidence": "high"},
        {"protein_id": "P_OTHER", "taxonomy_scope": "other", "kingdom": "Viruses", "species": "Virus", "taxonomy_source": "fixture", "taxonomy_confidence": "medium"},
        {"protein_id": "P_UNKNOWN", "taxonomy_scope": "unknown", "kingdom": "", "species": "", "taxonomy_source": "fixture", "taxonomy_confidence": "unknown"},
    ]).to_csv(registry, index=False)

    assert taxonomy_summary(registry) == {
        "version": "terpene-enzyme-taxonomy-scope-v1",
        "total": 4, "eukaryote": 1, "prokaryote": 1, "other": 1, "unknown": 1,
    }
    assert taxonomy_record("P_EUK", registry_path=registry).kingdom == "Plantae"
    keep_euk, audit_euk = filter_candidate_ids(["P_EUK", "P_PROK", "P_MISSING"], "eukaryote", registry_path=registry)
    keep_prok, audit_prok = filter_candidate_ids(["P_EUK", "P_PROK", "P_MISSING"], "prokaryote", registry_path=registry)
    assert keep_euk == [0]
    assert keep_prok == [1]
    assert audit_euk["unknown_count"] == audit_prok["unknown_count"] == 1
    with pytest.raises(ValueError, match="incompatible"):
        validate_seed_scope(["P_PROK"], "eukaryote", registry_path=registry)
    with pytest.raises(ValueError, match="unclassified"):
        validate_seed_scope(["P_UNKNOWN"], "prokaryote", registry_path=registry)
