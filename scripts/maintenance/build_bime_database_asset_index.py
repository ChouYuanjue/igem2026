from __future__ import annotations

import csv
import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "reproducibility/bime_rank/database_assets.json"
UNIVERSE = ROOT / "data/catalyst_candidate_universes/general_merged"
UNIVERSE_MANIFEST = UNIVERSE / "manifest.json"
RELEASE_MANIFEST = ROOT / "reproducibility/research_release_manifest.json"

TABLES = [
    ("manifest", "data/catalyst_candidate_universes/general_merged/manifest.json", None),
    ("protein_sequences", "data/catalyst_candidate_universes/general_merged/protein_sequences.tsv", "tsv"),
    ("protein_metadata", "data/catalyst_candidate_universes/general_merged/protein_metadata.csv", "csv"),
    ("reactions", "data/catalyst_candidate_universes/general_merged/reactions.csv", "csv"),
    ("associations", "data/catalyst_candidate_universes/general_merged/associations.csv", "csv"),
    ("sequence_version_conflicts", "data/catalyst_candidate_universes/general_merged/sequence_version_conflicts.csv", "csv"),
    ("protein_entries", "data/catalyst_candidate_universes/general_merged/proteins/entries.csv", "csv"),
]

MODEL_READY_PATHS = {
    "data/catalyst_candidate_universes/general_merged/proteins/embeddings.npy",
    "data/external/enzgfm_current/general_merged_650m_mean_v1/embeddings.npy",
    "data/catalyst_candidate_universes/general_merged/reaction_features/drfp_categorical_v1/reaction_feature_matrix.npy",
    "data/catalyst_candidate_universes/general_merged/reaction_features/drfp_categorical_rdkitplus_v1/reaction_feature_matrix.npy",
    "data/catalyst_candidate_universes/general_merged/reaction_features/drfp_categorical_rdkitplus_center_v1/reaction_feature_matrix.npy",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def tracked() -> set[str]:
    out = subprocess.check_output(["git", "ls-files", "-z"], cwd=ROOT)
    return {x.decode() for x in out.split(b"\0") if x}


def data_rows(path: Path, fmt: str | None) -> int | None:
    if fmt is None:
        return None
    delimiter = "\t" if fmt == "tsv" else ","
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle, delimiter=delimiter)
        try:
            next(reader)
        except StopIteration:
            return 0
        return sum(1 for _ in reader)


def main() -> None:
    tracked_paths = tracked()
    universe = json.loads(UNIVERSE_MANIFEST.read_text(encoding="utf-8"))
    release = json.loads(RELEASE_MANIFEST.read_text(encoding="utf-8"))
    direct_release = {str(x["path"]): x for x in release["direct_git_assets"]}

    tables = []
    for role, relative, fmt in TABLES:
        path = ROOT / relative
        if relative not in tracked_paths or not path.is_file():
            raise FileNotFoundError(f"canonical database table missing from Git: {relative}")
        record = {
            "role": role,
            "path": relative,
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
        rows = data_rows(path, fmt)
        if rows is not None:
            record["rows"] = rows
        release_record = direct_release.get(relative)
        if release_record is None or release_record["sha256"] != record["sha256"]:
            raise RuntimeError(f"canonical database table missing/drifted in direct release assets: {relative}")
        tables.append(record)

    rebuildable_release = {str(x["path"]): x for x in release.get("rebuildable_assets", [])}
    external_release = {str(x["target"]): x for x in release.get("external_assets", [])}
    source_files = []
    unresolved_sources: list[str] = []
    for relative, digest in sorted(universe.get("source_files", {}).items()):
        if relative in direct_release and relative in tracked_paths:
            role = "direct_git_input"
            coverage = "direct"
        elif relative in rebuildable_release:
            role = "rebuildable_input"
            coverage = "rebuildable"
        elif relative in external_release:
            role = "external_restorable_input"
            coverage = "external"
        else:
            role = "unresolved_input"
            coverage = "unresolved"
            unresolved_sources.append(relative)
        source_files.append({
            "path": relative,
            "sha256": digest,
            "portable_release_input": coverage == "direct",
            "role": role,
            "coverage": coverage,
        })
    if unresolved_sources:
        raise RuntimeError(
            "exact database assembly has undeclared source inputs: " + ", ".join(unresolved_sources)
        )

    model_ready = [
        dict(item)
        for item in release.get("rebuildable_assets", [])
        if str(item["path"]) in MODEL_READY_PATHS
    ]
    if {str(x["path"]) for x in model_ready} != MODEL_READY_PATHS:
        raise RuntimeError("database model-ready rebuildable asset set drifted")

    rxnmapper = json.loads((ROOT / "reproducibility/bime_rank/rxnmapper_general_merged_v1.json").read_text())
    mapped = rxnmapper["portable_precompute"]
    mapped_path = str(mapped["path"])
    if mapped_path not in tracked_paths or mapped_path not in direct_release:
        raise RuntimeError("RXNMapper portable precompute is not a direct Git release asset")

    counts = {
        "proteins": int(universe["protein_count"]),
        "reactions": int(universe["reaction_count"]),
        "associations": int(universe["association_count"]),
        "canonical_tables": len(tables),
        "exact_assembly_source_files": len(source_files),
        "exact_assembly_direct_inputs": sum(x["coverage"] == "direct" for x in source_files),
        "exact_assembly_rebuildable_inputs": sum(x["coverage"] == "rebuildable" for x in source_files),
        "exact_assembly_external_inputs": sum(x["coverage"] == "external" for x in source_files),
        "exact_assembly_unresolved_inputs": len(unresolved_sources),
        "model_ready_rebuildable_assets": len(model_ready),
        "portable_precomputed_assets": 1,
    }
    payload = {
        "schema_version": 2,
        "authority": "data/catalyst_candidate_universes/general_merged/manifest.json",
        "database_contract": universe["contract"],
        "database_version": universe["version"],
        "policy": {
            "portable_database": "Canonical protein/reaction/association tables are distributed directly in Git and remain the release database authority.",
            "exact_assembly": "Every source read by the canonical assembly builder must resolve to a direct Git input, a deterministic rebuildable input, or a hash-locked external restore contract; unresolved inputs are forbidden.",
            "tables_only_replay": "The canonical tables can be replayed without source embedding matrices via build_general_candidate_universe.py --tables-only; model-ready embeddings are second-stage deterministic derivatives.",
            "model_ready_derivatives": "Large embedding/feature matrices are rebuilt from canonical tables or explicitly declared upstream inputs with pinned builders/models; they are not separate database authorities.",
            "private_data": "No private/local candidate library is an input to the canonical release database contract.",
        },
        "counts": counts,
        "canonical_tables": tables,
        "exact_assembly": {
            "builder": "projects/active/terpene_screening/build_general_candidate_universe.py",
            "command": ".venv/bin/python projects/active/terpene_screening/build_general_candidate_universe.py --tables-only --output <output-dir>",
            "source_files": source_files,
            "unresolved_source_count": len(unresolved_sources),
            "note": "Exact canonical-table replay is a supported release path. Large ReactZyme intermediates are rebuilt byte-exactly from hash-locked official inputs before this builder runs.",
        },
        "model_ready_rebuildable_assets": model_ready,
        "portable_precomputed_assets": [rxnmapper],
    }
    OUT.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(counts, indent=2))


if __name__ == "__main__":
    main()
