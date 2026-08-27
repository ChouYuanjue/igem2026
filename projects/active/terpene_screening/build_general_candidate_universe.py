from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT = ROOT / "data/catalyst_candidate_universes/general_merged"

GENERAL_EMBEDDINGS = ROOT / "data/external/reactzyme_transfer/esmc600m_mean"
GENERAL_SEQUENCES = ROOT / "data/external/reactzyme_transfer/unique_sequences.tsv"
GENERAL_UNIPROT = ROOT / "data/external/reactzyme/cleaned_uniprot_rhea.tsv"
GENERAL_EXPANDED_PAIRS = ROOT / "results/terpene_reactzyme_transfer_audit_v1/reactzyme_expanded_pairs.csv"
# Despite living beside the leakage-safe transfer split, this audit is the *pre-filter*
# full ReactZyme reaction inventory (10k+ reactions). We intentionally consume only
# rhea_id/smiles_seq and never the filtered global_clean_v2/clean_reactions.csv. Product
# candidate coverage and benchmark-isolation subsets are separate concepts.
GENERAL_REACTIONS = ROOT / "data/external/reactzyme_transfer/global_clean_v2/reaction_overlap_audit.csv"


def load_full_general_reactions(path: Path = GENERAL_REACTIONS) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype=str).fillna("")
    required = {"rhea_id", "smiles_seq"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Full ReactZyme reaction inventory is missing columns: {missing}")
    # The full audit carries overlap labels for evaluation construction, but those
    # labels must never decide product candidate inclusion. Returning only the two
    # source fields makes that separation enforceable in code.
    return frame[["rhea_id", "smiles_seq"]].copy()

CURRENT_EMBEDDINGS = ROOT / "data/terpene_embeddings/esmc600m_mean"
CURRENT_SEQUENCES = ROOT / "data/terpene/all_seq_terpene_synthase.tsv"
REGISTERED_EMBEDDINGS = ROOT / "data/terpene_open_world_registry/proteins"
MARTS_ENZYMES = ROOT / "data/terpene_marts/marts_enzymes.tsv"
PRODUCTION = ROOT / "results/terpene_production_models/marts_adapted_drfp_pu"

TPS_PRIMARY_EMBEDDINGS = ROOT / "data/terpene_embeddings/uniprot_tps_primary_esmc600m"
TPS_PRIMARY_METADATA = ROOT / "data/terpene_uniprot_expansion/uniprot_tps_primary_embedding_candidates.tsv"
TPS_RESCUE_EMBEDDINGS = ROOT / "data/terpene_embeddings/uniprot_tps_domain_rescue_esmc600m"
TPS_RESCUE_METADATA = ROOT / "data/terpene_uniprot_expansion/uniprot_tps_domain_only_rescue_candidates.tsv"

BUILD_VERSION = "general-merged-v2"


def clean_sequence(value: object) -> str:
    return "".join(str(value or "").split()).upper()


def sequence_sha(sequence: str) -> str:
    return hashlib.sha256(sequence.encode("ascii", errors="ignore")).hexdigest()


def file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass
class ProteinRow:
    protein_id: str
    sequence: str
    embedding_dir: Path
    embedding_row: int
    source_layer: str
    priority: int
    aliases: tuple[str, ...] = ()
    evidence_scope: str = "candidate"
    pfam: str = ""
    domain_family: str = ""


class UniverseBuilder:
    def __init__(self, output: Path) -> None:
        self.output = output.resolve()
        self.proteins: dict[str, ProteinRow] = {}
        self.id_to_sha: dict[str, str] = {}
        self.source_counts: dict[str, int] = {}
        self.sequence_version_conflicts: list[dict[str, str | int | bool]] = []

    def _offer(self, row: ProteinRow) -> None:
        sequence = clean_sequence(row.sequence)
        if not sequence:
            return
        sha = sequence_sha(sequence)
        row.sequence = sequence
        current = self.proteins.get(sha)
        if current is None:
            self.proteins[sha] = row
        else:
            merged_aliases = tuple(
                dict.fromkeys(
                    value
                    for value in (
                        current.protein_id,
                        *current.aliases,
                        row.protein_id,
                        *row.aliases,
                    )
                    if value
                )
            )
            if row.priority > current.priority:
                row.aliases = merged_aliases
                self.proteins[sha] = row
            else:
                current.aliases = merged_aliases
        self.source_counts[row.source_layer] = self.source_counts.get(row.source_layer, 0) + 1

    @staticmethod
    def _embedding_entries(path: Path) -> tuple[pd.DataFrame, np.ndarray]:
        entries = pd.read_csv(path / "entries.csv", dtype=str)
        matrix = np.load(path / "embeddings.npy", mmap_mode="r")
        if len(entries) != len(matrix):
            raise ValueError(f"Embedding rows do not align in {path}")
        if "Entry" not in entries.columns:
            raise ValueError(f"Embedding entries missing Entry column in {path}")
        entries = entries.copy()
        entries["_row"] = np.arange(len(entries), dtype=int)
        return entries, matrix

    def add_project_current(self) -> None:
        seq = pd.read_csv(CURRENT_SEQUENCES, sep="\t", dtype=str).fillna("")
        seq_map = dict(zip(seq["Entry"].astype(str), seq["Sequence"].map(clean_sequence)))
        entries, _ = self._embedding_entries(CURRENT_EMBEDDINGS)
        for embedding_row, pid_value in enumerate(entries["Entry"].astype(str)):
            pid = str(pid_value)
            self._offer(
                ProteinRow(
                    protein_id=pid,
                    sequence=seq_map.get(pid, ""),
                    embedding_dir=CURRENT_EMBEDDINGS,
                    embedding_row=embedding_row,
                    source_layer="project_current",
                    priority=100,
                    aliases=(pid,),
                    evidence_scope="project_model_catalog",
                )
            )

    def add_project_registered(self) -> None:
        marts = pd.read_csv(MARTS_ENZYMES, sep="\t", dtype=str).fillna("")
        seq_map: dict[str, str] = {}
        alias_map: dict[str, tuple[str, ...]] = {}
        for row in marts.to_dict("records"):
            sequence = clean_sequence(row.get("sequence", ""))
            aliases = tuple(
                dict.fromkeys(
                    str(row.get(key, "")).strip()
                    for key in ("enzyme_id", "uniprot_id", "genbank_id")
                    if str(row.get(key, "")).strip()
                )
            )
            for value in aliases:
                seq_map[value] = sequence
                alias_map[value] = aliases
        entries, _ = self._embedding_entries(REGISTERED_EMBEDDINGS)
        for embedding_row, pid_value in enumerate(entries["Entry"].astype(str)):
            pid = str(pid_value)
            self._offer(
                ProteinRow(
                    protein_id=pid,
                    sequence=seq_map.get(pid, ""),
                    embedding_dir=REGISTERED_EMBEDDINGS,
                    embedding_row=embedding_row,
                    source_layer="project_registered",
                    priority=95,
                    aliases=alias_map.get(pid, (pid,)),
                    evidence_scope="project_registered_candidate",
                )
            )

    def _add_tps_embedding_layer(self, embedding_dir: Path, metadata_path: Path, source: str, priority: int) -> None:
        if not (embedding_dir / "embeddings.npy").is_file():
            return
        metadata = pd.read_csv(metadata_path, sep="\t", dtype=str).fillna("")
        meta = metadata.set_index("accession", drop=False).to_dict("index")
        entries, _ = self._embedding_entries(embedding_dir)
        for embedding_row, pid_value in enumerate(entries["Entry"].astype(str)):
            pid = str(pid_value)
            row = meta.get(pid, {})
            aliases = tuple(
                value for value in str(row.get("sequence_alias_accessions", "") or pid).split(";") if value
            ) or (pid,)
            self._offer(
                ProteinRow(
                    protein_id=pid,
                    sequence=clean_sequence(row.get("sequence", "")),
                    embedding_dir=embedding_dir,
                    embedding_row=embedding_row,
                    source_layer=source,
                    priority=priority,
                    aliases=aliases,
                    evidence_scope=str(row.get("evidence_quality_tier", "candidate")),
                    pfam=str(row.get("pfam_combination", "")),
                    domain_family=str(row.get("domain_family", "")),
                )
            )

    def add_tps_expansion(self) -> None:
        self._add_tps_embedding_layer(
            TPS_PRIMARY_EMBEDDINGS, TPS_PRIMARY_METADATA, "uniprot_tps_expansion", 85
        )
        self._add_tps_embedding_layer(
            TPS_RESCUE_EMBEDDINGS, TPS_RESCUE_METADATA, "uniprot_tps_domain_rescue", 80
        )

    def add_general(self) -> None:
        unique = pd.read_csv(GENERAL_SEQUENCES, sep="\t", dtype=str).fillna("")
        raw = pd.read_csv(
            GENERAL_UNIPROT,
            sep="\t",
            usecols=["Entry", "Sequence", "EC number", "Rhea ID"],
            dtype=str,
        ).fillna("")
        raw["Sequence"] = raw["Sequence"].map(clean_sequence)
        aliases_by_sequence = raw.groupby("Sequence")["Entry"].agg(
            lambda values: tuple(sorted(set(str(value) for value in values if str(value))))
        ).to_dict()
        sequence_by_synthetic = dict(zip(unique["Entry"].astype(str), unique["Sequence"].map(clean_sequence)))
        entries, _ = self._embedding_entries(GENERAL_EMBEDDINGS)
        for embedding_row, synthetic_value in enumerate(entries["Entry"].astype(str)):
            synthetic = str(synthetic_value)
            sequence = sequence_by_synthetic.get(synthetic, "")
            aliases = aliases_by_sequence.get(sequence, ())
            if not aliases:
                raise ValueError(f"Cannot map general sequence {synthetic} back to UniProt")
            self._offer(
                ProteinRow(
                    protein_id=aliases[0],
                    sequence=sequence,
                    embedding_dir=GENERAL_EMBEDDINGS,
                    embedding_row=embedding_row,
                    source_layer="uniprot_rhea_general",
                    priority=50,
                    aliases=aliases,
                    evidence_scope="uniprot_rhea_record",
                )
            )

    def finalize_proteins(self) -> tuple[dict[str, str], dict[str, str]]:
        protein_dir = self.output / "proteins"
        protein_dir.mkdir(parents=True, exist_ok=True)
        # Exact-sequence deduplication above can still leave the same stable accession
        # attached to different sequence snapshots. Treat that as a version conflict,
        # not as two biological entities: select the highest-priority source once,
        # preserve every alias, and map associations from all observed sequence
        # versions back to the selected stable entity. This rule is source-agnostic and
        # intentionally avoids accession-specific exceptions.
        by_id: dict[str, list[tuple[str, ProteinRow]]] = {}
        for sha, row in self.proteins.items():
            by_id.setdefault(row.protein_id, []).append((sha, row))

        chosen: list[tuple[str, ProteinRow]] = []
        sha_to_selected: dict[str, str] = {}
        conflict_rows: list[dict[str, str | int | bool]] = []
        for protein_id, variants in sorted(by_id.items()):
            variants = sorted(
                variants,
                key=lambda item: (-item[1].priority, item[1].source_layer, item[0]),
            )
            selected_sha, selected = variants[0]
            merged_aliases = tuple(
                dict.fromkeys(
                    alias
                    for _, variant in variants
                    for alias in (variant.protein_id, *variant.aliases)
                    if alias
                )
            )
            selected.aliases = merged_aliases
            chosen.append((selected_sha, selected))
            for sha, variant in variants:
                sha_to_selected[sha] = selected.protein_id
                if len(variants) > 1:
                    conflict_rows.append(
                        {
                            "protein_id": protein_id,
                            "sequence_sha256": sha,
                            "source_layer": variant.source_layer,
                            "source_priority": variant.priority,
                            "selected": sha == selected_sha,
                            "selected_sequence_sha256": selected_sha,
                            "selected_source_layer": selected.source_layer,
                        }
                    )
        self.sequence_version_conflicts = conflict_rows
        pd.DataFrame(conflict_rows).to_csv(self.output / "sequence_version_conflicts.csv", index=False)
        chosen.sort(key=lambda item: item[1].protein_id)
        ids = [row.protein_id for _, row in chosen]

        first_matrix = np.load(chosen[0][1].embedding_dir / "embeddings.npy", mmap_mode="r")
        dim = int(first_matrix.shape[1])
        out = np.lib.format.open_memmap(
            protein_dir / "embeddings.npy", mode="w+", dtype=np.float32, shape=(len(chosen), dim)
        )
        matrices: dict[Path, np.ndarray] = {}
        metadata_rows = []
        sequence_rows = []
        alias_to_canonical: dict[str, str] = {}
        sha_to_canonical: dict[str, str] = dict(sha_to_selected)
        for output_row, (sha, row) in enumerate(chosen):
            matrix = matrices.setdefault(
                row.embedding_dir,
                np.load(row.embedding_dir / "embeddings.npy", mmap_mode="r"),
            )
            vector = matrix[row.embedding_row]
            if vector.shape != (dim,):
                raise ValueError(f"Embedding dimension mismatch for {row.protein_id}")
            out[output_row] = vector.astype(np.float32, copy=False)
            aliases = tuple(dict.fromkeys((row.protein_id, *row.aliases)))
            for alias in aliases:
                alias_to_canonical[alias] = row.protein_id
            sha_to_canonical[sha] = row.protein_id
            metadata_rows.append(
                {
                    "protein_id": row.protein_id,
                    "canonical_accession": row.protein_id,
                    "aliases": ";".join(aliases),
                    "source_layer": row.source_layer,
                    "evidence_scope": row.evidence_scope,
                    "sequence_sha256": sha,
                    "sequence_length": len(row.sequence),
                    "pfam": row.pfam,
                    "domain_family": row.domain_family,
                    "model_ready": True,
                }
            )
            sequence_rows.append({"protein_id": row.protein_id, "sequence": row.sequence})
        del out
        pd.DataFrame({"row": range(len(ids)), "Entry": ids}).to_csv(protein_dir / "entries.csv", index=False)
        pd.DataFrame(metadata_rows).to_csv(self.output / "protein_metadata.csv", index=False)
        pd.DataFrame(sequence_rows).to_csv(self.output / "protein_sequences.tsv", sep="\t", index=False)
        return alias_to_canonical, sha_to_canonical

    def build_associations(self, alias_to_canonical: dict[str, str], sha_to_canonical: dict[str, str]) -> int:
        rows: list[dict[str, str]] = []
        general = pd.read_csv(GENERAL_EXPANDED_PAIRS, dtype=str).fillna("")
        for record in general.itertuples(index=False):
            sequence = clean_sequence(record.Sequence)
            protein_id = sha_to_canonical.get(sequence_sha(sequence))
            rhea_id = str(record.rhea_id).strip()
            if protein_id and rhea_id.startswith("RHEA:"):
                rows.append(
                    {
                        "protein_id": protein_id,
                        "reaction_id": rhea_id,
                        "source": "uniprot_rhea_cached",
                        "evidence_type": "recorded_association",
                    }
                )
        project = pd.read_csv(PRODUCTION / "training_pairs.csv", dtype=str).fillna("")
        for record in project.to_dict("records"):
            protein_id = alias_to_canonical.get(str(record.get("Entry", "")).strip(), str(record.get("Entry", "")).strip())
            reaction_id = str(record.get("rhea_id", "")).strip()
            if protein_id and reaction_id:
                rows.append(
                    {
                        "protein_id": protein_id,
                        "reaction_id": reaction_id,
                        "source": "project_catalog",
                        "evidence_type": "project_recorded_association",
                    }
                )
        frame = pd.DataFrame(rows).drop_duplicates(["protein_id", "reaction_id", "source"])
        frame = frame.sort_values(["reaction_id", "protein_id", "source"]).reset_index(drop=True)
        frame.to_csv(self.output / "associations.csv", index=False)
        return len(frame)

    def build_reactions(self) -> int:
        rows: list[dict[str, str]] = []
        general = load_full_general_reactions()
        for record in general.to_dict("records"):
            reaction_id = str(record.get("rhea_id", "")).strip()
            smiles = str(record.get("smiles_seq", "")).strip()
            if reaction_id.startswith("RHEA:") and smiles:
                rows.append(
                    {
                        "reaction_id": reaction_id,
                        "reaction_smiles": smiles,
                        "source_layer": "rhea_general",
                    }
                )
        production = pd.read_csv(PRODUCTION / "reaction_registry.csv", dtype=str).fillna("")
        for record in production.to_dict("records"):
            reaction_id = str(record.get("reaction_id", "")).strip()
            smiles = str(record.get("reaction_smiles", "")).strip()
            if reaction_id and smiles:
                rows.append(
                    {
                        "reaction_id": reaction_id,
                        "reaction_smiles": smiles,
                        "source_layer": "project_model_catalog",
                    }
                )
        registered_path = ROOT / "data/terpene_open_world_registry/reactions.csv"
        if registered_path.is_file():
            registered = pd.read_csv(registered_path, dtype=str).fillna("")
            for record in registered.to_dict("records"):
                reaction_id = str(record.get("reaction_id", "")).strip()
                smiles = str(record.get("reaction_smiles", "")).strip()
                if reaction_id and smiles:
                    rows.append(
                        {
                            "reaction_id": reaction_id,
                            "reaction_smiles": smiles,
                            "source_layer": "project_registered",
                        }
                    )
        priority = {"project_model_catalog": 3, "project_registered": 2, "rhea_general": 1}
        frame = pd.DataFrame(rows)
        frame["_priority"] = frame["source_layer"].map(priority).fillna(0)
        frame = (
            frame.sort_values(["reaction_id", "_priority"], ascending=[True, False])
            .drop_duplicates("reaction_id", keep="first")
            .drop(columns=["_priority"])
            .sort_values("reaction_id")
            .reset_index(drop=True)
        )
        frame.to_csv(self.output / "reactions.csv", index=False)
        return len(frame)

    def write_manifest(self, association_count: int, reaction_count: int) -> None:
        metadata = pd.read_csv(self.output / "protein_metadata.csv", dtype=str).fillna("")
        source_files = [
            GENERAL_EMBEDDINGS / "entries.csv",
            GENERAL_SEQUENCES,
            GENERAL_UNIPROT,
            GENERAL_EXPANDED_PAIRS,
            GENERAL_REACTIONS,
            CURRENT_EMBEDDINGS / "entries.csv",
            CURRENT_SEQUENCES,
            REGISTERED_EMBEDDINGS / "entries.csv",
            MARTS_ENZYMES,
            TPS_PRIMARY_EMBEDDINGS / "entries.csv",
            TPS_PRIMARY_METADATA,
        ]
        if (TPS_RESCUE_EMBEDDINGS / "entries.csv").is_file():
            source_files += [TPS_RESCUE_EMBEDDINGS / "entries.csv", TPS_RESCUE_METADATA]
        manifest = {
            "version": BUILD_VERSION,
            "contract": "candidate-universe-general-merged-v2",
            "deduplication": "exact canonical amino-acid sequence SHA-256; prefer project > TPS expansion > general UniProt/Rhea",
            "protein_count": int(len(metadata)),
            "reaction_count": int(reaction_count),
            "association_count": int(association_count),
            "sequence_version_conflict_rows": int(len(self.sequence_version_conflicts)),
            "sequence_version_conflict_entities": int(
                len({str(row["protein_id"]) for row in self.sequence_version_conflicts})
            ),
            "protein_source_counts": metadata["source_layer"].value_counts().to_dict(),
            "source_files": {
                str(path.relative_to(ROOT)): file_sha(path) for path in source_files if path.is_file()
            },
        }
        (self.output / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        (self.output / "summary.json").write_text(
            json.dumps(
                {
                    "version": BUILD_VERSION,
                    "protein_count": int(len(metadata)),
                    "reaction_count": int(reaction_count),
                    "association_count": int(association_count),
                    "sequence_version_conflict_rows": int(len(self.sequence_version_conflicts)),
                    "sequence_version_conflict_entities": int(
                        len({str(row["protein_id"]) for row in self.sequence_version_conflicts})
                    ),
                    "protein_source_counts": metadata["source_layer"].value_counts().to_dict(),
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    def run(self) -> None:
        self.output.mkdir(parents=True, exist_ok=True)
        self.add_project_current()
        self.add_project_registered()
        self.add_tps_expansion()
        self.add_general()
        alias_to_canonical, sha_to_canonical = self.finalize_proteins()
        association_count = self.build_associations(alias_to_canonical, sha_to_canonical)
        reaction_count = self.build_reactions()
        self.write_manifest(association_count, reaction_count)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the merged general Catalyst candidate universe.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    UniverseBuilder(args.output).run()
    print((args.output / "summary.json").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
