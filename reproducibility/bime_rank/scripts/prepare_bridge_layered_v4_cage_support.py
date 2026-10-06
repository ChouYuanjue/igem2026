from __future__ import annotations

import io
import json
import pickle
from pathlib import Path
from zipfile import ZipFile

import pandas as pd

from projects.active.bridge.model.assets import ROOT

CAND = ROOT / "results/bridge_layered_v4_candidates"
META = ROOT / "data/catalyst_candidate_universes/general_merged/protein_metadata.csv"
SEQS = ROOT / "data/catalyst_candidate_universes/general_merged/protein_sequences.tsv"
TARGETS = ROOT / "results/bridge_relation_unseen_max_v3/targets.csv"
ZIP = ROOT / "data/external/enzymecage_current/authors_drive_current/dataset.zip"
EXISTING = (
    ROOT
    / "data/external/enzymecage_current/cage_official_features/"
    "esmc600m_max_support_v1/mapping_audit.csv"
)
OUT = ROOT / "results/bridge_layered_v4_cage_support"

POCKET_PREFIX = "dataset/RHEA/2025-02-05/pockets/pocket/"
UID2SEQ_PATH = "dataset/RHEA/2025-02-05/uid2seq.pkl"


def metadata_maps() -> tuple[
    dict[str, set[str]],
    dict[str, str],
    dict[str, str],
]:
    meta = pd.read_csv(META, dtype=str).fillna("")
    seqf = pd.read_csv(SEQS, sep="\t", dtype=str).fillna("")
    sequence = dict(zip(seqf.protein_id.astype(str), seqf.sequence.astype(str)))
    aliases: dict[str, set[str]] = {}
    canonical: dict[str, str] = {}
    for row in meta[["protein_id", "canonical_accession", "aliases"]].itertuples(index=False):
        pid = str(row.protein_id)
        vals = {pid, str(row.canonical_accession)}
        vals |= {x for x in str(row.aliases).split(";") if x}
        aliases[pid] = {x for x in vals if x}
        canonical[pid] = str(row.canonical_accession) or pid
    return aliases, canonical, sequence


def choose_uid(
    pid: str,
    aliases: dict[str, set[str]],
    canonical: dict[str, str],
    existing: set[str],
    pockets: set[str],
) -> tuple[str, str]:
    vals = aliases.get(pid, {pid})
    can = canonical.get(pid, pid)
    existing_hits = sorted(vals & existing)
    if existing_hits:
        return (can if can in existing_hits else existing_hits[0], "existing_feature")
    pocket_hits = sorted(vals & pockets)
    if pocket_hits:
        return (can if can in pocket_hits else pocket_hits[0], "author_pocket")
    return (can, "fallback_structure")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    existing_frame = pd.read_csv(EXISTING, dtype=str).fillna("")
    existing = set(
        existing_frame.loc[
            existing_frame.mapping_ok.astype(str).str.lower().eq("true"),
            "UniprotID",
        ].astype(str)
    )

    aliases, canonical, sequence = metadata_maps()
    with ZipFile(ZIP) as zf:
        pockets = {
            Path(name).stem
            for name in zf.namelist()
            if name.startswith(POCKET_PREFIX) and name.endswith(".pdb")
        }
        with zf.open(UID2SEQ_PATH) as fh:
            author_sequence = pickle.loads(fh.read())

    author_sequence = {str(k): str(v) for k, v in author_sequence.items()}

    r2e_native = pd.read_csv(
        CAND / "r2e_native_candidates.csv.gz", dtype=str
    ).fillna("")
    r2e_broad = pd.read_csv(
        CAND / "r2e_broad_equal_budget_candidates.csv.gz", dtype=str
    ).fillna("")
    e2r_native = pd.read_csv(
        CAND / "e2r_similarity_gate_candidates.csv.gz", dtype=str
    ).fillna("")
    e2r_broad = pd.read_csv(
        CAND / "e2r_broad_equal_budget_candidates.csv.gz", dtype=str
    ).fillna("")
    targets = pd.read_csv(TARGETS, dtype=str).fillna("")

    required_general: dict[str, set[str]] = {}
    for pid in sorted(set(r2e_broad.protein_id.astype(str))):
        required_general.setdefault(pid, set()).add("r2e_broad_candidate")
    for pid in sorted(set(e2r_native.protein_id.astype(str)) | set(e2r_broad.protein_id.astype(str))):
        required_general.setdefault(pid, set()).add("e2r_query")
    for pid in sorted(set(targets.protein_id.astype(str))):
        if pid in required_general:
            required_general[pid].add("strict_target")

    rows: list[dict[str, object]] = []
    for pid, scopes in sorted(required_general.items()):
        uid, source = choose_uid(pid, aliases, canonical, existing, pockets)
        seq = sequence.get(pid, "") or author_sequence.get(uid, "")
        rows.append(
            {
                "entity_kind": "general_merged",
                "entity_id": pid,
                "chosen_uid": uid,
                "scopes": ";".join(sorted(scopes)),
                "sequence": seq,
                "existing_feature": uid in existing,
                "author_pocket": uid in pockets,
                "support_source": source,
                "sequence_available": bool(seq),
            }
        )

    native_uid_mapping: dict[str, set[str]] = {}
    for uid, values in r2e_native.groupby("candidate_uid", sort=False)[
        "candidate_protein_ids"
    ]:
        mapped: set[str] = set()
        for value in values.astype(str):
            mapped.update(x for x in value.split(";") if x)
        native_uid_mapping[str(uid)] = mapped

    native_uids = sorted(native_uid_mapping)
    for uid in native_uids:
        seq = author_sequence.get(uid, "")
        mapped_pids = sorted(native_uid_mapping.get(uid, set()))
        if not seq:
            for pid in mapped_pids:
                if sequence.get(pid):
                    seq = sequence[pid]
                    break
        rows.append(
            {
                "entity_kind": "native_cage_uid",
                "entity_id": uid,
                "chosen_uid": uid,
                "scopes": "r2e_native_candidate",
                "sequence": seq,
                "existing_feature": uid in existing,
                "author_pocket": uid in pockets,
                "support_source": (
                    "existing_feature"
                    if uid in existing
                    else "author_pocket"
                    if uid in pockets
                    else "fallback_structure"
                ),
                "sequence_available": bool(seq),
            }
        )

    registry = pd.DataFrame(rows)
    registry.to_csv(OUT / "protein_support_registry.csv.gz", index=False)

    targets_unique = (
        registry[
            ~registry.existing_feature.astype(bool)
        ][
            [
                "chosen_uid",
                "sequence",
                "author_pocket",
                "support_source",
                "sequence_available",
            ]
        ]
        .drop_duplicates("chosen_uid")
        .sort_values("chosen_uid", kind="stable")
        .reset_index(drop=True)
    )
    targets_unique.to_csv(OUT / "feature_targets.csv", index=False)

    by_source = (
        registry.groupby(["entity_kind", "support_source"], dropna=False)
        .size()
        .to_dict()
    )
    summary = {
        "schema": "bridge-layered-v4-cage-support",
        "status": "completed",
        "existing_feature_uids": int(len(existing)),
        "author_rhea2025_pocket_uids": int(len(pockets)),
        "registry_rows": int(len(registry)),
        "unique_chosen_uids": int(registry.chosen_uid.nunique()),
        "feature_targets": int(len(targets_unique)),
        "feature_targets_with_author_pocket": int(
            targets_unique.author_pocket.astype(bool).sum()
        ),
        "feature_targets_need_fallback_structure": int(
            (targets_unique.support_source == "fallback_structure").sum()
        ),
        "feature_targets_missing_sequence": int(
            (~targets_unique.sequence_available.astype(bool)).sum()
        ),
        "r2e_native_unique_uids": int(len(native_uids)),
        "r2e_broad_unique_proteins": int(r2e_broad.protein_id.nunique()),
        "e2r_query_proteins": int(
            len(set(e2r_native.protein_id.astype(str)) | set(e2r_broad.protein_id.astype(str)))
        ),
        "by_source": {
            f"{kind}:{source}": int(count)
            for (kind, source), count in sorted(by_source.items())
        },
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
