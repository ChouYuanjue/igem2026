from __future__ import annotations

"""Deterministically rebuild the frozen Rhea128→141 v2 support for release replay.

This is a release assembler, not a replacement scientific protocol.  It reuses the
original frozen v2 mapping/selection implementation and reconstructs the compact-2023
alignment witness from the tracked clean2023 boundary, avoiding a server-local cache as
a hard dependency.  No model scores or target-dependent tuning enter this step.
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from projects.active.terpene_screening.prepare_rhea_snapshot_delta_external_benchmark_v2 import (
    CELL,
    DEFAULT_CLEAN,
    DEFAULT_PROTEIN_META,
    DEFAULT_REACTIONS,
    build,
    load_release,
)

EXPECTED_TEST_PAIRS_SHA256 = "9a53a465e6327e2c04a4fdd6171abd7d076aec2a3441a34955bf0f4526bc3334"
EXPECTED_TEST_PAIRS = 1122
EXPECTED_QUERY_REACTIONS = 208


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def compact_alignment_witness(clean: pd.DataFrame) -> pd.DataFrame:
    """Represent the tracked clean boundary in the frozen builder's source-ID schema."""
    clean = clean[["protein_id", "reaction_id"]].drop_duplicates().copy()
    return clean.rename(
        columns={"protein_id": "protein_source_id", "reaction_id": "reaction_source_id"}
    )


def rebuild(
    release128_sprot: Path,
    release141_sprot: Path,
    output_root: Path,
    *,
    clean2023: Path = DEFAULT_CLEAN,
    protein_metadata: Path = DEFAULT_PROTEIN_META,
    reaction_entries: Path = DEFAULT_REACTIONS,
) -> dict[str, object]:
    old = load_release(release128_sprot.resolve())
    new = load_release(release141_sprot.resolve())
    clean = pd.read_csv(clean2023.resolve(), dtype=str).fillna("")
    compact = compact_alignment_witness(clean)
    meta = pd.read_csv(protein_metadata.resolve(), dtype=str).fillna("")
    reactions = pd.read_csv(reaction_entries.resolve(), dtype=str).fillna("")
    reaction_ids = set(reactions.reaction_id.astype(str))

    train, test, trigger, audit = build(old, new, compact, clean, meta, reaction_ids)
    cell = output_root.resolve() / CELL
    cell.mkdir(parents=True, exist_ok=True)
    train_path = cell / "train_pairs.csv"
    test_path = cell / "test_pairs.csv"
    trigger_path = cell / "delta_trigger_pairs.csv"
    train.to_csv(train_path, index=False)
    test.to_csv(test_path, index=False)
    trigger.to_csv(trigger_path, index=False)

    digest = sha256_file(test_path)
    if digest != EXPECTED_TEST_PAIRS_SHA256:
        raise AssertionError(
            f"strict-support rebuild drift: {digest} != {EXPECTED_TEST_PAIRS_SHA256}"
        )
    if int(audit["test_pairs"]) != EXPECTED_TEST_PAIRS:
        raise AssertionError(f"test-pair count drift: {audit['test_pairs']}")
    if int(audit["test_query_reactions"]) != EXPECTED_QUERY_REACTIONS:
        raise AssertionError(f"query count drift: {audit['test_query_reactions']}")

    result = {
        "schema_version": 1,
        "role": "release_rebuild_of_frozen_rhea128_to141_v2_support",
        "frozen_builder": "projects/active/terpene_screening/prepare_rhea_snapshot_delta_external_benchmark_v2.py",
        "frozen_protocol": "projects/active/terpene_screening/CLEANROOM_R2E_RHEA128_TO141_EXTERNAL_V2.json",
        "alignment_witness": "constructed deterministically from tracked clean2023 training_pairs.csv; no historical compact cache required",
        "release128_sprot": str(release128_sprot.resolve()),
        "release128_sprot_sha256": sha256_file(release128_sprot.resolve()),
        "release141_sprot": str(release141_sprot.resolve()),
        "release141_sprot_sha256": sha256_file(release141_sprot.resolve()),
        "clean2023": str(clean2023.resolve()),
        "clean2023_sha256": sha256_file(clean2023.resolve()),
        "protein_metadata": str(protein_metadata.resolve()),
        "protein_metadata_sha256": sha256_file(protein_metadata.resolve()),
        "reaction_entries": str(reaction_entries.resolve()),
        "reaction_entries_sha256": sha256_file(reaction_entries.resolve()),
        "test_pairs": EXPECTED_TEST_PAIRS,
        "test_query_reactions": EXPECTED_QUERY_REACTIONS,
        "test_pairs_sha256": digest,
        "model_scores_read": False,
        "scientific_protocol_modified": False,
    }
    (output_root.resolve() / "release_rebuild_manifest.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rebuild the frozen Rhea128→141 v2 strict support without model scoring."
    )
    parser.add_argument("--release128-sprot", type=Path, required=True)
    parser.add_argument("--release141-sprot", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--clean2023", type=Path, default=DEFAULT_CLEAN)
    parser.add_argument("--protein-metadata", type=Path, default=DEFAULT_PROTEIN_META)
    parser.add_argument("--reaction-entries", type=Path, default=DEFAULT_REACTIONS)
    args = parser.parse_args()
    result = rebuild(
        args.release128_sprot,
        args.release141_sprot,
        args.output_root,
        clean2023=args.clean2023,
        protein_metadata=args.protein_metadata,
        reaction_entries=args.reaction_entries,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
