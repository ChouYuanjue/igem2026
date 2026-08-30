from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_PAIRS = ROOT / "data/external/enzymecage_current/Enzyme-405.csv"
DEFAULT_REFERENCE = ROOT / "data/catalyst_candidate_universes/general_merged/protein_sequences.tsv"
DEFAULT_OUTPUT = ROOT / "results/enzymecage_405_sequence_consistency_v1"


def sequence_hash(sequence: str) -> str:
    return hashlib.sha256(str(sequence).encode("utf-8")).hexdigest()


def audit_sequences(pairs: pd.DataFrame, reference: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    required_pairs = {"UniprotID", "sequence", "Label"}
    if not required_pairs <= set(pairs.columns):
        raise ValueError(f"pairs missing {sorted(required_pairs - set(pairs.columns))}")
    required_reference = {"protein_id", "sequence"}
    if not required_reference <= set(reference.columns):
        raise ValueError(f"reference missing {sorted(required_reference - set(reference.columns))}")

    pairs = pairs.copy().fillna("")
    pairs["Label"] = pd.to_numeric(pairs["Label"], errors="raise").astype(int)
    uid_sequences = pairs[["UniprotID", "sequence"]].drop_duplicates()
    multi_sequence_uids = uid_sequences.groupby("UniprotID")["sequence"].nunique()
    if (multi_sequence_uids > 1).any():
        raise ValueError(f"Enzyme-405 contains UIDs with multiple sequences: {multi_sequence_uids[multi_sequence_uids > 1].index[:5].tolist()}")
    uid_sequences = uid_sequences.drop_duplicates("UniprotID", keep="first")
    reference = reference[["protein_id", "sequence"]].fillna("").drop_duplicates("protein_id", keep="first")
    reference = reference.rename(columns={"protein_id": "UniprotID", "sequence": "reference_sequence"})
    frame = uid_sequences.merge(reference, on="UniprotID", how="left", validate="one_to_one").fillna("")
    labels = pairs.groupby("UniprotID")["Label"].agg(positive_rows="sum", pair_rows="size").reset_index()
    frame = frame.merge(labels, on="UniprotID", validate="one_to_one")
    frame["reference_covered"] = frame["reference_sequence"].ne("")
    frame["exact_sequence_match"] = frame["reference_covered"] & frame["sequence"].eq(frame["reference_sequence"])
    frame["benchmark_sequence_length"] = frame["sequence"].str.len()
    frame["reference_sequence_length"] = frame["reference_sequence"].str.len()
    frame["benchmark_sequence_sha256"] = frame["sequence"].map(sequence_hash)
    frame["reference_sequence_sha256"] = frame["reference_sequence"].map(sequence_hash)
    frame["sequence_status"] = "reference_missing"
    frame.loc[frame["reference_covered"] & frame["exact_sequence_match"], "sequence_status"] = "exact_match"
    frame.loc[frame["reference_covered"] & ~frame["exact_sequence_match"], "sequence_status"] = "mismatch"

    groups = (
        frame.groupby(["benchmark_sequence_sha256", "benchmark_sequence_length"], dropna=False)
        .agg(
            uid_count=("UniprotID", "nunique"),
            positive_uid_count=("positive_rows", lambda values: int((values > 0).sum())),
            mismatch_uid_count=("sequence_status", lambda values: int((values == "mismatch").sum())),
            reference_missing_uid_count=("sequence_status", lambda values: int((values == "reference_missing").sum())),
            uids=("UniprotID", lambda values: ";".join(sorted(map(str, set(values))))),
        )
        .reset_index()
        .sort_values(["uid_count", "mismatch_uid_count", "benchmark_sequence_sha256"], ascending=[False, False, True])
        .reset_index(drop=True)
    )
    covered = frame[frame["reference_covered"]]
    mismatched = covered[~covered["exact_sequence_match"]]
    duplicated = groups[groups["uid_count"] > 1]
    summary = {
        "protocol": "Enzyme-405 sequence provenance audit; observation only, no automatic correction",
        "benchmark_uids": int(len(frame)),
        "reference_covered_uids": int(len(covered)),
        "reference_missing_uids": int((~frame["reference_covered"]).sum()),
        "exact_match_uids": int(covered["exact_sequence_match"].sum()),
        "mismatch_uids": int(len(mismatched)),
        "exact_match_fraction_among_reference_covered": float(covered["exact_sequence_match"].mean()) if len(covered) else None,
        "mismatched_positive_uids": int((mismatched["positive_rows"] > 0).sum()),
        "mismatched_positive_rows": int(mismatched["positive_rows"].sum()),
        "mismatched_negative_only_uids": int((mismatched["positive_rows"] == 0).sum()),
        "duplicate_sequence_groups": int(len(duplicated)),
        "uids_in_duplicate_sequence_groups": int(duplicated["uid_count"].sum()),
        "largest_duplicate_sequence_group": None if duplicated.empty else duplicated.iloc[0].to_dict(),
        "automatic_sequence_correction_performed": False,
        "interpretation_limit": (
            "A mismatch against the Catalyst reference is evidence of sequence-version or construction inconsistency, "
            "not proof that the benchmark source is wrong. The benchmark copy remains unchanged."
        ),
    }
    return frame, groups, summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit Enzyme-405 UID/sequence consistency without modifying benchmark data.")
    parser.add_argument("--pairs", type=Path, default=DEFAULT_PAIRS)
    parser.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    pairs = pd.read_csv(args.pairs, dtype=str).fillna("")
    reference = pd.read_csv(args.reference, sep="\t", dtype=str).fillna("")
    frame, groups, summary = audit_sequences(pairs, reference)
    out = args.output_dir.resolve(); out.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out / "uid_sequence_audit.csv", index=False)
    groups.to_csv(out / "duplicate_sequence_groups.csv", index=False)
    (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
