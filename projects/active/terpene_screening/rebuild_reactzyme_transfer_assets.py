from __future__ import annotations

import argparse
import hashlib
import sys
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile

import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Imports below intentionally happen after the repository root is made importable.
from projects.active.terpene_screening.common import canonicalize_reaction_smiles  # noqa: E402
from projects.active.terpene_screening.gate_matrix import (  # noqa: E402
    largest_organic_component,
    mol_fp,
    precursor_class_from_reaction,
    split_reaction_smiles,
    tanimoto,
)

RAW_ROOT = Path("data/external/reactzyme")
EXPANDED_PAIRS = Path("results/terpene_reactzyme_transfer_audit_v1/reactzyme_expanded_pairs.csv")
UNIQUE_SEQUENCES = Path("data/external/reactzyme_transfer/unique_sequences.tsv")
GENERAL_ENTRIES = Path("data/external/reactzyme_transfer/esmc600m_mean/entries.csv")
REACTION_AUDIT = Path("data/external/reactzyme_transfer/global_clean_v2/reaction_overlap_audit.csv")
TPS_POSITIVES = Path("data/terpene/enzyme_terpene_synthase.tsv")

EXPECTED = {
    str(EXPANDED_PAIRS): (118375762, "3eac711124e05fad84cd946734818e45b28cb23ca445d98984eff1b6e1d69cef"),
    str(UNIQUE_SEQUENCES): (77395594, "f934bafbf6ad451ff4cd480b512eb1543790cc9ffeb188196eef87b506483c99"),
    str(GENERAL_ENTRIES): (5952018, "ff65c94912bdc0fc93376189d0330e6dd7a0a391e211a40b2324840cfd9d4e80"),
    str(REACTION_AUDIT): (2926793, "78f2a3a8a8bb64b4d7277f996de2c2faf22dc1619bf78aaa95547f0382751ce1"),
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def target(workspace: Path, relative: Path) -> Path:
    path = workspace / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def build_expanded_pairs(raw_root: Path) -> pd.DataFrame:
    frame = pd.read_csv(raw_root / "cleaned_uniprot_rhea.tsv", sep="\t", dtype=str).fillna("")
    required = {"Entry", "Rhea ID", "Sequence"}
    if not required.issubset(frame.columns):
        raise ValueError(f"cleaned_uniprot_rhea.tsv missing columns: {sorted(required - set(frame.columns))}")
    rows: list[tuple[str, str, str]] = []
    for entry, rhea_ids, sequence in frame[["Entry", "Rhea ID", "Sequence"]].itertuples(index=False, name=None):
        for rhea_id in str(rhea_ids).split(";"):
            rhea_id = rhea_id.strip()
            if rhea_id:
                rows.append((str(entry), rhea_id, str(sequence)))
    return (
        pd.DataFrame(rows, columns=["Entry", "rhea_id", "Sequence"])
        .drop_duplicates(["Entry", "rhea_id", "Sequence"], keep="first")
        .reset_index(drop=True)
    )


def load_split_member(archive: Path, member: str) -> dict:
    with ZipFile(archive) as handle:
        if member not in handle.namelist():
            raise FileNotFoundError(f"{member} not found in {archive}")
        return torch.load(BytesIO(handle.read(member)), map_location="cpu", weights_only=False)


def build_unique_sequences(raw_root: Path) -> pd.DataFrame:
    archive = raw_root / "enzyme_smi_split.zip"
    sequences: set[str] = set()
    for member in ("positive_train_val_seq_smi.pt", "positive_test_seq_smi.pt"):
        payload = load_split_member(archive, member)
        values = payload.values() if isinstance(payload, dict) else payload
        for value in values:
            sequence = str(value[1]).strip()
            if sequence:
                sequences.add(sequence)
    rows = [
        ("RZSEQ_" + hashlib.sha1(sequence.encode("utf-8")).hexdigest()[:20], sequence)
        for sequence in sequences
    ]
    return pd.DataFrame(rows, columns=["Entry", "Sequence"]).sort_values("Entry").reset_index(drop=True)


def build_general_entries(unique_sequences: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({"row": range(len(unique_sequences)), "Entry": unique_sequences["Entry"].astype(str)})


def build_reaction_audit(raw_root: Path, expanded: pd.DataFrame, tps_path: Path) -> pd.DataFrame:
    rhea = pd.read_csv(raw_root / "rhea_molecules.tsv", sep="\t", dtype=str).fillna("")
    required = {"Rhea ID", "substrate", "product"}
    if not required.issubset(rhea.columns):
        raise ValueError(f"rhea_molecules.tsv missing columns: {sorted(required - set(rhea.columns))}")
    reaction_map: dict[str, str] = {}
    for rhea_id, substrate, product in rhea[["Rhea ID", "substrate", "product"]].itertuples(index=False, name=None):
        raw = f"{substrate}>>{product}"
        reaction_map[str(rhea_id)] = canonicalize_reaction_smiles(raw) or raw
    order = [rhea_id for rhea_id in dict.fromkeys(expanded["rhea_id"].astype(str)) if rhea_id in reaction_map]
    if len(order) != len(reaction_map):
        raise ValueError(f"ReactZyme Rhea ordering incomplete: {len(order)} != {len(reaction_map)}")

    tps = pd.read_csv(tps_path, sep="\t", dtype=str).fillna("")[["rhea_id", "smiles_seq"]]
    tps = tps.drop_duplicates("rhea_id", keep="first").reset_index(drop=True)
    tps["precursor"] = tps["smiles_seq"].map(precursor_class_from_reaction)
    tps["product_fp"] = tps["smiles_seq"].map(
        lambda value: mol_fp(largest_organic_component(split_reaction_smiles(value)[1]))
    )
    by_precursor = {name: list(group["product_fp"]) for name, group in tps.groupby("precursor", sort=False)}
    exact_ids = set(tps["rhea_id"].astype(str))

    rows: list[tuple[str, str, float, bool, bool]] = []
    for rhea_id in order:
        reaction = reaction_map[rhea_id]
        precursor = precursor_class_from_reaction(reaction)
        product_fp = mol_fp(largest_organic_component(split_reaction_smiles(reaction)[1]))
        similarity = max((tanimoto(product_fp, other) for other in by_precursor.get(precursor, [])), default=0.0)
        exact = rhea_id in exact_ids
        rows.append((rhea_id, reaction, similarity, exact, bool(exact or similarity >= 0.5)))
    return pd.DataFrame(
        rows,
        columns=["rhea_id", "smiles_seq", "max_tps_reaction_similarity", "exact_tps_rhea", "tps_related"],
    )


def verify_frozen(workspace: Path) -> None:
    failures: list[str] = []
    for relative, (expected_bytes, expected_sha) in EXPECTED.items():
        path = workspace / relative
        if not path.is_file():
            failures.append(f"missing output: {relative}")
            continue
        if path.stat().st_size != expected_bytes:
            failures.append(f"byte-size drift: {relative}: {path.stat().st_size} != {expected_bytes}")
        digest = sha256(path)
        if digest != expected_sha:
            failures.append(f"sha256 drift: {relative}: {digest} != {expected_sha}")
    if failures:
        raise RuntimeError("ReactZyme frozen rebuild failed:\n" + "\n".join(failures))


def main() -> None:
    parser = argparse.ArgumentParser(description="Rebuild byte-exact ReactZyme transfer intermediates used by BiME-Rank.")
    parser.add_argument("--workspace", type=Path, default=ROOT, help="Output workspace; relative paths mirror the repository.")
    parser.add_argument("--raw-root", type=Path, default=ROOT / RAW_ROOT)
    parser.add_argument("--tps-positives", type=Path, default=ROOT / TPS_POSITIVES)
    parser.add_argument("--skip-frozen-check", action="store_true")
    args = parser.parse_args()

    workspace = args.workspace.resolve()
    expanded = build_expanded_pairs(args.raw_root.resolve())
    expanded.to_csv(target(workspace, EXPANDED_PAIRS), index=False)
    unique = build_unique_sequences(args.raw_root.resolve())
    unique.to_csv(target(workspace, UNIQUE_SEQUENCES), sep="\t", index=False)
    build_general_entries(unique).to_csv(target(workspace, GENERAL_ENTRIES), index=False)
    build_reaction_audit(args.raw_root.resolve(), expanded, args.tps_positives.resolve()).to_csv(
        target(workspace, REACTION_AUDIT), index=False
    )
    if not args.skip_frozen_check:
        verify_frozen(workspace)
    for relative in EXPECTED:
        path = workspace / relative
        print(f"{relative}\t{path.stat().st_size}\t{sha256(path)}")


if __name__ == "__main__":
    main()
