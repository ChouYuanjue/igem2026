from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
from rdkit import Chem

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from projects.active.terpene_screening.gate_matrix import (  # noqa: E402
    canonical_or_raw_reaction,
    largest_organic_component,
    split_reaction_smiles,
)

DEFAULT_MARTS_REACTIONS = ROOT / "data/external/marts_db/v1.5/reactions.csv"
DEFAULT_MARTS_MECHANISMS = ROOT / "data/external/marts_db/v1.5/mechanisms.csv"
DEFAULT_CURRENT_POSITIVES = ROOT / "data/terpene/enzyme_terpene_synthase.tsv"
DEFAULT_CURRENT_CANDIDATES = ROOT / "data/terpene/all_seq_terpene_synthase.tsv"
DEFAULT_OUTPUT = ROOT / "data/terpene_marts"


def text(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    result = str(value).strip()
    return "" if result.lower() == "nan" else result


def clean_sequence(value: object) -> str:
    return "".join(text(value).upper().split()).rstrip("*")


def canonical_molecule(smiles: str) -> str:
    if not smiles:
        return ""
    try:
        molecule = Chem.MolFromSmiles(smiles)
        return Chem.MolToSmiles(molecule, isomericSmiles=True) if molecule is not None else ""
    except Exception:
        return ""


def reaction_signature(reaction_smiles: str) -> str:
    canonical = canonical_or_raw_reaction(reaction_smiles)
    reactants, products = split_reaction_smiles(canonical)
    substrate = canonical_molecule(largest_organic_component(reactants))
    product = canonical_molecule(largest_organic_component(products))
    return f"{substrate}>>{product}" if substrate and product else ""


def choose_enzyme_id(row: pd.Series) -> tuple[str, str]:
    uniprot = text(row.get("Uniprot_ID"))
    genbank = text(row.get("Genbank_ID"))
    marts = text(row.get("Enzyme_marts_ID"))
    if uniprot:
        return uniprot, "uniprot"
    if genbank:
        return genbank, "genbank"
    return marts, "marts"


def normalize_marts_reactions(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for _, row in frame.iterrows():
        enzyme_id, id_type = choose_enzyme_id(row)
        substrate = text(row.get("Substrate_smiles"))
        product = text(row.get("Product_smiles"))
        reaction = f"{substrate}>>{product}" if substrate and product else ""
        rows.append(
            {
                "enzyme_id": enzyme_id,
                "enzyme_id_type": id_type,
                "uniprot_id": text(row.get("Uniprot_ID")),
                "genbank_id": text(row.get("Genbank_ID")),
                "marts_enzyme_id": text(row.get("Enzyme_marts_ID")),
                "enzyme_name": text(row.get("Enzyme_name")),
                "sequence": clean_sequence(row.get("Aminoacid_sequence")),
                "species": text(row.get("Species")),
                "kingdom": text(row.get("Kingdom")),
                "terpene_type": text(row.get("Type")),
                "tps_class": text(row.get("Class")),
                "substrate_name": text(row.get("Substrate_name")),
                "substrate_smiles": substrate,
                "substrate_marts_id": text(row.get("Substrate_marts_ID")),
                "product_name": text(row.get("Product_name")),
                "product_smiles": product,
                "product_marts_id": text(row.get("Product_marts_ID")),
                "reaction_smiles": reaction,
                "canonical_reaction": canonical_or_raw_reaction(reaction),
                "reaction_signature": reaction_signature(reaction),
                "has_mechanism": bool(row.get("Reaction_has_mechanism")),
                "mechanism_marts_id": text(row.get("Mechanism_marts_ID")),
                "publication": text(row.get("Publication")),
            }
        )
    result = pd.DataFrame(rows)
    result = result[(result["enzyme_id"] != "") & (result["sequence"] != "")]
    return result.drop_duplicates(["enzyme_id", "reaction_signature", "product_marts_id"], keep="first").reset_index(drop=True)


def normalize_mechanisms(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result.columns = [str(column).strip() for column in result.columns]
    for column in result.columns:
        result[column] = result[column].map(text)
    result["step_reaction_smiles"] = result.apply(
        lambda row: f"{row['Substrate_smiles']}>>{row['Product_smiles']}"
        if row["Substrate_smiles"] and row["Product_smiles"]
        else "",
        axis=1,
    )
    result["step_reaction_signature"] = result["step_reaction_smiles"].map(reaction_signature)
    result["step_index"] = result.groupby("Mechanism_marts_id").cumcount() + 1
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Normalize MARTS-DB and build open-world overlap strata.")
    parser.add_argument("--marts-reactions", type=Path, default=DEFAULT_MARTS_REACTIONS)
    parser.add_argument("--marts-mechanisms", type=Path, default=DEFAULT_MARTS_MECHANISMS)
    parser.add_argument("--current-positives", type=Path, default=DEFAULT_CURRENT_POSITIVES)
    parser.add_argument("--current-candidates", type=Path, default=DEFAULT_CURRENT_CANDIDATES)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_reactions = pd.read_csv(args.marts_reactions, dtype=str).fillna("")
    raw_mechanisms = pd.read_csv(args.marts_mechanisms, dtype=str).fillna("")
    marts = normalize_marts_reactions(raw_reactions)
    mechanisms = normalize_mechanisms(raw_mechanisms)

    current_candidates = pd.read_csv(args.current_candidates, sep="\t", dtype=str).fillna("")
    current_candidates["Sequence"] = current_candidates["Sequence"].map(clean_sequence)
    current_ids = set(current_candidates["Entry"].astype(str))
    current_sequences = set(current_candidates["Sequence"].astype(str)) - {""}

    current_positives = pd.read_csv(args.current_positives, sep="\t", dtype=str).fillna("")
    current_positives["reaction_signature"] = current_positives["smiles_seq"].map(reaction_signature)
    current_signatures = set(current_positives["reaction_signature"]) - {""}
    current_exact_reactions = set(current_positives["smiles_seq"].map(canonical_or_raw_reaction)) - {""}

    marts["enzyme_seen_by_id"] = marts["enzyme_id"].isin(current_ids)
    marts["enzyme_seen_by_sequence"] = marts["sequence"].isin(current_sequences)
    marts["enzyme_seen"] = marts["enzyme_seen_by_id"] | marts["enzyme_seen_by_sequence"]
    marts["reaction_seen_exact"] = marts["canonical_reaction"].isin(current_exact_reactions)
    marts["reaction_seen_signature"] = marts["reaction_signature"].isin(current_signatures)
    marts["reaction_seen"] = marts["reaction_seen_exact"] | marts["reaction_seen_signature"]
    marts["open_world_category"] = marts.apply(
        lambda row: (
            "seen_enzyme_seen_reaction"
            if row["enzyme_seen"] and row["reaction_seen"]
            else "unseen_enzyme_seen_reaction"
            if not row["enzyme_seen"] and row["reaction_seen"]
            else "seen_enzyme_unseen_reaction"
            if row["enzyme_seen"] and not row["reaction_seen"]
            else "unseen_enzyme_unseen_reaction"
        ),
        axis=1,
    )

    marts.to_csv(output_dir / "marts_reaction_pairs.tsv", sep="\t", index=False)
    mechanisms.to_csv(output_dir / "marts_mechanism_steps.tsv", sep="\t", index=False)
    enzyme_table = (
        marts.sort_values(["enzyme_id", "reaction_signature"])
        .drop_duplicates("enzyme_id")
        [[
            "enzyme_id",
            "enzyme_id_type",
            "uniprot_id",
            "genbank_id",
            "marts_enzyme_id",
            "enzyme_name",
            "sequence",
            "species",
            "kingdom",
            "terpene_type",
            "tps_class",
            "enzyme_seen_by_id",
            "enzyme_seen_by_sequence",
            "enzyme_seen",
        ]]
    )
    enzyme_table.to_csv(output_dir / "marts_enzymes.tsv", sep="\t", index=False)
    reaction_table = (
        marts.sort_values(["reaction_signature", "enzyme_id"])
        .drop_duplicates("reaction_signature")
        [[
            "reaction_signature",
            "canonical_reaction",
            "substrate_name",
            "substrate_smiles",
            "product_name",
            "product_smiles",
            "terpene_type",
            "has_mechanism",
            "mechanism_marts_id",
            "reaction_seen_exact",
            "reaction_seen_signature",
            "reaction_seen",
        ]]
    )
    reaction_table.to_csv(output_dir / "marts_reactions.tsv", sep="\t", index=False)

    category_counts = marts["open_world_category"].value_counts().to_dict()
    category_unique_enzymes = marts.groupby("open_world_category")["enzyme_id"].nunique().to_dict()
    category_unique_reactions = marts.groupby("open_world_category")["reaction_signature"].nunique().to_dict()
    mechanism_sizes = mechanisms.groupby("Mechanism_marts_id").size() if not mechanisms.empty else pd.Series(dtype=int)
    summary = {
        "source_reaction_rows": int(len(raw_reactions)),
        "normalized_reaction_pairs": int(len(marts)),
        "unique_marts_enzymes": int(marts["enzyme_id"].nunique()),
        "unique_marts_sequences": int(marts["sequence"].nunique()),
        "unique_marts_reaction_signatures": int(marts["reaction_signature"].replace("", pd.NA).nunique()),
        "unique_products": int(marts["product_marts_id"].replace("", pd.NA).nunique()),
        "pairs_with_mechanism": int(marts["has_mechanism"].sum()),
        "mechanism_step_rows": int(len(mechanisms)),
        "unique_mechanisms": int(mechanisms["Mechanism_marts_id"].replace("", pd.NA).nunique()),
        "median_steps_per_mechanism": float(mechanism_sizes.median()) if len(mechanism_sizes) else 0.0,
        "max_steps_per_mechanism": int(mechanism_sizes.max()) if len(mechanism_sizes) else 0,
        "enzymes_seen_by_id": int(enzyme_table["enzyme_seen_by_id"].sum()),
        "enzymes_seen_by_sequence": int(enzyme_table["enzyme_seen_by_sequence"].sum()),
        "enzymes_unseen": int((~enzyme_table["enzyme_seen"]).sum()),
        "reactions_seen": int(reaction_table["reaction_seen"].sum()),
        "reactions_unseen": int((~reaction_table["reaction_seen"]).sum()),
        "category_pair_counts": {str(key): int(value) for key, value in category_counts.items()},
        "category_unique_enzymes": {str(key): int(value) for key, value in category_unique_enzymes.items()},
        "category_unique_reactions": {str(key): int(value) for key, value in category_unique_reactions.items()},
        "outputs": {
            "reaction_pairs": str(output_dir / "marts_reaction_pairs.tsv"),
            "enzymes": str(output_dir / "marts_enzymes.tsv"),
            "reactions": str(output_dir / "marts_reactions.tsv"),
            "mechanism_steps": str(output_dir / "marts_mechanism_steps.tsv"),
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
