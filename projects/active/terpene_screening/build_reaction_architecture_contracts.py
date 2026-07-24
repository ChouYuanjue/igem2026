from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from projects.active.terpene_screening.analyze_uniprot_expansion_quality import (  # noqa: E402
    pfam_architecture,
)

DEFAULT_MARTS = ROOT / "data/terpene_marts/marts_reaction_pairs.tsv"
DEFAULT_REGISTRY = ROOT / "data/terpene_open_world_registry/reactions.csv"
DEFAULT_NORMALIZED = ROOT / "data/terpene_uniprot_expansion/uniprot_tps_normalized.tsv"
DEFAULT_MMSEQS = ROOT / "data/assets/mmseqs2/mmseqs/bin/mmseqs"
DEFAULT_OUTPUT = ROOT / "data/terpene_uniprot_expansion/reaction_architecture_contracts"
CLASS_I_COMPLETE = {"bacterial_classI", "plant_tps_full", "classI_hybrid_full"}


def normalize_sequence(value: object) -> str:
    return "".join(str(value).upper().split()).rstrip("*")


def allowed_architectures(mapped: set[str]) -> tuple[set[str], str]:
    mapped = {value for value in mapped if value and value != "unsupported_architecture"}
    allowed: set[str] = set()
    if mapped & CLASS_I_COMPLETE:
        allowed.update(CLASS_I_COMPLETE)
    if "osc_full" in mapped:
        allowed.add("osc_full")
    if "classII_cyclase_single_domain" in mapped:
        allowed.update({"classII_cyclase_single_domain", "plant_tps_full"})
    if not allowed:
        return set(), "unsupported_or_unresolved_reference_family"
    if len(mapped) > 1:
        return allowed, "multi_architecture_reference"
    return allowed, "reference_supported"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build reaction-specific UniProt TPS architecture contracts from known positives."
    )
    parser.add_argument("--marts", type=Path, default=DEFAULT_MARTS)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--normalized", type=Path, default=DEFAULT_NORMALIZED)
    parser.add_argument("--mmseqs", type=Path, default=DEFAULT_MMSEQS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--minimum-identity", type=float, default=0.25)
    parser.add_argument("--minimum-query-coverage", type=float, default=0.75)
    parser.add_argument("--minimum-target-coverage", type=float, default=0.50)
    parser.add_argument("--maximum-evalue", type=float, default=1e-10)
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    marts = pd.read_csv(args.marts, sep="\t", dtype=str).fillna("")
    registry = pd.read_csv(args.registry, dtype=str).fillna("")
    normalized = pd.read_csv(args.normalized, sep="\t", dtype=str).fillna("")
    normalized["sequence_normalized"] = normalized["sequence"].map(normalize_sequence)
    normalized = normalized.drop_duplicates("accession")
    by_id = normalized.set_index("accession")
    by_sequence = normalized.drop_duplicates("sequence_normalized").set_index(
        "sequence_normalized"
    )
    registered = registry[["reaction_id", "reaction_signature"]].merge(
        marts[
            [
                "reaction_signature",
                "enzyme_id",
                "enzyme_name",
                "sequence",
                "substrate_name",
                "product_name",
                "terpene_type",
                "tps_class",
            ]
        ],
        on="reaction_signature",
        how="left",
    ).fillna("")
    registered["sequence_normalized"] = registered["sequence"].map(normalize_sequence)
    registered = registered.drop_duplicates(
        ["reaction_id", "enzyme_id", "sequence_normalized"]
    )

    evidence_rows: list[dict[str, object]] = []
    unresolved_rows: list[dict[str, object]] = []
    for row in registered.itertuples(index=False):
        base = {
            "reaction_id": str(row.reaction_id),
            "reaction_signature": str(row.reaction_signature),
            "enzyme_id": str(row.enzyme_id),
            "enzyme_name": str(row.enzyme_name),
            "sequence_length": len(str(row.sequence_normalized)),
            "substrate_name": str(row.substrate_name),
            "product_name": str(row.product_name),
            "terpene_type": str(row.terpene_type),
            "tps_class": str(row.tps_class),
        }
        if row.enzyme_id in by_id.index:
            match = by_id.loc[row.enzyme_id]
            evidence_rows.append(
                {
                    **base,
                    "mapping_source": "accession",
                    "matched_accession": str(row.enzyme_id),
                    "pident": 1.0,
                    "qcov": 1.0,
                    "tcov": 1.0,
                    "evalue": 0.0,
                    "pfam_combination": str(match.pfam_combination),
                    "reference_architecture": pfam_architecture(
                        str(match.pfam_combination)
                    ),
                }
            )
        elif row.sequence_normalized and row.sequence_normalized in by_sequence.index:
            match = by_sequence.loc[row.sequence_normalized]
            evidence_rows.append(
                {
                    **base,
                    "mapping_source": "exact_sequence",
                    "matched_accession": str(match.name),
                    "pident": 1.0,
                    "qcov": 1.0,
                    "tcov": 1.0,
                    "evalue": 0.0,
                    "pfam_combination": str(match.pfam_combination),
                    "reference_architecture": pfam_architecture(
                        str(match.pfam_combination)
                    ),
                }
            )
        elif row.sequence_normalized:
            unresolved_rows.append({**base, "sequence": row.sequence_normalized})
        else:
            evidence_rows.append(
                {
                    **base,
                    "mapping_source": "missing_sequence",
                    "matched_accession": "",
                    "pident": "",
                    "qcov": "",
                    "tcov": "",
                    "evalue": "",
                    "pfam_combination": "",
                    "reference_architecture": "unsupported_architecture",
                }
            )

    unresolved = pd.DataFrame(unresolved_rows)
    if len(unresolved):
        query_fasta = output_dir / "unresolved_reference_sequences.fasta"
        with query_fasta.open("w", encoding="utf-8") as handle:
            for index, row in unresolved.reset_index(drop=True).iterrows():
                handle.write(f">query_{index}\n{row['sequence']}\n")
        target_fasta = output_dir / "normalized_tps_sequences.fasta"
        if not target_fasta.exists():
            with target_fasta.open("w", encoding="utf-8") as handle:
                for row in normalized.itertuples(index=False):
                    handle.write(f">{row.accession}\n{row.sequence_normalized}\n")
        prefix = output_dir / "reference_search"
        temporary = output_dir / "mmseqs_tmp"
        shutil.rmtree(temporary, ignore_errors=True)
        subprocess.run(
            [
                str(args.mmseqs.resolve()),
                "easy-search",
                str(query_fasta),
                str(target_fasta),
                str(prefix.with_suffix(".tsv")),
                str(temporary),
                "--format-output",
                "query,target,pident,alnlen,qcov,tcov,evalue,bits",
                "--max-seqs",
                "5",
                "--threads",
                str(args.threads),
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        hits_path = prefix.with_suffix(".tsv")
        if hits_path.exists() and hits_path.stat().st_size:
            hits = pd.read_csv(
                hits_path,
                sep="\t",
                header=None,
                names=[
                    "query",
                    "target",
                    "pident",
                    "alnlen",
                    "qcov",
                    "tcov",
                    "evalue",
                    "bits",
                ],
            )
        else:
            hits = pd.DataFrame(
                columns=[
                    "query",
                    "target",
                    "pident",
                    "alnlen",
                    "qcov",
                    "tcov",
                    "evalue",
                    "bits",
                ]
            )
        if len(hits):
            hits["query_index"] = hits["query"].str.replace("query_", "", regex=False).astype(int)
            hits["pident_fraction"] = pd.to_numeric(hits["pident"]) / 100.0
            for column in ["qcov", "tcov", "evalue", "bits"]:
                hits[column] = pd.to_numeric(hits[column])
        metadata = normalized.set_index("accession")
        for index, row in unresolved.reset_index(drop=True).iterrows():
            local = hits[hits.get("query_index", pd.Series(dtype=int)).eq(index)].copy()
            if len(local):
                local = local.sort_values(["bits", "pident_fraction"], ascending=False)
                best = local.iloc[0]
                accepted = (
                    float(best["pident_fraction"]) >= args.minimum_identity
                    and float(best["qcov"]) >= args.minimum_query_coverage
                    and float(best["tcov"]) >= args.minimum_target_coverage
                    and float(best["evalue"]) <= args.maximum_evalue
                )
            else:
                best = None
                accepted = False
            if accepted and best is not None and str(best["target"]) in metadata.index:
                match = metadata.loc[str(best["target"])]
                evidence_rows.append(
                    {
                        **{key: row[key] for key in row.index if key != "sequence"},
                        "mapping_source": "mmseqs_high_coverage",
                        "matched_accession": str(best["target"]),
                        "pident": float(best["pident_fraction"]),
                        "qcov": float(best["qcov"]),
                        "tcov": float(best["tcov"]),
                        "evalue": float(best["evalue"]),
                        "pfam_combination": str(match.pfam_combination),
                        "reference_architecture": pfam_architecture(
                            str(match.pfam_combination)
                        ),
                    }
                )
            else:
                evidence_rows.append(
                    {
                        **{key: row[key] for key in row.index if key != "sequence"},
                        "mapping_source": "no_supported_five_pfam_match",
                        "matched_accession": str(best["target"]) if best is not None else "",
                        "pident": float(best["pident_fraction"]) if best is not None else "",
                        "qcov": float(best["qcov"]) if best is not None else "",
                        "tcov": float(best["tcov"]) if best is not None else "",
                        "evalue": float(best["evalue"]) if best is not None else "",
                        "pfam_combination": "",
                        "reference_architecture": "unsupported_architecture",
                    }
                )

    evidence = pd.DataFrame(evidence_rows)
    evidence.to_csv(output_dir / "reference_architecture_evidence.csv", index=False)
    contract_rows: list[dict[str, object]] = []
    for reaction_id, group in evidence.groupby("reaction_id", sort=True):
        mapped = set(
            group.loc[
                ~group["reference_architecture"].eq("unsupported_architecture"),
                "reference_architecture",
            ].astype(str)
        )
        allowed, status = allowed_architectures(mapped)
        representative = group.iloc[0]
        contract_rows.append(
            {
                "reaction_id": reaction_id,
                "reaction_signature": representative["reaction_signature"],
                "substrate_name": representative["substrate_name"],
                "product_name": representative["product_name"],
                "terpene_type": representative["terpene_type"],
                "tps_class": representative["tps_class"],
                "reference_positive_count": group["enzyme_id"].nunique(),
                "mapped_reference_count": int(
                    (~group["reference_architecture"].eq("unsupported_architecture")).sum()
                ),
                "mapping_sources": ";".join(sorted(set(group["mapping_source"].astype(str)))),
                "reference_architectures": ";".join(sorted(mapped)),
                "allowed_candidate_architectures": ";".join(sorted(allowed)),
                "contract_status": status,
                "rescue_supported": bool(allowed),
            }
        )
    contracts = pd.DataFrame(contract_rows).sort_values("reaction_id")
    contracts.to_csv(output_dir / "reaction_architecture_contracts.csv", index=False)
    summary = {
        "registered_reactions": contracts["reaction_id"].nunique(),
        "rescue_supported_reactions": int(contracts["rescue_supported"].sum()),
        "unsupported_or_unresolved_reactions": int((~contracts["rescue_supported"]).sum()),
        "contract_status": contracts["contract_status"].value_counts().to_dict(),
        "mapping_sources": evidence["mapping_source"].value_counts().to_dict(),
        "thresholds": {
            "minimum_identity": args.minimum_identity,
            "minimum_query_coverage": args.minimum_query_coverage,
            "minimum_target_coverage": args.minimum_target_coverage,
            "maximum_evalue": args.maximum_evalue,
        },
        "outputs": {
            "contracts": str(output_dir / "reaction_architecture_contracts.csv"),
            "evidence": str(output_dir / "reference_architecture_evidence.csv"),
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(contracts[~contracts["rescue_supported"]].to_string(index=False))


if __name__ == "__main__":
    main()
