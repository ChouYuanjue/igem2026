from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_OUTPUT = ROOT / "data/terpene_uniprot_expansion"
DEFAULT_CURRENT = ROOT / "data/terpene/all_seq_terpene_synthase.tsv"
DEFAULT_MARTS = ROOT / "data/terpene_marts/marts_reaction_pairs.tsv"
DEFAULT_MMSEQS = ROOT / "data/assets/mmseqs2/mmseqs/bin/mmseqs"
PFAM_DOMAINS = ("PF01397", "PF03936", "PF19086", "PF13249", "PF13243")
UNIPROT_FIELDS = (
    "accession,id,reviewed,protein_name,gene_names,organism_name,organism_id,"
    "length,sequence,xref_pfam,protein_existence"
)
PROTEIN_EXISTENCE_PRIORITY = {
    "Evidence at protein level": 1,
    "Evidence at transcript level": 2,
    "Inferred from homology": 3,
    "Predicted": 4,
    "Uncertain": 5,
}


def build_query() -> str:
    domains = " OR ".join(f"xref:pfam-{value}" for value in PFAM_DOMAINS)
    return f"({domains}) AND length:[200 TO 1000] AND fragment:false"


def download_uniprot(path: Path, query: str, force: bool) -> None:
    if path.exists() and path.stat().st_size > 0 and not force:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    url = "https://rest.uniprot.org/uniprotkb/stream"
    params = {
        "query": query,
        "format": "tsv",
        "fields": UNIPROT_FIELDS,
        "compressed": "false",
    }
    last_error: Exception | None = None
    for attempt in range(1, 6):
        try:
            with requests.get(url, params=params, stream=True, timeout=(30, 300)) as response:
                response.raise_for_status()
                with temporary.open("wb") as handle:
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            handle.write(chunk)
            temporary.replace(path)
            return
        except Exception as error:  # pragma: no cover - network retry path
            last_error = error
            if temporary.exists():
                temporary.unlink()
            if attempt < 5:
                time.sleep(2**attempt)
    raise RuntimeError(f"UniProt download failed after retries: {last_error}")


def clean_sequence(value: object) -> str:
    return re.sub(r"\s+", "", str(value)).upper()


def parse_pfam(value: object) -> tuple[str, ...]:
    return tuple(sorted({item for item in str(value).split(";") if item in PFAM_DOMAINS}))


def domain_family(domains: tuple[str, ...]) -> str:
    domain_set = set(domains)
    families = []
    if domain_set & {"PF01397", "PF03936"}:
        families.append("plant_like_classI_II")
    if "PF19086" in domain_set:
        families.append("bacterial_classI")
    if domain_set & {"PF13249", "PF13243"}:
        families.append("triterpene_cyclase")
    return "+".join(families) if families else "other"


def read_existing(current_path: Path, marts_path: Path) -> tuple[set[str], set[str]]:
    current = pd.read_csv(current_path, sep="\t", dtype=str).fillna("")
    marts = pd.read_csv(marts_path, sep="\t", dtype=str).fillna("")
    existing_ids = set(current["Entry"].astype(str)) | set(marts["enzyme_id"].astype(str))
    current_sequences = {clean_sequence(value) for value in current["Sequence"] if clean_sequence(value)}
    marts_sequences = {clean_sequence(value) for value in marts["sequence"] if clean_sequence(value)}
    return existing_ids, current_sequences | marts_sequences


def normalize_uniprot(frame: pd.DataFrame) -> pd.DataFrame:
    rename = {
        "Entry": "accession",
        "Entry Name": "entry_name",
        "Reviewed": "reviewed",
        "Protein names": "protein_name",
        "Gene Names": "gene_names",
        "Organism": "organism_name",
        "Organism (ID)": "organism_id",
        "Length": "length",
        "Sequence": "sequence",
        "Pfam": "pfam",
        "Protein existence": "protein_existence",
    }
    frame = frame.rename(columns=rename).fillna("")
    expected = set(rename.values())
    missing = sorted(expected - set(frame.columns))
    if missing:
        raise ValueError(f"UniProt TSV missing normalized fields: {missing}")
    frame["accession"] = frame["accession"].astype(str).str.strip()
    frame["sequence"] = frame["sequence"].map(clean_sequence)
    frame["length"] = pd.to_numeric(frame["length"], errors="coerce").astype("Int64")
    frame["reviewed"] = frame["reviewed"].astype(str).str.lower().eq("reviewed")
    frame["pfam_domains"] = frame["pfam"].map(parse_pfam)
    frame["pfam_combination"] = frame["pfam_domains"].map(lambda values: ";".join(values))
    frame["domain_family"] = frame["pfam_domains"].map(domain_family)
    frame["canonical_sequence"] = frame["sequence"].map(
        lambda value: not bool(set(value) - set("ACDEFGHIKLMNPQRSTVWY"))
    )
    frame["protein_existence_priority"] = frame["protein_existence"].map(
        PROTEIN_EXISTENCE_PRIORITY
    ).fillna(9).astype(int)
    frame["annotation_low_quality"] = frame["protein_name"].astype(str).str.lower().str.contains(
        r"uncharacterized|hypothetical|putative protein", regex=True
    )
    frame = frame[
        frame["accession"].ne("")
        & frame["sequence"].ne("")
        & frame["length"].between(200, 1000)
        & frame["canonical_sequence"]
        & frame["pfam_domains"].map(bool)
    ].copy()
    return frame.drop_duplicates("accession")


def evidence_quality_tier(row: pd.Series) -> str:
    if bool(row["reviewed"]):
        return "A_reviewed"
    priority = int(row["protein_existence_priority"])
    low_quality = bool(row["annotation_low_quality"])
    if priority <= 2 and not low_quality:
        return "B_experimental_or_transcript_named"
    if priority <= 3 and not low_quality:
        return "C_homology_named"
    if not low_quality:
        return "D_named_predicted"
    return "E_domain_only_uncharacterized"


def collapse_identical_sequences(frame: pd.DataFrame) -> pd.DataFrame:
    ordered = frame.sort_values(
        [
            "sequence",
            "reviewed",
            "protein_existence_priority",
            "annotation_low_quality",
            "accession",
        ],
        ascending=[True, False, True, True, True],
    )
    rows = []
    for sequence, group in ordered.groupby("sequence", sort=False):
        representative = group.iloc[0].copy()
        representative["sequence_alias_accessions"] = ";".join(sorted(group["accession"].astype(str)))
        representative["n_sequence_alias_accessions"] = len(group)
        representative["any_reviewed_alias"] = bool(group["reviewed"].any())
        rows.append(representative)
    return pd.DataFrame(rows).reset_index(drop=True)


def write_fasta(frame: pd.DataFrame, path: Path) -> None:
    lines = []
    for row in frame.itertuples(index=False):
        lines.append(
            f">{row.accession}|family={row.domain_family}|reviewed={str(bool(row.reviewed)).lower()}|pfam={row.pfam_combination}"
        )
        sequence = str(row.sequence)
        lines.extend(sequence[index : index + 80] for index in range(0, len(sequence), 80))
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def run_mmseqs(
    mmseqs: Path,
    fasta: Path,
    output_dir: Path,
    identity: float,
    coverage: float,
    threads: int,
    force: bool,
) -> Path:
    prefix = output_dir / f"uniprot_novel_id{int(round(identity * 100))}"
    cluster_tsv = Path(str(prefix) + "_cluster.tsv")
    if cluster_tsv.exists() and not force:
        return cluster_tsv
    temporary = output_dir / "mmseqs_tmp"
    for path in output_dir.glob(f"{prefix.name}*"):
        if path.is_file():
            path.unlink()
    subprocess.run(
        [
            str(mmseqs),
            "easy-cluster",
            str(fasta),
            str(prefix),
            str(temporary),
            "--min-seq-id",
            str(identity),
            "-c",
            str(coverage),
            "--cov-mode",
            "0",
            "--threads",
            str(threads),
        ],
        check=True,
    )
    if not cluster_tsv.exists():
        raise FileNotFoundError(cluster_tsv)
    return cluster_tsv


def choose_cluster_representatives(frame: pd.DataFrame, cluster_tsv: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    clusters = pd.read_csv(
        cluster_tsv,
        sep="\t",
        names=["mmseqs_representative", "accession"],
        dtype=str,
    )
    clusters["mmseqs_representative"] = clusters["mmseqs_representative"].str.split("|").str[0]
    clusters["accession"] = clusters["accession"].str.split("|").str[0]
    clusters = clusters.drop_duplicates("accession")
    merged = frame.merge(clusters, on="accession", how="left")
    merged["mmseqs_representative"] = merged["mmseqs_representative"].fillna(merged["accession"])
    merged["cluster_size"] = merged.groupby("mmseqs_representative")["accession"].transform("size")
    selected = []
    selection_rows = []
    for cluster_id, group in merged.groupby("mmseqs_representative", sort=True):
        reviewed = group[group["reviewed"]].copy()
        if len(reviewed):
            for _, row in reviewed.sort_values(
                ["protein_existence_priority", "annotation_low_quality", "accession"]
            ).iterrows():
                selected.append(row)
                selection_rows.append(
                    {
                        "accession": row["accession"],
                        "cluster_id": cluster_id,
                        "selection_reason": "all_reviewed_members",
                        "cluster_size": len(group),
                        "cluster_reviewed_members": len(reviewed),
                    }
                )
        else:
            row = group.sort_values(
                ["protein_existence_priority", "annotation_low_quality", "accession"]
            ).iloc[0]
            selected.append(row)
            selection_rows.append(
                {
                    "accession": row["accession"],
                    "cluster_id": cluster_id,
                    "selection_reason": "best_unreviewed_cluster_representative",
                    "cluster_size": len(group),
                    "cluster_reviewed_members": 0,
                }
            )
    selected_frame = pd.DataFrame(selected).drop(columns=["cluster_size"], errors="ignore")
    selection = pd.DataFrame(selection_rows)
    selected_frame = selected_frame.merge(selection, on="accession", how="left")
    return merged, selected_frame


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare a clustered UniProt TPS-domain expansion layer.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--current-candidates", type=Path, default=DEFAULT_CURRENT)
    parser.add_argument("--marts", type=Path, default=DEFAULT_MARTS)
    parser.add_argument("--mmseqs", type=Path, default=DEFAULT_MMSEQS)
    parser.add_argument("--identity", type=float, default=0.5)
    parser.add_argument("--coverage", type=float, default=0.8)
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--force-download", action="store_true")
    parser.add_argument("--force-cluster", action="store_true")
    args = parser.parse_args()

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = output_dir / "uniprot_tps_raw.tsv"
    query = build_query()
    download_uniprot(raw_path, query, args.force_download)
    raw = pd.read_csv(raw_path, sep="\t", dtype=str).fillna("")
    normalized = normalize_uniprot(raw)
    existing_ids, existing_sequences = read_existing(
        args.current_candidates.resolve(), args.marts.resolve()
    )
    normalized["existing_id"] = normalized["accession"].isin(existing_ids)
    normalized["existing_sequence"] = normalized["sequence"].isin(existing_sequences)
    normalized["novel_to_current_registry"] = ~(
        normalized["existing_id"] | normalized["existing_sequence"]
    )
    normalized["evidence_quality_tier"] = normalized.apply(
        evidence_quality_tier, axis=1
    )
    normalized.to_csv(output_dir / "uniprot_tps_normalized.tsv", sep="\t", index=False)
    sequence_unique = collapse_identical_sequences(normalized)
    sequence_unique.to_csv(output_dir / "uniprot_tps_sequence_unique.tsv", sep="\t", index=False)
    novel = sequence_unique[sequence_unique["novel_to_current_registry"]].copy()
    novel.to_csv(output_dir / "uniprot_tps_novel_sequence_unique.tsv", sep="\t", index=False)
    novel_fasta = output_dir / "uniprot_tps_novel_sequence_unique.fasta"
    write_fasta(novel, novel_fasta)
    cluster_tsv = run_mmseqs(
        args.mmseqs.resolve(),
        novel_fasta,
        output_dir,
        args.identity,
        args.coverage,
        args.threads,
        args.force_cluster,
    )
    clustered, selected = choose_cluster_representatives(novel, cluster_tsv)
    clustered.to_csv(output_dir / "uniprot_tps_novel_clusters.tsv", sep="\t", index=False)
    selected.to_csv(output_dir / "uniprot_tps_embedding_candidates.tsv", sep="\t", index=False)
    write_fasta(selected, output_dir / "uniprot_tps_embedding_candidates.fasta")
    primary = selected[
        ~selected["evidence_quality_tier"].eq("E_domain_only_uncharacterized")
    ].copy()
    rescue = selected[
        selected["evidence_quality_tier"].eq("E_domain_only_uncharacterized")
    ].copy()
    primary.to_csv(
        output_dir / "uniprot_tps_primary_embedding_candidates.tsv",
        sep="\t",
        index=False,
    )
    rescue.to_csv(
        output_dir / "uniprot_tps_domain_only_rescue_candidates.tsv",
        sep="\t",
        index=False,
    )
    write_fasta(
        primary, output_dir / "uniprot_tps_primary_embedding_candidates.fasta"
    )
    write_fasta(
        rescue, output_dir / "uniprot_tps_domain_only_rescue_candidates.fasta"
    )

    summary = {
        "query": query,
        "pfam_domains": PFAM_DOMAINS,
        "raw_rows": len(raw),
        "normalized_valid_rows": len(normalized),
        "reviewed_valid_rows": int(normalized["reviewed"].sum()),
        "sequence_unique_rows": len(sequence_unique),
        "existing_id_rows": int(normalized["existing_id"].sum()),
        "existing_sequence_rows": int(normalized["existing_sequence"].sum()),
        "novel_sequence_unique_rows": len(novel),
        "novel_reviewed_rows": int(novel["reviewed"].sum()),
        "mmseqs_identity": args.identity,
        "mmseqs_coverage": args.coverage,
        "novel_clusters": int(clustered["mmseqs_representative"].nunique()),
        "embedding_candidates": len(selected),
        "embedding_candidate_reviewed": int(selected["reviewed"].sum()),
        "primary_named_embedding_candidates": len(primary),
        "domain_only_rescue_candidates": len(rescue),
        "embedding_candidate_selection_reasons": selected["selection_reason"].value_counts().to_dict(),
        "domain_family_counts_normalized": normalized["domain_family"].value_counts().to_dict(),
        "domain_family_counts_embedding_candidates": selected["domain_family"].value_counts().to_dict(),
        "reviewed_status_embedding_candidates": selected["reviewed"].value_counts().rename(index={True: "reviewed", False: "unreviewed"}).to_dict(),
        "evidence_quality_embedding_candidates": selected["evidence_quality_tier"].value_counts().to_dict(),
        "protein_existence_embedding_candidates": selected["protein_existence"].value_counts().to_dict(),
        "outputs": {
            "raw": str(raw_path),
            "normalized": str(output_dir / "uniprot_tps_normalized.tsv"),
            "novel": str(output_dir / "uniprot_tps_novel_sequence_unique.tsv"),
            "clusters": str(output_dir / "uniprot_tps_novel_clusters.tsv"),
            "embedding_candidates": str(output_dir / "uniprot_tps_embedding_candidates.tsv"),
            "embedding_fasta": str(output_dir / "uniprot_tps_embedding_candidates.fasta"),
            "primary_embedding_candidates": str(output_dir / "uniprot_tps_primary_embedding_candidates.tsv"),
            "primary_embedding_fasta": str(output_dir / "uniprot_tps_primary_embedding_candidates.fasta"),
            "domain_only_rescue_candidates": str(output_dir / "uniprot_tps_domain_only_rescue_candidates.tsv"),
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
