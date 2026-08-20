from __future__ import annotations

import csv
import hashlib
import io
import re
from datetime import datetime, timezone

from ..http import get_json, get_text

UNIPARC = "https://rest.uniprot.org/uniparc/search"
UNIPROTKB = "https://rest.uniprot.org/uniprotkb/search"
ECO_EXP = "ECO:0000269"


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def normalize_accession(x: str) -> str:
    return re.sub(r"\.\d+$", "", str(x).strip())


def _pe_level(text: str) -> int | None:
    t = (text or "").lower()
    if "protein level" in t: return 1
    if "transcript level" in t: return 2
    if "homology" in t: return 3
    if "predicted" in t: return 4
    if "uncertain" in t: return 5
    return None


def fetch_uniparc_exact(md5s: list[str]) -> tuple[list[dict], dict]:
    if not md5s:
        return [], {}
    q = " OR ".join(f"checksum:{x}" for x in md5s)
    obj, headers = get_json(UNIPARC, {"query": f"({q})", "format": "json", "size": 500})
    wanted = set(x.lower() for x in md5s)
    out = []
    for r in obj.get("results", []):
        seq = ((r.get("sequence") or {}).get("value") or "").strip().upper().rstrip("*")
        returned_md5 = ((r.get("sequence") or {}).get("md5") or "").lower()
        if not returned_md5 and seq:
            returned_md5 = hashlib.md5(seq.encode()).hexdigest()
        if returned_md5 not in wanted:
            raise RuntimeError(f"UniParc returned sequence outside checksum batch: {returned_md5}")
        out.append({
            "sequence_md5": returned_md5,
            "found": 1,
            "uniparc_id": r.get("uniParcId", ""),
            "accessions": sorted(set(normalize_accession(a) for a in (r.get("uniProtKBAccessions") or []) if a)),
            "first_seen": r.get("oldestCrossRefCreated", ""),
            "last_seen": r.get("mostRecentCrossRefUpdated", ""),
            "release": headers.get("X-UniProt-Release", ""),
            "release_date": headers.get("X-UniProt-Release-Date", ""),
            "fetched_utc": utcnow(),
        })
    return out, headers


def _split_pdb(text: str) -> list[str]:
    return sorted(set(x for x in re.split(r"[;,\s]+", text or "") if x))


def fetch_uniprot_summary(accessions: list[str], rich: bool = False) -> tuple[list[dict], dict]:
    if not accessions:
        return [], {}
    q = " OR ".join(f"accession:{normalize_accession(a)}" for a in accessions)
    fields = ["accession", "reviewed", "protein_existence", "xref_pdb"]
    if rich:
        fields += [
            "cc_catalytic_activity", "cc_function", "cc_activity_regulation",
            "kinetics", "cc_mass_spectrometry", "lit_pubmed_id",
        ]
    text, headers = get_text(UNIPROTKB, {
        "query": f"({q})", "format": "tsv", "fields": ",".join(fields), "size": 500,
    })
    rows = []
    for z in csv.DictReader(io.StringIO(text), delimiter="\t"):
        acc = normalize_accession(z.get("Entry", ""))
        if not acc:
            continue
        pe = str(z.get("Protein existence", "") or "")
        catalytic = str(z.get("Catalytic activity", "") or "")
        function = str(z.get("Function [CC]", "") or "")
        regulation = str(z.get("Activity regulation", "") or "")
        kinetics = str(z.get("Kinetics", "") or "")
        ms = str(z.get("Mass spectrometry", "") or "")
        pmids = sorted(set(re.findall(r"\b\d{5,9}\b", str(z.get("PubMed ID", "") or ""))))
        rows.append({
            "accession": acc,
            "reviewed": 1 if str(z.get("Reviewed", "")).strip().lower() == "reviewed" else 0,
            "pe_level": _pe_level(pe),
            "pe_text": pe,
            "pdb_ids": _split_pdb(str(z.get("PDB", "") or "")),
            "functional_experimental": 1 if rich and ECO_EXP in (catalytic + " " + function + " " + regulation) else 0,
            "catalytic_experimental": 1 if rich and ECO_EXP in catalytic else 0,
            "kinetics_present": 1 if rich and bool(kinetics.strip()) else 0,
            "mass_spec_present": 1 if rich and bool(ms.strip()) else 0,
            "pubmed_ids": pmids,
            "evidence_depth": 2 if rich else 1,
            "release": headers.get("X-UniProt-Release", ""),
            "release_date": headers.get("X-UniProt-Release-Date", ""),
            "fetched_utc": utcnow(),
        })
    return rows, headers
