from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from datetime import datetime, timezone

from ..http import get_json, get_text

ESEARCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
EFETCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"

TERM_PATTERNS = {
    "protein_detection": re.compile(r"\b(mass spectrom|proteomic|western blot|immunoblot|protein level)\b", re.I),
    "expression": re.compile(r"\b(recombinant|heterologous(?:ly)? express|protein expression|overexpress)\b", re.I),
    "purification": re.compile(r"\b(purif(?:y|ied|ication)|affinity chromatograph|His[- ]?tag)\b", re.I),
    "enzyme_assay": re.compile(r"\b(enzyme assay|enzymatic activ|catalytic activ|assayed|cataly[sz](?:e|ed|es|ing)|converted|conversion of|product formation|incubat(?:e|ed|ion) with|GC[- ]?MS|LC[- ]?MS|gas chromatograph|liquid chromatograph)\b", re.I),
    "kinetics": re.compile(r"\b(kcat|K[mM]\b|Michaelis|kinetic parameter|turnover number|specific activity)\b", re.I),
    "structure": re.compile(r"\b(crystal structure|X[- ]?ray|cryo[- ]?EM|NMR structure)\b", re.I),
}


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def search_pmids(terms: list[str], max_results: int = 20) -> list[str]:
    clean = [t.strip() for t in terms if t and len(t.strip()) >= 3]
    if not clean:
        return []
    # Quoted OR query: aliases are supplied only for a targeted stage-3 shortlist.
    query = " OR ".join(f'"{x}"[All Fields]' for x in clean[:12])
    obj, _ = get_json(ESEARCH, {
        "db": "pubmed", "term": query, "retmode": "json", "retmax": int(max_results), "sort": "relevance",
    })
    return list((obj.get("esearchresult") or {}).get("idlist") or [])


def fetch_articles(pmids: list[str]) -> list[dict]:
    if not pmids:
        return []
    text, _ = get_text(EFETCH, {"db": "pubmed", "id": ",".join(pmids), "retmode": "xml"})
    root = ET.fromstring(text)
    out = []
    for article in root.findall(".//PubmedArticle"):
        pmid = "".join(article.findtext(".//PMID", default=""))
        title_node = article.find(".//ArticleTitle")
        title = "" if title_node is None else "".join(title_node.itertext())
        abstract = " ".join("".join(x.itertext()) for x in article.findall(".//Abstract/AbstractText"))
        blob = f"{title} {abstract}"
        terms = [k for k, pat in TERM_PATTERNS.items() if pat.search(blob)]
        suggested = None
        if "enzyme_assay" in terms and ("kinetics" in terms or "structure" in terms): suggested = "T5"
        elif "enzyme_assay" in terms: suggested = "T4"
        elif "structure" in terms: suggested = "T3"
        elif any(x in terms for x in ("protein_detection","expression","purification")): suggested = "T2"
        out.append({
            "pmid": pmid, "title": title, "abstract": abstract,
            "evidence_terms": terms, "suggested_touch": suggested, "fetched_utc": utcnow(),
        })
    return out
