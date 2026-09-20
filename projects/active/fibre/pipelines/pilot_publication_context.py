from __future__ import annotations

import json
from pathlib import Path
import re
import time
import xml.etree.ElementTree as ET

import pandas as pd
import requests

ROOT=Path(__file__).resolve().parents[4]
MARTS=ROOT/"data/terpene_marts/marts_reaction_pairs.tsv"
OUT=ROOT/"results/fibre_publication_context_pilot_v1"

TARGETS=(
    ("PMID","21818683"),
    ("PMID","23679205"),
    ("PMID","30254228"),
    ("PMID","30105900"),
    ("PMCID","PMC6945850"),
)

KEYWORDS={
    "pH":re.compile(r"\bpH\b",re.I),
    "temperature":re.compile(r"\b(?:temperature|°C|degrees? C)\b",re.I),
    "metal_or_cofactor":re.compile(
        r"\b(?:MgCl2|Mg\(2\+\)|Mg2\+|MnCl2|Mn\(2\+\)|Mn2\+|"
        r"cofactor|metal ion|Zn2\+|Fe2\+)\b",re.I
    ),
    "kinetic":re.compile(
        r"\b(?:kcat|K\s*m|Michaelis|specific activity|initial rate|turnover|kinetic)\b",
        re.I,
    ),
    "assay_context":re.compile(
        r"\b(?:enzyme assay|activity assay|reaction mixture|incubat|buffer|"
        r"substrate concentration|enzyme concentration)\b",
        re.I,
    ),
}


def _search(kind: str, identifier: str) -> dict | None:
    query=(
        f"EXT_ID:{identifier} AND SRC:MED"
        if kind=="PMID" else f"PMCID:{identifier}"
    )
    r=requests.get(
        "https://www.ebi.ac.uk/europepmc/webservices/rest/search",
        params={"query":query,"format":"json"},
        timeout=20,
    )
    r.raise_for_status()
    rows=r.json().get("resultList",{}).get("result",[])
    return rows[0] if rows else None


def _paragraphs(xml_text: str):
    root=ET.fromstring(xml_text)
    for source_index,p in enumerate(root.findall(".//p")):
        text=" ".join("".join(p.itertext()).split())
        if len(text)>=25:
            yield source_index,text


def main() -> None:
    OUT.mkdir(parents=True,exist_ok=True)
    marts=pd.read_csv(MARTS,sep="\t",dtype=str).fillna("")
    articles=[]
    candidates=[]

    for kind,identifier in TARGETS:
        meta=_search(kind,identifier)
        if meta is None:
            articles.append({"input":identifier,"status":"metadata_not_found"})
            continue
        pmid=str(meta.get("pmid") or "")
        pmcid=str(meta.get("pmcid") or "")
        if not pmcid:
            articles.append({
                "input":identifier,"pmid":pmid,
                "status":"no_open_fulltext_pmcid",
            })
            continue

        url=f"https://www.ebi.ac.uk/europepmc/webservices/rest/{pmcid}/fullTextXML"
        r=requests.get(url,timeout=30)
        if r.status_code!=200 or "<article" not in r.text[:1500]:
            articles.append({
                "input":identifier,"pmid":pmid,"pmcid":pmcid,
                "status":f"fulltext_http_{r.status_code}",
            })
            continue

        (OUT/f"{pmcid}.xml").write_text(r.text,encoding="utf-8")
        n=0
        for source_index,text in _paragraphs(r.text):
            hits=[name for name,rx in KEYWORDS.items() if rx.search(text)]
            if not hits:
                continue
            n+=1
            candidates.append({
                "pmcid":pmcid,
                "pmid":pmid,
                "source_paragraph_index":source_index,
                "attribute_families":"|".join(hits),
                "text":text,
                "source_url":url,
            })
        articles.append({
            "input":identifier,
            "pmid":pmid,
            "pmcid":pmcid,
            "status":"fulltext_ok",
            "candidate_paragraphs":n,
        })
        time.sleep(0.1)

    article_frame=pd.DataFrame(articles)
    candidate_frame=pd.DataFrame(candidates)
    article_frame.to_csv(OUT/"articles.csv",index=False)
    candidate_frame.to_csv(OUT/"candidate_paragraphs.csv",index=False)

    def article_ids(publication: str) -> str:
        hits=[]
        for a in articles:
            pmid=str(a.get("pmid") or "")
            pmcid=str(a.get("pmcid") or "")
            if pmid and pmid in publication:
                hits.append(pmcid or pmid)
            if pmcid and pmcid in publication:
                hits.append(pmcid)
        return "|".join(sorted(set(hits)))

    linked=marts.copy()
    linked["pilot_article_ids"]=[
        article_ids(str(x)) for x in linked.publication
    ]
    linked=linked[linked.pilot_article_ids.ne("")]
    linked[
        ["enzyme_id","reaction_signature","publication","pilot_article_ids"]
    ].to_csv(OUT/"linked_marts_rows.csv",index=False)

    summary={
        "schema":"fibre-publication-context-pilot-v1",
        "target_articles":len(TARGETS),
        "fulltext_articles":sum(
            a.get("status")=="fulltext_ok" for a in articles
        ),
        "candidate_paragraphs":len(candidates),
        "linked_marts_rows":int(len(linked)),
        "attribute_family_paragraph_counts":{
            name:sum(
                name in row["attribute_families"].split("|")
                for row in candidates
            )
            for name in KEYWORDS
        },
        "policy":(
            "Candidate paragraphs are source-linked extraction candidates only. "
            "No assay value is accepted without endpoint, unit, condition scope "
            "and enzyme/reaction/experiment linkage validation."
        ),
    }
    (OUT/"summary.json").write_text(
        json.dumps(summary,indent=2)+"\n",encoding="utf-8"
    )
    print(json.dumps(summary,indent=2))
    print(article_frame.to_string(index=False))


if __name__=="__main__":
    main()
