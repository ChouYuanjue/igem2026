from __future__ import annotations

import json
from pathlib import Path
import re

import pandas as pd

ROOT=Path(__file__).resolve().parents[4]
CACHE=ROOT/"data/terpene_marts_adaptation"
RHEA=ROOT/"results/fibre_rhea_mapping_v1/marts_exact_rhea_mapping.csv"
ANNOT=ROOT/"results/fibre_uniprot_state_v1/protein_annotations.jsonl"
OUT=ROOT/"results/fibre_cross_source_catalytic_evidence_v1"
RHEA_RE=re.compile(r"RHEA:\d+")


def _jsonl(path: Path):
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                yield json.loads(line)


def main() -> None:
    reactions=pd.read_csv(CACHE/"reaction_entities.csv",dtype=str).fillna("")
    pairs=pd.read_csv(CACHE/"marts_pair_folds.csv",dtype=str).fillna("").drop_duplicates(["rhea_id","Entry"])
    exact=pd.read_csv(RHEA,dtype=str).fillna("")

    sig_to_rhea={}
    for row in exact.itertuples(index=False):
        values=set()
        for field in (str(row.rhea_directed_ids),str(row.rhea_master_ids)):
            values.update(
                f"RHEA:{x.split(':')[-1]}" if not x.startswith("RHEA:") else x
                for x in field.split("|") if x
            )
        sig_to_rhea.setdefault(str(row.reaction_signature),set()).update(values)
    rid_to_sig=dict(zip(reactions.reaction_id.astype(str),reactions.reaction_signature.astype(str)))

    protein_catalytic={}
    for row in _jsonl(ANNOT):
        if str(row.get("predicate") or "")!="uniprot:CatalyticActivity":
            continue
        pid=str(row.get("canonical_subject_id") or "")
        if not pid:
            continue
        bucket=protein_catalytic.setdefault(pid,{"rhea":set(),"experimental":set(),"inferred":set(),"other":set()})
        ids=set(RHEA_RE.findall(str(row.get("value") or "")))
        bucket["rhea"].update(ids)
        evidence=set(str(x) for x in row.get("evidence") or [])
        if any(x.startswith("ECO:0000269|") for x in evidence):
            bucket["experimental"].update(ids)
        elif any(x.startswith(("ECO:0000250|","ECO:0000255|","ECO:0000256|")) for x in evidence):
            bucket["inferred"].update(ids)
        else:
            bucket["other"].update(ids)

    rows=[]
    for pair in pairs.itertuples(index=False):
        pid=str(pair.Entry); rid=str(pair.rhea_id)
        signature=rid_to_sig.get(rid,"")
        reaction_rhea=sig_to_rhea.get(signature,set())
        protein=protein_catalytic.get(pid,{"rhea":set(),"experimental":set(),"inferred":set(),"other":set()})
        overlap=reaction_rhea & protein["rhea"]
        experimental=overlap & protein["experimental"]
        inferred=overlap & protein["inferred"]
        other=overlap & protein["other"]
        if experimental:
            status="independent_experimental_uniprot_rhea_match"
        elif inferred:
            status="uniprot_inferred_rhea_match"
        elif other:
            status="uniprot_other_rhea_match"
        elif not reaction_rhea:
            status="reaction_not_strictly_rhea_mapped"
        elif not protein["rhea"]:
            status="protein_without_uniprot_catalytic_rhea"
        else:
            status="no_exact_cross_source_rhea_overlap"
        rows.append({
            "enzyme_id":pid,
            "reaction_id":rid,
            "strict_reaction_rhea_ids":"|".join(sorted(reaction_rhea)),
            "uniprot_catalytic_rhea_ids":"|".join(sorted(protein["rhea"])),
            "overlap_rhea_ids":"|".join(sorted(overlap)),
            "status":status,
            "independent_experimental_support":bool(experimental),
        })

    frame=pd.DataFrame(rows)
    OUT.mkdir(parents=True,exist_ok=True)
    frame.to_csv(OUT/"pair_evidence.csv",index=False)
    counts=frame.status.value_counts().to_dict()
    mapped=frame[frame.strict_reaction_rhea_ids.ne("")]
    summary={
        "schema":"fibre-cross-source-catalytic-evidence-v1",
        "canonical_positive_pairs":int(len(frame)),
        "strict_rhea_mapped_positive_pairs":int(len(mapped)),
        "strict_rhea_mapped_pair_fraction":float(len(mapped)/len(frame)),
        "independent_experimental_uniprot_matches":int(frame.independent_experimental_support.sum()),
        "independent_experimental_fraction_all_pairs":float(frame.independent_experimental_support.mean()),
        "independent_experimental_fraction_strict_rhea_mapped":(
            float(mapped.independent_experimental_support.mean()) if len(mapped) else 0.0
        ),
        "status_counts":{str(k):int(v) for k,v in counts.items()},
        "policy":(
            "cross-source corroboration only: canonical MARTS reaction signature must map "
            "by strict directed molecular identity to Rhea and the canonical protein's "
            "UniProt catalytic-activity annotation must contain the same Rhea id; "
            "ECO:0000269 is reported separately as experimental support. No missing "
            "cross-source match is treated as contradictory evidence or a ranking penalty."
        ),
        "ranking_effect":"none",
    }
    (OUT/"summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    print(json.dumps(summary,indent=2))


if __name__=="__main__":
    main()
