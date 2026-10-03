from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import re

import pandas as pd

from projects.active.fibre.evidence.observation_index import (
    build_uniprot_entity_annotations,
)

ROOT=Path(__file__).resolve().parents[4]
SNAPSHOT=ROOT/"results/fibre_uniprot_state_v1/uniprot_annotations.tsv"
RAW=ROOT/"data/terpene_marts/marts_reaction_pairs.tsv"
PROTEINS=ROOT/"data/terpene_marts_adaptation/protein_entities.csv"
OUT=ROOT/"results/fibre_uniprot_state_v1"


def main() -> None:
    if not SNAPSHOT.is_file():
        raise FileNotFoundError(
            f"UniProt snapshot missing: {SNAPSHOT}; acquire it before indexing"
        )

    uni=pd.read_csv(SNAPSHOT,sep="\t",dtype=str).fillna("")
    raw=pd.read_csv(RAW,sep="\t",dtype=str).fillna("")
    proteins=pd.read_csv(PROTEINS,dtype=str).fillna("")

    seq_to_pid={}
    for row in proteins.itertuples(index=False):
        seq=str(row.sequence)
        previous=seq_to_pid.get(seq)
        if previous is not None and previous != str(row.protein_id):
            raise ValueError("canonical protein sequence mapping is ambiguous")
        seq_to_pid[seq]=str(row.protein_id)

    uniprot_to_pid={}
    for row in raw.itertuples(index=False):
        accession=str(row.uniprot_id).strip()
        if not accession:
            continue
        pid=seq_to_pid.get(str(row.sequence))
        if pid is None:
            continue
        previous=uniprot_to_pid.get(accession)
        if previous is not None and previous != pid:
            raise ValueError(
                f"MARTS UniProt id maps to multiple canonical sequences: {accession}"
            )
        uniprot_to_pid[accession]=pid

    obs=build_uniprot_entity_annotations(
        uni,proteins,extra_subject_map=uniprot_to_pid
    )
    with (OUT/"protein_annotations.jsonl").open("w",encoding="utf-8") as fh:
        for row in obs:
            fh.write(
                json.dumps(row.to_dict(),ensure_ascii=False,sort_keys=True)+"\n"
            )

    predicate_counts=Counter(x.predicate for x in obs)
    evidence_counts=Counter(e for x in obs for e in x.evidence)
    mapped_subjects={
        x.source_subject_id for x in obs if x.canonical_subject_id is not None
    }
    all_subjects={x.source_subject_id for x in obs}

    cofactor_names=Counter()
    for x in obs:
        if x.predicate != "uniprot:Cofactor":
            continue
        for match in re.finditer(r"Name=([^;]+)",x.value):
            cofactor_names[match.group(1).strip()]+=1

    coverage={
        "schema":"fibre-uniprot-state-index-v1",
        "uniprot_entries":int(len(uni)),
        "annotation_observations":int(len(obs)),
        "annotated_subjects":len(all_subjects),
        "canonical_mapped_subjects":len(mapped_subjects),
        "canonical_mapping_fraction":(
            float(len(mapped_subjects)/len(all_subjects))
            if all_subjects else 0.0
        ),
        "predicate_counts":dict(sorted(predicate_counts.items())),
        "evidence_token_counts":dict(
            evidence_counts.most_common(40)
        ),
        "cofactor_name_counts":dict(
            cofactor_names.most_common(30)
        ),
        "important_semantics":{
            "mapping":"MARTS UniProt id -> exact MARTS sequence -> canonical protein id",
            "evidence":"ECO/source tokens are annotation provenance, not weights or tiers",
            "reviewed":"reviewed status is retained as an annotation, not a scalar quality score",
            "ranking_effect":"none",
        },
    }
    (OUT/"state_index_coverage.json").write_text(
        json.dumps(coverage,indent=2)+"\n"
    )
    print(json.dumps(coverage,indent=2))


if __name__=="__main__":
    main()
