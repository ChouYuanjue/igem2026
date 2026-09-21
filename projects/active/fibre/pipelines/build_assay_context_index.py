from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from projects.active.fibre.evidence.assay_context import (
    build_source_bound_assay_observations,
    marts_target_id,
)

ROOT=Path(__file__).resolve().parents[4]
DEFAULT_OBSERVATIONS=ROOT/"results/fibre_observation_index_v1/observations.jsonl"
DEFAULT_EXTRACTIONS=ROOT/"results/fibre_publication_context_deepseek_v1/extractions.jsonl"
DEFAULT_OUTPUT=ROOT/"results/fibre_assay_context_v1"


def load_jsonl(path: Path) -> list[dict]:
    rows=[]
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def canonical_target_map(observation_rows: list[dict]) -> dict[str,tuple[str,str]]:
    out={}
    for row in observation_rows:
        enzyme_source=str(row.get("enzyme_source_id") or "")
        reaction_signature=str(row.get("reaction_source_signature") or "")
        protein=str(row.get("canonical_enzyme_id") or "")
        reaction=str(row.get("canonical_reaction_id") or "")
        if not (enzyme_source and reaction_signature and protein and reaction):
            continue
        key=marts_target_id(enzyme_source,reaction_signature)
        value=(protein,reaction)
        previous=out.get(key)
        if previous is not None and previous!=value:
            raise RuntimeError(f"ambiguous target mapping for {key}: {previous} vs {value}")
        out[key]=value
    return out


def main() -> None:
    ap=argparse.ArgumentParser(
        description="Materialize conservative pair-specific assay contexts from source-bound publication extraction."
    )
    ap.add_argument("--observations",type=Path,default=DEFAULT_OBSERVATIONS)
    ap.add_argument("--extractions",type=Path,default=DEFAULT_EXTRACTIONS)
    ap.add_argument("--output-dir",type=Path,default=DEFAULT_OUTPUT)
    args=ap.parse_args()

    observations=load_jsonl(args.observations.resolve())
    extractions=load_jsonl(args.extractions.resolve())
    target_map=canonical_target_map(observations)
    assays=build_source_bound_assay_observations(extractions,target_map)

    out=args.output_dir.resolve()
    out.mkdir(parents=True,exist_ok=True)
    with (out/"assay_observations.jsonl").open("w",encoding="utf-8") as fh:
        for row in sorted(assays,key=lambda x:x.observation_id):
            fh.write(json.dumps(row.to_dict(),ensure_ascii=False,sort_keys=True)+"\n")

    outcome=Counter(x.outcome for x in assays)
    dimensions=Counter(
        dim for x in assays for dim in x.context.observed_dimensions
    )
    pairs={(x.enzyme_id,x.reaction_id) for x in assays}
    summary={
        "schema":"fibre-assay-context-v1",
        "source_observation_index":str(args.observations.resolve().relative_to(ROOT)),
        "source_publication_extractions":str(args.extractions.resolve().relative_to(ROOT)),
        "canonical_target_map_size":len(target_map),
        "materialized_assay_observations":len(assays),
        "materialized_canonical_pairs":len(pairs),
        "outcome_counts":dict(sorted(outcome.items())),
        "observed_dimension_counts":dict(sorted(dimensions.items())),
        "promotion_gate":{
            "paragraph_role":"catalytic_assay",
            "source_span_verified":True,
            "scope_resolved":True,
            "single_explicit_target":True,
            "canonical_pair_mapping_required":True,
        },
        "ranking_effect":"none",
        "model_role":(
            "first-class pair-specific assay context and future context-constraint evidence; "
            "sparse coverage never receives scalar fusion weight"
        ),
        "negative_policy":(
            "only an explicitly source-bound inactive/below-detection assay can become "
            "a context-specific contradiction; missing or mismatched conditions remain unresolved"
        ),
    }
    (out/"summary.json").write_text(
        json.dumps(summary,indent=2,ensure_ascii=False)+"\n",encoding="utf-8"
    )
    print(json.dumps(summary,indent=2,ensure_ascii=False))


if __name__=="__main__":
    main()
