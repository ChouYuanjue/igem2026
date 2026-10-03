from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

from projects.active.fibre.evidence.assay_context import (
    AssayContext, AssayObservation, NumericInterval,
)
from projects.active.fibre.evidence.catalytic_state import build_pair_catalytic_states

ROOT=Path(__file__).resolve().parents[4]
DEFAULT_OBSERVATIONS=ROOT/"results/fibre_observation_index_v1/observations.jsonl"
DEFAULT_MECHANISM=ROOT/"results/fibre_observation_index_v1/mechanism_steps.jsonl"
DEFAULT_PROTEIN=ROOT/"results/fibre_uniprot_state_v1/protein_annotations.jsonl"
DEFAULT_ASSAY=ROOT/"results/fibre_assay_context_v1/assay_observations.jsonl"
DEFAULT_OUT=ROOT/"results/fibre_catalytic_state_v1"


def _jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    rows=[]
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _interval(row: dict | None) -> NumericInterval | None:
    if not row:
        return None
    return NumericInterval(float(row["lower"]),float(row["upper"]),str(row["unit"]))


def _assay(row: dict) -> AssayObservation:
    c=row.get("context") or {}
    ctx=AssayContext(
        ph=_interval(c.get("ph")),
        temperature_c=_interval(c.get("temperature_c")),
        cofactors=tuple(c.get("cofactors") or ()),
        substrate_concentration=tuple(_interval(x) for x in c.get("substrate_concentration") or [] if x),
        enzyme_concentration=tuple(_interval(x) for x in c.get("enzyme_concentration") or [] if x),
        incubation_time_s=tuple(_interval(x) for x in c.get("incubation_time_s") or [] if x),
        raw_conditions=tuple(c.get("raw_conditions") or ()),
    )
    return AssayObservation(
        observation_id=str(row["observation_id"]),
        enzyme_id=str(row["enzyme_id"]),
        reaction_id=str(row["reaction_id"]),
        context=ctx,
        outcome=str(row["outcome"]),
        source_scope=str(row["source_scope"]),
        source_uri=row.get("source_uri"),
        source_record=str(row.get("source_record") or ""),
        evidence_texts=tuple(row.get("evidence_texts") or ()),
        target_binding_status=str(row.get("target_binding_status") or ""),
    )


def main() -> None:
    ap=argparse.ArgumentParser(description="Build sparse enzymology state sheets for canonical FIBRE positive pairs.")
    ap.add_argument("--observations",type=Path,default=DEFAULT_OBSERVATIONS)
    ap.add_argument("--mechanism-steps",type=Path,default=DEFAULT_MECHANISM)
    ap.add_argument("--protein-annotations",type=Path,default=DEFAULT_PROTEIN)
    ap.add_argument("--assay-observations",type=Path,default=DEFAULT_ASSAY)
    ap.add_argument("--output-dir",type=Path,default=DEFAULT_OUT)
    args=ap.parse_args()

    correspondence=_jsonl(args.observations.resolve())
    mechanisms=_jsonl(args.mechanism_steps.resolve())
    proteins=_jsonl(args.protein_annotations.resolve())
    assays=[_assay(x) for x in _jsonl(args.assay_observations.resolve())]

    states=build_pair_catalytic_states(correspondence,mechanisms,proteins,assays)
    out=args.output_dir.resolve(); out.mkdir(parents=True,exist_ok=True)
    with (out/"pair_states.jsonl").open("w",encoding="utf-8") as fh:
        for row in states:
            fh.write(json.dumps(row.to_dict(),ensure_ascii=False,sort_keys=True)+"\n")

    component_counts=Counter(c for row in states for c in row.observed_components)
    mechanism_pairs=sum(bool(x.mechanism_ids) for x in states)
    direct_mechanism_pairs=sum(
        any(v=="experiment" for v in x.mechanism_source_evidence)
        for x in states
    )
    cofactor_pairs=sum(bool(x.protein_cofactors) for x in states)
    catalytic_pairs=sum(bool(x.protein_catalytic_rhea_ids) for x in states)
    summary={
        "schema":"fibre-catalytic-state-v1",
        "canonical_pair_states":len(states),
        "component_pair_counts":dict(sorted(component_counts.items())),
        "pair_mechanism_count":mechanism_pairs,
        "pair_mechanism_with_experimental_step_evidence":direct_mechanism_pairs,
        "protein_cofactor_annotation_pair_count":cofactor_pairs,
        "protein_catalytic_activity_annotation_pair_count":catalytic_pairs,
        "pair_assay_context_count":sum(bool(x.pair_assay_observation_ids) for x in states),
        "semantics":{
            "pair_mechanism":"pair-associated MARTS mechanism record; original evidence labels retained",
            "protein_annotations":"UniProt entity-level state, not pair-specific assay evidence",
            "assay_context":"source-bound pair-specific experimental context",
            "missing":"unresolved, never negative",
            "ranking_effect":"none by itself; ordering authority belongs only to separately validated FIBRE coordinates/constraints",
        },
    }
    (out/"summary.json").write_text(json.dumps(summary,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
    print(json.dumps(summary,indent=2,ensure_ascii=False))


if __name__=="__main__":
    main()
