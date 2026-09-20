from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from projects.active.fibre.evidence.observation_index import (
    build_marts_observations,
    build_marts_mechanism_steps,
    canonical_positive_pairs,
    dumps_observation,
    observation_property_terms,
)

ROOT=Path(__file__).resolve().parents[4]
RAW=ROOT/'data/terpene_marts/marts_reaction_pairs.tsv'
PROTEINS=ROOT/'data/terpene_marts_adaptation/protein_entities.csv'
REACTIONS=ROOT/'data/terpene_marts_adaptation/reaction_entities.csv'
OMEGA=ROOT/'data/terpene_marts_adaptation/marts_pair_folds.csv'
MECHANISM_STEPS=ROOT/'data/terpene_marts/marts_mechanism_steps.tsv'
OUT=ROOT/'results/fibre_observation_index_v1'

EXPECTED_NOT_MATERIALISED=(
    'strenda:reaction_conditions.temperature',
    'strenda:reaction_conditions.pH_value',
    'strenda:reaction_conditions.solvent_description',
    'strenda:reaction_conditions.ionic_strength',
    'strenda:results.turnover_number',
    'strenda:results.michaelis_constant',
    'strenda:results.catalytic_efficiency',
    'strenda:results.specific_activity',
    'strenda:results.initial_reaction_rate',
    'strenda:results.conversion',
    'strenda:results.stereoselectivity',
)


def main() -> None:
    raw=pd.read_csv(RAW,sep='\t',dtype=str).fillna('')
    proteins=pd.read_csv(PROTEINS,dtype=str).fillna('')
    reactions=pd.read_csv(REACTIONS,dtype=str).fillna('')
    omega=pd.read_csv(OMEGA,dtype=str).fillna('')
    mechanism_steps=pd.read_csv(MECHANISM_STEPS,sep='\t',dtype=str).fillna('')

    obs=build_marts_observations(raw,proteins,reactions)
    steps=build_marts_mechanism_steps(mechanism_steps)
    projected=canonical_positive_pairs(obs)
    canonical=set(zip(omega.Entry.astype(str),omega.rhea_id.astype(str)))
    if projected != canonical:
        raise RuntimeError(
            f'observation projection changed canonical Omega: '
            f'projected={len(projected)} canonical={len(canonical)} '
            f'extra={len(projected-canonical)} missing={len(canonical-projected)}'
        )

    OUT.mkdir(parents=True,exist_ok=True)
    with (OUT/'observations.jsonl').open('w',encoding='utf-8') as fh:
        for row in obs:
            fh.write(dumps_observation(row)+'\n')

    pd.DataFrame(
        sorted(projected),columns=['enzyme_id','reaction_id']
    ).to_csv(OUT/'canonical_pair_projection.csv',index=False)
    with (OUT/'mechanism_steps.jsonl').open('w',encoding='utf-8') as fh:
        for row in steps:
            fh.write(json.dumps(row.to_dict(),ensure_ascii=False,sort_keys=True)+'\n')

    publication=raw.loc[raw.publication.ne('')].copy()
    queue=(
        publication.groupby('publication',as_index=False)
        .agg(
            source_row_count=('publication','size'),
            unique_enzyme_count=('enzyme_id','nunique'),
            unique_reaction_count=('reaction_signature','nunique'),
            uniprot_row_count=('uniprot_id',lambda s:int(s.ne('').sum())),
        )
        .sort_values(['source_row_count','publication'],ascending=[False,True])
    )
    queue.to_csv(OUT/'publication_queue.csv',index=False)

    terms=sorted(observation_property_terms(obs))
    coverage={
        'schema':'fibre-observation-coverage-v1',
        'source':'MARTS',
        'source_rows':int(len(raw)),
        'observation_count':int(len(obs)),
        'canonical_mapped_observations':int(sum(
            x.canonical_enzyme_id is not None and x.canonical_reaction_id is not None
            for x in obs
        )),
        'canonical_pair_projection_count':int(len(projected)),
        'canonical_pair_reference_count':int(len(canonical)),
        'canonical_projection_exact':True,
        'unmapped_reaction_observations':int(sum(x.canonical_reaction_id is None for x in obs)),
        'unmapped_enzyme_observations':int(sum(x.canonical_enzyme_id is None for x in obs)),
        'publication_present_fraction':float(raw.publication.ne('').mean()),
        'unique_publications':int(raw.loc[raw.publication.ne(''),'publication'].nunique()),
        'uniprot_present_fraction':float(raw.uniprot_id.ne('').mean()),
        'unique_uniprot_ids':int(raw.loc[raw.uniprot_id.ne(''),'uniprot_id'].nunique()),
        'mechanism_placeholder_nonempty_fraction':float(raw.mechanism_marts_id.ne('').mean()),
        'pair_rows_with_real_mechanism_steps':int(raw.mechanism_marts_id.isin(set(mechanism_steps.Mechanism_marts_id)).sum()),
        'pair_fraction_with_real_mechanism_steps':float(raw.mechanism_marts_id.isin(set(mechanism_steps.Mechanism_marts_id)).mean()),
        'mechanism_step_observation_count':int(len(steps)),
        'mechanism_step_unique_mechanisms':int(mechanism_steps.Mechanism_marts_id.nunique()),
        'mechanism_step_evidence_counts':{str(k):int(v) for k,v in mechanism_steps.Evidence.value_counts(dropna=False).to_dict().items()},
        'mechanism_step_publication_fraction':float(mechanism_steps.Publication.ne('').mean()),
        'source_species_present_fraction':float(raw.species.ne('').mean()),
        'materialised_property_terms':terms,
        'standard_terms_not_materialised_by_MARTS':list(EXPECTED_NOT_MATERIALISED),
        'important_semantics':{
            'reported_positive':'MARTS reports an enzyme-reaction association; this is not upgraded to a quantitative assay.',
            'missing_property':'unknown/not materialised, never negative evidence.',
            'source_organism':'mapped to STRENDA origin_organism, not EnzymeML Protein.organism (expression host).',
            'mechanism':'only mechanism ids with actual step records count as observed mechanism; no_mechanism is missing.',
            'mechanism_evidence':'MARTS Evidence labels are preserved verbatim as source provenance, not ranked or scalarised.',
            'ranking_effect':'none; this index is read-only provenance and future context infrastructure.',
        },
    }
    (OUT/'coverage.json').write_text(
        json.dumps(coverage,indent=2,ensure_ascii=False)+'\n',encoding='utf-8'
    )
    print(json.dumps(coverage,indent=2,ensure_ascii=False))


if __name__=='__main__':
    main()
