from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd
from Bio import Align, SeqIO
from Bio.Align import substitution_matrices

ROOT = Path(__file__).resolve().parents[4]
CONTRACTS = ROOT / 'data/terpene_uniprot_expansion/reaction_architecture_contracts/reaction_architecture_contracts.csv'
REFERENCE = ROOT / 'data/terpene_uniprot_expansion/reaction_architecture_contracts/reference_architecture_evidence.csv'
REFERENCE_FASTA = ROOT / 'data/terpene_uniprot_expansion/reaction_architecture_contracts/normalized_tps_sequences.fasta'
EXPANSION = ROOT / 'data/terpene_uniprot_expansion/uniprot_tps_primary_embedding_candidates.tsv'

# Evidence-only motifs.  They are deliberately broader than the historical
# descriptor extractor: class-I TPSs can carry DDXX(D/E) or DDXXX(D/E)
# aspartate-rich variants (e.g. the experimentally established geosmin DDHFLE
# motif).  Matches are reported as sequence observables, never as rank gates.
TYPE_I_ASPARTATE = re.compile(r'DD.{2,3}[DE]')
NSE = re.compile(r'[ND]D..[ST]...E')
DTE = re.compile(r'DTE')
DXDD = re.compile(r'D.DD')
QW = re.compile(r'QW')
RY = re.compile(r'RY')


def _matches(sequence: str, pattern: re.Pattern[str], flank: int = 8) -> list[dict[str, object]]:
    out=[]
    for m in pattern.finditer(sequence):
        lo=max(0,m.start()-flank); hi=min(len(sequence),m.end()+flank)
        out.append({'start_1based':m.start()+1,'end_1based':m.end(),'match':m.group(0),'context':sequence[lo:hi]})
    return out


def family_motif_observables(sequence: str, domain_family: str) -> dict[str, object]:
    family=str(domain_family or '')
    if not family:
        return {'scope':'not_applicable_without_tps_family_annotation'}
    common={
        'typeI_aspartate_rich_variants':_matches(sequence,TYPE_I_ASPARTATE),
        'nse_like':_matches(sequence,NSE),
        'dte':_matches(sequence,DTE),
        'dxdd':_matches(sequence,DXDD),
        'qw':_matches(sequence,QW),
        'ry':_matches(sequence,RY),
    }
    if 'bacterial_classI' in family or 'plant_like_classI' in family:
        scope='class_I_terpene_synthase_observables'
    elif 'triterpene_cyclase' in family:
        scope='triterpene_cyclase_observables'
    else:
        scope='annotated_terpene_synthase_family_observables'
    return {'scope':scope,'domain_family':family,'motifs':common,
            'interpretation':'sequence observables only; motif presence/absence does not change ranking'}


def _reference_sequences() -> dict[str,str]:
    return {r.id:str(r.seq) for r in SeqIO.parse(REFERENCE_FASTA,'fasta')}


def _sequence_homology(candidate: str, reference: str) -> dict[str, float | int]:
    matrix=substitution_matrices.load('BLOSUM62')
    g=Align.PairwiseAligner(); g.mode='global'; g.substitution_matrix=matrix; g.open_gap_score=-10.0; g.extend_gap_score=-0.5
    ga=g.align(candidate,reference)[0]; gc=ga.counts(); gl=int(ga.length)
    l=Align.PairwiseAligner(); l.mode='local'; l.substitution_matrix=matrix; l.open_gap_score=-10.0; l.extend_gap_score=-0.5
    la=l.align(candidate,reference)[0]; lc=la.counts(); ll=int(la.length)
    ca=int(sum(int(e-b) for b,e in la.aligned[0])); ra=int(sum(int(e-b) for b,e in la.aligned[1]))
    return {
        'global_identity':float(gc.identities/max(gl,1)),
        'global_alignment_length':gl,
        'local_identity':float(lc.identities/max(ll,1)),
        'local_alignment_length':ll,
        'candidate_local_coverage':float(ca/max(len(candidate),1)),
        'reference_local_coverage':float(ra/max(len(reference),1)),
    }


def build_sheet(reaction_id: str, candidate_id: str, max_references: int = 12) -> dict[str, object]:
    contracts=pd.read_csv(CONTRACTS,dtype=str).fillna('')
    crows=contracts[contracts.reaction_id.eq(reaction_id)]
    if len(crows)!=1:
        raise ValueError(f'reaction contract missing/nonunique: {reaction_id}')
    c=crows.iloc[0]

    candidates=pd.read_csv(EXPANSION,sep='\t',dtype=str).fillna('')
    prows=candidates[candidates.accession.eq(candidate_id)]
    if len(prows)!=1:
        raise ValueError(f'candidate missing/nonunique in TPS expansion: {candidate_id}')
    p=prows.iloc[0]; sequence=str(p.sequence)

    refs=pd.read_csv(REFERENCE,dtype=str).fillna('')
    refs=refs[refs.reaction_id.eq(reaction_id)].copy()
    seqs=_reference_sequences()
    ref_records=[]
    for r in refs.head(max_references).itertuples(index=False):
        rid=str(r.enzyme_id); rseq=seqs.get(rid,'')
        ref_records.append({
            'enzyme_id':rid,'enzyme_name':str(r.enzyme_name),
            'mapping_source':str(r.mapping_source),
            'pfam_combination':str(r.pfam_combination),
            'reference_architecture':str(r.reference_architecture),
            'mapping_identity':None if str(r.pident)=='' else float(r.pident),
            'mapping_query_coverage':None if str(r.qcov)=='' else float(r.qcov),
            'mapping_target_coverage':None if str(r.tcov)=='' else float(r.tcov),
            'candidate_to_reference_sequence_homology':(
                _sequence_homology(sequence,rseq) if rseq else None
            ),
            'family_motif_observables':(
                family_motif_observables(rseq, str(r.reference_architecture)) if rseq else {'scope':'sequence_unavailable'}
            ),
        })

    reference_architectures=sorted(set(x for x in refs.reference_architecture.astype(str) if x))
    candidate_family=str(p.domain_family)
    architecture_observed=bool(candidate_family and candidate_family in reference_architectures)
    return {
        'schema':'terpene-reaction-conditioned-mechanism-sheet-v1',
        'role':'ranking-neutral application evidence; no compatibility gate and no score modification',
        'reaction':{
            'reaction_id':reaction_id,
            'substrate_name':str(c.substrate_name),'product_name':str(c.product_name),
            'terpene_type':str(c.terpene_type),'tps_class':str(c.tps_class),
            'reaction_signature':str(c.reaction_signature),
            'known_positive_count':int(c.reference_positive_count),
            'mapped_reference_count':int(c.mapped_reference_count),
            'reference_architectures':reference_architectures,
            'reference_evidence_status':str(c.contract_status),
        },
        'candidate':{
            'accession':candidate_id,'entry_name':str(p.entry_name),'protein_name':str(p.protein_name),
            'organism_name':str(p.organism_name),'reviewed':str(p.reviewed).lower()=='true',
            'protein_existence':str(p.protein_existence),'evidence_quality_tier':str(p.evidence_quality_tier),
            'sequence_length':int(p.length),'pfam_combination':str(p.pfam_combination),
            'domain_family':candidate_family,
            'architecture_observed_among_known_references':architecture_observed,
            'family_motif_observables':family_motif_observables(sequence,candidate_family),
        },
        'known_positive_references':ref_records,
        'evidence_interpretation':{
            'architecture_statement':'reports whether the candidate family architecture has been observed among known positives for this reaction; it is not an admissibility rule',
            'motif_statement':'reports family-aware catalytic motif-like sequence contexts; variants and missing motifs require biological review and do not change rank',
            'missingness_statement':'missing reference mapping, sequence, structure, motif, or Pfam is missing evidence rather than negative evidence',
        },
        'ranking_modified':False,
    }


def main() -> None:
    ap=argparse.ArgumentParser(); ap.add_argument('--reaction',required=True); ap.add_argument('--protein',required=True); ap.add_argument('--max-references',type=int,default=12); ap.add_argument('--output',type=Path,default=None)
    a=ap.parse_args(); result=build_sheet(a.reaction,a.protein,a.max_references); text=json.dumps(result,indent=2,ensure_ascii=False)+'\n'
    if a.output: a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text(text)
    print(text,end='')

if __name__=='__main__': main()
