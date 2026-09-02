from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
from drfp import DrfpEncoder

from projects.active.terpene_screening.prepare_marts_dataset import normalize_marts_reactions
from projects.active.terpene_screening.build_sequence_clusters import resolve_mmseqs

ROOT = Path(__file__).resolve().parents[3]
PROTOCOL = ROOT / 'projects/active/terpene_screening/CATALYST_TPS_TEMPORAL_R2E_V1.json'
DEFAULT_OLD = ROOT / 'data/external/marts_db/v1.5/reactions.csv'
DEFAULT_NEW = ROOT / 'data/external/marts_db/v2.1/extracted/reactions.csv'
DEFAULT_OUT = ROOT / 'results/tps_temporal_r2e_v1'


def sha256_file(path: Path) -> str:
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1<<20),b''): h.update(chunk)
    return h.hexdigest()


def seq_key(sequence: str) -> str:
    return 'seq_' + hashlib.sha256(str(sequence).encode()).hexdigest()[:20]


def write_unique_fasta(sequences: set[str], path: Path) -> dict[str,str]:
    mapping={seq_key(s):s for s in sorted(sequences) if s}
    if len(mapping)!=len({s for s in sequences if s}): raise AssertionError('sequence hash collision')
    with path.open('w') as f:
        for key,seq in mapping.items(): f.write(f'>{key}\n{seq}\n')
    return mapping


def mmseqs_cold_sequences(train_sequences:set[str], target_sequences:set[str], out:Path, mmseqs:Path, threads:int)->tuple[set[str],pd.DataFrame]:
    train_fa=out/'v15_train_sequences.fasta'; target_fa=out/'v21_target_sequences.fasta'
    train_map=write_unique_fasta(train_sequences,train_fa); target_map=write_unique_fasta(target_sequences,target_fa)
    hits=out/'target_vs_v15_mmseqs_id50.tsv'; tmp=out/'mmseqs_tmp'
    shutil.rmtree(tmp,ignore_errors=True)
    cmd=[str(mmseqs),'easy-search',str(target_fa),str(train_fa),str(hits),str(tmp),
         '--min-seq-id','0.5','-c','0.8','--cov-mode','2','--format-output','query,target,pident,qcov,tcov,evalue,bits','--threads',str(threads),'--remove-tmp-files','1']
    subprocess.run(cmd,cwd=ROOT,check=True)
    if hits.exists() and hits.stat().st_size:
        h=pd.read_csv(hits,sep='\t',names=['query','target','pident','qcov','tcov','evalue','bits'],dtype={'query':str,'target':str})
        hit_keys=set(h['query'].astype(str))
    else:
        h=pd.DataFrame(columns=['query','target','pident','qcov','tcov','evalue','bits']); hit_keys=set()
    cold={seq for key,seq in target_map.items() if key not in hit_keys}
    audit=pd.DataFrame({'sequence_key':list(target_map),'sequence':[target_map[k] for k in target_map]})
    audit['has_v15_id50_hit']=audit.sequence_key.isin(hit_keys); audit['protein_cold']=~audit.has_v15_id50_hit
    audit.to_csv(out/'protein_cold_audit.csv',index=False)
    return cold,h


def drfp_matrix(reactions:list[str])->np.ndarray:
    if not reactions: return np.empty((0,2048),dtype=np.uint8)
    return np.asarray(DrfpEncoder.encode(reactions,n_folded_length=2048),dtype=np.uint8)>0


def max_binary_tanimoto(query:np.ndarray, train:np.ndarray, chunk:int=512)->float:
    if not len(train): return 0.0
    q=query.astype(bool,copy=False); qn=int(q.sum()); best=0.0
    for start in range(0,len(train),chunk):
        x=train[start:start+chunk].astype(bool,copy=False)
        inter=np.logical_and(x,q).sum(1); union=x.sum(1)+qn-inter
        sims=np.divide(inter,union,out=np.zeros_like(inter,dtype=float),where=union>0)
        if len(sims): best=max(best,float(sims.max()))
    return best


def reaction_cold_map(old_reactions:set[str], target_reactions:set[str], out:Path)->pd.DataFrame:
    old_ids=sorted(r for r in old_reactions if r); target_ids=sorted(r for r in target_reactions if r)
    old_x=drfp_matrix(old_ids); target_x=drfp_matrix(target_ids)
    rows=[]
    for rid,q in zip(target_ids,target_x):
        best=max_binary_tanimoto(q,old_x)
        rows.append({'reaction_signature':rid,'max_v15_drfp_tanimoto':best,'reaction_cold':best<0.5})
    frame=pd.DataFrame(rows); frame.to_csv(out/'reaction_cold_audit.csv',index=False); return frame


def materialize(old_path:Path,new_path:Path,out:Path,mmseqs:Path,threads:int)->dict[str,object]:
    out.mkdir(parents=True,exist_ok=True)
    old=normalize_marts_reactions(pd.read_csv(old_path,dtype=str).fillna(''))
    new=normalize_marts_reactions(pd.read_csv(new_path,dtype=str).fillna(''))
    old=old[(old.sequence!='')&(old.reaction_signature!='')].copy(); new=new[(new.sequence!='')&(new.reaction_signature!='')].copy()
    old_pair=set(old[['sequence','reaction_signature']].itertuples(index=False,name=None))
    new['is_future_pair']=[pair not in old_pair for pair in new[['sequence','reaction_signature']].itertuples(index=False,name=None)]
    future=new[new.is_future_pair].copy()
    cold_sequences,hits=mmseqs_cold_sequences(set(old.sequence),set(new.sequence),out,mmseqs,threads)
    future['protein_cold']=future.sequence.isin(cold_sequences)
    rxn_audit=reaction_cold_map(set(old.reaction_signature),set(future.reaction_signature),out)
    rxn_cold=dict(zip(rxn_audit.reaction_signature,rxn_audit.reaction_cold))
    future['reaction_cold']=future.reaction_signature.map(rxn_cold).fillna(False).astype(bool)
    future['strict_double_cold']=future.protein_cold & future.reaction_cold

    # Candidate identities are target-snapshot enzyme IDs; historical masking is sequence-aware.
    candidates=(new.sort_values(['enzyme_id','reaction_signature']).drop_duplicates('enzyme_id')
        [['enzyme_id','enzyme_id_type','sequence','uniprot_id','genbank_id','marts_enzyme_id','enzyme_name','species','kingdom','terpene_type','tps_class']])
    candidates.to_csv(out/'candidate_enzymes.csv',index=False)
    historical_by_rxn={r:set(g.sequence) for r,g in old.groupby('reaction_signature',sort=True)}
    target_by_rxn={r:g.copy() for r,g in future.groupby('reaction_signature',sort=True)}
    candidate_by_sequence=new.groupby('sequence').enzyme_id.agg(lambda s:sorted(set(map(str,s)))).to_dict()

    query_rows=[]; pos_rows=[]; mask_rows=[]
    for rid,g in sorted(target_by_rxn.items()):
        future_ids=sorted(set(g.enzyme_id.astype(str)))
        if not future_ids: continue
        known_sequences=historical_by_rxn.get(rid,set())
        known_ids=sorted({eid for seq in known_sequences for eid in candidate_by_sequence.get(seq,[])})
        any_pc=bool(g.protein_cold.any()); rc=bool(g.reaction_cold.iloc[0]); any_dc=bool(g.strict_double_cold.any())
        query_rows.append({'query_id':rid,'reaction_signature':rid,'n_future_positive_ids':len(future_ids),'n_masked_historical_ids':len(known_ids),'temporal_new_pair':True,'temporal_protein_cold':any_pc,'temporal_strict_double_cold':any_dc and rc,'max_v15_drfp_tanimoto':float(rxn_audit.loc[rxn_audit.reaction_signature.eq(rid),'max_v15_drfp_tanimoto'].iloc[0])})
        for eid in future_ids:
            gg=g[g.enzyme_id.astype(str).eq(eid)]
            pos_rows.append({'query_id':rid,'enzyme_id':eid,'protein_cold':bool(gg.protein_cold.any()),'reaction_cold':rc,'strict_double_cold':bool(gg.strict_double_cold.any())})
        for eid in known_ids: mask_rows.append({'query_id':rid,'enzyme_id':eid})
    queries=pd.DataFrame(query_rows); positives=pd.DataFrame(pos_rows); masks=pd.DataFrame(mask_rows)
    queries.to_csv(out/'queries.csv',index=False); positives.to_csv(out/'future_positives.csv',index=False); masks.to_csv(out/'historical_masks.csv',index=False)
    future.to_csv(out/'future_pair_rows.csv',index=False)

    support={
      'temporal_new_pair_queries':int(queries.temporal_new_pair.sum()),
      'temporal_protein_cold_queries':int(queries.temporal_protein_cold.sum()),
      'temporal_strict_double_cold_queries':int(queries.temporal_strict_double_cold.sum()),
      'future_pair_rows':int(len(future)), 'future_unique_sequence_reaction_pairs':int(future[['sequence','reaction_signature']].drop_duplicates().shape[0]),
      'candidate_enzyme_ids':int(candidates.enzyme_id.nunique()), 'candidate_unique_sequences':int(candidates.sequence.nunique()),
      'protein_cold_target_sequences':int(len(cold_sequences)), 'mmseqs_hits_at_id50':int(len(hits)),
      'target_reaction_queries':int(queries.query_id.nunique())
    }
    p=json.loads(PROTOCOL.read_text()); mins=p['minimum_support']
    checks={k:int(support[k])>=int(v) for k,v in mins.items()}
    manifest={'status':'support_met' if all(checks.values()) else 'underpowered','protocol':str(PROTOCOL.relative_to(ROOT)),'protocol_sha256':sha256_file(PROTOCOL),'sources':{'v1.5_reactions':str(old_path.relative_to(ROOT)),'v1.5_sha256':sha256_file(old_path),'v2.1_reactions':str(new_path.relative_to(ROOT)),'v2.1_sha256':sha256_file(new_path)},'mmseqs':str(mmseqs),'protein_rule':'no MMseqs2 hit to any v1.5 sequence at min-seq-id=0.5, coverage=0.8, cov-mode=2','reaction_rule':'max binary 2048-d DRFP Tanimoto to any v1.5 reaction signature <0.5','support':support,'minimum_support_checks':checks,'model_scores_read':False,'selection_allowed':False}
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    return manifest


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--old',type=Path,default=DEFAULT_OLD); ap.add_argument('--new',type=Path,default=DEFAULT_NEW); ap.add_argument('--output-dir',type=Path,default=DEFAULT_OUT); ap.add_argument('--mmseqs',default=None); ap.add_argument('--threads',type=int,default=8); a=ap.parse_args()
    result=materialize(a.old.resolve(),a.new.resolve(),a.output_dir.resolve(),resolve_mmseqs(a.mmseqs),a.threads); print(json.dumps(result,indent=2))
if __name__=='__main__': main()
