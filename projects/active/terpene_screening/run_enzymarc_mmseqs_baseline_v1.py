from __future__ import annotations
import argparse, hashlib, json, subprocess
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT=Path(__file__).resolve().parents[3]
SUPPORT=ROOT/'results/enzymarc_open_world_v1/support'
GATE=ROOT/'results/enzymarc_open_world_v1/sequence_form_gate'
OUT=ROOT/'results/enzymarc_open_world_v1/mmseqs_same_task_baseline'
CLEAN=ROOT/'data/external/enzymecage_current/catalyst_features/clean2023/training_pairs.csv'
SEQ=ROOT/'data/catalyst_candidate_universes/general_merged/protein_sequences.tsv'
MM=ROOT/'data/assets/mmseqs2/mmseqs/bin/mmseqs'
PROTO=ROOT/'projects/active/terpene_screening/CATALYST_OPEN_WORLD_ENZYMARC_MMSEQS_BASELINE_V1.json'

def sha(p:Path): return hashlib.sha256(p.read_bytes()).hexdigest()
def write_fasta(path, rows):
    with open(path,'w') as f:
        for ident,seq in rows: f.write(f'>{ident}\n{seq}\n')
def qid_parent(p): return f'P::{p}'
def qid_decoy(p,c): return f'D::{p}::{c}'
def metrics(x):
    ps=x.parent_score.to_numpy(float); ds=x.decoy_score.to_numpy(float); d=ps-ds
    labels=np.r_[np.ones(len(x),int),np.zeros(len(x),int)]; scores=np.r_[ps,ds]
    return {'instances':int(len(x)),'paired_parent_win_rate':float(np.mean(ps>ds)),'paired_score_delta_mean':float(d.mean()),'paired_score_delta_median':float(np.median(d)),'parent_vs_decoy_AUROC':float(roc_auc_score(labels,scores)),'decoy_score_retention_ratio_median':float(np.median((ds+1)/np.maximum(ps+1,1e-8)))}

def prepare(out:Path):
    if (out/'summary.json').exists(): raise RuntimeError('baseline already finalized')
    out.mkdir(parents=True,exist_ok=True)
    elig=set(pd.read_csv(GATE/'eligible_parents.csv',dtype=str).parent_accession.astype(str))
    p=pd.read_csv(SUPPORT/'parents.csv',dtype=str).fillna(''); p=p[p.parent_accession.isin(elig)].sort_values('parent_accession')
    d=pd.read_csv(SUPPORT/'decoys.csv',dtype=str).fillna(''); d=d[d.parent_accession.isin(elig)].sort_values(['parent_accession','category'])
    r=pd.read_csv(SUPPORT/'parent_reaction_relations.csv',dtype=str).fillna(''); r=r[r.parent_accession.isin(elig)].drop_duplicates()
    clean=pd.read_csv(CLEAN,dtype=str).fillna('').drop_duplicates(); clean=clean[clean.reaction_id.isin(set(r.reaction_id))]
    seq=pd.read_csv(SEQ,sep='\t',dtype=str).fillna(''); sm=dict(zip(seq.protein_id.astype(str),seq.sequence.astype(str)))
    clean=clean[clean.protein_id.isin(sm)].copy()
    targets=sorted(clean.protein_id.unique())
    write_fasta(out/'queries.fasta',[(qid_parent(x.parent_accession),x.parent_sequence) for x in p.itertuples(index=False)]+[(qid_decoy(x.parent_accession,x.category),x.decoy_sequence) for x in d.itertuples(index=False)])
    write_fasta(out/'targets.fasta',[(x,sm[x]) for x in targets])
    pd.concat([
        p[['parent_accession']].assign(category='parent',query_id=lambda z:z.parent_accession.map(qid_parent)),
        d[['parent_accession','category']].assign(query_id=lambda z:[qid_decoy(p,c) for p,c in zip(z.parent_accession,z.category)])
    ],ignore_index=True).to_csv(out/'query_entries.csv',index=False)
    clean[['protein_id','reaction_id']].drop_duplicates().to_csv(out/'target_reactions.csv',index=False)
    m={'status':'prepared_scores_unread','eligible_parents':len(p),'eligible_decoys':len(d),'query_sequences':len(p)+len(d),'target_proteins':len(targets),'target_pairs':len(clean),'target_reactions':r.reaction_id.nunique(),'protocol_sha256':sha(PROTO),'gate_sha256':sha(GATE/'manifest.json'),'model_scores_read':False,'baseline_scores_read':False}
    (out/'prepare_manifest.json').write_text(json.dumps(m,indent=2)+'\n'); print(json.dumps(m,indent=2))

def search(out:Path):
    if not (out/'prepare_manifest.json').exists(): raise RuntimeError('prepare first')
    if (out/'all_search.tsv').exists(): raise RuntimeError('search output already exists')
    cmd=[str(MM),'easy-search',str(out/'queries.fasta'),str(out/'targets.fasta'),str(out/'all_search.tsv'),str(out/'tmp'),'-s','7.5','--max-seqs','500','--threads','16','--format-output','query,target,fident,qcov,tcov,alnlen,evalue,bits']
    print(' '.join(cmd),flush=True); subprocess.run(cmd,check=True)

def finalize(out:Path):
    if (out/'summary.json').exists(): raise RuntimeError('already finalized')
    cols=['query_id','target','fident','qcov','tcov','alnlen','evalue','bits']
    h=pd.read_csv(out/'all_search.tsv',sep='\t',names=cols,dtype={'query_id':str,'target':str})
    for c in cols[2:]: h[c]=pd.to_numeric(h[c],errors='coerce')
    if h.fident.max()>1: h['fident']/=100.0
    qe=pd.read_csv(out/'query_entries.csv',dtype=str).fillna(''); tr=pd.read_csv(out/'target_reactions.csv',dtype=str).drop_duplicates(); rel=pd.read_csv(SUPPORT/'parent_reaction_relations.csv',dtype=str).drop_duplicates(); elig=set(pd.read_csv(GATE/'eligible_parents.csv',dtype=str).parent_accession.astype(str)); rel=rel[rel.parent_accession.isin(elig)]
    # Coverage is checked before aggregate metrics. Every derived query must report its own parent target.
    own=qe[['query_id','parent_accession']].merge(h[['query_id','target']].drop_duplicates(),on='query_id',how='left')
    own_ok=own[own.target.eq(own.parent_accession)].query_id.unique(); missing=sorted(set(qe.query_id)-set(own_ok))
    coverage={'queries':len(qe),'own_parent_hit_queries':len(own_ok),'missing_own_parent_hit_queries':len(missing),'complete':not missing}
    (out/'coverage.json').write_text(json.dumps({**coverage,'missing_query_ids':missing[:1000]},indent=2)+'\n')
    if missing: raise RuntimeError(f'coverage guard failed for {len(missing)} queries; aggregate metrics remain unread')
    # Attach target reactions, then take exact maximum reported identity for each candidate/reaction.
    z=h[['query_id','target','fident']].merge(tr,left_on='target',right_on='protein_id',how='inner')
    best=z.groupby(['query_id','reaction_id'],as_index=False).fident.max().rename(columns={'fident':'score'})
    relp=rel.assign(parent_query_id=rel.parent_accession.map(qid_parent))
    ps=relp.merge(best,left_on=['parent_query_id','reaction_id'],right_on=['query_id','reaction_id'],how='left',validate='many_to_one')[['parent_accession','reaction_id','score']].rename(columns={'score':'parent_score'})
    dec=qe[qe.category.ne('parent')][['parent_accession','category','query_id']].merge(rel,on='parent_accession',how='inner',validate='many_to_many')
    ds=dec.merge(best,on=['query_id','reaction_id'],how='left',validate='many_to_one')
    inst=ds.merge(ps,on=['parent_accession','reaction_id'],how='left',validate='many_to_one').rename(columns={'score':'decoy_score'})[['parent_accession','category','reaction_id','parent_score','decoy_score']]
    if inst[['parent_score','decoy_score']].isna().any().any(): raise RuntimeError('same-reaction score missing despite coverage guard')
    inst.to_csv(out/'pair_scores.csv',index=False)
    cats={c:metrics(g) for c,g in inst.groupby('category',sort=True)}; micro=metrics(inst)
    summary={'status':'evaluated_frozen_mmseqs_same_task_baseline','protocol_sha256':sha(PROTO),'search_sha256':sha(out/'all_search.tsv'),'coverage':coverage,'per_category':cats,'micro_all_decoys':micro,'selection_allowed':False,'threshold_tuning_used':False,'catalyst_scores_read_by_this_runner':False}
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n'); print(json.dumps(summary,indent=2))

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('action',choices=['prepare','search','finalize']); ap.add_argument('--output-dir',type=Path,default=OUT); a=ap.parse_args(); out=a.output_dir.resolve(); {'prepare':prepare,'search':search,'finalize':finalize}[a.action](out)
if __name__=='__main__': main()
