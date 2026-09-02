from __future__ import annotations
import argparse,json,hashlib
from pathlib import Path
import numpy as np,pandas as pd,torch
from sklearn.metrics import roc_auc_score
ROOT=Path(__file__).resolve().parents[3]
from projects.active.terpene_screening.evaluate_broad_rhea_benchmark import encode_chunks
from projects.active.terpene_screening.rank_open_world import load_models,load_feature_schema,load_protein_library,load_registered_reaction_feature_library,normalize_rows
PROTOCOL=ROOT/'projects/active/terpene_screening/CATALYST_OPEN_WORLD_ENZYMARC_V1.json'
SUPPORT=ROOT/'results/enzymarc_open_world_v1/support'; GATE=ROOT/'results/enzymarc_open_world_v1/sequence_form_gate'; FEATURES=ROOT/'results/enzymarc_open_world_v1/esmc_features'; OUT=ROOT/'results/enzymarc_open_world_v1/catalyst_primary_source'
GENERAL=ROOT/'data/catalyst_candidate_universes/general_merged/proteins'; REACTIONS=ROOT/'data/catalyst_candidate_universes/general_merged/reaction_features/drfp_categorical_rdkitplus_center_v1'; MODEL=ROOT/'results/catalyst_clean_mainline_v1/r2e_center_bounded_cap0p1'

def sha(p:Path): return hashlib.sha256(p.read_bytes()).hexdigest()
def metrics(x:pd.DataFrame)->dict:
    ps=x.parent_score.to_numpy(float); ds=x.decoy_score.to_numpy(float); delta=ps-ds
    labels=np.concatenate([np.ones(len(x),dtype=int),np.zeros(len(x),dtype=int)]); scores=np.concatenate([ps,ds])
    ratio=(ds+1.0)/np.maximum(ps+1.0,1e-8)
    return {'instances':int(len(x)),'paired_parent_win_rate':float(np.mean(ps>ds)),'paired_score_delta_mean':float(delta.mean()),'paired_score_delta_median':float(np.median(delta)),'parent_vs_decoy_AUROC':float(roc_auc_score(labels,scores)),'decoy_score_retention_ratio_median':float(np.median(ratio))}
def bootstrap(x:pd.DataFrame,n:int=1000,seed:int=20260902)->dict:
    # Collapse sufficient statistics by parent; bootstrap primary mean metrics only.
    g=x.assign(win=(x.parent_score>x.decoy_score).astype(float),delta=x.parent_score-x.decoy_score).groupby('parent_accession').agg(n=('win','size'),win_sum=('win','sum'),delta_sum=('delta','sum')).reset_index(drop=True)
    rng=np.random.default_rng(seed); vals=np.empty((n,2),float); N=len(g)
    a=g[['n','win_sum','delta_sum']].to_numpy(float)
    for i in range(n):
        ids=rng.integers(0,N,size=N); z=a[ids].sum(axis=0); vals[i]=[z[1]/z[0],z[2]/z[0]]
    lo,hi=np.quantile(vals,[.025,.975],axis=0)
    return {'replicates':n,'seed':seed,'paired_parent_win_rate_95ci':[float(lo[0]),float(hi[0])],'paired_score_delta_mean_95ci':[float(lo[1]),float(hi[1])]}
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--device',default='cuda'); a=ap.parse_args(); out=OUT; out.mkdir(parents=True,exist_ok=True)
    if (out/'summary.json').exists(): raise RuntimeError('EnzymARC source scores already exist; refusing accidental rerun')
    proto=json.loads(PROTOCOL.read_text()); feat_manifest=json.loads((FEATURES/'manifest.json').read_text())
    if feat_manifest['status']!='ready' or not feat_manifest['parity']['cosine_min']>=0.99999: raise RuntimeError('ESM-C external features not parity validated')
    gate=json.loads((GATE/'manifest.json').read_text());
    if gate.get('status')!='sequence_form_gate_frozen': raise RuntimeError('sequence-form gate not finalized')
    eligible=set(pd.read_csv(GATE/'eligible_parents.csv',dtype=str).parent_accession.astype(str))
    parents=pd.read_csv(SUPPORT/'parents.csv',dtype=str).fillna(''); parents=parents[parents.parent_accession.isin(eligible)]
    decoys=pd.read_csv(SUPPORT/'decoys.csv',dtype=str).fillna(''); decoys=decoys[decoys.parent_accession.isin(eligible)]
    rel=pd.read_csv(SUPPORT/'parent_reaction_relations.csv',dtype=str).fillna(''); rel=rel[rel.parent_accession.isin(eligible)]
    pfeat,pids=load_protein_library(GENERAL); pidx={x:i for i,x in enumerate(pids)}
    de=pd.read_csv(FEATURES/'entries.csv',dtype=str).sort_values('row'); dx=np.load(FEATURES/'embeddings.npy').astype(np.float32); dx=normalize_rows(dx); didx={(str(r.parent_accession),str(r.category)):int(r.row) for r in de.itertuples(index=False)}
    schema=load_feature_schema(MODEL); rx,rids=load_registered_reaction_feature_library(REACTIONS,schema); ridx={x:i for i,x in enumerate(rids)}; models=load_models(MODEL/'models','production',torch.device(a.device))
    if len(models)!=1: raise RuntimeError('expected exactly one frozen primary model')
    model=models[0]; pe=encode_chunks(model,pfeat,kind='protein',device=torch.device(a.device),chunk_size=8192); de_emb=encode_chunks(model,dx,kind='protein',device=torch.device(a.device),chunk_size=8192); re=encode_chunks(model,rx,kind='reaction',device=torch.device(a.device),chunk_size=8192)
    inst=decoys[['parent_accession','category']].merge(rel,on='parent_accession',validate='many_to_many'); pr=np.asarray([pidx[x] for x in inst.parent_accession],int); dr=np.asarray([didx[(p,c)] for p,c in zip(inst.parent_accession,inst.category)],int); rr=np.asarray([ridx[x] for x in inst.reaction_id],int)
    with torch.no_grad():
        ps=(pe[torch.as_tensor(pr,device=pe.device)]*re[torch.as_tensor(rr,device=re.device)]).sum(1).float().cpu().numpy(); ds=(de_emb[torch.as_tensor(dr,device=de_emb.device)]*re[torch.as_tensor(rr,device=re.device)]).sum(1).float().cpu().numpy()
    inst=inst.assign(parent_score=ps,decoy_score=ds); inst.to_csv(out/'pair_scores.csv',index=False)
    cats={c:{**metrics(g),'bootstrap':bootstrap(g)} for c,g in inst.groupby('category',sort=True)}; micro={**metrics(inst),'bootstrap':bootstrap(inst)}
    summary={'status':'evaluated_frozen_primary_source','protocol_sha256':sha(PROTOCOL),'model_checkpoint_dir':str(MODEL),'model_checkpoint_sha256':sha(next((MODEL/'models').glob('production*.pt'))),'feature_manifest_sha256':sha(FEATURES/'manifest.json'),'sequence_form_gate_sha256':sha(GATE/'manifest.json'),'per_category':cats,'micro_all_decoys':micro,'selection_allowed':False,'threshold_tuning_used':False,'lambdarank_pair_score_fabricated':False}
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n'); print(json.dumps(summary,indent=2))
if __name__=='__main__': main()
