from __future__ import annotations

import argparse, json, sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT=Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from projects.active.terpene_screening.gate_matrix import canonical_or_raw_reaction

POS=ROOT/'data/terpene/enzyme_terpene_synthase.tsv'
EMB=ROOT/'data/terpene_embeddings/esmc600m_mean'
CAGE=ROOT/'results/terpene_cage_screen/all_rhea_gate/all_pair_scores.csv'
SIMDIR=ROOT/'results/terpene_zero_shot_cold'
FOLDS=ROOT/'data/terpene_cold_splits/reaction_cluster_folds.csv'
OUT=ROOT/'results/terpene_cage_residual_rescue'


def norm(x):
    d=np.linalg.norm(x,axis=1,keepdims=True); d[d==0]=1
    return x/d


def tied_rank(x, ids):
    order=np.lexsort((ids,-x)); sx=x[order]; out=np.empty(len(x),np.float32)
    if len(x)==1: out[0]=1; return out
    i=0
    while i<len(x):
        j=i+1
        while j<len(x) and sx[j]==sx[i]: j+=1
        out[order[i:j]]=1-((i+j-1)/2)/(len(x)-1)
        i=j
    return out


def load_inputs():
    e=pd.read_csv(EMB/'entries.csv',dtype={'Entry':str}).sort_values('row')
    pids=e.Entry.astype(str).tolist(); P=norm(np.load(EMB/'embeddings.npy').astype(np.float32))
    r=pd.read_csv(SIMDIR/'reaction_similarity_entries.csv',dtype={'rhea_id':str}).sort_values('row')
    rids=r.rhea_id.astype(str).tolist(); S=np.load(SIMDIR/'reaction_similarity_matrix.npy').astype(np.float32)
    pos=pd.read_csv(POS,sep='\t',dtype=str).fillna('')[['rhea_id','Entry','smiles_seq']].drop_duplicates(['rhea_id','Entry'])
    return P,pids,S,rids,pos


def label_matrix(pos,rids,pids):
    ri={x:i for i,x in enumerate(rids)}; pi={x:i for i,x in enumerate(pids)}
    y=np.zeros((len(rids),len(pids)),np.uint8)
    for z in pos.itertuples(index=False):
        if z.rhea_id in ri and z.Entry in pi: y[ri[z.rhea_id],pi[z.Entry]]=1
    return y


def base_scores(P,pids,S,rids,pos,topk=5):
    pi={x:i for i,x in enumerate(pids)}
    by=pos.groupby('rhea_id').Entry.apply(lambda s:sorted(set(s)&set(pi))).to_dict()
    can=pos.groupby('rhea_id').smiles_seq.first().map(canonical_or_raw_reaction).to_dict()
    C=P@P.T; out=np.zeros((len(rids),len(pids)),np.float32)
    for i,rid in enumerate(rids):
        neigh=[]
        for j,sid in enumerate(rids):
            if sid==rid: continue
            if can.get(rid) and can.get(rid)==can.get(sid): continue
            neigh.append((sid,float(S[i,j])))
        neigh.sort(key=lambda z:(-z[1],z[0])); weights={}
        for sid,w in neigh[:topk]:
            for uid in by.get(sid,[]): weights[uid]=max(weights.get(uid,0),w)
        if weights:
            ids=sorted(weights); cols=np.array([pi[x] for x in ids]); w=np.array([weights[x] for x in ids])
            out[i]=(C[:,cols]*w[None,:]).max(axis=1)
    return out


def cage_features(rids,pids):
    ri={x:i for i,x in enumerate(rids)}; pi={x:i for i,x in enumerate(pids)}
    raw=np.full((len(rids),len(pids)),np.nan,np.float64)
    d=pd.read_csv(CAGE,dtype=str).fillna(''); d.cage_score=pd.to_numeric(d.cage_score,errors='coerce')
    for z in d.itertuples(index=False):
        if z.reaction_id in ri and z.uniprot_id in pi and np.isfinite(z.cage_score): raw[ri[z.reaction_id],pi[z.uniprot_id]]=z.cage_score
    rank=np.zeros(raw.shape,np.float32); rel=np.zeros(len(rids),np.float32); diag=[]; ids=np.array(pids)
    for i,rid in enumerate(rids):
        m=np.isfinite(raw[i]); v=raw[i,m]; uniq=len(np.unique(v)) if len(v) else 0; spread=float(v.max()-v.min()) if len(v) else 0
        if len(v): rank[i,m]=tied_rank(v,ids[m]); rel[i]=min(1,np.log1p(uniq)/np.log(20))*float(spread>0)
        diag.append({'reaction_id':rid,'n_cage_candidates':int(m.sum()),'unique_scores':uniq,'spread':spread,'reliability':float(rel[i])})
    return raw,rank,rel,pd.DataFrame(diag)


def feature_tensor(base,raw,crank,rel,pids):
    ids=np.array(pids); brank=np.stack([tied_rank(x,ids) for x in base])
    avail=np.isfinite(raw).astype(np.float32); clipped=np.nan_to_num(raw,nan=0.0)
    logp=np.zeros(base.shape,np.float32); m=avail.astype(bool)
    logp[m]=np.clip(np.log10(np.clip(clipped[m],1e-40,1))/40+1,0,1)
    R=np.repeat(rel[:,None],base.shape[1],axis=1)
    X=np.stack([base,brank,avail,crank,logp,R,brank*crank,crank*R],axis=-1).astype(np.float32)
    return X,brank


def sample_pairs(rx,y,brank,avail,rng):
    rr=[]; pp=[]; n=y.shape[1]
    for i in rx:
        pos=np.flatnonzero(y[i]); hard=np.argpartition(-brank[i],min(100,n-1))[:100]
        cage=np.flatnonzero(avail[i]); neg=np.flatnonzero(y[i]==0); rnd=rng.choice(neg,min(50,len(neg)),replace=False)
        cols=np.unique(np.concatenate([pos,hard,cage,rnd])); rr.append(np.full(len(cols),i)); pp.append(cols)
    return np.concatenate(rr).astype(int),np.concatenate(pp).astype(int)


def calibrate_oof(X,y,brank,avail,folds,seed):
    oof=np.zeros(y.shape,np.float32); rows=[]
    for f in sorted(set(folds)):
        tr=np.flatnonzero(folds!=f); te=np.flatnonzero(folds==f)
        rr,pp=sample_pairs(tr,y,brank,avail,np.random.default_rng(seed+int(f)))
        model=make_pipeline(StandardScaler(),LogisticRegression(C=.2,class_weight='balanced',max_iter=500,random_state=seed+int(f)))
        model.fit(X[rr,pp],y[rr,pp]); oof[te]=model.predict_proba(X[te].reshape(-1,X.shape[-1]))[:,1].reshape(len(te),y.shape[1])
        lr=model.named_steps['logisticregression']
        rows.append({'fold':int(f),'n_train_pairs':len(rr),'n_positive':int(y[rr,pp].sum()),'intercept':float(lr.intercept_[0]),**{f'coef_{k}':float(v) for k,v in enumerate(lr.coef_[0])}})
    return oof,pd.DataFrame(rows)


def order(scores,ids): return np.lexsort((ids,-scores))


def rescue_panel(base_order,rescue,avail,B,k,ids):
    panel=list(base_order[:B-k]); used=set(panel)
    for c in order(rescue,ids):
        if len(panel)>=B: break
        if c not in used and avail[c]: panel.append(int(c)); used.add(int(c))
    for c in base_order:
        if len(panel)>=B: break
        if int(c) not in used: panel.append(int(c)); used.add(int(c))
    return np.array(panel)


def evaluate(y,base,brank,cal,crank,avail,rids,pids,budgets):
    ids=np.array(pids); gain=cal-brank; rows=[]; panels=[]
    for i,rid in enumerate(rids):
        bo=order(base[i],ids); co=order(cal[i],ids); has=bool(avail[i].any()); pos=y[i].astype(bool)
        for B in budgets:
            methods={'base_only':bo[:B],'calibrated_full_rerank':co[:B]}
            for k in ([1,2] if B<20 else [1,2,5,10]):
                methods[f'base_plus_calibrated_rescue_k{k}']=rescue_panel(bo,gain[i],avail[i],B,k,ids)
                methods[f'base_plus_cage_rank_rescue_k{k}']=rescue_panel(bo,crank[i],avail[i],B,k,ids)
                methods[f'base_plus_cage_rank_base_tiebreak_k{k}']=rescue_panel(
                    bo,
                    crank[i] + brank[i] * 1e-6,
                    avail[i],
                    B,
                    k,
                    ids,
                )
            for name,panel in methods.items():
                hits=int(pos[panel].sum())
                rows.append({'reaction_id':rid,'scope':'all513','has_cage':has,'method':name,'B':B,'hits':hits,'hit':hits>0,'precision':hits/B,'positive_recall':hits/max(1,int(pos.sum()))})
                for rank,c in enumerate(panel,1): panels.append({'reaction_id':rid,'method':name,'B':B,'rank':rank,'uniprot_id':pids[int(c)],'label':int(pos[c]),'base_score':float(base[i,c]),'base_rank':float(brank[i,c]),'cage_available':bool(avail[i,c]),'cage_rank':float(crank[i,c]),'calibrated_score':float(cal[i,c]),'residual_gain':float(gain[i,c])})
    long=pd.DataFrame(rows); cov=long[long.has_cage].copy(); cov.scope='cage_available_reactions'; long=pd.concat([long,cov],ignore_index=True)
    metrics=long.groupby(['scope','method','B']).agg(n_reactions=('reaction_id','size'),hit_probability=('hit','mean'),expected_hits=('hits','mean'),precision=('precision','mean'),positive_recall=('positive_recall','mean')).reset_index()
    return metrics,pd.DataFrame(panels)


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--output-dir',type=Path,default=OUT); ap.add_argument('--topk-seed-reactions',type=int,default=5); ap.add_argument('--seed',type=int,default=20260723); ap.add_argument('--budgets',default='5,10,20'); a=ap.parse_args()
    budgets=tuple(int(x) for x in a.budgets.split(',') if x); out=a.output_dir.resolve(); out.mkdir(parents=True,exist_ok=True)
    P,pids,S,rids,pos=load_inputs(); y=label_matrix(pos,rids,pids); base=base_scores(P,pids,S,rids,pos,a.topk_seed_reactions)
    raw,crank,rel,diag=cage_features(rids,pids); X,brank=feature_tensor(base,raw,crank,rel,pids)
    fd=pd.read_csv(FOLDS,dtype={'reaction_id':str}); fm=dict(zip(fd.reaction_id,pd.to_numeric(fd.fold).astype(int)))
    folds=np.array([fm[x] for x in rids]); cal,coef=calibrate_oof(X,y,brank,np.isfinite(raw),folds,a.seed)
    metrics,panels=evaluate(y,base,brank,cal,crank,np.isfinite(raw),rids,pids,budgets)
    metrics.to_csv(out/'metrics.csv',index=False); panels.to_csv(out/'panels.csv',index=False); diag.to_csv(out/'cage_diagnostics.csv',index=False); coef.to_csv(out/'calibrator_coefficients.csv',index=False)
    comp=[]
    for (scope,B),g in metrics.groupby(['scope','B']):
        base_row=g[g.method.eq('base_only')].iloc[0]
        for z in g.itertuples(index=False): comp.append({'scope':scope,'B':int(B),'method':z.method,'hit_probability':z.hit_probability,'expected_hits':z.expected_hits,'delta_hit_vs_base':z.hit_probability-base_row.hit_probability,'delta_expected_vs_base':z.expected_hits-base_row.expected_hits})
    comp=pd.DataFrame(comp); comp.to_csv(out/'comparison.csv',index=False)
    best=comp.sort_values(['scope','B','hit_probability','expected_hits'],ascending=[True,True,False,False]).groupby(['scope','B'],as_index=False).head(1); best.to_csv(out/'best_methods.csv',index=False)
    summary={'n_reactions':len(rids),'n_proteins':len(pids),'n_positive_pairs':int(y.sum()),'n_reactions_with_cage':int((np.isfinite(raw).sum(1)>0).sum()),'n_cage_pairs':int(np.isfinite(raw).sum()),'topk_seed_reactions':a.topk_seed_reactions,'budgets':budgets,'feature_names':['base_score','base_rank','cage_available','cage_rank','scaled_log_probability','cage_reliability','base_x_cage_rank','cage_rank_x_reliability'],'limitations':['Legacy CAGE outputs contain sigmoid probabilities only; exact-zero logits cannot be recovered.','Only the light calibrator is reaction-cluster OOF; checkpoint data overlap remains possible.','Unannotated pairs are evaluated as negatives although some may be unknown positives.'],'outputs':{'metrics':str(out/'metrics.csv'),'comparison':str(out/'comparison.csv'),'best_methods':str(out/'best_methods.csv'),'panels':str(out/'panels.csv'),'diagnostics':str(out/'cage_diagnostics.csv'),'coefficients':str(out/'calibrator_coefficients.csv')}}
    (out/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8'); print(json.dumps(summary,indent=2)); print(best.to_string(index=False))

if __name__=='__main__': main()
