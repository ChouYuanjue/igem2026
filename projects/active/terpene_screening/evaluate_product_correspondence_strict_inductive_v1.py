from __future__ import annotations

import json, math, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path

ROOT=Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))

from projects.active.terpene_screening.multiscale_geometry import (
    _self_tuning_scale,
    one_step_diffusion_distance,
    partial_observation_pullback_affinity,
)
from projects.active.terpene_screening.out_of_sample_geometry import (
    attach_query_to_reference,
    query_geodesic_to_reference,
)
from projects.active.terpene_screening.evaluate_zero_shot_retrieval_cold import (
    reaction_features as molecule_side_features,
    best_match_similarity,
)
from projects.active.terpene_screening.train_dual_tower_cold import rank_metrics

CACHE=ROOT/'data/terpene_marts_adaptation'
GLOBAL=ROOT/'data/terpene_global_esmc_aligned_v1'
POCKET=ROOT/'data/terpene_pocket_local_aligned_v2'
MOTIF=ROOT/'data/terpene_family_aware_motif_coordinates_v1'
STRUCT=ROOT/'data/terpene_structural_observations_v1'
OUT=ROOT/'results/terpene_product_correspondence_strict_inductive_v1'
BUD=(3,10,20)


def bools(s): return s.astype(str).str.lower().isin({'1','true','yes'})


def chordal_full(emb: np.ndarray, available: np.ndarray) -> np.ndarray:
    x=np.asarray(emb,dtype=np.float64); a=np.asarray(available,bool)
    norm=np.linalg.norm(x,axis=1,keepdims=True)
    good=a&(norm[:,0]>1e-12)
    y=np.zeros_like(x); y[good]=x[good]/norm[good]
    s=np.clip(y@y.T,-1,1)
    d=np.sqrt(np.maximum(2-2*s,0))
    d[~(good[:,None]&good[None,:])]=np.inf
    np.fill_diagonal(d,0)
    return d


def diffusion_reference_and_cross(
    similarity: np.ndarray,
    available: np.ndarray,
    ref: np.ndarray,
) -> tuple[np.ndarray,np.ndarray,np.ndarray,np.ndarray]:
    """Strict reference-only t=1 diffusion geometry plus all->reference cross distance."""
    sim=np.asarray(similarity,dtype=np.float64)
    av=np.asarray(available,bool)
    rr=np.asarray(sim[np.ix_(ref,ref)],dtype=np.float64)
    rav=av[ref]
    rr=np.where(np.outer(rav,rav),np.maximum(rr,0),0)
    np.fill_diagonal(rr,0)
    A=csr_matrix(rr)
    degree=np.asarray(A.sum(1)).reshape(-1)
    relational=rav&(degree>0)
    dref=one_step_diffusion_distance(A)
    dref[~(relational[:,None]&relational[None,:])]=np.inf
    np.fill_diagonal(dref,0)

    n=len(av); m=len(ref)
    cross=np.full((n,m),np.inf,dtype=np.float32)
    cross[ref,:]=dref.astype(np.float32)

    total=float(degree.sum())
    if total<=1e-12:
        return dref,relational,cross,np.zeros(n,dtype=bool)
    safe=np.maximum(degree,1e-12)
    pi=np.maximum(degree/total,1e-12)
    Pref=rr/safe[:,None]
    Wref=Pref/np.sqrt(pi[None,:])
    refnorm=np.sum(Wref*Wref,axis=1)

    ref_mask=np.zeros(n,dtype=bool); ref_mask[ref]=True
    q_available=np.zeros(n,dtype=bool); q_available[ref]=relational
    ext=np.flatnonzero(~ref_mask & av)
    if len(ext):
        aq=np.asarray(sim[np.ix_(ext,ref)],dtype=np.float64)
        aq[:,~rav]=0
        qdeg=aq.sum(1)
        ok=qdeg>1e-12
        if np.any(ok):
            Pq=aq[ok]/qdeg[ok,None]
            Wq=Pq/np.sqrt(pi[None,:])
            qnorm=np.sum(Wq*Wq,axis=1)
            d2=qnorm[:,None]+refnorm[None,:]-2*(Wq@Wref.T)
            d=np.sqrt(np.maximum(d2,0))
            d[:,~relational]=np.inf
            good_ext=ext[ok]
            cross[good_ext,:]=d.astype(np.float32)
            q_available[good_ext]=True
    return dref,relational,cross,q_available


def coordinate_reference_and_cross(
    full_distance: np.ndarray,
    available: np.ndarray,
    ref: np.ndarray,
) -> tuple[np.ndarray,np.ndarray,np.ndarray,np.ndarray]:
    a=np.asarray(available,bool)
    d=np.asarray(full_distance,dtype=np.float64)
    rav=a[ref]
    dref=np.asarray(d[np.ix_(ref,ref)],dtype=np.float64)
    cross=np.asarray(d[:,ref],dtype=np.float32)
    qav=a.copy()
    return dref,rav,cross,qav


def length_graph_from_affinity(w: csr_matrix) -> tuple[csr_matrix,float]:
    w=w.astype(np.float64).maximum(w.astype(np.float64).T).tocsr()
    w.setdiag(0);w.eliminate_zeros()
    coo=w.tocoo(); upper=coo.row<coo.col
    edge=np.sqrt(np.maximum(-np.log(np.clip(coo.data[upper],1e-300,1)),1e-12))
    ell=float(np.median(edge))
    lengths=np.sqrt(np.maximum(-np.log(np.clip(w.data,1e-300,1)),1e-12))
    return csr_matrix((lengths,w.indices,w.indptr),shape=w.shape),ell


def factor_all_to_reference(
    ref: np.ndarray,
    view_data: list[tuple[np.ndarray,np.ndarray,np.ndarray,np.ndarray]],
) -> tuple[np.ndarray,dict]:
    """Build reference-only atlas, then attach every excluded point out of sample."""
    ref_dist=[x[0] for x in view_data]
    ref_avail=[x[1] for x in view_data]
    cross=[x[2] for x in view_data]
    qall=[x[3] for x in view_data]
    affinity,diag=partial_observation_pullback_affinity(ref_dist,ref_avail)
    length_graph,ell=length_graph_from_affinity(affinity)
    scales=[_self_tuning_scale(d,a,epsilon=1e-8) for d,a in zip(ref_dist,ref_avail)]
    dref=shortest_path(length_graph,directed=False,unweighted=False)
    if not np.all(np.isfinite(dref)):
        raise RuntimeError('reference atlas disconnected')
    n=cross[0].shape[0]
    out=np.full((n,len(ref)),np.inf,dtype=np.float32)
    refpos={int(g):i for i,g in enumerate(ref)}
    for g,local in refpos.items():
        out[g]=np.asarray(dref[local]/ell,dtype=np.float32)
    external=[i for i in range(n) if i not in refpos]
    for i in external:
        qdist=[np.asarray(x[i],dtype=np.float64) for x in cross]
        qavail=[bool(x[i]) for x in qall]
        attachment=attach_query_to_reference(
            qdist,qavail,ref_avail,scales,
        )
        # factor characteristic-length normalization is the same as canonical.
        from dataclasses import replace
        attachment=replace(attachment,edge_lengths=attachment.edge_lengths/ell)
        unit_graph=length_graph.copy(); unit_graph.data/=ell
        out[i]=query_geodesic_to_reference(unit_graph,attachment).astype(np.float32)
    if not np.all(np.isfinite(out)):
        raise RuntimeError('nonfinite all-to-reference geodesics')
    return out,{
        'reference_count':int(len(ref)),
        'external_count':int(n-len(ref)),
        'graph_k':int(diag['graph_k']),
        'characteristic_length':ell,
    }


def symmetric_side_similarity(reactions: pd.DataFrame,key: str) -> np.ndarray:
    feats=[molecule_side_features(x) for x in reactions.reaction_smiles.astype(str)]
    n=len(feats); sim=np.zeros((n,n),dtype=np.float32)
    for i in range(n):
        if feats[i][key]: sim[i,i]=1
        for j in range(i):
            if not(feats[i][key] and feats[j][key]): continue
            a=float(best_match_similarity(feats[i][key],feats[j][key]))
            b=float(best_match_similarity(feats[j][key],feats[i][key]))
            sim[i,j]=sim[j,i]=0.5*(a+b)
    return sim


def build_protein_factor_inputs():
    views=[]
    g=np.load(GLOBAL/'embeddings.npy',mmap_mode='r'); ga=np.load(GLOBAL/'available.npy').astype(bool)
    views.append(('global_esmc','coordinate',chordal_full(g,ga),ga))
    p=np.load(POCKET/'embeddings.npy',mmap_mode='r'); pa=np.load(POCKET/'available.npy').astype(bool)
    views.append(('pocket_local_esmc','coordinate',chordal_full(p,pa),pa))
    for name in ['typeI_aspartate','nse_dte','dxdd','qw']:
        e=np.load(MOTIF/f'{name}_embeddings.npy',mmap_mode='r'); a=np.load(MOTIF/f'{name}_available.npy').astype(bool)
        views.append((name,'coordinate',chordal_full(e,a),a))
    for name in ['whole_3di','pocket_3di','pocket_ot']:
        s=np.load(STRUCT/name/'similarity.npy',mmap_mode='r')
        a=np.load(STRUCT/name/'available.npy').astype(bool)
        views.append((name,'diffusion',s,a))
    return views


def build_reaction_factor_inputs(reactions: pd.DataFrame):
    feat=np.load(CACHE/'reaction_features.npy',mmap_mode='r')[:,:2048]
    norms=np.linalg.norm(feat,axis=1,keepdims=True); a=norms[:,0]>1e-12
    y=np.zeros_like(feat,dtype=np.float64); y[a]=feat[a]/norms[a]
    sim=np.clip(y@y.T,-1,1); d=np.sqrt(np.maximum(2-2*sim,0)); d[~np.outer(a,a)]=np.inf; np.fill_diagonal(d,0)
    react=symmetric_side_similarity(reactions,'reactant_fps')
    prod=symmetric_side_similarity(reactions,'product_fps')
    all_av=np.ones(len(reactions),dtype=bool)
    return [
        ('drfp','coordinate',d,a),
        ('reactant','diffusion',react,all_av),
        ('product','diffusion',prod,all_av),
    ]


def materialize_views(raw_views,ref):
    out=[]
    for name,kind,data,avail in raw_views:
        if kind=='coordinate':
            x=coordinate_reference_and_cross(data,avail,ref)
        else:
            x=diffusion_reference_and_cross(data,avail,ref)
        out.append(x)
    return out


def score_r2e(q,train,Dr2,De2,r_global_to_local,e_global_to_local,ri,pi,nE):
    marg_r=min(float(Dr2[q,r_global_to_local[ri[str(x)] ]]) for x in train.rhea_id.unique())
    e_support=np.array([e_global_to_local[pi[str(x)]] for x in train.Entry.unique()],dtype=int)
    marg_e=np.min(De2[:,e_support],axis=1)
    joint=np.full(nE,np.inf)
    for rid,g in train.groupby('rhea_id',sort=False):
        rr=r_global_to_local[ri[str(rid)]]
        es=np.array([e_global_to_local[pi[str(x)]] for x in g.Entry.unique()],dtype=int)
        joint=np.minimum(joint,float(Dr2[q,rr])+np.min(De2[:,es],axis=1))
    return -np.maximum(joint-marg_r-marg_e,0)


def score_e2r(q,train,Dr2,De2,r_global_to_local,e_global_to_local,ri,pi,nR):
    marg_e=min(float(De2[q,e_global_to_local[pi[str(x)] ]]) for x in train.Entry.unique())
    r_support=np.array([r_global_to_local[ri[str(x)]] for x in train.rhea_id.unique()],dtype=int)
    marg_r=np.min(Dr2[:,r_support],axis=1)
    joint=np.full(nR,np.inf)
    for pid,g in train.groupby('Entry',sort=False):
        ee=e_global_to_local[pi[str(pid)]]
        rs=np.array([r_global_to_local[ri[str(x)]] for x in g.rhea_id.unique()],dtype=int)
        joint=np.minimum(joint,float(De2[q,ee])+np.min(Dr2[:,rs],axis=1))
    return -np.maximum(joint-marg_e-marg_r,0)


def main():
    t0=time.time()
    proteins=pd.read_csv(CACHE/'protein_entities.csv',dtype=str).fillna('')
    reactions=pd.read_csv(CACHE/'reaction_entities.csv',dtype=str).fillna('')
    pairs=pd.read_csv(CACHE/'marts_pair_folds.csv',dtype=str).fillna('')
    pairs[['protein_fold','reaction_fold']]=pairs[['protein_fold','reaction_fold']].astype(int)
    pairs['protein_seen']=bools(pairs.protein_seen);pairs['reaction_seen']=bools(pairs.reaction_seen)
    pids=proteins.protein_id.astype(str).tolist();rids=reactions.reaction_id.astype(str).tolist()
    pi={x:i for i,x in enumerate(pids)};ri={x:i for i,x in enumerate(rids)}

    # Fold membership is defined only for labelled entities. Unlabelled proteins
    # remain legitimate label-free reference points in every split.
    pfold={str(k):int(v) for k,v in pairs[['Entry','protein_fold']].drop_duplicates().itertuples(index=False)}
    rfold={str(k):int(v) for k,v in pairs[['rhea_id','reaction_fold']].drop_duplicates().itertuples(index=False)}

    print('precomputing raw measurement views',flush=True)
    p_raw=build_protein_factor_inputs()
    r_raw=build_reaction_factor_inputs(reactions)

    p_cache={}; r_cache={}
    for f in range(5):
        pref=np.array([i for i,pid in enumerate(pids) if pfold.get(pid)!=f],dtype=int)
        print('protein fold',f,'reference',len(pref),flush=True)
        p_cache[f]=factor_all_to_reference(pref,materialize_views(p_raw,pref))
        rref=np.array([i for i,rid in enumerate(rids) if rfold.get(rid)!=f],dtype=int)
        print('reaction fold',f,'reference',len(rref),flush=True)
        r_cache[f]=factor_all_to_reference(rref,materialize_views(r_raw,rref))

    rows=[]; split_rows=[]
    for pf in range(5):
      for rf in range(5):
        if not(pf==4 or rf==4): continue
        sid=f'p{pf}_r{rf}'
        train=pairs[pairs.protein_fold.ne(pf)&pairs.reaction_fold.ne(rf)].drop_duplicates(['rhea_id','Entry'])
        test=pairs[pairs.protein_fold.eq(pf)&pairs.reaction_fold.eq(rf)&~pairs.protein_seen&~pairs.reaction_seen].drop_duplicates(['rhea_id','Entry'])
        Dp,pinfo=p_cache[pf]; Dr,rinfo=r_cache[rf]
        pref=np.array([i for i,pid in enumerate(pids) if pfold.get(pid)!=pf],dtype=int)
        rref=np.array([i for i,rid in enumerate(rids) if rfold.get(rid)!=rf],dtype=int)
        pmap={int(g):i for i,g in enumerate(pref)}; rmap={int(g):i for i,g in enumerate(rref)}
        Dr2=np.asarray(Dr,dtype=np.float64)**2; Dp2=np.asarray(Dp,dtype=np.float64)**2
        for rid,g in test.groupby('rhea_id',sort=True):
            sc=score_r2e(ri[rid],train,Dr2,Dp2,rmap,pmap,ri,pi,len(pids))
            rows.append({'split_id':sid,'direction':'reaction_to_enzyme','query_id':rid,**rank_metrics(sc,pids,set(g.Entry.astype(str)),set(),BUD)})
        for pid,g in test.groupby('Entry',sort=True):
            sc=score_e2r(pi[pid],train,Dr2,Dp2,rmap,pmap,ri,pi,len(rids))
            rows.append({'split_id':sid,'direction':'enzyme_to_reaction','query_id':pid,**rank_metrics(sc,rids,set(g.rhea_id.astype(str)),set(),BUD)})
        split_rows.append({'split_id':sid,'train_pairs':len(train),'test_pairs':len(test),
                           'protein_reference_count':pinfo['reference_count'],'reaction_reference_count':rinfo['reference_count'],
                           'protein_overlap':len(set(train.Entry)&set(test.Entry)),
                           'reaction_overlap':len(set(train.rhea_id)&set(test.rhea_id))})
    q=pd.DataFrame(rows); sp=pd.DataFrame(split_rows)
    metrics=[]
    for direction,g in q.groupby('direction'):
        rec={'direction':direction,'n_queries':len(g),'mrr':float(g.reciprocal_rank.mean()),
             'median_rank':float(g.best_positive_rank.median())}
        for k in BUD:
            rec[f'hit{k}']=float(g[f'hit_at_{k}'].mean()); rec[f'recall{k}']=float(g[f'positive_recall_at_{k}'].mean())
        metrics.append(rec)
    OUT.mkdir(parents=True,exist_ok=True)
    q.to_csv(OUT/'query_metrics.csv',index=False); sp.to_csv(OUT/'split_summary.csv',index=False)
    pd.DataFrame(metrics).to_csv(OUT/'metrics.csv',index=False)
    summary={
        'version':'terpene-product-correspondence-strict-inductive-v1',
        'purpose':'inductive transfer audit; does not replace canonical transductive-side-information development result',
        'protocol':'for each held-out factor fold, rebuild the reference atlas without those entities; each excluded entity is attached out of sample using only its query-to-reference molecular observations; pair labels remain double-cold',
        'protein_views':['global_esmc','pocket_local_esmc','typeI_aspartate','nse_dte','dxdd','qw','whole_3di','pocket_3di','pocket_ot'],
        'reaction_views':['drfp','reactant_neighbourhood','product_neighbourhood'],
        'structural_policy':'reference diffusion coordinates are rebuilt from reference-only raw Foldseek/OT affinity subgraphs; held-out query diffusion coordinates use only query-to-reference affinities',
        'pair_field':'same correspondence defect F=-Delta_Omega',
        'metrics':metrics,
        'all_train_test_entity_overlaps_zero':bool(((sp.protein_overlap==0)&(sp.reaction_overlap==0)).all()),
        'elapsed_seconds':time.time()-t0,
    }
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2),flush=True)

if __name__=='__main__': main()
