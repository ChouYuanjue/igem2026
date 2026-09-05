from __future__ import annotations
import hashlib,json,math,sys
from pathlib import Path
import numpy as np,pandas as pd,torch,xgboost as xgb
ROOT=Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from projects.active.terpene_screening import run_r2e_lambdarank_fusion_v1 as base
from projects.active.terpene_screening import run_bime_r2e_clipzyme_expert_v1 as structural
from projects.active.terpene_screening.audit_bime_r2e_homology_admission_v1 import top_reaction_neighbors,CELL
from projects.active.terpene_screening.broad_rhea_metrics import evaluate_full_candidate_ranks

OUT=ROOT/'results/bime_rank_unified_v1/r2e_homology_context_v1';DIAG=ROOT/'results/bime_rank_unified_v1/r2e_homology_admission_v1';DEV=base.DEV_ROOT
FOLDS=(0,1,2);POOL=100;PREFIX=100
FEATURE_NAMES=['base_log_rank','base_reciprocal_rank','base_top10','base_top50','base_top100','homology_log_rank','homology_reciprocal_rank','homology_top10','homology_top50','homology_top100','best_log_rank','rank_gap_homology_minus_base','both_top10','both_top50','both_top100','max_train_reaction_similarity']

def stable(s):return int.from_bytes(hashlib.blake2b(s.encode(),digest_size=8).digest(),'big')%(2**31-1)
def lexical(ids):o=np.argsort(np.asarray(ids,dtype=object),kind='stable');r=np.empty(len(ids),dtype=np.int32);r[o]=np.arange(len(ids),dtype=np.int32);return r
def rank_features(rows,binv,hinv,n,sim):
 br=binv[rows].astype(float);hr=hinv[rows].astype(float);den=math.log1p(n);bl=np.log1p(br)/den;hl=np.log1p(hr)/den
 return np.column_stack([bl,1/br,br<=10,br<=50,br<=100,hl,1/hr,hr<=10,hr<=50,hr<=100,np.minimum(bl,hl),hl-bl,(br<=10)&(hr<=10),(br<=50)&(hr<=50),(br<=100)&(hr<=100),np.full(len(rows),sim)]).astype(np.float32)
def metric_map(d):
 return {k:float(d[c].mean()) for k,c in [('mrr','reciprocal_rank'),('map','average_precision'),('macro_roc_auc','roc_auc'),('ndcg_at_10','ndcg_at_10'),('hit_at_10','hit_at_10'),('hit_at_20','hit_at_20'),('hit_at_50','hit_at_50')]}
def structural_model(holdout):return structural._train([structural._load_cache(f) for f in FOLDS if f!=holdout],base._stable_seed(f'bime-r2e-clip|fold{holdout}'))
def homology_maps(fold,qids,pids,pidx):
 cell=CELL(fold);train=pd.read_csv(DEV/'benchmarks'/cell/'train_pairs.csv',dtype=str).fillna('');schema=base.load_feature_schema(base.REACTIONS);rf,rids=base.load_registered_reaction_feature_library(base.REACTIONS,schema);train_rids=sorted(set(train.reaction_id.astype(str))&set(rids));neigh=top_reaction_neighbors(qids,train_rids,rids,rf,int(schema.get('drfp_dimension',2048)));enz=train.groupby('reaction_id').protein_id.agg(lambda s:sorted(set(s.astype(str)))).to_dict();z=np.load(DIAG/f'fold{fold}_seed_top100.npz');seed_rows=z['seed_rows'].astype(np.int64);inds=z['indices'].astype(np.int32);vals=z['values'].astype(np.float32);cache={int(r):(inds[i],vals[i]) for i,r in enumerate(seed_rows)};out={}
 for q in qids:
  score={}
  for rid,rw in neigh[q]:
   if rw<=0:continue
   for p in enz.get(rid,[]):
    if p not in pidx or pidx[p] not in cache:continue
    ii,vv=cache[pidx[p]];ww=vv*float(rw)
    for r,v in zip(ii,ww,strict=True):
     r=int(r);v=float(v)
     if v>score.get(r,-1e9):score[r]=v
  order=np.asarray(sorted(score,key=lambda r:(-score[r],pids[r]))[:POOL],dtype=np.int32);out[q]=(order,float(neigh[q][0][1] if neigh[q] else 0.))
 return out

def prepare_fold(fold,device_name):
 device=torch.device(device_name);ids,_,reaction_ids,pe0,pe1,re0,re1=base._load_fold_embeddings(fold,device);pidx={p:i for i,p in enumerate(ids)};ridx={r:i for i,r in enumerate(reaction_ids)};lex=lexical(ids)
 dev=pd.read_csv(DEV/'baseline_base'/f'fold{fold}'/'dev_pairs.csv',dtype=str).fillna('');qids=sorted(dev.reaction_id.unique());pos=dev.groupby('reaction_id').protein_id.agg(lambda s:set(s.astype(str))).to_dict();hm=homology_maps(fold,qids,ids,pidx);smodel=structural_model(fold);clip_pt,clip_rows,clip_lookup,clip_rmat,clip_ridx=structural._load_clip_assets(ids,device);clip_lex=lex[clip_rows]
 diff=pd.read_csv(DEV/'difficulty'/CELL(fold)/'reaction_slices.csv',dtype={'reaction_id':str});simmap=dict(zip(diff.reaction_id.astype(str),diff.max_train_drfp_tanimoto.astype(float)));qrows=[ridx[q] for q in qids]
 Xs=[];ys=[];rowsall=[];brall=[];ptr=[0];prowsall=[];pbrall=[];pptr=[0];aud=[];baseq=[]
 for st in range(0,len(qids),32):
  stop=min(st+32,len(qids));rt=torch.as_tensor(qrows[st:stop],dtype=torch.long,device=device)
  with torch.no_grad():s0b=(re0[rt]@pe0.T).cpu().numpy();s1b=(re1[rt]@pe1.T).cpu().numpy()
  clip_local=[];cq=[]
  for j,q in enumerate(qids[st:stop]):
   if q in clip_ridx:cq.append(np.asarray(clip_rmat[clip_ridx[q]],dtype=np.float32));clip_local.append(j)
  cdict={}
  if cq:
   with torch.no_grad():cb=(torch.as_tensor(np.stack(cq),device=device)@clip_pt.T).cpu().numpy()
   for k,j in enumerate(clip_local):cdict[j]=cb[k]
  for j,q in enumerate(qids[st:stop]):
   s0=s0b[j].astype(np.float32);s1=s1b[j].astype(np.float32);o0,i0=base._full_order(s0,lex);o1,i1=base._full_order(s1,lex);sim=float(simmap[q]);use=sim<base.ROUTER_THRESHOLD;fb=o1 if use else o0;fbinv=i1 if use else i0
   cs=cdict.get(j);cinv=None;ctop=np.empty(0,dtype=np.int32);qsup=cs is not None
   if qsup:
    co=np.lexsort((clip_lex,-cs)).astype(np.int32);cinv=np.empty(len(co),dtype=np.int32);cinv[co]=np.arange(1,len(co)+1,dtype=np.int32);ctop=clip_rows[co[:POOL]]
   u=np.unique(np.concatenate([o0[:POOL],o1[:POOL],ctop])).astype(np.int32);bx=base._build_features(s0,s1,u,i0,i1,use,sim);sx=structural._clip_features(bx,u,cs,cinv,clip_lookup,qsup,len(ids));pred=smodel.predict(xgb.DMatrix(sx));ordidx=np.lexsort((lex[u],-pred));sel=u[ordidx[:min(PREFIX,len(u))]];mask=np.zeros(len(ids),dtype=bool);mask[sel]=1;bord=np.concatenate([sel,fb[~mask[fb]]]);binv=np.empty(len(ids),dtype=np.int32);binv[bord]=np.arange(1,len(ids)+1,dtype=np.int32)
   hord,hsim=hm[q];hinv=np.full(len(ids),len(ids)+1,dtype=np.int32);hinv[hord]=np.arange(1,len(hord)+1,dtype=np.int32);union=np.unique(np.concatenate([bord[:POOL],hord])).astype(np.int32);X=rank_features(union,binv,hinv,len(ids),hsim);pr=np.asarray(sorted(pidx[p] for p in pos[q]),dtype=np.int32);y=np.isin(union,pr).astype(np.uint8)
   Xs.append(X);ys.append(y);rowsall.append(union);brall.append(binv[union]);ptr.append(ptr[-1]+len(union));prowsall.append(pr);pbrall.append(binv[pr]);pptr.append(pptr[-1]+len(pr));baseq.append({'fold':fold,'query_id':q,**evaluate_full_candidate_ranks(binv[pr],len(ids))});aud.append({'fold':fold,'query_id':q,'union_size':len(union),'homology_top100_size':len(hord),'positive_in_union':int(y.sum()),'homology_similarity':hsim})
  print('prepare homology context fold',fold,stop,'/',len(qids),flush=True)
 out=OUT/'prepared'/f'fold{fold}';out.mkdir(parents=True,exist_ok=True);np.savez(out/'cache.npz',X=np.concatenate(Xs),labels=np.concatenate(ys),candidate_rows=np.concatenate(rowsall),base_ranks=np.concatenate(brall),query_ptr=np.asarray(ptr),positive_rows=np.concatenate(prowsall),positive_base_ranks=np.concatenate(pbrall),pos_ptr=np.asarray(pptr),lexical_rank=lex);pd.DataFrame({'query_id':qids}).to_csv(out/'queries.csv',index=False);pd.DataFrame(baseq).to_csv(out/'base_query_metrics.csv',index=False);pd.DataFrame(aud).to_csv(out/'audit.csv',index=False)

def load_cache(f):
 p=OUT/'prepared'/f'fold{f}';z=np.load(p/'cache.npz');return {'z':{k:z[k] for k in z.files},'queries':pd.read_csv(p/'queries.csv',dtype=str).query_id.tolist()}
def train(caches,seed):
 xs=[];ys=[];groups=[]
 for c in caches:
  z=c['z'];ptr=z['query_ptr']
  for i,q in enumerate(c['queries']):
   a,b=map(int,ptr[i:i+2]);X=z['X'][a:b];y=z['labels'][a:b];rows=z['candidate_rows'][a:b];pos=np.flatnonzero(y);neg=np.flatnonzero(~y.astype(bool));
   if not len(pos):continue
   hardness=np.minimum(X[neg,0],X[neg,5]);hard=neg[np.lexsort((rows[neg],hardness))][:128];rem=np.setdiff1d(neg,hard);rng=np.random.default_rng(stable(f'homctx|{q}'));rnd=rng.choice(rem,size=min(32,len(rem)),replace=False) if len(rem) else np.empty(0,dtype=int);keep=np.concatenate([pos,hard,rnd]);xs.append(X[keep]);ys.append(y[keep].astype(np.float32));groups.append(len(keep))
 X=np.concatenate(xs);y=np.concatenate(ys);d=xgb.DMatrix(X,label=y);d.set_group(groups);params={'objective':'rank:ndcg','eval_metric':'ndcg@10','tree_method':'hist','device':'cuda' if torch.cuda.is_available() else 'cpu','max_depth':2,'eta':.12,'min_child_weight':5.,'lambda':1.,'subsample':.9,'colsample_bytree':.9,'lambdarank_pair_method':'topk','lambdarank_num_pair_per_sample':20,'seed':seed,'verbosity':0};return xgb.train(params,d,num_boost_round=80)
def evaluate(model,c,fold):
 z=c['z'];pred=model.predict(xgb.DMatrix(z['X']));out=[]
 for i,q in enumerate(c['queries']):
  a,b=map(int,z['query_ptr'][i:i+2]);rows=z['candidate_rows'][a:b];fb=z['base_ranks'][a:b];take=np.lexsort((z['lexical_rank'][rows],-pred[a:b]))[:min(PREFIX,len(rows))];sel=rows[take];selfb=fb[take];sp={int(r):j+1 for j,r in enumerate(sel)};pa,pb=map(int,z['pos_ptr'][i:i+2]);pr=z['positive_rows'][pa:pb];pbr=z['positive_base_ranks'][pa:pb];ranks=[]
  for r,fr in zip(pr,pbr,strict=True):r=int(r);fr=int(fr);ranks.append(sp[r] if r in sp else len(sel)+fr-int(np.count_nonzero(selfb<fr)))
  out.append({'fold':fold,'query_id':q,**evaluate_full_candidate_ranks(np.asarray(ranks),len(z['lexical_rank']))})
 return pd.DataFrame(out)
def crossfit():
 cs={f:load_cache(f) for f in FOLDS};new=[]
 for h in FOLDS:m=train([cs[f] for f in FOLDS if f!=h],stable(f'homctx-fold{h}'));new.append(evaluate(m,cs[h],h));print('crossfit',h,flush=True)
 new=pd.concat(new,ignore_index=True);baseq=pd.concat([pd.read_csv(OUT/'prepared'/f'fold{f}'/'base_query_metrics.csv') for f in FOLDS]);nm,bm=metric_map(new),metric_map(baseq);delta={k:nm[k]-bm[k] for k in nm};folds={}
 for f in FOLDS:folds[str(f)]={k:metric_map(new[new.fold.eq(f)])[k]-metric_map(baseq[baseq.fold.eq(f)])[k] for k in nm}
 safe=all(v['mrr']>=-.005 and v['map']>=-.005 and v['hit_at_20']>=-.005 and v['hit_at_50']>=-.005 for v in folds.values());pooled=delta['mrr']>=-.002 and delta['map']>=-.002 and delta['hit_at_20']>=-.005 and delta['hit_at_50']>=-.005;material=delta['mrr']>=.003 or delta['map']>=.003 or delta['hit_at_20']>=.01 or delta['hit_at_50']>=.01;adm=bool(safe and pooled and material);new.to_csv(OUT/'development_oof_query_metrics.csv',index=False);res={'status':'admitted_internal' if adm else 'not_admitted','comparison':'second-stage homology context after crossfit structural BiME; top5 train-only reaction neighbors -> ESM-C transfer Top100; union with BiME Top100; fixed-capacity prefix reranker','base_structural_bime':bm,'homology_context':nm,'delta':delta,'fold_delta':folds,'fold_safe':safe,'pooled_safe':pooled,'material':material,'admitted':adm,'external_metrics_used':False,'next_if_admitted':'fit final on all folds then one frozen strict temporal retention'};(OUT/'development_result.json').write_text(json.dumps(res,indent=2)+'\n');print(json.dumps(res,indent=2));return adm
def fit_final():
 r=json.load(open(OUT/'development_result.json'));assert r['admitted'];m=train([load_cache(f) for f in FOLDS],stable('homctx-final'));o=OUT/'selected';o.mkdir(parents=True,exist_ok=True);m.save_model(o/'ranker.json');sha=hashlib.sha256((o/'ranker.json').read_bytes()).hexdigest();cfg={'feature_names':FEATURE_NAMES,'pool_k':POOL,'prefix_k':PREFIX,'ranker_sha256':sha,'training_folds':list(FOLDS),'homology_semantics':'top5 train reaction neighbors by binary-DRFP; train-only positive enzymes; ESM-C transfer','external_metrics_used':False};(o/'config.json').write_text(json.dumps(cfg,indent=2)+'\n');print(json.dumps(cfg,indent=2))
def main():
 import argparse;ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['prepare','crossfit','fit-final']);ap.add_argument('--fold',type=int);ap.add_argument('--device',default='cuda');a=ap.parse_args()
 if a.stage=='prepare':prepare_fold(a.fold,a.device)
 elif a.stage=='crossfit':crossfit()
 else:fit_final()
if __name__=='__main__':main()
