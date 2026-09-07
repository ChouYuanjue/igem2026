from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path
import numpy as np,pandas as pd,torch,xgboost as xgb
from projects.active.terpene_screening import run_unified_safe_system_e2r_anchored_lambdamart_v3 as v3

ROOT=v3.ROOT
PROTOCOL=ROOT/'projects/active/terpene_screening/UNIFIED_SAFE_SYSTEM_E2R_ANCHORED_LAMBDAMART_V3_CONFIRMATION.json'
DEV_OUT=ROOT/'results/unified_safe_system_v1/e2r_anchored_lambdamart_v3_dev/anchored'
CONF_ROOT=ROOT/'results/unified_safe_system_v1/e2r_anchored_lambdamart_v3_confirmation'
CONF_ER=CONF_ROOT/'experts'; OUT=CONF_ROOT/'anchored'

def protocol(): return json.loads(PROTOCOL.read_text())
def sha(path:Path)->str: return hashlib.sha256(path.read_bytes()).hexdigest()
def assert_frozen_selection(p:dict)->dict:
 d=p['development_selection']; s=ROOT/d['selection_result']; r=ROOT/d['search_results']
 assert sha(s)==d['selection_result_sha256']; assert sha(r)==d['search_results_sha256']
 actual=json.loads(s.read_text()); assert actual['status']=='selected_development_pass_confirmation_authorized'; assert actual['configuration_count']==d['configuration_count']; assert actual['feasible_count']==d['feasible_count']; assert actual['selected_config']==d['selected_config']; return d['selected_config']

def train_final_ranker():
 p=protocol(); selected=assert_frozen_selection(p); cfg=selected['ranker_config']; sampled=[]
 old=v3.OUT; v3.OUT=DEV_OUT
 try:
  for f in [0,1,2]: sampled.append(v3.sampled_training(v3.load_cache(f),f))
 finally: v3.OUT=old
 xs=[];ys=[];groups=[]
 for X,y,g in sampled: xs.append(X); ys.append(y); groups.extend(g)
 model=v3.train_model(np.concatenate(xs),np.concatenate(ys),groups,cfg,int(p['final_ranker']['seed'])); OUT.mkdir(parents=True,exist_ok=True); path=OUT/'final_ranker.json'; model.save_model(path); meta={'selected_config':selected,'seed':int(p['final_ranker']['seed']),'training_folds':[0,1,2],'confirmation_labels_used':False,'checkpoint':str(path),'checkpoint_sha256':sha(path)}; (OUT/'final_ranker_meta.json').write_text(json.dumps(meta,indent=2)+'\n'); print(json.dumps(meta,indent=2))

def prepare_confirmation():
 p=protocol(); assert_frozen_selection(p); split=p['confirmation_split']; fold=int(split['dev_fold']); old_er,old_out=v3.ER,v3.OUT; v3.ER,v3.OUT=CONF_ER,OUT
 try:
  dev,common,qids,emb=v3.load_fold_embeddings(fold)
 finally: v3.ER,v3.OUT=old_er,old_out
 positives=dev.groupby('protein_id').reaction_id.apply(set).to_dict(); ridx={r:i for i,r in enumerate(common)}; device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); rt={n:torch.from_numpy(emb[n][1]).to(device) for n in v3.NAMES}
 X=[]; rows_all=[]; ranks_all=[]; labels=[]; offsets=[0]; pos_rows=[]; pos_base_ranks=[]; pos_offsets=[0]; baseline=[]; audits=[]
 with torch.no_grad():
  for st in range(0,len(qids),64):
   sc={n:(torch.from_numpy(emb[n][0][st:st+64]).to(device)@rt[n].T).cpu().numpy() for n in v3.NAMES}; nloc=len(next(iter(sc.values())))
   for j,q in enumerate(qids[st:st+nloc]):
    S=np.stack([sc[n][j] for n in v3.NAMES]).astype(np.float32); ranks_full=np.stack([v3.full_ranks(S[e]) for e in range(4)],axis=1); union=np.asarray(sorted(set().union(*(set(map(int,v3.top_rows(S[e],v3.MAX_POOL))) for e in range(4)))),dtype=np.int32); uranks=ranks_full[union].astype(np.int16); feat=v3.feature_matrix(S,union,ranks_full)
    pos=np.asarray(sorted({ridx[r] for r in positives[q]}),dtype=np.int32); pset=set(map(int,pos)); y=np.asarray([int(int(r) in pset) for r in union],dtype=np.uint8); br=ranks_full[pos,0].astype(np.int32)
    X.append(feat); rows_all.append(union); ranks_all.append(uranks); labels.append(y); offsets.append(offsets[-1]+len(union)); pos_rows.append(pos); pos_base_ranks.append(br); pos_offsets.append(pos_offsets[-1]+len(pos)); baseline.append({'query_id':q,**v3.evaluate_full_candidate_ranks(br,v3.N_CANDIDATES)}); audits.append({'query_id':q,'union_size':len(union),'positive_count':len(pos),'positive_in_union':int(y.sum()),'baseline_best_rank':int(br.min())})
 out=OUT/'prepared'/f'fold{fold}'; out.mkdir(parents=True,exist_ok=True); np.save(out/'X.npy',np.concatenate(X)); np.save(out/'rows.npy',np.concatenate(rows_all)); np.save(out/'ranks.npy',np.concatenate(ranks_all)); np.save(out/'labels.npy',np.concatenate(labels)); np.save(out/'offsets.npy',np.asarray(offsets,dtype=np.int64)); np.save(out/'positive_rows.npy',np.concatenate(pos_rows)); np.save(out/'positive_base_ranks.npy',np.concatenate(pos_base_ranks)); np.save(out/'positive_offsets.npy',np.asarray(pos_offsets,dtype=np.int64)); pd.DataFrame({'query_id':qids}).to_csv(out/'queries.csv',index=False); pd.DataFrame(baseline).to_csv(out/'baseline_query_metrics.csv',index=False); pd.DataFrame(audits).to_csv(out/'audit.csv',index=False); (out/'candidate_reactions.txt').write_text('\n'.join(common)+'\n'); (out/'feature_names.json').write_text(json.dumps(v3.FEATURE_NAMES,indent=2)+'\n')
 summary={'fold':fold,'queries':len(qids),'candidate_count':len(common),'rows':int(offsets[-1]),'mean_union_size':float(np.mean([a['union_size'] for a in audits])),'query_positive_in_union_fraction':float(np.mean([a['positive_in_union']>0 for a in audits])),'dev_pairs':len(dev),'split_salt':split['split_salt'],'folds':int(split['folds']),'external_metrics_used':False}; (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n'); print(json.dumps(summary,indent=2))

def gate(delta:dict[str,float],tol:float=-1e-12)->dict:
 checks={k:delta[k]>=tol for k in v3.METRICS}; material=delta['mrr']>=.003 or delta['map']>=.003 or delta['hit10']>=.01; return {'checks':checks,'material_gain':bool(material),'pass':bool(all(checks.values()) and material)}
def evaluate_confirmation():
 p=protocol(); selected=assert_frozen_selection(p); fold=int(p['confirmation_split']['dev_fold']); meta=json.loads((OUT/'final_ranker_meta.json').read_text()); assert meta['selected_config']==selected and meta['confirmation_labels_used'] is False; assert sha(OUT/'final_ranker.json')==meta['checkpoint_sha256']
 old=v3.OUT; v3.OUT=OUT
 try: cache=v3.load_cache(fold)
 finally: v3.OUT=old
 booster=xgb.Booster(); booster.load_model(OUT/'final_ranker.json'); pred=booster.predict(xgb.DMatrix(np.asarray(cache['X'],dtype=np.float32))); cand=v3.candidate_query_metrics(cache,pred,selected); base=cache['baseline']; assert cand.query_id.tolist()==base.query_id.tolist(); cm=v3.metric_map(cand); bm=v3.metric_map(base); delta={k:cm[k]-bm[k] for k in v3.METRICS}; decision=gate(delta,float(p['confirmation_gate']['numeric_tolerance'])); cand.to_csv(OUT/'confirmation_query_metrics.csv',index=False)
 result={'status':'passed_confirmation' if decision['pass'] else 'rejected_confirmation_no_retuning','selected_config':selected,'baseline':bm,'candidate':cm,'delta':delta,'gate':decision,'query_count':len(cand),'candidate_count':v3.N_CANDIDATES,'split':p['confirmation_split'],'final_ranker_sha256':meta['checkpoint_sha256'],'external_metrics_used':False,'same_confirmation_retuning_allowed':False}; (OUT/'confirmation_result.json').write_text(json.dumps(result,indent=2)+'\n'); print(json.dumps(result,indent=2))

def main():
 ap=argparse.ArgumentParser(); ap.add_argument('stage',choices=['train-ranker','prepare','evaluate']); a=ap.parse_args()
 if a.stage=='train-ranker': train_final_ranker()
 elif a.stage=='prepare': prepare_confirmation()
 else: evaluate_confirmation()
if __name__=='__main__': main()
