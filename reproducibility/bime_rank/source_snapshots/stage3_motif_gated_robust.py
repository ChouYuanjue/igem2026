from pathlib import Path
import glob,json
import numpy as np,pandas as pd
OUT=Path('results/requested_r2e20_bime_v2_20260906')
E=np.load(OUT/'enzgfm_stage2_530/embeddings.npy',mmap_mode='r'); ids=pd.read_csv(OUT/'enzgfm_stage2_530/entries.csv',dtype=str).sort_values('row').Entry.astype(str).tolist();idx={x:i for i,x in enumerate(ids)}
E=np.asarray(E,dtype=np.float32).copy();E/=np.maximum(np.linalg.norm(E,axis=1,keepdims=True),1e-8)
rng=np.random.default_rng(20260906)
def rankscore(v):
 v=np.asarray(v,float);o=np.argsort(-v,kind='stable');r=np.empty(len(v),int);r[o]=np.arange(1,len(v)+1);n=max(len(v),2);return r,(1-np.log(r)/np.log(n)).astype(np.float32)
def diverse(df,n=3):
 out=[]
 for _,r in df.iterrows():
  if not out:out.append(r);continue
  sims=[float(E[idx[r.candidate_id]]@E[idx[x.candidate_id]]) for x in out]
  if max(sims)<(0.95 if len(out)==1 else 0.92):out.append(r)
  if len(out)>=n:break
 for _,r in df.iterrows():
  if len(out)>=n:break
  if r.candidate_id not in {x.candidate_id for x in out}:out.append(r)
 return out
summ=[];sels=[]
for p in sorted(glob.glob(str(OUT/'stage2_rank_*.csv'))):
 d=pd.read_csv(p,dtype=str).fillna('');g=d.requested_group.iloc[0]
 # mechanism gate: canonical class-I motif + no tier-C severe risk
 m=d[d.classI_motif.astype(str).str.lower().eq('true') & d.feasibility_tier.ne('C')].copy()
 if len(m)<3: raise RuntimeError(f'{g}: only {len(m)} motif-gated candidates')
 sig=[]
 for name,col in [('stage1','practical_score'),('esmc_direct','esmc_direct_score'),('enzgfm_direct','enzgfm_direct_score')]:
  raw=pd.to_numeric(m[col],errors='coerce').to_numpy(float);r,s=rankscore(raw);m[f'gated_{name}_rank']=r;sig.append((name,s))
 if pd.to_numeric(m.get('enzgfm_seed_similarity'),errors='coerce').notna().any():
  raw=pd.to_numeric(m.enzgfm_seed_similarity,errors='coerce').fillna(-1).to_numpy(float);r,s=rankscore(raw);m['gated_seed_rank']=r;sig.append(('enzgfm_seed',s))
 else:m['gated_seed_rank']=np.nan
 if pd.to_numeric(m.get('clip_transfer_score'),errors='coerce').notna().any():
  c=pd.to_numeric(m.clip_transfer_score,errors='coerce');raw=c.fillna(c.min()-1e-6).to_numpy(float);r,s=rankscore(raw);m['gated_clip_rank']=r;sig.append(('clip_transfer',s))
 else:m['gated_clip_rank']=np.nan
 X=np.column_stack([s for _,s in sig]); W=np.exp(rng.normal(0,.55,size=(50000,len(sig)))).astype(np.float32);W/=W.sum(1,keepdims=True);Z=X@W.T
 top1=np.argmax(Z,axis=0);k=min(3,len(m));topk=np.argpartition(-Z,kth=k-1,axis=0)[:k]
 m['gated_top1_robustness']=np.bincount(top1,minlength=len(m))/Z.shape[1];m['gated_top3_robustness']=np.bincount(topk.ravel(),minlength=len(m))/Z.shape[1];m['gated_mean_consensus']=Z.mean(1);m['gated_signal_names']=';'.join(n for n,_ in sig)
 m=m.sort_values(['gated_top3_robustness','gated_top1_robustness','gated_mean_consensus'],ascending=False).reset_index(drop=True);chosen=diverse(m)
 m['final_role']='';
 for role,r in zip(['PRIMARY','BACKUP_A','BACKUP_B'],chosen):m.loc[m.candidate_id.eq(r.candidate_id),'final_role']=role
 m.to_csv(OUT/f'final_gated_rank_{g}.csv',index=False)
 sel=m[m.final_role.ne('')].copy();omap={r.candidate_id:i for i,r in enumerate(chosen)};sel['role_order']=sel.candidate_id.map(omap);sel=sel.sort_values('role_order');sels.append(sel)
 r=chosen[0];summ.append({'requested_group':g,'product':r['product'],'eligible_classI_count':len(m),'signals':';'.join(n for n,_ in sig),'primary':r.candidate_id,'backup_a':chosen[1].candidate_id,'backup_b':chosen[2].candidate_id,'primary_top1_robustness':r.gated_top1_robustness,'primary_top3_robustness':r.gated_top3_robustness,'stage1_rank':r.gated_stage1_rank,'esmc_direct_rank':r.gated_esmc_direct_rank,'enzgfm_direct_rank':r.gated_enzgfm_direct_rank,'seed_rank':r.gated_seed_rank,'clip_rank':r.gated_clip_rank,'feasibility':r.feasibility_tier,'length':r.length,'instability':r.instability_index,'source_files':r.source_files,'locus_tags':r.locus_tags,'screening_sources':r.screening_sources,'mmseqs_ref_id':r.mmseqs_ref_id,'mmseqs_fident':r.mmseqs_fident})
 print(g,[(x.candidate_id,round(float(x.gated_top1_robustness),3),round(float(x.gated_top3_robustness),3),int(x.gated_stage1_rank),int(x.gated_esmc_direct_rank),int(x.gated_enzgfm_direct_rank),None if pd.isna(x.gated_seed_rank) else int(x.gated_seed_rank),None if pd.isna(x.gated_clip_rank) else int(x.gated_clip_rank),x.feasibility_tier) for x in chosen])
S=pd.DataFrame(summ);S.to_csv(OUT/'final_recommendations_17.csv',index=False);pd.concat(sels,ignore_index=True).to_csv(OUT/'final_candidates_17x3.csv',index=False)
# map to 20 rows
base=pd.read_csv('results/requested_r2e20_nju_lab_final_v2_20260813/top10_predictions_20_rows_enriched.csv',dtype=str).fillna('');mp=S.set_index('requested_group');rows=[]
for _,r in base[['number','substrate','product','requested_group']].drop_duplicates('number').sort_values('number',key=lambda s:pd.to_numeric(s)).iterrows():
 x=mp.loc[r.requested_group];rows.append({'number':r.number,'substrate':r.substrate,'product':r['product'],'requested_group':r.requested_group,'primary_candidate_id':x.primary,'backup_a_candidate_id':x.backup_a,'backup_b_candidate_id':x.backup_b,'primary_top1_robustness':x.primary_top1_robustness,'primary_top3_robustness':x.primary_top3_robustness,'primary_source_files':x.source_files,'primary_locus_tags':x.locus_tags,'primary_feasibility':x.feasibility})
pd.DataFrame(rows).to_csv(OUT/'final_experiment_plan_20_rows.csv',index=False)
(OUT/'final_strategy.json').write_text(json.dumps({'status':'complete','pipeline':['131532 NJU_LAB prokaryotic candidates','BiME teacher + seed/CLIP transfer + feasibility -> top50 per reaction','530 unique candidates exact EnzGFM-650M','exact frozen ESM-C and EnzGFM reaction-specific source scores','hard class-I TPS DDxxD+NSE/DTE motif gate and severe-risk exclusion','50000 lognormal weight perturbations over available rank signals','diversity-aware 1 primary + 2 backups'],'robustness_interpretation':'ranking stability under plausible expert weighting, not activity probability'},indent=2)+'\n')
