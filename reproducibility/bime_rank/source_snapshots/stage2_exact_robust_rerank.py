from __future__ import annotations
from pathlib import Path
import sys, json, math, glob
import numpy as np, pandas as pd, torch
ROOT=Path('/home/s241850073/igem2026');sys.path.insert(0,str(ROOT))
OUT=ROOT/'results/requested_r2e20_bime_v2_20260906'; OLD=ROOT/'results/requested_r2e20_nju_lab_final_v2_20260813'; LAB=ROOT/'local_candidate_libraries/nju_lab_gbk_20260812/candidates'
from projects.active.terpene_screening.rank_open_world import load_feature_schema,load_models,encode_reaction_with_audit,ensemble_similarity

def rank_score(v, higher=True):
    v=np.asarray(v,float); order=np.argsort(-v if higher else v,kind='stable'); rank=np.empty(len(v),int); rank[order]=np.arange(1,len(v)+1); n=max(len(v),2)
    # smooth log-rank score: top ranks matter, but a single rank jump cannot dominate.
    score=1-np.log(rank)/np.log(n)
    return rank,score.astype(np.float32)

def normalize_rows(x):
    x=np.asarray(x,dtype=np.float32); n=np.linalg.norm(x,axis=1,keepdims=True); n[n==0]=1; return x/n

def choose_diverse(df, emb, row_by_id, n=3):
    chosen=[]
    for _,r in df.iterrows():
        if not chosen: chosen.append(r); continue
        rv=emb[row_by_id[r.candidate_id]]
        sims=[float(rv@emb[row_by_id[x.candidate_id]]) for x in chosen]
        threshold=0.95 if len(chosen)==1 else 0.92
        if max(sims)<threshold: chosen.append(r)
        if len(chosen)>=n: break
    if len(chosen)<n:
        for _,r in df.iterrows():
            if r.candidate_id not in {x.candidate_id for x in chosen}: chosen.append(r)
            if len(chosen)>=n:break
    return chosen

# authoritative 530 order + exact protein features
se=pd.read_csv(OUT/'enzgfm_stage2_530/entries.csv',dtype=str).sort_values('row'); stage_ids=se.Entry.astype(str).tolist(); sidx={x:i for i,x in enumerate(stage_ids)}
enz=normalize_rows(np.asarray(np.load(OUT/'enzgfm_stage2_530/embeddings.npy',mmap_mode='r'),dtype=np.float32))
le=pd.read_csv(LAB/'esmc600m/entries.csv',dtype=str).sort_values('row'); lidx={x:i for i,x in enumerate(le.Entry.astype(str))}; lab_es=np.load(LAB/'esmc600m/embeddings.npy',mmap_mode='r'); es=np.asarray(lab_es[[lidx[x] for x in stage_ids]],dtype=np.float32)
# exact prok seed EnzGFM map
sme=pd.read_csv(OUT/'enzgfm_prok_seeds/entries.csv',dtype=str).sort_values('row'); smap=pd.read_csv(OUT/'prok_seed_map.csv',dtype=str).fillna(''); seedE=normalize_rows(np.asarray(np.load(OUT/'enzgfm_prok_seeds/embeddings.npy',mmap_mode='r'),dtype=np.float32)); seedrow={x:i for i,x in enumerate(sme.Entry.astype(str))}; hash_to_seedrow={h:seedrow[pid] for pid,h in smap[['protein_id','sequence_sha256']].itertuples(index=False)}
kp=pd.read_csv(OLD/'known_positive_sequence_audit.csv',dtype=str).fillna('')
base=pd.read_csv(OLD/'top10_predictions_20_rows_enriched.csv',dtype=str).fillna(''); targets=base.groupby('requested_group',sort=False).first().reset_index()
# load exact source models once
p_bundle=ROOT/'results/catalyst_clean_mainline_v1/r2e_center_bounded_cap0p1'; s_bundle=ROOT/'results/catalyst_clean_mainline_v1/r2e_enzgfm_center_router_v1';dev=torch.device('cuda')
p_schema=load_feature_schema(p_bundle); s_schema=load_feature_schema(s_bundle); p_models=load_models(p_bundle/'models','production',dev); s_models=load_models(s_bundle/'models','production',dev)
# score matrices across 17 queries x 530 candidates
P=[];S=[]; audits=[]
for _,t in targets.iterrows():
    smi=str(t.reaction_smiles_from_sheet); qp,ap=encode_reaction_with_audit(smi,p_schema,failure_policy='warn'); qs,ass=encode_reaction_with_audit(smi,s_schema,failure_policy='warn')
    P.append(ensemble_similarity(p_models,es,qp[None,:],dev)[0]);S.append(ensemble_similarity(s_models,enz,qs[None,:],dev)[0]);audits.append({'group':t.requested_group,'primary':ap.__dict__,'secondary':ass.__dict__})
P=np.stack(P);S=np.stack(S);np.save(OUT/'stage2_primary_direct_scores.npy',P);np.save(OUT/'stage2_secondary_direct_scores.npy',S);(OUT/'stage2_reaction_feature_audits.json').write_text(json.dumps(audits,indent=2,default=str))
# per-target robust rank aggregation in its stage1 Top50 only
rng=np.random.default_rng(20260906); allsel=[]; summaries=[]; details=[]
for qi,t in targets.iterrows():
    g=str(t.requested_group); d=pd.read_csv(OUT/f'lab_rank_{g}.csv',dtype=str).fillna('').copy(); ids=d.candidate_id.astype(str).tolist(); rows=np.array([sidx[x] for x in ids]);
    # numeric stage1 and exact direct signals
    practical=pd.to_numeric(d.practical_score).to_numpy(float); pr_rank,pr_sc=rank_score(practical)
    p_raw=P[qi,rows];p_rank,p_sc=rank_score(p_raw); s_raw=S[qi,rows];s_rank,s_sc=rank_score(s_raw)
    signals=[('stage1_practical',pr_sc),('esmc_direct',p_sc),('enzgfm_direct',s_sc)]
    d['stage1_rank50']=pr_rank;d['esmc_direct_score']=p_raw;d['esmc_direct_rank50']=p_rank;d['enzgfm_direct_score']=s_raw;d['enzgfm_direct_rank50']=s_rank
    # exact enzyme-specific known-positive similarity
    pg=kp[(kp.requested_group==g)&(kp.taxonomy_scope=='prokaryote')&kp.sequence.ne('')].drop_duplicates('sequence_sha256');
    if len(pg):
        sr=[hash_to_seedrow[h] for h in pg.sequence_sha256.astype(str)]; raw=(enz[rows]@seedE[sr].T).max(axis=1);r,sc=rank_score(raw);signals.append(('enzgfm_seed',sc));d['enzgfm_seed_similarity']=raw;d['enzgfm_seed_rank50']=r
    else: d['enzgfm_seed_similarity']=np.nan;d['enzgfm_seed_rank50']=np.nan
    # stage1 CLIP transfer is already a lab-side structure-anchor similarity and is admitted only where available
    clip=pd.to_numeric(d.clip_transfer_score,errors='coerce') if 'clip_transfer_score' in d else pd.Series(np.nan,index=d.index)
    if clip.notna().any():
        raw=clip.fillna(clip.min()-1e-6).to_numpy(float);r,sc=rank_score(raw);signals.append(('clip_transfer',sc));d['clip_rank50']=r
    else:d['clip_rank50']=np.nan
    X=np.column_stack([x for _,x in signals]).astype(np.float32);m=X.shape[1]
    # Equal-center lognormal perturbations: no single fixed weight is privileged; extreme one-expert domination is suppressed.
    W=np.exp(rng.normal(0,0.55,size=(30000,m))).astype(np.float32);W/=W.sum(axis=1,keepdims=True);scores=X@W.T
    top1=np.argmax(scores,axis=0); top3=np.argpartition(-scores,kth=min(2,len(d)-1),axis=0)[:3,:]
    t1=np.bincount(top1,minlength=len(d))/scores.shape[1];t3=np.bincount(top3.ravel(),minlength=len(d))/scores.shape[1]
    mean=scores.mean(axis=1); d['weight_robust_top1']=t1;d['weight_robust_top3']=t3;d['robust_mean_rank_score']=mean;d['stage2_signal_count']=m;d['stage2_signal_names']=';'.join(n for n,_ in signals)
    # rank: avoid tier C; prioritize robust top3, then top1 and mean consensus. Small stage1 feasibility remains embedded in stage1 signal.
    safe=d[d.feasibility_tier.ne('C')].copy();safe=safe.sort_values(['weight_robust_top3','weight_robust_top1','robust_mean_rank_score','practical_score'],ascending=[False,False,False,False])
    chosen=choose_diverse(safe,enz,sidx,3)
    d['stage2_selected_role']='';
    for role,r in zip(['PRIMARY','BACKUP_A','BACKUP_B'],chosen):d.loc[d.candidate_id.eq(r.candidate_id),'stage2_selected_role']=role
    d.to_csv(OUT/f'stage2_rank_{g}.csv',index=False)
    chosen_ids=[x.candidate_id for x in chosen];sel=d[d.candidate_id.isin(chosen_ids)].copy();ordm={x:i for i,x in enumerate(chosen_ids)};sel['role_order']=sel.candidate_id.map(ordm);sel=sel.sort_values('role_order');allsel.append(sel)
    summaries.append({'requested_group':g,'product':t['product'],'signals':';'.join(n for n,_ in signals),'primary':chosen[0].candidate_id,'backup_a':chosen[1].candidate_id,'backup_b':chosen[2].candidate_id,'primary_top1_robustness':float(chosen[0].weight_robust_top1),'primary_top3_robustness':float(chosen[0].weight_robust_top3),'primary_stage1_rank':int(chosen[0].stage1_rank50),'primary_esmc_rank':int(chosen[0].esmc_direct_rank50),'primary_enzgfm_rank':int(chosen[0].enzgfm_direct_rank50),'primary_seed_rank':None if pd.isna(chosen[0].enzgfm_seed_rank50) else int(chosen[0].enzgfm_seed_rank50),'primary_clip_rank':None if pd.isna(chosen[0].clip_rank50) else int(chosen[0].clip_rank50),'primary_feasibility':chosen[0].feasibility_tier,'primary_source':chosen[0].source_files,'primary_locus':chosen[0].locus_tags})
    print(g,[(x.candidate_id,round(float(x.weight_robust_top1),3),round(float(x.weight_robust_top3),3),int(x.stage1_rank50),int(x.esmc_direct_rank50),int(x.enzgfm_direct_rank50),None if pd.isna(x.enzgfm_seed_rank50) else int(x.enzgfm_seed_rank50),x.feasibility_tier) for x in chosen],flush=True)
sel=pd.concat(allsel,ignore_index=True);sel.to_csv(OUT/'stage2_recommended_candidates_17x3.csv',index=False);pd.DataFrame(summaries).to_csv(OUT/'stage2_reaction_recommendations_17.csv',index=False)
# cross-reaction specificity audit: rank each selected candidate under all 17 exact source scores.
for qi,t in targets.iterrows():
    _,rp=rank_score(P[qi]);_,rs=rank_score(S[qi]);details.append((str(t.requested_group),rp,rs))
# selected cross-target rank summary
rows=[]
for _,r in sel.iterrows():
    cid=r.candidate_id; ci=sidx[cid]; target_g=r.requested_group; target_i=int(targets.index[targets.requested_group.eq(target_g)][0]);
    p_ranks=[];s_ranks=[]
    for g,rp,rs in details:p_ranks.append((g,int(rp[ci])));s_ranks.append((g,int(rs[ci])))
    tp=dict(p_ranks)[target_g];ts=dict(s_ranks)[target_g];other_p=sorted(v for g,v in p_ranks if g!=target_g);other_s=sorted(v for g,v in s_ranks if g!=target_g)
    rows.append({'requested_group':target_g,'candidate_id':cid,'role':r.stage2_selected_role,'target_primary_rank530':tp,'target_secondary_rank530':ts,'best_other_primary_rank530':other_p[0],'best_other_secondary_rank530':other_s[0],'target_primary_advantage_vs_best_other':other_p[0]-tp,'target_secondary_advantage_vs_best_other':other_s[0]-ts,'top20_other_primary_count':sum(v<=20 for v in other_p),'top20_other_secondary_count':sum(v<=20 for v in other_s)})
pd.DataFrame(rows).to_csv(OUT/'stage2_cross_reaction_specificity.csv',index=False)
# map back to 20 rows
map17=pd.DataFrame(summaries).set_index('requested_group');out=[]
for _,r in base[['number','substrate','product','requested_group']].drop_duplicates('number').sort_values('number',key=lambda s:pd.to_numeric(s)).iterrows():
    s=map17.loc[r.requested_group];out.append({'number':r.number,'substrate':r.substrate,'product':r['product'],'requested_group':r.requested_group,'primary_candidate_id':s.primary,'backup_a_candidate_id':s.backup_a,'backup_b_candidate_id':s.backup_b,'primary_top1_robustness':s.primary_top1_robustness,'primary_top3_robustness':s.primary_top3_robustness,'primary_source':s.primary_source,'primary_locus':s.primary_locus})
pd.DataFrame(out).to_csv(OUT/'stage2_experiment_plan_20_rows.csv',index=False)
(OUT/'stage2_summary.json').write_text(json.dumps({'status':'complete','candidate_cascade':'131532 -> stage1 top50/reaction -> 530 unique exact EnzGFM -> exact frozen ESM-C/EnzGFM source scoring -> robustness rank aggregation','unique_stage2_proteins':len(stage_ids),'weight_samples_per_reaction':30000,'weight_sampling':'equal-center independent lognormal sigma=0.55, normalized; robustness is ranking stability, not activity probability','selection':'exclude feasibility C from preferred set; maximize top3 robustness, then top1 robustness, mean rank score; enforce EnzGFM embedding diversity for backups'},indent=2)+'\n')
