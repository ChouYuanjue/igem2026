from __future__ import annotations
import json,math,sys
from pathlib import Path
import numpy as np,pandas as pd,torch
from Bio.SeqUtils.ProtParam import ProteinAnalysis
ROOT=Path('/home/s241850073/igem2026');OUT=ROOT/'results/requested_r2e20_bime_v2_20260906';OLD=ROOT/'results/requested_r2e20_nju_lab_final_v2_20260813';LAB=ROOT/'local_candidate_libraries/nju_lab_gbk_20260812/candidates'
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from projects.active.terpene_screening.rank_open_world import encode_external_enzymes_with_audit

SEED_W=0.25; OLD_W=0.12; CLIP_W=0.13
KD={'A':1.8,'R':-4.5,'N':-3.5,'D':-3.5,'C':2.5,'Q':-3.5,'E':-3.5,'G':-0.4,'H':-3.2,'I':4.5,'L':3.8,'K':-3.9,'M':1.9,'F':2.8,'P':-1.6,'S':-0.8,'T':-0.7,'W':-0.9,'Y':-1.3,'V':4.2}
def norm(x):
 x=np.asarray(x,dtype=np.float32);n=np.linalg.norm(x,axis=1,keepdims=True);n[n==0]=1;return x/n
def log_rank_score(v,descending=True):
 v=np.asarray(v,float); order=np.argsort(-v if descending else v,kind='stable');rank=np.empty(len(v),dtype=np.int64);rank[order]=np.arange(1,len(v)+1);return (1-np.log(rank)/math.log(len(v))).astype(np.float32),rank
def maxkd(seq,w=19):
 vals=np.asarray([KD.get(a,0.) for a in seq],float)
 if len(vals)<w:return float(vals.mean()) if len(vals) else 0.
 return float(np.convolve(vals,np.ones(w)/w,mode='valid').max())
def feasibility(seq):
 L=len(seq);pa=ProteinAnalysis(seq);inst=float(pa.instability_index());gravy=float(pa.gravy());mx=maxkd(seq);nt=maxkd(seq[:60]);severe=[];moderate=[]
 if L<200 or L>750: severe.append('length_outside_200_750')
 elif L<250 or L>650: moderate.append('length_outside_250_650')
 if mx>=2.0: severe.append('strong_hydrophobic_segment')
 elif mx>=1.6: moderate.append('hydrophobic_segment')
 if nt>=1.8: severe.append('nterm_hydrophobicity')
 if inst>55: moderate.append('instability_gt_55')
 tier='C' if severe else ('B' if moderate else 'A');score={'A':1.0,'B':0.72,'C':0.25}[tier]
 return tier,score,L,inst,gravy,mx,nt,';'.join(severe),';'.join(moderate)
def choose_diverse(frame,lab_norm,idx,n=3):
 chosen=[]
 for _,r in frame.iterrows():
  ri=idx[r.candidate_id]
  if not chosen:chosen.append(r);continue
  sims=[float(lab_norm[ri]@lab_norm[idx[x.candidate_id]]) for x in chosen]
  thresh=.95 if len(chosen)==1 else .92
  if max(sims)<thresh: chosen.append(r)
  if len(chosen)>=n:break
 if len(chosen)<n:
  for _,r in frame.iterrows():
   if r.candidate_id not in {x.candidate_id for x in chosen}:chosen.append(r)
   if len(chosen)>=n:break
 return chosen

# assets
labmeta_raw=pd.read_csv(LAB/'candidates_metadata.tsv.gz',sep='\t',dtype=str).fillna('').drop_duplicates('enzyme_id')
le=pd.read_csv(LAB/'esmc600m/entries.csv',dtype=str).sort_values('row');lab_ids=le.Entry.astype(str).tolist();idx={x:i for i,x in enumerate(lab_ids)}
labmeta=labmeta_raw.set_index('enzyme_id').reindex(lab_ids).reset_index();assert labmeta.enzyme_id.astype(str).tolist()==lab_ids and labmeta.sequence.ne('').all()
lab_norm=norm(np.load(LAB/'esmc600m/embeddings.npy',mmap_mode='r'));assert len(lab_norm)==len(lab_ids)
ge=pd.read_csv(ROOT/'data/catalyst_candidate_universes/general_merged/proteins/entries.csv',dtype=str).sort_values('row');gen_ids=ge.Entry.astype(str).tolist();gidx={x:i for i,x in enumerate(gen_ids)};gen_emb=np.load(ROOT/'data/catalyst_candidate_universes/general_merged/proteins/embeddings.npy',mmap_mode='r')
old=pd.read_csv(OLD/'unique_reaction_rankings.csv',dtype=str).fillna('');old['rank']=pd.to_numeric(old['rank']); oldmap={(g,c):int(r) for g,c,r in old[['requested_group','candidate_id','rank']].itertuples(index=False)}
rs=pd.read_csv(OLD/'reaction_summary.csv',dtype=str).fillna('').drop_duplicates('requested_group');rinfo=rs.set_index('requested_group').to_dict('index')
enr=pd.read_csv(OLD/'top10_predictions_20_rows_enriched.csv',dtype=str).fillna('');target=enr.groupby('requested_group',sort=False).first().reset_index()
kp=pd.read_csv(OLD/'known_positive_sequence_audit.csv',dtype=str).fillna('')
# encode all unique prokaryotic positive sequences exactly once; small enough that this is cheap.
pro=kp[(kp.taxonomy_scope=='prokaryote')&kp.sequence.ne('')].drop_duplicates('sequence_sha256').copy();seed_vec,_=encode_external_enzymes_with_audit(pro.rename(columns={'known_positive_id':'enzyme_id'})[['enzyme_id','sequence']],device='cuda',model_name='esmc_600m');seed_by_hash={h:seed_vec[i] for i,h in enumerate(pro.sequence_sha256.astype(str))}
torch.cuda.empty_cache()
# CLIP structural anchors
cp_e=pd.read_csv(ROOT/'results/bime_rank_unified_v1/clipzyme_r2e_candidate_asset_v1/entries.csv',dtype=str).sort_values('row');cp_ids=cp_e.protein_id.astype(str).tolist();cp=np.load(ROOT/'results/bime_rank_unified_v1/clipzyme_r2e_candidate_asset_v1/embeddings.npy',mmap_mode='r');cpn=norm(cp)
cr_e=pd.read_csv(ROOT/'results/clipzyme_native_extension_v1/full_hplus_candidate_reactions/clipzyme_embeddings_gpu_v1/entries.csv',dtype=str).fillna('');cr=np.load(ROOT/'results/clipzyme_native_extension_v1/full_hplus_candidate_reactions/clipzyme_embeddings_gpu_v1/embeddings.npy',mmap_mode='r');cridx={rid:int(row) for rid,row,sup in cr_e[['reaction_id','row','clipzyme_supported']].itertuples(index=False) if str(sup).lower()=='true'}
# experimental touch optional metadata
try:touch=pd.read_csv(LAB/'experimental_touch_v2.tsv.gz',sep='\t',dtype=str).fillna('').drop_duplicates('candidate_id').set_index('candidate_id')
except Exception:touch=None
allrows=[]; summaries=[]
for ti,t in target.iterrows():
 g=str(t.requested_group); teacher=pd.read_csv(OUT/'teacher'/f'{g}.csv',dtype=str).fillna('').sort_values('rank'); aids=[x for x in teacher.candidate_id.astype(str).head(100) if x in gidx]
 A=norm(np.asarray(gen_emb[[gidx[x] for x in aids]],dtype=np.float32)); Lt=torch.as_tensor(lab_norm,dtype=torch.float32,device='cuda'); At=torch.as_tensor(A[:20],dtype=torch.float32,device='cuda')
 with torch.no_grad(): sim=(Lt@At.T);tmax=sim.max(1).values.cpu().numpy();t3=sim.topk(k=min(3,sim.shape[1]),dim=1).values.mean(1).cpu().numpy()
 ts1,_=log_rank_score(tmax);ts3,_=log_rank_score(t3);teacher_score=.7*ts1+.3*ts3
 # all prokaryotic known-positive sequences, not only public IDs accepted by teacher
 pg=kp[(kp.requested_group==g)&(kp.taxonomy_scope=='prokaryote')&kp.sequence.ne('')].drop_duplicates('sequence_sha256'); seed_score=None
 if len(pg):
  S=np.stack([seed_by_hash[h] for h in pg.sequence_sha256.astype(str)]).astype(np.float32);St=torch.as_tensor(S,dtype=torch.float32,device='cuda')
  with torch.no_grad(): sraw=(Lt@St.T).max(1).values.cpu().numpy()
  seed_score,_=log_rank_score(sraw)
 # CLIP structural teacher only for registered Rhea targets with supported reaction embedding
 rhea=str(rinfo[g]['query_ref']) if str(rinfo[g]['query_type'])=='current_rhea' else '';clip_score=None;clip_supported=False
 if rhea in cridx:
  q=np.asarray(cr[cridx[rhea]],dtype=np.float32).copy();q/=max(np.linalg.norm(q),1e-8); cs=cpn@q; top=np.argsort(-cs,kind='stable')[:20]; canch=[cp_ids[i] for i in top if cp_ids[i] in gidx]
  if canch:
   C=norm(np.asarray(gen_emb[[gidx[x] for x in canch]],dtype=np.float32));Ct=torch.as_tensor(C,dtype=torch.float32,device='cuda')
   with torch.no_grad(): craw=(Lt@Ct.T).max(1).values.cpu().numpy()
   clip_score,_=log_rank_score(craw);clip_supported=True
 # old wet-lab rank evidence: missing from old top200 = rank N
 oldr=np.asarray([oldmap.get((g,c),len(lab_ids)) for c in lab_ids],dtype=np.int64);oldscore=(1-np.log(oldr)/math.log(len(lab_ids))).clip(0,1).astype(np.float32)
 # availability-aware weighted functional score
 parts=[(.50,teacher_score),(.12,oldscore)]
 if seed_score is not None:parts.append((.25,seed_score))
 if clip_score is not None:parts.append((.13,clip_score))
 sw=sum(w for w,_ in parts);functional=sum(w*x for w,x in parts)/sw
 # preselect top 1200 functionally then add old top50 explicitly
 cand=set(np.argsort(-functional,kind='stable')[:1200].tolist());cand.update(i for i,c in enumerate(lab_ids) if oldmap.get((g,c),999999)<=50)
 records=[]
 for i in cand:
  m=labmeta.iloc[i];tier,fscore,L,inst,gravy,mx,nt,severe,moderate=feasibility(str(m.sequence));motif=1.0 if str(m.classI_motif).lower()=='true' else 0.0;mm=1.0 if 'mmseqs_homolog' in str(m.screening_sources) else 0.0;tpsprior=max(motif,.75*mm);final=.78*float(functional[i])+.16*fscore+.06*tpsprior
  cid=lab_ids[i];rec={'requested_group':g,'sheet_rows':rinfo[g]['sheet_rows'],'number':str(t['number']),'substrate':t['substrate'],'product':t['product'],'candidate_id':cid,'practical_score':final,'functional_score':float(functional[i]),'teacher_score':float(teacher_score[i]),'seed_score':float(seed_score[i]) if seed_score is not None else np.nan,'clip_transfer_score':float(clip_score[i]) if clip_score is not None else np.nan,'old_lab_rank':int(oldr[i]) if oldr[i]<len(lab_ids) else np.nan,'known_prok_seed_count':int(len(pg)),'clip_structural_teacher':clip_supported,'feasibility_tier':tier,'feasibility_score':fscore,'length':L,'instability_index':inst,'gravy':gravy,'max_kd_19aa':mx,'nterm_max_kd_19aa':nt,'severe_risk_flags':severe,'moderate_risk_flags':moderate,'classI_motif':m.classI_motif,'screening_sources':m.screening_sources,'mmseqs_ref_id':m.mmseqs_ref_id,'mmseqs_fident':m.mmseqs_fident,'mmseqs_qcov':m.mmseqs_qcov,'mmseqs_tcov':m.mmseqs_tcov,'source_files':m.source_files,'locus_tags':m.locus_tags,'organisms':m.organisms,'sequence_sha256':m.sequence_sha256,'sequence':m.sequence}
  if touch is not None and cid in touch.index:
   for c in ['experimental_touch_level','experimental_touch_label','uniparc_exact_found','uniparc_id','uniprot_accessions','best_pe_level','pdb_exact_ids','alphafold_exact_ids','pubmed_ids']: rec[c]=touch.at[cid,c] if c in touch.columns else ''
  records.append(rec)
 df=pd.DataFrame(records).sort_values(['practical_score','functional_score'],ascending=False).reset_index(drop=True)
 # Primary should not be tier C if a near-score A/B exists; practical score already penalizes, plus hard prefer A/B within top candidates.
 safe=df[df.feasibility_tier!='C'];ranking=pd.concat([safe,df[df.feasibility_tier=='C']],ignore_index=True)
 chosen=choose_diverse(ranking,lab_norm,idx,3)
 selected_ids={r.candidate_id for r in chosen};df['selected_role']='';
 for role,r in zip(['PRIMARY','BACKUP_A','BACKUP_B'],chosen):df.loc[df.candidate_id.eq(r.candidate_id),'selected_role']=role
 df.head(50).to_csv(OUT/f'lab_rank_{g}.csv',index=False)
 sel=df[df.candidate_id.isin(selected_ids)].copy(); order={r.candidate_id:i for i,r in enumerate(chosen)};sel['role_order']=sel.candidate_id.map(order);sel=sel.sort_values('role_order');allrows.append(sel)
 summaries.append({'requested_group':g,'sheet_rows':rinfo[g]['sheet_rows'],'product':t['product'],'known_prok_seed_count':len(pg),'public_teacher_seed_count':int(teacher.teacher_seed_count.iloc[0]) if 'teacher_seed_count' in teacher else np.nan,'clip_structural_teacher':clip_supported,'primary':chosen[0].candidate_id,'backup_a':chosen[1].candidate_id,'backup_b':chosen[2].candidate_id,'primary_score':float(chosen[0].practical_score),'primary_tier':chosen[0].feasibility_tier,'primary_source':chosen[0].source_files,'primary_locus':chosen[0].locus_tags,'old_primary':old[old.requested_group.eq(g)].sort_values('rank').candidate_id.iloc[0] if (old.requested_group==g).any() else ''})
 print(f'[{ti+1}/{len(target)}] {g}: '+', '.join(f'{r.candidate_id}({r.feasibility_tier},{r.practical_score:.3f})' for r in chosen),flush=True)
 del Lt,At
 torch.cuda.empty_cache()
sel=pd.concat(allrows,ignore_index=True);sel.to_csv(OUT/'recommended_candidates_17x3.csv',index=False);pd.DataFrame(summaries).to_csv(OUT/'reaction_recommendations_17.csv',index=False)
# 20 sheet rows map duplicated reactions to same candidate panels.
rows=[]
for _,r in enr[['number','substrate','product','requested_group']].drop_duplicates('number').sort_values('number',key=lambda s:pd.to_numeric(s)).iterrows():
 s=sel[sel.requested_group.eq(r.requested_group)].sort_values('role_order')
 rec={'number':r.number,'substrate':r.substrate,'product':r['product'],'requested_group':r.requested_group}
 for j,role in enumerate(['PRIMARY','BACKUP_A','BACKUP_B']):
  x=s.iloc[j];rec[f'{role.lower()}_candidate_id']=x.candidate_id;rec[f'{role.lower()}_source_files']=x.source_files;rec[f'{role.lower()}_locus_tags']=x.locus_tags;rec[f'{role.lower()}_feasibility']=x.feasibility_tier;rec[f'{role.lower()}_score']=x.practical_score
 rows.append(rec)
pd.DataFrame(rows).to_csv(OUT/'experiment_plan_20_rows.csv',index=False)
summary={'status':'complete','strategy':'current BiME-Rank v2 full-public-universe teacher -> ESM-C transfer to immutable 131532 NJU_LAB prokaryotic candidate universe; all prokaryotic positive seed sequences; CLIP structural teacher when Rhea-native embedding exists; old lab rank as secondary evidence; wet-lab feasibility penalty; diversity-aware 1+2 shortlist','unique_reactions':int(len(target)),'sheet_rows':20,'lab_candidates':len(lab_ids),'recommended_unique_construct_slots':int(len(target)),'recommended_with_two_backups':int(len(target)*3),'weights':{'teacher':.50,'seed_if_available':.25,'old_lab_rank':.12,'clip_if_available':.13,'functional_fraction_final':.78,'feasibility_fraction_final':.16,'tps_prior_fraction_final':.06},'note':'weights are a transparent practical prioritization heuristic, not calibrated catalytic probabilities; raw score values should not be interpreted as activity probabilities'};(OUT/'summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False)+'\n')
print(json.dumps(summary,indent=2,ensure_ascii=False))
