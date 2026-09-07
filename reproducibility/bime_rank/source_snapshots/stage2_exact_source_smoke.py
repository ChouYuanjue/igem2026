from pathlib import Path
import sys,pandas as pd,numpy as np,torch
ROOT=Path('/home/s241850073/igem2026');sys.path.insert(0,str(ROOT))
from projects.active.terpene_screening.rank_open_world import load_feature_schema,load_models,encode_reaction_with_audit,ensemble_similarity
OUT=ROOT/'results/requested_r2e20_bime_v2_20260906'; LAB=ROOT/'local_candidate_libraries/nju_lab_gbk_20260812/candidates'
se=pd.read_csv(OUT/'enzgfm_stage2_530/entries.csv',dtype=str).sort_values('row'); ids=se.Entry.astype(str).tolist()
# align ESM-C
le=pd.read_csv(LAB/'esmc600m/entries.csv',dtype=str).sort_values('row'); lidx={x:i for i,x in enumerate(le.Entry.astype(str))}; base=np.load(LAB/'esmc600m/embeddings.npy',mmap_mode='r'); es=np.asarray(base[[lidx[x] for x in ids]],dtype=np.float32)
enz=np.asarray(np.load(OUT/'enzgfm_stage2_530/embeddings.npy',mmap_mode='r'),dtype=np.float32)
old=pd.read_csv(ROOT/'results/requested_r2e20_nju_lab_final_v2_20260813/top10_predictions_20_rows_enriched.csv',dtype=str).fillna(''); smi=old[old.requested_group.eq('delta_cadinene')].reaction_smiles_from_sheet.iloc[0]
for name,bundle,prot in [('primary',ROOT/'results/catalyst_clean_mainline_v1/r2e_center_bounded_cap0p1',es),('secondary',ROOT/'results/catalyst_clean_mainline_v1/r2e_enzgfm_center_router_v1',enz)]:
 schema=load_feature_schema(bundle); q,a=encode_reaction_with_audit(smi,schema,failure_policy='warn'); models=load_models(bundle/'models','production',torch.device('cuda')); s=ensemble_similarity(models,prot,q[None,:],torch.device('cuda'))[0]; order=np.argsort(-s)[:10];print(name,'qdim',len(q),'pdim',prot.shape[1],'top',[(ids[i],float(s[i])) for i in order])
