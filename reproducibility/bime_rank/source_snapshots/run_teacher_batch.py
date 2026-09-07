from __future__ import annotations
import subprocess,sys,json
from pathlib import Path
import pandas as pd
ROOT=Path('/home/s241850073/igem2026');OUT=ROOT/'results/requested_r2e20_bime_v2_20260906';OLD=ROOT/'results/requested_r2e20_nju_lab_final_v2_20260813'
enr=pd.read_csv(OLD/'top10_predictions_20_rows_enriched.csv',dtype=str).fillna('')
targets=enr.groupby('requested_group',sort=False).first().reset_index()[['requested_group','number','substrate','product','reaction_smiles_from_sheet']]
kp=pd.read_csv(OLD/'known_positive_sequence_audit.csv',dtype=str).fillna(''); gen=pd.read_csv(ROOT/'data/catalyst_candidate_universes/general_merged/protein_sequences.tsv',sep='\t',dtype=str).fillna('');gids=set(gen.protein_id.astype(str))
records=[]
for i,row in targets.iterrows():
 g=str(row.requested_group); pro=kp[(kp.requested_group==g)&(kp.taxonomy_scope=='prokaryote')&kp.sequence.ne('')].drop_duplicates('sequence_sha256');seed_ids=[x for x in pro.known_positive_id.astype(str) if x in gids]
 out=OUT/'teacher';out.mkdir(exist_ok=True);csv=out/f'{g}.csv';audit=out/f'{g}.audit.json';log=out/f'{g}.log'
 cmd=[sys.executable,str(ROOT/'projects/active/terpene_screening/rank_open_world.py'),'rank-enzymes','--reaction-smiles',str(row.reaction_smiles_from_sheet),'--top-k','100','--ranking-objective','top10','--protein-dir','data/catalyst_candidate_universes/general_merged/proteins','--registered-protein-dir','data/catalyst_candidate_universes/general_merged/proteins','--output',str(csv),'--audit-output',str(audit)]
 if seed_ids: cmd += ['--known-enzyme-ids',*seed_ids]
 print(f'[{i+1}/{len(targets)}] {g} seeds={seed_ids}',flush=True)
 with log.open('w') as h: cp=subprocess.run(cmd,cwd=ROOT,stdout=h,stderr=subprocess.STDOUT)
 if cp.returncode: raise SystemExit(f'{g} failed; see {log}')
 d=pd.read_csv(csv,dtype=str).fillna('');a=json.load(open(audit));records.append({'requested_group':g,'number':row.number,'substrate':row.substrate,'product':row['product'],'teacher_seed_ids':';'.join(seed_ids),'teacher_seed_count':len(seed_ids),'teacher_score_source':str(d.score_source.iloc[0]),'teacher_route_id':str(d.route_id.iloc[0]),'teacher_candidate_universe':int(d.candidate_universe_size.iloc[0]),'structure_expert_applied':bool(a.get('structure_expert_applied',False)),'top1':str(d.candidate_id.iloc[0])})
pd.DataFrame(records).to_csv(OUT/'teacher_summary.csv',index=False)
print(pd.DataFrame(records).to_string(index=False),flush=True)
