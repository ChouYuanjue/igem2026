from __future__ import annotations
import json
from collections import defaultdict
from pathlib import Path
import pandas as pd
from projects.active.bridge.model.assets import ROOT

TRAIN=ROOT/'data/external/enzymecage_current/catalyst_features/clean2023/training_pairs.csv'
BENCH=ROOT/'results/broad_rhea_fair_benchmarks_v1'
OUT=ROOT/'results/bridge_relation_unseen_max_v3'

def main():
 OUT.mkdir(parents=True,exist_ok=True)
 train=pd.read_csv(TRAIN,dtype=str).fillna('')[['protein_id','reaction_id']].drop_duplicates()
 seen=set(zip(train.protein_id.astype(str),train.reaction_id.astype(str)))
 sources=defaultdict(set); frames=[]
 for d in sorted(x for x in BENCH.iterdir() if x.is_dir()):
  for split in ('train','test'):
   p=d/f'{split}_pairs.csv'
   if not p.exists(): continue
   x=pd.read_csv(p,dtype=str).fillna('')[['protein_id','reaction_id']].drop_duplicates()
   for protein,reaction in x.itertuples(index=False):
    key=(str(protein),str(reaction)); sources[key].add(f'{d.name}:{split}')
   frames.append(x)
 pool=pd.concat(frames,ignore_index=True).drop_duplicates(['protein_id','reaction_id'])
 mask=[(str(p),str(r)) not in seen for p,r in pool.itertuples(index=False)]
 targets=pool.loc[mask].copy().sort_values(['reaction_id','protein_id']).reset_index(drop=True)
 targets['source_memberships']=[';'.join(sorted(sources[(str(p),str(r))])) for p,r in targets[['protein_id','reaction_id']].itertuples(index=False)]
 targets.to_csv(OUT/'targets.csv',index=False)
 train.groupby('reaction_id').protein_id.apply(lambda x:';'.join(sorted(set(map(str,x))))).rename('known_proteins').to_csv(OUT/'r2e_train_seed_bank.csv')
 train.groupby('protein_id').reaction_id.apply(lambda x:';'.join(sorted(set(map(str,x))))).rename('known_reactions').to_csv(OUT/'e2r_train_seed_bank.csv')
 summary={
  'schema':'bridge-relation-unseen-max-v3',
  'rule':'union every available train/test relation from the seven fair benchmark sources; remove only exact protein-reaction pairs present in clean2023 training relations; no protein-seen or reaction-seen constraint',
  'training_relations':int(len(train)),
  'union_relations':int(len(pool)),
  'relation_unseen_targets':int(len(targets)),
  'r2e_queries':int(targets.reaction_id.nunique()),
  'e2r_queries':int(targets.protein_id.nunique()),
  'source_membership_retained_for_audit':True,
 }
 (OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
