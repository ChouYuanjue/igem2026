from __future__ import annotations
import json
from pathlib import Path
import numpy as np,pandas as pd
from projects.active.bridge.model.assets import ROOT
from projects.active.bridge.model.index import FibreCandidateIndex

SNAPSHOT=pd.Timestamp('2023-07-12')
SOURCE=ROOT/'data/external/reactzyme/cleaned_uniprot_rhea.tsv'
TRAIN=ROOT/'data/external/enzymecage_current/catalyst_features/clean2023/training_pairs.csv'
CAGE=ROOT/'data/external/enzymecage_current/rhea_2023_compact.csv.gz'
META=ROOT/'data/catalyst_candidate_universes/general_merged/protein_metadata.csv'
OUT=ROOT/'results/bridge_post2023_growth_benchmark_v1'

def main():
 OUT.mkdir(parents=True,exist_ok=True)
 tr=pd.read_csv(TRAIN,dtype=str).fillna('').drop_duplicates(['protein_id','reaction_id'])
 s=pd.read_csv(SOURCE,sep='\t',dtype=str).fillna('');s['created']=pd.to_datetime(s['Date of creation'],format='%Y%m%d',errors='coerce');s=s[s.created>SNAPSHOT].copy();s['reaction_id']=s['Rhea ID'].str.split(';');x=s[['Entry','created','reaction_id']].explode('reaction_id').rename(columns={'Entry':'protein_id'});x['reaction_id']=x.reaction_id.str.strip();x=x[x.reaction_id.ne('')].drop_duplicates(['protein_id','reaction_id'])
 raw_edges=len(x);raw_proteins=x.protein_id.nunique()
 idx=FibreCandidateIndex(device='cpu');pu=set(idx.protein_ids);ru=set(idx.reaction_ids);x['protein_in_universe']=x.protein_id.isin(pu);x['reaction_in_universe']=x.reaction_id.isin(ru);dropped=x[~(x.protein_in_universe & x.reaction_in_universe)].copy();x=x[x.protein_in_universe & x.reaction_in_universe].copy()
 sr=set(tr.reaction_id.astype(str));x['reaction_seen_2023']=x.reaction_id.isin(sr)
 ev=x.groupby('protein_id').agg(created=('created','min'),edges=('reaction_id','size'),seen_edges=('reaction_seen_2023','sum'),frontier_edges=('reaction_seen_2023',lambda z:int((~z).sum()))).reset_index();ev['event_type']=np.where(ev.frontier_edges>0,'frontier','attachment');x=x.merge(ev[['protein_id','event_type','edges']],on='protein_id',how='left',validate='many_to_one').rename(columns={'edges':'event_size'})
 # CAGE 2023 association support (alias aware).
 meta=pd.read_csv(META,dtype=str).fillna('');aliases={r.protein_id:[r.protein_id,r.canonical_accession,*str(r.aliases).split(';')] for r in meta.itertuples(index=False)};c=pd.read_csv(CAGE,dtype=str).fillna('');cp=set(c.UniprotID.astype(str));cr=set(c.reaction_id.astype(str));pairs=set(zip(c.UniprotID.astype(str),c.reaction_id.astype(str)))
 def ps(p):return any(a and a in cp for a in aliases.get(p,[p]))
 def prs(p,r):return any(a and (a,r) in pairs for a in aliases.get(p,[p]))
 x['cage_protein_seen_2023']=[ps(p) for p in x.protein_id];x['cage_reaction_seen_2023']=x.reaction_id.isin(cr);x['cage_pair_seen_2023']=[prs(p,r) for p,r in x[['protein_id','reaction_id']].itertuples(index=False)]
 x.to_csv(OUT/'targets.csv',index=False);ev.to_csv(OUT/'events.csv',index=False);dropped.to_csv(OUT/'dropped_outside_universe.csv',index=False)
 summary={'schema':'bridge-post2023-growth-benchmark-v1','snapshot':'2023-07-12 official clean2023','source':'current ReactZyme/UniProt-Rhea projection using UniProt protein creation date','scope':'entity-arrival growth benchmark; does not claim to observe old-old relation curation timestamps','raw':{'proteins':raw_proteins,'edges':raw_edges},'eligible':{'proteins':int(x.protein_id.nunique()),'reactions':int(x.reaction_id.nunique()),'edges':len(x),'dropped_edges':len(dropped)},'events':{k:{'proteins':int(len(g)),'edges':int(g.edges.sum()),'mean_edges':float(g.edges.mean()),'median_edges':float(g.edges.median()),'max_edges':int(g.edges.max())} for k,g in ev.groupby('event_type')},'edge_types':{'reaction_seen_2023':int(x.reaction_seen_2023.sum()),'reaction_unseen_2023':int((~x.reaction_seen_2023).sum())},'cage_2023_support':{'protein_seen_unique':int(x.loc[x.cage_protein_seen_2023,'protein_id'].nunique()),'protein_total':int(x.protein_id.nunique()),'pair_seen_edges':int(x.cage_pair_seen_2023.sum()),'reaction_seen_edges':int(x.cage_reaction_seen_2023.sum())}}
 (OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
