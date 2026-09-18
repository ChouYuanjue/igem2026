from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

ROOT=Path(__file__).resolve().parents[3]
DEFAULT_CLEAN=ROOT/'data/external/enzymecage_current/catalyst_features/clean2023/training_pairs.csv'
DEFAULT_PROTEINS=ROOT/'data/catalyst_candidate_universes/general_merged/proteins/entries.csv'
DEFAULT_REACTIONS=ROOT/'data/catalyst_candidate_universes/general_merged/reaction_features/drfp_categorical_rdkitplus_center_v1/entries.csv'
DEFAULT_OUTPUT=ROOT/'results/rhea128_to141_external_v1'
CELL='rhea128_to141_sprot_strict_double_cold_v1'


def sha256_file(path:Path)->str:
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1<<20),b''): h.update(chunk)
    return h.hexdigest()


def load_rhea_sprot(path:Path)->pd.DataFrame:
    frame=pd.read_csv(path,sep='\t',dtype=str).fillna('')
    required={'MASTER_ID','ID'}
    if not required<=set(frame.columns):
        raise ValueError(f'{path} missing required columns {sorted(required-set(frame.columns))}')
    local=frame[['MASTER_ID','ID']].copy()
    local=local[local.MASTER_ID.str.fullmatch(r'\d+') & local.ID.str.len().gt(0)]
    local['reaction_id']='RHEA:'+local.MASTER_ID.astype(str)
    local['protein_id']=local.ID.astype(str)
    return local[['protein_id','reaction_id']].drop_duplicates().reset_index(drop=True)


def build(old:pd.DataFrame,new:pd.DataFrame,clean:pd.DataFrame,protein_ids:set[str],reaction_ids:set[str])->tuple[pd.DataFrame,pd.DataFrame,pd.DataFrame,dict[str,object]]:
    clean=clean[['protein_id','reaction_id']].drop_duplicates().copy()
    clean_pairs=set(map(tuple,clean.itertuples(index=False,name=None)))
    old_pairs=set(map(tuple,old[['protein_id','reaction_id']].itertuples(index=False,name=None)))
    train_p=set(clean.protein_id); train_r=set(clean.reaction_id)
    supported=new[new.protein_id.isin(protein_ids)&new.reaction_id.isin(reaction_ids)].drop_duplicates().copy()
    supported['is_new_vs_release128']=[(p,r) not in old_pairs for p,r in supported[['protein_id','reaction_id']].itertuples(index=False,name=None)]
    supported['is_absent_clean2023_pair']=[(p,r) not in clean_pairs for p,r in supported[['protein_id','reaction_id']].itertuples(index=False,name=None)]
    supported['protein_cold_vs_clean2023']=~supported.protein_id.isin(train_p)
    supported['reaction_cold_vs_clean2023']=~supported.reaction_id.isin(train_r)
    trigger=supported[
        supported.is_new_vs_release128 & supported.is_absent_clean2023_pair &
        supported.protein_cold_vs_clean2023 & supported.reaction_cold_vs_clean2023
    ].copy()
    trigger_queries=set(trigger.reaction_id)
    eligible=[]
    for reaction_id,group in supported[supported.reaction_id.isin(trigger_queries)].groupby('reaction_id',sort=True):
        # Complete fixed-universe 141 positive support for the query must be protein-cold.
        if bool(group.protein_cold_vs_clean2023.all()) and reaction_id not in train_r:
            eligible.append(reaction_id)
    eligible=set(eligible)
    test=supported[supported.reaction_id.isin(eligible)][['protein_id','reaction_id']].drop_duplicates().sort_values(['reaction_id','protein_id']).reset_index(drop=True)
    trigger=trigger[trigger.reaction_id.isin(eligible)][['protein_id','reaction_id']].drop_duplicates().sort_values(['reaction_id','protein_id']).reset_index(drop=True)
    if set(test.protein_id)&train_p: raise AssertionError('selected test support is not protein-cold')
    if set(test.reaction_id)&train_r: raise AssertionError('selected test support is not reaction-cold')
    if set(map(tuple,test.itertuples(index=False,name=None)))&clean_pairs: raise AssertionError('selected test pair overlaps clean2023')
    audit={
        'release128_pairs':int(len(old)),'release141_pairs':int(len(new)),
        'release141_fixed_universe_pairs':int(len(supported)),
        'new_vs_release128_fixed_universe_pairs':int(supported.is_new_vs_release128.sum()),
        'strict_new_trigger_pairs':int(len(trigger)),
        'test_pairs':int(len(test)),'test_query_reactions':int(test.reaction_id.nunique()),'test_positive_proteins':int(test.protein_id.nunique()),
        'clean2023_train_pairs':int(len(clean)),'clean2023_train_proteins':int(len(train_p)),'clean2023_train_reactions':int(len(train_r)),
        'exact_train_test_pair_overlap':0,'train_test_protein_overlap':0,'train_test_reaction_overlap':0,
        'query_selection':'at least one release141-vs-128 new supported pair that is pair/protein/reaction cold vs clean2023; all release141 positives inside the fixed candidate universe for that query must also be protein-cold',
        'label_completion':'for each selected query use every release141 Swiss-Prot positive inside the frozen 185,918-protein candidate universe, not only the delta-trigger pair',
    }
    return clean,test,trigger,audit


def main()->None:
    ap=argparse.ArgumentParser(description='Materialize the preregistered Rhea release128->141 Swiss-Prot strict-double-cold external benchmark without scoring a model.')
    ap.add_argument('--release128-sprot',type=Path,required=True); ap.add_argument('--release141-sprot',type=Path,required=True)
    ap.add_argument('--clean2023',type=Path,default=DEFAULT_CLEAN); ap.add_argument('--protein-entries',type=Path,default=DEFAULT_PROTEINS); ap.add_argument('--reaction-entries',type=Path,default=DEFAULT_REACTIONS)
    ap.add_argument('--output-root',type=Path,default=DEFAULT_OUTPUT); args=ap.parse_args()
    old=load_rhea_sprot(args.release128_sprot.resolve()); new=load_rhea_sprot(args.release141_sprot.resolve())
    clean=pd.read_csv(args.clean2023.resolve(),dtype=str).fillna('')
    proteins=pd.read_csv(args.protein_entries.resolve(),dtype=str).fillna(''); pcol='Entry' if 'Entry' in proteins else 'protein_id'
    reactions=pd.read_csv(args.reaction_entries.resolve(),dtype=str).fillna(''); rcol='reaction_id'
    train,test,trigger,audit=build(old,new,clean,set(proteins[pcol].astype(str)),set(reactions[rcol].astype(str)))
    cell=args.output_root.resolve()/CELL; cell.mkdir(parents=True,exist_ok=True)
    train.to_csv(cell/'train_pairs.csv',index=False); test.to_csv(cell/'test_pairs.csv',index=False); trigger.to_csv(cell/'delta_trigger_pairs.csv',index=False)
    manifest={
      'name':CELL,'claim_tier':'fresh_external_temporal_snapshot','strict_clean':True,'outer_benchmark_labels_used':True,
      'release128_sprot':str(args.release128_sprot.resolve()),'release128_sprot_sha256':sha256_file(args.release128_sprot.resolve()),
      'release141_sprot':str(args.release141_sprot.resolve()),'release141_sprot_sha256':sha256_file(args.release141_sprot.resolve()),
      'clean2023':str(args.clean2023.resolve()),'clean2023_sha256':sha256_file(args.clean2023.resolve()),
      'fixed_candidate_proteins':int(len(proteins)),'fixed_candidate_reactions':int(len(reactions)),
      'audit':audit,'model_performance_read_by_builder':False,
    }
    (cell/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    (args.output_root.resolve()/'summary.json').write_text(json.dumps({'protocol':'rhea128_to141_external_v1','cell':manifest},indent=2)+'\n')
    print(json.dumps(manifest,indent=2))

if __name__=='__main__': main()
