from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import pandas as pd

ROOT=Path(__file__).resolve().parents[3]
DEFAULT_COMPACT=ROOT/'data/external/enzymecage_current/rhea_2023_compact.csv.gz'
DEFAULT_CLEAN=ROOT/'data/external/enzymecage_current/catalyst_features/clean2023/training_pairs.csv'
DEFAULT_PROTEIN_META=ROOT/'data/catalyst_candidate_universes/general_merged/protein_metadata.csv'
DEFAULT_REACTIONS=ROOT/'data/catalyst_candidate_universes/general_merged/reaction_features/drfp_categorical_rdkitplus_center_v1/entries.csv'
DEFAULT_OUTPUT=ROOT/'results/rhea128_to141_external_v2'
CELL='rhea128_to141_sprot_strict_double_cold_v2'


def sha256_file(path:Path)->str:
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1<<20),b''): h.update(chunk)
    return h.hexdigest()


def build_unambiguous_protein_alias_map(meta:pd.DataFrame)->tuple[dict[str,str],dict[str,object]]:
    alias_to_ids:dict[str,set[str]]={}
    for row in meta[['protein_id','canonical_accession','aliases']].fillna('').itertuples(index=False):
        vals={str(row.protein_id),str(row.canonical_accession)}
        vals.update(x for x in re.split(r'[;]+',str(row.aliases)) if x)
        for alias in vals:
            if alias:
                alias_to_ids.setdefault(alias,set()).add(str(row.protein_id))
    ambiguous={alias:ids for alias,ids in alias_to_ids.items() if len(ids)>1}
    mapping={alias:next(iter(ids)) for alias,ids in alias_to_ids.items() if len(ids)==1}
    return mapping,{
        'metadata_rows':int(len(meta)),
        'nontrivial_alias_rows':int(((meta.aliases.fillna('')!=meta.protein_id.fillna('')) & meta.aliases.fillna('').ne('')).sum()),
        'alias_keys':int(len(alias_to_ids)),
        'unambiguous_alias_keys':int(len(mapping)),
        'ambiguous_alias_keys':int(len(ambiguous)),
        'ambiguous_examples':{k:sorted(v) for k,v in list(sorted(ambiguous.items()))[:10]},
    }


def load_release(path:Path)->pd.DataFrame:
    frame=pd.read_csv(path,sep='\t',dtype=str).fillna('')
    required={'RHEA_ID','ID'}
    if not required<=set(frame.columns):
        raise ValueError(f'{path} missing required columns {sorted(required-set(frame.columns))}')
    local=frame[['RHEA_ID','ID']].copy()
    local=local[local.RHEA_ID.str.fullmatch(r'\d+') & local.ID.str.len().gt(0)]
    local['reaction_source_id']='RHEA:'+local.RHEA_ID.astype(str)
    local['protein_source_id']=local.ID.astype(str)
    return local[['protein_source_id','reaction_source_id']].drop_duplicates().reset_index(drop=True)


def load_compact(path:Path)->pd.DataFrame:
    frame=pd.read_csv(path,dtype=str).fillna('')
    required={'UniprotID','reaction_id'}
    if not required<=set(frame.columns):
        raise ValueError(f'{path} missing required columns {sorted(required-set(frame.columns))}')
    return frame[['UniprotID','reaction_id']].rename(columns={'UniprotID':'protein_source_id','reaction_id':'reaction_source_id'}).drop_duplicates().reset_index(drop=True)


def map_to_frozen_universe(source:pd.DataFrame,protein_alias:dict[str,str],reaction_ids:set[str])->tuple[pd.DataFrame,dict[str,int]]:
    mapped=source.copy()
    mapped['protein_id']=mapped.protein_source_id.map(protein_alias).fillna('')
    mapped['reaction_id']=mapped.reaction_source_id.where(mapped.reaction_source_id.isin(reaction_ids),'')
    ok=mapped.protein_id.ne('') & mapped.reaction_id.ne('')
    out=mapped.loc[ok,['protein_id','reaction_id']].drop_duplicates().reset_index(drop=True)
    return out,{
        'source_pairs':int(len(source)),
        'protein_alias_mappable_source_pairs':int(mapped.protein_id.ne('').sum()),
        'reaction_supported_source_pairs':int(mapped.reaction_id.ne('').sum()),
        'mapped_unique_pairs':int(len(out)),
    }


def build(old:pd.DataFrame,new:pd.DataFrame,compact:pd.DataFrame,clean:pd.DataFrame,protein_meta:pd.DataFrame,reaction_ids:set[str]):
    protein_alias,alias_audit=build_unambiguous_protein_alias_map(protein_meta)
    old_m,old_audit=map_to_frozen_universe(old,protein_alias,reaction_ids)
    new_m,new_audit=map_to_frozen_universe(new,protein_alias,reaction_ids)
    compact_m,compact_audit=map_to_frozen_universe(compact,protein_alias,reaction_ids)
    clean=clean[['protein_id','reaction_id']].drop_duplicates().copy()
    clean_set=set(map(tuple,clean.itertuples(index=False,name=None)))
    compact_set=set(map(tuple,compact_m.itertuples(index=False,name=None)))
    if compact_set != clean_set:
        raise ValueError(f'2023 compact alias mapping does not exactly reconstruct clean2023: missing={len(clean_set-compact_set)} extra={len(compact_set-clean_set)}')
    old_set=set(map(tuple,old_m.itertuples(index=False,name=None)))
    train_p=set(clean.protein_id); train_r=set(clean.reaction_id)
    supported=new_m.copy()
    supported['is_new_vs_release128']=[pair not in old_set for pair in supported[['protein_id','reaction_id']].itertuples(index=False,name=None)]
    supported['is_absent_clean2023_pair']=[pair not in clean_set for pair in supported[['protein_id','reaction_id']].itertuples(index=False,name=None)]
    supported['protein_cold_vs_clean2023']=~supported.protein_id.isin(train_p)
    supported['reaction_cold_vs_clean2023']=~supported.reaction_id.isin(train_r)
    trigger=supported[
        supported.is_new_vs_release128 & supported.is_absent_clean2023_pair &
        supported.protein_cold_vs_clean2023 & supported.reaction_cold_vs_clean2023
    ].copy()
    eligible=[]
    for reaction_id,group in supported[supported.reaction_id.isin(set(trigger.reaction_id))].groupby('reaction_id',sort=True):
        if reaction_id not in train_r and bool(group.protein_cold_vs_clean2023.all()):
            eligible.append(reaction_id)
    eligible=set(eligible)
    test=supported[supported.reaction_id.isin(eligible)][['protein_id','reaction_id']].drop_duplicates().sort_values(['reaction_id','protein_id']).reset_index(drop=True)
    trigger=trigger[trigger.reaction_id.isin(eligible)][['protein_id','reaction_id']].drop_duplicates().sort_values(['reaction_id','protein_id']).reset_index(drop=True)
    test_set=set(map(tuple,test.itertuples(index=False,name=None)))
    if test_set & clean_set: raise AssertionError('selected test pair overlaps clean2023')
    if set(test.protein_id)&train_p: raise AssertionError('selected test support is not protein-cold')
    if set(test.reaction_id)&train_r: raise AssertionError('selected test support is not reaction-cold')
    audit={
        'alias_mapping':alias_audit,
        'release128_mapping':old_audit,
        'release141_mapping':new_audit,
        'clean2023_compact_mapping':compact_audit,
        'clean2023_exact_reconstruction':True,
        'release128_mapped_pairs':int(len(old_m)),
        'release141_mapped_pairs':int(len(new_m)),
        'release141_new_vs_release128_mapped_pairs':int(supported.is_new_vs_release128.sum()),
        'strict_new_trigger_pairs':int(len(trigger)),
        'test_pairs':int(len(test)),
        'test_query_reactions':int(test.reaction_id.nunique()),
        'test_positive_proteins':int(test.protein_id.nunique()),
        'clean2023_train_pairs':int(len(clean)),
        'clean2023_train_proteins':int(len(train_p)),
        'clean2023_train_reactions':int(len(train_r)),
        'exact_train_test_pair_overlap':0,'train_test_protein_overlap':0,'train_test_reaction_overlap':0,
        'query_selection':'mapped release141 pair absent from mapped release128 and clean2023, with protein+reaction entities cold vs clean2023; every mapped release141 positive for the selected reaction must be protein-cold',
        'label_completion':'all mapped release141 Swiss-Prot positives inside the frozen candidate universe are positives for each selected query',
    }
    return clean,test,trigger,audit


def main()->None:
    ap=argparse.ArgumentParser(description='Materialize provenance-correct Rhea128→141 strict-double-cold external benchmark without model scoring.')
    ap.add_argument('--release128-sprot',type=Path,required=True); ap.add_argument('--release141-sprot',type=Path,required=True)
    ap.add_argument('--compact2023',type=Path,default=DEFAULT_COMPACT); ap.add_argument('--clean2023',type=Path,default=DEFAULT_CLEAN)
    ap.add_argument('--protein-metadata',type=Path,default=DEFAULT_PROTEIN_META); ap.add_argument('--reaction-entries',type=Path,default=DEFAULT_REACTIONS)
    ap.add_argument('--output-root',type=Path,default=DEFAULT_OUTPUT); args=ap.parse_args()
    old=load_release(args.release128_sprot.resolve()); new=load_release(args.release141_sprot.resolve()); compact=load_compact(args.compact2023.resolve())
    clean=pd.read_csv(args.clean2023.resolve(),dtype=str).fillna(''); meta=pd.read_csv(args.protein_metadata.resolve(),dtype=str).fillna('')
    reactions=pd.read_csv(args.reaction_entries.resolve(),dtype=str).fillna(''); reaction_ids=set(reactions.reaction_id.astype(str))
    train,test,trigger,audit=build(old,new,compact,clean,meta,reaction_ids)
    cell=args.output_root.resolve()/CELL; cell.mkdir(parents=True,exist_ok=True)
    train.to_csv(cell/'train_pairs.csv',index=False); test.to_csv(cell/'test_pairs.csv',index=False); trigger.to_csv(cell/'delta_trigger_pairs.csv',index=False)
    manifest={
      'name':CELL,'claim_tier':'fresh_external_temporal_snapshot','strict_clean':True,'outer_benchmark_labels_used':True,
      'identifier_semantics':'direction-specific RHEA_ID plus unambiguous protein alias mapping into the frozen candidate universe',
      'release128_sprot':str(args.release128_sprot.resolve()),'release128_sprot_sha256':sha256_file(args.release128_sprot.resolve()),
      'release141_sprot':str(args.release141_sprot.resolve()),'release141_sprot_sha256':sha256_file(args.release141_sprot.resolve()),
      'compact2023':str(args.compact2023.resolve()),'compact2023_sha256':sha256_file(args.compact2023.resolve()),
      'clean2023':str(args.clean2023.resolve()),'clean2023_sha256':sha256_file(args.clean2023.resolve()),
      'protein_metadata':str(args.protein_metadata.resolve()),'protein_metadata_sha256':sha256_file(args.protein_metadata.resolve()),
      'fixed_candidate_proteins':int(len(meta)),'fixed_candidate_reactions':int(len(reactions)),
      'audit':audit,'model_performance_read_by_builder':False,
    }
    (cell/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    (args.output_root.resolve()/'summary.json').write_text(json.dumps({'protocol':'rhea128_to141_external_v2','cell':manifest},indent=2)+'\n')
    print(json.dumps(manifest,indent=2))

if __name__=='__main__': main()
