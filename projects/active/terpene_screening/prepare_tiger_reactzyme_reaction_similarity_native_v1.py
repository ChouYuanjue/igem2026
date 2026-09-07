from __future__ import annotations

import json
from collections import defaultdict

import numpy as np
import pandas as pd
import torch

from projects.active.terpene_screening.tiger_reactzyme_reaction_similarity_native_v1_common import (
    BASE_FEATURE_ROOT, CONTRACT, EXPECTED_ARCHIVE_MD5, PREPARED, PROTEIN_SEQUENCE_TSV, ROOT, TEST_POS, TRAIN_POS,
    bag_feature, md5_file, normalize_reaction_bag, normalize_sequence, sha256_file,
)


def load_positive(path):
    payload = torch.load(path, map_location='cpu', weights_only=False)
    rows=[]
    for source_key, value in payload.items():
        if len(value) < 2:
            raise ValueError(f'unexpected ReactZyme tuple at {source_key}')
        rows.append({'source_key':str(source_key),'reaction_bag':normalize_reaction_bag(value[0]),'sequence':normalize_sequence(value[1])})
    return pd.DataFrame(rows)


def main():
    PREPARED.mkdir(parents=True, exist_ok=True)
    archive=ROOT/'data/external/reactzyme/reaction_smi_split.zip'
    if md5_file(archive) != EXPECTED_ARCHIVE_MD5:
        raise AssertionError('official reaction_smi_split archive MD5 drift')
    train_raw=load_positive(TRAIN_POS); test_raw=load_positive(TEST_POS)
    if len(train_raw)!=163771 or len(test_raw)!=14692:
        raise AssertionError('official source row count drift')
    train_bags=set(train_raw.reaction_bag); test_bags=set(test_raw.reaction_bag)
    if len(train_bags)!=7340 or len(test_bags)!=386 or train_bags & test_bags:
        raise AssertionError('reaction-novel support identity drift')

    seq_table=pd.read_csv(PROTEIN_SEQUENCE_TSV,sep='\t',dtype=str).fillna('')
    entries=pd.read_csv(BASE_FEATURE_ROOT/'entries.csv',dtype=str)
    if len(seq_table)!=185918 or len(entries)!=185918 or list(seq_table.protein_id.astype(str))!=list(entries.Entry.astype(str)):
        raise AssertionError('fixed general EnzGFM feature registry drift')
    seq_to_rows=defaultdict(list)
    for i,s in enumerate(seq_table.sequence.astype(str)):
        seq_to_rows[normalize_sequence(s)].append(i)

    train_sequences=sorted(set(train_raw.sequence)); test_sequences=sorted(set(test_raw.sequence))
    missing_train=[s for s in train_sequences if s not in seq_to_rows]
    missing_test=[s for s in test_sequences if s not in seq_to_rows]
    if len(missing_train)!=1 or missing_test:
        raise AssertionError(f'protein mapping support drift: missing_train={len(missing_train)} missing_test={len(missing_test)}')
    base_count=len(entries); overlay_map={s:base_count+i for i,s in enumerate(sorted(missing_train))}

    all_bags=sorted(train_bags|test_bags); bag_to_idx={b:i for i,b in enumerate(all_bags)}
    feats=[]; audit=[]
    for i,bag in enumerate(all_bags):
        v,valid,invalid=bag_feature(bag); feats.append(v); audit.append({'reaction_idx':i,'reaction_bag':bag,'valid_molecules':valid,'invalid_molecules':invalid,'zero_feature':valid==0,'split':'test' if bag in test_bags else 'train'})
    np.save(PREPARED/'reaction_features.npy',np.stack(feats).astype(np.float32))
    pd.DataFrame(audit).to_csv(PREPARED/'reaction_entries.csv',index=False)

    def protein_idx(seq):
        rows=seq_to_rows.get(seq,[])
        return min(rows) if rows else overlay_map[seq]
    def mapped(frame):
        out=frame.copy(); out['protein_idx']=out.sequence.map(protein_idx).astype(int); out['reaction_idx']=out.reaction_bag.map(bag_to_idx).astype(int); return out
    train=mapped(train_raw); test=mapped(test_raw)
    train.to_csv(PREPARED/'train_pairs.csv',index=False); test.to_csv(PREPARED/'test_pairs.csv',index=False)

    test_proteins=test[['sequence','protein_idx']].drop_duplicates().sort_values('sequence').reset_index(drop=True)
    test_reactions=test[['reaction_bag','reaction_idx']].drop_duplicates().sort_values('reaction_bag').reset_index(drop=True)
    test_proteins.to_csv(PREPARED/'test_proteins.csv',index=False); test_reactions.to_csv(PREPARED/'test_reactions.csv',index=False)
    unique_pairs=test[['protein_idx','reaction_idx']].drop_duplicates()
    if len(test_proteins)!=14688 or len(test_reactions)!=386 or len(unique_pairs)!=14689:
        raise AssertionError('native test label-matrix support drift')

    overlay_ids=pd.DataFrame({'protein_id':[f'RXNSIM_TRAIN_OVERLAY_{i:03d}' for i in range(len(missing_train))]})
    overlay_seqs=overlay_ids.copy(); overlay_seqs['sequence']=sorted(missing_train)
    overlay_ids.to_csv(PREPARED/'overlay_protein_ids.csv',index=False)
    overlay_seqs.to_csv(PREPARED/'overlay_protein_sequences.tsv',sep='\t',index=False)
    overlay_map_frame=overlay_seqs.copy(); overlay_map_frame['protein_idx']=[overlay_map[s] for s in sorted(missing_train)]
    overlay_map_frame.to_csv(PREPARED/'overlay_mapping.csv',index=False)

    summary={
      'status':'prepared_support_only_no_model_scores',
      'contract':str(CONTRACT.relative_to(ROOT)),
      'official_archive_md5':md5_file(archive),
      'source_rows':{'train':len(train_raw),'test':len(test_raw)},
      'normalized_support':{
        'train_unique_reactions':len(train_bags),'test_unique_reactions':len(test_bags),'train_test_reaction_overlap':0,
        'train_unique_sequences':len(train_sequences),'test_unique_sequences':len(test_sequences),
        'train_test_sequence_overlap':len(set(train_sequences)&set(test_sequences)),
        'test_unique_positive_pairs':len(unique_pairs),
        'native_score_shape':[len(test_proteins),len(test_reactions)]
      },
      'feature_mapping':{'base_proteins':base_count,'missing_train_sequences':len(missing_train),'missing_test_sequences':len(missing_test),'overlay_rows_required':len(missing_train)},
      'reaction_features':{'rows':len(all_bags),'dimension':4096,'zero_rows':int(sum(x['zero_feature'] for x in audit)),'invalid_molecule_tokens':int(sum(x['invalid_molecules'] for x in audit))},
      'test_labels_read_for_support_only':True,'model_scores_read':False,'test_performance_used_for_selection':False,
      'source_sha256':{
        'train':sha256_file(TRAIN_POS),'test':sha256_file(TEST_POS),'protein_sequences':sha256_file(PROTEIN_SEQUENCE_TSV),'base_feature_manifest':sha256_file(BASE_FEATURE_ROOT/'manifest.json')
      }
    }
    (PREPARED/'summary.json').write_text(json.dumps(summary,indent=2)+'\n'); print(json.dumps(summary,indent=2))

if __name__=='__main__': main()
