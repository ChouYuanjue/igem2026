from __future__ import annotations
import json
import numpy as np
import pandas as pd
import torch
from projects.active.terpene_screening.train_enzgfm_native_same_support_v1 import train_dual
from projects.active.terpene_screening.tiger_reactzyme_reaction_similarity_native_v1_common import (
    CONTRACT, PREPARED, RESULT_ROOT, DualTowerNative, load_combined_features, seed_all, sha256_file, SEED,
)

def main():
    seed_all(SEED)
    prep=json.loads((PREPARED/'summary.json').read_text())
    if prep['status']!='prepared_support_only_no_model_scores' or prep['test_performance_used_for_selection']:
        raise AssertionError('invalid prepared support provenance')
    pairs=pd.read_csv(PREPARED/'train_pairs.csv',dtype={'reaction_idx':int,'protein_idx':int}).rename(columns={'protein_idx':'protein_row'})
    if len(pairs)!=163771:
        raise AssertionError('official train support not preserved')
    reaction=np.load(PREPARED/'reaction_features.npy',mmap_mode='r'); proteins=load_combined_features(require_overlay=True)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model,history,config=train_dual(pairs,reaction,proteins,device)
    expected={'epochs':8,'batch_size':512,'lr':3e-4,'weight_decay':1e-4,'temperature':.05}
    if config!=expected:
        raise AssertionError(f'frozen recipe drift: {config}')
    out=RESULT_ROOT/'model'; out.mkdir(parents=True,exist_ok=True)
    torch.save({'model_type':'dual_tower_native_v1','candidate':'dual_tower','stage':'reaction_similarity_descriptive','seed':SEED,'config':config,'state_dict':model.state_dict()},out/'model.pt')
    pd.DataFrame(history).to_csv(out/'training_history.csv',index=False)
    summary={'status':'trained_single_prefrozen_recipe_without_native_test_scores','contract':str(CONTRACT),'training_rows':len(pairs),'unique_training_reactions':int(pairs.reaction_idx.nunique()),'unique_training_proteins':int(pairs.protein_row.nunique()),'config':config,'seed':SEED,'train_pairs_sha256':sha256_file(PREPARED/'train_pairs.csv'),'target_native_test_pairs_read':False,'target_native_test_scores_read':False,'target_native_test_performance_used_for_training_or_selection':False}
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n'); print(json.dumps(summary,indent=2))
if __name__=='__main__': main()
