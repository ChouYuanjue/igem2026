import json
from pathlib import Path
import numpy as np
import pandas as pd
from projects.active.terpene_screening.package_catalyst_clean_mainline_runtime import package_runtime

def test_runtime_packaging_expands_query_registry_without_changing_training_schema(tmp_path: Path):
    bundle=tmp_path/'bundle'; bundle.mkdir(); features=tmp_path/'features'; features.mkdir()
    schema={'reaction_ids':['R2','R1'],'reaction_feature_dimension':3,'protein_feature_dimension':4}
    (bundle/'feature_schema.json').write_text(json.dumps(schema))
    pd.DataFrame({'row':[0,1,2],'reaction_id':['R0','R1','R2']}).to_csv(features/'entries.csv',index=False)
    np.save(features/'reaction_feature_matrix.npy',np.arange(9,dtype=np.float32).reshape(3,3))
    result=package_runtime(bundle,features)
    assert result['runtime_reaction_count']==3 and result['training_schema_reaction_count']==2
    assert json.loads((bundle/'training_feature_schema.json').read_text())['reaction_ids']==['R2','R1']
    runtime=json.loads((bundle/'feature_schema.json').read_text()); assert runtime['reaction_ids']==['R0','R1','R2']
    assert (bundle/'reaction_feature_matrix.npy').is_symlink() and np.load(bundle/'reaction_feature_matrix.npy').shape==(3,3)

def test_runtime_packaging_rejects_training_reaction_missing_from_registry(tmp_path: Path):
    bundle=tmp_path/'bundle'; bundle.mkdir(); features=tmp_path/'features'; features.mkdir()
    (bundle/'feature_schema.json').write_text(json.dumps({'reaction_ids':['MISSING'],'reaction_feature_dimension':2}))
    pd.DataFrame({'row':[0],'reaction_id':['R0']}).to_csv(features/'entries.csv',index=False); np.save(features/'reaction_feature_matrix.npy',np.zeros((1,2),dtype=np.float32))
    try: package_runtime(bundle,features)
    except ValueError as e: assert 'missing from runtime registry' in str(e)
    else: raise AssertionError('expected fail closed')
