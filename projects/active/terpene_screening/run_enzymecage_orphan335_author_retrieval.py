from __future__ import annotations

import argparse
import collections
import hashlib
import importlib
import json
import pickle
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
AUTHOR = ROOT / "external_repos/EnzymeCAGE"


def sha256_file(path: Path) -> str:
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1<<20),b''): h.update(chunk)
    return h.hexdigest()


def load_author_module():
    if str(AUTHOR) not in sys.path: sys.path.insert(0,str(AUTHOR))
    mod=importlib.import_module('retrieve')
    # Upstream master references Counter but imports only defaultdict.
    # Supply the missing stdlib symbol; no retrieval/scoring logic is changed.
    mod.Counter = collections.Counter
    return mod


def main() -> None:
    ap=argparse.ArgumentParser(description='Run the official EnzymeCAGE Orphan-335 candidate retrieval/Selenzyme-style score with one import-only compatibility patch.')
    ap.add_argument('--data',type=Path,default=ROOT/'data/external/enzymecage_current/Orphan-335.csv')
    ap.add_argument('--db-2023',type=Path,required=True)
    ap.add_argument('--proevi',type=Path,required=True)
    ap.add_argument('--taxdis',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--topk',type=int,default=10)
    args=ap.parse_args()
    if args.topk!=10: raise ValueError('Orphan-335 author protocol fixes topk=10')
    data=pd.read_csv(args.data,dtype=str).fillna(''); db=pd.read_csv(args.db_2023,dtype=str).fillna('')
    pro=pickle.load(open(args.proevi,'rb')); tax=pickle.load(open(args.taxdis,'rb'))
    author=load_author_module()
    result=author.run_retrieval(data,db,'CANO_RXN_SMILES',pro,tax,topk=10,exclude_rxns=set())
    args.output.parent.mkdir(parents=True,exist_ok=True); result.to_csv(args.output,index=False)
    summary={'name':'enzymecage_orphan335_author_retrieval','author_source':'external_repos/EnzymeCAGE/retrieve.py','compatibility_patch':'supply collections.Counter missing from upstream imports only','data':str(args.data.resolve()),'data_sha256':sha256_file(args.data.resolve()),'db_2023':str(args.db_2023.resolve()),'db_2023_sha256':sha256_file(args.db_2023.resolve()),'top_similar_reactions':10,'query_count':int(result.CANO_RXN_SMILES.nunique()),'rows':int(len(result)),'candidate_uids':int(result.UniprotID.nunique()),'selection_uses_test_scores':False,'score_formula':'author retrieve.py: rxn_similarity*100 - tax_dis - 0.1*pro_evi'}
    (args.output.parent/'summary.json').write_text(json.dumps(summary,indent=2)+'\n'); print(json.dumps(summary,indent=2))

if __name__=='__main__': main()
