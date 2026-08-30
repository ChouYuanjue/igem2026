from pathlib import Path

import pandas as pd
import torch

from projects.active.terpene_screening.evaluate_enzymecage_405_cleanroom import filter_author_valid_pocket_reservoir


def test_author_valid_pocket_filter_mirrors_infer_uid_filter(tmp_path: Path) -> None:
    gvp=tmp_path/'gvp.pt'; torch.save({'P1': ('dummy',), 'P3': ('dummy',)}, gvp)
    frame=pd.DataFrame({
        'UniprotID':['P1','P2','P3','P2'],
        'reaction_id':['R1','R1','R2','R2'],
        'label':[1,0,1,0],
    })
    filtered,audit=filter_author_valid_pocket_reservoir(frame,gvp)
    assert filtered.UniprotID.tolist()==['P1','P3']
    assert audit['removed_candidate_uids']==1
    assert audit['filtered_queries']==2
    assert audit['queries_without_positive_after_filter']==0


def test_author_valid_pocket_filter_reports_lost_positive_query(tmp_path: Path) -> None:
    gvp=tmp_path/'gvp.pt'; torch.save({'N1': ('dummy',)}, gvp)
    frame=pd.DataFrame({'UniprotID':['P1','N1'],'reaction_id':['R1','R1'],'label':[1,0]})
    _,audit=filter_author_valid_pocket_reservoir(frame,gvp)
    assert audit['queries_without_positive_after_filter']==1
