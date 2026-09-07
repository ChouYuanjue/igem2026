import pandas as pd
from projects.active.terpene_screening.audit_rhea128_to141_external_failure_modes import merge_query_audit,aggregate


def test_failure_audit_merges_predefined_slices_without_selection_logic():
    base=pd.DataFrame([{'direction':'reaction_to_enzyme','query_id':'R1','reciprocal_rank':.5,'average_precision':.4,'roc_auc':.8,'ndcg_at_10':.3,'hit_at_10':1.,'hit_at_20':1.,'hit_at_50':1.,'best_positive_rank':2.,'best_positive_rank_fraction':.02}])
    cand=base.copy(); cand.loc[0,'reciprocal_rank']=.4; cand.loc[0,'roc_auc']=.9; cand.loc[0,'best_positive_rank']=3.; cand.loc[0,'best_positive_rank_fraction']=.03
    rs=pd.DataFrame([{'reaction_id':'R1','max_train_drfp_tanimoto':'0.2','reaction_similarity_bucket':'lt0p3'}])
    ps=pd.DataFrame([{'protein_id':'P1','reaction_id':'R1','mmseqs_fident':'','protein_identity_bucket':'no_hit'},{'protein_id':'P2','reaction_id':'R1','mmseqs_fident':'0.35','protein_identity_bucket':'20_40'}])
    ca=pd.DataFrame([{'reaction_id':'R1','status':'valid','warning':'','feature_nonzero':'10'}])
    merged=merge_query_audit(base,cand,rs,ps,ca)
    assert merged.loc[0,'n_positives']==2
    assert merged.loc[0,'positive_count_bucket']=='2'
    assert merged.loc[0,'center_status']=='valid'
    assert merged.loc[0,'max_positive_identity_bucket']=='20_40'
    assert merged.loc[0,'no_hit_positive_fraction']==.5
    summary=aggregate(merged)
    assert summary['mrr']['delta']<0 and summary['macro_roc_auc']['delta']>0


def test_zero_fallback_center_status_is_explicit():
    base=pd.DataFrame([{'direction':'reaction_to_enzyme','query_id':'R1','reciprocal_rank':.1,'average_precision':.1,'roc_auc':.5,'ndcg_at_10':0.,'hit_at_10':0.,'hit_at_20':0.,'hit_at_50':0.,'best_positive_rank':100.,'best_positive_rank_fraction':.1}])
    rs=pd.DataFrame([{'reaction_id':'R1','max_train_drfp_tanimoto':'1','reaction_similarity_bucket':'ge0p9'}])
    ps=pd.DataFrame([{'protein_id':'P1','reaction_id':'R1','mmseqs_fident':'','protein_identity_bucket':'no_hit'}])
    ca=pd.DataFrame([{'reaction_id':'R1','status':'zero_fallback','warning':'missing_mapping','feature_nonzero':''}])
    merged=merge_query_audit(base,base.copy(),rs,ps,ca)
    assert merged.loc[0,'center_status']=='zero_fallback'
    assert merged.loc[0,'max_positive_identity_bucket']=='all_no_hit'
