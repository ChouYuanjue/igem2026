from __future__ import annotations

import json
from pathlib import Path
from collections import defaultdict
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
OUT=ROOT/'results/terpene_cage_fair'
OUT.mkdir(parents=True,exist_ok=True)
GATE_PATH=ROOT/'data/terpene_gate_matrix/gate_candidate_pools_with_evidence.csv'
if not GATE_PATH.exists():
    GATE_PATH=ROOT/'data/terpene_gate_matrix/gate_candidate_pools.csv'
FAIR_POOL=ROOT/'data/terpene_cage_fair/fair_cage_candidate_pairs.csv'
OLD=ROOT/'results/terpene_cage_screen/all_rhea_gate/all_pair_scores.csv'
NEW=ROOT/'results/terpene_cage_fair/new_cage_scores_parseable/all_pair_scores.csv'
UNPARSE=ROOT/'results/terpene_cage_fair/cage_unparseable_reactions.csv'
SUMMARY=ROOT/'results/terpene_cage_fair/fair_cage_pool_summary.json'
SELECT_GATES=['rxn_balanced_top20','rxn_balanced_top50','weighted_top100','recall_union_core']
BUDGETS=[5,10,20]


def norm_numeric(df, cols):
    for c in cols:
        if c in df.columns:
            df[c]=pd.to_numeric(df[c],errors='coerce').fillna(0)
    return df

def load_cage_scores():
    fair=pd.read_csv(FAIR_POOL, dtype=str, usecols=['reaction_id','uniprot_id','label']).fillna('')
    fair_keys=set(map(tuple, fair[['reaction_id','uniprot_id']].values))
    parts=[]
    for path,src in [(OLD,'old'),(NEW,'new')]:
        df=pd.read_csv(path,dtype=str).fillna('')
        keep_cols=[c for c in ['reaction_id','uniprot_id','label','cage_score'] if c in df.columns]
        df=df[keep_cols].copy()
        df['source']=src
        df['cage_score']=pd.to_numeric(df['cage_score'], errors='coerce')
        df['label']=pd.to_numeric(df.get('label',0), errors='coerce').fillna(0).astype(int)
        parts.append(df)
    allc=pd.concat(parts, ignore_index=True)
    allc=allc[allc[['reaction_id','uniprot_id']].apply(tuple,axis=1).isin(fair_keys)].copy()
    # Deduplicate, prefer new if any accidental overlap.
    allc['source_priority']=allc['source'].map({'old':0,'new':1}).fillna(0)
    allc=allc.sort_values(['reaction_id','uniprot_id','source_priority']).drop_duplicates(['reaction_id','uniprot_id'], keep='last')
    allc=allc.drop(columns=['source_priority'])
    allc=allc.sort_values(['reaction_id','cage_score','uniprot_id'], ascending=[True,False,True])
    allc['cage_rank']=allc.groupby('reaction_id').cumcount()+1
    allc['cage_n']=allc.groupby('reaction_id')['uniprot_id'].transform('size')
    denom=(allc['cage_n']-1).replace(0,1)
    allc['cage_rank_score']=1-(allc['cage_rank']-1)/denom
    allc.to_csv(OUT/'fair_cage_all_scores.csv', index=False)
    return allc

def hit_metrics(rows, reaction_universe, B):
    if len(rows)==0:
        hits=pd.Series(0,index=pd.Index(sorted(reaction_universe)))
    else:
        hits=rows.groupby('reaction_id')['label'].sum().reindex(pd.Index(sorted(reaction_universe)),fill_value=0)
    return float((hits>0).mean()), float(hits.mean()), float(hits.mean()/B)

def sort_top(df, score_col, B, extra_cols=None):
    cols=['reaction_id', score_col]
    asc=[True, False]
    if extra_cols:
        for c, a in extra_cols:
            cols.append(c); asc.append(a)
    cols.append('uniprot_id'); asc.append(True)
    return df.sort_values(cols, ascending=asc).groupby('reaction_id',sort=False).head(B)

def eval_reaction_only():
    cage=load_cage_scores()
    cage_cols=cage[['reaction_id','uniprot_id','cage_score','cage_rank_score','source']]
    C=pd.read_csv(GATE_PATH,dtype=str).fillna('')
    C=C[C['gate_id'].isin(SELECT_GATES)].copy()
    for c in ['label','gate_score','reaction_similarity','sequence_kmer','motif_score','precursor_match','product_skeleton','mechanism']:
        if c not in C.columns: C[c]=0
    C=norm_numeric(C,['label','gate_score','reaction_similarity','sequence_kmer','motif_score','precursor_match','product_skeleton','mechanism'])
    C['fusion_no_cage']=0.35*C['reaction_similarity']+0.25*C['sequence_kmer']+0.15*C['precursor_match']+0.10*C['product_skeleton']+0.10*C['motif_score']
    M=C.merge(cage_cols,on=['reaction_id','uniprot_id'],how='left')
    M['has_cage']=M['cage_score'].notna()
    M['cage_score_fill0']=M['cage_score'].fillna(0)
    M['cage_rank_score_fill0']=M['cage_rank_score'].fillna(0)
    for alpha in [0.005,0.01,0.02,0.05,0.1,0.2,0.5]:
        M[f'rxn_plus_cage_{alpha}']=M['reaction_similarity']+alpha*M['cage_rank_score_fill0']
        M[f'fusion_plus_cage_{alpha}']=M['fusion_no_cage']+alpha*M['cage_rank_score_fill0']
    all_rxns=set(pd.read_csv(ROOT/'results/terpene_gate_matrix/gate_reaction_level.csv',usecols=['reaction_id'],dtype=str)['reaction_id'])
    rows=[]
    panel_rows=[]
    for gate,g in M.groupby('gate_id'):
        gate_rxns=set(g['reaction_id'])
        cage_rxns=set(g.loc[g['has_cage'],'reaction_id'])
        for scope, universe in [('all513',all_rxns),('gate_rxns',gate_rxns),('cage_available_rxns',cage_rxns)]:
            for B in BUDGETS:
                methods={
                    'reaction_similarity':'reaction_similarity',
                    'fusion_no_cage':'fusion_no_cage',
                    'cage_only_rank':'cage_rank_score_fill0',
                    'cage_only_raw':'cage_score_fill0',
                    'rxn_plus_cage_0.005':'rxn_plus_cage_0.005',
                    'rxn_plus_cage_0.01':'rxn_plus_cage_0.01',
                    'rxn_plus_cage_0.02':'rxn_plus_cage_0.02',
                    'rxn_plus_cage_0.05':'rxn_plus_cage_0.05',
                    'rxn_plus_cage_0.1':'rxn_plus_cage_0.1',
                    'fusion_plus_cage_0.01':'fusion_plus_cage_0.01',
                    'fusion_plus_cage_0.05':'fusion_plus_cage_0.05',
                    'fusion_plus_cage_0.1':'fusion_plus_cage_0.1',
                }
                for name,col in methods.items():
                    top=sort_top(g,col,B)
                    hp,eh,pr=hit_metrics(top,universe,B)
                    rows.append({'gate_id':gate,'scope':scope,'method':name,'B':B,'n_reactions':len(universe),'hit_probability':hp,'expected_hits':eh,'precision':pr})
                # priority by CAGE threshold, then reaction similarity
                for q in [0.7,0.8,0.9,0.95]:
                    tmp=g.copy()
                    tmp[f'cage_priority_q{q}']=(tmp['cage_rank_score_fill0']>=q).astype(int)
                    top=tmp.sort_values(['reaction_id',f'cage_priority_q{q}','reaction_similarity','uniprot_id'],ascending=[True,False,False,True]).groupby('reaction_id',sort=False).head(B)
                    hp,eh,pr=hit_metrics(top,universe,B)
                    rows.append({'gate_id':gate,'scope':scope,'method':f'cage_priority_q{q}','B':B,'n_reactions':len(universe),'hit_probability':hp,'expected_hits':eh,'precision':pr})
                # rescue panels: main rxn B-k plus CAGE k
                ks=[1,2] if B==5 else ([2,5] if B==10 else [2,5,10])
                for k in ks:
                    panels=[]
                    for rid,grp in g.groupby('reaction_id'):
                        main=grp.sort_values(['reaction_similarity','uniprot_id'],ascending=[False,True]).head(max(0,B-k))
                        used=set(main['uniprot_id'])
                        rescue=grp[~grp['uniprot_id'].isin(used)].sort_values(['cage_rank_score_fill0','uniprot_id'],ascending=[False,True]).head(k)
                        panels.append(pd.concat([main,rescue],ignore_index=True))
                    top=pd.concat(panels,ignore_index=True) if panels else pd.DataFrame()
                    hp,eh,pr=hit_metrics(top,universe,B)
                    rows.append({'gate_id':gate,'scope':scope,'method':f'rxn_main_plus_cage_rescue_k{k}','B':B,'n_reactions':len(universe),'hit_probability':hp,'expected_hits':eh,'precision':pr})
    R=pd.DataFrame(rows)
    R.to_csv(OUT/'fair_cage_reaction_only_metrics.csv',index=False)
    # Compare best CAGE-aware against rxn baseline.
    comp=[]
    for (gate,scope,B),grp in R.groupby(['gate_id','scope','B']):
        base=grp[grp['method'].eq('reaction_similarity')].iloc[0]
        best=grp.sort_values(['hit_probability','expected_hits','precision'],ascending=False).iloc[0]
        cage_grp=grp[~grp['method'].isin(['reaction_similarity','fusion_no_cage'])]
        best_cage=cage_grp.sort_values(['hit_probability','expected_hits','precision'],ascending=False).iloc[0]
        comp.append({
            'gate_id':gate,'scope':scope,'B':B,
            'rxn_hit':base.hit_probability,'rxn_expected':base.expected_hits,
            'best_method':best.method,'best_hit':best.hit_probability,'best_expected':best.expected_hits,
            'best_cage_method':best_cage.method,'best_cage_hit':best_cage.hit_probability,'best_cage_expected':best_cage.expected_hits,
            'delta_hit_vs_rxn':best_cage.hit_probability-base.hit_probability,
            'delta_expected_vs_rxn':best_cage.expected_hits-base.expected_hits,
        })
    Comp=pd.DataFrame(comp)
    Comp.to_csv(OUT/'fair_cage_reaction_only_comparison.csv',index=False)
    coverage=[]
    for gate,g in M.groupby('gate_id'):
        coverage.append({
            'gate_id':gate,
            'rows':len(g),'rows_with_cage':int(g['has_cage'].sum()),'row_cage_coverage':float(g['has_cage'].mean()),
            'reactions':g['reaction_id'].nunique(),'reactions_with_cage':g.loc[g['has_cage'],'reaction_id'].nunique(),
            'positive_rows':int(g['label'].sum()),'positive_rows_with_cage':int(g.loc[g['has_cage'],'label'].sum())
        })
    Cov=pd.DataFrame(coverage)
    Cov.to_csv(OUT/'fair_cage_coverage_by_gate.csv',index=False)
    summary={
        'n_cage_scores':int(len(cage)),
        'n_cage_reactions':int(cage['reaction_id'].nunique()),
        'n_cage_positive_pairs':int(cage['label'].sum()),
        'selected_gates':SELECT_GATES,
        'outputs': {
            'scores': str(OUT/'fair_cage_all_scores.csv'),
            'metrics': str(OUT/'fair_cage_reaction_only_metrics.csv'),
            'comparison': str(OUT/'fair_cage_reaction_only_comparison.csv'),
            'coverage': str(OUT/'fair_cage_coverage_by_gate.csv')
        }
    }
    (OUT/'fair_cage_reaction_only_summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False))
    print(json.dumps(summary,indent=2,ensure_ascii=False))
    print('\nTOP all513 B=10')
    print(R[(R.scope=='all513')&(R.B==10)].sort_values(['hit_probability','expected_hits'],ascending=False).head(20).to_string(index=False))
    print('\nCOMPARISON all513')
    print(Comp[Comp.scope.eq('all513')].sort_values(['B','delta_hit_vs_rxn','delta_expected_vs_rxn'],ascending=[True,False,False]).to_string(index=False))

if __name__=='__main__':
    eval_reaction_only()
