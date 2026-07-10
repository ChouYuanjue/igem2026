from __future__ import annotations

import json, random, re
from collections import defaultdict
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
OUT=ROOT/'results/terpene_cage_fair'
POS=ROOT/'data/terpene/enzyme_terpene_synthase.tsv'
CAND=ROOT/'data/terpene/all_seq_terpene_synthase.tsv'
GATE=ROOT/'data/terpene_gate_matrix/gate_candidate_pools.csv'
CAGE=OUT/'fair_cage_all_scores.csv'
BUDGETS=[5,10,20]
M_VALUES=[1,2,3,5]
REPEATS=20
SEED=20260707

def kmers(seq,k=3):
    seq=re.sub('[^A-Z]','',str(seq).upper())
    return {seq[i:i+k] for i in range(len(seq)-k+1)} if len(seq)>=k else set()

def build_seq_index(seq):
    ck={u:kmers(s) for u,s in seq.items()}
    sz={u:len(v) for u,v in ck.items()}
    inv=defaultdict(set)
    for u,ks in ck.items():
        for k in ks: inv[k].add(u)
    return ck,sz,inv

def seed_scores(seed_ids, ck, sz, inv):
    best=defaultdict(float)
    for sid in seed_ids:
        sk=ck.get(sid,set())
        if not sk: continue
        counts=defaultdict(int)
        for k in sk:
            for u in inv.get(k,()): counts[u]+=1
        ss=len(sk)
        for u,inter in counts.items():
            den=ss+sz.get(u,0)-inter
            if den>0:
                sc=inter/den
                if sc>best[u]: best[u]=sc
    return dict(best)

def reciprocal_rank_score(ids, weight=1.0):
    return {u: weight/(i+1) for i,u in enumerate(ids)}

def pool_ids(pool, gate, rid):
    g=pool.get(gate,{}).get(rid)
    if g is None: return []
    return g.sort_values(['reaction_similarity','gate_score','uniprot_id'],ascending=[False,False,True])['uniprot_id'].astype(str).tolist()

def rank_union(name, components, seed_sc=None, cage_sc=None, topn=None, cage_alpha=0.0, seed_alpha=0.0, priority_q=None):
    score=defaultdict(float)
    seen=set()
    for weight, ids in components:
        for i,u in enumerate(ids):
            seen.add(u); score[u]+=weight/(i+1)
    if seed_sc and seed_alpha:
        for u in list(seen): score[u]+=seed_alpha*seed_sc.get(u,0)
    if cage_sc and cage_alpha:
        for u in list(seen): score[u]+=cage_alpha*cage_sc.get(u,0)
    if priority_q is not None and cage_sc:
        ordered=sorted(seen,key=lambda u:(-(cage_sc.get(u,0)>=priority_q),-score[u],u))
    else:
        ordered=sorted(seen,key=lambda u:(-score[u],u))
    return ordered[:topn] if topn else ordered

def eval_rank(rows, m, rid, rep, method, ranked, seeds, hidden):
    ranked=[u for u in ranked if u not in seeds]
    for B in BUDGETS:
        top=ranked[:B]
        hits=sum(1 for u in top if u in hidden)
        rows.append({'m':m,'reaction_id':rid,'rep':rep,'method':method,'B':B,'n_hidden':len(hidden),'hits':hits,'hit':hits>0,'precision':hits/B,'hidden_recall':hits/len(hidden) if hidden else 0})

def eval_panel(rows, m, rid, rep, method, main_ranked, cage_ranked, seeds, hidden, B, k):
    main=[u for u in main_ranked if u not in seeds]
    panel=[]; used=set()
    for u in main:
        if len(panel)>=max(0,B-k): break
        if u not in used:
            panel.append(u); used.add(u)
    for u in cage_ranked:
        if len(panel)>=B: break
        if u not in seeds and u not in used:
            panel.append(u); used.add(u)
    hits=sum(1 for u in panel if u in hidden)
    rows.append({'m':m,'reaction_id':rid,'rep':rep,'method':method,'B':B,'n_hidden':len(hidden),'hits':hits,'hit':hits>0,'precision':hits/B,'hidden_recall':hits/len(hidden) if hidden else 0})

def main():
    random.seed(SEED)
    pos=pd.read_csv(POS,sep='\t',dtype=str).fillna('')
    cand=pd.read_csv(CAND,sep='\t',dtype=str).fillna('')
    seq=dict(zip(cand['Entry'].astype(str), cand['Sequence'].astype(str)))
    cand_set=set(seq)
    ck,sz,inv=build_seq_index(seq)
    pos_by_rxn=pos.groupby('rhea_id')['Entry'].apply(lambda s: sorted(set(s.astype(str)) & cand_set)).to_dict()
    G=pd.read_csv(GATE,usecols=['gate_id','reaction_id','uniprot_id','gate_score','reaction_similarity'],dtype=str).fillna('')
    for c in ['gate_score','reaction_similarity']:
        G[c]=pd.to_numeric(G[c],errors='coerce').fillna(0)
    pools={}
    for gate in ['rxn_balanced_top20','rxn_balanced_top50','weighted_top100','recall_union_core']:
        sub=G[G.gate_id.eq(gate)]
        pools[gate]={rid:grp for rid,grp in sub.groupby('reaction_id')}
    cage=pd.read_csv(CAGE,dtype=str).fillna('')
    cage['cage_rank_score']=pd.to_numeric(cage['cage_rank_score'],errors='coerce').fillna(0)
    cage_by_rxn={rid:dict(zip(g['uniprot_id'].astype(str),g['cage_rank_score'])) for rid,g in cage.groupby('reaction_id')}
    cage_rank_by_rxn={rid:g.sort_values(['cage_rank_score','uniprot_id'],ascending=[False,True])['uniprot_id'].astype(str).tolist() for rid,g in cage.groupby('reaction_id')}
    rows=[]; eligible_counts={}
    seed_score_cache={}
    for m in M_VALUES:
        eligible=[rid for rid,ids in pos_by_rxn.items() if len(ids)>=m+1]
        eligible_counts[m]=len(eligible)
        print('m',m,'eligible',len(eligible),flush=True)
        for rid in eligible:
            positives=pos_by_rxn[rid]
            rxn20=pool_ids(pools,'rxn_balanced_top20',rid)
            rxn50=pool_ids(pools,'rxn_balanced_top50',rid)
            weighted=pool_ids(pools,'weighted_top100',rid)
            recall=pool_ids(pools,'recall_union_core',rid)
            cage_sc=cage_by_rxn.get(rid,{})
            cage_rank=cage_rank_by_rxn.get(rid,[])
            for rep in range(REPEATS):
                rng=random.Random(hash((rid,m,rep)) & ((1<<32)-1))
                seeds=tuple(sorted(rng.sample(positives,m)))
                hidden=set(positives)-set(seeds)
                if seeds not in seed_score_cache:
                    seed_score_cache[seeds]=seed_scores(seeds,ck,sz,inv)
                ss=seed_score_cache[seeds]
                seed_order=sorted(ss,key=lambda u:(-ss[u],u))
                # Baselines
                eval_rank(rows,m,rid,rep,'seed_seq_top20',seed_order[:20],set(seeds),hidden)
                eval_rank(rows,m,rid,rep,'seed_seq_top50',seed_order[:50],set(seeds),hidden)
                eval_rank(rows,m,rid,rep,'seed_seq_top100',seed_order[:100],set(seeds),hidden)
                eval_rank(rows,m,rid,rep,'rxn_top20_only',rxn20,set(seeds),hidden)
                seed50_rxn20=rank_union('seed50_union_rxn20',[(1,seed_order[:50]),(1,rxn20)])
                seed100_rxn50=rank_union('seed100_union_rxn50',[(1,seed_order[:100]),(1,rxn50)])
                seed50_recall=rank_union('seed50_union_recall',[(1,seed_order[:50]),(0.7,recall)])
                seeded50=rank_union('seeded_fusion_top50',[(1,seed_order[:100]),(1,weighted),(0.8,rxn50)],seed_sc=ss,seed_alpha=0.5,topn=50)
                seeded100=rank_union('seeded_fusion_top100',[(1,seed_order[:100]),(1,weighted),(0.8,rxn50),(0.5,recall)],seed_sc=ss,seed_alpha=0.5,topn=100)
                for name,ranked in [('seed50_union_rxn20',seed50_rxn20),('seed100_union_rxn50',seed100_rxn50),('seed50_union_recall',seed50_recall),('seeded_fusion_top50',seeded50),('seeded_fusion_top100',seeded100)]:
                    eval_rank(rows,m,rid,rep,name,ranked,set(seeds),hidden)
                # CAGE-only in same union reservoir
                union_res=list(dict.fromkeys(seed_order[:100]+weighted+rxn50+recall+cage_rank))
                cage_only=sorted(union_res,key=lambda u:(-cage_sc.get(u,0),u))
                eval_rank(rows,m,rid,rep,'cage_only_on_union',cage_only,set(seeds),hidden)
                for alpha in [0.01,0.05,0.1,0.2]:
                    ranked=rank_union(f'seeded_fusion100_plus_cage_{alpha}',[(1,seed_order[:100]),(1,weighted),(0.8,rxn50),(0.5,recall)],seed_sc=ss,seed_alpha=0.5,cage_sc=cage_sc,cage_alpha=alpha,topn=100)
                    eval_rank(rows,m,rid,rep,f'seeded_fusion100_plus_cage_{alpha}',ranked,set(seeds),hidden)
                for q in [0.8,0.9,0.95]:
                    ranked=rank_union(f'seeded_fusion100_cage_priority_q{q}',[(1,seed_order[:100]),(1,weighted),(0.8,rxn50),(0.5,recall)],seed_sc=ss,seed_alpha=0.5,cage_sc=cage_sc,priority_q=q,topn=100)
                    eval_rank(rows,m,rid,rep,f'seeded_fusion100_cage_priority_q{q}',ranked,set(seeds),hidden)
                # Rescue panels
                for B in BUDGETS:
                    ks=[1,2] if B==5 else ([2,5] if B==10 else [2,5,10])
                    for k in ks:
                        eval_panel(rows,m,rid,rep,f'seeded_fusion100_plus_cage_rescue_k{k}',seeded100,cage_only,set(seeds),hidden,B,k)
    long=pd.DataFrame(rows)
    long.to_csv(OUT/'fair_cage_fewshot_metrics_long.csv',index=False)
    agg=long.groupby(['m','method','B']).agg(n_trials=('hit','size'),hit_probability=('hit','mean'),expected_hits=('hits','mean'),precision=('precision','mean'),hidden_recall=('hidden_recall','mean')).reset_index()
    agg.to_csv(OUT/'fair_cage_fewshot_metrics.csv',index=False)
    comp=[]
    for (m,B),grp in agg.groupby(['m','B']):
        # Use no-CAGE best from previous known methods in this script as baseline.
        baseline_methods=['seed_seq_top20','seed_seq_top50','seed_seq_top100','rxn_top20_only','seed50_union_rxn20','seed100_union_rxn50','seed50_union_recall','seeded_fusion_top50','seeded_fusion_top100']
        base=grp[grp.method.isin(baseline_methods)].sort_values(['hit_probability','expected_hits'],ascending=False).iloc[0]
        best=grp.sort_values(['hit_probability','expected_hits'],ascending=False).iloc[0]
        cagegrp=grp[~grp.method.isin(baseline_methods)].sort_values(['hit_probability','expected_hits'],ascending=False)
        bestc=cagegrp.iloc[0]
        comp.append({'m':m,'B':B,'best_no_cage_method':base.method,'best_no_cage_hit':base.hit_probability,'best_no_cage_expected':base.expected_hits,'best_method':best.method,'best_hit':best.hit_probability,'best_expected':best.expected_hits,'best_cage_method':bestc.method,'best_cage_hit':bestc.hit_probability,'best_cage_expected':bestc.expected_hits,'delta_hit_vs_no_cage':bestc.hit_probability-base.hit_probability,'delta_expected_vs_no_cage':bestc.expected_hits-base.expected_hits})
    comp=pd.DataFrame(comp)
    comp.to_csv(OUT/'fair_cage_fewshot_comparison.csv',index=False)
    summary={'eligible_counts':eligible_counts,'n_rows_long':len(long),'outputs':{'metrics':str(OUT/'fair_cage_fewshot_metrics.csv'),'comparison':str(OUT/'fair_cage_fewshot_comparison.csv')}}
    (OUT/'fair_cage_fewshot_summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False))
    print(json.dumps(summary,indent=2,ensure_ascii=False))
    print('\nCOMPARISON')
    print(comp.to_string(index=False))
    for m in M_VALUES:
        for B in BUDGETS:
            print('\nTOP m',m,'B',B)
            print(agg[(agg.m==m)&(agg.B==B)].sort_values(['hit_probability','expected_hits'],ascending=False).head(10).to_string(index=False))

if __name__=='__main__':
    main()
