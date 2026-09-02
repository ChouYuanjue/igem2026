from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from projects.active.terpene_screening.core.engine import RetrievalEngine
from projects.active.terpene_screening.e2r_anchored_lambdamart_runtime import AnchoredE2RRuntime

ROOT=Path(__file__).resolve().parents[3]
PROTOCOL=ROOT/'projects/active/terpene_screening/CATALYST_E2R_ANCHORED_LAMBDAMART_V3_PRODUCTION.json'
OUT=ROOT/'results/catalyst_clean_mainline_v1/e2r_anchored_lambdamart_v3/runtime_gate'
GENERAL=ROOT/'data/catalyst_candidate_universes/general_merged/proteins/entries.csv'
OLD_SCHEMA=ROOT/'results/terpene_production_models/marts_adapted_drfp_pu_e2r/feature_schema.json'
SALT='e2r-v3-production-runtime-gate-v1'
N_QUERIES=12


def _hash(value:str)->str:
    return hashlib.blake2b(f'{SALT}|{value}'.encode(),digest_size=16).hexdigest()


def fixed_query_ids()->list[str]:
    general=pd.read_csv(GENERAL,dtype={'Entry':str}).sort_values('row').Entry.astype(str).tolist()
    schema=json.loads(OLD_SCHEMA.read_text())
    entries=Path(str(schema['protein_ids_file']))
    old_current=set(pd.read_csv(entries,dtype={'Entry':str}).Entry.astype(str))
    eligible=[value for value in general if value not in old_current]
    chosen=sorted(eligible,key=lambda value:(_hash(value),value))[:N_QUERIES]
    if len(chosen)!=N_QUERIES: raise RuntimeError('not enough fixed gate queries')
    return chosen


def rss_bytes()->int:
    with open('/proc/self/statm') as handle:
        resident=int(handle.read().split()[1])
    return resident*os.sysconf('SC_PAGE_SIZE')


def p95(values:list[float])->float:
    return float(np.quantile(np.asarray(values,dtype=float),0.95,method='linear'))


def main()->None:
    ap=argparse.ArgumentParser(); ap.add_argument('--device',default='cuda'); a=ap.parse_args()
    p=json.loads(PROTOCOL.read_text()); gate=p['production_gates']; qids=fixed_query_ids(); OUT.mkdir(parents=True,exist_ok=True)
    engine=RetrievalEngine()
    base_times={key:[] for key in ['top3','top10','top20']}
    # Prewarm old route once per objective; timing begins only after all old caches are resident.
    for objective,k in [('top3',3),('top10',10),('top20',20)]:
        engine.rank_frame('rank-reactions',{'enzyme_id':qids[0],'candidate_universe':'general_merged','top_k':k,'ranking_objective':objective,'conformal_mode':'disabled','device':a.device})
    old_warm_rss=rss_bytes(); old_warm_gpu=int(torch.cuda.memory_reserved()) if torch.cuda.is_available() else 0
    old_top_ids={}
    for objective,k in [('top3',3),('top10',10),('top20',20)]:
        for qid in qids:
            t=time.perf_counter(); frame=engine.rank_frame('rank-reactions',{'enzyme_id':qid,'candidate_universe':'general_merged','top_k':k,'ranking_objective':objective,'conformal_mode':'disabled','device':a.device}); base_times[objective].append(time.perf_counter()-t)
            old_top_ids[f'{objective}|{qid}']=frame.candidate_id.astype(str).tolist()
    t=time.perf_counter(); runtime=AnchoredE2RRuntime(device=a.device); candidate_init=time.perf_counter()-t
    candidate_warm_rss=rss_bytes(); candidate_warm_gpu=int(torch.cuda.memory_reserved()) if torch.cuda.is_available() else 0
    # One untimed pass fully warms the candidate query/ranker path.
    for qid in qids: runtime.rank_registered(qid)
    candidate_times=[]; first={}
    for repeat in range(2):
        for qid in qids:
            t=time.perf_counter(); result=runtime.rank_registered(qid); candidate_times.append(time.perf_counter()-t)
            ids=result.top_ids(20)
            if repeat==0: first[qid]=ids
            elif ids!=first[qid]: raise AssertionError(f'non-deterministic candidate Top20 for {qid}')
    base_summary={key:{'median_s':float(statistics.median(vals)),'p95_s':p95(vals)} for key,vals in base_times.items()}
    min_base_median=min(v['median_s'] for v in base_summary.values()); min_base_p95=min(v['p95_s'] for v in base_summary.values())
    cand_median=float(statistics.median(candidate_times)); cand_p95=p95(candidate_times)
    median_ratio=cand_median/min_base_median; p95_ratio=cand_p95/min_base_p95
    rss_delta=max(0,candidate_warm_rss-old_warm_rss); gpu_delta=max(0,candidate_warm_gpu-old_warm_gpu)
    checks={
      'warm_median_ratio':median_ratio<=float(gate['latency']['registered_query_warm_median_ratio_vs_existing_route_max']),
      'warm_p95_ratio':p95_ratio<=float(gate['latency']['registered_query_warm_p95_ratio_vs_existing_route_max']),
      'rss_delta':rss_delta<=float(gate['memory']['incremental_rss_gib_max'])*(1024**3),
      'gpu_reserved_delta':gpu_delta<=float(gate['memory']['incremental_gpu_reserved_gib_max'])*(1024**3),
      'determinism':True,
    }
    result={
      'status':'pass' if all(checks.values()) else 'fail',
      'query_selection':{'salt':SALT,'rule':'12 smallest blake2b(salt|protein_id) among general_merged IDs excluding old current-training IDs; no labels','query_ids':qids},
      'existing_route':base_summary,
      'candidate':{'init_s':candidate_init,'warm_median_s':cand_median,'warm_p95_s':cand_p95,'timed_calls':len(candidate_times)},
      'ratios_vs_fastest_existing_objective':{'median':median_ratio,'p95':p95_ratio},
      'memory':{'old_warm_rss_bytes':old_warm_rss,'candidate_warm_rss_bytes':candidate_warm_rss,'incremental_rss_bytes':rss_delta,'old_warm_gpu_reserved_bytes':old_warm_gpu,'candidate_warm_gpu_reserved_bytes':candidate_warm_gpu,'incremental_gpu_reserved_bytes':gpu_delta},
      'checks':checks,
      'candidate_top20':first,
      'existing_top_ids':old_top_ids,
      'external_labels_used':False,
    }
    (OUT/'result.json').write_text(json.dumps(result,indent=2)+'\n'); print(json.dumps(result,indent=2))

if __name__=='__main__': main()
