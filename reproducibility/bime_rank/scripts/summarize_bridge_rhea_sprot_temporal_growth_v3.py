from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
BENCH = ROOT / 'results/bridge_rhea_sprot_temporal_growth_v3'
R2E = ROOT / 'results/bridge_rhea_sprot_temporal_r2e_v3/edge_metrics.csv.gz'
E2R = ROOT / 'results/bridge_rhea_sprot_temporal_e2r_v3/edge_metrics.csv.gz'
BROAD_E2R = ROOT / 'results/bridge_rhea_sprot_temporal_broad_v3/e2r_edges.csv.gz'
CAGE = ROOT / 'results/bridge_rhea_sprot_temporal_cage_gate_v3/summary.json'
OUT = ROOT / 'reproducibility/bime_rank/records/BRIDGE_RHEA_SPROT_TEMPORAL_GROWTH_V3_RESULT.json'


def metrics(frame: pd.DataFrame, col: str) -> dict[str, float | int]:
    r = frame[col].astype(int).to_numpy()
    return {'edges': int(len(r)), 'mrr': float((1 / r).mean()), 'hit10': float((r <= 10).mean()), 'hit100': float((r <= 100).mean()), 'hit1000': float((r <= 1000).mean()), 'median_rank': float(np.median(r))}


def bootstrap_delta(frame: pd.DataFrame, seed: int) -> dict[str, object]:
    rng = np.random.default_rng(seed)
    b = 1 / frame.broad_rank.astype(int).to_numpy(); f = 1 / frame.full_rank.astype(int).to_numpy(); delta = f - b; n = len(delta)
    samples = np.empty(10000, dtype=np.float64)
    for i in range(len(samples)):
        samples[i] = delta[rng.integers(0, n, n)].mean()
    return {'delta_mrr': float(delta.mean()), 'bootstrap_replicates': 10000, 'ci95': [float(np.quantile(samples, .025)), float(np.quantile(samples, .975))], 'bootstrap_nonpositive_fraction': float((samples <= 0).mean())}


def block(frame: pd.DataFrame, seed: int) -> dict[str, object]:
    return {'broad': metrics(frame, 'broad_rank'), 'bridge': metrics(frame, 'full_rank'), 'paired_gain': bootstrap_delta(frame, seed)}


def main() -> None:
    bench = json.load(open(BENCH / 'summary.json'))
    target = pd.read_csv(BENCH / 'targets.csv', dtype=str)[['protein_id','reaction_id','growth_class','orientation','first_observed_release']]
    r2e = pd.read_csv(R2E, dtype=str).merge(target, on=['protein_id','reaction_id'], validate='one_to_one')
    e2r = pd.read_csv(E2R, dtype=str).merge(target, on=['protein_id','reaction_id'], validate='one_to_one')
    broad_e2r = pd.read_csv(BROAD_E2R, dtype=str)[['protein_id','reaction_id','rank']].rename(columns={'rank':'broad_rank'})
    e2r = e2r.merge(broad_e2r, on=['protein_id','reaction_id'], validate='one_to_one')
    cage = json.load(open(CAGE))

    result = {
        'schema': 'bridge-rhea-sprot-temporal-growth-v3-result',
        'status': 'completed',
        'headline_protocol': {
            'historical_source': 'official Rhea Swiss-Prot mappings, release128 -> release142',
            'baseline_release': bench['baseline_release'], 'final_release': bench['final_release'],
            'candidate_universe': {'r2e_proteins': 185918, 'e2r_reactions': 11081},
            'evaluation_unit': 'each future relation edge; filtered full-candidate rank',
            'graph_completion': 'both protein and reaction already exist in release128; relation appears later',
            'graph_expansion': 'at least one endpoint does not exist in release128',
            'clean2023_target_pair_overlap': 0,
            'selection_or_tuning_on_temporal_labels': False,
        },
        'support': bench['targets'],
        'headline': {},
        'expansion_diagnostic': {},
        'temporal_stability': {},
        'cage_candidate_generation': cage,
        'interpretation': {
            'primary': 'Historical release deltas separate fixed-node relation completion from open graph expansion without changing the candidate universe.',
            'cage_boundary': 'CAGE Top-10 similar-reaction retrieval retains substantial recall for graph completion but loses most future positives when new protein entities enter the graph; new reactions remain partially transferable through reaction similarity.',
            'difficulty_boundary': 'Graph completion and graph expansion are growth mechanisms, not a guaranteed scalar difficulty ordering; direction and which endpoint is new still matter.',
        },
        'reproduction': {
            'build_benchmark': 'reproducibility/bime_rank/scripts/build_bridge_rhea_sprot_temporal_growth_v3.py',
            'broad_evaluator': 'reproducibility/bime_rank/scripts/evaluate_bridge_relation_unseen_edgewise_broad_v1.py',
            'r2e_evaluator': 'reproducibility/bime_rank/scripts/evaluate_bridge_r2e_edgewise_v13.py',
            'e2r_evaluator': 'reproducibility/bime_rank/scripts/evaluate_bridge_e2r_edgewise_v4.py',
            'cage_gate': 'reproducibility/bime_rank/scripts/evaluate_bridge_rhea_sprot_temporal_cage_gate_v3.py',
            'summarize': 'reproducibility/bime_rank/scripts/summarize_bridge_rhea_sprot_temporal_growth_v3.py',
        },
    }
    for i, name in enumerate(('graph_completion', 'graph_expansion')):
        if name == 'graph_completion':
            rmask = r2e.growth_class.eq('edge_completion'); emask = e2r.growth_class.eq('edge_completion')
        else:
            rmask = r2e.growth_class.ne('edge_completion'); emask = e2r.growth_class.ne('edge_completion')
        rblock = r2e[rmask].copy(); eblock = e2r[emask].copy()
        rkeys = set(map(tuple, rblock[['protein_id','reaction_id']].to_numpy()))
        ekeys = set(map(tuple, eblock[['protein_id','reaction_id']].to_numpy()))
        if rkeys != ekeys:
            raise AssertionError(f'{name} R2E/E2R edge-set mismatch: {len(rkeys)} vs {len(ekeys)}')
        result['headline'][name] = {'r2e': block(rblock, 20261005 + i), 'e2r': block(eblock, 20261105 + i), 'cage_gate': cage['headline'][name]}
    for name, cls in [('single_endpoint_expansion','single_endpoint_expansion'), ('double_endpoint_expansion','double_endpoint_expansion')]:
        result['expansion_diagnostic'][name] = {'edges': int((r2e.growth_class == cls).sum()), 'r2e': {'broad': metrics(r2e[r2e.growth_class == cls], 'broad_rank'), 'bridge': metrics(r2e[r2e.growth_class == cls], 'full_rank')}, 'e2r': {'broad': metrics(e2r[e2r.growth_class == cls], 'broad_rank'), 'bridge': metrics(e2r[e2r.growth_class == cls], 'full_rank')}, 'cage_gate': cage['by_growth_class'][cls]}
    result['expansion_diagnostic']['orientation'] = cage['by_orientation']
    for release in sorted(set(r2e.first_observed_release)):
        rm = r2e.first_observed_release.eq(release) & r2e.growth_class.ne('edge_completion'); em = e2r.first_observed_release.eq(release) & e2r.growth_class.ne('edge_completion')
        result['temporal_stability'][str(release)] = {'expansion_edges': int(rm.sum()), 'r2e_broad': metrics(r2e[rm], 'broad_rank'), 'r2e_bridge': metrics(r2e[rm], 'full_rank'), 'e2r_broad': metrics(e2r[em], 'broad_rank'), 'e2r_bridge': metrics(e2r[em], 'full_rank')}
    OUT.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
