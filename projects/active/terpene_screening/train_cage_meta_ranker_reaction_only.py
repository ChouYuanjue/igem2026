from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / 'results' / 'terpene_cage_fair' / 'meta_ranker'
OUT.mkdir(parents=True, exist_ok=True)
GATE_POOL = ROOT / 'data' / 'terpene_gate_matrix' / 'gate_candidate_pools_with_evidence.csv'
if not GATE_POOL.exists():
    GATE_POOL = ROOT / 'data' / 'terpene_gate_matrix' / 'gate_candidate_pools.csv'
CAGE = ROOT / 'results' / 'terpene_cage_fair' / 'fair_cage_all_scores.csv'
GATE_REACTION = ROOT / 'results' / 'terpene_gate_matrix' / 'gate_reaction_level.csv'
SELECT_GATES = ['rxn_balanced_top20', 'rxn_balanced_top50', 'weighted_top100', 'recall_union_core']
BUDGETS = [5, 10, 20]
SEED = 20260707
N_SPLITS = 5

NUMERIC_COLS = [
    'gate_score', 'reaction_similarity', 'sequence_kmer', 'motif_score',
    'precursor_match', 'product_skeleton', 'mechanism', 'evidence_channels',
    'cage_score_fill0', 'cage_rank_score_fill0', 'has_cage',
    'rxn_x_cage', 'seq_x_cage', 'motif_x_cage', 'evidence_x_cage',
    'rxn_minus_cage', 'cage_top_80', 'cage_top_90', 'cage_top_95',
]
CAT_COLS = ['gate_id']


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    for col in ['label', 'gate_score', 'reaction_similarity', 'sequence_kmer', 'motif_score', 'precursor_match', 'product_skeleton', 'mechanism', 'evidence_channels']:
        if col not in df.columns:
            df[col] = 0
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    df['has_cage'] = df['cage_rank_score'].notna().astype(float)
    df['cage_score_fill0'] = pd.to_numeric(df['cage_score'], errors='coerce').fillna(0)
    df['cage_rank_score_fill0'] = pd.to_numeric(df['cage_rank_score'], errors='coerce').fillna(0)
    df['fusion_no_cage'] = 0.35 * df['reaction_similarity'] + 0.25 * df['sequence_kmer'] + 0.15 * df['precursor_match'] + 0.10 * df['product_skeleton'] + 0.10 * df['motif_score']
    df['rxn_x_cage'] = df['reaction_similarity'] * df['cage_rank_score_fill0']
    df['seq_x_cage'] = df['sequence_kmer'] * df['cage_rank_score_fill0']
    df['motif_x_cage'] = df['motif_score'] * df['cage_rank_score_fill0']
    df['evidence_x_cage'] = df['evidence_channels'] * df['cage_rank_score_fill0']
    df['rxn_minus_cage'] = df['reaction_similarity'] - df['cage_rank_score_fill0']
    df['cage_top_80'] = (df['cage_rank_score_fill0'] >= 0.80).astype(float)
    df['cage_top_90'] = (df['cage_rank_score_fill0'] >= 0.90).astype(float)
    df['cage_top_95'] = (df['cage_rank_score_fill0'] >= 0.95).astype(float)
    return df


def make_data() -> pd.DataFrame:
    gate = pd.read_csv(GATE_POOL, dtype=str).fillna('')
    gate = gate[gate['gate_id'].isin(SELECT_GATES)].copy()
    cage = pd.read_csv(CAGE, dtype=str).fillna('')
    cage['cage_score'] = pd.to_numeric(cage['cage_score'], errors='coerce')
    cage['cage_rank_score'] = pd.to_numeric(cage['cage_rank_score'], errors='coerce')
    cage = cage[['reaction_id', 'uniprot_id', 'cage_score', 'cage_rank_score']].drop_duplicates(['reaction_id', 'uniprot_id'])
    df = gate.merge(cage, on=['reaction_id', 'uniprot_id'], how='left')
    return add_features(df)


def build_model(name: str, pos_weight: float):
    numeric_pipe = Pipeline([('impute', SimpleImputer(strategy='median')), ('scale', StandardScaler())])
    pre = ColumnTransformer([
        ('num', numeric_pipe, NUMERIC_COLS),
        ('cat', OneHotEncoder(handle_unknown='ignore'), CAT_COLS),
    ])
    if name == 'logreg':
        clf = LogisticRegression(max_iter=1000, class_weight='balanced', C=0.3, n_jobs=4, random_state=SEED)
    elif name == 'hgb':
        clf = HistGradientBoostingClassifier(max_iter=220, learning_rate=0.04, max_leaf_nodes=15, min_samples_leaf=30, l2_regularization=0.08, random_state=SEED)
    elif name == 'rf':
        clf = RandomForestClassifier(n_estimators=250, max_depth=14, min_samples_leaf=15, class_weight='balanced_subsample', n_jobs=8, random_state=SEED)
    elif name == 'extratrees':
        clf = ExtraTreesClassifier(n_estimators=350, max_depth=14, min_samples_leaf=15, class_weight='balanced_subsample', n_jobs=8, random_state=SEED)
    else:
        raise ValueError(name)
    return Pipeline([('pre', pre), ('clf', clf)])


def cross_fit(df: pd.DataFrame, model_name: str) -> pd.Series:
    X = df[NUMERIC_COLS + CAT_COLS]
    y = df['label'].astype(int).values
    groups = df['reaction_id'].astype(str).values
    oof = np.zeros(len(df), dtype=float)
    pos = max(1, int(y.sum()))
    neg = max(1, len(y) - pos)
    pos_weight = min(100.0, neg / pos)
    gkf = GroupKFold(n_splits=N_SPLITS)
    for fold, (tr, te) in enumerate(gkf.split(X, y, groups), start=1):
        model = build_model(model_name, pos_weight)
        if model_name == 'hgb':
            weights = np.ones(len(tr), dtype=float)
            weights[y[tr] == 1] = pos_weight
            model.fit(X.iloc[tr], y[tr], clf__sample_weight=weights)
        else:
            model.fit(X.iloc[tr], y[tr])
        oof[te] = model.predict_proba(X.iloc[te])[:, 1]
        print(f'{model_name} fold={fold} test_rows={len(te)} test_pos={int(y[te].sum())}', flush=True)
    return pd.Series(oof, index=df.index)


def add_group_rank(df: pd.DataFrame, col: str, out_col: str) -> None:
    rank = df.groupby(['gate_id', 'reaction_id'])[col].rank(method='first', ascending=False)
    n = df.groupby(['gate_id', 'reaction_id'])[col].transform('size')
    df[out_col] = 1 - (rank - 1) / (n - 1).replace(0, 1)


def hit_metrics(top: pd.DataFrame, universe: set[str], B: int) -> tuple[float, float, float]:
    hits = top.groupby('reaction_id')['label'].sum().reindex(pd.Index(sorted(universe)), fill_value=0)
    expected = float(hits.mean())
    return float((hits > 0).mean()), expected, expected / B


def top_by_score(df: pd.DataFrame, score_col: str, B: int) -> pd.DataFrame:
    return df.sort_values(['reaction_id', score_col, 'uniprot_id'], ascending=[True, False, True]).groupby('reaction_id', sort=False).head(B)


def top_rescue(df: pd.DataFrame, main_col: str, rescue_col: str, B: int, k: int) -> pd.DataFrame:
    panels = []
    for _, g in df.groupby('reaction_id', sort=False):
        main = g.sort_values([main_col, 'uniprot_id'], ascending=[False, True]).head(max(0, B-k))
        used = set(main['uniprot_id'])
        rescue = g[~g['uniprot_id'].isin(used)].sort_values([rescue_col, 'uniprot_id'], ascending=[False, True]).head(k)
        panels.append(pd.concat([main, rescue], ignore_index=True))
    return pd.concat(panels, ignore_index=True) if panels else pd.DataFrame()


def evaluate(df: pd.DataFrame, score_cols: dict[str, str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    all_rxns = set(pd.read_csv(GATE_REACTION, usecols=['reaction_id'], dtype=str)['reaction_id'])
    rows = []
    for gate, g in df.groupby('gate_id'):
        gate_rxns = set(g['reaction_id'].astype(str))
        for scope, universe in [('all513', all_rxns), ('gate_rxns', gate_rxns)]:
            for B in BUDGETS:
                for method, col in score_cols.items():
                    top = top_by_score(g, col, B)
                    hp, eh, pr = hit_metrics(top, universe, B)
                    rows.append({'gate_id': gate, 'scope': scope, 'method': method, 'B': B, 'n_reactions': len(universe), 'hit_probability': hp, 'expected_hits': eh, 'precision': pr})
                # explicit rescue panels: main reaction ranking plus learned model slots
                for model_name in ['logreg', 'hgb', 'rf', 'extratrees']:
                    for k in ([1, 2] if B == 5 else ([2, 5] if B == 10 else [2, 5, 10])):
                        top = top_rescue(g, 'reaction_similarity', f'oof_{model_name}_rank', B, k)
                        hp, eh, pr = hit_metrics(top, universe, B)
                        rows.append({'gate_id': gate, 'scope': scope, 'method': f'rxn_main_plus_{model_name}_rescue_k{k}', 'B': B, 'n_reactions': len(universe), 'hit_probability': hp, 'expected_hits': eh, 'precision': pr})
    metrics = pd.DataFrame(rows)
    comp = []
    for (gate, scope, B), grp in metrics.groupby(['gate_id', 'scope', 'B']):
        base = grp[grp['method'].eq('reaction_similarity')].iloc[0]
        best = grp.sort_values(['hit_probability', 'expected_hits', 'precision'], ascending=False).iloc[0]
        learned_grp = grp[~grp['method'].isin(['reaction_similarity', 'fusion_no_cage', 'cage_rank_only'])]
        best_learned = learned_grp.sort_values(['hit_probability', 'expected_hits', 'precision'], ascending=False).iloc[0]
        comp.append({
            'gate_id': gate, 'scope': scope, 'B': B,
            'rxn_hit': base.hit_probability, 'rxn_expected': base.expected_hits,
            'best_method': best.method, 'best_hit': best.hit_probability, 'best_expected': best.expected_hits,
            'best_learned_method': best_learned.method, 'best_learned_hit': best_learned.hit_probability, 'best_learned_expected': best_learned.expected_hits,
            'delta_hit_vs_rxn': best_learned.hit_probability - base.hit_probability,
            'delta_expected_vs_rxn': best_learned.expected_hits - base.expected_hits,
        })
    return metrics, pd.DataFrame(comp)


def main() -> None:
    df = make_data()
    print('data rows', len(df), 'reactions', df['reaction_id'].nunique(), 'positives', int(df['label'].sum()), flush=True)
    model_names = ['logreg', 'hgb', 'rf', 'extratrees']
    for name in model_names:
        df[f'oof_{name}'] = cross_fit(df, name)
        add_group_rank(df, f'oof_{name}', f'oof_{name}_rank')
    score_cols = {
        'reaction_similarity': 'reaction_similarity',
        'fusion_no_cage': 'fusion_no_cage',
        'cage_rank_only': 'cage_rank_score_fill0',
    }
    for name in model_names:
        score_cols[f'{name}_only'] = f'oof_{name}'
        score_cols[f'{name}_rank_only'] = f'oof_{name}_rank'
        for alpha in [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.10, 0.20, 0.50]:
            df[f'rxn_plus_{name}_{alpha}'] = df['reaction_similarity'] + alpha * df[f'oof_{name}_rank']
            df[f'fusion_plus_{name}_{alpha}'] = df['fusion_no_cage'] + alpha * df[f'oof_{name}_rank']
            score_cols[f'rxn_plus_{name}_{alpha}'] = f'rxn_plus_{name}_{alpha}'
            score_cols[f'fusion_plus_{name}_{alpha}'] = f'fusion_plus_{name}_{alpha}'
    df.to_csv(OUT / 'reaction_only_oof_meta_scores.csv', index=False)
    metrics, comp = evaluate(df, score_cols)
    metrics.to_csv(OUT / 'reaction_only_meta_ranker_metrics.csv', index=False)
    comp.to_csv(OUT / 'reaction_only_meta_ranker_comparison.csv', index=False)
    summary = {
        'n_rows': int(len(df)),
        'n_reactions': int(df['reaction_id'].nunique()),
        'n_positives': int(df['label'].sum()),
        'models': model_names,
        'outputs': {
            'oof_scores': str(OUT / 'reaction_only_oof_meta_scores.csv'),
            'metrics': str(OUT / 'reaction_only_meta_ranker_metrics.csv'),
            'comparison': str(OUT / 'reaction_only_meta_ranker_comparison.csv'),
        },
    }
    (OUT / 'reaction_only_meta_ranker_summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    print('\nTOP all513 B=10')
    print(metrics[(metrics.scope.eq('all513')) & (metrics.B.eq(10))].sort_values(['hit_probability','expected_hits'], ascending=False).head(25).to_string(index=False))
    print('\nCOMPARISON all513')
    print(comp[comp.scope.eq('all513')].sort_values(['B','delta_hit_vs_rxn','delta_expected_vs_rxn'], ascending=[True, False, False]).to_string(index=False))

if __name__ == '__main__':
    main()
