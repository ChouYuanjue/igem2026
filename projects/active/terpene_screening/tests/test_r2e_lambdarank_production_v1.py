import hashlib
import json
from pathlib import Path

import numpy as np
import yaml

from projects.active.terpene_screening.core.engine import payload_to_argv
from projects.active.terpene_screening.core.routing import resolve_route
from projects.active.terpene_screening import rank_open_world
from projects.active.terpene_screening.r2e_lambdarank_runtime import (
    build_features,
    full_order,
    lexical_rank,
    load_ranker,
)
from projects.active.terpene_screening.run_r2e_lambdarank_fusion_v1 import (
    _build_features as training_build_features,
    _full_order as training_full_order,
    _lexical_rank as training_lexical_rank,
)

ROOT=Path(__file__).resolve().parents[4]
CANDIDATE=ROOT/'configs/production_routes/terpene_lambdarank_candidate_v1.yaml'
DEFAULT=ROOT/'configs/production_routes/terpene_v1.yaml'
LEGACY_V3=ROOT/'configs/production_routes/terpene_similarity_router_v3.yaml'
BUNDLE=ROOT/'results/catalyst_clean_mainline_v1/r2e_lambdarank_fusion_v1'
EXPECTED='86b6fc7ff43fe1c59916dc6692cb38f513c877e1beed2c88902f00909cb7bb6e'


def test_packaged_ranker_is_exact_confirmed_artifact():
    assert hashlib.sha256((BUNDLE/'ranker.json').read_bytes()).hexdigest()==EXPECTED
    config=json.loads((BUNDLE/'config.json').read_text())
    assert config['config_id']=='cfg_07_392fe119' and config['pool_k']==100 and config['prefix_k']==100
    booster,loaded=load_ranker(str(BUNDLE),EXPECTED)
    assert booster is not None and loaded==config


def test_runtime_feature_contract_matches_frozen_training_implementation_exactly():
    rng=np.random.default_rng(20260902); n=257
    ids=[f'P{v:04d}' for v in rng.permutation(n)]
    p=rng.normal(size=n).astype(np.float32); s=rng.normal(size=n).astype(np.float32)
    lex=lexical_rank(ids); train_lex=training_lexical_rank(ids)
    assert np.array_equal(lex,train_lex)
    po,pr=full_order(p,lex); so,sr=full_order(s,lex)
    tpo,tpr=training_full_order(p,train_lex); tso,tsr=training_full_order(s,train_lex)
    assert np.array_equal(po,tpo) and np.array_equal(so,tso) and np.array_equal(pr,tpr) and np.array_equal(sr,tsr)
    rows=np.unique(np.concatenate([po[:100],so[:100]])).astype(np.int32)
    for fallback in [False,True]:
        a=build_features(p,s,rows,pr,sr,fallback,.731)
        b=training_build_features(p,s,rows,tpr,tsr,fallback,.731)
        assert np.array_equal(a,b)


def test_candidate_manifest_enables_fusion_only_in_confirmed_general_scope():
    args=rank_open_world.build_parser().parse_args(payload_to_argv(
        'rank-enzymes',
        {'reaction_smiles':'CC>>CO','candidate_universe':'general_merged','top_k':10,'route_manifest':str(CANDIDATE)},
        allow_overrides=True,
    ))
    route=resolve_route(direction='reaction_to_enzyme',objective='top10',is_current=False,manifest_path=CANDIDATE)
    spec=rank_open_world._r2e_lambdarank_fusion_spec(args,route)
    assert spec is not None and spec['config_id']=='cfg_07_392fe119'
    assert route.model_bundle_version=='catalyst-r2e-lambdarank-fusion-v1'
    assert route.route_version=='terpene-production-routes-v4-lambdarank-candidate'

    subset_args=rank_open_world.build_parser().parse_args(payload_to_argv(
        'rank-enzymes',
        {'reaction_smiles':'CC>>CO','candidate_universe':'general_merged','candidate_ids':['P1'],'route_manifest':str(CANDIDATE)},
        allow_overrides=True,
    ))
    assert rank_open_world._r2e_lambdarank_fusion_spec(subset_args,route) is None


def test_default_v3_manifest_does_not_activate_learned_fusion():
    args=rank_open_world.build_parser().parse_args(payload_to_argv(
        'rank-enzymes',{'reaction_smiles':'CC>>CO','candidate_universe':'general_merged','top_k':10}
    ))
    route=resolve_route(direction='reaction_to_enzyme',objective='top10',is_current=False,manifest_path=LEGACY_V3)
    assert rank_open_world._r2e_lambdarank_fusion_spec(args,route) is None
    assert route.model_bundle_version=='catalyst-r2e-clean-center-router-v1'
    assert route.route_version=='terpene-production-routes-v3'


def test_candidate_manifest_retains_existing_similarity_router_for_ineligible_scope():
    payload=yaml.safe_load(CANDIDATE.read_text())
    for objective,spec in payload['routes']['reaction_to_enzyme']['external'].items():
        assert spec['lambdarank_fusion']['config_id']=='cfg_07_392fe119'
        assert 'similarity_model_router' in spec
        assert spec['similarity_model_router']['threshold']==.9


def test_default_v5_manifest_retains_promoted_r2e_learned_fusion():
    args=rank_open_world.build_parser().parse_args(payload_to_argv('rank-enzymes',{'reaction_smiles':'CC>>CO','candidate_universe':'general_merged','top_k':10}))
    route=resolve_route(direction='reaction_to_enzyme',objective='top10',is_current=False,manifest_path=DEFAULT)
    spec=rank_open_world._r2e_lambdarank_fusion_spec(args,route)
    assert spec is not None and spec['config_id']=='cfg_07_392fe119'
    assert route.route_version=='terpene-production-routes-v5'
    assert route.model_bundle_version=='catalyst-r2e-lambdarank-fusion-v1'
