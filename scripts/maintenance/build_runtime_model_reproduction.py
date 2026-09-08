from __future__ import annotations
import hashlib, json, subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'reproducibility/bime_rank/runtime_model_reproduction.json'

def sha(p: Path) -> str:
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''): h.update(b)
    return h.hexdigest()

def tracked() -> set[str]:
    raw=subprocess.check_output(['git','ls-files','-z'],cwd=ROOT)
    return {x.decode() for x in raw.split(b'\0') if x}

def files_under(rel: str) -> list[str]:
    p=ROOT/rel
    if p.is_file(): return [rel]
    return sorted(str(x.relative_to(ROOT)) for x in p.rglob('*') if x.is_file())

release=json.loads((ROOT/'reproducibility/research_release_manifest.json').read_text())
direct={x['path'] for x in release['direct_git_assets']}
rebuild={x['path'] for x in release['rebuildable_assets']}
external={x['target'] for x in release['external_assets']}
tracked_paths=tracked()

def coverage(rel: str) -> str:
    p=ROOT/rel
    if p.is_dir():
        members=files_under(rel)
        if members and all(x in direct and x in tracked_paths for x in members): return 'direct'
    if rel in direct and rel in tracked_paths: return 'direct'
    if rel in rebuild: return 'rebuildable'
    if rel in external: return 'external'
    raise RuntimeError(f'uncovered reproduction input: {rel}')

COMMON=[
 'data/terpene_marts/marts_reaction_pairs.tsv',
 'data/terpene/enzyme_terpene_synthase.tsv',
 'data/terpene_embeddings/esmc600m_mean/entries.csv',
 'data/terpene_embeddings/esmc600m_mean/embeddings.npy',
 'data/terpene_embeddings/marts_unseen_esmc600m/entries.csv',
 'data/terpene_embeddings/marts_unseen_esmc600m/embeddings.npy',
 'data/terpene/all_seq_terpene_synthase.tsv',
 'data/terpene_sequence_clusters/clusters_id50.csv',
 'data/terpene_cold_splits/reaction_cluster_folds.csv',
 'data/terpene_marts_adaptation/protein_entities.csv',
 'data/terpene_marts_adaptation/reaction_entities.csv',
 'results/terpene_production_models/drfp_categorical',
]
ADAPTED='projects/active/terpene_screening/train_marts_adapted_production.py'
EXACT='projects/active/terpene_screening/train_marts_horizyn_exact_residual_production.py'
DUAL='projects/active/terpene_screening/prepare_production_dual_kernel_assets.py'
SOURCE_SHA={
 ADAPTED:'733e93f6a714be87c5e7088147c5e44876a4df0d6b2fa1b479e4453ce7e32660',
 EXACT:'363a8c39df7ef69052f850e2376a09fe84683effd31b364f8f12f1db9d9cdfa1',
 DUAL:'24bb741394a4e6f4839f601adce06a0314e02e74797999ef5bf63293b63614d4',
}
for p,d in SOURCE_SHA.items():
    if sha(ROOT/p)!=d: raise RuntimeError(f'source drift: {p}')

specs=[
 dict(bundle='results/terpene_production_models/marts_adapted_drfp_pu_e2r',kind='trained',source=ADAPTED,
      command='.venv/bin/python projects/active/terpene_screening/train_marts_adapted_production.py --output-dir results/terpene_production_models/marts_adapted_drfp_pu_e2r --pu-group-mask --freeze-reaction-tower',
      expected={'epochs':100,'learning_rate':0.0001,'pu_group_mask':True,'freeze_reaction_tower':True,'n_training_pairs':3439},inputs=COMMON),
 dict(bundle='results/terpene_production_models/marts_adapted_drfp_pu_e2r_hardneg128',kind='trained',source=ADAPTED,
      command='.venv/bin/python projects/active/terpene_screening/train_marts_adapted_production.py --output-dir results/terpene_production_models/marts_adapted_drfp_pu_e2r_hardneg128 --epochs 50 --pu-group-mask --hard-negative-k 128',
      expected={'epochs':50,'learning_rate':0.0001,'pu_group_mask':True,'hard_negative_k':128,'n_training_pairs':3439},inputs=COMMON),
 dict(bundle='results/terpene_production_models/marts_adapted_drfp_pu_r2e075',kind='trained',source=ADAPTED,
      command='.venv/bin/python projects/active/terpene_screening/train_marts_adapted_production.py --output-dir results/terpene_production_models/marts_adapted_drfp_pu_r2e075 --pu-group-mask --reaction-loss-weight 0.75',
      expected={'epochs':100,'learning_rate':0.0001,'pu_group_mask':True,'reaction_loss_weight':0.75,'n_training_pairs':3439},inputs=COMMON),
 dict(bundle='results/terpene_production_models/marts_adapted_drfp_pu_r2e_exact_residual',kind='trained_external_dependent',source=EXACT,
      command='.venv/bin/python projects/active/terpene_screening/train_marts_horizyn_exact_residual_production.py --output-dir results/terpene_production_models/marts_adapted_drfp_pu_r2e_exact_residual --pu-group-mask --horizyn-checkpoint external/horizyn/checkpoints/horizyn_v1_0_dev.ckpt --fallback-distiller results/terpene_horizyn_reaction_feature_distillation/reaction_feature_distiller.pt',
      expected={'epochs':50,'reaction_loss_weight':0.75,'hard_negative_k':0,'pu_group_mask':True,'n_training_pairs':3439},
      inputs=COMMON+['external/horizyn/checkpoints/horizyn_v1_0_dev.ckpt','external/horizyn/configs/sota.yaml','results/terpene_horizyn_reaction_feature_distillation']),
 dict(bundle='results/terpene_production_models/marts_dual_kernel_e2r_top20',kind='deterministic_built',source=DUAL,
      command='.venv/bin/python projects/active/terpene_screening/prepare_production_dual_kernel_assets.py --output-dir results/terpene_production_models/marts_dual_kernel_e2r_top20',
      expected={'reaction_k':50,'protein_k':5,'temperature':0.03,'degree_power':1.0,'n_training_pairs':3439},
      inputs=['results/terpene_production_models/marts_adapted_drfp_pu_e2r','data/terpene_embeddings/esmc600m_mean/entries.csv','data/terpene_embeddings/esmc600m_mean/embeddings.npy','data/terpene_open_world_registry/proteins/entries.csv','data/terpene_open_world_registry/proteins/embeddings.npy']),
]
rows=[]
for s in specs:
    summary=json.loads((ROOT/s['bundle']/'summary.json').read_text())
    for k,v in s['expected'].items():
        if summary.get(k)!=v: raise RuntimeError(f"summary mismatch {s['bundle']} {k}: {summary.get(k)!r} != {v!r}")
    inputs=[{'path':p,'coverage':coverage(p)} for p in s['inputs']]
    outputs=[]
    for rel in files_under(s['bundle']):
        # Lineage output identity must be portable: server-only external copies (for example
        # the >100 MB Horizyn checkpoint copied into one deployment directory) are covered
        # by the external restore contract, not by the Git-reproducible project output set.
        if rel not in tracked_paths:
            continue
        p=ROOT/rel
        outputs.append({'path':rel,'bytes':p.stat().st_size,'sha256':sha(p)})
    rows.append({
      'bundle':s['bundle'],'kind':s['kind'],'source':s['source'],'source_sha256':SOURCE_SHA[s['source']],
      'source_identity':'byte-identical to its first tracked implementation',
      'command':s['command'],'frozen_summary':s['bundle']+'/summary.json','expected_summary_fields':s['expected'],
      'inputs':inputs,'outputs':outputs,
    })
# Current BiME / clean-mainline bundles. Their fixed run scripts and frozen metadata are the
# executable authority; model output identity comes from model_assets.json.
model_index=json.loads((ROOT/'reproducibility/bime_rank/model_assets.json').read_text())
def model_artifacts(bundle: str) -> list[dict[str, object]]:
    out=[]
    for rec in model_index['project_owned_assets']:
        if bundle in rec.get('bundles',[]):
            out.append({k:rec[k] for k in ('path','role','bytes','sha256')})
    if not out: raise RuntimeError(f'no indexed model artifacts for {bundle}')
    return out

def source_record(rel: str) -> dict[str,str]:
    if rel not in tracked_paths or not (ROOT/rel).is_file(): raise RuntimeError(f'untracked lineage source: {rel}')
    return {'path':rel,'sha256':sha(ROOT/rel)}

def upstream_record(rel: str) -> dict[str,str]:
    p=ROOT/rel
    if p.is_file(): return {'path':rel,'coverage':coverage(rel)}
    prefix=rel.rstrip('/')+'/'
    if any(x.startswith(prefix) for x in direct): cov='direct_or_mixed_release_contract'
    elif any(x.startswith(prefix) for x in rebuild): cov='rebuildable_release_contract'
    elif any(x.startswith(prefix) for x in external): cov='external_release_contract'
    else: raise RuntimeError(f'uncovered upstream root: {rel}')
    return {'path':rel,'coverage':cov}

CENTER='projects/active/terpene_screening/train_cleanroom_directional_identity_aux_residual.py'
R2LR='projects/active/terpene_screening/run_r2e_lambdarank_fusion_v1.py'
E2V3='projects/active/terpene_screening/run_e2r_anchored_lambdamart_v3_production_experts.py'
R2CLIP='projects/active/terpene_screening/run_bime_r2e_clipzyme_expert_v1.py'
R2SEED='projects/active/terpene_screening/run_bime_r2e_seed_context_v1.py'
E2V4='projects/active/terpene_screening/run_e2r_clipzyme_anchored_lambdamart_v4.py'
E2SEED='projects/active/terpene_screening/run_bime_e2r_seed_context_v1.py'
current_specs=[
 dict(bundle='results/catalyst_clean_mainline_v1/r2e_center_bounded_cap0p1',kind='trained_from_project_ancestor',sources=[CENTER],
      commands=['.venv/bin/python '+CENTER+' --base-dir results/catalyst_clean_mainline_v1/r2e_base_rdkitplus --training-pairs results/catalyst_clean_mainline_v1/r2e_base_rdkitplus/training_pairs.csv --protein-feature-dir data/catalyst_candidate_universes/general_merged/proteins --reaction-feature-dir data/catalyst_candidate_universes/general_merged/reaction_features/drfp_categorical_rdkitplus_center_v1 --output-dir results/catalyst_clean_mainline_v1/r2e_center_bounded_cap0p1 --direction r2e --dev-fold -1 --max-residual-ratio 0.1 --epochs 2 --steps-per-epoch 60 --learning-rate 3e-5 --weight-decay 1e-4 --temperature 0.07 --batch-size 64 --topk-k 10 --topk-weight 0.1 --topk-margin 0 --all-positive-weight 0.05 --anchor-weight 0.1 --anchor-batch-size 256 --historical-query-repeat 2 --seed 20260723'],
      metadata=['results/catalyst_clean_mainline_v1/r2e_center_bounded_cap0p1/summary.json'],
      upstream=['results/catalyst_clean_mainline_v1/r2e_base_rdkitplus','data/catalyst_candidate_universes/general_merged/proteins','data/catalyst_candidate_universes/general_merged/reaction_features/drfp_categorical_rdkitplus_center_v1']),
 dict(bundle='results/catalyst_clean_mainline_v1/r2e_enzgfm_center_router_v1',kind='trained_from_project_ancestor',sources=[CENTER],
      commands=['.venv/bin/python '+CENTER+' --base-dir results/catalyst_clean_mainline_v1/r2e_enzgfm_base_router_v1 --training-pairs results/catalyst_clean_mainline_v1/r2e_enzgfm_base_router_v1/training_pairs.csv --protein-feature-dir data/external/enzgfm_current/general_merged_650m_mean_v1 --reaction-feature-dir data/catalyst_candidate_universes/general_merged/reaction_features/drfp_categorical_rdkitplus_center_v1 --output-dir results/catalyst_clean_mainline_v1/r2e_enzgfm_center_router_v1 --direction r2e --dev-fold -1 --max-residual-ratio 0.1 --epochs 2 --steps-per-epoch 60 --learning-rate 3e-5 --weight-decay 1e-4 --temperature 0.07 --batch-size 64 --topk-k 10 --topk-weight 0.1 --topk-margin 0 --all-positive-weight 0.05 --anchor-weight 0.1 --anchor-batch-size 256 --historical-query-repeat 2 --seed 20260723'],
      metadata=['results/catalyst_clean_mainline_v1/r2e_enzgfm_center_router_v1/summary.json'],
      upstream=['results/catalyst_clean_mainline_v1/r2e_enzgfm_base_router_v1','data/external/enzgfm_current/general_merged_650m_mean_v1','data/catalyst_candidate_universes/general_merged/reaction_features/drfp_categorical_rdkitplus_center_v1']),
 dict(bundle='results/catalyst_clean_mainline_v1/r2e_lambdarank_fusion_v1',kind='fixed_internal_ranker_pipeline',sources=[R2LR],
      commands=[*(f'.venv/bin/python {R2LR} prepare --fold {f}' for f in (0,1,2)),f'.venv/bin/python {R2LR} search',f'.venv/bin/python {R2LR} fit-selected','cp results/r2e_lambdarank_fusion_v1/selected/ranker.json results/catalyst_clean_mainline_v1/r2e_lambdarank_fusion_v1/ranker.json','cp results/r2e_lambdarank_fusion_v1/selected/config.json results/catalyst_clean_mainline_v1/r2e_lambdarank_fusion_v1/config.json'],
      metadata=['results/catalyst_clean_mainline_v1/r2e_lambdarank_fusion_v1/config.json','results/catalyst_clean_mainline_v1/r2e_lambdarank_fusion_v1/manifest.json'],
      upstream=['results/catalyst_clean_mainline_v1/r2e_center_bounded_cap0p1','results/catalyst_clean_mainline_v1/r2e_enzgfm_center_router_v1']),
 dict(bundle='results/catalyst_clean_mainline_v1/e2r_anchored_lambdamart_v3',kind='trained_experts_plus_frozen_ranker',sources=[E2V3],
      commands=[f'.venv/bin/python {E2V3} --expert all'],
      metadata=['projects/active/terpene_screening/CATALYST_E2R_ANCHORED_LAMBDAMART_V3_PRODUCTION.json'],
      upstream=['results/unified_safe_system_v1/e2r_anchored_lambdamart_v3_confirmation/anchored/final_ranker.json','data/external/enzgfm_current/general_merged_650m_mean_v1','data/catalyst_candidate_universes/general_merged/proteins','data/external/enzgfm_current/general_merged_esmc_enzgfm_equalblock_v1']),
 dict(bundle='results/bime_rank_unified_v1/r2e_clipzyme_expert_v1/selected',kind='fixed_internal_ranker_pipeline',sources=[R2CLIP],
      commands=[*(f'.venv/bin/python {R2CLIP} prepare --fold {f}' for f in (0,1,2)),f'.venv/bin/python {R2CLIP} crossfit',f'.venv/bin/python {R2CLIP} fit-final'],
      metadata=['results/bime_rank_unified_v1/r2e_clipzyme_expert_v1/selected/config.json','results/bime_rank_unified_v1/r2e_clipzyme_expert_v1/development_result.json'],
      upstream=['results/catalyst_clean_mainline_v1/r2e_lambdarank_fusion_v1','results/bime_rank_unified_v1/clipzyme_r2e_candidate_asset_v1','results/clipzyme_native_extension_v1/full_hplus_candidate_reactions/clipzyme_embeddings_gpu_v1']),
 dict(bundle='results/bime_rank_unified_v1/r2e_seed_context_v1/selected',kind='fixed_internal_ranker_pipeline',sources=[R2SEED],
      commands=[*(f'.venv/bin/python {R2SEED} prepare --fold {f}' for f in (0,1,2)),f'.venv/bin/python {R2SEED} crossfit',f'.venv/bin/python {R2SEED} fit-final'],
      metadata=['results/bime_rank_unified_v1/r2e_seed_context_v1/selected/config.json','results/bime_rank_unified_v1/r2e_seed_context_v1/development_result.json'],
      upstream=['results/bime_rank_unified_v1/r2e_clipzyme_expert_v1/selected','data/catalyst_candidate_universes/general_merged/proteins']),
 dict(bundle='results/unified_safe_system_v1/e2r_clipzyme_anchored_lambdamart_v4_dev/selected',kind='fixed_internal_ranker_pipeline',sources=[E2V4],
      commands=[*(f'.venv/bin/python {E2V4} prepare --fold {f}' for f in (0,1,2)),f'.venv/bin/python {E2V4} search',f'.venv/bin/python {E2V4} fit-selected'],
      metadata=['results/unified_safe_system_v1/e2r_clipzyme_anchored_lambdamart_v4_dev/selected/config.json','results/unified_safe_system_v1/e2r_clipzyme_anchored_lambdamart_v4_dev/selection_result.json'],
      upstream=['results/catalyst_clean_mainline_v1/e2r_anchored_lambdamart_v3','results/bime_rank_unified_v1/clipzyme_e2r_query_asset_v1','results/clipzyme_native_extension_v1/full_hplus_candidate_reactions/clipzyme_embeddings_gpu_v1']),
 dict(bundle='results/bime_rank_unified_v1/e2r_seed_context_v1/selected',kind='fixed_internal_ranker_pipeline',sources=[E2SEED],
      commands=[*(f'.venv/bin/python {E2SEED} prepare --fold {f}' for f in (0,1,2)),f'.venv/bin/python {E2SEED} crossfit',f'.venv/bin/python {E2SEED} fit-final'],
      metadata=['results/bime_rank_unified_v1/e2r_seed_context_v1/selected/config.json','results/bime_rank_unified_v1/e2r_seed_context_v1/development_result.json'],
      upstream=['results/unified_safe_system_v1/e2r_clipzyme_anchored_lambdamart_v4_dev/selected']),
]
for s in current_specs:
    for m in s['metadata']:
        if m not in tracked_paths or not (ROOT/m).is_file(): raise RuntimeError(f'untracked frozen metadata: {m}')
    rows.append({'bundle':s['bundle'],'kind':s['kind'],'sources':[source_record(x) for x in s['sources']],
                 'commands':s['commands'],'frozen_metadata':s['metadata'],'upstream':[upstream_record(x) for x in s['upstream']],
                 'primary_artifacts':model_artifacts(s['bundle'])})
payload={'schema_version':2,'scope':'all unique production model bundles in configs/production_routes/terpene_v1.yaml','bundle_count':len(rows),'bundles':rows,
         'policy':'Every production bundle has an executable generator/materializer or a frozen project-owned artifact lineage. Training/search is not rerun during packaging; the release preserves fixed code, commands, frozen metadata, ancestors, and model hashes needed for independent replay.'}
OUT.write_text(json.dumps(payload,indent=2)+'\n')
print(json.dumps({'output':str(OUT.relative_to(ROOT)),'bundle_count':len(rows),'declared_upstreams':sum(len(x.get('inputs', x.get('upstream', []))) for x in rows),'model_artifacts':sum(len(x.get('outputs', x.get('primary_artifacts', []))) for x in rows)},indent=2))
