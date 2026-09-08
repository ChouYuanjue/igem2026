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
        p=ROOT/rel
        outputs.append({'path':rel,'bytes':p.stat().st_size,'sha256':sha(p)})
    rows.append({
      'bundle':s['bundle'],'kind':s['kind'],'source':s['source'],'source_sha256':SOURCE_SHA[s['source']],
      'source_identity':'byte-identical to its first tracked implementation',
      'command':s['command'],'frozen_summary':s['bundle']+'/summary.json','expected_summary_fields':s['expected'],
      'inputs':inputs,'outputs':outputs,
    })
payload={'schema_version':1,'scope':'deployed legacy TPS fallback bundles only','bundle_count':len(rows),'bundles':rows,
         'policy':'Commands are executable from the repository root after restoring external assets. Learned bundles are not rerun during release packaging because their original trainers and frozen artifacts are retained; deterministic dual-kernel outputs were independently replayed byte-exactly.'}
OUT.write_text(json.dumps(payload,indent=2)+'\n')
print(json.dumps({'output':str(OUT.relative_to(ROOT)),'bundle_count':len(rows),'covered_inputs':sum(len(x['inputs']) for x in rows),'outputs':sum(len(x['outputs']) for x in rows)},indent=2))
