from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from projects.active.terpene_screening.blend_general_evidence_models import ASSET_FILES
from projects.active.terpene_screening.merge_general_evidence_fisher import (
    DEFAULT_BROAD_UNIVERSE,
    _load_one_model,
    _scale_tag,
    _source_r2e_data,
    _target_r2e_data,
)
from projects.active.terpene_screening.third_party.fusionbench_regmean import (
    merging_with_regmean_weights,
)


def _sample_rows(ids: list[str], max_queries: int, seed: int) -> list[int]:
    rows = list(range(len(ids)))
    if max_queries <= 0 or len(rows) <= max_queries:
        return rows
    rng = random.Random(seed)
    return sorted(rng.sample(rows, max_queries))


def _collect_linear_grams(
    model: nn.Module,
    query_features: np.ndarray,
    query_ids: list[str],
    *,
    module_prefix: str,
    max_queries: int,
    sample_seed: int,
    batch_size: int,
    device: torch.device,
) -> tuple[dict[str, torch.Tensor], dict[str, object]]:
    selected = _sample_rows(query_ids, max_queries, sample_seed)
    if not selected:
        raise ValueError('No RegMean query rows selected')
    modules = {
        name: module
        for name, module in model.named_modules()
        if name.startswith(module_prefix) and isinstance(module, nn.Linear)
    }
    if not modules:
        raise ValueError(f'No Linear modules under {module_prefix}')
    sums: dict[str, torch.Tensor] = {}
    counts: dict[str, int] = {}
    handles = []

    def make_hook(name: str):
        def hook(_module, inputs, _output):
            x = inputs[0].detach().reshape(-1, inputs[0].shape[-1]).float()
            gram = x.transpose(0, 1) @ x
            gram = gram.detach().cpu()
            sums[name] = sums.get(name, torch.zeros_like(gram)) + gram
            counts[name] = counts.get(name, 0) + x.shape[0]
        return hook

    for name, module in modules.items():
        handles.append(module.register_forward_hook(make_hook(name)))
    model.eval()
    with torch.no_grad():
        for start in range(0, len(selected), batch_size):
            rows = selected[start:start + batch_size]
            batch = torch.as_tensor(query_features[rows], dtype=torch.float32, device=device)
            model.encode_reactions(batch)
    for handle in handles:
        handle.remove()
    grams = {name: sums[name] / float(counts[name]) for name in modules}
    diag = {
        'n_queries': len(selected),
        'module_prefix': module_prefix,
        'sample_seed': sample_seed,
        'trace': {name: float(torch.trace(value)) for name, value in grams.items()},
        'rank': {name: int(torch.linalg.matrix_rank(value).item()) for name, value in grams.items()},
    }
    return grams, diag


def main() -> None:
    parser = argparse.ArgumentParser(description='Source-weighted RegMean consolidation for broad R2E continuation.')
    parser.add_argument('--source-dir', type=Path, required=True)
    parser.add_argument('--adapted-dir', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--universe-dir', type=Path, default=DEFAULT_BROAD_UNIVERSE)
    parser.add_argument('--source-scales', default='1,2,5,10')
    parser.add_argument('--source-max-queries', type=int, default=0)
    parser.add_argument('--target-max-queries', type=int, default=1024)
    parser.add_argument('--sample-seed', type=int, default=20260723)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--reduce-non-diagonal-ratio', type=float, default=1.0)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    source_dir=args.source_dir.resolve(); adapted_dir=args.adapted_dir.resolve(); universe_dir=args.universe_dir.resolve(); output_root=args.output_root.resolve(); output_root.mkdir(parents=True,exist_ok=True)
    device=torch.device(args.device)
    source_model, source_payload, source_checkpoint = _load_one_model(source_dir, device)
    adapted_model, adapted_payload, adapted_checkpoint = _load_one_model(adapted_dir, device)
    if source_payload['model_config'] != adapted_payload['model_config']:
        raise ValueError('Source/adapted configs differ')

    s_query, s_ids, *_ = _source_r2e_data(source_dir)
    t_query, t_ids, *_ = _target_r2e_data(source_dir, universe_dir)
    # Only unseen broad reaction queries are required for the target activation geometry.
    historical=set(pd.read_csv(source_dir/'reaction_registry.csv',dtype=str).fillna('')['reaction_id'].astype(str))
    mask=np.asarray([rid not in historical for rid in t_ids],dtype=bool)
    t_query=t_query[mask]; t_ids=[rid for rid,keep in zip(t_ids,mask,strict=True) if keep]

    source_grams, source_diag = _collect_linear_grams(source_model,s_query,s_ids,module_prefix='reaction_tower.',max_queries=args.source_max_queries,sample_seed=args.sample_seed,batch_size=args.batch_size,device=device)
    target_grams, target_diag = _collect_linear_grams(adapted_model,t_query,t_ids,module_prefix='reaction_tower.',max_queries=args.target_max_queries,sample_seed=args.sample_seed,batch_size=args.batch_size,device=device)

    source_state={k:v.detach().cpu() for k,v in source_model.state_dict().items()}; adapted_state={k:v.detach().cpu() for k,v in adapted_model.state_dict().items()}
    params={name:[source_state[name],adapted_state[name]] for name in source_state if name.startswith('reaction_tower.')}
    outputs=[]
    for source_scale in [float(x) for x in args.source_scales.split(',') if x.strip()]:
        weighted_source={name:value*source_scale for name,value in source_grams.items()}
        merged=merging_with_regmean_weights(params,[weighted_source,target_grams],reduce_non_diagonal_ratio=args.reduce_non_diagonal_ratio)
        state={name:value.clone() for name,value in source_state.items()}
        # RegMean only has a closed-form solution for Linear weights. Bias/LayerNorm are source-preserved,
        # avoiding arbitrary averaging of source-critical offsets.
        for name,value in merged.items():
            if name.endswith('.weight') and name[:-len('.weight')] in source_grams:
                state[name]=value.to(dtype=state[name].dtype)
        tag=_scale_tag(source_scale); out=output_root/f'source_{tag}'; (out/'models').mkdir(parents=True,exist_ok=True)
        payload={k:v for k,v in source_payload.items() if k!='model_state_dict'}; payload['model_state_dict']=state
        payload['general_evidence_regmean_merge']={'source_checkpoint':str(source_checkpoint),'adapted_checkpoint':str(adapted_checkpoint),'source_scale':source_scale,'reduce_non_diagonal_ratio':args.reduce_non_diagonal_ratio,'upstream':'https://github.com/tanganke/fusion_bench','upstream_commit':'54c9e8c9d9621620c720452cd8533332a32d3689'}
        target=out/'models'/source_checkpoint.name; torch.save(payload,target)
        for filename in ASSET_FILES:
            src=source_dir/filename
            if src.exists(): shutil.copy2(src,out/filename)
        (out/'summary.json').write_text(json.dumps({'model_type':'general_evidence_source_weighted_regmean','source_scale':source_scale,'source_activation':source_diag,'target_activation':target_diag,'checkpoint':str(target)},indent=2),encoding='utf-8')
        outputs.append(str(out))
    (output_root/'summary.json').write_text(json.dumps({'source_activation':source_diag,'target_activation':target_diag,'outputs':outputs},indent=2),encoding='utf-8')
    print(json.dumps({'source_activation':source_diag,'target_activation':target_diag,'outputs':outputs},indent=2))

if __name__=='__main__': main()
