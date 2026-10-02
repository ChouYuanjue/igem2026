from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from projects.active.fibre.evidence.pair_scores import enzgfm_pair_evidence
from projects.active.fibre.model.assets import ROOT
from projects.active.fibre.model.index import DEFAULT_INDEX, FibreCandidateIndex


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser(description='Rebind existing R2E OOF evidence to the current frozen FIBRE broad core.')
    ap.add_argument('--template', type=Path, default=ROOT/'results/fibre_score_evidence_main_v1/r2e/source_template.csv')
    ap.add_argument('--output', type=Path, default=ROOT/'results/fibre_score_evidence_main_v1/r2e')
    ap.add_argument('--device', default='cuda')
    args = ap.parse_args()

    frame = pd.read_csv(args.template, dtype={'query_id': str, 'candidate_id': str})
    required = {'query_id','candidate_id','evidence_score','available','label','fold'}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f'missing template columns: {sorted(missing)}')

    broad = FibreCandidateIndex(device=args.device)
    enzgfm = enzgfm_pair_evidence('r2e', device=args.device)
    core_scores = np.zeros(len(frame), dtype=np.float64)
    enzgfm_scores = np.zeros(len(frame), dtype=np.float64)
    enzgfm_available = np.zeros(len(frame), dtype=bool)

    for i, (query_id, group) in enumerate(frame.groupby('query_id', sort=False)):
        positions = group.index.to_numpy(np.int64)
        candidates = group['candidate_id'].astype(str).tolist()
        rrow = broad.reaction_index.get(str(query_id))
        if rrow is None:
            raise KeyError(f'broad core missing reaction {query_id}')
        prows = torch.as_tensor(
            [broad.protein_index[x] for x in candidates],
            dtype=torch.long,
            device=broad.device,
        )
        with torch.no_grad():
            q = broad.reaction_embeddings[rrow]
            score = broad.protein_embeddings.index_select(0, prows) @ q
        core_scores[positions] = score.float().cpu().numpy()

        e = enzgfm.score(direction='r2e', query_id=str(query_id), candidate_ids=candidates)
        e.validate(len(candidates))
        enzgfm_scores[positions] = e.score
        enzgfm_available[positions] = e.available
        if (i + 1) % 250 == 0:
            print(f'queries={i + 1}/{frame.query_id.nunique()}', flush=True)

    args.output.mkdir(parents=True, exist_ok=True)
    core = frame[['query_id','candidate_id','label','fold']].copy()
    core['core_score'] = core_scores
    core = core[['query_id','candidate_id','core_score','label','fold']]
    core.to_csv(args.output/'core.csv', index=False)
    core_center = float(core_scores.mean())
    core_scale = float(core_scores.std())
    if not np.isfinite(core_scale) or core_scale <= 1e-12:
        raise ValueError('broad core scores have no usable fixed scale')
    core_calibrated = core.copy()
    core_calibrated['core_score'] = (core_scores - core_center) / core_scale
    core_calibrated.to_csv(args.output/'core_calibrated.csv', index=False)
    baseline_id = f'fibre-broad-rankstrong-r2e98-{sha256(DEFAULT_INDEX)[:8]}'
    (args.output/'core_calibration.json').write_text(json.dumps({
        'method': 'fixed_global_affine_v1',
        'baseline_id': baseline_id,
        'center': core_center,
        'scale': core_scale,
        'ranking_invariant': True,
        'checkpoint_sha256': sha256(DEFAULT_INDEX),
    }, indent=2)+'\n')

    clip = pd.DataFrame({
        'direction': 'r2e',
        'query_id': frame['query_id'].astype(str),
        'candidate_id': frame['candidate_id'].astype(str),
        'score': pd.to_numeric(frame['evidence_score']).astype(float),
        'available': frame['available'],
    })
    clip.to_csv(args.output/'clipzyme.csv', index=False)

    enz = pd.DataFrame({
        'direction': 'r2e',
        'query_id': frame['query_id'].astype(str),
        'candidate_id': frame['candidate_id'].astype(str),
        'score': enzgfm_scores,
        'available': enzgfm_available,
    })
    enz.to_csv(args.output/'enzgfm.csv', index=False)

    clip_desc = {
        'name': 'clipzyme_structure', 'kind': 'structural', 'role': 'rerank',
        'directions': ['r2e'], 'score_semantics': 'higher cosine means stronger CLIPZyme structural compatibility',
        'availability_semantics': 'both reaction and protein have valid CLIPZyme embeddings',
        'quality_semantics': None, 'provenance': 'frozen CLIPZyme structure/reaction assets already present in igem2026',
        'score_direction': 'higher_is_better',
    }
    enz_desc = {
        'name': 'enzgfm_r2e', 'kind': 'molecular_view', 'role': 'rerank',
        'directions': ['r2e'], 'score_semantics': 'higher frozen EnzGFM dual-tower cosine means stronger enzyme-reaction compatibility',
        'availability_semantics': 'query reaction and candidate protein exist in the frozen EnzGFM production universe',
        'quality_semantics': None, 'provenance': 'frozen EnzGFM R2E production checkpoint already present in igem2026',
        'score_direction': 'higher_is_better',
    }
    (args.output/'clipzyme_descriptor.json').write_text(json.dumps(clip_desc, indent=2)+'\n')
    (args.output/'enzgfm_descriptor.json').write_text(json.dumps(enz_desc, indent=2)+'\n')

    manifest = {
        'schema': 'fibre-score-evidence-r2e-v1',
        'rows': int(len(frame)), 'queries': int(frame.query_id.nunique()),
        'positive_rows': int(pd.to_numeric(frame.label).sum()),
        'baseline_id': baseline_id,
        'baseline_checkpoint': str(DEFAULT_INDEX.relative_to(ROOT)),
        'baseline_checkpoint_sha256': sha256(DEFAULT_INDEX),
        'experts': ['clipzyme_structure','enzgfm_r2e'],
        'tps_policy': 'application-only; excluded from frozen broad admission',
        'core_csv': str((args.output/'core_calibrated.csv').relative_to(ROOT)),
        'core_calibration': str((args.output/'core_calibration.json').relative_to(ROOT)),
    }
    (args.output/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == '__main__':
    main()
