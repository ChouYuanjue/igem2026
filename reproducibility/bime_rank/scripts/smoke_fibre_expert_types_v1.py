from __future__ import annotations

import gc
import json
from pathlib import Path

import numpy as np
import torch

from projects.active.fibre.evidence.experts import (
    CageFamilyPairEvidence,
    ReactionCenterPairEvidence,
    SeedHomologyPairEvidence,
)
from projects.active.fibre.evidence.pair_scores import ClipzymePairEvidence, enzgfm_pair_evidence
from projects.active.fibre.model.assets import ROOT
from projects.active.fibre.model.index import FibreCandidateIndex

OUT = ROOT / "results/fibre_expert_types_v1"


def check_output(name: str, output) -> dict[str, object]:
    output.validate(len(output.score))
    available = np.asarray(output.available, dtype=bool)
    score = np.asarray(output.score, dtype=float)
    if not available.any():
        raise RuntimeError(f"{name} smoke produced no available score")
    if not np.isfinite(score[available]).all():
        raise RuntimeError(f"{name} smoke produced non-finite available score")
    return {
        "available": int(available.sum()),
        "score_min": float(score[available].min()),
        "score_max": float(score[available].max()),
    }


def release(*objects) -> None:
    for obj in objects:
        del obj
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    result: dict[str, object] = {
        "schema": "fibre-expert-types-v1-smoke",
        "status": "running",
        "types": {},
    }

    functional = enzgfm_pair_evidence("r2e", device="cuda")
    query = functional.r_ids[0]
    candidates = functional.p_ids[:16]
    result["types"]["functional_foundation"] = {
        "module": functional.name,
        "query": query,
        **check_output(
            functional.name,
            functional.score(direction="r2e", query_id=query, candidate_ids=candidates),
        ),
    }
    release(functional)

    structural = ClipzymePairEvidence(device="cuda")
    r_rows = np.flatnonzero(structural.r_supported)
    p_rows = np.flatnonzero(structural.p_supported)
    if not len(r_rows) or not len(p_rows):
        raise RuntimeError("CLIPZyme smoke has no supported assets")
    r_ids = [k for k, _ in sorted(structural.r_index.items(), key=lambda kv: kv[1])]
    p_ids = [k for k, _ in sorted(structural.p_index.items(), key=lambda kv: kv[1])]
    query = r_ids[int(r_rows[0])]
    candidates = [p_ids[int(i)] for i in p_rows[:16]]
    result["types"]["structural_geometry"] = {
        "module": structural.name,
        "query": query,
        **check_output(
            structural.name,
            structural.score(direction="r2e", query_id=query, candidate_ids=candidates),
        ),
    }
    release(structural)

    mechanism = ReactionCenterPairEvidence(device="cuda")
    query = mechanism.r_ids[0]
    candidates = mechanism.p_ids[:16]
    result["types"]["mechanistic"] = {
        "module": mechanism.name,
        "query": query,
        **check_output(
            mechanism.name,
            mechanism.score(direction="r2e", query_id=query, candidate_ids=candidates),
        ),
        "cache": "results/fibre_expert_assets_v1/reaction_center",
    }
    release(mechanism)

    family_result: dict[str, object] = {}
    for family in ("p450", "phosphatase", "terpene"):
        expert = CageFamilyPairEvidence(family)
        query, group = next(iter(expert._groups.items()))
        candidates = list(group)[:16]
        family_result[family] = {
            "module": expert.name,
            "query": query,
            **check_output(
                expert.name,
                expert.score(direction="r2e", query_id=query, candidate_ids=candidates),
            ),
        }
    result["types"]["family_domain"] = family_result

    index = FibreCandidateIndex(device="cuda")
    seed = index.protein_ids[0]
    candidates = index.protein_ids[:16]
    context = SeedHomologyPairEvidence([seed], index=index, device="cuda")
    result["types"]["contextual_observational"] = {
        "module": context.name,
        "seed": seed,
        **check_output(
            context.name,
            context.score(direction="r2e", query_id=index.reaction_ids[0], candidate_ids=candidates),
        ),
    }
    release(context, index)

    result["status"] = "completed"
    (OUT / "smoke.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
