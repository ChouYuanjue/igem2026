from __future__ import annotations

import argparse
import json
import math
import pickle

import numpy as np
import pandas as pd
import torch

from projects.active.bridge.model.index import FibreCandidateIndex
from reproducibility.bime_rank.scripts import evaluate_bridge_reciprocal_relation_context_v4 as base_eval
from reproducibility.bime_rank.scripts import evaluate_bridge_adaptive_relation_context_v8 as adaptive

ROOT = adaptive.ROOT
TRAIN = adaptive.TRAIN
TARGETS = base_eval.TARGETS
ASSET = adaptive.ASSET
OUT = ROOT / "results/bridge_adaptive_relation_context_outer_v8"


def _predict_alpha(bundle: dict, direction: str, features: np.ndarray) -> tuple[float, float, float]:
    asset = bundle["directions"][direction]
    x = asset["scaler"].transform(features[None, :])
    p = float(asset["permission_model"].predict_proba(x)[0, 1])
    strength = float(np.expm1(asset["strength_model"].predict(x)[0]))
    strength = float(np.clip(strength, 0.0, max(asset["alpha_support"])))
    alpha = float(np.clip((p ** float(asset["permission_power"])) * strength, 0.0, max(asset["alpha_support"])))
    return alpha, p, strength


def _best_rank(score: np.ndarray, target_rows: np.ndarray, lexical: np.ndarray) -> int:
    values = score[target_rows]
    best_value = float(values.max())
    tied_targets = target_rows[values == best_value]
    best_target = int(tied_targets[np.argmin(lexical[tied_targets])])
    return int(
        np.count_nonzero(score > best_value)
        + np.count_nonzero((score == best_value) & (lexical < lexical[best_target]))
        + 1
    )


def evaluate(
    direction: str,
    index: FibreCandidateIndex,
    frame: pd.DataFrame,
    train: pd.DataFrame,
    pproto: torch.Tensor,
    pmask: torch.Tensor,
    rproto: torch.Tensor,
    rmask: torch.Tensor,
    bundle: dict,
    batch_size: int,
) -> dict:
    P = index.protein_embeddings
    R = index.reaction_embeddings
    if direction == "r2e":
        targets = frame.groupby("reaction_id").protein_id.apply(lambda s: set(map(str, s))).to_dict()
        qids = sorted(targets)
        qindex = index.reaction_index
        cindex = index.protein_index
        candidate_ids = index.protein_ids
        candidate_embeddings = P
        known = train.groupby("reaction_id").protein_id.apply(lambda s: set(map(str, s))).to_dict()
    else:
        targets = frame.groupby("protein_id").reaction_id.apply(lambda s: set(map(str, s))).to_dict()
        qids = sorted(targets)
        qindex = index.protein_index
        cindex = index.reaction_index
        candidate_ids = index.reaction_ids
        candidate_embeddings = R
        known = train.groupby("protein_id").reaction_id.apply(lambda s: set(map(str, s))).to_dict()

    lexical = np.empty(len(candidate_ids), dtype=np.int64)
    lexical_order = np.argsort(np.asarray(candidate_ids, dtype=object), kind="stable")
    lexical[lexical_order] = np.arange(len(lexical_order))
    base_ranks: list[int] = []
    adaptive_ranks: list[int] = []
    audit: list[dict] = []

    for start in range(0, len(qids), batch_size):
        qs = [q for q in qids[start : start + batch_size] if q in qindex]
        if not qs:
            continue
        qi = torch.as_tensor([qindex[q] for q in qs], dtype=torch.long, device=index.device)
        with torch.no_grad():
            if direction == "r2e":
                qemb = R.index_select(0, qi)
                raw_base = qemb @ P.T
                raw_a = rproto.index_select(0, qi) @ P.T
                av_a = rmask.index_select(0, qi)[:, None].expand_as(raw_a)
                raw_b = qemb @ pproto.T
                av_b = pmask[None, :].expand_as(raw_b)
            else:
                qemb = P.index_select(0, qi)
                raw_base = qemb @ R.T
                raw_a = pproto.index_select(0, qi) @ R.T
                av_a = pmask.index_select(0, qi)[:, None].expand_as(raw_a)
                raw_b = qemb @ rproto.T
                av_b = rmask[None, :].expand_as(raw_b)

            base_z = base_eval.z_masked(raw_base, torch.ones_like(raw_base, dtype=torch.bool))
            za = base_eval.z_masked(raw_a, av_a)
            zb = base_eval.z_masked(raw_b, av_b)
            den = av_a.float() + av_b.float()
            context = (za + zb) / den.clamp_min(1.0)
            context[den == 0] = 0

        B = base_z.cpu().numpy().astype(np.float64, copy=False)
        C = context.cpu().numpy().astype(np.float64, copy=False)
        A = za.cpu().numpy().astype(np.float64, copy=False)
        ZB = zb.cpu().numpy().astype(np.float64, copy=False)
        AV = (den > 0).cpu().numpy()
        BOTH = (av_a & av_b).cpu().numpy()

        for j, query_id in enumerate(qs):
            known_ids = [x for x in known.get(query_id, set()) if x in cindex]
            seed_rows = [cindex[x] for x in known_ids]
            target_rows = np.asarray(
                [cindex[x] for x in targets[query_id] if x in cindex], dtype=np.int64
            )
            if not len(target_rows):
                continue

            base = B[j].copy()
            ctx = C[j].copy()
            if seed_rows:
                seed_arr = np.asarray(seed_rows, dtype=np.int64)
                base[seed_arr] = -np.inf
                ctx[seed_arr] = 0.0
            finite = np.isfinite(base)
            work_base = base.copy()
            if not np.all(finite):
                floor = float(np.min(work_base[finite]) - 100.0) if np.any(finite) else -100.0
                work_base[~finite] = floor

            coherence = adaptive._query_neighbor_coherence(seed_rows, candidate_embeddings)
            features = adaptive._features(
                work_base,
                ctx,
                A[j],
                ZB[j],
                AV[j] & finite,
                BOTH[j] & finite,
                len(seed_rows),
                coherence,
            )
            alpha, probability, conditional_strength = _predict_alpha(bundle, direction, features)
            adaptive_score = work_base + alpha * ctx

            br = _best_rank(work_base, target_rows, lexical)
            ar = _best_rank(adaptive_score, target_rows, lexical)
            base_ranks.append(br)
            adaptive_ranks.append(ar)
            audit.append(
                {
                    "query_id": query_id,
                    "query_degree": len(seed_rows),
                    "permission_probability": probability,
                    "conditional_strength": conditional_strength,
                    "alpha": alpha,
                    "base_rank": br,
                    "adaptive_rank": ar,
                }
            )

        print(direction, min(start + batch_size, len(qids)), "/", len(qids), flush=True)

    audit_frame = pd.DataFrame(audit)
    audit_frame.to_csv(OUT / f"{direction}_query_audit.csv.gz", index=False)
    return {
        "queries": int(len(base_ranks)),
        "base": adaptive._metric(np.asarray(base_ranks)),
        "adaptive_relation": adaptive._metric(np.asarray(adaptive_ranks)),
        "alpha": {
            "mean": float(audit_frame.alpha.mean()),
            "median": float(audit_frame.alpha.median()),
            "near_zero_fraction": float((audit_frame.alpha < 0.05).mean()),
            "p90": float(audit_frame.alpha.quantile(0.90)),
            "p99": float(audit_frame.alpha.quantile(0.99)),
        },
        "permission_probability": {
            "mean": float(audit_frame.permission_probability.mean()),
            "median": float(audit_frame.permission_probability.median()),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    with open(ASSET, "rb") as handle:
        bundle = pickle.load(handle)

    train = pd.read_csv(TRAIN, dtype=str).fillna("").drop_duplicates(["protein_id", "reaction_id"])
    outer = pd.read_csv(TARGETS, dtype=str).fillna("")[["protein_id", "reaction_id"]].drop_duplicates()

    index = FibreCandidateIndex(device=args.device)
    pproto, pmask, rproto, rmask = base_eval.build_prototypes(index, train)

    result = {
        "schema": "bridge-adaptive-relation-context-outer-v8",
        "protocol": {
            "gate_asset": str(ASSET.relative_to(ROOT)),
            "gate_frozen_before_outer": True,
            "outer_used_for_selection": False,
            "post_outer_retuning_allowed": False,
            "context_source": "clean2023 exact relation graph only",
            "authority": "alpha(q)=p_help(q)^gamma * conditional_strength(q)",
            "fixed_rank_band": False,
        },
        "r2e": evaluate("r2e", index, outer, train, pproto, pmask, rproto, rmask, bundle, 24),
        "e2r": evaluate("e2r", index, outer, train, pproto, pmask, rproto, rmask, bundle, 192),
    }
    (OUT / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
