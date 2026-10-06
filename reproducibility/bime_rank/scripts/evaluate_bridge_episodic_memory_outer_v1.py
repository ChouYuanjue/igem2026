from __future__ import annotations

import argparse
import json
import pickle

import numpy as np
import pandas as pd
import torch

from projects.active.bridge.model.assets import ROOT
from projects.active.bridge.model.index import FibreCandidateIndex
from projects.active.bridge.runtime.memory import (
    EPISODIC_FEATURE_NAMES,
    build_episodic_memory,
    episodic_authority,
)
from reproducibility.bime_rank.scripts import evaluate_bridge_adaptive_relation_context_v8 as long_eval
from reproducibility.bime_rank.scripts import evaluate_bridge_episodic_memory_v1 as validation
from reproducibility.bime_rank.scripts import evaluate_bridge_reciprocal_relation_context_v4 as base_eval

OUT = ROOT / "results/bridge_episodic_memory_outer_v1"
EPISODIC_ASSET = ROOT / "projects/active/bridge/release/runtime/final_bridge_v1/episodic_memory_gate.production.pkl"


def metric(ranks: np.ndarray) -> dict[str, float]:
    r = np.asarray(ranks, dtype=np.int64)
    return {
        "queries": int(len(r)),
        "mrr": float(np.mean(1.0 / r)),
        "hit10": float(np.mean(r <= 10)),
        "hit100": float(np.mean(r <= 100)),
        "hit1000": float(np.mean(r <= 1000)),
        "median_rank": float(np.median(r)),
    }


def best_rank(score: np.ndarray, targets: np.ndarray, lex: np.ndarray) -> int:
    values = score[targets]
    best = float(values.max())
    tied = targets[values == best]
    row = int(tied[np.argmin(lex[tied])])
    return int(
        np.count_nonzero(score > best)
        + np.count_nonzero((score == best) & (lex < lex[row]))
        + 1
    )


def evaluate_direction(
    direction: str,
    index: FibreCandidateIndex,
    outer: pd.DataFrame,
    train: pd.DataFrame,
    pproto: torch.Tensor,
    pmask: torch.Tensor,
    rproto: torch.Tensor,
    rmask: torch.Tensor,
    long_gate: dict,
    episodic_gate: dict,
    batch_size: int,
) -> tuple[pd.DataFrame, dict]:
    P, R = index.protein_embeddings, index.reaction_embeddings
    if direction == "r2e":
        groups = outer.groupby("reaction_id").protein_id.apply(lambda s: sorted(set(map(str, s)))).to_dict()
        qindex, cindex = index.reaction_index, index.protein_index
        candidate_ids, candidates = index.protein_ids, P
        known = train.groupby("reaction_id").protein_id.apply(lambda s: set(map(str, s))).to_dict()
        seen_mask = pmask.detach().cpu().numpy().astype(bool)
    else:
        groups = outer.groupby("protein_id").reaction_id.apply(lambda s: sorted(set(map(str, s)))).to_dict()
        qindex, cindex = index.protein_index, index.reaction_index
        candidate_ids, candidates = index.reaction_ids, R
        known = train.groupby("protein_id").reaction_id.apply(lambda s: set(map(str, s))).to_dict()
        seen_mask = rmask.detach().cpu().numpy().astype(bool)

    query_ids = [
        q for q in sorted(groups)
        if q in qindex and sum(x in cindex for x in groups[q]) >= 2
    ]
    lex = np.empty(len(candidate_ids), dtype=np.int64)
    order = np.argsort(np.asarray(candidate_ids, dtype=object), kind="stable")
    lex[order] = np.arange(len(order))
    records: list[dict[str, object]] = []

    for start in range(0, len(query_ids), batch_size):
        qs = query_ids[start : start + batch_size]
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
            bz = base_eval.z_masked(raw_base, torch.ones_like(raw_base, dtype=torch.bool))
            za = base_eval.z_masked(raw_a, av_a)
            zb = base_eval.z_masked(raw_b, av_b)
            den = av_a.float() + av_b.float()
            ctx = (za + zb) / den.clamp_min(1.0)
            ctx[den == 0] = 0

        B = bz.cpu().numpy().astype(np.float64, copy=False)
        C = ctx.cpu().numpy().astype(np.float64, copy=False)
        A = za.cpu().numpy().astype(np.float64, copy=False)
        BB = zb.cpu().numpy().astype(np.float64, copy=False)
        AV = (den > 0).cpu().numpy()
        BOTH = (av_a & av_b).cpu().numpy()

        for j, query_id in enumerate(qs):
            positives = [x for x in validation.stable_order(query_id, groups[query_id]) if x in cindex]
            n_support = validation.support_size(query_id, len(positives))
            if n_support <= 0:
                continue
            support = positives[:n_support]
            targets = positives[n_support:]
            target_rows = np.asarray([cindex[x] for x in targets], dtype=np.int64)
            if not len(target_rows):
                continue

            train_ids = {x for x in known.get(query_id, set()) if x in cindex}
            train_rows = [cindex[x] for x in train_ids]
            coherence = long_eval._query_neighbor_coherence(train_rows, candidates)
            long_weight = validation.relation_weight(
                direction=direction,
                gate=long_gate,
                base_z=B[j],
                ctx=C[j],
                comp_a=A[j],
                comp_b=BB[j],
                available=AV[j],
                both=BOTH[j],
                query_degree=len(train_rows),
                coherence=coherence,
            )
            prior = B[j] + long_weight * C[j]
            memory = build_episodic_memory(
                query_embedding=qemb[j],
                candidate_embeddings=candidates,
                candidate_ids=candidate_ids,
                candidate_index=cindex,
                requested_support_ids=support,
                training_positive_ids=train_ids,
                entity_seen_mask=seen_mask,
                base_score=prior,
                train_memory_weight=long_weight,
            )
            if memory is None:
                continue
            gate, raw_gate = episodic_authority(
                episodic_gate,
                direction,
                memory.features,
            )
            final = prior + gate * (memory.score - B[j])
            masked_rows = np.asarray(
                sorted(set(train_rows + [cindex[x] for x in support])),
                dtype=np.int64,
            )
            prior_rank_score = prior.copy()
            final_rank_score = final.copy()
            if len(masked_rows):
                prior_rank_score[masked_rows] = -np.inf
                final_rank_score[masked_rows] = -np.inf

            records.append(
                {
                    "direction": direction,
                    "query_id": query_id,
                    "support_count": len(support),
                    "target_count": len(targets),
                    "base_rank": best_rank(prior_rank_score, target_rows, lex),
                    "episodic_rank": best_rank(final_rank_score, target_rows, lex),
                    "gate": gate,
                    "raw_gate": raw_gate,
                    "effective_support_count": memory.effective_support_count,
                    "support_coherence": memory.support_coherence,
                }
            )

        print(direction, min(start + batch_size, len(query_ids)), "/", len(query_ids), flush=True)

    frame = pd.DataFrame(records)
    summary = {
        "queries": int(len(frame)),
        "base": metric(frame.base_rank.to_numpy(np.int64)),
        "episodic": metric(frame.episodic_rank.to_numpy(np.int64)),
        "gate": {
            "mean": float(frame.gate.mean()),
            "median": float(frame.gate.median()),
            "p10": float(frame.gate.quantile(0.10)),
            "p90": float(frame.gate.quantile(0.90)),
            "near_zero_fraction": float((frame.gate < 0.05).mean()),
        },
    }
    return frame, summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    index = FibreCandidateIndex(device=args.device)
    train = pd.read_csv(base_eval.TRAIN, dtype=str).fillna("").drop_duplicates(["protein_id", "reaction_id"])
    outer = pd.read_csv(base_eval.TARGETS, dtype=str).fillna("")[["protein_id", "reaction_id"]].drop_duplicates()
    pproto, pmask, rproto, rmask = base_eval.build_prototypes(index, train)
    with open(long_eval.ASSET, "rb") as handle:
        long_gate = pickle.load(handle)
    with open(EPISODIC_ASSET, "rb") as handle:
        episodic_gate = pickle.load(handle)

    r_frame, r_summary = evaluate_direction(
        "r2e", index, outer, train, pproto, pmask, rproto, rmask, long_gate, episodic_gate, 16
    )
    r_frame.to_csv(OUT / "r2e_query_metrics.csv.gz", index=False)
    e_frame, e_summary = evaluate_direction(
        "e2r", index, outer, train, pproto, pmask, rproto, rmask, long_gate, episodic_gate, 128
    )
    e_frame.to_csv(OUT / "e2r_query_metrics.csv.gz", index=False)

    result = {
        "schema": "bridge-episodic-memory-frozen-outer-v1",
        "protocol": {
            "gate_asset": str(EPISODIC_ASSET.relative_to(ROOT)),
            "gate_frozen_before_outer": True,
            "outer_used_for_model_or_hyperparameter_selection": False,
            "alternative_gate_values_evaluated_on_outer": False,
            "episode_rule": "same deterministic validation rule: one support set from outer positives, remaining positives are prediction targets",
            "scope": "few-shot episodic-memory layer diagnostic; separate from zero-shot BRIDGE headline",
            "training_relations_as_support": False,
        },
        "r2e": r_summary,
        "e2r": e_summary,
    }
    (OUT / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
