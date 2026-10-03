from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from projects.active.bridge.model.assets import ROOT
from projects.active.bridge.runtime.relational import FibreRelationalRuntime
from projects.active.bridge.runtime.ranking_metrics import (
    evaluate_full_candidate_ranks,
    summarize_query_metrics,
)
from projects.active.bridge.training.train import split_double_cold

DEFAULT_CHECKPOINT = ROOT / "results/fibre_relational_main_v1/best.pt"
DEFAULT_PAIRS = ROOT / "data/external/enzymecage_current/catalyst_features/clean2023/training_pairs.csv"
DEFAULT_OUTPUT = ROOT / "results/fibre_relational_main_v1/final_evaluation"
STRICT_ROOT = ROOT / "results/clipzyme_native_extension_v1"
STRICT_PAIRS = ROOT / "results/rhea128_to141_external_v2/rhea128_to141_sprot_strict_double_cold_v2/test_pairs.csv"


def _positions(order: list[str], positives: set[str]) -> np.ndarray:
    pos = {value: i + 1 for i, value in enumerate(order)}
    ranks = sorted(pos[value] for value in positives if value in pos)
    if not ranks:
        raise ValueError("query has no positive in candidate support")
    return np.asarray(ranks, dtype=np.int64)


def _rerank_supported_slots(
    base_order: list[str],
    supported: list[bool],
    relation_scores: np.ndarray,
) -> list[str]:
    slots = [i for i, ok in enumerate(supported) if ok]
    if len(slots) != len(relation_scores):
        raise ValueError("relation scores do not align to supported shortlist")
    supported_ids = [base_order[i] for i in slots]
    rel_order = np.argsort(-np.asarray(relation_scores), kind="stable")
    result = list(base_order)
    for slot, local in zip(slots, rel_order, strict=True):
        result[slot] = supported_ids[int(local)]
    return result


class FinalEvaluator:
    def __init__(self, checkpoint: Path, *, shortlist: int, relation_batch: int, device: str) -> None:
        self.runtime = FibreRelationalRuntime(
            checkpoint,
            device=device,
            shortlist=shortlist,
            relation_batch=relation_batch,
        )
        self.shortlist = int(shortlist)
        self.device = self.runtime.device

    def _base_r2e(self, reaction_id: str, candidate_ids: list[str]) -> list[str]:
        idx = self.runtime.index
        rows = torch.as_tensor(
            [idx.protein_index[x] for x in candidate_ids],
            dtype=torch.long,
            device=self.device,
        )
        q = idx.reaction_embeddings[idx.reaction_index[str(reaction_id)]]
        scores = idx.protein_embeddings.index_select(0, rows) @ q
        order = torch.argsort(scores, descending=True, stable=True).cpu().numpy()
        return [candidate_ids[int(i)] for i in order]

    def _base_e2r(self, protein_id: str, candidate_ids: list[str]) -> list[str]:
        idx = self.runtime.index
        rows = torch.as_tensor(
            [idx.reaction_index[x] for x in candidate_ids],
            dtype=torch.long,
            device=self.device,
        )
        q = idx.protein_embeddings[idx.protein_index[str(protein_id)]]
        scores = idx.reaction_embeddings.index_select(0, rows) @ q
        order = torch.argsort(scores, descending=True, stable=True).cpu().numpy()
        return [candidate_ids[int(i)] for i in order]

    def rank_r2e(self, reaction_id: str, candidate_ids: list[str]) -> list[str]:
        base = self._base_r2e(reaction_id, candidate_ids)
        if not self.runtime.assets.tokens.available(reaction_id):
            return base
        n = min(self.shortlist, len(base))
        head = base[:n]
        scores = self.runtime._scores(head, [str(reaction_id)] * n)
        order = np.argsort(-scores, kind="stable")
        return [head[int(i)] for i in order] + base[n:]

    def rank_e2r(self, protein_id: str, candidate_ids: list[str]) -> list[str]:
        base = self._base_e2r(protein_id, candidate_ids)
        n = min(self.shortlist, len(base))
        head = base[:n]
        supported = [self.runtime.assets.tokens.available(rid) for rid in head]
        supported_ids = [rid for rid, ok in zip(head, supported, strict=True) if ok]
        if supported_ids:
            scores = self.runtime._scores(
                [str(protein_id)] * len(supported_ids),
                supported_ids,
            )
            head = _rerank_supported_slots(head, supported, scores)
        return head + base[n:]


def _evaluate_direction(
    evaluator: FinalEvaluator,
    *,
    direction: str,
    positives: dict[str, set[str]],
    candidate_ids: list[str],
) -> tuple[pd.DataFrame, dict[str, object]]:
    rows: list[dict[str, object]] = []
    for i, (query, pos) in enumerate(sorted(positives.items())):
        if direction == "r2e":
            order = evaluator.rank_r2e(query, candidate_ids)
        elif direction == "e2r":
            order = evaluator.rank_e2r(query, candidate_ids)
        else:
            raise ValueError(direction)
        metrics = evaluate_full_candidate_ranks(_positions(order, pos), len(candidate_ids))
        rows.append({"query_id": query, **metrics})
        if (i + 1) % 25 == 0:
            print(f"direction={direction} queries={i + 1}/{len(positives)}", flush=True)
    frame = pd.DataFrame(rows)
    return frame, summarize_query_metrics(frame)


def internal_protocol(evaluator: FinalEvaluator, pairs_path: Path) -> dict[str, object]:
    pairs = pd.read_csv(pairs_path, dtype=str).fillna("")
    pairs = evaluator.runtime.assets.supported_pairs(pairs)
    _, _, test = split_double_cold(pairs)
    r_pos = test.groupby("reaction_id").protein_id.agg(lambda x: set(x.astype(str))).to_dict()
    e_pos = test.groupby("protein_id").reaction_id.agg(lambda x: set(x.astype(str))).to_dict()
    r_frame, r_summary = _evaluate_direction(
        evaluator,
        direction="r2e",
        positives=r_pos,
        candidate_ids=list(evaluator.runtime.index.protein_ids),
    )
    e_frame, e_summary = _evaluate_direction(
        evaluator,
        direction="e2r",
        positives=e_pos,
        candidate_ids=list(evaluator.runtime.index.reaction_ids),
    )
    return {"r2e": r_summary, "e2r": e_summary, "r2e_frame": r_frame, "e2r_frame": e_frame}


def strict_temporal_protocol(evaluator: FinalEvaluator) -> dict[str, object]:
    r_root = STRICT_ROOT / "r2e_strict650_same_support_v1"
    r_queries = [x.strip() for x in (r_root / "mutual_cold_query_ids.txt").read_text().splitlines() if x.strip()]
    r_candidates = [x.strip() for x in (STRICT_ROOT / "r2e_strict650_candidate_ids.txt").read_text().splitlines() if x.strip()]
    r_pairs = pd.read_csv(r_root / "mutual_cold_test_pairs.csv", dtype=str).fillna("")
    r_pos = r_pairs[r_pairs.reaction_id.isin(r_queries)].groupby("reaction_id").protein_id.agg(lambda x: set(x.astype(str))).to_dict()

    clip_q = pd.read_csv(
        STRICT_ROOT / "e2r_strict650_mutual_cold_10131_v2_lexical/official_clipzyme_query_metrics.csv",
        dtype={"query_id": str},
    )
    e_queries = clip_q.query_id.astype(str).tolist()
    clip_entries = pd.read_csv(
        ROOT / "results/clipzyme_native_extension_v1/full_hplus_candidate_reactions/clipzyme_embeddings_gpu_v1/entries.csv",
        dtype=str,
    ).fillna("")
    clip_entries = clip_entries[clip_entries.clipzyme_supported.astype(str).str.lower().eq("true")]
    e_candidates = sorted(clip_entries.reaction_id.astype(str).tolist())
    strict_pairs = pd.read_csv(STRICT_PAIRS, dtype=str).fillna("")
    e_set = set(e_candidates)
    e_pos = strict_pairs[
        strict_pairs.protein_id.isin(e_queries) & strict_pairs.reaction_id.isin(e_set)
    ].groupby("protein_id").reaction_id.agg(lambda x: set(x.astype(str))).to_dict()

    r_frame, r_summary = _evaluate_direction(
        evaluator,
        direction="r2e",
        positives=r_pos,
        candidate_ids=r_candidates,
    )
    e_frame, e_summary = _evaluate_direction(
        evaluator,
        direction="e2r",
        positives=e_pos,
        candidate_ids=e_candidates,
    )
    return {"r2e": r_summary, "e2r": e_summary, "r2e_frame": r_frame, "e2r_frame": e_frame}


def _write_protocol(root: Path, name: str, result: dict[str, object]) -> dict[str, object]:
    root.mkdir(parents=True, exist_ok=True)
    result["r2e_frame"].to_csv(root / f"{name}_r2e_query_metrics.csv", index=False)
    result["e2r_frame"].to_csv(root / f"{name}_e2r_query_metrics.csv", index=False)
    return {"r2e": result["r2e"], "e2r": result["e2r"]}


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate the single final BRIDGE relational model.")
    ap.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    ap.add_argument("--pairs", type=Path, default=DEFAULT_PAIRS)
    ap.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    ap.add_argument("--shortlist", type=int, default=4096)
    ap.add_argument("--relation-batch", type=int, default=32)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    evaluator = FinalEvaluator(
        args.checkpoint,
        shortlist=args.shortlist,
        relation_batch=args.relation_batch,
        device=args.device,
    )
    internal = _write_protocol(args.output, "internal_test", internal_protocol(evaluator, args.pairs))
    temporal = _write_protocol(args.output, "strict_temporal", strict_temporal_protocol(evaluator))
    summary = {
        "method": "BRIDGE multimodal relational learning",
        "checkpoint": str(args.checkpoint),
        "shortlist": int(args.shortlist),
        "ranking": "full broad-index order with relation-supported shortlist slots reranked by BRIDGE; unsupported ERAM reactions retain broad-index slots",
        "internal_test": internal,
        "strict_temporal_rhea128_to141": temporal,
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
