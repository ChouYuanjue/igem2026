from __future__ import annotations

import argparse
import copy
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from projects.active.terpene_screening.evaluate_interaction_retriever_marts import PairResidualHead  # noqa: E402
from projects.active.terpene_screening.fair_benchmark import sha256_file  # noqa: E402
from projects.active.terpene_screening.rank_open_world import (  # noqa: E402
    load_feature_schema,
    load_models,
    load_protein_library,
    load_registered_reaction_feature_library,
)
from projects.active.terpene_screening.run_internal_top2000_pair_reranker_v1 import (  # noqa: E402
    DEFAULT_BENCH,
    DEFAULT_COARSE_EVAL,
    DEFAULT_DIFFICULTY,
    DEFAULT_MODEL_ROOT,
    DEFAULT_PROTEINS,
    DEFAULT_REACTIONS,
    encode_library,
    evaluate_reranker,
    query_metrics_from_positive_rank_frame,
)

DEFAULT_PROTOCOL = ROOT / "projects/active/terpene_screening/CLEANROOM_R2E_TOP2000_BOUNDED_RESIDUAL_V2.json"
DEFAULT_V1_RESULTS = ROOT / "results/cleanroom_internal_top2000_pair_reranker_v1"
DEFAULT_OUTPUT = ROOT / "results/cleanroom_internal_top2000_bounded_residual_v2"


def scale_slug(scale: float) -> str:
    return f"scale_{scale:g}".replace(".", "p")


def _relative_delta(candidate: float, baseline: float) -> float:
    if baseline == 0:
        return 0.0 if candidate == 0 else math.inf if candidate > 0 else -math.inf
    return (candidate - baseline) / abs(baseline)


def select_residual_scale(
    fold_metrics: dict[int, dict[float, dict[str, float]]],
    *,
    scales: list[float],
    primary_metrics: list[str],
) -> dict[str, object]:
    if 0.0 not in scales:
        raise ValueError("Scale grid must contain 0.0 fallback")
    folds = sorted(fold_metrics)
    if not folds:
        raise ValueError("No fold metrics")
    for fold in folds:
        missing_scales = set(scales) - set(fold_metrics[fold])
        if missing_scales:
            raise ValueError(f"fold{fold} missing scales: {sorted(missing_scales)}")

    eligibility: dict[str, dict[str, object]] = {}
    eligible: list[float] = []
    for scale in scales:
        if scale == 0.0:
            continue
        mean_deltas: dict[str, float] = {}
        for metric in primary_metrics:
            deltas = [
                float(fold_metrics[fold][scale][metric]) - float(fold_metrics[fold][0.0][metric])
                for fold in folds
            ]
            mean_deltas[metric] = float(np.mean(deltas))
        nonnegative_all = all(value >= -1e-12 for value in mean_deltas.values())
        strictly_positive = sum(value > 1e-12 for value in mean_deltas.values())
        guard_ok = True
        guard_details: dict[str, dict[str, float]] = {}
        for fold in folds:
            guard_details[str(fold)] = {}
            for metric in ("mrr", "map", "hit_at_10"):
                rel = _relative_delta(
                    float(fold_metrics[fold][scale][metric]),
                    float(fold_metrics[fold][0.0][metric]),
                )
                guard_details[str(fold)][metric] = float(rel)
                if rel < -0.02 - 1e-12:
                    guard_ok = False
        is_eligible = nonnegative_all and strictly_positive >= 4 and guard_ok
        eligibility[f"{scale:g}"] = {
            "mean_deltas": mean_deltas,
            "strictly_positive_primary_metrics": strictly_positive,
            "all_primary_mean_deltas_nonnegative": nonnegative_all,
            "per_fold_relative_guard": guard_details,
            "guard_ok": guard_ok,
            "eligible": is_eligible,
        }
        if is_eligible:
            eligible.append(scale)

    # Percentile score is computed only after the fixed eligibility gate.  Percentiles
    # use every registered scale including the exact coarse fallback.
    fold_percentiles: dict[int, dict[float, float]] = {fold: {} for fold in folds}
    for fold in folds:
        metric_percentiles: dict[str, dict[float, float]] = {}
        for metric in primary_metrics:
            values = pd.Series({scale: fold_metrics[fold][scale][metric] for scale in scales}, dtype=float)
            pct = values.rank(method="average", pct=True)
            metric_percentiles[metric] = {float(scale): float(pct.loc[scale]) for scale in scales}
        for scale in scales:
            fold_percentiles[fold][scale] = float(
                np.mean([metric_percentiles[metric][scale] for metric in primary_metrics])
            )

    candidate_scores: dict[str, dict[str, float]] = {}
    for scale in eligible:
        per_fold = [fold_percentiles[fold][scale] for fold in folds]
        candidate_scores[f"{scale:g}"] = {
            "mean_fold_metric_percentile": float(np.mean(per_fold)),
            "worst_fold_metric_percentile": float(min(per_fold)),
        }

    if eligible:
        selected = max(
            eligible,
            key=lambda scale: (
                candidate_scores[f"{scale:g}"]["mean_fold_metric_percentile"],
                candidate_scores[f"{scale:g}"]["worst_fold_metric_percentile"],
                -scale,
            ),
        )
        promoted = True
    else:
        selected = 0.0
        promoted = False

    return {
        "selected_residual_scale": float(selected),
        "pair_residual_expert_promoted": promoted,
        "fallback_used": not promoted,
        "eligibility": eligibility,
        "candidate_scores": candidate_scores,
    }


def load_v1_head(fold: int, device: torch.device) -> PairResidualHead:
    payload = torch.load(
        DEFAULT_V1_RESULTS / f"fold{fold}" / "pair_residual_head.pt",
        map_location="cpu",
        weights_only=True,
    )
    recipe = dict(payload["recipe"])
    head_cfg = dict(recipe["head"])
    head = PairResidualHead(
        int(payload["embedding_dim"]),
        int(head_cfg["hidden_dim"]),
        float(head_cfg["dropout"]),
    ).to(device)
    head.load_state_dict(payload["head_state_dict"])
    head.eval()
    return head


def main() -> None:
    ap = argparse.ArgumentParser(description="Internal-only bounded residual scale sweep for frozen Top-2000 pair heads.")
    ap.add_argument("--folds", default="0,1,2")
    ap.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    protocol = json.loads(DEFAULT_PROTOCOL.read_text(encoding="utf-8"))
    scales = [float(value) for value in protocol["residual_scales"]]
    primary_metrics = [str(value) for value in protocol["primary_metrics"]]
    folds = [int(x) for x in args.folds.split(",") if x.strip()]
    device = torch.device(args.device)
    protein_features, protein_ids = load_protein_library(DEFAULT_PROTEINS)
    fold_metrics: dict[int, dict[float, dict[str, float]]] = {}

    for fold in folds:
        cell = f"clean2023_internal_double_cold_fold{fold}"
        model_dir = DEFAULT_MODEL_ROOT / f"fold{fold}"
        schema = load_feature_schema(model_dir)
        reaction_features, reaction_ids = load_registered_reaction_feature_library(DEFAULT_REACTIONS, schema)
        models = load_models(model_dir / "models", "production", device)
        if len(models) != 1:
            raise ValueError(f"Expected one frozen coarse model for fold{fold}, got {len(models)}")
        model = models[0]
        p_emb = encode_library(model, protein_features, kind="protein", device=device)
        r_emb = encode_library(model, reaction_features, kind="reaction", device=device)
        head = load_v1_head(fold, device)
        test_pairs = pd.read_csv(DEFAULT_BENCH / cell / "test_pairs.csv", dtype=str).fillna("")
        fold_metrics[fold] = {}
        v1_summary = json.loads((DEFAULT_V1_RESULTS / f"fold{fold}" / "summary.json").read_text(encoding="utf-8"))

        for scale in scales:
            out = args.output_root.resolve() / f"fold{fold}" / scale_slug(scale)
            out.mkdir(parents=True, exist_ok=True)
            if scale == 0.0:
                # Exact fallback means exact stored coarse ranks, not a fresh GPU TopK
                # reconstruction whose boundary/tie order can differ at float epsilon.
                coarse_pos = pd.read_csv(
                    DEFAULT_COARSE_EVAL / cell / "positive_ranks.csv",
                    dtype={"query_id": str, "positive_id": str},
                )
                coarse_pos = coarse_pos[coarse_pos["direction"] == "reaction_to_enzyme"].copy()
                frame = query_metrics_from_positive_rank_frame(coarse_pos)
                metrics = copy.deepcopy(v1_summary["metrics"])
                metrics["reranked"] = copy.deepcopy(metrics["coarse"])
                for slice_metrics in metrics.get("reaction_similarity_slices", {}).values():
                    slice_metrics["reranked"] = copy.deepcopy(slice_metrics["coarse"])
                support = pd.DataFrame(
                    {
                        "query_id": frame["query_id"].astype(str),
                        "residual_scale": np.zeros(len(frame), dtype=float),
                        "exact_coarse_fallback": np.ones(len(frame), dtype=bool),
                    }
                )
            else:
                frame, metrics, support = evaluate_reranker(
                    head=head,
                    reaction_embeddings=r_emb,
                    protein_embeddings=p_emb,
                    reaction_ids=reaction_ids,
                    protein_ids=protein_ids,
                    test_pairs=test_pairs,
                    coarse_positive_ranks_csv=DEFAULT_COARSE_EVAL / cell / "positive_ranks.csv",
                    coarse_query_metrics_csv=DEFAULT_COARSE_EVAL / cell / "query_metrics.csv",
                    reaction_slices_csv=DEFAULT_DIFFICULTY / cell / "reaction_slices.csv",
                    shortlist_size=int(protocol["shortlist_size"]),
                    residual_scale=scale,
                    device=device,
                )
            frame.to_csv(out / "query_metrics.csv", index=False)
            support.to_csv(out / "support_audit.csv", index=False)
            summary = {
                "fold": fold,
                "cell": cell,
                "residual_scale": scale,
                "outer_labels_used": False,
                "protocol": str(DEFAULT_PROTOCOL.resolve()),
                "protocol_sha256": sha256_file(DEFAULT_PROTOCOL),
                "v1_head": str((DEFAULT_V1_RESULTS / f"fold{fold}" / "pair_residual_head.pt").resolve()),
                "metrics": metrics,
            }
            (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
            fold_metrics[fold][scale] = {metric: float(metrics["reranked"][metric]) for metric in primary_metrics}
            print(f"fold={fold} scale={scale:g} " + " ".join(f"{m}={fold_metrics[fold][scale][m]:.6g}" for m in primary_metrics), flush=True)

    selection = select_residual_scale(
        fold_metrics,
        scales=scales,
        primary_metrics=primary_metrics,
    )
    payload = {
        "protocol": str(DEFAULT_PROTOCOL.resolve()),
        "protocol_sha256": sha256_file(DEFAULT_PROTOCOL),
        "outer_labels_used": False,
        "folds": folds,
        "residual_scales": scales,
        "primary_metrics": primary_metrics,
        "fold_metrics": {
            str(fold): {f"{scale:g}": values for scale, values in metrics.items()}
            for fold, metrics in fold_metrics.items()
        },
        **selection,
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "selection_summary.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
