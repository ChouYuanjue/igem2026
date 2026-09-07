from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import xgboost as xgb

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from projects.active.terpene_screening.broad_rhea_metrics import (  # noqa: E402
    DEFAULT_BUDGETS,
    DEFAULT_TOP_PERCENTS,
    evaluate_full_candidate_ranks,
    summarize_query_metrics,
)
from projects.active.terpene_screening.evaluate_broad_rhea_benchmark import encode_chunks  # noqa: E402
from projects.active.terpene_screening.rank_open_world import (  # noqa: E402
    load_feature_schema,
    load_models,
    load_protein_library,
    load_registered_reaction_feature_library,
)
from projects.active.terpene_screening.run_r2e_lambdarank_fusion_v1 import (  # noqa: E402
    Config,
    PRIMARY_METRICS,
    ROUTER_THRESHOLD,
    _build_features,
    _full_order,
    _geomean_ratio,
    _lexical_rank,
)

PROTOCOL = ROOT / "projects/active/terpene_screening/CATALYST_R2E_LAMBDARANK_FUSION_V1.json"
DEV_RESULT = ROOT / "projects/active/terpene_screening/CATALYST_R2E_LAMBDARANK_FUSION_V1_DEVELOPMENT_RESULT.json"
DEFAULT_ROOT = ROOT / "results/r2e_lambdarank_fusion_v1"
PRIMARY_PROTEINS = ROOT / "data/catalyst_candidate_universes/general_merged/proteins"
SECONDARY_PROTEINS = ROOT / "data/external/enzgfm_current/general_merged_650m_mean_v1"
REACTIONS = ROOT / "data/catalyst_candidate_universes/general_merged/reaction_features/drfp_categorical_rdkitplus_center_v1"
ALL_GATE_METRICS = ("mrr", "map", "macro_roc_auc", "ndcg_at_10", "hit_at_10", "hit_at_20", "hit_at_50")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def metric_map(frame: pd.DataFrame) -> dict[str, float]:
    s = summarize_query_metrics(frame, budgets=DEFAULT_BUDGETS, top_percents=DEFAULT_TOP_PERCENTS)
    return {
        "mrr": float(s["mrr"]),
        "map": float(s["map"]),
        "macro_roc_auc": float(s["macro_roc_auc"]),
        "ndcg_at_10": float(s["ndcg_at_10"]),
        "hit_at_10": float(s["hit_at_10"]),
        "hit_at_20": float(s["hit_at_20"]),
        "hit_at_50": float(s["hit_at_50"]),
        "median_best_positive_rank": float(s["median_best_positive_rank"]),
    }


def selected_config_and_ranker(root: Path) -> tuple[Config, Path, dict[str, object]]:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    dev = json.loads(DEV_RESULT.read_text(encoding="utf-8"))
    if dev["status"] != "passed_frozen_development_material_gate_confirmation_allowed":
        raise RuntimeError("Development result does not allow confirmation")
    selected = dict(dev["selected_config"])
    config = Config(
        str(selected["config_id"]), int(selected["pool_k"]), int(selected["prefix_k"]),
        int(selected["max_depth"]), float(selected["learning_rate"]),
        float(selected["min_child_weight"]), float(selected["reg_lambda"]),
        int(selected["rounds"]), int(selected["lambdarank_pairs"]),
    )
    ranker = root / "selected/ranker.json"
    if sha256_file(ranker) != str(dev["selected_ranker_sha256"]):
        raise RuntimeError("Selected ranker hash differs from frozen development result")
    conf = protocol["confirmation"]
    if str(conf["salt"]) != "r2e_lambdarank_fusion_v1_confirm_20260902" or int(conf["fold"]) != 6 or int(conf["folds"]) != 7:
        raise RuntimeError("Unexpected frozen confirmation split")
    return config, ranker, protocol


def apply_prefix_rerank_positive_ranks(
    *,
    primary_scores: np.ndarray,
    secondary_scores: np.ndarray,
    positive_rows: np.ndarray,
    lexical_rank: np.ndarray,
    similarity: float,
    booster: xgb.Booster,
    config: Config,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    p_order, p_inv = _full_order(primary_scores, lexical_rank)
    s_order, s_inv = _full_order(secondary_scores, lexical_rank)
    use_secondary = float(similarity) < ROUTER_THRESHOLD
    fallback_inv = s_inv if use_secondary else p_inv
    fallback_positive = fallback_inv[positive_rows].astype(np.int64)
    union = np.unique(
        np.concatenate([p_order[: config.pool_k], s_order[: config.pool_k]])
    ).astype(np.int32)
    features = _build_features(
        primary_scores, secondary_scores, union, p_inv, s_inv, use_secondary, float(similarity)
    )
    predicted = booster.predict(xgb.DMatrix(features))
    learned_order = union[np.lexsort((lexical_rank[union], -predicted))]
    selected = learned_order[: min(config.prefix_k, len(learned_order))]
    selected_fb = fallback_inv[selected]
    selected_position = {int(row): i + 1 for i, row in enumerate(selected)}
    reranked: list[int] = []
    for row, fallback_rank in zip(positive_rows, fallback_positive, strict=True):
        row = int(row); fallback_rank = int(fallback_rank)
        if row in selected_position:
            new_rank = selected_position[row]
        else:
            promoted_before = int(np.count_nonzero(selected_fb < fallback_rank))
            new_rank = len(selected) + fallback_rank - promoted_before
        reranked.append(new_rank)
    return fallback_positive, np.asarray(reranked, dtype=np.int64), {
        "use_secondary_fallback": bool(use_secondary),
        "union_size": int(len(union)),
        "selected_prefix_size": int(len(selected)),
        "positive_in_union": int(np.isin(positive_rows, union).sum()),
        "positive_in_selected_prefix": int(np.isin(positive_rows, selected).sum()),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="One-shot frozen confirmation for Catalyst R2E LambdaRank fusion V1")
    ap.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    root = args.root.resolve(); conf_root = root / "confirmation"; device = torch.device(args.device)
    config, ranker_path, protocol = selected_config_and_ranker(root)
    frozen = protocol["confirmation"]; salt = str(frozen["salt"]); fold = int(frozen["fold"])
    cell = f"clean2023_internal_double_cold_salted_{salt}_fold{fold}"
    benchmark = conf_root / "benchmarks" / cell
    manifest = json.loads((benchmark / "manifest.json").read_text(encoding="utf-8"))
    if not bool(manifest.get("valid")) or str(manifest.get("split_salt")) != salt or int(manifest.get("dev_fold")) != fold or int(manifest.get("folds")) != int(frozen["folds"]):
        raise RuntimeError("Confirmation benchmark does not match the frozen valid split")
    test = pd.read_csv(benchmark / "test_pairs.csv", dtype=str).fillna("")
    primary_dir = conf_root / "baseline_center" / f"fold{fold}"
    secondary_dir = conf_root / "candidate_center" / f"fold{fold}"
    primary_features, candidate_ids = load_protein_library(PRIMARY_PROTEINS)
    secondary_features, candidate_ids2 = load_protein_library(SECONDARY_PROTEINS)
    if candidate_ids != candidate_ids2:
        raise ValueError("Protein candidate order differs between source models")
    candidate_index = {value: i for i, value in enumerate(candidate_ids)}
    lexical_rank = _lexical_rank(candidate_ids)
    ps = load_feature_schema(primary_dir); ss = load_feature_schema(secondary_dir)
    reactions, reaction_ids = load_registered_reaction_feature_library(REACTIONS, ps)
    reactions2, reaction_ids2 = load_registered_reaction_feature_library(REACTIONS, ss)
    if reaction_ids != reaction_ids2 or not np.array_equal(reactions, reactions2):
        raise ValueError("Reaction libraries differ between confirmation sources")
    reaction_index = {value: i for i, value in enumerate(reaction_ids)}
    pm = load_models(primary_dir / "models", "production", device)
    sm = load_models(secondary_dir / "models", "production", device)
    if len(pm) != 1 or len(sm) != 1:
        raise ValueError("Confirmation expects one checkpoint per source")
    pe = encode_chunks(pm[0], primary_features, kind="protein", device=device, chunk_size=8192)
    se = encode_chunks(sm[0], secondary_features, kind="protein", device=device, chunk_size=8192)
    pre = encode_chunks(pm[0], reactions, kind="reaction", device=device, chunk_size=8192)
    sre = encode_chunks(sm[0], reactions, kind="reaction", device=device, chunk_size=8192)
    booster = xgb.Booster(); booster.load_model(ranker_path)
    diff = pd.read_csv(conf_root / "difficulty" / cell / "reaction_slices.csv", dtype={"reaction_id": str})
    similarity = dict(zip(diff["reaction_id"].astype(str), diff["max_train_drfp_tanimoto"].astype(float)))
    positives = test.groupby("reaction_id")["protein_id"].apply(lambda x: set(map(str, x))).to_dict()
    query_ids = sorted(positives)
    rows = [reaction_index[q] for q in query_ids]
    baseline_records: list[dict[str, object]] = []
    candidate_records: list[dict[str, object]] = []
    audits: list[dict[str, object]] = []
    for start in range(0, len(query_ids), 32):
        stop = min(start + 32, len(query_ids))
        qrows = torch.as_tensor(rows[start:stop], dtype=torch.long, device=device)
        with torch.no_grad():
            pscore = (pre[qrows] @ pe.T).float().cpu().numpy()
            sscore = (sre[qrows] @ se.T).float().cpu().numpy()
        for local, q in enumerate(query_ids[start:stop]):
            pos_rows = np.asarray(sorted(candidate_index[x] for x in positives[q]), dtype=np.int32)
            base_ranks, cand_ranks, audit = apply_prefix_rerank_positive_ranks(
                primary_scores=pscore[local], secondary_scores=sscore[local],
                positive_rows=pos_rows, lexical_rank=lexical_rank,
                similarity=float(similarity[q]), booster=booster, config=config,
            )
            bm = evaluate_full_candidate_ranks(base_ranks, len(candidate_ids))
            cm = evaluate_full_candidate_ranks(cand_ranks, len(candidate_ids))
            baseline_records.append({"query_id": q, **bm})
            candidate_records.append({"query_id": q, **cm})
            audits.append({"query_id": q, "similarity": float(similarity[q]), **audit})
        print(f"confirmation {stop}/{len(query_ids)}", flush=True)
    baseline_frame = pd.DataFrame(baseline_records); candidate_frame = pd.DataFrame(candidate_records)
    baseline = metric_map(baseline_frame); candidate = metric_map(candidate_frame)
    delta = {k: candidate[k] - baseline[k] for k in ALL_GATE_METRICS}
    no_regression = {k: bool(delta[k] >= -1e-12) for k in ALL_GATE_METRICS}
    gm = _geomean_ratio(candidate, baseline)
    material = bool(delta["mrr"] >= 0.003 or delta["map"] >= 0.003 or delta["hit_at_10"] >= 0.01)
    checks = {
        **{f"no_regression_{k}": value for k, value in no_regression.items()},
        "primary_geomean_ratio_gt_1": bool(gm > 1.0),
        "material_gate": material,
    }
    passed = bool(all(checks.values()))
    output = conf_root / "lambdarank"; output.mkdir(parents=True, exist_ok=True)
    merged = baseline_frame.merge(candidate_frame, on="query_id", suffixes=("_baseline", "_candidate"), validate="one_to_one").merge(pd.DataFrame(audits), on="query_id", validate="one_to_one")
    merged.to_csv(output / "query_metrics.csv", index=False)
    pd.DataFrame(audits).to_csv(output / "audit.csv", index=False)
    summary = {
        "status": "passed_frozen_confirmation" if passed else "failed_frozen_confirmation",
        "pass": passed,
        "split_salt": salt,
        "fold": fold,
        "cell": cell,
        "query_count": len(query_ids),
        "candidate_count": len(candidate_ids),
        "selected_config": config.__dict__,
        "selected_ranker": str(ranker_path),
        "selected_ranker_sha256": sha256_file(ranker_path),
        "benchmark_manifest_sha256": sha256_file(benchmark / "manifest.json"),
        "baseline": baseline,
        "candidate": candidate,
        "delta": delta,
        "primary_geomean_ratio": gm,
        "checks": checks,
        "candidate_fraction_low_similarity": float(pd.DataFrame(audits)["use_secondary_fallback"].mean()),
        "mean_union_size": float(pd.DataFrame(audits)["union_size"].mean()),
        "mean_positive_in_union": float(pd.DataFrame(audits)["positive_in_union"].mean()),
        "external_or_outer_metrics_used": False,
        "retuning_after_reveal_allowed": False,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    if not passed:
        raise SystemExit(2)

if __name__ == "__main__":
    main()
