from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import rankdata

from projects.active.terpene_screening.rank_open_world import load_feature_schema, load_models, load_protein_library

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_POSITIVE_SUPPORT = ROOT / "results/clipzyme_protein_support_v1/reactzyme_reaction_projected_double_cold_protein_support.csv"
DEFAULT_CLIP_SCREEN = ROOT / "external_models/clipzyme_audit/clipzyme_data/clipzyme_screening_set.p"
DEFAULT_CLIP_REACTIONS = ROOT / "data/external/clipzyme_current/general_merged_reaction_embeddings_v1"
DEFAULT_UNIVERSE = ROOT / "data/catalyst_candidate_universes/general_merged"
DEFAULT_CATALYST_REACTIONS = DEFAULT_UNIVERSE / "reaction_features/drfp_categorical_rdkitplus_center_v1"
DEFAULT_MODEL = ROOT / "results/catalyst_clean_mainline_v1/r2e_center_bounded_cap0p1"
DEFAULT_OUTPUT = ROOT / "results/clipzyme_catalyst_common_support_v1"


def _norm(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return x / n


def _load_entries_feature_dir(path: Path, filename: str) -> tuple[list[str], np.ndarray]:
    entries = pd.read_csv(path / "entries.csv", dtype=str).fillna("").sort_values("row")
    ids = entries[filename].astype(str).tolist()
    mat = np.load(path / ("embeddings.npy" if filename == "Entry" else "reaction_feature_matrix.npy"), mmap_mode="r")
    if len(ids) != len(mat):
        raise ValueError(f"entry/feature row mismatch: {path}")
    return ids, mat


def _metrics(score: np.ndarray, query_ids: list[str], candidate_ids: list[str], positives: dict[str, set[str]]) -> tuple[dict[str, float | int], pd.DataFrame]:
    cindex = {c: i for i, c in enumerate(candidate_ids)}
    rows: list[dict[str, float | int | str]] = []
    for qi, q in enumerate(query_ids):
        pos_ids = positives.get(q, set()) & cindex.keys()
        if not pos_ids:
            continue
        pos_idx = np.fromiter((cindex[p] for p in pos_ids), dtype=np.int64)
        s = np.asarray(score[qi], dtype=np.float64)
        # Average ascending ranks make AUROC tie-aware; descending ordinal ranks drive Top-K metrics.
        asc = rankdata(s, method="average")
        p = len(pos_idx); nneg = len(s) - p
        auc = float((asc[pos_idx].sum() - p * (p + 1) / 2.0) / (p * nneg)) if nneg > 0 else float("nan")
        order = np.argsort(-s, kind="stable")
        inv = np.empty_like(order)
        inv[order] = np.arange(len(order))
        ranks = np.sort(inv[pos_idx] + 1)
        mrr = 1.0 / float(ranks[0])
        ap = float(np.mean(np.arange(1, p + 1, dtype=np.float64) / ranks))
        k = min(10, len(s))
        gains = (np.isin(order[:k], pos_idx)).astype(np.float64)
        discounts = 1.0 / np.log2(np.arange(2, k + 2, dtype=np.float64))
        dcg = float((gains * discounts).sum())
        ideal = float(discounts[: min(p, k)].sum())
        ndcg10 = dcg / ideal if ideal else 0.0
        row: dict[str, float | int | str] = {
            "query_id": q,
            "positive_count": p,
            "candidate_count": len(s),
            "mrr": mrr,
            "map": ap,
            "macro_roc_auc": auc,
            "ndcg_at_10": ndcg10,
            "best_positive_rank": int(ranks[0]),
            "mean_positive_rank": float(ranks.mean()),
        }
        for budget in (1, 5, 10, 20, 50):
            row[f"hit_at_{budget}"] = float(ranks[0] <= budget)
        rows.append(row)
    qdf = pd.DataFrame(rows)
    if qdf.empty:
        raise ValueError("no evaluable queries")
    summary: dict[str, float | int] = {
        "query_count": int(len(qdf)),
        "candidate_count": int(qdf["candidate_count"].iloc[0]),
        "positive_rows": int(qdf["positive_count"].sum()),
        "mrr": float(qdf["mrr"].mean()),
        "map": float(qdf["map"].mean()),
        "macro_roc_auc": float(qdf["macro_roc_auc"].mean()),
        "ndcg_at_10": float(qdf["ndcg_at_10"].mean()),
        "median_best_positive_rank": float(qdf["best_positive_rank"].median()),
        "mean_positive_rank": float((qdf["mean_positive_rank"] * qdf["positive_count"]).sum() / qdf["positive_count"].sum()),
    }
    for budget in (1, 5, 10, 20, 50):
        summary[f"hit_at_{budget}"] = float(qdf[f"hit_at_{budget}"].mean())
    return summary, qdf


def main() -> None:
    ap = argparse.ArgumentParser(description="Official native CLIPZyme vs Catalyst V3 on frozen priority-1 common support.")
    ap.add_argument("--positive-support", type=Path, default=DEFAULT_POSITIVE_SUPPORT)
    ap.add_argument("--clip-screen", type=Path, default=DEFAULT_CLIP_SCREEN)
    ap.add_argument("--clip-reactions", type=Path, default=DEFAULT_CLIP_REACTIONS)
    ap.add_argument("--universe", type=Path, default=DEFAULT_UNIVERSE)
    ap.add_argument("--catalyst-reactions", type=Path, default=DEFAULT_CATALYST_REACTIONS)
    ap.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL)
    ap.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    pos = pd.read_csv(args.positive_support, dtype=str).fillna("")
    pos = pos[pos["common_executable_positive"].astype(str).str.lower().eq("true")][["reaction_id", "protein_id"]].drop_duplicates()
    if pos["reaction_id"].nunique() != 109 or pos["protein_id"].nunique() != 4161 or len(pos) != 4915:
        raise AssertionError("frozen priority-1 positive support drift")

    clip = pickle.load(args.clip_screen.open("rb"))
    clip_p_ids = [str(x) for x in clip["uniprots"]]
    clip_p = np.asarray(clip["hiddens"], dtype=np.float32)
    clip_p_index = {x: i for i, x in enumerate(clip_p_ids)}

    cat_p_raw, cat_p_ids = load_protein_library(args.universe / "proteins")
    cat_p_index = {x: i for i, x in enumerate(cat_p_ids)}
    common_p = sorted(set(clip_p_ids) & set(cat_p_ids))

    clip_re = pd.read_csv(args.clip_reactions / "entries.csv", dtype=str).fillna("").sort_values("row")
    clip_r_all = np.load(args.clip_reactions / "embeddings.npy", mmap_mode="r")
    clip_supported = clip_re[clip_re["clipzyme_supported"].astype(str).str.lower().eq("true")]
    clip_r_index = {rid: int(row) for rid, row in clip_supported[["reaction_id", "row"]].itertuples(index=False)}

    cat_re = pd.read_csv(args.catalyst_reactions / "entries.csv", dtype=str).fillna("").sort_values("row")
    cat_r_all = np.load(args.catalyst_reactions / "reaction_feature_matrix.npy", mmap_mode="r")
    cat_r_index = {rid: int(row) for rid, row in cat_re[["reaction_id", "row"]].itertuples(index=False)}
    common_r = sorted(set(clip_r_index) & set(cat_r_index))

    r2e_queries = sorted(pos["reaction_id"].unique())
    e2r_queries = sorted(pos["protein_id"].unique())
    if not set(r2e_queries) <= set(common_r) or not set(e2r_queries) <= set(common_p):
        raise AssertionError("positive query lost from common candidate support")

    # Author model features are already normalized by native CLIPZyme encoders; re-normalization here
    # is numerically idempotent and protects against serialization roundoff only.
    clip_p_common = _norm(clip_p[[clip_p_index[x] for x in common_p]])
    clip_r_common = _norm(np.asarray(clip_r_all[[clip_r_index[x] for x in common_r]], dtype=np.float32))
    clip_r_query = _norm(np.asarray(clip_r_all[[clip_r_index[x] for x in r2e_queries]], dtype=np.float32))
    clip_p_query = _norm(clip_p[[clip_p_index[x] for x in e2r_queries]])

    clip_r2e = clip_r_query @ clip_p_common.T
    clip_e2r = clip_p_query @ clip_r_common.T

    device = torch.device(args.device)
    schema = load_feature_schema(args.model_dir)
    if int(schema["protein_feature_dimension"]) != int(cat_p_raw.shape[1]):
        raise ValueError("Catalyst protein feature dimension mismatch")
    if int(schema["reaction_feature_dimension"]) != int(cat_r_all.shape[1]):
        raise ValueError("Catalyst reaction feature dimension mismatch")
    models = load_models(args.model_dir / "models", "production", device)
    if len(models) != 1:
        raise ValueError("clean mainline V3 expected exactly one production member")
    model = models[0]
    with torch.no_grad():
        p_raw = torch.as_tensor(np.asarray(cat_p_raw[[cat_p_index[x] for x in common_p]], dtype=np.float32), device=device)
        r_raw = torch.as_tensor(np.asarray(cat_r_all[[cat_r_index[x] for x in common_r]], dtype=np.float32), device=device)
        p_emb = model.encode_proteins(p_raw)
        r_emb = model.encode_reactions(r_raw)
        r2e_q_idx = torch.as_tensor([common_r.index(x) for x in r2e_queries], dtype=torch.long, device=device)
        e2r_q_idx = torch.as_tensor([common_p.index(x) for x in e2r_queries], dtype=torch.long, device=device)
        cat_r2e = (r_emb[r2e_q_idx] @ p_emb.T).cpu().numpy()
        cat_e2r = (p_emb[e2r_q_idx] @ r_emb.T).cpu().numpy()

    r2e_pos = pos.groupby("reaction_id")["protein_id"].agg(lambda s: set(s.astype(str))).to_dict()
    e2r_pos = pos.groupby("protein_id")["reaction_id"].agg(lambda s: set(s.astype(str))).to_dict()
    out = args.output_dir; out.mkdir(parents=True, exist_ok=True)
    result: dict[str, object] = {
        "status": "descriptive_revealed_fallback_same_support_no_model_selection",
        "cell": "reactzyme_reaction_projected_double_cold",
        "selection_priority": 1,
        "common_support": {
            "protein_candidates": len(common_p), "reaction_candidates": len(common_r),
            "positive_rows": len(pos), "r2e_queries": len(r2e_queries), "e2r_queries": len(e2r_queries),
        },
        "score_semantics": {
            "clipzyme": "dot product of official normalized native reaction/protein hiddens",
            "catalyst": "dot product of frozen Catalyst-Clean-Mainline-v1 V3 embeddings",
        },
        "models": {},
        "selection_allowed": False,
        "router_tuning_allowed": False,
    }
    for name, r2e_score, e2r_score in [("official_clipzyme", clip_r2e, clip_e2r), ("catalyst_v3", cat_r2e, cat_e2r)]:
        rsum, rdf = _metrics(r2e_score, r2e_queries, common_p, r2e_pos)
        esum, edf = _metrics(e2r_score, e2r_queries, common_r, e2r_pos)
        rdf.to_csv(out / f"{name}_r2e_query_metrics.csv", index=False)
        edf.to_csv(out / f"{name}_e2r_query_metrics.csv", index=False)
        result["models"][name] = {"reaction_to_enzyme": rsum, "enzyme_to_reaction": esum}
    deltas: dict[str, dict[str, float]] = {}
    for direction in ("reaction_to_enzyme", "enzyme_to_reaction"):
        a = result["models"]["official_clipzyme"][direction]
        b = result["models"]["catalyst_v3"][direction]
        deltas[direction] = {k: float(b[k]) - float(a[k]) for k in a if isinstance(a[k], (int, float)) and k in b and k not in {"query_count", "candidate_count", "positive_rows"}}
    result["catalyst_minus_clipzyme"] = deltas
    (out / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
