from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from projects.active.terpene_screening import run_bime_r2e_clipzyme_expert_v1 as clipmod
from projects.active.terpene_screening import run_r2e_lambdarank_fusion_v1 as base

OUT = ROOT / "results/bime_rank_unified_v1/cost_aware_shortlist_retention_v1"
FOLDS = (0, 1, 2)
K_VALUES = (20, 50, 100, 200, 500, 1000, 2000)
SPECIALIST_K = 100


def _one_fold(fold: int, device_name: str) -> pd.DataFrame:
    device = torch.device(device_name)
    ids, _, reaction_ids, pe0, pe1, re0, re1 = base._load_fold_embeddings(fold, device)
    candidate_index = {p: i for i, p in enumerate(ids)}
    reaction_index = {r: i for i, r in enumerate(reaction_ids)}
    lex = base._lexical_rank(ids)
    clip_pt, clip_candidate_rows, _, clip_rmat, clip_ridx = clipmod._load_clip_assets(ids, device)
    clip_lex = lex[clip_candidate_rows]

    dev_pairs = pd.read_csv(base.DEV_ROOT / "baseline_base" / f"fold{fold}" / "dev_pairs.csv", dtype=str).fillna("")
    query_ids = sorted(dev_pairs["reaction_id"].astype(str).unique())
    positives = dev_pairs.groupby("reaction_id")["protein_id"].apply(lambda x: set(map(str, x))).to_dict()
    qrows = [reaction_index[q] for q in query_ids]

    records: list[dict[str, object]] = []
    batch = 32
    for st in range(0, len(query_ids), batch):
        stop = min(st + batch, len(query_ids))
        rows_t = torch.as_tensor(qrows[st:stop], dtype=torch.long, device=device)
        with torch.no_grad():
            s0b = (re0[rows_t] @ pe0.T).float().cpu().numpy()
            s1b = (re1[rows_t] @ pe1.T).float().cpu().numpy()

        supported_local: list[int] = []
        clip_q: list[np.ndarray] = []
        for j, q in enumerate(query_ids[st:stop]):
            if q in clip_ridx:
                supported_local.append(j)
                clip_q.append(np.asarray(clip_rmat[clip_ridx[q]], dtype=np.float32))
        clip_scores_by_local: dict[int, np.ndarray] = {}
        if clip_q:
            qt = torch.as_tensor(np.stack(clip_q), dtype=torch.float32, device=device)
            with torch.no_grad():
                cb = (qt @ clip_pt.T).float().cpu().numpy()
            for row, j in enumerate(supported_local):
                clip_scores_by_local[j] = cb[row].astype(np.float32, copy=False)

        for j, q in enumerate(query_ids[st:stop]):
            positive_rows = np.asarray(sorted(candidate_index[p] for p in positives[q]), dtype=np.int32)
            s0 = s0b[j]
            s1 = s1b[j]
            o0 = np.lexsort((lex, -s0)).astype(np.int32)
            o1 = np.lexsort((lex, -s1)).astype(np.int32)
            query_supported = j in clip_scores_by_local
            clip_top = np.empty(0, dtype=np.int32)
            if query_supported:
                cs = clip_scores_by_local[j]
                co = np.lexsort((clip_lex, -cs)).astype(np.int32)
                clip_top = clip_candidate_rows[co[:SPECIALIST_K]]
            full_env = np.unique(np.concatenate([o0[:100], o1[:100], clip_top])).astype(np.int32)
            full_hit = bool(np.isin(positive_rows, full_env).any())
            base100 = np.unique(np.concatenate([o0[:100], o1[:100]])).astype(np.int32)
            base100_hit = bool(np.isin(positive_rows, base100).any())
            specialist_rescue = bool(full_hit and not base100_hit)
            rec: dict[str, object] = {
                "fold": fold,
                "query_id": q,
                "positives": int(len(positive_rows)),
                "clip_query_supported": bool(query_supported),
                "full_specialist_envelope_hit": full_hit,
                "base100_hit": base100_hit,
                "specialist_rescue_over_base100": specialist_rescue,
            }
            for k in K_VALUES:
                pool = np.unique(np.concatenate([o0[:k], o1[:k]])).astype(np.int32)
                rec[f"base{k}_hit"] = bool(np.isin(positive_rows, pool).any())
                rec[f"cliptop100_in_base{k}"] = int(np.isin(clip_top, pool).sum()) if query_supported else 0
            records.append(rec)
        print(f"fold={fold} {stop}/{len(query_ids)}", flush=True)
    return pd.DataFrame(records)


def summarize(frame: pd.DataFrame) -> dict[str, object]:
    supported = frame[frame.clip_query_supported.astype(bool)].copy()
    full_positive = supported[supported.full_specialist_envelope_hit.astype(bool)].copy()
    rescue = supported[supported.specialist_rescue_over_base100.astype(bool)].copy()
    rows: list[dict[str, object]] = []
    for k in K_VALUES:
        col = f"base{k}_hit"
        retained = int(full_positive[col].astype(bool).sum())
        rescue_retained = int(rescue[col].astype(bool).sum())
        overlap = supported[f"cliptop100_in_base{k}"].astype(float)
        rows.append({
            "k_per_base_expert": k,
            "max_base_pool_candidates": 2 * k,
            "max_pool_fraction_of_185918": (2.0 * k) / 185918.0,
            "all_supported_query_hit_rate": float(supported[col].astype(bool).mean()),
            "retention_of_full_specialist_positive_queries": float(retained / len(full_positive)) if len(full_positive) else None,
            "specialist_rescue_queries_retained_without_specialist_candidate_discovery": float(rescue_retained / len(rescue)) if len(rescue) else None,
            "mean_clip_top100_overlap": float(overlap.mean() / SPECIALIST_K),
            "median_clip_top100_overlap": float(overlap.median() / SPECIALIST_K),
        })
    return {
        "status": "complete",
        "protocol": "clean internal R2E double-cold development folds; no external labels; compare base ESM-C/EnzGFM top-K envelope with current structural candidate envelope (base top100 union CLIPZyme top100)",
        "query_count": int(len(frame)),
        "clip_supported_queries": int(len(supported)),
        "full_specialist_positive_queries": int(len(full_positive)),
        "base100_positive_queries": int(supported.base100_hit.astype(bool).sum()),
        "specialist_rescue_queries_over_base100": int(len(rescue)),
        "specialist_k": SPECIALIST_K,
        "sweep": rows,
        "interpretation_boundary": "This is shortlist/oracle-retention evidence for on-demand specialist execution. It does not claim that CLIPZyme itself should be restricted at runtime when its candidate embeddings are already cached, and it does not admit EnzymeCAGE as an active expert.",
    }


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    frames = [_one_fold(f, args.device) for f in FOLDS]
    frame = pd.concat(frames, ignore_index=True)
    OUT.mkdir(parents=True, exist_ok=True)
    frame.to_csv(OUT / "query_retention.csv", index=False)
    payload = summarize(frame)
    (OUT / "summary.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
