from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from projects.active.terpene_screening.enzgfm_native_same_support_v1_common import (
    AuthorPairwiseNative, DualTowerNative, PREPARED, RESULT_ROOT, build_label_matrix, load_protein_memmap, query_metrics, sha256_file,
)

PAPER = {
    "e2r_map": 0.5156,
    "e2r_hit_at_5": 0.6636,
    "r2e_map": 0.8211,
    "source": "EnzGFM Nature Communications 2026, ReactZyme enzyme-similarity split",
    "note": "Only unambiguous same-task values are used for direct deltas; paper NDCG is not assigned a K here.",
}


def load_model(candidate: str, stage: str, device: torch.device):
    path = RESULT_ROOT / stage / candidate / "model.pt"
    payload = torch.load(path, map_location=device, weights_only=False)
    if payload["candidate"] != candidate or payload["stage"] != stage:
        raise AssertionError("checkpoint provenance mismatch")
    if candidate == "dual_tower":
        model = DualTowerNative().to(device)
    elif candidate == "author_pairwise":
        model = AuthorPairwiseNative().to(device)
    else:
        raise ValueError(candidate)
    model.load_state_dict(payload["state_dict"]); model.eval()
    return model, path, payload


def score_dual(model: DualTowerNative, reactions: np.ndarray, proteins: np.ndarray, device: torch.device) -> np.ndarray:
    rr, pp = [], []
    with torch.inference_mode():
        for s in range(0, len(reactions), 1024):
            rr.append(model.encode_reactions(torch.from_numpy(np.asarray(reactions[s:s+1024], dtype=np.float32)).to(device)).cpu().numpy())
        for s in range(0, len(proteins), 2048):
            pp.append(model.encode_proteins(torch.from_numpy(np.asarray(proteins[s:s+2048], dtype=np.float32)).to(device)).cpu().numpy())
    r = np.concatenate(rr).astype(np.float32); p = np.concatenate(pp).astype(np.float32)
    return (p @ r.T).astype(np.float32)


def score_pairwise(model: AuthorPairwiseNative, reactions: np.ndarray, proteins: np.ndarray, device: torch.device) -> np.ndarray:
    n_p, n_r = len(proteins), len(reactions)
    out = np.empty((n_p, n_r), dtype=np.float32)
    batch = 8192
    with torch.inference_mode():
        total = n_p * n_r
        for start in range(0, total, batch):
            stop = min(total, start + batch)
            flat = np.arange(start, stop, dtype=np.int64)
            pi = flat // n_r; ri = flat % n_r
            rx = torch.from_numpy(np.asarray(reactions[ri], dtype=np.float32)).to(device)
            px = torch.from_numpy(np.asarray(proteins[pi], dtype=np.float32)).to(device)
            out.reshape(-1)[start:stop] = model(rx, px).float().cpu().numpy()
            if start and start % (batch * 100) == 0:
                print(f"pairwise scoring {start}/{total}", flush=True)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidate", choices=("dual_tower", "author_pairwise"), required=True)
    ap.add_argument("--stage", choices=("development", "final"), required=True)
    args = ap.parse_args()
    split = "dev" if args.stage == "development" else "test"
    out_dir = RESULT_ROOT / args.stage / args.candidate
    out_path = out_dir / f"{split}_evaluation.json"
    if out_path.exists():
        raise SystemExit(f"refusing to overwrite frozen evaluation: {out_path}")
    if args.stage == "final":
        sel_path = RESULT_ROOT / "selection/summary.json"
        if not sel_path.exists():
            raise SystemExit("final reveal blocked: frozen development selection missing")
        sel = json.loads(sel_path.read_text())
        if sel.get("selected_candidate") != args.candidate:
            raise SystemExit("final reveal blocked: candidate is not the frozen selected candidate")
        # Final reveal is allowed only after final retraining, never with the development checkpoint.
        if not (RESULT_ROOT / "final" / args.candidate / "summary.json").exists():
            raise SystemExit("final reveal blocked: final all-train model missing")

    if split == "dev":
        proteins_df = pd.read_csv(PREPARED / "dev_eval_proteins.csv", dtype={"protein_row": int})
        reactions_df = pd.read_csv(PREPARED / "dev_eval_reactions.csv", dtype={"reaction_idx": int})
        pairs = pd.read_csv(PREPARED / "dev_pairs.csv", dtype={"protein_row": int, "reaction_idx": int})
    else:
        proteins_df = pd.read_csv(PREPARED / "test_proteins.csv", dtype={"protein_row": int})
        reactions_df = pd.read_csv(PREPARED / "test_reactions.csv", dtype={"reaction_idx": int})
        pairs = pd.read_csv(PREPARED / "test_pairs.csv", dtype={"protein_row": int, "reaction_idx": int})

    reaction_all = np.load(PREPARED / "reaction_features.npy", mmap_mode="r")
    protein_all = load_protein_memmap()
    reactions = np.asarray(reaction_all[reactions_df.reaction_idx.to_numpy(np.int64)], dtype=np.float32)
    proteins = np.asarray(protein_all[proteins_df.protein_row.to_numpy(np.int64)], dtype=np.float32)
    labels = build_label_matrix(pairs, proteins_df, reactions_df)
    if split == "test":
        if labels.shape != (8734, 1573) or int(labels.sum()) != 8739:
            raise AssertionError(f"native support drift: shape={labels.shape} positives={labels.sum()}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, model_path, payload = load_model(args.candidate, args.stage, device)
    scores = score_dual(model, reactions, proteins, device) if args.candidate == "dual_tower" else score_pairwise(model, reactions, proteins, device)
    if scores.shape != labels.shape:
        raise AssertionError("score matrix shape mismatch")
    e2r = query_metrics(scores, labels)
    r2e = query_metrics(scores.T, labels.T)

    result = {
        "status": "development_internal_only" if split == "dev" else "revealed_native_external_frozen_no_retuning",
        "candidate": args.candidate,
        "stage": args.stage,
        "support": {"protein_queries": len(proteins_df), "reaction_queries": len(reactions_df), "positive_pairs": int(labels.sum())},
        "e2r": e2r,
        "r2e": r2e,
        "model_sha256": sha256_file(model_path),
        "test_performance_used_for_model_selection": False,
    }
    if split == "test":
        np.save(out_dir / "native_test_scores.npy", scores.astype(np.float32))
        result["authoritative_external_baseline"] = "EnzGFM-1.5B"
        result["paper_reference"] = PAPER
        result["direct_same_support_paper_deltas"] = {
            "e2r_map": float(e2r["map"] - PAPER["e2r_map"]),
            "e2r_hit_at_5": float(e2r["hit_at_5"] - PAPER["e2r_hit_at_5"]),
            "r2e_map": float(r2e["map"] - PAPER["r2e_map"]),
        }
        result["material_gain_gt_5pp"] = {
            "e2r_hit_at_5": bool(e2r["hit_at_5"] - PAPER["e2r_hit_at_5"] > 0.05),
        }
        result["post_reveal_retuning_allowed"] = False
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
