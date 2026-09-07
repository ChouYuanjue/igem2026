from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

from projects.active.terpene_screening.enzgfm_native_same_support_v1_common import (
    AuthorPairwiseNative, DualTowerNative, PREPARED, RESULT_ROOT, SEED, load_protein_memmap, seed_all, sha256_file,
)

CANDIDATES = ("dual_tower", "author_pairwise")


def multi_positive_contrastive(sim: torch.Tensor, reaction_idx: torch.Tensor, protein_row: torch.Tensor) -> torch.Tensor:
    positive = (reaction_idx[:, None] == reaction_idx[None, :]) | (protein_row[:, None] == protein_row[None, :])
    neg_inf = torch.finfo(sim.dtype).min
    row_pos = torch.logsumexp(sim.masked_fill(~positive, neg_inf), dim=1)
    row_all = torch.logsumexp(sim, dim=1)
    col_pos = torch.logsumexp(sim.T.masked_fill(~positive.T, neg_inf), dim=1)
    col_all = torch.logsumexp(sim.T, dim=1)
    return 0.5 * ((row_all - row_pos).mean() + (col_all - col_pos).mean())


def train_dual(pairs: pd.DataFrame, reaction_features: np.ndarray, protein_features: np.ndarray, device: torch.device):
    model = DualTowerNative().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    rng = np.random.default_rng(SEED)
    history = []
    batch_size, epochs, temperature = 512, 8, 0.05
    rid = pairs.reaction_idx.to_numpy(np.int64)
    pro = pairs.protein_row.to_numpy(np.int64)
    for epoch in range(1, epochs + 1):
        model.train(); order = rng.permutation(len(pairs)); losses = []
        for start in range(0, len(order), batch_size):
            idx = order[start:start + batch_size]
            if len(idx) < 2:
                continue
            rids, pros = rid[idx], pro[idx]
            rx = torch.from_numpy(np.asarray(reaction_features[rids], dtype=np.float32)).to(device)
            px = torch.from_numpy(np.asarray(protein_features[pros], dtype=np.float32)).to(device)
            rr = model.encode_reactions(rx); pp = model.encode_proteins(px)
            sim = rr @ pp.T / temperature
            loss = multi_positive_contrastive(sim, torch.from_numpy(rids).to(device), torch.from_numpy(pros).to(device))
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); losses.append(float(loss.detach().cpu()))
        history.append({"epoch": epoch, "loss": float(np.mean(losses))})
        print(f"dual epoch={epoch} loss={history[-1]['loss']:.6f}", flush=True)
    return model, history, {"epochs": epochs, "batch_size": batch_size, "lr": 3e-4, "weight_decay": 1e-4, "temperature": temperature}


def train_pairwise(pairs: pd.DataFrame, reaction_features: np.ndarray, protein_features: np.ndarray, device: torch.device):
    model = AuthorPairwiseNative().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=5e-10)
    rng = np.random.default_rng(SEED)
    history = []
    batch_size, epochs = 256, 4
    rid = pairs.reaction_idx.to_numpy(np.int64)
    pro = pairs.protein_row.to_numpy(np.int64)
    reaction_pool = np.unique(rid); protein_pool = np.unique(pro)
    positive_pairs = set(zip(rid.tolist(), pro.tolist()))

    def sample_negatives(rids: np.ndarray, pros: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        neg_p = rng.choice(protein_pool, size=len(rids), replace=True)
        neg_r = rng.choice(reaction_pool, size=len(rids), replace=True)
        for i in range(len(rids)):
            tries = 0
            while (int(rids[i]), int(neg_p[i])) in positive_pairs:
                neg_p[i] = rng.choice(protein_pool); tries += 1
                if tries > 1000: raise RuntimeError("protein-negative rejection exhausted")
            tries = 0
            while (int(neg_r[i]), int(pros[i])) in positive_pairs:
                neg_r[i] = rng.choice(reaction_pool); tries += 1
                if tries > 1000: raise RuntimeError("reaction-negative rejection exhausted")
        return neg_p.astype(np.int64), neg_r.astype(np.int64)

    for epoch in range(1, epochs + 1):
        model.train(); order = rng.permutation(len(pairs)); losses = []
        for start in range(0, len(order), batch_size):
            idx = order[start:start + batch_size]
            if len(idx) < 2:
                continue
            rids, pros = rid[idx], pro[idx]
            neg_p, neg_r = sample_negatives(rids, pros)
            rr = np.concatenate([rids, rids, neg_r])
            pp = np.concatenate([pros, neg_p, pros])
            yy = np.concatenate([np.ones(len(idx), np.float32), np.zeros(2 * len(idx), np.float32)])
            perm = rng.permutation(len(yy)); rr, pp, yy = rr[perm], pp[perm], yy[perm]
            rx = torch.from_numpy(np.asarray(reaction_features[rr], dtype=np.float32)).to(device)
            px = torch.from_numpy(np.asarray(protein_features[pp], dtype=np.float32)).to(device)
            y = torch.from_numpy(yy).to(device)
            logits = model(rx, px)
            loss = F.binary_cross_entropy_with_logits(logits, y)
            opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step()
            losses.append(float(loss.detach().cpu()))
        history.append({"epoch": epoch, "loss": float(np.mean(losses))})
        print(f"pairwise epoch={epoch} loss={history[-1]['loss']:.6f}", flush=True)
    return model, history, {"epochs": epochs, "batch_size": batch_size, "lr": 1e-4, "weight_decay": 5e-10, "negatives_per_positive": 2, "hidden": 128, "dropout": 0.0}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidate", choices=CANDIDATES, required=True)
    ap.add_argument("--stage", choices=("development", "final"), required=True)
    args = ap.parse_args()
    seed_all(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pair_file = PREPARED / ("train_pairs.csv" if args.stage == "development" else "all_train_pairs.csv")
    pairs = pd.read_csv(pair_file, dtype={"reaction_idx": int, "protein_row": int})
    reaction_features = np.load(PREPARED / "reaction_features.npy", mmap_mode="r")
    protein_features = load_protein_memmap()
    if args.candidate == "dual_tower":
        model, history, config = train_dual(pairs, reaction_features, protein_features, device)
        model_type = "dual_tower_native_v1"
    else:
        model, history, config = train_pairwise(pairs, reaction_features, protein_features, device)
        model_type = "author_pairwise_native_v1"
    out = RESULT_ROOT / args.stage / args.candidate
    out.mkdir(parents=True, exist_ok=True)
    torch.save({"model_type": model_type, "candidate": args.candidate, "stage": args.stage, "seed": SEED, "config": config, "state_dict": model.state_dict()}, out / "model.pt")
    pd.DataFrame(history).to_csv(out / "training_history.csv", index=False)
    summary = {
        "status": "trained_without_native_test_scores",
        "candidate": args.candidate,
        "stage": args.stage,
        "model_type": model_type,
        "training_pairs": int(len(pairs)),
        "unique_training_reactions": int(pairs.reaction_idx.nunique()),
        "unique_training_proteins": int(pairs.protein_row.nunique()),
        "pair_source": str(pair_file),
        "pair_source_sha256": sha256_file(pair_file),
        "config": config,
        "seed": SEED,
        "target_native_test_pairs_read": False,
        "target_native_test_scores_read": False,
        "target_native_test_performance_used_for_training_or_selection": False,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
