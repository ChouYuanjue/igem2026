from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from projects.active.bridge.model import ExpertSpec, FibreRelationalModel
from projects.active.bridge.model.assets import FibreAssetStore, ROOT


DEFAULT_PAIRS = ROOT / "data/external/enzymecage_current/catalyst_features/clean2023/training_pairs.csv"
DEFAULT_OUTPUT = ROOT / "results/fibre_relational_main_v1"


def stable_fold(value: str, folds: int = 10) -> int:
    digest = hashlib.blake2b(str(value).encode(), digest_size=8).digest()
    return int.from_bytes(digest, "big") % folds


def split_double_cold(
    pairs: pd.DataFrame,
    *,
    folds: int = 10,
    val_fold: int = 0,
    test_fold: int = 1,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    p = pairs.protein_id.astype(str).map(lambda x: stable_fold(x, folds))
    r = pairs.reaction_id.astype(str).map(lambda x: stable_fold(x, folds))
    held = {val_fold, test_fold}
    train = pairs[(~p.isin(held)) & (~r.isin(held))]
    val = pairs[(p == val_fold) & (r == val_fold)]
    test = pairs[(p == test_fold) & (r == test_fold)]
    for a, b in ((train, val), (train, test), (val, test)):
        if set(a.protein_id) & set(b.protein_id):
            raise AssertionError("protein leakage")
        if set(a.reaction_id) & set(b.reaction_id):
            raise AssertionError("reaction leakage")
    return (
        train.reset_index(drop=True),
        val.reset_index(drop=True),
        test.reset_index(drop=True),
    )


def positive_maps(frame: pd.DataFrame) -> tuple[dict[str, set[str]], dict[str, set[str]]]:
    by_r: dict[str, set[str]] = defaultdict(set)
    by_p: dict[str, set[str]] = defaultdict(set)
    for p, r in frame[["protein_id", "reaction_id"]].itertuples(index=False):
        by_r[str(r)].add(str(p))
        by_p[str(p)].add(str(r))
    return dict(by_r), dict(by_p)


def reaction_neighbors(
    assets: FibreAssetStore,
    query_ids: set[str],
    train_ids: set[str],
    *,
    topk: int,
    device: torch.device,
) -> dict[str, list[str]]:
    idx = assets.reaction_index
    train = sorted(train_ids & set(idx))
    query = sorted(query_ids & set(idx))
    train_rows = np.asarray([idx[x] for x in train], dtype=np.int64)
    train_x = torch.as_tensor(
        np.asarray(assets.reaction_features[train_rows, :2048], dtype=np.float32),
        device=device,
    )
    train_n = (train_x * train_x).sum(1).clamp_min(1e-8)
    out: dict[str, list[str]] = {}
    k = min(topk + 1, len(train))
    with torch.no_grad():
        for start in range(0, len(query), 128):
            ids = query[start : start + 128]
            rows = np.asarray([idx[x] for x in ids], dtype=np.int64)
            x = torch.as_tensor(
                np.asarray(assets.reaction_features[rows, :2048], dtype=np.float32),
                device=device,
            )
            dot = x @ train_x.T
            xn = (x * x).sum(1, keepdim=True).clamp_min(1e-8)
            sim = dot / (xn + train_n[None, :] - dot).clamp_min(1e-8)
            cols = torch.topk(sim, k=k, dim=1).indices.cpu().numpy()
            for i, q in enumerate(ids):
                values = [train[int(c)] for c in cols[i] if train[int(c)] != q]
                out[q] = values[:topk]
    return out


def _sample_excluding(
    rng: random.Random,
    universe: list[str],
    forbidden: set[str],
    count: int,
) -> list[str]:
    target = min(max(0, count), max(0, len(universe) - len(forbidden)))
    chosen: set[str] = set()
    attempts = 0
    while len(chosen) < target and attempts < max(100, target * 50):
        candidate = rng.choice(universe)
        attempts += 1
        if candidate not in forbidden:
            chosen.add(candidate)
    return list(chosen)


def sample_negatives(
    batch: pd.DataFrame,
    *,
    rng: random.Random,
    by_r: dict[str, set[str]],
    by_p: dict[str, set[str]],
    all_by_r: dict[str, set[str]],
    all_by_p: dict[str, set[str]],
    neighbors: dict[str, list[str]],
    proteins: list[str],
    reactions: list[str],
    hard: int,
    random_count: int,
) -> tuple[list[tuple[str, str, int]], list[tuple[str, str, int]]]:
    r2e: list[tuple[str, str, int]] = []
    e2r: list[tuple[str, str, int]] = []
    for origin, row in enumerate(batch.itertuples(index=False)):
        p, r = str(row.protein_id), str(row.reaction_id)
        forbidden_p = all_by_r.get(r, set())
        hard_p: list[str] = []
        for nr in neighbors.get(r, []):
            for candidate in by_r.get(nr, set()):
                if candidate not in forbidden_p and candidate not in hard_p:
                    hard_p.append(candidate)
                    if len(hard_p) >= hard:
                        break
            if len(hard_p) >= hard:
                break
        r2e.extend((candidate, r, origin) for candidate in hard_p)
        for candidate in _sample_excluding(rng, proteins, forbidden_p, random_count):
            r2e.append((candidate, r, origin))

        forbidden_r = all_by_p.get(p, set())
        hard_r = [
            nr for nr in neighbors.get(r, [])
            if nr not in forbidden_r
        ][:hard]
        e2r.extend((p, candidate, origin) for candidate in hard_r)
        for candidate in _sample_excluding(rng, reactions, forbidden_r, random_count):
            e2r.append((p, candidate, origin))
    return r2e, e2r


def pair_distances(
    model: FibreRelationalModel,
    assets: FibreAssetStore,
    pairs: list[tuple[str, str]],
    device: torch.device,
    *,
    chunk: int,
) -> torch.Tensor:
    values: list[torch.Tensor] = []
    for start in range(0, len(pairs), chunk):
        part = pairs[start : start + chunk]
        kwargs = assets.batch(
            [x[0] for x in part],
            [x[1] for x in part],
            device,
            expert_names=set(model.expert_adapters.keys()),
        )
        values.append(model.relation_distance(**kwargs))
    return torch.cat(values)


def eval_sample(
    model: FibreRelationalModel,
    assets: FibreAssetStore,
    frame: pd.DataFrame,
    *,
    all_by_r: dict[str, set[str]],
    all_by_p: dict[str, set[str]],
    train_by_r: dict[str, set[str]],
    train_by_p: dict[str, set[str]],
    neighbors: dict[str, list[str]],
    train_proteins: list[str],
    train_reactions: list[str],
    device: torch.device,
    margin: float,
    seed: int,
    max_pairs: int = 512,
) -> float:
    model.eval()
    sample = frame.sample(n=min(max_pairs, len(frame)), random_state=seed)
    rng = random.Random(seed)
    r2e, e2r = sample_negatives(
        sample,
        rng=rng,
        by_r=train_by_r,
        by_p=train_by_p,
        all_by_r=all_by_r,
        all_by_p=all_by_p,
        neighbors=neighbors,
        proteins=train_proteins,
        reactions=train_reactions,
        hard=2,
        random_count=2,
    )
    with torch.no_grad():
        pos_pairs = list(sample[["protein_id", "reaction_id"]].itertuples(index=False, name=None))
        pos = pair_distances(model, assets, pos_pairs, device, chunk=96)
        losses = []
        for negs in (r2e, e2r):
            if not negs:
                continue
            neg = pair_distances(model, assets, [(p, r) for p, r, _ in negs], device, chunk=96)
            origin = torch.tensor([i for _, _, i in negs], device=device)
            losses.append(F.softplus(margin + pos[origin] - neg).mean())
        return float(torch.stack(losses).mean().cpu())


def main() -> None:
    ap = argparse.ArgumentParser(description="Train the single ERAM-style plug-in BRIDGE relational model.")
    ap.add_argument("--pairs", type=Path, default=DEFAULT_PAIRS)
    ap.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--hard-negatives", type=int, default=2)
    ap.add_argument("--random-negatives", type=int, default=1)
    ap.add_argument("--neighbor-k", type=int, default=32)
    ap.add_argument("--margin", type=float, default=0.2)
    ap.add_argument("--learning-rate", type=float, default=3e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=20260723)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    assets = FibreAssetStore()
    print("phase=assets_loaded", flush=True)
    pairs = pd.read_csv(args.pairs, dtype=str).fillna("")
    pairs = assets.supported_pairs(pairs)
    train, val, test = split_double_cold(pairs)
    print(
        f"phase=split train={len(train)} val={len(val)} test={len(test)}",
        flush=True,
    )
    all_by_r, all_by_p = positive_maps(pairs)
    train_by_r, train_by_p = positive_maps(train)
    train_proteins = sorted(train.protein_id.unique())
    train_reactions = sorted(train.reaction_id.unique())
    query_reactions = set(train.reaction_id) | set(val.reaction_id)
    args.output.mkdir(parents=True, exist_ok=True)
    neighbor_cache = args.output / "reaction_neighbors.json"
    if neighbor_cache.exists():
        neighbors = {
            str(key): [str(value) for value in values]
            for key, values in json.loads(neighbor_cache.read_text()).items()
        }
        print("phase=neighbors_loaded", flush=True)
    else:
        print("phase=neighbors_build_start", flush=True)
        neighbors = reaction_neighbors(
            assets,
            query_reactions,
            set(train_reactions),
            topk=args.neighbor_k,
            device=device,
        )
        neighbor_cache.write_text(json.dumps(neighbors))
        print("phase=neighbors_built", flush=True)

    model = FibreRelationalModel(
        protein_input_dim=1152,
        experts={
            "enzgfm": ExpertSpec(2048),
            "reaction_center": ExpertSpec(1280),
            "clipzyme": ExpertSpec(1280),
            "tps": ExpertSpec(768),
        },
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    split_record = {
        "train_pairs": len(train),
        "val_pairs": len(val),
        "test_pairs": len(test),
        "train_proteins": int(train.protein_id.nunique()),
        "train_reactions": int(train.reaction_id.nunique()),
        "val_proteins": int(val.protein_id.nunique()),
        "val_reactions": int(val.reaction_id.nunique()),
        "test_proteins": int(test.protein_id.nunique()),
        "test_reactions": int(test.reaction_id.nunique()),
    }
    (args.output / "split.json").write_text(json.dumps(split_record, indent=2))

    best = float("inf")
    history = []
    rng = random.Random(args.seed)
    for epoch in range(1, args.epochs + 1):
        model.train()
        order = np.random.default_rng(args.seed + epoch).permutation(len(train))
        losses = []
        for start in range(0, len(order), args.batch_size):
            if start == 0:
                print(f"epoch={epoch} phase=first_batch_start", flush=True)
            rows = order[start : start + args.batch_size]
            batch = train.iloc[rows]
            pos_pairs = list(batch[["protein_id", "reaction_id"]].itertuples(index=False, name=None))
            r2e, e2r = sample_negatives(
                batch,
                rng=rng,
                by_r=train_by_r,
                by_p=train_by_p,
                all_by_r=all_by_r,
                all_by_p=all_by_p,
                neighbors=neighbors,
                proteins=train_proteins,
                reactions=train_reactions,
                hard=args.hard_negatives,
                random_count=args.random_negatives,
            )
            optimizer.zero_grad(set_to_none=True)
            pos = pair_distances(model, assets, pos_pairs, device, chunk=32)
            direction_losses = []
            for negs in (r2e, e2r):
                if not negs:
                    continue
                neg = pair_distances(
                    model,
                    assets,
                    [(p, r) for p, r, _ in negs],
                    device,
                    chunk=32,
                )
                origin = torch.tensor([i for _, _, i in negs], device=device)
                direction_losses.append(
                    F.softplus(args.margin + pos[origin] - neg).mean()
                )
            loss = torch.stack(direction_losses).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
            if (start // args.batch_size) % 250 == 0:
                print(
                    f"epoch={epoch} step={start // args.batch_size} "
                    f"loss={losses[-1]:.5f}",
                    flush=True,
                )

        val_loss = eval_sample(
            model,
            assets,
            val,
            all_by_r=all_by_r,
            all_by_p=all_by_p,
            train_by_r=train_by_r,
            train_by_p=train_by_p,
            neighbors=neighbors,
            train_proteins=train_proteins,
            train_reactions=train_reactions,
            device=device,
            margin=args.margin,
            seed=args.seed,
        )
        record = {
            "epoch": epoch,
            "train_loss": float(np.mean(losses)),
            "val_loss": val_loss,
        }
        history.append(record)
        print(json.dumps(record), flush=True)
        if val_loss < best:
            best = val_loss
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "experts": {
                        "enzgfm": 2048,
                        "reaction_center": 1280,
                        "clipzyme": 1280,
                        "tps": 768,
                    },
                    "config": vars(args),
                    "split": split_record,
                    "epoch": epoch,
                    "val_loss": val_loss,
                },
                args.output / "best.pt",
            )

    summary = {
        "method": "BRIDGE multimodal relational learning",
        "relational_core": "ERAM EnzymaticModel",
        "plug_in_contract": "variable-length available biochemical evidence tokens",
        "experts": ["enzgfm", "reaction_center", "clipzyme", "tps"],
        "runtime_plugins": ["context"],
        "selection": "single fixed architecture; validation selects checkpoint only",
        "best_val_loss": best,
        "history": history,
        "split": split_record,
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
