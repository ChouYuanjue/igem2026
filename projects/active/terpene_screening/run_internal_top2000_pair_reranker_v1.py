from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from projects.active.terpene_screening.broad_rhea_metrics import (  # noqa: E402
    evaluate_full_candidate_ranks,
    summarize_query_metrics,
)
from projects.active.terpene_screening.evaluate_interaction_retriever_marts import (  # noqa: E402
    PairResidualHead,
)
from projects.active.terpene_screening.fair_benchmark import (  # noqa: E402
    DEFAULT_BUDGETS,
    DEFAULT_TOP_PERCENTS,
    sha256_file,
)
from projects.active.terpene_screening.rank_open_world import (  # noqa: E402
    load_feature_schema,
    load_models,
    load_protein_library,
    load_registered_reaction_feature_library,
)

DEFAULT_BENCH = ROOT / "results/cleanroom_internal_full_candidate_benchmarks_v1"
DEFAULT_DIFFICULTY = ROOT / "results/cleanroom_internal_full_candidate_difficulty_v1"
DEFAULT_COARSE_EVAL = ROOT / "results/cleanroom_internal_full_candidate_rdkitplus_v1"
DEFAULT_MODEL_ROOT = ROOT / "results/enzymecage_cleanroom_rdkitplus_v1"
DEFAULT_PROTEINS = ROOT / "data/catalyst_candidate_universes/general_merged/proteins"
DEFAULT_REACTIONS = ROOT / "data/catalyst_candidate_universes/general_merged/reaction_features/drfp_categorical_rdkitplus_v1"
DEFAULT_RECIPE = ROOT / "projects/active/terpene_screening/CLEANROOM_R2E_TOP2000_RERANKER_RECIPE_V1.json"
DEFAULT_OUTPUT = ROOT / "results/cleanroom_internal_top2000_pair_reranker_v1"


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def zero_initialize_residual(head: PairResidualHead) -> None:
    last = head.network[-1]
    if not isinstance(last, nn.Linear):
        raise TypeError("PairResidualHead final module must be Linear")
    nn.init.zeros_(last.weight)
    if last.bias is not None:
        nn.init.zeros_(last.bias)


def score_pair_residual(
    head: PairResidualHead, reaction: torch.Tensor, protein: torch.Tensor
) -> torch.Tensor:
    return head(reaction, protein)


def reconstruct_positive_ranks(
    *,
    positives: set[str],
    reranked_top_ids: list[str],
    coarse_positive_ranks: dict[str, int],
) -> np.ndarray:
    position = {candidate: index + 1 for index, candidate in enumerate(reranked_top_ids)}
    ranks: list[int] = []
    for positive in sorted(positives):
        if positive in position:
            ranks.append(position[positive])
        elif positive in coarse_positive_ranks:
            ranks.append(int(coarse_positive_ranks[positive]))
        else:
            raise KeyError(f"Missing coarse positive rank for {positive}")
    if len(set(ranks)) != len(ranks):
        raise ValueError(f"Reranked positive ranks are not unique: {ranks}")
    return np.asarray(ranks, dtype=np.int64)


def query_metrics_from_positive_rank_frame(
    positive_ranks: pd.DataFrame,
    *,
    budgets: tuple[int, ...] = DEFAULT_BUDGETS,
    top_percents: tuple[float, ...] = DEFAULT_TOP_PERCENTS,
) -> pd.DataFrame:
    """Rebuild per-query metrics from exact positive ranks using the current metric schema.

    Older coarse-evaluation artifacts predate some cutoffs (notably Hit@2/4).  Positive
    ranks are the stable sufficient statistic for deterministic full rankings, so derive
    the coarse side again instead of silently dropping newly registered metrics.
    """
    required = {"query_id", "positive_rank", "candidate_count"}
    missing = sorted(required - set(positive_ranks.columns))
    if missing:
        raise ValueError(f"Positive-rank frame missing columns: {missing}")
    if positive_ranks.empty:
        raise ValueError("Positive-rank frame is empty")

    records: list[dict[str, object]] = []
    for query_id, group in positive_ranks.groupby("query_id", sort=True):
        candidate_counts = group["candidate_count"].astype(int).unique()
        if len(candidate_counts) != 1:
            raise ValueError(
                f"Query {query_id} has inconsistent candidate_count values: {candidate_counts.tolist()}"
            )
        metrics = evaluate_full_candidate_ranks(
            group["positive_rank"].astype(int).to_numpy(),
            int(candidate_counts[0]),
            budgets=budgets,
            top_percents=top_percents,
        )
        records.append({"direction": "reaction_to_enzyme", "query_id": str(query_id), **metrics})
    return pd.DataFrame(records)


def encode_library(
    model: torch.nn.Module,
    values: np.ndarray,
    *,
    kind: str,
    device: torch.device,
    chunk_size: int = 8192,
) -> np.ndarray:
    blocks: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(values), chunk_size):
            x = torch.as_tensor(values[start : start + chunk_size], dtype=torch.float32, device=device)
            if kind == "protein":
                z = model.encode_proteins(x)
            elif kind == "reaction":
                z = model.encode_reactions(x)
            else:
                raise ValueError(kind)
            blocks.append(z.detach().cpu().numpy().astype(np.float32, copy=False))
    return np.concatenate(blocks, axis=0)


def build_training_triples(
    *,
    train_pairs: pd.DataFrame,
    protein_ids: list[str],
    reaction_ids: list[str],
    protein_embeddings: np.ndarray,
    reaction_embeddings: np.ndarray,
    shortlist_size: int,
    negatives_per_positive: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
    train_pairs = train_pairs[["protein_id", "reaction_id"]].drop_duplicates().copy()
    train_proteins = sorted(set(train_pairs["protein_id"].astype(str)))
    train_reactions = sorted(set(train_pairs["reaction_id"].astype(str)))
    p_index = {value: i for i, value in enumerate(protein_ids)}
    r_index = {value: i for i, value in enumerate(reaction_ids)}
    missing_p = sorted(set(train_proteins) - set(p_index))
    missing_r = sorted(set(train_reactions) - set(r_index))
    if missing_p or missing_r:
        raise ValueError(f"Training entities missing from feature universe: proteins={missing_p[:5]} reactions={missing_r[:5]}")

    train_p_rows = np.asarray([p_index[x] for x in train_proteins], dtype=np.int64)
    train_r_rows = np.asarray([r_index[x] for x in train_reactions], dtype=np.int64)
    positives_by_reaction = {
        str(rid): set(group["protein_id"].astype(str))
        for rid, group in train_pairs.groupby("reaction_id", sort=True)
    }

    p_tensor = torch.as_tensor(protein_embeddings[train_p_rows], dtype=torch.float32, device=device)
    r_tensor = torch.as_tensor(reaction_embeddings[train_r_rows], dtype=torch.float32, device=device)
    k = min(int(shortlist_size), len(train_proteins))
    negs_by_reaction: dict[str, list[int]] = {}
    with torch.no_grad():
        for start in range(0, len(train_reactions), 32):
            scores = r_tensor[start : start + 32] @ p_tensor.T
            _, local_rows = torch.topk(scores, k=k, dim=1, largest=True, sorted=True)
            local_rows = local_rows.cpu().numpy()
            for offset in range(len(local_rows)):
                rid = train_reactions[start + offset]
                positives = positives_by_reaction[rid]
                chosen: list[int] = []
                for local_row in local_rows[offset]:
                    pid = train_proteins[int(local_row)]
                    if pid in positives:
                        continue
                    chosen.append(int(train_p_rows[int(local_row)]))
                    if len(chosen) >= negatives_per_positive:
                        break
                if len(chosen) < negatives_per_positive:
                    raise RuntimeError(f"Only {len(chosen)} hard negatives for {rid}")
                negs_by_reaction[rid] = chosen

    r_rows: list[int] = []
    p_rows: list[int] = []
    n_rows: list[int] = []
    for rid, group in train_pairs.groupby("reaction_id", sort=True):
        rid = str(rid)
        rr = r_index[rid]
        negatives = negs_by_reaction[rid]
        for pid in sorted(set(group["protein_id"].astype(str))):
            pr = p_index[pid]
            for nr in negatives:
                r_rows.append(rr)
                p_rows.append(pr)
                n_rows.append(nr)

    audit = {
        "train_pairs": int(len(train_pairs)),
        "train_proteins": int(len(train_proteins)),
        "train_reactions": int(len(train_reactions)),
        "shortlist_size": int(k),
        "negatives_per_positive": int(negatives_per_positive),
        "training_triples": int(len(r_rows)),
        "candidate_scope": "train proteins only",
        "dev_entities_used_in_mining": False,
    }
    return (
        np.asarray(r_rows, dtype=np.int32),
        np.asarray(p_rows, dtype=np.int32),
        np.asarray(n_rows, dtype=np.int32),
        audit,
    )


def train_head(
    *,
    reaction_embeddings: np.ndarray,
    protein_embeddings: np.ndarray,
    triples: tuple[np.ndarray, np.ndarray, np.ndarray],
    recipe: dict[str, object],
    device: torch.device,
) -> tuple[PairResidualHead, list[dict[str, float]]]:
    opt_cfg = dict(recipe["optimization"])
    head_cfg = dict(recipe["head"])
    seed_everything(int(opt_cfg["seed"]))
    dim = int(reaction_embeddings.shape[1])
    if protein_embeddings.shape[1] != dim:
        raise ValueError("Frozen reaction/protein embedding dimensions differ")
    head = PairResidualHead(dim, int(head_cfg["hidden_dim"]), float(head_cfg["dropout"])).to(device)
    zero_initialize_residual(head)
    optimizer = torch.optim.AdamW(
        head.parameters(),
        lr=float(opt_cfg["learning_rate"]),
        weight_decay=float(opt_cfg["weight_decay"]),
    )
    r_all = torch.as_tensor(reaction_embeddings, dtype=torch.float32, device=device)
    p_all = torch.as_tensor(protein_embeddings, dtype=torch.float32, device=device)
    r_rows, p_rows, n_rows = triples
    rng = np.random.default_rng(int(opt_cfg["seed"]))
    history: list[dict[str, float]] = []
    batch_size = int(opt_cfg["batch_size"])
    margin = float(opt_cfg["margin"])
    penalty = float(opt_cfg["residual_penalty"])
    for epoch in range(1, int(opt_cfg["epochs"]) + 1):
        order = rng.permutation(len(r_rows))
        loss_sum = pair_sum = residual_sum = 0.0
        count = 0
        head.train()
        for start in range(0, len(order), batch_size):
            idx = order[start : start + batch_size]
            rt = r_all[torch.as_tensor(r_rows[idx], dtype=torch.long, device=device)]
            pt = p_all[torch.as_tensor(p_rows[idx], dtype=torch.long, device=device)]
            nt = p_all[torch.as_tensor(n_rows[idx], dtype=torch.long, device=device)]
            coarse_p = (rt * pt).sum(dim=-1)
            coarse_n = (rt * nt).sum(dim=-1)
            res_p = score_pair_residual(head, rt, pt)
            res_n = score_pair_residual(head, rt, nt)
            delta = (coarse_p + res_p) - (coarse_n + res_n)
            pair_loss = F.softplus(margin - delta).mean()
            residual_loss = 0.5 * (res_p.square().mean() + res_n.square().mean())
            loss = pair_loss + penalty * residual_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), max_norm=float(opt_cfg["gradient_clip_norm"]))
            optimizer.step()
            b = len(idx)
            loss_sum += float(loss.detach().cpu()) * b
            pair_sum += float(pair_loss.detach().cpu()) * b
            residual_sum += float(residual_loss.detach().cpu()) * b
            count += b
        history.append({
            "epoch": float(epoch),
            "loss": loss_sum / max(count, 1),
            "pair_loss": pair_sum / max(count, 1),
            "residual_l2": residual_sum / max(count, 1),
        })
    return head, history


def evaluate_reranker(
    *,
    head: PairResidualHead,
    reaction_embeddings: np.ndarray,
    protein_embeddings: np.ndarray,
    reaction_ids: list[str],
    protein_ids: list[str],
    test_pairs: pd.DataFrame,
    coarse_positive_ranks_csv: Path,
    coarse_query_metrics_csv: Path,
    reaction_slices_csv: Path,
    shortlist_size: int,
    device: torch.device,
) -> tuple[pd.DataFrame, dict[str, object], pd.DataFrame]:
    r_index = {value: i for i, value in enumerate(reaction_ids)}
    p_index = {value: i for i, value in enumerate(protein_ids)}
    candidate_ids = np.asarray(protein_ids, dtype=object)
    positives_by_reaction = {
        str(rid): set(group["protein_id"].astype(str))
        for rid, group in test_pairs.groupby("reaction_id", sort=True)
    }
    coarse_pos = pd.read_csv(coarse_positive_ranks_csv, dtype={"query_id": str, "positive_id": str})
    coarse_pos = coarse_pos[coarse_pos["direction"] == "reaction_to_enzyme"].copy()
    coarse_rank_map = {
        (str(row.query_id), str(row.positive_id)): int(row.positive_rank)
        for row in coarse_pos.itertuples(index=False)
    }
    # Reconstruct the coarse query table from exact positive ranks.  This keeps old
    # coarse artifacts comparable when the shared evaluator gains new cutoffs.
    coarse_query = query_metrics_from_positive_rank_frame(coarse_pos)

    p_all = torch.as_tensor(protein_embeddings, dtype=torch.float32, device=device)
    r_all = torch.as_tensor(reaction_embeddings, dtype=torch.float32, device=device)
    head.eval()
    records: list[dict[str, object]] = []
    support_records: list[dict[str, object]] = []
    queries = sorted(positives_by_reaction)
    k = min(int(shortlist_size), len(protein_ids))
    with torch.no_grad():
        for start in range(0, len(queries), 16):
            batch = queries[start : start + 16]
            rr = torch.as_tensor([r_index[q] for q in batch], dtype=torch.long, device=device)
            coarse = r_all[rr] @ p_all.T
            top_values, top_rows = torch.topk(coarse, k=k, dim=1, largest=True, sorted=False)
            for i, qid in enumerate(batch):
                rows = top_rows[i].detach().cpu().numpy().astype(np.int64)
                values = top_values[i].detach().cpu().numpy().astype(np.float64)
                # Resolve order inside the selected shortlist deterministically by coarse score then ID.
                local_order = np.lexsort((candidate_ids[rows], -values))
                rows = rows[local_order]
                values = values[local_order]
                q = r_all[r_index[qid]].unsqueeze(0).expand(k, -1)
                p = p_all[torch.as_tensor(rows, dtype=torch.long, device=device)]
                residual = head(q, p).detach().cpu().numpy().astype(np.float64)
                final = values + residual
                rerank_order = np.lexsort((candidate_ids[rows], -final))
                reranked_ids = candidate_ids[rows[rerank_order]].astype(str).tolist()
                positives = positives_by_reaction[qid]
                old = {pid: coarse_rank_map[(qid, pid)] for pid in positives}
                ranks = reconstruct_positive_ranks(
                    positives=positives,
                    reranked_top_ids=reranked_ids,
                    coarse_positive_ranks=old,
                )
                metrics = evaluate_full_candidate_ranks(
                    ranks,
                    len(protein_ids),
                    budgets=DEFAULT_BUDGETS,
                    top_percents=DEFAULT_TOP_PERCENTS,
                )
                records.append({"direction": "reaction_to_enzyme", "query_id": qid, **metrics})
                support_records.append({
                    "query_id": qid,
                    "positive_count": len(positives),
                    "coarse_positive_in_top2000": int(sum(rank <= k for rank in old.values())),
                    "coarse_query_hit_top2000": int(any(rank <= k for rank in old.values())),
                    "residual_rms": float(np.sqrt(np.mean(residual**2))),
                    "residual_max_abs": float(np.max(np.abs(residual))),
                })
    frame = pd.DataFrame(records)
    support = pd.DataFrame(support_records)
    overall = {
        "coarse": summarize_query_metrics(coarse_query, budgets=DEFAULT_BUDGETS, top_percents=DEFAULT_TOP_PERCENTS),
        "reranked": summarize_query_metrics(frame, budgets=DEFAULT_BUDGETS, top_percents=DEFAULT_TOP_PERCENTS),
    }
    slices = pd.read_csv(reaction_slices_csv, dtype={"reaction_id": str}).rename(columns={"reaction_id": "query_id"})
    coarse_join = coarse_query.merge(slices[["query_id", "reaction_similarity_bucket"]], on="query_id", how="left", validate="one_to_one")
    new_join = frame.merge(slices[["query_id", "reaction_similarity_bucket"]], on="query_id", how="left", validate="one_to_one")
    by_slice: dict[str, object] = {}
    for bucket in sorted(set(slices["reaction_similarity_bucket"].astype(str))):
        c = coarse_join[coarse_join["reaction_similarity_bucket"] == bucket]
        n = new_join[new_join["reaction_similarity_bucket"] == bucket]
        if c.empty or n.empty:
            continue
        by_slice[bucket] = {
            "coarse": summarize_query_metrics(c, budgets=DEFAULT_BUDGETS, top_percents=DEFAULT_TOP_PERCENTS),
            "reranked": summarize_query_metrics(n, budgets=DEFAULT_BUDGETS, top_percents=DEFAULT_TOP_PERCENTS),
        }
    overall["reaction_similarity_slices"] = by_slice
    return frame, overall, support


def main() -> None:
    ap = argparse.ArgumentParser(description="Train/evaluate preregistered train-only Top-2000 R2E pair residual reranker on internal clean folds.")
    ap.add_argument("--folds", default="0,1,2")
    ap.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    folds = [int(x) for x in args.folds.split(",") if x.strip()]
    recipe = json.loads(DEFAULT_RECIPE.read_text(encoding="utf-8"))
    shortlist_size = int(recipe["shortlist_size"])
    negs = int(recipe["negative_mining"]["per_positive_negatives"])
    device = torch.device(args.device)
    protein_features, protein_ids = load_protein_library(DEFAULT_PROTEINS)

    for fold in folds:
        out = args.output_root.resolve() / f"fold{fold}"
        if (out / "summary.json").is_file():
            print(f"skip fold{fold}: summary exists", flush=True)
            continue
        out.mkdir(parents=True, exist_ok=True)
        cell = f"clean2023_internal_double_cold_fold{fold}"
        bench = DEFAULT_BENCH / cell
        model_dir = DEFAULT_MODEL_ROOT / f"fold{fold}"
        schema = load_feature_schema(model_dir)
        reaction_features, reaction_ids = load_registered_reaction_feature_library(DEFAULT_REACTIONS, schema)
        models = load_models(model_dir / "models", "production", device)
        if len(models) != 1:
            raise ValueError(f"Expected one frozen coarse model for fold{fold}, got {len(models)}")
        model = models[0]
        p_emb = encode_library(model, protein_features, kind="protein", device=device)
        r_emb = encode_library(model, reaction_features, kind="reaction", device=device)
        train_pairs = pd.read_csv(bench / "train_pairs.csv", dtype=str).fillna("")
        test_pairs = pd.read_csv(bench / "test_pairs.csv", dtype=str).fillna("")
        train_p, test_p = set(train_pairs.protein_id), set(test_pairs.protein_id)
        train_r, test_r = set(train_pairs.reaction_id), set(test_pairs.reaction_id)
        if train_p & test_p or train_r & test_r:
            raise RuntimeError("Internal fold is not entity-disjoint")
        r_rows, p_rows, n_rows, mining_audit = build_training_triples(
            train_pairs=train_pairs,
            protein_ids=protein_ids,
            reaction_ids=reaction_ids,
            protein_embeddings=p_emb,
            reaction_embeddings=r_emb,
            shortlist_size=shortlist_size,
            negatives_per_positive=negs,
            device=device,
        )
        head, history = train_head(
            reaction_embeddings=r_emb,
            protein_embeddings=p_emb,
            triples=(r_rows, p_rows, n_rows),
            recipe=recipe,
            device=device,
        )
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
            shortlist_size=shortlist_size,
            device=device,
        )
        frame.to_csv(out / "query_metrics.csv", index=False)
        support.to_csv(out / "support_audit.csv", index=False)
        pd.DataFrame(history).to_csv(out / "training_history.csv", index=False)
        torch.save({
            "head_state_dict": {k: v.detach().cpu() for k, v in head.state_dict().items()},
            "recipe": recipe,
            "fold": fold,
            "embedding_dim": int(p_emb.shape[1]),
        }, out / "pair_residual_head.pt")
        payload = {
            "fold": fold,
            "cell": cell,
            "outer_labels_used": False,
            "recipe": str(DEFAULT_RECIPE.resolve()),
            "recipe_sha256": sha256_file(DEFAULT_RECIPE),
            "train_pairs_sha256": sha256_file(bench / "train_pairs.csv"),
            "test_pairs_sha256": sha256_file(bench / "test_pairs.csv"),
            "coarse_model_dir": str(model_dir.resolve()),
            "coarse_model_checkpoint_sha256": sha256_file(model_dir / "models" / "production_seed20260723.pt"),
            "training_mining_audit": mining_audit,
            "metrics": metrics,
            "history": history,
        }
        (out / "summary.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        print(json.dumps(payload, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
