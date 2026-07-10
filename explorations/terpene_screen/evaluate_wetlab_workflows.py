from __future__ import annotations

import json
import random
import re
from collections import defaultdict
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
BASE = PROJECT_ROOT / "results" / "terpene_gate_matrix"
DATA = PROJECT_ROOT / "data" / "terpene_gate_matrix"
POSITIVE_PATH = PROJECT_ROOT / "data" / "terpene" / "enzyme_terpene_synthase.tsv"
CANDIDATE_PATH = PROJECT_ROOT / "data" / "terpene" / "all_seq_terpene_synthase.tsv"

BUDGETS = [5, 10, 20, 30, 50]
FEWSHOT_BUDGETS = [5, 10, 20]
FEWSHOT_SEEDS = [1, 2, 3, 5]
FEWSHOT_REPEATS = 20
RANDOM_SEED = 20260707


def kmers(seq: str, k: int = 3) -> set[str]:
    seq = re.sub(r"[^A-Z]", "", str(seq).upper())
    if len(seq) < k:
        return set()
    return {seq[i : i + k] for i in range(len(seq) - k + 1)}


def md_table(df: pd.DataFrame, cols: list[str], max_rows: int | None = None) -> str:
    if max_rows is not None:
        df = df.head(max_rows)
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in df.iterrows():
        values = []
        for col in cols:
            value = row[col]
            if isinstance(value, float):
                values.append(f"{value:.4g}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def compute_reaction_only_budget_metrics() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    summary = json.loads((BASE / "run_summary.json").read_text())
    gate_reaction = pd.read_csv(BASE / "gate_reaction_level.csv", usecols=["reaction_id"])
    all_reactions = sorted(set(gate_reaction["reaction_id"].astype(str)))
    n_reactions = len(all_reactions)

    gate_metrics = pd.read_csv(BASE / "gate_metrics.csv")
    gate_metrics["coverage_per_100_pool"] = gate_metrics["micro_positive_coverage"] / gate_metrics["mean_pool_size"] * 100
    gate_metrics["positive_pair_precision_pct"] = gate_metrics["positive_candidates_total"] / gate_metrics["candidate_pairs_total"] * 100
    gate_metrics.to_csv(BASE / "gate_metrics_with_efficiency.csv", index=False)

    candidates = pd.read_csv(DATA / "gate_candidate_pools.csv")
    numeric_cols = [
        "gate_score",
        "reaction_similarity",
        "sequence_kmer",
        "motif_score",
        "label",
        "precursor_match",
        "product_skeleton",
        "mechanism",
    ]
    for col in numeric_cols:
        if col in candidates.columns:
            candidates[col] = pd.to_numeric(candidates[col], errors="coerce").fillna(0)
    candidates["fusion"] = (
        0.35 * candidates.get("reaction_similarity", 0)
        + 0.25 * candidates.get("sequence_kmer", 0)
        + 0.15 * candidates.get("precursor_match", 0)
        + 0.10 * candidates.get("product_skeleton", 0)
        + 0.10 * candidates.get("motif_score", 0)
    )
    score_cols = {
        "gate_score": "gate_score",
        "reaction_similarity": "reaction_similarity",
        "sequence_kmer": "sequence_kmer",
        "motif": "motif_score",
        "fusion": "fusion",
    }
    rows = []
    all_reaction_index = pd.Index(all_reactions)
    for gate_id, gate_df in candidates.groupby("gate_id", sort=False):
        print(f"[reaction-only] scoring {gate_id}", flush=True)
        for reranker, score_col in score_cols.items():
            sorted_df = gate_df.sort_values(["reaction_id", score_col, "uniprot_id"], ascending=[True, False, True])
            for budget in BUDGETS:
                top_df = sorted_df.groupby("reaction_id", sort=False).head(budget)
                hits = top_df.groupby("reaction_id")["label"].sum().reindex(all_reaction_index, fill_value=0)
                hit_probability = float((hits > 0).mean())
                expected_hits = float(hits.mean())
                rows.append(
                    {
                        "gate_id": gate_id,
                        "reranker": reranker,
                        "B": budget,
                        "hit_probability": hit_probability,
                        "expected_hits": expected_hits,
                        "precision": expected_hits / budget,
                        "n_reactions": n_reactions,
                    }
                )
    wetlab = pd.DataFrame(rows)
    wetlab.to_csv(BASE / "wetlab_budget_metrics.csv", index=False)

    evidence = candidates.copy()
    evidence["e_reaction"] = evidence["reaction_similarity"] > 0
    evidence["e_sequence"] = evidence["sequence_kmer"] > 0
    evidence["e_precursor"] = evidence.get("precursor_match", 0) > 0
    evidence["e_product"] = evidence.get("product_skeleton", 0) > 0
    evidence["e_motif"] = evidence["motif_score"] > 0
    evidence["e_mechanism"] = evidence.get("mechanism", 0) > 0
    evidence["evidence_channels"] = evidence[["e_reaction", "e_sequence", "e_precursor", "e_product", "e_motif", "e_mechanism"]].sum(axis=1)
    evidence.to_csv(DATA / "gate_candidate_pools_with_evidence.csv", index=False)
    evidence_rows = []
    for gate_id, gate_df in evidence.groupby("gate_id"):
        for threshold in [1, 2, 3, 4]:
            mask = gate_df["evidence_channels"] >= threshold
            n_candidates = int(mask.sum())
            n_positives = int(gate_df.loc[mask, "label"].sum())
            evidence_rows.append(
                {
                    "gate_id": gate_id,
                    "threshold": threshold,
                    "n_candidates": n_candidates,
                    "n_positives": n_positives,
                    "precision_pct": 100 * n_positives / n_candidates if n_candidates else 0,
                }
            )
    evidence_metrics = pd.DataFrame(evidence_rows)
    evidence_metrics.to_csv(BASE / "evidence_enrichment.csv", index=False)
    return gate_metrics, wetlab, evidence_metrics


def compute_fewshot_metrics() -> pd.DataFrame:
    positive = pd.read_csv(POSITIVE_PATH, sep="\t", dtype=str).fillna("")
    candidate = pd.read_csv(CANDIDATE_PATH, sep="\t", dtype=str).fillna("")
    pos_by_rxn = positive.groupby("rhea_id")["Entry"].apply(lambda s: sorted(set(s.astype(str)))).to_dict()
    seq = dict(zip(candidate["Entry"].astype(str), candidate["Sequence"].astype(str)))
    candidate_ids = sorted(seq)
    candidate_id_set = set(candidate_ids)

    candidate_kmers = {uid: kmers(sequence) for uid, sequence in seq.items()}
    candidate_sizes = {uid: len(kset) for uid, kset in candidate_kmers.items()}
    inverted_index: dict[str, set[str]] = defaultdict(set)
    for uid, kset in candidate_kmers.items():
        for kmer in kset:
            inverted_index[kmer].add(uid)

    positive_seq = positive.drop_duplicates("Entry").set_index("Entry")["Sequence"].astype(str).to_dict()

    def seed_sequence_scores(seed_ids: list[str]) -> dict[str, float]:
        best_scores: dict[str, float] = defaultdict(float)
        for seed_id in seed_ids:
            seed_kmers = candidate_kmers.get(seed_id) or kmers(positive_seq.get(seed_id, ""))
            if not seed_kmers:
                continue
            intersection_counts: dict[str, int] = defaultdict(int)
            for kmer in seed_kmers:
                for uid in inverted_index.get(kmer, set()):
                    intersection_counts[uid] += 1
            seed_size = len(seed_kmers)
            for uid, intersection in intersection_counts.items():
                denom = seed_size + candidate_sizes.get(uid, 0) - intersection
                if denom <= 0:
                    continue
                score = intersection / denom
                if score > best_scores[uid]:
                    best_scores[uid] = score
        return best_scores

    candidate_pools = pd.read_csv(
        DATA / "gate_candidate_pools.csv",
        usecols=["gate_id", "reaction_id", "uniprot_id", "gate_score", "reaction_similarity"],
    )
    for col in ["gate_score", "reaction_similarity"]:
        candidate_pools[col] = pd.to_numeric(candidate_pools[col], errors="coerce").fillna(0)
    rxn_pools = {}
    for gate in ["rxn_balanced_top20", "rxn_balanced_top50", "recall_union_core", "weighted_top100"]:
        sub = candidate_pools[candidate_pools["gate_id"] == gate]
        rxn_pools[gate] = {rid: group for rid, group in sub.groupby("reaction_id")}

    def pool_ids(gate: str, reaction_id: str) -> list[str]:
        group = rxn_pools.get(gate, {}).get(reaction_id)
        if group is None:
            return []
        return group.sort_values(["reaction_similarity", "gate_score", "uniprot_id"], ascending=[False, False, True])["uniprot_id"].astype(str).tolist()

    rows = []
    eligible_counts = {}
    random.seed(RANDOM_SEED)
    for m in FEWSHOT_SEEDS:
        eligible = [rid for rid, ids in pos_by_rxn.items() if len(set(ids) & candidate_id_set) >= m + 1]
        eligible_counts[m] = len(eligible)
        print(f"[few-shot] m={m}, eligible reactions={len(eligible)}", flush=True)
        for rid in eligible:
            positives = sorted(set(pos_by_rxn[rid]) & candidate_id_set)
            for rep in range(FEWSHOT_REPEATS):
                rng = random.Random(hash((rid, m, rep)) & ((1 << 32) - 1))
                seeds = sorted(rng.sample(positives, m))
                hidden = set(positives) - set(seeds)
                seed_scores = seed_sequence_scores(seeds)
                ordered_seed = sorted(seed_scores, key=lambda uid: (-seed_scores[uid], uid))
                rxn20_ids = pool_ids("rxn_balanced_top20", rid)
                rxn50_ids = pool_ids("rxn_balanced_top50", rid)
                recall_ids = pool_ids("recall_union_core", rid)
                weighted_ids = pool_ids("weighted_top100", rid)

                def evaluate_ranked(method: str, ranked: list[str]) -> None:
                    ranked = [uid for uid in ranked if uid not in seeds]
                    for budget in FEWSHOT_BUDGETS:
                        top = ranked[:budget]
                        hits = sum(1 for uid in top if uid in hidden)
                        rows.append(
                            {
                                "m": m,
                                "reaction_id": rid,
                                "rep": rep,
                                "method": method,
                                "B": budget,
                                "n_hidden": len(hidden),
                                "hits": hits,
                                "hit": hits > 0,
                                "precision": hits / budget,
                                "hidden_recall": hits / len(hidden) if hidden else 0,
                            }
                        )

                def union_rank(name: str, weighted_lists: list[tuple[float, list[str]]], topn: int | None = None, add_seed_score: bool = False) -> None:
                    scores: dict[str, float] = defaultdict(float)
                    seen = set()
                    for weight, ids in weighted_lists:
                        for rank, uid in enumerate(ids):
                            seen.add(uid)
                            scores[uid] += weight / (rank + 1)
                    if add_seed_score:
                        for uid in seen:
                            scores[uid] += 0.5 * seed_scores.get(uid, 0)
                    ordered = sorted(seen, key=lambda uid: (-scores[uid], uid))
                    if topn is not None:
                        ordered = ordered[:topn]
                    evaluate_ranked(name, ordered)

                evaluate_ranked("seed_seq_top20", ordered_seed[:20])
                evaluate_ranked("seed_seq_top50", ordered_seed[:50])
                evaluate_ranked("seed_seq_top100", ordered_seed[:100])
                evaluate_ranked("rxn_top20_only", rxn20_ids)
                union_rank("seed50_union_rxn20", [(1.0, ordered_seed[:50]), (1.0, rxn20_ids)])
                union_rank("seed100_union_rxn50", [(1.0, ordered_seed[:100]), (1.0, rxn50_ids)])
                union_rank("seed50_union_recall", [(1.0, ordered_seed[:50]), (0.7, recall_ids)])
                union_rank("seeded_fusion_top50", [(1.0, ordered_seed[:100]), (1.0, weighted_ids), (0.8, rxn50_ids)], topn=50, add_seed_score=True)
                union_rank("seeded_fusion_top100", [(1.0, ordered_seed[:100]), (1.0, weighted_ids), (0.8, rxn50_ids), (0.5, recall_ids)], topn=100, add_seed_score=True)

    long_df = pd.DataFrame(rows)
    long_df.to_csv(BASE / "fewshot_budget_metrics_long.csv", index=False)
    agg = (
        long_df.groupby(["m", "method", "B"])
        .agg(
            n_trials=("hit", "size"),
            hit_probability=("hit", "mean"),
            expected_hits=("hits", "mean"),
            precision=("precision", "mean"),
            hidden_recall=("hidden_recall", "mean"),
        )
        .reset_index()
    )
    agg.to_csv(BASE / "fewshot_budget_metrics.csv", index=False)
    (BASE / "fewshot_eligible_counts.json").write_text(json.dumps(eligible_counts, indent=2, ensure_ascii=False))
    return agg


def write_report(gate_metrics: pd.DataFrame, wetlab: pd.DataFrame, evidence: pd.DataFrame, fewshot: pd.DataFrame) -> None:
    summary = json.loads((BASE / "run_summary.json").read_text())
    lines = ["# Wet-lab oriented experimental evaluation", ""]
    lines.append(
        f"- Full reactions: {summary['n_evaluated_reactions']}; candidates: {summary['n_candidate_enzymes']}; positives: {summary['n_positive_pairs']}; debug_subset: {summary['debug_subset']}."
    )
    lines.append("- Reaction-only metrics use all 513 reactions as denominator, including empty candidate-pool cases.")
    lines.append("")
    lines.append("## 1. Gate efficiency")
    cols = ["gate_id", "mean_pool_size", "reaction_hit_rate", "micro_positive_coverage", "coverage_per_100_pool", "positive_pair_precision_pct"]
    lines.append(md_table(gate_metrics.sort_values("coverage_per_100_pool", ascending=False)[cols], cols))
    lines.append("")
    lines.append("## 2. Reaction-only wet-lab budget metrics")
    cols = ["gate_id", "reranker", "B", "hit_probability", "expected_hits", "precision"]
    for budget in [5, 10, 20]:
        lines.append(f"### B={budget}")
        view = wetlab[wetlab["B"] == budget].sort_values(["hit_probability", "expected_hits"], ascending=False).head(10)
        lines.append(md_table(view, cols))
        lines.append("")
    lines.append("## 3. Few-shot seed-expansion metrics")
    cols = ["m", "method", "B", "n_trials", "hit_probability", "expected_hits", "precision", "hidden_recall"]
    for m in FEWSHOT_SEEDS:
        for budget in [5, 10, 20]:
            lines.append(f"### m={m}, B={budget}")
            view = fewshot[(fewshot["m"] == m) & (fewshot["B"] == budget)].sort_values(["hit_probability", "expected_hits"], ascending=False).head(8)
            lines.append(md_table(view, cols))
            lines.append("")
    lines.append("## 4. Evidence-channel enrichment")
    cols = ["gate_id", "threshold", "n_candidates", "n_positives", "precision_pct"]
    lines.append(md_table(evidence[evidence["threshold"] == 3].sort_values("precision_pct", ascending=False), cols))
    (BASE / "wetlab_oriented_evaluation.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    print("Computing reaction-only budget metrics...", flush=True)
    gate_metrics, wetlab, evidence = compute_reaction_only_budget_metrics()
    print("Computing few-shot metrics...", flush=True)
    fewshot = compute_fewshot_metrics()
    print("Writing report...", flush=True)
    write_report(gate_metrics, wetlab, evidence, fewshot)
    print("Done.", flush=True)
    print(BASE / "wetlab_oriented_evaluation.md")


if __name__ == "__main__":
    main()
