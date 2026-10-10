from __future__ import annotations

"""Fit E2R query-level authority using all safely combinable heldout + independent Rhea gates.

E2R Broad, EnzGFM, CLIPZyme and reaction candidate assets are immutable.
Use query-disjoint development labels from the fixed 2,268 heldout parent edges
plus all nonoverlapping Rhea rows that cannot expose an evaluation protein query.
Retain the original 23,773 source, original rank cache, and all model weights.
"""

import argparse
import hashlib
import itertools
import json
import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

from projects.active.bridge.model.assets import ROOT
from projects.active.bridge.runtime.e2r_query_gate import (
    CHANNELS, FEATURE_NAMES, GRID, MAX_AUTHORITY, FrozenE2RSurface,
    filtered_target_ranks, predict_authority, reorder_head, monotone,
)
from projects.active.bridge.runtime.final_system import FinalBridgeRuntime
from reproducibility.bime_rank.scripts.analyze_bridge_difficulty_standardized_v3 import annotate

OUT = ROOT / "results/bridge_e2r_query_gate_v3"
ASSET = OUT / "e2r_gate.v3.pkl"
TEST = ROOT / "results/bridge_relation_unseen_max_v3/targets.csv"
TRAIN = ROOT / "data/external/enzymecage_current/catalyst_features/clean2023/training_pairs.csv"
SPLIT = ROOT / "results/bridge_gate_split_v2"
SOURCE = SPLIT / "membership.csv.gz"
RHEA = ROOT / "results/bridge_rhea_longitudinal_growth_v4/targets.csv"
FROZEN = ROOT / "results/bridge_e2r_four_group_ablation_v1/edge_metrics.csv.gz"


def checksum(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def gate_relations() -> tuple[pd.DataFrame, dict]:
    # The 23,773 parent, its frozen hash split, and clean2023 do not change.
    parent = pd.read_csv(TEST, usecols=["protein_id", "reaction_id"], dtype=str).drop_duplicates()
    train = pd.read_csv(TRAIN, usecols=["protein_id", "reaction_id"], dtype=str).drop_duplicates()
    frozen = pd.read_csv(SOURCE, dtype=str).drop_duplicates(["protein_id", "reaction_id"])
    manifest = json.loads((SPLIT / "manifest.json").read_text())
    assert checksum(TEST) == manifest["original_sha256"]
    assert len(parent) == len(frozen) == 23773
    assert set(map(tuple, parent[["protein_id", "reaction_id"]].to_numpy())) == set(
        map(tuple, frozen[["protein_id", "reaction_id"]].to_numpy())
    )
    p2partition = frozen.groupby("protein_id").partition.first().to_dict()
    if not frozen.groupby("protein_id").partition.nunique().eq(1).all():
        raise AssertionError("Internal query split has leakage")
    external = pd.read_csv(RHEA, usecols=["protein_id", "reaction_id"], dtype=str).drop_duplicates()
    banned = set(zip(train.protein_id, train.reaction_id)) | set(zip(parent.protein_id, parent.reaction_id))
    external = external.loc[[
        (p, r) not in banned for p, r in zip(external.protein_id, external.reaction_id)
    ]].copy()
    if len(external) != 3937:
        raise AssertionError("Official Rhea disjoint relation inventory drifted")

    def external_partition(protein_id: str) -> str:
        known = p2partition.get(protein_id)
        if known is not None:
            return known
        # Disjoint new query keys are assigned to training/validation without
        # using relation labels, ranks, or heldout benchmark scores.
        b = int.from_bytes(
            hashlib.blake2b(("IGEM26:E2R:EXT_VALID:20261010:" + protein_id).encode(),
                            digest_size=8).digest(), "big"
        )
        return "gate_validation" if b % 4 == 0 else "gate_learn"

    external["partition"] = external.protein_id.map(external_partition)
    rejected = external.loc[external.partition.eq("evaluation")]
    external = external.loc[external.partition.ne("evaluation")].copy()
    if len(external) != 2948 or len(rejected) != 989:
        raise AssertionError("Changed external-to-test protein-query overlap")
    external = annotate(
        external,
        train.groupby("protein_id").size().to_dict(),
        train.groupby("reaction_id").size().to_dict(),
    )
    external["source"] = "rhea_independent"
    original = frozen.loc[frozen.partition.ne("evaluation")].copy()
    original["source"] = "parent_dev"
    joined = pd.concat([
        original[["protein_id", "reaction_id", "partition", "novelty", "difficulty_stratum", "source"]],
        external[["protein_id", "reaction_id", "partition", "novelty", "difficulty_stratum", "source"]],
    ], ignore_index=True).sort_values(["protein_id", "reaction_id"]).reset_index(drop=True)
    if len(joined) != 5216 or joined.duplicated(["protein_id", "reaction_id"]).any():
        raise AssertionError("Combined gate development relation inventory drifted")
    assert joined.groupby("protein_id").partition.nunique().eq(1).all()
    eval_proteins = set(frozen.loc[frozen.partition.eq("evaluation"), "protein_id"])
    assert not eval_proteins & set(joined.protein_id)
    assert not set(zip(joined.protein_id, joined.reaction_id)) & set(
        zip(frozen.loc[frozen.partition.eq("evaluation"), "protein_id"],
            frozen.loc[frozen.partition.eq("evaluation"), "reaction_id"])
    )
    assert not set(zip(joined.loc[joined.source.eq("rhea_independent"), "protein_id"],
                       joined.loc[joined.source.eq("rhea_independent"), "reaction_id"])) & banned
    selected = joined.copy()
    selected["partition"] = selected.partition.map({
        "gate_learn": "learn", "gate_validation": "validation"
    })
    assert selected.partition.notna().all()
    metadata = {
        "schema": "bridge-e2r-query-dev-v3-combined",
        "original_target_sha256": checksum(TEST),
        "train_source_sha256": checksum(TRAIN),
        "rhea_source_sha256": checksum(RHEA),
        "parent_partition_sha256": checksum(SPLIT / "manifest.json"),
        "original_parent_edges": 23773,
        "heldout_evaluation_edges": 21505,
        "parent_development_edges_included": len(original),
        "external_independent_pairs_before_query_filter": 3937,
        "external_pair_unique_but_evaluation_query_overlapping_rejected": len(rejected),
        "external_independent_pairs_included": len(external),
        "combined_development_edges": len(selected),
        "combined_development_queries": int(selected.protein_id.nunique()),
        "source_novelty_counts": {
            source: {
                novelty: int(count)
                for novelty, count in rows.novelty.value_counts().items()
            }
            for source, rows in selected.groupby("source")
        },
        "split_novelty_counts": {
            subset: {
                novelty: int(count)
                for novelty, count in rows.novelty.value_counts().items()
            }
            for subset, rows in selected.groupby("partition")
        },
        "strict_evaluation_protein_query_overlap": 0,
        "strict_evaluation_relation_overlap": 0,
        "parent_23773_and_all_original_scores_retained": True,
        "no_previous_test_contaminated_adaptive_relation_gate": True,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    selected.to_csv(OUT / "gate_development_pairs.csv.gz", index=False)
    (OUT / "gate_development_manifest.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return selected, metadata



def target_rows(surface: FrozenE2RSurface, ids: list[str]) -> np.ndarray:
    idx = surface.rt.index.reaction_index
    if not all(key in idx for key in ids):
        raise RuntimeError("Positive gate development reaction missing frozen Broad universe")
    return np.asarray([idx[rid] for rid in ids], dtype=np.int64)


def ranks_from_weights(surface: FrozenE2RSurface, item: dict,
                       target: np.ndarray, coeff: np.ndarray) -> np.ndarray:
    order = reorder_head(
        broad_order=item["order"], lexical=surface.lex,
        core=item["core"], functional=item["functional"],
        structure=item["structure"], relation=item["relation"],
        authority=coeff,
    )
    return filtered_target_ranks(order, target)


def utility(ranks: np.ndarray) -> float:
    r = np.asarray(ranks, dtype=np.int64)
    return float(np.mean(1.0 / r + 0.20 * (r <= 10) + 0.04 * (r <= 100)))


def teacher(surface: FrozenE2RSurface, item: dict, targets: np.ndarray) -> tuple[np.ndarray, float]:
    """Label one *training* query by paired top-1000 rank preference, never test labels."""
    baseline = utility(filtered_target_ranks(item["order"], targets))
    best_value = baseline
    best_authority = np.zeros(3, dtype=np.float64)
    # Deterministic predeclared small grid; ties favor less authority and a safe Broad fallback.
    for values in itertools.product(*(GRID[channel] for channel in CHANNELS)):
        authority = np.asarray(values, dtype=np.float64)
        if not item["functional_av"] and authority[0] > 0:
            continue
        if not item["structure_av"] and authority[1] > 0:
            continue
        if not item["relation_av"] and authority[2] > 0:
            continue
        score = utility(ranks_from_weights(surface, item, targets, authority))
        # Absolute protective penalty is fixed before any test score is inspected.
        candidate = score - 0.0005 * float(np.sum(authority / MAX_AUTHORITY))
        if candidate > best_value + 1e-12:
            best_value = candidate
            best_authority = authority
    return best_authority, best_value - baseline


def compute_development() -> pd.DataFrame:
    development, manifest = gate_relations()
    queries = development.groupby("protein_id").agg({
        "reaction_id": lambda x: list(x),
        "partition": "first",
        "novelty": lambda x: list(x),
        "difficulty_stratum": lambda x: list(x),
        "source": lambda x: sorted(set(x)),
    })
    rt = FinalBridgeRuntime(device="cuda")
    surface = FrozenE2RSurface(rt)
    rows = []
    for count, (query, sample) in enumerate(queries.iterrows(), 1):
        item = surface.score(query)
        targets = target_rows(surface, sample.reaction_id)
        base_ranks = filtered_target_ranks(item["order"], targets)
        weights = None
        gain = np.nan
        if sample.partition == "learn":
            weights, gain = teacher(surface, item, targets)
        row = {
            "query_id": query,
            "partition": sample.partition,
            "target_count": len(targets),
            "target_reactions": json.dumps(sample.reaction_id),
            "target_novelty": json.dumps(sample.novelty),
            "target_degree_strata": json.dumps(sample.difficulty_stratum),
            "source_groups": json.dumps(sample.source),
            "broad_ranks": json.dumps(base_ranks.tolist()),
            "oracle_gain": gain,
        }
        for i, name in enumerate(FEATURE_NAMES):
            row[name] = float(item["features"][i])
        for i, name in enumerate(CHANNELS):
            row[name + "_authority"] = float(weights[i]) if weights is not None else np.nan
        rows.append(row)
        if count % 200 == 0 or count == len(queries):
            print(f"GATE_DEV_FEATURES {count}/{len(queries)}", flush=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(OUT / "development_query_features.csv.gz", index=False)
    return frame


def balanced_query_weights(frame: pd.DataFrame) -> tuple[np.ndarray, dict]:
    """Balance four novelty types without discarding external Rhea relationships.

    Predeclared class target = 70% frozen parent novelty distribution +
    30% four-way uniform. All sampling weights come from degree/novelty
    metadata and source provenance, never validation/test ranking scores.
    """
    membership = pd.read_csv(SOURCE, usecols=["novelty"], dtype=str)
    dev = pd.read_csv(OUT / "gate_development_pairs.csv.gz",
                      usecols=["partition", "novelty", "source"])
    train_rows = dev[dev.partition.eq("learn")]
    order = ("both_seen", "protein_cold", "reaction_cold", "double_cold")
    reference = membership.novelty.value_counts(normalize=True).to_dict()
    observed = train_rows.novelty.value_counts(normalize=True).to_dict()
    target = {key: .70 * reference[key] + .30 / len(order) for key in order}
    scale = {key: float(np.clip(target[key] / observed[key], .25, 4.0))
             for key in order}
    qweight = []
    for row in frame.itertuples(index=False):
        labels = json.loads(row.target_novelty)
        if not labels:
            raise RuntimeError("Missing positive novelty labels for train query")
        # A query receives one weight, even if several relationships are known.
        mean = float(np.mean([scale[x] for x in labels]))
        qweight.append(mean * min(2.0, np.sqrt(len(labels))))
    weights = np.asarray(qweight, dtype=np.float64)
    weights *= len(weights) / weights.sum()
    if not np.isfinite(weights).all() or (weights <= 0).any():
        raise RuntimeError("Invalid nonnegative query balancing weights")
    summary = {
        "reference": reference, "observed_learner_pairs": observed,
        "predeclared_target_mix": target, "novelty_relative_weights_clipped": scale,
        "effective_query_weight_range": [float(weights.min()),float(weights.max())],
        "effective_query_weight_total": float(weights.sum()),
    }
    sources = {"parent_dev": 0.0, "rhea_independent": 0.0, "parent_and_rhea": 0.0}
    for row, w in zip(frame.itertuples(index=False), weights):
        provenance = json.loads(row.source_groups)
        key = "parent_and_rhea" if len(provenance)>1 else provenance[0]
        sources[key] += float(w)
    summary["effective_source_weight_fraction"] = {
        key: value / weights.sum() for key, value in sources.items()
    }
    return weights, summary


def train_models(frame: pd.DataFrame) -> dict:
    learn = frame.loc[frame.partition.eq("learn")].copy()
    valid = frame.loc[frame.partition.eq("validation")].copy()
    if len(learn) < 300 or len(valid) < 100:
        raise ValueError("Independent Rhea query partitions insufficient for gate learning")
    X = learn.loc[:, FEATURE_NAMES].to_numpy(np.float64)
    sample_weight, balancing = balanced_query_weights(learn)
    channels = {}
    audit = {}
    for channel, maximum in zip(CHANNELS, MAX_AUTHORITY):
        truth = learn[channel + "_authority"].to_numpy(np.float64)
        y = truth > 1e-8
        if len(np.unique(y)) == 1:
            clf = float(y[0])
        else:
            clf = HistGradientBoostingClassifier(
                max_iter=75, max_leaf_nodes=9, min_samples_leaf=28,
                l2_regularization=10, learning_rate=0.045, random_state=26,
            )
            clf.fit(X, y, sample_weight=sample_weight)
        positive = truth[y]
        if len(positive) < 45 or float(positive.std()) < 1e-8:
            reg = float(positive.mean()) if len(positive) else 0.0
        else:
            reg = HistGradientBoostingRegressor(
                max_iter=75, max_leaf_nodes=9, min_samples_leaf=28,
                l2_regularization=10, learning_rate=0.045, random_state=26,
            )
            reg.fit(X[y], positive, sample_weight=sample_weight[y])
        channels[channel] = {"permission": clf, "strength": reg}
        audit[channel] = {
            "learn_activation_rate": float(y.mean()),
            "mean_learn_oracle_authority": float(truth.mean()),
            "upper_authority_limit": maximum,
        }
    asset = {
        "schema": "bridge-e2r-query-authority-v3",
        "feature_names": FEATURE_NAMES,
        "channels": channels,
        "permission_threshold": 0.5,
        "topk_broad_reranking": 1000,
        "fit_relations": "combined parent minority and official disjoint Rhea; train/validation/evaluation protein queries strictly disjoint",
        "fit_query_count": len(learn),
        "heldout_query_count": len(valid),
        "train_audit": audit,
        "training_balance": balancing,
    }
    with ASSET.open("wb") as handle:
        pickle.dump(asset, handle)
    (OUT / "gate_fit.json").write_text(json.dumps({
        "schema": asset["schema"],
        "fit_query_count": len(learn),
        "heldout_query_count": len(valid),
        "channel_audit": audit,
        "training_balance": balancing,
        "permission_threshold": asset["permission_threshold"],
        "feature_count": len(FEATURE_NAMES),
    }, indent=2) + "\n")
    print(json.dumps(audit, indent=2))
    return asset


def score_report(ranks: list[int]) -> dict[str, float]:
    arr = np.asarray(ranks, dtype=np.int64)
    return {
        "edges": int(len(arr)),
        "mrr": float((1 / arr).mean()),
        "hit3": float((arr <= 3).mean()),
        "hit10": float((arr <= 10).mean()),
        "hit100": float((arr <= 100).mean()),
    }


def validation_check(frame: pd.DataFrame, asset: dict) -> dict:
    """Only pre-frozen gate_validation protein queries enter this acceptance report."""
    valid = frame.loc[frame.partition.eq("validation")]
    rt = FinalBridgeRuntime(device="cuda")
    surface = FrozenE2RSurface(rt)
    metrics: dict[str, list[int]] = defaultdict(list)
    gate_rows = []
    edge_rows = []
    for count, row in enumerate(valid.itertuples(index=False), 1):
        query = row.query_id
        item = surface.score(query)
        targets = target_rows(surface, json.loads(row.target_reactions))
        reference_features = np.asarray(
            [getattr(row, name) for name in FEATURE_NAMES]
        )
        feature_drift = float(np.max(np.abs(reference_features - item["features"])))
        if feature_drift > .05:
            raise AssertionError(
                f"E2R query evidence changes beyond bounded numerical sorting tolerance: "
                f"{query} drift={feature_drift}"
            )
        # Sub-float32-tie differences in rank-overlap and margins are audited;
        # live inference always recomputes features with no label-dependent lookup.
        auth = predict_authority(asset, item["features"])
        auth *= np.asarray([
            item["functional_av"], item["structure_av"], item["relation_av"]
        ], dtype=np.float64)
        broad = filtered_target_ranks(item["order"], targets)
        gated = ranks_from_weights(surface, item, targets, auth)
        fixed = ranks_from_weights(surface, item, targets, np.asarray([1.0, 1.0, 1.0]))
        metrics["broad"].extend(broad.tolist())
        metrics["gated"].extend(gated.tolist())
        metrics["fixed_reaction_head"].extend(fixed.tolist())
        for i, (novelty, difficulty) in enumerate(
            zip(json.loads(row.target_novelty),
                json.loads(row.target_degree_strata))
        ):
            edge_rows.append({
                "query_id": query, "novelty": novelty,
                "difficulty_stratum": difficulty,
                "broad_rank": int(broad[i]),
                "gated_rank": int(gated[i]),
                "fixed_rank": int(fixed[i]),
            })
        gate_rows.append({"query_id": query, "positive_count": len(targets),
                          "feature_drift_max": feature_drift,
                          "functional": float(auth[0]), "structure": float(auth[1]),
                          "relation": float(auth[2]),
                          "broad_top10": float((broad <= 10).mean()),
                          "gated_top10": float((gated <= 10).mean())})
        if count % 100 == 0 or count == len(valid):
            print(f"GATE_INTERNAL_VALIDATION {count}/{len(valid)}", flush=True)
    report = {
        "heldout_queries": len(valid),
        "methods": {key: score_report(values) for key, values in metrics.items()},
        "validation_relations_overlap_final_test": 0,
        "heldout_query_ids_unseen_during_gate_fit": True,
        "no_model_selection_from_benchmark": True,
        "max_feature_recomputation_drift": float(pd.DataFrame(gate_rows).feature_drift_max.max()),
    }
    evidence = pd.DataFrame(edge_rows)
    novelty_report = {}
    for novelty, group in evidence.groupby("novelty"):
        novelty_report[novelty] = {
            key: score_report(group[column].to_numpy(np.int64))
            for key, column in (
                ("broad", "broad_rank"),
                ("gated", "gated_rank"),
                ("fixed", "fixed_rank"),
            )
        }
    report["novelty_diagnostics"] = novelty_report
    from reproducibility.bime_rank.scripts.analyze_bridge_balanced_monotone_v4 import balanced
    report["degree_balanced"] = {
        method: balanced(evidence, rank_column)["balanced"]
        for method, rank_column in (
            ("broad", "broad_rank"), ("gated", "gated_rank"),
            ("fixed", "fixed_rank"),
        )
    }
    evidence.to_csv(OUT / "validation_edges.csv.gz", index=False)
    pd.DataFrame(gate_rows).to_csv(OUT / "validation_gate_query_audit.csv.gz", index=False)
    (OUT / "heldout_validation.json").write_text(json.dumps(report, indent=2) + "\n")
    print("HELDOUT_VALIDATION", json.dumps(report, indent=2), flush=True)
    return report


def evaluate_fixed_test(asset: dict, expected_hash: str) -> dict:
    """One evaluation-only pass over the retained query-heldout E2R relations.

    The complete original 23,773 parent and all frozen existing rank assets
    remain unchanged and reusable for audits.
    """
    if checksum(TEST) != expected_hash:
        raise RuntimeError("Frozen outer benchmark changed after selecting independent gate set")
    benchmark = pd.read_csv(
        SPLIT / "evaluation_pairs.csv.gz", dtype=str
    ).drop_duplicates(["protein_id", "reaction_id"])
    split = json.loads((SPLIT / "manifest.json").read_text())
    if checksum(TEST) != split["original_sha256"]:
        raise AssertionError("Changed parent benchmark")
    if len(benchmark) != split["parts"]["evaluation"]["edges"]:
        raise AssertionError("Changed frozen evaluation denominator")
    dev = pd.read_csv(OUT / "gate_development_pairs.csv.gz",
                      usecols=["protein_id", "reaction_id"], dtype=str)
    if set(zip(benchmark.protein_id, benchmark.reaction_id)) & set(zip(dev.protein_id, dev.reaction_id)) or set(benchmark.protein_id) & set(dev.protein_id):
        raise RuntimeError("Test labels accidentally present in E2R gate fitting surface")
    positive = benchmark.groupby("protein_id")["reaction_id"].apply(list)
    frozen = pd.read_csv(FROZEN, dtype={"protein_id": str, "reaction_id": str})
    ref = frozen.set_index(["protein_id", "reaction_id"])[["broad_rank", "full_rank"]]
    rt = FinalBridgeRuntime(device="cuda")
    surface = FrozenE2RSurface(rt)
    rows = []
    audit = []
    numerical_tie_drifts = []
    missing_q = 0
    for count, (q, pos) in enumerate(positive.items(), 1):
        if q not in rt.index.protein_index:
            missing_q += 1
            continue
        item = surface.score(q)
        tgt = target_rows(surface, pos)
        auth = predict_authority(asset, item["features"])
        auth *= np.asarray([
            item["functional_av"], item["structure_av"], item["relation_av"]
        ], dtype=np.float64)
        broad = filtered_target_ranks(item["order"], tgt)
        variants = {
            "full_rank": auth,
            "minus_functional_rank": auth * np.asarray([0., 1., 1.]),
            "minus_structure_mechanism_rank": auth * np.asarray([1., 0., 1.]),
            "minus_relational_memory_rank": auth * np.asarray([1., 1., 0.]),
        }
        results = {name: ranks_from_weights(surface, item, tgt, weights)
                   for name, weights in variants.items()}
        for i, reaction_id in enumerate(pos):
            # Immutable old Broad cache is authoritative for its own result.
            # Per-query fp32 GPU matvec vs previous batched GEMM may move
            # near-tied reaction scores by one or a few rank positions.
            # This cannot be used to select/tune gate weights.
            original_broad_rank = int(ref.loc[(q, reaction_id), "broad_rank"])
            local_broad_rank = int(broad[i])
            if local_broad_rank != original_broad_rank:
                numerical_tie_drifts.append({
                    "protein_id": q,
                    "reaction_id": reaction_id,
                    "old_cached_broad_rank": original_broad_rank,
                    "per_query_broad_rank": local_broad_rank,
                    "rank_difference": local_broad_rank - original_broad_rank,
                })
            rec = {"protein_id": q, "reaction_id": reaction_id,
                   "broad_rank": original_broad_rank,
                   "fixed_evidence_previous_rank": int(ref.loc[(q, reaction_id), "full_rank"])}
            rec.update({name: int(array[i]) for name, array in results.items()})
            rows.append(rec)
        audit.append({"protein_id": q, "positive_count": len(pos),
                      "functional_weight": float(auth[0]),
                      "structure_weight": float(auth[1]),
                      "relation_weight": float(auth[2])})
        if count % 500 == 0 or count == len(positive):
            print(f"E2R_FIXED_BENCHMARK_RESCORING {count}/{len(positive)} "
                  f"edges={len(rows)}", flush=True)
    edge = pd.DataFrame(rows)
    if len(edge) != len(benchmark) or missing_q:
        raise AssertionError(f"Test coverage changed: edges={len(edge)} missing queries={missing_q}")
    mismatched_fraction = len(numerical_tie_drifts) / len(edge)
    if mismatched_fraction > 0:
        print(
            f"NUMERICAL_BROAD_TIE_AUDIT fraction={mismatched_fraction:.4%}; "
            "reuse immutable batched Broad ranks as the published baseline, "
            "retain separate single-query predictions and drift diagnostics",
            flush=True,
        )
    edge.to_csv(OUT / "edge_metrics.csv.gz", index=False)
    pd.DataFrame(audit).to_csv(OUT / "query_audit.csv.gz", index=False)
    if numerical_tie_drifts:
        pd.DataFrame(numerical_tie_drifts).to_csv(
            OUT / "broad_fp32_tie_differences.csv.gz", index=False
        )
    methods = {
        name: score_report(edge[name].tolist())
        for name in ("broad_rank", "fixed_evidence_previous_rank", "full_rank",
                     "minus_functional_rank", "minus_structure_mechanism_rank",
                     "minus_relational_memory_rank")
    }
    report = {
        "schema": "bridge-e2r-query-authority-heldout-main-v3",
        "status": "complete",
        "outer_relations": len(edge),
        "original_parent_relations": 23773,
        "gate_dev_relations_excluded": split["parts"]["gate_learn"]["edges"] + split["parts"]["gate_validation"]["edges"],
        "unchanged_existing_rank_asset_restored_by_key_subset": True,
        "broad_cached_authoritative": True,
        "broad_fp32_near_tie_differences": len(numerical_tie_drifts),
        "broad_fp32_near_tie_fraction": mismatched_fraction,
        "broad_fp32_max_abs_rank_difference": int(
            max((abs(x["rank_difference"]) for x in numerical_tie_drifts), default=0)
        ),
        "new_gate_parameters_fixed_before_outer_evaluation": True,
        "outer_queries": len(positive),
        "sha256_outer_unchanged": checksum(TEST),
        "gate_training_pair_overlap": 0,
        "gate_training_query_overlap": 0,
        "external_rhea_gate_development_included": True,
        "old_long_term_relation_gate_used": False,
        "production_expert_checkpoints_modified": False,
        "candidate_universe": len(rt.index.reaction_ids),
        "broad_shortlist": 1000,
        "direct": methods,
        "query_authority": {
            channel: {
                "mean": float(pd.DataFrame(audit)[channel + "_weight"].mean()),
                "silent_query_fraction": float(
                    (pd.DataFrame(audit)[channel + "_weight"] < 1e-8).mean()
                ),
            } for channel in CHANNELS
        },
    }
    (OUT / "final_test_summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print("FINAL_FROZEN_TEST", json.dumps(report, indent=2), flush=True)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("prepare", "fit", "validate", "evaluate"), required=True)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.phase == "prepare":
        compute_development()
        return
    with ASSET.open("rb") if args.phase in ("validate", "evaluate") else open(
        "/dev/null", "rb"
    ) as f:
        asset = pickle.load(f) if args.phase in ("validate", "evaluate") else None
    if args.phase == "fit":
        frame = pd.read_csv(OUT / "development_query_features.csv.gz")
        train_models(frame)
    elif args.phase == "validate":
        frame = pd.read_csv(OUT / "development_query_features.csv.gz")
        validation_check(frame, asset)
    else:
        manifest = json.loads((OUT / "gate_development_manifest.json").read_text())
        evaluate_fixed_test(asset, manifest["original_target_sha256"])


if __name__ == "__main__":
    main()
