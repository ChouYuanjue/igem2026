from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from projects.active.terpene_screening.rank_open_world import (  # noqa: E402
    load_external_reaction_rows,
    load_protein_library,
)
from projects.active.terpene_screening.rank_registry_batch import (  # noqa: E402
    build_known_association_maps,
)

DEFAULT_BATCH = ROOT / "results/terpene_registry_batch"
DEFAULT_CURRENT_PROTEINS = ROOT / "data/terpene_embeddings/esmc600m_mean"
DEFAULT_REGISTERED_PROTEINS = ROOT / "data/terpene_open_world_registry/proteins"
DEFAULT_CURRENT_CANDIDATES = ROOT / "data/terpene/all_seq_terpene_synthase.tsv"
DEFAULT_REGISTERED_REACTIONS = ROOT / "data/terpene_open_world_registry/reactions.csv"
DEFAULT_MARTS = ROOT / "data/terpene_marts/marts_reaction_pairs.tsv"
DEFAULT_POSITIVES = ROOT / "data/terpene/enzyme_terpene_synthase.tsv"
DEFAULT_OUTPUT = ROOT / "results/terpene_wetlab_discovery_panels"


def minmax(values: pd.Series, higher_is_better: bool = True) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    if numeric.notna().sum() == 0:
        result = pd.Series(np.zeros(len(values)), index=values.index, dtype=float)
    else:
        fill = numeric.median()
        numeric = numeric.fillna(fill)
        low, high = float(numeric.min()), float(numeric.max())
        result = (numeric - low) / (high - low) if high > low else pd.Series(0.5, index=values.index)
    return result if higher_is_better else 1.0 - result


def load_candidate_metadata(
    current_candidates_path: Path,
    marts_path: Path,
    current_ids: set[str],
    registered_ids: set[str],
) -> pd.DataFrame:
    current = pd.read_csv(current_candidates_path, sep="\t", dtype=str).fillna("")
    current = current[["Entry", "Sequence"]].rename(columns={"Sequence": "sequence"})
    current["candidate_source"] = "current"
    marts = pd.read_csv(marts_path, sep="\t", dtype=str).fillna("")
    annotation_columns = [
        "enzyme_id",
        "enzyme_name",
        "sequence",
        "species",
        "kingdom",
        "terpene_type",
        "tps_class",
        "publication",
    ]
    annotation = (
        marts[annotation_columns]
        .sort_values(["enzyme_id", "publication"], ascending=[True, False])
        .drop_duplicates("enzyme_id")
        .rename(columns={"enzyme_id": "Entry"})
    )
    rows = pd.DataFrame({"Entry": sorted(current_ids | registered_ids)})
    rows["candidate_source"] = np.where(rows["Entry"].isin(current_ids), "current", "registered_external")
    rows = rows.merge(current[["Entry", "sequence"]], on="Entry", how="left", suffixes=("", "_current"))
    rows = rows.merge(annotation, on="Entry", how="left", suffixes=("_current", "_marts"))
    rows["sequence"] = rows["sequence_current"].fillna("")
    marts_sequence = rows["sequence_marts"].fillna("")
    rows.loc[rows["sequence"].eq(""), "sequence"] = marts_sequence[rows["sequence"].eq("")]
    cleaned_sequence = rows["sequence"].astype(str).str.replace(r"\s+", "", regex=True).str.upper()
    rows["sequence"] = cleaned_sequence
    rows["sequence_length"] = cleaned_sequence.str.len()
    rows["contains_noncanonical_residue"] = cleaned_sequence.map(
        lambda value: bool(set(value) - set("ACDEFGHIKLMNPQRSTVWY"))
    )
    rows["sequence_review_reason"] = np.select(
        [
            rows["sequence_length"].lt(200),
            rows["sequence_length"].gt(1000),
            rows["contains_noncanonical_residue"],
        ],
        ["very_short", "very_long", "noncanonical_residue"],
        default="",
    )
    rows["sequence_eligible"] = rows["sequence_review_reason"].eq("")
    return rows.drop(columns=[column for column in ["sequence_current", "sequence_marts"] if column in rows])


def load_reaction_metadata(registered_reactions_path: Path, marts_path: Path) -> pd.DataFrame:
    reactions = load_external_reaction_rows(registered_reactions_path)
    registry = pd.read_csv(registered_reactions_path, dtype=str).fillna("")
    reactions = reactions.merge(registry[["reaction_id", "reaction_signature"]], on="reaction_id", how="left")
    marts = pd.read_csv(marts_path, sep="\t", dtype=str).fillna("")
    annotation = (
        marts[
            [
                "reaction_signature",
                "substrate_name",
                "product_name",
                "terpene_type",
                "tps_class",
                "has_mechanism",
                "publication",
            ]
        ]
        .sort_values(["reaction_signature", "publication"], ascending=[True, False])
        .drop_duplicates("reaction_signature")
    )
    return reactions.merge(annotation, on="reaction_signature", how="left")


def pairwise_distance_summary(features: np.ndarray) -> tuple[float, float]:
    if len(features) < 2:
        return 0.0, 0.0
    similarity = features @ features.T
    upper = 1.0 - similarity[np.triu_indices(len(features), 1)]
    return float(np.mean(upper)), float(np.min(upper))


def choose_with_similarity_guard(
    ordered_indices: list[int],
    features: np.ndarray,
    selected: list[int],
    count: int,
    maximum_similarity: float,
) -> list[int]:
    chosen: list[int] = []
    for index in ordered_indices:
        if index in selected or index in chosen:
            continue
        previous = selected + chosen
        if previous and float(np.max(features[previous] @ features[index])) >= maximum_similarity:
            continue
        chosen.append(index)
        if len(chosen) >= count:
            break
    if len(chosen) < count:
        for index in ordered_indices:
            if index not in selected and index not in chosen:
                chosen.append(index)
                if len(chosen) >= count:
                    break
    return chosen


def select_panel(
    pool: pd.DataFrame,
    feature_matrix: np.ndarray,
    id_to_row: dict[str, int],
    hub_frequency: dict[str, int],
    n_exploit: int,
    n_uncertainty: int,
    n_diversity: int,
    maximum_similarity: float,
    eligible_ids: set[str],
) -> tuple[pd.DataFrame, dict[str, float]]:
    original_pool_size = int(pool["candidate_id"].nunique())
    pool = pool[pool["candidate_id"].astype(str).isin(eligible_ids)].copy()
    pool = pool.sort_values("rank").drop_duplicates("candidate_id").reset_index(drop=True)
    desired = n_exploit + n_uncertainty + n_diversity
    if len(pool) < desired:
        raise ValueError(
            f"Only {len(pool)} sequence-eligible candidates remain; {desired} are required"
        )
    feature_rows = np.asarray([id_to_row[value] for value in pool["candidate_id"].astype(str)], dtype=np.int64)
    features = feature_matrix[feature_rows]
    max_hub = max(hub_frequency.values()) if hub_frequency else 1
    pool["hub_frequency_top20"] = pool["candidate_id"].map(hub_frequency).fillna(0).astype(int)
    pool["hub_penalty"] = np.log1p(pool["hub_frequency_top20"]) / math.log1p(max_hub)
    pool["rank_quality"] = 1.0 - (pool["rank"].astype(float) - 1.0) / max(1.0, float(pool["rank"].max() - 1.0))
    pool["uncertainty_signal"] = minmax(pool["ensemble_rank_std"])
    pool["agreement_signal"] = minmax(pool["ensemble_topk_vote_fraction"])
    pool["exploitation_utility"] = pool["rank_quality"] - 0.12 * pool["hub_penalty"]
    pool["uncertainty_utility"] = (
        0.50 * pool["rank_quality"]
        + 0.35 * pool["uncertainty_signal"]
        + 0.15 * pool["agreement_signal"]
        - 0.12 * pool["hub_penalty"]
    )

    selected: list[int] = []
    roles: dict[int, str] = {}
    exploit_order = pool.sort_values(
        ["exploitation_utility", "rank", "candidate_id"], ascending=[False, True, True]
    ).index.tolist()
    for index in choose_with_similarity_guard(
        exploit_order, features, selected, n_exploit, maximum_similarity
    ):
        selected.append(index)
        roles[index] = "exploitation"

    uncertainty_order = pool.sort_values(
        ["uncertainty_utility", "rank", "candidate_id"], ascending=[False, True, True]
    ).index.tolist()
    for index in choose_with_similarity_guard(
        uncertainty_order, features, selected, n_uncertainty, maximum_similarity
    ):
        selected.append(index)
        roles[index] = "uncertainty"

    while len([value for value in roles.values() if value == "diversity"]) < n_diversity:
        remaining = [index for index in pool.index if index not in selected]
        if not remaining:
            break
        best_index, best_value = None, -np.inf
        for index in remaining:
            if selected:
                min_distance = float(np.min(1.0 - features[selected] @ features[index]))
            else:
                min_distance = 1.0
            source_bonus = 0.03 if str(pool.loc[index, "candidate_source"]) == "registered_external" else 0.0
            value = (
                min_distance
                + 0.15 * float(pool.loc[index, "rank_quality"])
                - 0.08 * float(pool.loc[index, "hub_penalty"])
                + source_bonus
            )
            if value > best_value or (
                value == best_value
                and (int(pool.loc[index, "rank"]), str(pool.loc[index, "candidate_id"]))
                < (int(pool.loc[best_index, "rank"]), str(pool.loc[best_index, "candidate_id"]))
            ):
                best_index, best_value = index, value
        assert best_index is not None
        selected.append(best_index)
        roles[best_index] = "diversity"

    if len(selected) < desired:
        for index in pool.sort_values(["rank", "candidate_id"]).index:
            if index not in selected:
                selected.append(index)
                roles[index] = "fallback"
                if len(selected) >= desired:
                    break

    result = pool.loc[selected].copy()
    result["panel_role"] = [roles[index] for index in selected]
    result["panel_order"] = np.arange(1, len(result) + 1)
    selected_features = features[np.asarray(selected, dtype=np.int64)]
    mean_distance, minimum_distance = pairwise_distance_summary(selected_features)
    diagnostics = {
        "panel_size": len(result),
        "mean_pairwise_esm_distance": mean_distance,
        "minimum_pairwise_esm_distance": minimum_distance,
        "external_candidate_fraction": float(result["candidate_source"].eq("registered_external").mean()),
        "mean_hub_frequency": float(result["hub_frequency_top20"].mean()),
        "maximum_hub_frequency": int(result["hub_frequency_top20"].max()),
        "mean_original_rank": float(result["rank"].mean()),
        "sequence_risk_candidates_excluded": original_pool_size - len(pool),
        "sequence_eligible_pool_size": len(pool),
    }
    return result, diagnostics


def choose_positive_control(
    known_ids: set[str],
    metadata: pd.DataFrame,
    id_to_row: dict[str, int],
) -> pd.Series | None:
    candidates = metadata[metadata["Entry"].isin(known_ids & set(id_to_row))].copy()
    if candidates.empty:
        return None
    candidates["source_priority"] = candidates["candidate_source"].map(
        {"current": 0, "registered_external": 1}
    ).fillna(2)
    candidates["length_priority"] = (pd.to_numeric(candidates["sequence_length"], errors="coerce") - 500).abs()
    candidates["annotation_priority"] = candidates["enzyme_name"].fillna("").eq("").astype(int)
    return candidates.sort_values(
        ["source_priority", "annotation_priority", "length_priority", "Entry"]
    ).iloc[0]


def select_balanced_campaign(
    summary: pd.DataFrame,
    count: int,
    core_types: set[str],
    class2_minimum: int,
) -> pd.DataFrame:
    working = summary[summary["terpene_type"].astype(str).isin(core_types)].copy()
    if len(working) <= count:
        return working.sort_values(["campaign_priority_score", "reaction_id"], ascending=[False, True])
    selected: list[int] = []
    type_counts: dict[str, int] = {}
    substrate_counts: dict[str, int] = {}
    control_counts: dict[str, int] = {}

    def add(index: int) -> None:
        if index in selected:
            return
        selected.append(index)
        row = working.loc[index]
        terpene_type = str(row["terpene_type"])
        substrate = str(row["substrate_name"])
        control = str(row["positive_control_id"])
        type_counts[terpene_type] = type_counts.get(terpene_type, 0) + 1
        substrate_counts[substrate] = substrate_counts.get(substrate, 0) + 1
        if control:
            control_counts[control] = control_counts.get(control, 0) + 1

    # Guarantee one high-quality representative for every core terpene type.
    for _, group in working.groupby("terpene_type", sort=True):
        index = int(
            group.sort_values(
                ["campaign_priority_score", "reaction_id"], ascending=[False, True]
            ).index[0]
        )
        add(index)

    # Preserve a minimum number of class-II reactions where available.
    class2_target = min(class2_minimum, int(working["tps_class"].astype(str).eq("2").sum()))
    while sum(str(working.loc[index, "tps_class"]) == "2" for index in selected) < class2_target:
        candidates = working[
            working["tps_class"].astype(str).eq("2") & ~working.index.isin(selected)
        ]
        if candidates.empty:
            break
        index = int(
            candidates.sort_values(
                ["campaign_priority_score", "reaction_id"], ascending=[False, True]
            ).index[0]
        )
        add(index)

    while len(selected) < count:
        best_index, best_score = None, -np.inf
        for index, row in working[~working.index.isin(selected)].iterrows():
            terpene_type = str(row["terpene_type"])
            substrate = str(row["substrate_name"])
            control = str(row["positive_control_id"])
            score = float(row["campaign_priority_score"])
            score -= 0.055 * type_counts.get(terpene_type, 0)
            score -= 0.030 * substrate_counts.get(substrate, 0)
            score -= 0.020 * control_counts.get(control, 0)
            if str(row["tps_class"]) == "2" and sum(
                str(working.loc[value, "tps_class"]) == "2" for value in selected
            ) < class2_target + 1:
                score += 0.03
            if score > best_score or (
                score == best_score
                and str(row["reaction_id"]) < str(working.loc[best_index, "reaction_id"])
            ):
                best_index, best_score = int(index), score
        if best_index is None:
            break
        add(best_index)

    result = working.loc[selected].copy()
    result["balanced_campaign_order"] = np.arange(1, len(result) + 1)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Build diversified wet-lab panels from masked TPS discovery rankings.")
    parser.add_argument("--batch-dir", type=Path, default=DEFAULT_BATCH)
    parser.add_argument("--current-protein-dir", type=Path, default=DEFAULT_CURRENT_PROTEINS)
    parser.add_argument("--registered-protein-dir", type=Path, default=DEFAULT_REGISTERED_PROTEINS)
    parser.add_argument("--current-candidates", type=Path, default=DEFAULT_CURRENT_CANDIDATES)
    parser.add_argument("--registered-reactions", type=Path, default=DEFAULT_REGISTERED_REACTIONS)
    parser.add_argument("--marts", type=Path, default=DEFAULT_MARTS)
    parser.add_argument("--positives", type=Path, default=DEFAULT_POSITIVES)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--n-exploitation", type=int, default=6)
    parser.add_argument("--n-uncertainty", type=int, default=3)
    parser.add_argument("--n-diversity", type=int, default=3)
    parser.add_argument("--maximum-sequence-embedding-similarity", type=float, default=0.95)
    parser.add_argument("--campaign-reactions", type=int, default=24)
    parser.add_argument("--extended-campaign-reactions", type=int, default=8)
    parser.add_argument("--class2-minimum", type=int, default=4)
    args = parser.parse_args()

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    batch_dir = args.batch_dir.resolve()
    rankings = pd.read_csv(batch_dir / "reaction_to_enzyme_rankings.csv", dtype=str).fillna("")
    rankings = rankings[rankings["ranking_objective"].eq("top20")].copy()
    numeric_columns = [
        "rank",
        "score",
        "ensemble_score_mean",
        "ensemble_score_std",
        "ensemble_rank_mean",
        "ensemble_rank_std",
        "ensemble_topk_vote_fraction",
        "ensemble_top1_vote_fraction",
        "ensemble_top1_rank_std",
        "ensemble_top1_margin_z",
        "ensemble_topk_jaccard",
        "ensemble_topk_vote_mean",
        "known_associations_masked",
    ]
    for column in numeric_columns:
        if column in rankings:
            rankings[column] = pd.to_numeric(rankings[column], errors="coerce")

    current_features, current_ids = load_protein_library(args.current_protein_dir.resolve())
    registered_features, registered_ids = load_protein_library(args.registered_protein_dir.resolve())
    feature_matrix = np.concatenate([current_features, registered_features], axis=0)
    candidate_ids = current_ids + registered_ids
    id_to_row = {value: index for index, value in enumerate(candidate_ids)}
    current_id_set, registered_id_set = set(current_ids), set(registered_ids)
    metadata = load_candidate_metadata(
        args.current_candidates.resolve(),
        args.marts.resolve(),
        current_id_set,
        registered_id_set,
    )
    metadata_map = metadata.set_index("Entry")
    eligible_ids = set(metadata.loc[metadata["sequence_eligible"], "Entry"].astype(str))
    reaction_metadata = load_reaction_metadata(
        args.registered_reactions.resolve(), args.marts.resolve()
    ).set_index("reaction_id")
    registered_reaction_ids = set(reaction_metadata.index.astype(str))
    _, known_enzymes_by_reaction = build_known_association_maps(
        args.marts.resolve(), args.positives.resolve(), registered_reaction_ids
    )
    hub_frequency = rankings["candidate_id"].value_counts().astype(int).to_dict()

    panel_rows: list[pd.DataFrame] = []
    control_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    for reaction_id, pool in rankings.groupby("query_id", sort=True):
        pool = pool[pool["candidate_id"].isin(id_to_row)].copy()
        pool["candidate_source"] = np.where(
            pool["candidate_id"].isin(current_id_set), "current", "registered_external"
        )
        panel, diagnostics = select_panel(
            pool,
            feature_matrix,
            id_to_row,
            hub_frequency,
            args.n_exploitation,
            args.n_uncertainty,
            args.n_diversity,
            args.maximum_sequence_embedding_similarity,
            eligible_ids,
        )
        panel.insert(0, "reaction_id", reaction_id)
        panel["is_known_positive_control"] = False
        panel = panel.merge(metadata, left_on="candidate_id", right_on="Entry", how="left", suffixes=("", "_metadata"))
        panel_rows.append(panel)

        known_ids = set(known_enzymes_by_reaction.get(reaction_id, set()))
        control = choose_positive_control(known_ids, metadata, id_to_row)
        control_id = None
        if control is not None:
            control_id = str(control["Entry"])
            control_rows.append(
                {
                    "reaction_id": reaction_id,
                    "candidate_id": control_id,
                    "panel_role": "positive_control",
                    "candidate_source": control["candidate_source"],
                    "sequence": control["sequence"],
                    "sequence_length": control["sequence_length"],
                    "enzyme_name": control.get("enzyme_name", ""),
                    "species": control.get("species", ""),
                    "kingdom": control.get("kingdom", ""),
                    "terpene_type": control.get("terpene_type", ""),
                    "tps_class": control.get("tps_class", ""),
                    "publication": control.get("publication", ""),
                }
            )
        reaction_row = reaction_metadata.loc[reaction_id] if reaction_id in reaction_metadata.index else pd.Series(dtype=object)
        query_row = pool.iloc[0]
        summary_rows.append(
            {
                "reaction_id": reaction_id,
                "reaction_signature": reaction_row.get("reaction_signature", ""),
                "reaction_smiles": reaction_row.get("reaction_smiles", ""),
                "substrate_name": reaction_row.get("substrate_name", ""),
                "product_name": reaction_row.get("product_name", ""),
                "terpene_type": reaction_row.get("terpene_type", ""),
                "tps_class": reaction_row.get("tps_class", ""),
                "has_mechanism": reaction_row.get("has_mechanism", ""),
                "known_associations_masked": int(query_row.get("known_associations_masked", len(known_ids))),
                "positive_control_id": control_id,
                "positive_control_available": control_id is not None,
                "ensemble_top1_vote_fraction": float(query_row["ensemble_top1_vote_fraction"]),
                "ensemble_top1_rank_std": float(query_row["ensemble_top1_rank_std"]),
                "ensemble_top1_margin_z": float(query_row["ensemble_top1_margin_z"]),
                "ensemble_topk_jaccard": float(query_row["ensemble_topk_jaccard"]),
                "ensemble_topk_vote_mean": float(query_row["ensemble_topk_vote_mean"]),
                **diagnostics,
            }
        )

    panels = pd.concat(panel_rows, ignore_index=True)
    controls = pd.DataFrame(control_rows)
    summary = pd.DataFrame(summary_rows)
    summary["diversity_percentile"] = summary["mean_pairwise_esm_distance"].rank(pct=True)
    summary["agreement_percentile"] = summary["ensemble_topk_jaccard"].rank(pct=True)
    summary["vote_percentile"] = summary["ensemble_topk_vote_mean"].rank(pct=True)
    summary["nonhub_percentile"] = summary["maximum_hub_frequency"].rank(pct=True, ascending=False)
    summary["control_score"] = summary["positive_control_available"].astype(float)
    summary["campaign_priority_score"] = (
        0.30 * summary["diversity_percentile"]
        + 0.25 * summary["agreement_percentile"]
        + 0.20 * summary["vote_percentile"]
        + 0.15 * summary["nonhub_percentile"]
        + 0.10 * summary["control_score"]
    )
    summary = summary.sort_values(
        ["campaign_priority_score", "reaction_id"], ascending=[False, True]
    ).reset_index(drop=True)
    summary["campaign_priority_rank"] = np.arange(1, len(summary) + 1)
    priority_only = summary.head(args.campaign_reactions).copy()
    core_types = {"mono", "sesq", "di", "sester", "tri", "sesquar"}
    core_campaign = select_balanced_campaign(
        summary,
        args.campaign_reactions,
        core_types,
        args.class2_minimum,
    )
    extended_campaign = (
        summary[~summary["terpene_type"].astype(str).isin(core_types)]
        .head(args.extended_campaign_reactions)
        .copy()
    )
    campaign_ids = set(core_campaign["reaction_id"].astype(str))
    extended_ids = set(extended_campaign["reaction_id"].astype(str))
    campaign_panels = panels[panels["reaction_id"].isin(campaign_ids)].copy()
    campaign_controls = controls[controls["reaction_id"].isin(campaign_ids)].copy()
    extended_panels = panels[panels["reaction_id"].isin(extended_ids)].copy()
    extended_controls = controls[controls["reaction_id"].isin(extended_ids)].copy()

    panels.to_csv(output_dir / "reaction_discovery_panels.csv", index=False)
    controls.to_csv(output_dir / "reaction_positive_controls.csv", index=False)
    summary.to_csv(output_dir / "reaction_panel_summary.csv", index=False)
    priority_only.to_csv(output_dir / "campaign_reactions_priority_only.csv", index=False)
    core_campaign.to_csv(output_dir / "campaign_reactions.csv", index=False)
    campaign_panels.to_csv(output_dir / "campaign_discovery_candidates.csv", index=False)
    campaign_controls.to_csv(output_dir / "campaign_positive_controls.csv", index=False)
    extended_campaign.to_csv(output_dir / "extended_pathway_reactions.csv", index=False)
    extended_panels.to_csv(output_dir / "extended_pathway_candidates.csv", index=False)
    extended_controls.to_csv(output_dir / "extended_pathway_positive_controls.csv", index=False)

    result = {
        "n_reactions": int(summary["reaction_id"].nunique()),
        "discovery_candidates_per_reaction": args.n_exploitation + args.n_uncertainty + args.n_diversity,
        "n_discovery_rows": len(panels),
        "n_positive_controls": len(controls),
        "campaign_reactions": len(core_campaign),
        "campaign_scope": "balanced_core_tps",
        "campaign_discovery_rows": len(campaign_panels),
        "extended_pathway_reactions": len(extended_campaign),
        "extended_pathway_discovery_rows": len(extended_panels),
        "selection_allocation": {
            "exploitation": args.n_exploitation,
            "uncertainty": args.n_uncertainty,
            "diversity": args.n_diversity,
        },
        "known_associations_are_discovery_masked": True,
        "masked_query_empirical_reliability_used": False,
        "sequence_risk_filter": {
            "minimum_length": 200,
            "maximum_length": 1000,
            "canonical_amino_acids_only": True,
            "total_risky_candidates_excluded_from_pools": int(summary["sequence_risk_candidates_excluded"].sum()),
        },
        "outputs": {
            "panels": str(output_dir / "reaction_discovery_panels.csv"),
            "controls": str(output_dir / "reaction_positive_controls.csv"),
            "summary": str(output_dir / "reaction_panel_summary.csv"),
            "campaign_reactions": str(output_dir / "campaign_reactions.csv"),
            "campaign_candidates": str(output_dir / "campaign_discovery_candidates.csv"),
            "extended_pathway_reactions": str(output_dir / "extended_pathway_reactions.csv"),
            "extended_pathway_candidates": str(output_dir / "extended_pathway_candidates.csv"),
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(core_campaign[
        [
            "balanced_campaign_order",
            "campaign_priority_rank",
            "reaction_id",
            "terpene_type",
            "tps_class",
            "substrate_name",
            "product_name",
            "campaign_priority_score",
            "mean_pairwise_esm_distance",
            "positive_control_id",
        ]
    ].to_string(index=False))


if __name__ == "__main__":
    main()
