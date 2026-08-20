from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from projects.active.terpene_screening.rank_open_world import load_protein_library  # noqa: E402

DEFAULT_MANIFEST = ROOT / "results/terpene_wetlab_plate_manifest/assay_manifest.csv"
DEFAULT_RANKINGS = ROOT / "results/terpene_registry_batch/reaction_to_enzyme_rankings.csv"
DEFAULT_CURRENT_PROTEINS = ROOT / "data/terpene_embeddings/esmc600m_mean"
DEFAULT_REGISTERED_PROTEINS = ROOT / "data/terpene_open_world_registry/proteins"
DEFAULT_TEMPLATE = ROOT / "results/terpene_wetlab_plate_manifest/assay_results_template.csv"
DEFAULT_OUTPUT = ROOT / "results/terpene_wetlab_feedback"
ALLOWED_EXPRESSION = {"", "not_measured", "failed", "low", "adequate", "high"}
BOOLEAN_TRUE = {"true", "1", "yes", "y"}
BOOLEAN_FALSE = {"false", "0", "no", "n"}
DISCOVERY_ASSAY_ROLES = {"discovery_candidate", "uniprot_rescue_candidate"}
ASSAY_OUTCOME_STATES = {
    "not_tested",
    "control_failed",
    "expression_failed",
    "detection_inconclusive",
    "expressed_no_target_product",
    "positive_target_product",
    "positive_other_product",
}
OPTIONAL_RESULT_COLUMNS = {
    "operator_label": "",
    "other_product_detected": "",
    "other_product_annotation": "",
    "expression_host": "",
    "expression_vector": "",
    "expression_temperature_c": "",
    "induction_condition": "",
    "substrate_identity": "",
    "substrate_concentration": "",
    "metal_ion_condition": "",
    "detection_limit": "",
    "instrument_run_id": "",
    "experiment_batch_id": "",
    "biological_replicate": "",
}


def parse_bool(value: object) -> bool | None:
    text = str(value).strip().lower()
    if text in BOOLEAN_TRUE:
        return True
    if text in BOOLEAN_FALSE:
        return False
    return None


def initialize_template(manifest_path: Path, output_path: Path) -> pd.DataFrame:
    manifest = pd.read_csv(manifest_path, dtype=str).fillna("")
    template = manifest.copy()
    additions = {
        "expression_status": "not_measured",
        "soluble_expression": "",
        "assay_signal": "",
        "background_signal": "",
        "target_product_detected": "",
        "other_product_detected": "",
        "other_product_annotation": "",
        "product_identity_confidence": "",
        "technical_issue": "",
        "operator_label": "",
        "expression_host": "",
        "expression_vector": "",
        "expression_temperature_c": "",
        "induction_condition": "",
        "substrate_identity": "",
        "substrate_concentration": "",
        "metal_ion_condition": "",
        "detection_limit": "",
        "instrument_run_id": "",
        "experiment_batch_id": "",
        "biological_replicate": "",
        "notes": "",
    }
    for column, default in additions.items():
        template[column] = default
    output_path.parent.mkdir(parents=True, exist_ok=True)
    template.to_csv(output_path, index=False)
    return template


def ensure_optional_result_columns(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    for column, default in OPTIONAL_RESULT_COLUMNS.items():
        if column not in frame:
            frame[column] = default
    return frame


def validate_results(frame: pd.DataFrame) -> None:
    required = {
        "plate_id",
        "well",
        "reaction_id",
        "assay_role",
        "candidate_id",
        "expression_status",
        "target_product_detected",
        "product_identity_confidence",
        "technical_issue",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Wet-lab results are missing required columns: {missing}")
    duplicate = frame.duplicated(["plate_id", "well"])
    if duplicate.any():
        raise ValueError(f"Duplicate plate wells: {frame.loc[duplicate, ['plate_id', 'well']].to_dict('records')[:10]}")
    invalid_expression = sorted(set(frame["expression_status"].astype(str)) - ALLOWED_EXPRESSION)
    if invalid_expression:
        raise ValueError(f"Unsupported expression_status values: {invalid_expression}")


def infer_signal_ratio(frame: pd.DataFrame) -> pd.Series:
    signal = pd.to_numeric(frame["assay_signal"], errors="coerce")
    background = pd.to_numeric(frame["background_signal"], errors="coerce")
    denominator = background.where(background > 0)
    return signal / denominator


def reaction_qc(group: pd.DataFrame, identity_threshold: float) -> dict[str, object]:
    positives = group[group["assay_role"].astype(str).str.startswith("positive_control")]
    negatives = group[group["assay_role"].eq("empty_vector_negative")]
    blanks = group[group["assay_role"].eq("substrate_process_blank")]

    def control_detected(rows: pd.DataFrame) -> pd.Series:
        detected = rows["target_product_detected"].map(parse_bool).fillna(False)
        confidence = pd.to_numeric(rows["product_identity_confidence"], errors="coerce").fillna(0)
        technical = rows["technical_issue"].map(parse_bool).fillna(False)
        return detected & confidence.ge(identity_threshold) & ~technical

    positive_pass_count = int(control_detected(positives).sum())
    positive_pass = positive_pass_count >= 1
    negative_pass = bool(
        len(negatives)
        and negatives["target_product_detected"].map(parse_bool).fillna(False).eq(False).all()
        and negatives["technical_issue"].map(parse_bool).fillna(False).eq(False).all()
    )
    blank_pass = bool(
        len(blanks)
        and blanks["target_product_detected"].map(parse_bool).fillna(False).eq(False).all()
        and blanks["technical_issue"].map(parse_bool).fillna(False).eq(False).all()
    )
    return {
        "reaction_id": str(group.iloc[0]["reaction_id"]),
        "positive_control_wells": len(positives),
        "positive_control_pass_count": positive_pass_count,
        "positive_control_pass": positive_pass,
        "empty_vector_negative_pass": negative_pass,
        "substrate_process_blank_pass": blank_pass,
        "reaction_qc_pass": positive_pass and negative_pass and blank_pass,
    }


def classify_discovery_rows(
    frame: pd.DataFrame,
    qc_by_reaction: dict[str, bool],
    identity_threshold: float,
) -> pd.DataFrame:
    frame = ensure_optional_result_columns(frame)
    discovery = frame[frame["assay_role"].isin(DISCOVERY_ASSAY_ROLES)].copy()
    detected = discovery["target_product_detected"].map(parse_bool)
    other_detected = discovery["other_product_detected"].map(parse_bool)
    confidence = pd.to_numeric(discovery["product_identity_confidence"], errors="coerce")
    technical = discovery["technical_issue"].map(parse_bool).fillna(False)
    expression = discovery["expression_status"].astype(str)
    soluble = discovery["soluble_expression"].map(parse_bool)
    discovery["reaction_qc_pass"] = discovery["reaction_id"].map(qc_by_reaction).fillna(False)
    expression_qualified = expression.isin({"adequate", "high"}) | (
        expression.eq("low") & soluble.fillna(False)
    )
    has_measurement = (
        discovery["assay_signal"].astype(str).str.strip().ne("")
        | detected.notna()
        | other_detected.notna()
        | discovery["operator_label"].astype(str).str.strip().ne("")
    )
    not_tested = ~has_measurement & expression.isin({"", "not_measured"})
    control_failed = ~not_tested & ~discovery["reaction_qc_pass"]
    expression_failed = (
        ~not_tested
        & discovery["reaction_qc_pass"]
        & (expression.eq("failed") | soluble.eq(False))
    )
    positive_target = (
        discovery["reaction_qc_pass"]
        & detected.fillna(False)
        & confidence.ge(identity_threshold)
        & ~technical
        & ~expression.eq("failed")
    )
    positive_other = (
        discovery["reaction_qc_pass"]
        & ~detected.fillna(False)
        & other_detected.eq(True)
        & expression_qualified
        & ~technical
    )
    expressed_no_target = (
        discovery["reaction_qc_pass"]
        & detected.eq(False)
        & ~other_detected.eq(True)
        & expression_qualified
        & ~technical
    )
    outcome = np.select(
        [
            not_tested,
            control_failed,
            expression_failed,
            positive_target,
            positive_other,
            expressed_no_target,
        ],
        [
            "not_tested",
            "control_failed",
            "expression_failed",
            "positive_target_product",
            "positive_other_product",
            "expressed_no_target_product",
        ],
        default="detection_inconclusive",
    )
    discovery["assay_outcome_state"] = outcome
    if not set(discovery["assay_outcome_state"]).issubset(ASSAY_OUTCOME_STATES):
        raise ValueError("Unexpected wet-lab outcome state")
    discovery["feedback_label"] = np.select(
        [
            discovery["assay_outcome_state"].eq("positive_target_product"),
            discovery["assay_outcome_state"].eq("expressed_no_target_product"),
            discovery["assay_outcome_state"].eq("positive_other_product"),
        ],
        [
            "confirmed_positive",
            "expression_qualified_negative",
            "alternative_product_observation",
        ],
        default="inconclusive",
    )
    discovery["signal_to_background"] = infer_signal_ratio(discovery)
    discovery["training_weight"] = np.select(
        [
            discovery["feedback_label"].eq("confirmed_positive"),
            discovery["feedback_label"].eq("expression_qualified_negative"),
        ],
        [1.0, 0.5],
        default=0.0,
    )
    return discovery


def normalized(series: pd.Series, higher_is_better: bool = True) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    if values.notna().sum() == 0:
        result = pd.Series(0.5, index=series.index)
    else:
        values = values.fillna(values.median())
        low, high = float(values.min()), float(values.max())
        result = (values - low) / (high - low) if high > low else pd.Series(0.5, index=series.index)
    return result if higher_is_better else 1.0 - result


def select_next_iteration(
    feedback: pd.DataFrame,
    rankings: pd.DataFrame,
    feature_matrix: np.ndarray,
    candidate_ids: list[str],
    next_panel_size: int,
    ranking_objective: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    id_to_row = {value: index for index, value in enumerate(candidate_ids)}
    feedback_reaction_ids = set(feedback["reaction_id"].astype(str))
    rankings = rankings[
        rankings["ranking_objective"].eq(ranking_objective)
        & rankings["query_id"].astype(str).isin(feedback_reaction_ids)
    ].copy()
    numeric = [
        "rank",
        "score",
        "ensemble_rank_std",
        "ensemble_topk_vote_fraction",
        "known_associations_masked",
    ]
    for column in numeric:
        rankings[column] = pd.to_numeric(rankings[column], errors="coerce")
    tested_by_reaction = feedback.groupby("reaction_id")["candidate_id"].apply(set).to_dict()
    positive_by_reaction = (
        feedback[feedback["feedback_label"].eq("confirmed_positive")]
        .groupby("reaction_id")["candidate_id"]
        .apply(set)
        .to_dict()
    )
    negative_by_reaction = (
        feedback[feedback["feedback_label"].eq("expression_qualified_negative")]
        .groupby("reaction_id")["candidate_id"]
        .apply(set)
        .to_dict()
    )
    qc_by_reaction = feedback.groupby("reaction_id")["reaction_qc_pass"].first().to_dict()
    selected_rows = []
    summary_rows = []
    for reaction_id, pool in rankings.groupby("query_id", sort=True):
        tested = tested_by_reaction.get(reaction_id, set())
        pool = pool[~pool["candidate_id"].isin(tested) & pool["candidate_id"].isin(id_to_row)].copy()
        positives = positive_by_reaction.get(reaction_id, set()) & set(id_to_row)
        negatives = negative_by_reaction.get(reaction_id, set()) & set(id_to_row)
        if not bool(qc_by_reaction.get(reaction_id, False)):
            summary_rows.append(
                {
                    "reaction_id": reaction_id,
                    "reaction_qc_pass": False,
                    "confirmed_positives": len(positives),
                    "qualified_negatives": len(negatives),
                    "untested_ranked_candidates": len(pool),
                    "next_iteration_action": "rerun_controls_and_current_panel",
                    "next_candidates_selected": 0,
                }
            )
            continue
        if pool.empty:
            summary_rows.append(
                {
                    "reaction_id": reaction_id,
                    "reaction_qc_pass": True,
                    "confirmed_positives": len(positives),
                    "qualified_negatives": len(negatives),
                    "untested_ranked_candidates": 0,
                    "next_iteration_action": "expand_candidate_retrieval_beyond_top20",
                    "next_candidates_selected": 0,
                }
            )
            continue
        pool = pool.sort_values("rank").drop_duplicates("candidate_id").reset_index(drop=True)
        rows = np.asarray([id_to_row[value] for value in pool["candidate_id"]], dtype=np.int64)
        features = feature_matrix[rows]
        pool["rank_quality"] = normalized(pool["rank"], higher_is_better=False)
        pool["uncertainty_signal"] = normalized(pool["ensemble_rank_std"], higher_is_better=True)
        pool["agreement_signal"] = normalized(pool["ensemble_topk_vote_fraction"], higher_is_better=True)
        if positives:
            positive_rows = np.asarray([id_to_row[value] for value in sorted(positives)], dtype=np.int64)
            positive_similarity = (features @ feature_matrix[positive_rows].T).max(axis=1)
        else:
            positive_similarity = np.zeros(len(pool), dtype=np.float32)
        if negatives:
            negative_rows = np.asarray([id_to_row[value] for value in sorted(negatives)], dtype=np.int64)
            negative_similarity = (features @ feature_matrix[negative_rows].T).max(axis=1)
        else:
            negative_similarity = np.zeros(len(pool), dtype=np.float32)
        pool["positive_neighbor_similarity"] = positive_similarity
        pool["negative_neighbor_similarity"] = negative_similarity
        pool["outcome_utility"] = (
            0.45 * pool["rank_quality"]
            + 0.25 * pool["positive_neighbor_similarity"]
            - 0.20 * pool["negative_neighbor_similarity"]
            + 0.10 * pool["agreement_signal"]
        )
        selected: list[int] = []
        roles: dict[int, str] = {}
        n_exploit = next_panel_size // 2
        n_uncertainty = next_panel_size // 4
        n_diversity = next_panel_size - n_exploit - n_uncertainty
        for index in pool.sort_values(
            ["outcome_utility", "rank", "candidate_id"], ascending=[False, True, True]
        ).index:
            selected.append(int(index))
            roles[int(index)] = "outcome_exploitation" if positives else "model_exploitation"
            if len(selected) >= n_exploit:
                break
        for index in pool.sort_values(
            ["uncertainty_signal", "rank", "candidate_id"], ascending=[False, True, True]
        ).index:
            if int(index) in selected:
                continue
            selected.append(int(index))
            roles[int(index)] = "uncertainty"
            if len([value for value in roles.values() if value == "uncertainty"]) >= n_uncertainty:
                break
        while len([value for value in roles.values() if value == "diversity"]) < n_diversity:
            remaining = [int(index) for index in pool.index if int(index) not in selected]
            if not remaining:
                break
            values = []
            for index in remaining:
                min_distance = float(np.min(1.0 - features[selected] @ features[index])) if selected else 1.0
                values.append((min_distance + 0.10 * float(pool.loc[index, "rank_quality"]), index))
            _, index = max(values, key=lambda item: (item[0], -int(pool.loc[item[1], "rank"])))
            selected.append(index)
            roles[index] = "diversity"
        chosen = pool.loc[selected].copy()
        chosen["next_panel_role"] = [roles[index] for index in selected]
        chosen["next_panel_order"] = np.arange(1, len(chosen) + 1)
        chosen.insert(0, "reaction_id", reaction_id)
        selected_rows.append(chosen)
        summary_rows.append(
            {
                "reaction_id": reaction_id,
                "reaction_qc_pass": True,
                "confirmed_positives": len(positives),
                "qualified_negatives": len(negatives),
                "untested_ranked_candidates": len(pool),
                "next_iteration_action": "test_outcome_guided_panel" if len(chosen) else "expand_candidate_retrieval_beyond_top20",
                "next_candidates_selected": len(chosen),
            }
        )
    selected_frame = pd.concat(selected_rows, ignore_index=True) if selected_rows else pd.DataFrame()
    return selected_frame, pd.DataFrame(summary_rows)


def analyze_results(args: argparse.Namespace) -> None:
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    frame = ensure_optional_result_columns(
        pd.read_csv(args.results, dtype=str).fillna("")
    )
    validate_results(frame)
    qc_rows = [reaction_qc(group, args.identity_threshold) for _, group in frame.groupby("reaction_id")]
    qc = pd.DataFrame(qc_rows)
    qc_map = dict(zip(qc["reaction_id"], qc["reaction_qc_pass"].astype(bool)))
    feedback = classify_discovery_rows(frame, qc_map, args.identity_threshold)
    positives = feedback[feedback["feedback_label"].eq("confirmed_positive")].copy()
    negatives = feedback[feedback["feedback_label"].eq("expression_qualified_negative")].copy()
    alternative_products = feedback[
        feedback["feedback_label"].eq("alternative_product_observation")
    ].copy()
    inconclusive = feedback[feedback["feedback_label"].eq("inconclusive")].copy()
    training = feedback[
        feedback["feedback_label"].isin(
            {"confirmed_positive", "expression_qualified_negative"}
        )
    ][
        [
            "reaction_id",
            "candidate_id",
            "assay_outcome_state",
            "feedback_label",
            "training_weight",
            "signal_to_background",
            "product_identity_confidence",
            "experiment_batch_id",
            "biological_replicate",
        ]
    ].rename(columns={"reaction_id": "rhea_id", "candidate_id": "Entry"})

    current_features, current_ids = load_protein_library(args.current_protein_dir.resolve())
    registered_features, registered_ids = load_protein_library(args.registered_protein_dir.resolve())
    feature_blocks = [current_features, registered_features]
    candidate_ids = current_ids + registered_ids
    if args.additional_protein_dir is not None:
        additional_features, additional_ids = load_protein_library(
            args.additional_protein_dir.resolve()
        )
        duplicate_ids = sorted(set(candidate_ids) & set(additional_ids))
        if duplicate_ids:
            raise ValueError(
                f"Additional protein embedding IDs overlap existing candidates: {duplicate_ids[:20]}"
            )
        feature_blocks.append(additional_features)
        candidate_ids.extend(additional_ids)
    feature_matrix = np.concatenate(feature_blocks, axis=0)
    rankings = pd.read_csv(args.rankings, dtype=str).fillna("")
    next_candidates, next_summary = select_next_iteration(
        feedback,
        rankings,
        feature_matrix,
        candidate_ids,
        args.next_panel_size,
        args.ranking_objective,
    )

    qc.to_csv(output_dir / "reaction_qc.csv", index=False)
    feedback.to_csv(output_dir / "discovery_feedback.csv", index=False)
    positives.to_csv(output_dir / "confirmed_positive_assays.csv", index=False)
    negatives.to_csv(output_dir / "expression_qualified_negative_assays.csv", index=False)
    alternative_products.to_csv(
        output_dir / "alternative_product_observations.csv", index=False
    )
    inconclusive.to_csv(output_dir / "inconclusive_assays.csv", index=False)
    training.to_csv(output_dir / "wetlab_training_feedback.tsv", sep="\t", index=False)
    next_candidates.to_csv(output_dir / "next_iteration_candidates.csv", index=False)
    next_summary.to_csv(output_dir / "next_iteration_summary.csv", index=False)
    summary = {
        "n_reactions": int(frame["reaction_id"].nunique()),
        "reaction_qc_pass": int(qc["reaction_qc_pass"].sum()),
        "reaction_qc_fail": int((~qc["reaction_qc_pass"]).sum()),
        "confirmed_positive_assays": len(positives),
        "expression_qualified_negative_assays": len(negatives),
        "alternative_product_observations": len(alternative_products),
        "inconclusive_assays": len(inconclusive),
        "assay_outcome_states": feedback["assay_outcome_state"].value_counts().to_dict(),
        "next_iteration_candidates": len(next_candidates),
        "ranking_objective_used_for_next_iteration": args.ranking_objective,
        "additional_protein_embedding_dir": (
            str(args.additional_protein_dir.resolve())
            if args.additional_protein_dir is not None
            else None
        ),
        "label_policy": {
            "positive_target_product": "controls pass; target detected; identity threshold met; no technical issue",
            "positive_other_product": "controls pass; target absent; another product detected with qualified expression",
            "expressed_no_target_product": "controls pass; no target/other product; expression qualified; no technical issue",
            "control_failed": "reaction-level positive, negative, or blank controls failed",
            "expression_failed": "expression failed or protein was explicitly insoluble",
            "detection_inconclusive": "measurement exists but cannot support a target/other-product/qualified-negative conclusion",
            "not_tested": "no assay measurement and expression not measured",
        },
        "unlabelled_pairs_are_not_negative": True,
        "outputs": {
            "qc": str(output_dir / "reaction_qc.csv"),
            "feedback": str(output_dir / "discovery_feedback.csv"),
            "training_feedback": str(output_dir / "wetlab_training_feedback.tsv"),
            "next_candidates": str(output_dir / "next_iteration_candidates.csv"),
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="TPS wet-lab result template, QC and active-learning feedback.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    init_parser = subparsers.add_parser("init", help="Create a blank wet-lab result template.")
    init_parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    init_parser.add_argument("--output", type=Path, default=DEFAULT_TEMPLATE)

    analyze_parser = subparsers.add_parser("analyze", help="Analyze completed wet-lab results.")
    analyze_parser.add_argument("--results", type=Path, required=True)
    analyze_parser.add_argument("--rankings", type=Path, default=DEFAULT_RANKINGS)
    analyze_parser.add_argument("--current-protein-dir", type=Path, default=DEFAULT_CURRENT_PROTEINS)
    analyze_parser.add_argument("--registered-protein-dir", type=Path, default=DEFAULT_REGISTERED_PROTEINS)
    analyze_parser.add_argument("--additional-protein-dir", type=Path, default=None)
    analyze_parser.add_argument("--ranking-objective", default="top20")
    analyze_parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    analyze_parser.add_argument("--identity-threshold", type=float, default=0.8)
    analyze_parser.add_argument("--next-panel-size", type=int, default=8)
    args = parser.parse_args()

    if args.command == "init":
        template = initialize_template(args.manifest.resolve(), args.output.resolve())
        print(json.dumps({"output": str(args.output.resolve()), "rows": len(template)}, indent=2))
    else:
        analyze_results(args)


if __name__ == "__main__":
    main()
