from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from projects.active.terpene_screening.analyze_uniprot_expansion_quality import (  # noqa: E402
    pfam_architecture,
)

DEFAULT_CANONICAL = ROOT / "results/terpene_wetlab_plate_manifest/assay_manifest.csv"
DEFAULT_RESCUE = ROOT / "results/terpene_uniprot_rescue_campaign/assay_manifest.csv"
DEFAULT_RESCUE_METADATA = (
    ROOT
    / "results/terpene_uniprot_rescue_campaign/selected_uniprot_rescue_candidates.csv"
)
DEFAULT_OUTPUT = ROOT / "results/terpene_wetlab_plate_balanced"


@dataclass(frozen=True)
class CampaignSpec:
    scope: str
    candidate_role: str
    role_column: str
    reactions_per_plate: int
    columns_per_reaction: int


CANONICAL_SPEC = CampaignSpec(
    scope="canonical_discovery",
    candidate_role="discovery_candidate",
    role_column="panel_role",
    reactions_per_plate=6,
    columns_per_reaction=2,
)
RESCUE_SPEC = CampaignSpec(
    scope="uniprot_rescue",
    candidate_role="uniprot_rescue_candidate",
    role_column="rescue_role",
    reactions_per_plate=12,
    columns_per_reaction=1,
)


def clean_sequence(value: object) -> str:
    return "".join(str(value).upper().split()).rstrip("*")


def stable_fraction(*parts: object) -> float:
    value = "|".join(map(str, parts))
    return int(hashlib.sha1(value.encode("utf-8")).hexdigest()[:12], 16) / float(16**12)


def attach_rescue_architecture(
    manifest: pd.DataFrame,
    metadata_path: Path,
) -> pd.DataFrame:
    """Attach exact Pfam combinations and architectures to rescue candidate wells."""
    metadata = pd.read_csv(metadata_path, dtype=str).fillna("")
    required = {"reaction_id", "candidate_id", "pfam_combination"}
    missing = sorted(required - set(metadata.columns))
    if missing:
        raise ValueError(f"Rescue metadata missing required columns: {missing}")
    metadata = metadata[["reaction_id", "candidate_id", "pfam_combination"]].drop_duplicates(
        ["reaction_id", "candidate_id"]
    )
    metadata["pfam_architecture"] = metadata["pfam_combination"].map(pfam_architecture)
    result = manifest.drop(
        columns=["pfam_combination", "pfam_architecture"], errors="ignore"
    ).merge(
        metadata,
        on=["reaction_id", "candidate_id"],
        how="left",
        validate="many_to_one",
    )
    candidate_mask = result["assay_role"].eq(RESCUE_SPEC.candidate_role)
    missing_candidates = result.loc[
        candidate_mask & result["pfam_architecture"].fillna("").eq(""),
        ["reaction_id", "candidate_id"],
    ]
    if len(missing_candidates):
        raise ValueError(
            "Missing Pfam architecture for rescue candidates: "
            f"{missing_candidates.head(20).to_dict('records')}"
        )
    result[["pfam_combination", "pfam_architecture"]] = result[
        ["pfam_combination", "pfam_architecture"]
    ].fillna("")
    return result


def reaction_features(frame: pd.DataFrame, spec: CampaignSpec) -> pd.DataFrame:
    working = frame.copy().fillna("")
    working["sequence_clean"] = working["sequence"].map(clean_sequence)
    working["sequence_length_calc"] = working["sequence_clean"].str.len()
    rows: list[dict[str, object]] = []
    for reaction_id, group in working.groupby("reaction_id", sort=False):
        candidates = group[group["assay_role"].eq(spec.candidate_role)].copy()
        controls = group[group["assay_role"].eq("positive_control_primary")]
        if len(candidates) == 0:
            raise ValueError(f"No candidate wells for {reaction_id}")
        source = candidates.get("candidate_source", pd.Series([""] * len(candidates), index=candidates.index))
        tier = candidates.get("evidence_quality_tier", pd.Series([""] * len(candidates), index=candidates.index))
        architecture = candidates.get("pfam_architecture", pd.Series([""] * len(candidates), index=candidates.index))
        rows.append(
            {
                "reaction_id": str(reaction_id),
                "original_plate_id": str(group["plate_id"].iloc[0]),
                "original_reaction_order": int(pd.to_numeric(group["reaction_order"], errors="raise").iloc[0]),
                "terpene_type": str(group["terpene_type"].iloc[0]),
                "tps_class": str(group["tps_class"].iloc[0]),
                "substrate_name": str(group["substrate_name"].iloc[0]),
                "product_name": str(group["product_name"].iloc[0]),
                "positive_control_id": str(controls["candidate_id"].iloc[0]) if len(controls) else "",
                "candidate_median_length": float(candidates["sequence_length_calc"].median()),
                "candidate_q90_length": float(candidates["sequence_length_calc"].quantile(0.9)),
                "candidate_external_fraction": float(source.isin(["registered_external", "uniprot_primary"]).mean()),
                "candidate_unique_sequences": int(candidates["sequence_clean"].nunique()),
                "A_count": int(tier.eq("A_reviewed").sum()),
                "B_count": int(tier.eq("B_experimental_or_transcript_named").sum()),
                "C_count": int(tier.eq("C_homology_named").sum()),
                "D_count": int(tier.eq("D_named_predicted").sum()),
                "bacterial_classI_count": int(architecture.eq("bacterial_classI").sum()),
                "plant_tps_full_count": int(architecture.eq("plant_tps_full").sum()),
                "classI_hybrid_full_count": int(architecture.eq("classI_hybrid_full").sum()),
                "osc_full_count": int(architecture.eq("osc_full").sum()),
            }
        )
    result = pd.DataFrame(rows).sort_values("original_reaction_order").reset_index(drop=True)
    plate_count = frame["plate_id"].nunique()
    expected = plate_count * spec.reactions_per_plate
    if len(result) != expected:
        raise ValueError(
            f"{spec.scope}: expected {expected} reactions for {plate_count} plates, found {len(result)}"
        )
    return result


def feature_vectors(reactions: pd.DataFrame, spec: CampaignSpec) -> list[tuple[str, np.ndarray, float]]:
    features: list[tuple[str, np.ndarray, float]] = []

    def add_categorical(column: str, weight: float) -> None:
        values = reactions[column].astype(str)
        for category in sorted(set(values) - {""}):
            features.append(
                (f"{column}::{category}", values.eq(category).astype(float).to_numpy(), weight)
            )

    add_categorical("terpene_type", 30.0)
    add_categorical("tps_class", 45.0)
    add_categorical("substrate_name", 6.0)
    add_categorical("positive_control_id", 5.0)

    numeric_weights = {
        "candidate_median_length": 16.0,
        "candidate_q90_length": 8.0,
        "candidate_external_fraction": 8.0,
        "candidate_unique_sequences": 2.0,
        "A_count": 10.0,
        "B_count": 8.0,
        "C_count": 6.0,
        "D_count": 6.0,
        "bacterial_classI_count": 8.0,
        "plant_tps_full_count": 8.0,
        "classI_hybrid_full_count": 8.0,
        "osc_full_count": 10.0,
    }
    for column, weight in numeric_weights.items():
        values = pd.to_numeric(reactions[column], errors="coerce").fillna(0.0).to_numpy(float)
        if np.allclose(values, values[0]):
            continue
        scale = float(np.std(values))
        if scale < 1e-8:
            scale = max(float(np.mean(np.abs(values))), 1.0)
        normalized = (values - float(np.mean(values))) / scale
        features.append((column, normalized, weight))
    return features


def optimize_assignment(
    reactions: pd.DataFrame,
    plate_ids: list[str],
    spec: CampaignSpec,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    n = len(reactions)
    p = len(plate_ids)
    features = feature_vectors(reactions, spec)
    n_x = n * p
    n_dev = len(features) * p
    total_variables = n_x + n_dev

    objective = np.zeros(total_variables, dtype=float)
    original_plate = reactions["original_plate_id"].astype(str).tolist()
    for i, reaction_id in enumerate(reactions["reaction_id"].astype(str)):
        for plate_index, plate_id in enumerate(plate_ids):
            index = i * p + plate_index
            objective[index] = (
                (0.03 if plate_id != original_plate[i] else 0.0)
                + 1e-5 * stable_fraction(seed, spec.scope, reaction_id, plate_id)
            )
    for feature_index, (_, _, weight) in enumerate(features):
        for plate_index in range(p):
            objective[n_x + feature_index * p + plate_index] = weight

    lower = np.zeros(total_variables, dtype=float)
    upper = np.full(total_variables, np.inf, dtype=float)
    upper[:n_x] = 1.0
    integrality = np.zeros(total_variables, dtype=int)
    integrality[:n_x] = 1

    row_indices: list[int] = []
    column_indices: list[int] = []
    data: list[float] = []
    constraint_lower: list[float] = []
    constraint_upper: list[float] = []
    row = 0

    # Each reaction is assigned to exactly one plate.
    for i in range(n):
        for plate_index in range(p):
            row_indices.append(row)
            column_indices.append(i * p + plate_index)
            data.append(1.0)
        constraint_lower.append(1.0)
        constraint_upper.append(1.0)
        row += 1

    # Each plate has exact capacity.
    for plate_index in range(p):
        for i in range(n):
            row_indices.append(row)
            column_indices.append(i * p + plate_index)
            data.append(1.0)
        constraint_lower.append(float(spec.reactions_per_plate))
        constraint_upper.append(float(spec.reactions_per_plate))
        row += 1

    # Absolute deviation from each feature target.
    for feature_index, (_, values, _) in enumerate(features):
        target = float(np.sum(values) / p)
        for plate_index in range(p):
            deviation_index = n_x + feature_index * p + plate_index
            # sum(values*x) - deviation <= target
            for i, value in enumerate(values):
                if value != 0:
                    row_indices.append(row)
                    column_indices.append(i * p + plate_index)
                    data.append(float(value))
            row_indices.append(row)
            column_indices.append(deviation_index)
            data.append(-1.0)
            constraint_lower.append(-np.inf)
            constraint_upper.append(target)
            row += 1
            # -sum(values*x) - deviation <= -target
            for i, value in enumerate(values):
                if value != 0:
                    row_indices.append(row)
                    column_indices.append(i * p + plate_index)
                    data.append(float(-value))
            row_indices.append(row)
            column_indices.append(deviation_index)
            data.append(-1.0)
            constraint_lower.append(-np.inf)
            constraint_upper.append(-target)
            row += 1

    matrix = coo_matrix(
        (data, (row_indices, column_indices)), shape=(row, total_variables)
    ).tocsr()
    result = milp(
        c=objective,
        integrality=integrality,
        bounds=Bounds(lower, upper),
        constraints=LinearConstraint(
            matrix,
            np.asarray(constraint_lower, dtype=float),
            np.asarray(constraint_upper, dtype=float),
        ),
        options={"time_limit": 60.0, "mip_rel_gap": 0.0},
    )
    if not result.success or result.x is None:
        raise RuntimeError(
            f"MILP failed for {spec.scope}: status={result.status}, message={result.message}"
        )
    assignment_matrix = result.x[:n_x].reshape(n, p)
    assigned_indices = assignment_matrix.argmax(axis=1)
    if not np.all(assignment_matrix[np.arange(n), assigned_indices] > 0.5):
        raise RuntimeError("MILP returned non-integral reaction assignments")
    assigned = reactions.copy()
    assigned["balanced_plate_id"] = [plate_ids[index] for index in assigned_indices]
    assigned["plate_changed"] = assigned["balanced_plate_id"].ne(
        assigned["original_plate_id"]
    )
    assigned = assigned.sort_values(
        ["balanced_plate_id", "original_reaction_order", "reaction_id"]
    )
    assigned["balanced_plate_position"] = (
        assigned.groupby("balanced_plate_id").cumcount() + 1
    )
    assigned = assigned.sort_values("original_reaction_order").reset_index(drop=True)

    audit_rows: list[dict[str, object]] = []
    for layout, plate_column in [
        ("before", "original_plate_id"),
        ("after", "balanced_plate_id"),
    ]:
        for plate_id, group in assigned.groupby(plate_column, sort=True):
            payload: dict[str, object] = {
                "campaign_scope": spec.scope,
                "layout": layout,
                "plate_id": plate_id,
                "n_reactions": len(group),
                "terpene_type_counts": json.dumps(
                    group["terpene_type"].value_counts().to_dict(), sort_keys=True
                ),
                "tps_class_counts": json.dumps(
                    group["tps_class"].value_counts().to_dict(), sort_keys=True
                ),
                "unique_substrates": group["substrate_name"].nunique(),
                "unique_positive_controls": group["positive_control_id"].nunique(),
                "candidate_median_length_mean": float(
                    group["candidate_median_length"].mean()
                ),
                "candidate_q90_length_mean": float(
                    group["candidate_q90_length"].mean()
                ),
                "candidate_external_fraction_mean": float(
                    group["candidate_external_fraction"].mean()
                ),
                "A_total": int(group["A_count"].sum()),
                "B_total": int(group["B_count"].sum()),
                "C_total": int(group["C_count"].sum()),
                "D_total": int(group["D_count"].sum()),
                "bacterial_classI_total": int(group["bacterial_classI_count"].sum()),
                "plant_tps_full_total": int(group["plant_tps_full_count"].sum()),
                "classI_hybrid_full_total": int(group["classI_hybrid_full_count"].sum()),
                "osc_full_total": int(group["osc_full_count"].sum()),
            }
            audit_rows.append(payload)
    audit = pd.DataFrame(audit_rows)
    summary = {
        "campaign_scope": spec.scope,
        "solver_success": bool(result.success),
        "solver_status": int(result.status),
        "solver_message": str(result.message),
        "objective_value": float(result.fun),
        "n_reactions": n,
        "n_plates": p,
        "reactions_per_plate": spec.reactions_per_plate,
        "reactions_changed_plate": int(assigned["plate_changed"].sum()),
        "feature_count": len(features),
        "features": [name for name, _, _ in features],
    }
    return assigned, audit, summary


def remap_manifest(
    manifest: pd.DataFrame,
    assignment: pd.DataFrame,
    spec: CampaignSpec,
) -> pd.DataFrame:
    result = manifest.copy().fillna("")
    result.insert(0, "original_plate_id", result["plate_id"].astype(str))
    result.insert(1, "original_well", result["well"].astype(str))
    result.insert(2, "plate_balance_method", "capacity_constrained_milp_v1")
    mapping = assignment.set_index("reaction_id")
    result["balanced_plate_position"] = result["reaction_id"].map(
        mapping["balanced_plate_position"]
    ).astype(int)
    result["plate_id"] = result["reaction_id"].map(
        mapping["balanced_plate_id"]
    )
    new_wells: list[str] = []
    for row in result.itertuples(index=False):
        old_row = str(row.original_well)[0]
        old_column = int(str(row.original_well)[1:])
        old_group = manifest[manifest["reaction_id"].astype(str).eq(str(row.reaction_id))]
        old_columns = sorted(
            {int(str(value)[1:]) for value in old_group["well"].astype(str)}
        )
        local_column = old_columns.index(old_column)
        start_column = (int(row.balanced_plate_position) - 1) * spec.columns_per_reaction + 1
        new_column = start_column + local_column
        new_wells.append(f"{old_row}{new_column}")
    result["well"] = new_wells
    if result.duplicated(["plate_id", "well"]).any():
        duplicate = result.loc[
            result.duplicated(["plate_id", "well"], keep=False),
            ["plate_id", "well", "reaction_id", "assay_role"],
        ]
        raise ValueError(f"Balanced layout has duplicate wells: {duplicate.to_dict('records')[:20]}")
    plate_sizes = result.groupby("plate_id").size()
    if not plate_sizes.eq(96).all():
        raise ValueError(f"Balanced plates are not 96 wells: {plate_sizes.to_dict()}")
    return result


def compact_balance_summary(
    audit: pd.DataFrame,
    diagnostics: pd.DataFrame,
) -> dict[str, object]:
    compact: dict[str, object] = {}
    for scope, scope_audit in audit.groupby("campaign_scope", sort=True):
        scope_payload: dict[str, object] = {}
        for layout in ("before", "after"):
            group = scope_audit[scope_audit["layout"].eq(layout)]
            type_dicts = [json.loads(value) for value in group["terpene_type_counts"]]
            class_dicts = [json.loads(value) for value in group["tps_class_counts"]]
            type_categories = sorted(set().union(*(set(value) for value in type_dicts)))
            class_categories = sorted(set().union(*(set(value) for value in class_dicts)))
            scope_payload[layout] = {
                "type_range_sum": int(
                    sum(
                        max(value.get(category, 0) for value in type_dicts)
                        - min(value.get(category, 0) for value in type_dicts)
                        for category in type_categories
                    )
                ),
                "class_range_sum": int(
                    sum(
                        max(value.get(category, 0) for value in class_dicts)
                        - min(value.get(category, 0) for value in class_dicts)
                        for category in class_categories
                    )
                ),
            }
        for metric in [
            "candidate_median_length_mean",
            "candidate_q90_length_mean",
            "candidate_external_fraction_mean",
            "A_total",
            "B_total",
            "C_total",
            "D_total",
            "bacterial_classI_total",
            "plant_tps_full_total",
            "classI_hybrid_full_total",
            "osc_full_total",
        ]:
            rows = diagnostics[
                diagnostics["campaign_scope"].eq(scope)
                & diagnostics["metric"].eq(metric)
            ].set_index("layout")
            if {"before", "after"}.issubset(rows.index):
                scope_payload[metric] = {
                    "before_range": float(rows.loc["before", "range"]),
                    "after_range": float(rows.loc["after", "range"]),
                }
        compact[str(scope)] = scope_payload
    return compact


def campaign_diagnostics(audit: pd.DataFrame) -> pd.DataFrame:
    numeric = [
        "unique_substrates",
        "unique_positive_controls",
        "candidate_median_length_mean",
        "candidate_q90_length_mean",
        "candidate_external_fraction_mean",
        "A_total",
        "B_total",
        "C_total",
        "D_total",
        "bacterial_classI_total",
        "plant_tps_full_total",
        "classI_hybrid_full_total",
        "osc_full_total",
    ]
    rows: list[dict[str, object]] = []
    for (scope, layout), group in audit.groupby(["campaign_scope", "layout"]):
        for column in numeric:
            values = pd.to_numeric(group[column], errors="coerce")
            rows.append(
                {
                    "campaign_scope": scope,
                    "layout": layout,
                    "metric": column,
                    "minimum": float(values.min()),
                    "maximum": float(values.max()),
                    "range": float(values.max() - values.min()),
                    "coefficient_of_variation": (
                        float(values.std(ddof=0) / values.mean())
                        if values.mean() != 0
                        else 0.0
                    ),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Balance TPS reaction blocks across plates with exact capacity-constrained MILP."
    )
    parser.add_argument("--canonical-manifest", type=Path, default=DEFAULT_CANONICAL)
    parser.add_argument("--rescue-manifest", type=Path, default=DEFAULT_RESCUE)
    parser.add_argument("--rescue-metadata", type=Path, default=DEFAULT_RESCUE_METADATA)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=20260723)
    args = parser.parse_args()

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    all_audits: list[pd.DataFrame] = []
    summaries: dict[str, object] = {}
    for spec, path, output_name in [
        (CANONICAL_SPEC, args.canonical_manifest, "canonical_balanced_assay_manifest.csv"),
        (RESCUE_SPEC, args.rescue_manifest, "uniprot_balanced_assay_manifest.csv"),
    ]:
        manifest = pd.read_csv(path, dtype=str).fillna("")
        if spec == RESCUE_SPEC:
            manifest = attach_rescue_architecture(manifest, args.rescue_metadata.resolve())
        reactions = reaction_features(manifest, spec)
        plate_ids = sorted(manifest["plate_id"].astype(str).unique())
        assignment, audit, summary = optimize_assignment(
            reactions, plate_ids, spec, args.seed
        )
        balanced = remap_manifest(manifest, assignment, spec)
        assignment.to_csv(
            output_dir / f"{spec.scope}_reaction_assignment.csv", index=False
        )
        audit.to_csv(output_dir / f"{spec.scope}_plate_audit.csv", index=False)
        balanced.to_csv(output_dir / output_name, index=False)
        for plate_id, group in balanced.groupby("plate_id", sort=True):
            group.sort_values(
                "well",
                key=lambda values: values.map(
                    lambda value: (int(str(value)[1:]), str(value)[0])
                ),
            ).to_csv(output_dir / f"{plate_id}_balanced_layout.csv", index=False)
        all_audits.append(audit)
        summaries[spec.scope] = summary

    combined_audit = pd.concat(all_audits, ignore_index=True)
    combined_audit.to_csv(output_dir / "plate_balance_audit.csv", index=False)
    diagnostics = campaign_diagnostics(combined_audit)
    diagnostics.to_csv(output_dir / "plate_balance_diagnostics.csv", index=False)
    compact = compact_balance_summary(combined_audit, diagnostics)
    (output_dir / "compact_balance_summary.json").write_text(
        json.dumps(compact, indent=2), encoding="utf-8"
    )
    summary = {
        "seed": args.seed,
        "method": "capacity_constrained_milp_v1",
        "campaigns": summaries,
        "outputs": {
            "canonical_manifest": str(
                output_dir / "canonical_balanced_assay_manifest.csv"
            ),
            "rescue_manifest": str(
                output_dir / "uniprot_balanced_assay_manifest.csv"
            ),
            "audit": str(output_dir / "plate_balance_audit.csv"),
            "diagnostics": str(output_dir / "plate_balance_diagnostics.csv"),
            "compact_summary": str(output_dir / "compact_balance_summary.json"),
        },
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    print(combined_audit.to_string(index=False))
    print(diagnostics.to_string(index=False))


if __name__ == "__main__":
    main()
