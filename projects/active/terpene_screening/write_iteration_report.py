from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
OUTPUT = ROOT / "results/terpene_research_iteration_report.md"
OUTPUT_JSON = ROOT / "results/terpene_research_iteration_summary.json"


def pct(value: float) -> str:
    return f"{100 * float(value):.1f}%"


def row_for(frame: pd.DataFrame, **filters: str) -> pd.Series:
    selected = frame
    for column, value in filters.items():
        selected = selected[selected[column].astype(str).eq(str(value))]
    if len(selected) != 1:
        raise ValueError(f"Expected one row for {filters}, found {len(selected)}")
    return selected.iloc[0]


def metric_cells(row: pd.Series) -> list[str]:
    return [
        pct(row["hit_probability_at_3"]),
        pct(row["hit_probability_at_10"]),
        pct(row["hit_probability_at_20"]),
        f"{float(row['mean_reciprocal_rank']):.3f}",
        f"{float(row['median_best_positive_rank']):.1f}",
    ]


def main() -> None:
    standard = pd.read_csv(ROOT / "results/terpene_marts_domain_adaptation_cartesian_drfp/metrics.csv")
    pu = pd.read_csv(ROOT / "results/terpene_marts_domain_adaptation_cartesian_pu/metrics.csv")
    e2r_specialized = pd.read_csv(ROOT / "results/terpene_marts_domain_adaptation_freeze_reaction/metrics.csv")
    r2e_top3_10 = pd.read_csv(ROOT / "results/terpene_marts_domain_adaptation_r2e075/metrics.csv")
    hybrid_r2e = pd.read_csv(ROOT / "results/terpene_marts_adapted_neighbor_hybrid/metrics.csv")
    hybrid_e2r = pd.read_csv(ROOT / "results/terpene_marts_freeze_reaction_neighbor_hybrid/metrics.csv")
    mechanism = pd.read_csv(ROOT / "results/terpene_marts_mechanism_rescue_pu/metrics.csv")
    retention_r2e = pd.read_csv(ROOT / "results/terpene_production_retention/comparison.csv")
    retention_e2r = pd.read_csv(ROOT / "results/terpene_production_retention_e2r/comparison.csv")
    deployment_r2e = json.loads((ROOT / "results/terpene_deployment_validation_pu.json").read_text(encoding="utf-8"))
    deployment_e2r = json.loads((ROOT / "results/terpene_deployment_validation_e2r.json").read_text(encoding="utf-8"))
    deployment_r2e_top3_10 = json.loads((ROOT / "results/terpene_deployment_validation_r2e075.json").read_text(encoding="utf-8"))
    deployment_r2e_exact = json.loads((ROOT / "results/terpene_deployment_validation_r2e_exact_residual.json").read_text(encoding="utf-8"))
    deployment_e2r_hardneg = json.loads((ROOT / "results/terpene_deployment_validation_marts_adapted_drfp_pu_e2r_hardneg128.json").read_text(encoding="utf-8"))
    production_r2e = json.loads(
        (ROOT / "results/terpene_production_models/marts_adapted_drfp_pu/summary.json").read_text(encoding="utf-8")
    )
    production_e2r = json.loads(
        (ROOT / "results/terpene_production_models/marts_adapted_drfp_pu_e2r/summary.json").read_text(encoding="utf-8")
    )
    production_r2e_top3_10 = json.loads(
        (ROOT / "results/terpene_production_models/marts_adapted_drfp_pu_r2e075/summary.json").read_text(encoding="utf-8")
    )
    production_r2e_exact = json.loads(
        (ROOT / "results/terpene_production_models/marts_adapted_drfp_pu_r2e_exact_residual/summary.json").read_text(encoding="utf-8")
    )
    production_e2r_hardneg = json.loads(
        (ROOT / "results/terpene_production_models/marts_adapted_drfp_pu_e2r_hardneg128/summary.json").read_text(encoding="utf-8")
    )
    production_e2r_dual_kernel = json.loads(
        (ROOT / "results/terpene_production_models/marts_dual_kernel_e2r_top20/summary.json").read_text(encoding="utf-8")
    )
    exact_r2e_features = pd.read_csv(
        ROOT / "results/terpene_exact_residual_uncertainty/query_uncertainty_features.csv"
    )
    rrf_legacy = pd.read_csv(
        ROOT / "results/terpene_e2r_route_interleaving_locked_rrf_summary.csv"
    )
    rrf_confirmatory = json.loads(
        (ROOT / "results/terpene_e2r_route_interleaving_confirmatory20260725/locked_rrf_confirmatory_summary.json").read_text(encoding="utf-8")
    )
    dual_kernel_route = json.loads(
        (ROOT / "results/terpene_marts_dual_kernel_rescue_route_v1/summary.json").read_text(encoding="utf-8")
    )
    dual_kernel_confirmatory = json.loads(
        (ROOT / "results/terpene_marts_dual_kernel_confirmatory20260726/locked_confirmatory_summary.json").read_text(encoding="utf-8")
    )
    registry_entries = pd.read_csv(ROOT / "data/terpene_open_world_registry/proteins/entries.csv")
    registry_reactions = pd.read_csv(ROOT / "data/terpene_open_world_registry/reactions.csv")
    uncertainty_summary = pd.read_csv(
        ROOT / "results/terpene_open_world_uncertainty_rrf_routing/calibration_summary.csv"
    )
    selective_performance = pd.read_csv(
        ROOT / "results/terpene_open_world_uncertainty_rrf_routing/selective_performance.csv"
    )
    uniprot_report = json.loads(
        (ROOT / "results/terpene_uniprot_expansion_report_summary.json").read_text(
            encoding="utf-8"
        )
    )
    uniprot_free = pd.DataFrame(uniprot_report["free_merge_paired_retention"])
    uniprot_quota = pd.DataFrame(uniprot_report["selected_quota"])
    randomization = json.loads(
        (ROOT / "results/terpene_wetlab_randomized_layout/summary.json").read_text(
            encoding="utf-8"
        )
    )
    combined_campaign = json.loads(
        (ROOT / "results/terpene_combined_wetlab_campaign/summary.json").read_text(
            encoding="utf-8"
        )
    )
    plate_balance = json.loads(
        (ROOT / "results/terpene_wetlab_plate_balanced/summary.json").read_text(
            encoding="utf-8"
        )
    )
    plate_balance_compact = json.loads(
        (
            ROOT
            / "results/terpene_wetlab_plate_balanced/compact_balance_summary.json"
        ).read_text(encoding="utf-8")
    )

    external_rows = []
    e2r_base = row_for(standard, method="current_production", direction="enzyme_to_reaction")
    e2r_pu = row_for(pu, method="marts_adapted", direction="enzyme_to_reaction")
    e2r_selected = row_for(e2r_specialized, method="marts_adapted", direction="enzyme_to_reaction")
    for label, row in [
        ("Current production", e2r_base),
        ("Shared MARTS + PU", e2r_pu),
        ("E2R specialized: frozen reaction tower", e2r_selected),
    ]:
        external_rows.append(["enzyme_to_reaction", label, *metric_cells(row)])
    r2e_base = row_for(standard, method="current_production", direction="reaction_to_enzyme")
    r2e_standard = row_for(standard, method="marts_adapted", direction="reaction_to_enzyme")
    r2e_selected = row_for(pu, method="marts_adapted", direction="reaction_to_enzyme")
    r2e_short = row_for(r2e_top3_10, method="marts_adapted", direction="reaction_to_enzyme")
    for label, row in [
        ("Current production", r2e_base),
        ("MARTS adaptation", r2e_standard),
        ("Shared R2E MARTS + PU", r2e_selected),
        ("R2E short-list specialized: loss weight 0.75", r2e_short),
    ]:
        external_rows.append(["reaction_to_enzyme", label, *metric_cells(row)])

    e2r_route_metrics = hybrid_e2r[hybrid_e2r["direction"].eq("enzyme_to_reaction")].copy()
    e2r_top3 = e2r_route_metrics[e2r_route_metrics["method"].eq("rank_hybrid_direct_0.75")].iloc[0]
    e2r_top20_calibration = uncertainty_summary[
        uncertainty_summary["calibrator"].eq("enzyme_to_reaction_top20")
    ].iloc[0]
    rrf_legacy_row = rrf_legacy[rrf_legacy["split_assignment"].eq("legacy")].iloc[0]
    rrf_confirmatory_20260724 = rrf_legacy[
        rrf_legacy["split_assignment"].eq("confirmatory20260724")
    ].iloc[0]
    exact_top10 = exact_r2e_features[exact_r2e_features["budget"].eq(10)]
    exact_top20 = exact_r2e_features[exact_r2e_features["budget"].eq(20)]
    route_rows = [
        ["enzyme_to_reaction", "Top-3", "freeze-reaction + 5-neighbor hybrid (direct 0.75)", pct(e2r_top3["hit_probability_at_3"])],
        ["enzyme_to_reaction", "Top-10", "RRF: 0.35 freeze-route + 0.65 hard-negative route, c=60", pct(rrf_legacy_row["rrf_hit_at_10"])],
        ["enzyme_to_reaction", "Top-20", "RRF: 0.70 freeze-route + 0.30 dual-kernel collaborative support, c=60", pct(e2r_top20_calibration["base_hit_rate"])],
        ["reaction_to_enzyme", "Top-3", "reaction-loss-0.75 direct", pct(r2e_short["hit_probability_at_3"])],
        ["reaction_to_enzyme", "Top-10", "Horizyn exact-residual direct", pct(exact_top10["hit"].mean())],
        ["reaction_to_enzyme", "Top-20", "Horizyn exact-residual direct", pct(exact_top20["hit"].mean())],
    ]
    rrf_confirmation_rows = [
        [
            "legacy development/evaluation split",
            int(rrf_legacy_row["n_query_cells"]),
            pct(rrf_legacy_row["rrf_hit_at_10"]),
            pct(rrf_legacy_row["production_hit_at_10"]),
            f"{100 * float(rrf_legacy_row['absolute_delta']):+.2f} pp",
            f"[{100 * float(rrf_legacy_row['ci_low']):+.2f}, {100 * float(rrf_legacy_row['ci_high']):+.2f}] pp",
        ],
        [
            "confirmatory fold seed 20260724",
            int(rrf_confirmatory_20260724["n_query_cells"]),
            pct(rrf_confirmatory_20260724["rrf_hit_at_10"]),
            pct(rrf_confirmatory_20260724["production_hit_at_10"]),
            f"{100 * float(rrf_confirmatory_20260724['absolute_delta']):+.2f} pp",
            f"[{100 * float(rrf_confirmatory_20260724['ci_low']):+.2f}, {100 * float(rrf_confirmatory_20260724['ci_high']):+.2f}] pp",
        ],
        [
            "locked confirmatory fold seed 20260725",
            int(rrf_confirmatory["n_query_cells"]),
            pct(rrf_confirmatory["rrf_hit_at_10"]),
            pct(rrf_confirmatory["production_hit_at_10"]),
            f"{100 * float(rrf_confirmatory['absolute_delta']):+.2f} pp",
            f"[{100 * float(rrf_confirmatory['bootstrap_ci_low']):+.2f}, {100 * float(rrf_confirmatory['bootstrap_ci_high']):+.2f}] pp",
        ],
    ]


    dual_kernel_confirmation_rows = [
        [
            "development cells (parameter selection)",
            "—",
            pct(dual_kernel_route["selected_development"]["hit_at_20"]),
            "—",
            "—",
            f"MRR {dual_kernel_route['selected_development']['mrr']:.3f}",
        ],
        [
            "original frozen 16 cells",
            int(dual_kernel_route["frozen"]["n"]),
            pct(dual_kernel_route["frozen"]["selected_hit"]),
            pct(dual_kernel_route["frozen"]["production_hit"]),
            f"{100 * float(dual_kernel_route['frozen']['difference']):+.2f} pp",
            f"[{100 * float(dual_kernel_route['frozen']['bootstrap_ci_low']):+.2f}, {100 * float(dual_kernel_route['frozen']['bootstrap_ci_high']):+.2f}] pp",
        ],
        [
            "locked independent fold seed 20260726",
            int(dual_kernel_confirmatory["n_query_cells"]),
            pct(dual_kernel_confirmatory["fused_hit"]),
            pct(dual_kernel_confirmatory["production_hit"]),
            f"{100 * float(dual_kernel_confirmatory['difference']):+.2f} pp",
            f"[{100 * float(dual_kernel_confirmatory['bootstrap_ci_low']):+.2f}, {100 * float(dual_kernel_confirmatory['bootstrap_ci_high']):+.2f}] pp",
        ],
    ]

    reliability_rows = []
    for row in uncertainty_summary.itertuples(index=False):
        direction, budget_text = str(row.calibrator).rsplit("_top", 1)
        budget = int(budget_text)
        selective = selective_performance[
            selective_performance["direction"].eq(direction)
            & selective_performance["budget"].eq(budget)
            & selective_performance["coverage"].eq(0.25)
        ]
        high_quartile_hit = (
            pct(selective.iloc[0]["hit_rate"]) if len(selective) else "—"
        )
        reliability_rows.append(
            [
                direction,
                f"Top-{budget}",
                "deployed" if bool(row.deployable) else "not deployed",
                f"{float(row.roc_auc):.3f}",
                f"[{float(row.roc_auc_ci_low):.3f}, {float(row.roc_auc_ci_high):.3f}]",
                pct(row.base_hit_rate),
                high_quartile_hit,
            ]
        )

    mechanism_direct = row_for(mechanism, coverage="mechanism_available", method="adapted_direct")
    mechanism_transfer = row_for(mechanism, coverage="mechanism_available", method="mechanism_transfer")
    retention_rows = []
    selected_retention = pd.concat(
        [
            retention_e2r[retention_e2r["direction"].eq("enzyme_to_reaction")],
            retention_r2e[retention_r2e["direction"].eq("reaction_to_enzyme")],
        ],
        ignore_index=True,
    )
    for row in selected_retention.itertuples(index=False):
        retention_rows.append(
            [
                row.direction,
                row.evaluation_level,
                f"{100 * row.delta_hit_at_3:+.1f} pp",
                f"{100 * row.delta_hit_at_10:+.1f} pp",
                f"{100 * row.delta_hit_at_20:+.1f} pp",
            ]
        )

    lines = [
        "# Terpene Synthase Retrieval — Iteration Report",
        "",
        "## Current decision",
        "",
        "The active production system uses direction- and objective-specific three-seed open-world ensembles. External E2R Top-10 is a locked reciprocal-rank fusion of two independently trained neural routes: a freeze-reaction-tower model with five-neighbor transfer and a hard-negative K=128 model with three-neighbor transfer. External E2R Top-20 now uses a separately confirmed reciprocal-rank fusion of the freeze-reaction route and a nonparametric dual-kernel collaborative-support route. The Top-20 auxiliary source combines reaction similarity, the training association graph and protein sequence similarity; its locked parameters are reaction-k 50, protein-k 5, temperature 0.03, degree power 1, primary weight 0.70, auxiliary weight 0.30 and RRF constant 60. It improved the independent confirmation split from 34.77% to 43.37% Hit@20 with a paired bootstrap 95% interval of +5.02 to +12.54 percentage points. R2E Top-10/20 uses the packaged Horizyn exact-residual reaction route, while R2E Top-3 remains the reaction-loss-0.75 shortlist model. External zero-shot queries expose seed disagreement, nearest-library novelty and bootstrap-gated empirical reliability. A 5,672-sequence UniProt TPS layer remains controlled rescue only and is never free-merged into canonical ranking.",
        "",
        "## Data and deployment",
        "",
        f"- Current proteins: {production_r2e['n_current_proteins']}",
        f"- Registered MARTS proteins: {production_r2e['n_external_proteins']}",
        f"- Canonical protein candidate space: {deployment_r2e.get('n_total_active_proteins', deployment_r2e.get('n_total_proteins'))}",
        f"- Controlled UniProt TPS rescue layer: {uniprot_report['candidate_expansion']['primary_named_embedding_candidates']}",
        f"- Architecture-contract-supported registered reactions: {uniprot_report['architecture_contracts']['rescue_supported_reactions']}",
        f"- Architecture-contract-unsupported registered reactions: {uniprot_report['architecture_contracts']['unsupported_or_unresolved_reactions']}",
        f"- Current reactions: {production_r2e['n_current_reactions']}",
        f"- Registered MARTS reactions: {production_r2e['n_external_reactions']}",
        f"- Active reaction candidate space: {deployment_r2e['n_total_reactions']}",
        f"- Rehearsal associations: {production_r2e['n_training_pairs']}",
        f"- Objective/direction-specific production checkpoints: R2E shared {deployment_r2e['n_models']} + R2E Top-3 {deployment_r2e_top3_10['n_models']} + R2E Top-10/20 exact-residual {deployment_r2e_exact['n_models']} + E2R primary {deployment_e2r['n_models']} + E2R hard-negative secondary {deployment_e2r_hardneg['n_models']}",
        f"- Persistent user registry: {len(registry_entries)} proteins and {len(registry_reactions)} reactions",
        f"- Wet-lab execution: {combined_campaign['n_plates']} plates, {combined_campaign['n_wells']} wells and {combined_campaign['sequence_deduplicated_constructs']} sequence-deduplicated master constructs",
        "",
        "## Strict external double-cold results",
        "",
        "Every external-enzyme/external-reaction positive pair is evaluated exactly once across the 5 × 5 protein-cluster/reaction-cluster Cartesian split.",
        "",
        "| Direction | Model | Hit@3 | Hit@10 | Hit@20 | MRR | Median best rank |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    lines.extend("| " + " | ".join(map(str, row)) + " |" for row in external_rows)
    lines.extend(
        [
            "",
            "PU masking improves both directions without expanding the model: unlabeled candidates in the same 50% identity protein cluster or reaction cluster as a positive are removed only from the contrastive denominator.",
            "",
            "## Routing selected after adaptation",
            "",
            "| Direction | Objective | Selected route | Hit |",
            "|---|---|---|---:|",
        ]
    )
    lines.extend("| " + " | ".join(map(str, row)) + " |" for row in route_rows)
    lines.extend(
        [
            "",
            "External E2R Top-10 RRF confirmation:",
            "",
            "| Split assignment | Query-cells | RRF Hit@10 | Previous production Hit@10 | Delta | Cell-bootstrap 95% CI |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    lines.extend("| " + " | ".join(map(str, row)) + " |" for row in rrf_confirmation_rows)
    lines.extend(
        [
            "",
            "External E2R Top-20 dual-kernel RRF confirmation:",
            "",
            "| Split role | Query-cells | Fused Hit@20 | Previous production Hit@20 | Delta | Diagnostic / cell-bootstrap 95% CI |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    lines.extend("| " + " | ".join(map(str, row)) + " |" for row in dual_kernel_confirmation_rows)
    lines.extend(
        [
            "",
            "The identifier `20260726` is the locked fold-seed value, not an execution date. The Top-20 parameters were selected before this split was generated and were not retuned on it.",
            "",
            "Production `auto` routing uses the reaction-loss-0.75 direct model only for external R2E Top-3, the packaged Horizyn exact-residual model for external R2E Top-10/20, and the shared PU model for current-library reactions. External E2R Top-3 uses the freeze-reaction route with direct weight 0.75. External E2R Top-10 uses locked RRF between the freeze-reaction route (five neighbors, direct weight 0.5) and the hard-negative route (three neighbors, direct weight 0.9). External E2R Top-20 uses locked RRF between the freeze-reaction route (five neighbors, direct weight 0.75) and dual-kernel collaborative support (reaction-k 50, protein-k 5, temperature 0.03, degree power 1) with weights 0.70/0.30 and constant 60. Current-library enzymes remain direct; few-shot, masked-known-association and manual overrides bypass or invalidate external reliability annotation as appropriate.",
            "",
            "## External-query reliability",
            "",
            "Reliability is learned only from query-grouped predictions on the strict 25-cell external double-cold benchmark. The value is an empirical ranking-reliability score, not a biochemical activity probability. Deployment requires the bootstrap 95% ROC-AUC lower bound to exceed 0.5.",
            "",
            "| Direction | Objective | Status | CV ROC-AUC | Bootstrap 95% CI | Overall hit | Highest-score quartile hit |",
            "|---|---|---|---:|---:|---:|---:|",
        ]
    )
    lines.extend("| " + " | ".join(map(str, row)) + " |" for row in reliability_rows)
    lines.extend(
        [
            "",
            "E2R Top-3 uses nearest-train protein similarity alone; E2R Top-10 RRF and Top-20 use novelty plus ensemble agreement. R2E Top-10/20 exact-residual routes use ensemble agreement and both pass the bootstrap deployment threshold; R2E Top-3 remains uncalibrated. The reliability value is a ranking-evidence score rather than biochemical activity probability. The CLI supports annotation-only, require-calibrated, require-intermediate and require-higher policies.",
            "",
            "## Controlled UniProt candidate expansion",
            "",
            "The UniProt layer was built from five TPS-related Pfam domains, exact-sequence deduplicated, filtered against the current/MARTS universe and compressed to 50% identity representatives. Of 6,494 representatives, 5,672 named A–D evidence-tier sequences were embedded; 822 domain-only sequences remain inactive.",
            "",
            "Free merging is rejected under strict double-cold stress. The added sequences are unlabelled in the benchmark, so this test measures preservation of known external positives rather than UniProt activity yield.",
            "",
            "| Budget | Canonical hits | Free-merge hits | Original hits retained | Controlled slots: canonical + UniProt | Controlled retention |",
            "|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for budget in [3, 10, 20]:
        free_row = uniprot_free[uniprot_free["budget"].eq(budget)].iloc[0]
        quota_row = uniprot_quota[uniprot_quota["budget"].eq(budget)].iloc[0]
        lines.append(
            f"| {budget} | {int(free_row['canonical_hits'])} | {int(free_row['expanded_hits'])} | {pct(free_row['hit_retention_fraction'])} | {int(quota_row['canonical_slots'])} + {int(quota_row['rescue_slots'])} | {pct(quota_row['hit_retention_fraction'])} |"
        )
    lines.extend(
        [
            "",
            "A/B-only expansion preserves every Top-20 cutoff hit but still worsens ranks and MRR for most queries. Adding C-tier homologs causes the major collapse. Candidate mean centering, z-scoring and local-density hub correction were tested with training-reaction statistics and rejected. Known-positive accession, exact-sequence and high-coverage MMseqs evidence define a reaction-specific Pfam architecture contract: 208 registered reactions support the five-Pfam rescue layer, while 32 remain canonical-only. Complete OSCs require PF13243+PF13249; single-domain OSC and plant-TPS fragments are excluded. The deployed policy is canonical-only Top-3, nine canonical plus one UniProt at Top-10, and eighteen canonical plus two UniProt at Top-20 only for contract-supported reactions. The full decision record is `results/terpene_uniprot_expansion_report.md`.",
            "",
            "## Wet-lab execution design",
            "",
            f"The four canonical and two UniProt rescue plates are procured through a shared exact-sequence-deduplicated master manifest containing {combined_campaign['sequence_deduplicated_constructs']} constructs ({combined_campaign['total_amino_acids']:,} aa). Canonical and UniProt rescue results remain separate QC scopes.",
            "",
            f"Complete reaction blocks are first assigned to plates by exact-capacity MILP. Canonical balancing moves {plate_balance['campaigns']['canonical_discovery']['reactions_changed_plate']} reactions, reduces summed terpene-type imbalance from {plate_balance_compact['canonical_discovery']['before']['type_range_sum']} to {plate_balance_compact['canonical_discovery']['after']['type_range_sum']}, eliminates TPS-class imbalance ({plate_balance_compact['canonical_discovery']['before']['class_range_sum']} to {plate_balance_compact['canonical_discovery']['after']['class_range_sum']}), and reduces the mean candidate-length range from {plate_balance_compact['canonical_discovery']['candidate_median_length_mean']['before_range']:.1f} to {plate_balance_compact['canonical_discovery']['candidate_median_length_mean']['after_range']:.1f} aa. Rescue balancing moves {plate_balance['campaigns']['uniprot_rescue']['reactions_changed_plate']} reactions, reduces type/class imbalance from {plate_balance_compact['uniprot_rescue']['before']['type_range_sum']}/{plate_balance_compact['uniprot_rescue']['before']['class_range_sum']} to {plate_balance_compact['uniprot_rescue']['after']['type_range_sum']}/{plate_balance_compact['uniprot_rescue']['after']['class_range_sum']}, equalizes B/C/D evidence counts, and removes exact Pfam architecture imbalance: bacterial class-I range {plate_balance_compact['uniprot_rescue']['bacterial_classI_total']['before_range']:.0f}→{plate_balance_compact['uniprot_rescue']['bacterial_classI_total']['after_range']:.0f}, plant-TPS-full {plate_balance_compact['uniprot_rescue']['plant_tps_full_total']['before_range']:.0f}→{plate_balance_compact['uniprot_rescue']['plant_tps_full_total']['after_range']:.0f}, complete OSC {plate_balance_compact['uniprot_rescue']['osc_full_total']['before_range']:.0f}→{plate_balance_compact['uniprot_rescue']['osc_full_total']['after_range']:.0f}.",
            "",
            f"Within the balanced reaction blocks, the original role-ordered layouts were rejected because selection roles were perfectly or strongly coupled to rows and local columns. Deterministic Hungarian assignment with seed {randomization['seed']} raises mean normalized role-slot entropy from {randomization['mean_normalized_entropy_before']:.3f} to {randomization['mean_normalized_entropy_after']:.3f}, reduces the maximum single-slot role share from {pct(randomization['maximum_slot_share_before'])} to {pct(randomization['maximum_slot_share_after'])}, and reduces the maximum role slot-count range from {randomization['maximum_role_slot_count_range_before']} to {randomization['maximum_role_slot_count_range_after']}. All controls and blanks remain fixed.",
            "",
            "The randomized manifests and matching result templates are the operational inputs. The original layouts remain only for provenance.",
            "",
            "## Rejected ablations",
            "",
            f"- Aggregated MARTS mechanism-step transfer: Hit@10 {pct(mechanism_transfer['hit_probability_at_10'])} versus {pct(mechanism_direct['hit_probability_at_10'])} for adapted direct on mechanism-covered queries. It is not deployed.",
            "- Multiview reaction fingerprints underperformed DRFP after MARTS adaptation and are not the production reaction tower.",
            "- CAGE sigmoid probabilities were saturated; raw logits and rank diagnostics are retained, but CAGE remains an optional structural evidence channel rather than the main ranker.",
            "- Embedding-anchor weights 0.01, 0.05 and 0.1 all reduced strict external Hit@10. Anchor weight remains zero.",
            "- Freezing the protein tower and adapting only the reaction tower reduced Top-10. The reverse configuration was retained only for E2R.",
            "- Free merging all 5,672 UniProt candidates lost 42–50% of canonical cutoff hits. C/D evidence tiers are rescue-only.",
            "- Candidate mean centering, z-scoring and local-density correction did not repair full candidate-universe expansion.",
            "- Carbon-count or coarse domain-family compatibility was too broad: it admitted PF13243/PF13249 fragments and reactions whose reference enzymes belong to PF00348 or PF00494. Production now uses known-positive architecture contracts.",
            "- The original reaction-to-plate allocation was rejected because TPS class, terpene type and sequence length were unevenly concentrated across plates. Execution uses exact-capacity MILP block balancing.",
            "- Role-ordered plate placement was rejected because candidate-selection strategies were confounded with row and column positions. Execution uses deterministic balanced randomization.",
            "",
            "## Current-database retention sanity check",
            "",
            "This is a training-retention check, not an unbiased cold-start estimate.",
            "",
            "| Direction | Evaluation | ΔHit@3 | ΔHit@10 | ΔHit@20 |",
            "|---|---|---:|---:|---:|",
        ]
    )
    lines.extend("| " + " | ".join(map(str, row)) + " |" for row in retention_rows)
    lines.extend(
        [
            "",
            "The adapted model slightly reduces current-library Top-1/3 memorization while preserving Top-10/20, which is accepted because strict external generalization improves substantially.",
            "",
            "## Persistent extension workflow",
            "",
            "```bash",
            ".venv/bin/python projects/active/terpene_screening/manage_open_world_registry.py init --force",
            "",
            ".venv/bin/python projects/active/terpene_screening/manage_open_world_registry.py add-enzymes \\",
            "  --enzyme-id NEW_ENZYME --sequence 'MSEQUENCE...'",
            "",
            ".venv/bin/python projects/active/terpene_screening/manage_open_world_registry.py add-reactions \\",
            "  --reaction-id NEW_REACTION --reaction-smiles 'SUBSTRATE>>PRODUCT'",
            "```",
            "",
            "After registration, use `rank_open_world.py` with the new ID directly. A persistent duplicate-entity integration test placed a newly registered enzyme at rank 3 and a newly registered reaction at rank 5, then removed both and restored the registry baseline.",
            "",
            "## Validation",
            "",
            f"Deployment status: R2E shared `{deployment_r2e['status']}`, R2E Top-3 `{deployment_r2e_top3_10['status']}`, R2E exact-residual `{deployment_r2e_exact['status']}`, E2R primary `{deployment_e2r['status']}`, E2R hard-negative secondary `{deployment_e2r_hardneg['status']}`. Models: {deployment_r2e['n_models']} + {deployment_r2e_top3_10['n_models']} + {deployment_r2e_exact['n_models']} + {deployment_e2r['n_models']} + {deployment_e2r_hardneg['n_models']}; protein input: {deployment_r2e['model_configs'][0]['protein_input_dim']}; base reaction input: {deployment_r2e['model_configs'][0]['reaction_input_dim']}.",
        ]
    )
    OUTPUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    summary = {
        "production": {"r2e_shared": production_r2e, "r2e_top3": production_r2e_top3_10, "r2e_exact": production_r2e_exact, "e2r_primary": production_e2r, "e2r_hardnegative": production_e2r_hardneg, "e2r_top20_dual_kernel": production_e2r_dual_kernel},
        "deployment": {"r2e_shared": deployment_r2e, "r2e_top3": deployment_r2e_top3_10, "r2e_exact": deployment_r2e_exact, "e2r_primary": deployment_e2r, "e2r_hardnegative": deployment_e2r_hardneg},
        "e2r_top10_rrf_confirmation": rrf_confirmation_rows,
        "e2r_top20_dual_kernel_confirmation": {
            "development_and_original_frozen": dual_kernel_route,
            "independent_locked_seed": dual_kernel_confirmatory,
            "report_rows": dual_kernel_confirmation_rows,
        },
        "external_double_cold": external_rows,
        "selected_routes": route_rows,
        "external_reliability": reliability_rows,
        "uniprot_expansion": uniprot_report,
        "wetlab_plate_balance": plate_balance,
        "wetlab_plate_balance_compact": plate_balance_compact,
        "wetlab_randomization": randomization,
        "combined_wetlab_campaign": combined_campaign,
        "mechanism_direct_hit10": float(mechanism_direct["hit_probability_at_10"]),
        "mechanism_transfer_hit10": float(mechanism_transfer["hit_probability_at_10"]),
        "report": str(OUTPUT),
    }
    OUTPUT_JSON.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(OUTPUT)
    print(OUTPUT_JSON)


if __name__ == "__main__":
    main()
