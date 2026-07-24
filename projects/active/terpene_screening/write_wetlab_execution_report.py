from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
OUTPUT = ROOT / "results/terpene_wetlab_execution_report.md"
OUTPUT_JSON = ROOT / "results/terpene_wetlab_execution_summary.json"


def pct(value: float) -> str:
    return f"{100 * float(value):.1f}%"


def main() -> None:
    batch_root = ROOT / "results/terpene_registry_batch"
    panel_root = ROOT / "results/terpene_wetlab_discovery_panels"
    plate_root = ROOT / "results/terpene_wetlab_plate_manifest"
    rescue_root = ROOT / "results/terpene_uniprot_rescue_campaign"
    combined_root = ROOT / "results/terpene_combined_wetlab_campaign"
    randomized_root = ROOT / "results/terpene_wetlab_randomized_layout"
    plate_balance_root = ROOT / "results/terpene_wetlab_plate_balanced"
    batch_summary = json.loads((batch_root / "summary.json").read_text(encoding="utf-8"))
    discovery_audit = json.loads((batch_root / "discovery_audit.json").read_text(encoding="utf-8"))
    concentration = pd.read_csv(batch_root / "discovery_concentration_summary.csv")
    panel_summary_json = json.loads((panel_root / "summary.json").read_text(encoding="utf-8"))
    plate_summary_json = json.loads((plate_root / "summary.json").read_text(encoding="utf-8"))
    campaign = pd.read_csv(panel_root / "campaign_reactions.csv", dtype=str).fillna("")
    campaign["balanced_campaign_order"] = pd.to_numeric(
        campaign["balanced_campaign_order"], errors="raise"
    ).astype(int)
    extended = pd.read_csv(panel_root / "extended_pathway_reactions.csv", dtype=str).fillna("")
    panel_summary = pd.read_csv(panel_root / "reaction_panel_summary.csv")
    plate_summary = pd.read_csv(plate_root / "plate_summary.csv")
    constructs = pd.read_csv(plate_root / "sequence_deduplicated_constructs.csv", dtype=str).fillna("")
    rescue_summary = json.loads((rescue_root / "summary.json").read_text(encoding="utf-8"))
    combined_summary = json.loads((combined_root / "summary.json").read_text(encoding="utf-8"))
    combined_plates = pd.read_csv(combined_root / "master_plate_summary.csv")
    procurement = pd.read_csv(combined_root / "procurement_summary.csv")
    randomization_summary = json.loads(
        (randomized_root / "summary.json").read_text(encoding="utf-8")
    )
    randomization_audit = pd.read_csv(
        randomized_root / "role_slot_balance_audit.csv"
    )
    plate_balance_summary = json.loads(
        (plate_balance_root / "summary.json").read_text(encoding="utf-8")
    )
    plate_balance_compact = json.loads(
        (plate_balance_root / "compact_balance_summary.json").read_text(
            encoding="utf-8"
        )
    )
    plate_balance_audit = pd.read_csv(
        plate_balance_root / "plate_balance_audit.csv"
    )

    type_counts = campaign["terpene_type"].replace("", "missing").value_counts().to_dict()
    class_counts = campaign["tps_class"].replace("", "missing").value_counts().to_dict()
    extended_counts = extended["terpene_type"].replace("", "missing").value_counts().to_dict()
    concentration_rows = []
    for row in concentration.itertuples(index=False):
        concentration_rows.append(
            [
                row.direction,
                row.objective,
                int(row.n_queries),
                int(row.unique_top1),
                pct(row.top1_top_candidate_share),
                pct(row.top10_candidates_share),
                f"{float(row.effective_top1_candidates):.1f}",
                pct(row.top1_external_share),
            ]
        )

    lines = [
        "# Terpene Synthase Wet-Lab Discovery Execution Report",
        "",
        "## Scope and data contract",
        "",
        "This report covers the persistent MARTS registry discovery run, candidate-panel construction, balanced core-TPS campaign, four canonical 96-well plates, two architecture-contract-filtered UniProt rescue plates, combined sequence procurement and the result-feedback workflow. Known MARTS enzyme-reaction associations are masked from every discovery ranking and appear only as explicitly labelled positive controls.",
        "",
        f"- Registered enzyme queries: {batch_summary['enzyme_to_reaction']['n_unique_queries']}",
        f"- Registered reaction queries: {batch_summary['reaction_to_enzyme']['n_unique_queries']}",
        f"- Known-association leakage found: {discovery_audit['known_association_leaks']}",
        "- Reliability after known-positive masking: not reused; masked discovery is explicitly marked uncalibrated.",
        "",
        "## Candidate concentration audit",
        "",
        "| Direction | Objective | Queries | Unique Top-1 | Largest single-candidate share | Top-10 candidates share | Effective Top-1 candidates | External Top-1 share |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    lines.extend("| " + " | ".join(map(str, row)) + " |" for row in concentration_rows)
    lines.extend(
        [
            "",
            "The largest single candidate covers only approximately 4.5–5.0% of queries, so the discovery output is not dominated by one universal hub. Objective-specific Top-3/10/20 lists are not forced to be nested because the production routes differ by cutoff.",
            "",
            "## Full reaction panels",
            "",
            f"- Reactions with panels: {panel_summary_json['n_reactions']}",
            f"- Discovery candidates per reaction: {panel_summary_json['discovery_candidates_per_reaction']}",
            f"- Total discovery assays represented: {panel_summary_json['n_discovery_rows']}",
            f"- Positive controls available: {panel_summary_json['n_positive_controls']}",
            "- Allocation: 6 exploitation, 3 uncertainty, 3 ESM-C diversity candidates.",
            f"- Sequence-risk candidates removed from Top-20 pools: {panel_summary_json['sequence_risk_filter']['total_risky_candidates_excluded_from_pools']}",
            "- Eligibility rule: 200–1000 aa and canonical amino-acid alphabet; excluded candidates are replaced from the remaining masked Top-20 pool.",
            "",
            "## Balanced core-TPS campaign",
            "",
            f"- Core reactions: {len(campaign)}",
            f"- Terpene-type distribution: {json.dumps(type_counts, ensure_ascii=False)}",
            f"- TPS-class distribution: {json.dumps(class_counts, ensure_ascii=False)}",
            f"- Unique positive-control IDs: {campaign['positive_control_id'].nunique()} of {len(campaign)} reactions",
            f"- Extended pathway slate: {len(extended)} reactions, {json.dumps(extended_counts, ensure_ascii=False)}",
            "",
            "The primary campaign excludes PSY/SQS/PT/tetraterpene pathway enzymes from the 24-reaction core slate and preserves them in a separate eight-reaction exploratory slate. The core selector guarantees type coverage, preserves at least four class-II reactions and penalizes repeated substrate and positive-control choices.",
            "",
            "| Order | Reaction | Type | Class | Substrate | Product | Positive control |",
            "|---:|---|---|---|---|---|---|",
        ]
    )
    for row in campaign.sort_values("balanced_campaign_order").itertuples(index=False):
        lines.append(
            f"| {row.balanced_campaign_order} | {row.reaction_id} | {row.terpene_type} | {row.tps_class} | {row.substrate_name} | {row.product_name} | {row.positive_control_id} |"
        )
    lines.extend(
        [
            "",
            "## Construct procurement",
            "",
            f"- Candidate-ID constructs: {plate_summary_json['n_candidate_id_constructs']}",
            f"- Sequence-deduplicated constructs: {plate_summary_json['n_sequence_deduplicated_constructs']}",
            f"- Alias IDs collapsed: {plate_summary_json['redundant_candidate_ids_collapsed']}",
            f"- Total protein length: {plate_summary_json['total_amino_acids_sequence_deduplicated']:,} aa",
            f"- Total coding length without stops: {plate_summary_json['total_coding_nucleotides_without_stop']:,} nt",
            f"- Sequence-ready constructs: {plate_summary_json['n_sequence_ready']}",
            f"- Constructs requiring manual sequence review: {plate_summary_json['n_constructs_needing_manual_review']}",
            "- Protein FASTA is complete. No codon optimization has been performed because the expression host and vector architecture are not fixed.",
            "",
            "## Four-plate layout",
            "",
            "Each reaction occupies two adjacent columns. Candidates 1–8 occupy A–H of the first column; candidates 9–12 occupy A–D of the second. The remaining wells are positive control primary, positive control replicate, empty-vector negative and substrate/process blank.",
            "",
            "| Plate | Reactions | Discovery wells | Positive-control wells | Empty-vector wells | Process blanks | Unique protein constructs |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in plate_summary.itertuples(index=False):
        lines.append(
            f"| {row.plate_id} | {row.n_reactions} | {row.n_discovery_assays} | {row.n_positive_control_wells} | {row.n_negative_control_wells} | {row.n_process_blank_wells} | {row.unique_protein_constructs} |"
        )
    lines.extend(
        [
            "",
            "## Reaction-to-plate balancing",
            "",
            "Before assigning candidate wells, complete reaction blocks are redistributed across plates with an exact-capacity mixed-integer linear program. Canonical plates retain six reactions each and UniProt rescue plates retain twelve reactions each. The objective balances terpene type, TPS class, substrate, positive-control reuse, candidate sequence length, candidate-source fraction and, for the rescue campaign, evidence-tier and architecture counts.",
            "",
            f"Canonical balancing moves {plate_balance_summary['campaigns']['canonical_discovery']['reactions_changed_plate']} of 24 reaction blocks. The summed per-category terpene-type range falls from {plate_balance_compact['canonical_discovery']['before']['type_range_sum']} to {plate_balance_compact['canonical_discovery']['after']['type_range_sum']}; the TPS-class range falls from {plate_balance_compact['canonical_discovery']['before']['class_range_sum']} to {plate_balance_compact['canonical_discovery']['after']['class_range_sum']}. The between-plate range of mean candidate median length falls from {plate_balance_compact['canonical_discovery']['candidate_median_length_mean']['before_range']:.1f} aa to {plate_balance_compact['canonical_discovery']['candidate_median_length_mean']['after_range']:.1f} aa.",
            "",
            f"UniProt rescue balancing moves {plate_balance_summary['campaigns']['uniprot_rescue']['reactions_changed_plate']} of 24 reaction blocks. The terpene-type range falls from {plate_balance_compact['uniprot_rescue']['before']['type_range_sum']} to {plate_balance_compact['uniprot_rescue']['after']['type_range_sum']}, the TPS-class range from {plate_balance_compact['uniprot_rescue']['before']['class_range_sum']} to {plate_balance_compact['uniprot_rescue']['after']['class_range_sum']}, and the between-plate mean candidate-length range from {plate_balance_compact['uniprot_rescue']['candidate_median_length_mean']['before_range']:.1f} aa to {plate_balance_compact['uniprot_rescue']['candidate_median_length_mean']['after_range']:.1f} aa. B/C/D evidence counts are equalized between the two rescue plates. Exact Pfam architecture imbalance is also removed: bacterial class-I range {plate_balance_compact['uniprot_rescue']['bacterial_classI_total']['before_range']:.0f}→{plate_balance_compact['uniprot_rescue']['bacterial_classI_total']['after_range']:.0f}, plant-TPS-full {plate_balance_compact['uniprot_rescue']['plant_tps_full_total']['before_range']:.0f}→{plate_balance_compact['uniprot_rescue']['plant_tps_full_total']['after_range']:.0f}, and complete OSC {plate_balance_compact['uniprot_rescue']['osc_full_total']['before_range']:.0f}→{plate_balance_compact['uniprot_rescue']['osc_full_total']['after_range']:.0f}.",
            "",
            "Every reaction retains exactly the same candidate and control set; only its plate/block assignment changes. Each resulting plate remains exactly 96 wells.",
            "",
            "## Candidate-position randomization",
            "",
            "The original generated layouts placed candidate-selection roles in fixed rows and local columns: canonical exploitation, uncertainty and diversity roles occupied disjoint slot sets, while UniProt evidence, homology, predicted and diversity candidates were fixed to A/B/C/D rows. This would confound selection strategy with plate-position effects.",
            "",
            f"The operational layouts therefore use deterministic within-reaction-block Hungarian assignment with seed `{randomization_summary['seed']}`. Positive controls, empty-vector negatives and substrate/process blanks remain in their original wells; only candidate wells are reassigned.",
            "",
            "| Balance diagnostic | Before | After |",
            "|---|---:|---:|",
            f"| Mean normalized role-slot entropy | {randomization_summary['mean_normalized_entropy_before']:.3f} | {randomization_summary['mean_normalized_entropy_after']:.3f} |",
            f"| Maximum single-slot share for any role | {pct(randomization_summary['maximum_slot_share_before'])} | {pct(randomization_summary['maximum_slot_share_after'])} |",
            f"| Maximum role slot-count range | {randomization_summary['maximum_role_slot_count_range_before']} | {randomization_summary['maximum_role_slot_count_range_after']} |",
            f"| Candidate assignments | — | {randomization_summary['candidate_assignments']} |",
            f"| Control or blank wells moved | — | {randomization_summary['control_and_blank_wells_moved']} |",
            "",
            "The mean role-slot entropy increases from approximately 0.20 to 0.97, and the maximum count difference for a role across candidate slots falls from 24 to 1. The randomized manifests and matching result templates are the execution inputs; the original role-ordered layouts are retained only for provenance and audit.",
            "",
            "## Combined six-plate procurement campaign",
            "",
            f"The four canonical plates and two UniProt rescue plates contain {combined_summary['n_wells']} wells across {combined_summary['n_reactions']} distinct reactions. They contain {combined_summary['protein_assay_wells']} protein assay wells and {combined_summary['candidate_id_constructs']} candidate IDs. Cross-campaign exact-sequence deduplication reduces these to {combined_summary['sequence_deduplicated_constructs']} master constructs, including {combined_summary['constructs_shared_between_campaigns']} sequences shared by both campaigns.",
            "",
            f"Total procurement size is {combined_summary['total_amino_acids']:,} aa or {combined_summary['total_coding_nucleotides_without_stop']:,} coding nucleotides without stop codons. The master FASTA contains protein sequences only; codon optimization remains deferred until the host and vector architecture are fixed.",
            "",
            "| Master order | Scope | Plate | Reactions | Wells | Discovery wells | Positive controls | Negative controls | Process blanks | Unique constructs |",
            "|---:|---|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in combined_plates.itertuples(index=False):
        lines.append(
            f"| {row.master_plate_order} | {row.campaign_scope} | {row.plate_id} | {row.n_reactions} | {row.n_wells} | {row.discovery_wells} | {row.positive_control_wells} | {row.negative_control_wells} | {row.process_blank_wells} | {row.unique_constructs} |"
        )
    lines.extend(
        [
            "",
            "Procurement length tiers:",
            "",
            "| Length tier | Usage | Constructs | Total aa | Total coding nt | Median length | Maximum length |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in procurement.itertuples(index=False):
        lines.append(
            f"| {row.procurement_length_tier} | {row.construct_usage_class} | {row.unique_constructs} | {row.total_amino_acids} | {row.total_coding_nt} | {row.median_length:.0f} | {row.max_length} |"
        )
    lines.extend(
        [
            "",
            "The master manifest is for procurement and plate tracking only. Canonical and UniProt rescue feedback remain separate QC scopes, so a failed control or contamination event in one experimental batch cannot relabel assays from the other batch.",
            "",
            "## Result feedback and iteration",
            "",
            "The assay template contains expression status, soluble-expression status, assay/background signals, target-product detection, product-identity confidence, technical-issue flag and notes.",
            "",
            "- Confirmed positive: reaction controls pass; target product is detected; identity confidence is at least the configured threshold; no technical issue.",
            "- Expression-qualified negative: controls pass; target is not detected; expression is adequate/high, or low but soluble; no technical issue.",
            "- Inconclusive: failed expression, failed controls, missing evidence or technical issue.",
            "- Untested or unlabeled pairs are never converted into negatives.",
            "- Failed-control reactions are routed to control/current-panel rerun rather than candidate expansion.",
            "- Passed reactions receive an eight-candidate next panel: 4 outcome/model exploitation, 2 uncertainty and 2 diversity candidates.",
            "",
            "## Canonical files",
            "",
            "- `results/terpene_registry_batch/reaction_to_enzyme_rankings.csv`",
            "- `results/terpene_wetlab_discovery_panels/campaign_reactions.csv`",
            "- `results/terpene_wetlab_discovery_panels/campaign_discovery_candidates.csv`",
            "- `results/terpene_wetlab_plate_manifest/assay_manifest.csv` (pre-randomization provenance)",
            "- `results/terpene_wetlab_plate_balanced/canonical_balanced_assay_manifest.csv`",
            "- `results/terpene_wetlab_plate_balanced/uniprot_balanced_assay_manifest.csv`",
            "- `results/terpene_wetlab_plate_balanced/plate_balance_audit.csv`",
            "- `results/terpene_wetlab_randomized_layout/canonical_randomized_assay_manifest.csv`",
            "- `results/terpene_wetlab_randomized_layout/canonical_randomized_assay_results_template.csv`",
            "- `results/terpene_wetlab_plate_manifest/sequence_deduplicated_constructs.fasta`",
            "- `results/terpene_wetlab_plate_manifest/TPS_DISCOVERY_P01_layout.csv` through `P04`",
            "- `results/terpene_uniprot_rescue_campaign/assay_manifest.csv` (pre-randomization provenance)",
            "- `results/terpene_wetlab_randomized_layout/uniprot_randomized_assay_manifest.csv`",
            "- `results/terpene_wetlab_randomized_layout/uniprot_randomized_assay_results_template.csv`",
            "- `results/terpene_wetlab_randomized_layout/candidate_well_assignments.csv`",
            "- `results/terpene_wetlab_randomized_layout/role_slot_balance_audit.csv`",
            "- `results/terpene_combined_wetlab_campaign/master_assay_manifest.csv`",
            "- `results/terpene_combined_wetlab_campaign/master_sequence_constructs.fasta`",
            "- `results/terpene_combined_wetlab_campaign/feedback_scopes.csv`",
            "",
            "## Remaining experimental decisions",
            "",
            "Expression host, vector, tag, subcellular-targeting truncation policy, precursor-supply strategy, assay matrix, analytical detection limits and product-confirmation criteria remain wet-lab decisions. They are intentionally not inferred by the retrieval pipeline.",
        ]
    )
    OUTPUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    summary = {
        "batch": batch_summary,
        "discovery_audit": discovery_audit,
        "panels": panel_summary_json,
        "plates": plate_summary_json,
        "uniprot_rescue": rescue_summary,
        "combined_campaign": combined_summary,
        "reaction_plate_balance": plate_balance_summary,
        "reaction_plate_balance_compact": plate_balance_compact,
        "reaction_plate_balance_audit": plate_balance_audit.to_dict("records"),
        "candidate_position_randomization": randomization_summary,
        "candidate_position_balance_audit": randomization_audit.to_dict("records"),
        "core_type_counts": type_counts,
        "core_class_counts": class_counts,
        "extended_type_counts": extended_counts,
        "report": str(OUTPUT),
    }
    OUTPUT_JSON.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(OUTPUT)
    print(OUTPUT_JSON)


if __name__ == "__main__":
    main()
