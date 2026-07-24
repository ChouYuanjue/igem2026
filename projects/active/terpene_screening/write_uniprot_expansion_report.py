from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
OUTPUT = ROOT / "results/terpene_uniprot_expansion_report.md"
OUTPUT_JSON = ROOT / "results/terpene_uniprot_expansion_report_summary.json"


def pct(value: float) -> str:
    return f"{100 * float(value):.1f}%"


def main() -> None:
    expansion = json.loads(
        (ROOT / "data/terpene_uniprot_expansion/summary.json").read_text(encoding="utf-8")
    )
    embedding = json.loads(
        (
            ROOT
            / "data/terpene_embeddings/uniprot_tps_primary_esmc600m/summary.json"
        ).read_text(encoding="utf-8")
    )
    free_stress = json.loads(
        (ROOT / "results/terpene_uniprot_expanded_double_cold/summary.json").read_text(
            encoding="utf-8"
        )
    )
    tiered = pd.read_csv(
        ROOT / "results/terpene_uniprot_tiered_double_cold/paired_retention.csv"
    )
    tiered_supported = pd.read_csv(
        ROOT
        / "results/terpene_uniprot_tiered_double_cold/paired_retention_contract_supported_queries.csv"
    )
    hub = pd.read_csv(
        ROOT
        / "results/terpene_candidate_hub_normalization_double_cold/comparison_to_canonical_raw.csv"
    )
    controlled = json.loads(
        (
            ROOT / "results/terpene_uniprot_controlled_rescue_batch/summary.json"
        ).read_text(encoding="utf-8")
    )
    controlled_audit = json.loads(
        (ROOT / "results/terpene_uniprot_controlled_rescue_batch/audit.json").read_text(
            encoding="utf-8"
        )
    )
    rescue = json.loads(
        (ROOT / "results/terpene_uniprot_rescue_campaign/summary.json").read_text(
            encoding="utf-8"
        )
    )
    architecture_contracts = json.loads(
        (
            ROOT
            / "data/terpene_uniprot_expansion/reaction_architecture_contracts/summary.json"
        ).read_text(encoding="utf-8")
    )
    sequence_integrity = json.loads(
        (
            ROOT / "results/terpene_uniprot_rescue_sequence_integrity/summary.json"
        ).read_text(encoding="utf-8")
    )
    replacements = pd.read_csv(
        ROOT / "results/terpene_uniprot_rescue_campaign/replaced_unsupported_reactions.csv",
        dtype=str,
    ).fillna("")

    free_metrics = pd.DataFrame(free_stress["metrics"])
    free_paired = pd.DataFrame(free_stress["paired_hit_retention"])
    quota = pd.read_csv(
        ROOT / "results/terpene_uniprot_expanded_double_cold/rescue_slot_retention.csv"
    )
    selected_quota = quota[
        ((quota["budget"] == 3) & (quota["rescue_slots"] == 0))
        | ((quota["budget"] == 10) & (quota["rescue_slots"] == 1))
        | ((quota["budget"] == 20) & (quota["rescue_slots"] == 2))
    ].sort_values("budget")

    lines = [
        "# UniProt TPS Candidate Expansion — Evaluation and Deployment Decision",
        "",
        "## Decision",
        "",
        "The full UniProt TPS layer is not merged into the canonical production ranking. The active policy keeps the current+MARTS ranking as an unchanged prefix and exposes UniProt candidates only through validated tail slots: zero slots at Top-3, one at Top-10 and two at Top-20. A separate four-candidate-per-reaction wet-lab rescue campaign remains available for deliberate discovery experiments.",
        "",
        "## Candidate construction",
        "",
        "The source query uses the five TPS-related Pfam domains used by the MARTS curation workflow and excludes fragments and sequences outside 200–1000 aa.",
        "",
        "| Stage | Count |",
        "|---|---:|",
        f"| Raw UniProt rows | {expansion['raw_rows']:,} |",
        f"| Valid normalized rows | {expansion['normalized_valid_rows']:,} |",
        f"| Exact-sequence unique rows | {expansion['sequence_unique_rows']:,} |",
        f"| Novel after existing ID/sequence removal | {expansion['novel_sequence_unique_rows']:,} |",
        f"| 50% identity clusters | {expansion['novel_clusters']:,} |",
        f"| Named primary embedding candidates | {expansion['primary_named_embedding_candidates']:,} |",
        f"| Domain-only rescue candidates | {expansion['domain_only_rescue_candidates']:,} |",
        "",
        "Primary-layer evidence tiers:",
        "",
    ]
    for tier, count in expansion["evidence_quality_embedding_candidates"].items():
        if tier != "E_domain_only_uncharacterized":
            lines.append(f"- `{tier}`: {count:,}")
    lines.extend(
        [
            "",
            f"ESM-C embeddings: {embedding['collated_embeddings']:,}/{embedding['input_sequences']:,} completed, {embedding['failed']} failed and {len(embedding['missing_after_collation'])} missing. The extractor uses length-bucketed low-level ESM-C transformer batches and is numerically aligned with the original SDK path.",
            "",
            "## Free-merge strict double-cold stress test",
            "",
            "The 5,672 UniProt candidates are treated as unlabelled decoys because the strict MARTS benchmark contains no labels for them. This test therefore measures preservation of known external positives, not UniProt activity yield.",
            "",
            "| Budget | Canonical Hit | Free-expanded Hit | Original hits retained | Median positive rank: canonical → expanded |",
            "|---:|---:|---:|---:|---:|",
        ]
    )
    for budget in [3, 10, 20]:
        canonical_row = free_metrics[
            free_metrics["candidate_universe"].eq("canonical")
            & free_metrics["budget"].eq(budget)
        ].iloc[0]
        expanded_row = free_metrics[
            free_metrics["candidate_universe"].eq("expanded")
            & free_metrics["budget"].eq(budget)
        ].iloc[0]
        paired_row = free_paired[free_paired["budget"].eq(budget)].iloc[0]
        lines.append(
            "| "
            + " | ".join(
                [
                    str(budget),
                    pct(canonical_row["hit_probability"]),
                    pct(expanded_row["hit_probability"]),
                    pct(paired_row["hit_retention_fraction"]),
                    f"{canonical_row['median_best_positive_rank']:.0f} → {expanded_row['median_best_positive_rank']:.0f}",
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "Free merging loses 42–50% of the canonical cutoff hits and increases the median true-positive rank by several hundred positions. It is rejected.",
            "",
            "## Evidence-tier and architecture-contract ablation",
            "",
            "| Budget | Evidence layer | Unconstrained retention | Contract retention, all strict queries | Contract retention, supported queries | Supported-query MRR ratio | Supported median rank inflation |",
            "|---:|---|---:|---:|---:|---:|---:|",
        ]
    )
    layer_specs = [
        ("A/B only", "expanded_ab", "expanded_ab_contract"),
        ("A/B/C", "expanded_abc", "expanded_abc_contract"),
        ("A–D", "expanded_abcd", "expanded_abcd_contract"),
    ]
    for budget in [3, 10, 20]:
        for label, unconstrained_name, contract_name in layer_specs:
            unconstrained = tiered[
                tiered["budget"].eq(budget)
                & tiered["candidate_universe"].eq(unconstrained_name)
            ].iloc[0]
            contract_all = tiered[
                tiered["budget"].eq(budget)
                & tiered["candidate_universe"].eq(contract_name)
            ].iloc[0]
            contract_supported = tiered_supported[
                tiered_supported["budget"].eq(budget)
                & tiered_supported["candidate_universe"].eq(contract_name)
            ].iloc[0]
            lines.append(
                f"| {budget} | {label} | {pct(unconstrained['hit_retention_fraction'])} | {pct(contract_all['hit_retention_fraction'])} | {pct(contract_supported['hit_retention_fraction'])} | {contract_supported['mrr_ratio_to_canonical']:.3f} | {contract_supported['median_positive_rank_inflation']:.0f} |"
            )
    supported_ab20 = tiered_supported[
        tiered_supported["budget"].eq(20)
        & tiered_supported["candidate_universe"].eq("expanded_ab_contract")
    ].iloc[0]
    supported_ab10 = tiered_supported[
        tiered_supported["budget"].eq(10)
        & tiered_supported["candidate_universe"].eq("expanded_ab_contract")
    ].iloc[0]
    supported_abc20 = tiered_supported[
        tiered_supported["budget"].eq(20)
        & tiered_supported["candidate_universe"].eq("expanded_abc_contract")
    ].iloc[0]
    lines.extend(
        [
            "",
            f"On the {int(supported_ab20['n_queries'])} contract-supported strict queries, the architecture-constrained A/B layer retains all {int(supported_ab20['canonical_hits'])} Top-20 hits and {int(supported_ab10['retained_hits'])}/{int(supported_ab10['canonical_hits'])} Top-10 hits. It still reduces Top-20 MRR to {supported_ab20['mrr_ratio_to_canonical']:.3f} of canonical and moves the median true positive back by {supported_ab20['median_positive_rank_inflation']:.0f} ranks, so it is not allowed to freely reorder the prefix. Adding C-tier homologs is the main failure boundary: contract-constrained A/B/C retains only {int(supported_abc20['retained_hits'])}/{int(supported_abc20['canonical_hits'])} Top-20 hits on supported queries.",
            "",
            "## Rejected hub-normalization methods",
            "",
            "Candidate mean centering, candidate z-scoring and top-20 local-density correction were computed from training-reaction scores only. None improved the full A–D expansion enough for deployment.",
            "",
            "| Budget | Best full-expansion normalization | Hit retention | MRR ratio to canonical | Median rank change |",
            "|---:|---|---:|---:|---:|",
        ]
    )
    full_hub = hub[hub["candidate_universe"].eq("expanded_abcd")].copy()
    for budget in [3, 10, 20]:
        row = full_hub[full_hub["budget"].eq(budget)].sort_values(
            ["hit_retention_fraction", "mrr_ratio_to_baseline"], ascending=False
        ).iloc[0]
        lines.append(
            f"| {budget} | {row['score_normalization']} | {pct(row['hit_retention_fraction'])} | {row['mrr_ratio_to_baseline']:.3f} | {row['median_rank_change']:.0f} |"
        )
    lines.extend(
        [
            "",
            "## Reaction-specific architecture contracts",
            "",
            f"Known positive enzymes define the admissible Pfam architecture for each registered reaction. Accession matches are preferred, followed by exact sequence matches and then high-coverage MMseqs matches. The resulting contracts support {architecture_contracts['rescue_supported_reactions']} of 240 registered reactions; {architecture_contracts['unsupported_or_unresolved_reactions']} reactions belong to enzyme families outside the five-Pfam expansion or lack sufficiently reliable mapping.",
            "",
            "Complete OSCs require PF13243+PF13249. PF13249-only OSC fragments and single PF01397/PF03936 plant-TPS fragments are excluded. Unsupported reactions receive no UniProt tail slots and remain canonical-only.",
            "",
            "## Controlled rescue quota",
            "",
            "Canonical candidates occupy the ranking prefix and UniProt candidates occupy only reserved tail slots. The quota is applied only when the reaction architecture contract is supported.",
            "",
            "| Budget | Canonical slots | UniProt slots | Strict hit retention | Resulting strict Hit |",
            "|---:|---:|---:|---:|---:|",
        ]
    )
    for row in selected_quota.itertuples(index=False):
        lines.append(
            f"| {int(row.budget)} | {int(row.canonical_slots)} | {int(row.rescue_slots)} | {pct(row.hit_retention_fraction)} | {pct(row.hit_probability)} |"
        )
    lines.extend(
        [
            "",
            "Batch validation across all registered reactions:",
            "",
            f"- Query/objective combinations: {controlled['query_objectives']}",
            f"- Contract-supported reactions: {controlled['contract_supported_reactions']}",
            f"- Contract-unsupported reactions kept canonical-only: {controlled['contract_unsupported_reactions']}",
            f"- Actual UniProt tail rows: {controlled['uniprot_rescue_rows']}",
            f"- Unique selected UniProt candidates: {controlled['unique_uniprot_rescue_candidates']}",
            f"- Maximum selected candidate usage: {controlled['maximum_selected_candidate_query_appearances']} reactions",
            f"- Known-association leakage: {controlled_audit['known_association_leaks']}",
            f"- Canonical-prefix mismatches: {controlled_audit['canonical_prefix_mismatches']}",
            "",
            "## Wet-lab rescue campaign",
            "",
            f"The separate campaign contains {rescue['n_selected_candidates']} UniProt candidates across {rescue['n_reactions']} reactions, using {rescue['n_plates']} complete 96-well plates. It contains {rescue['n_unique_selected_candidates']} unique candidates, and no candidate is used for more than {rescue['maximum_selected_candidate_usage']} reactions.",
            "",
            f"Of the original 24 balanced targets, {rescue['supported_base_reactions_retained']} were retained and {rescue['unsupported_base_reactions_replaced']} were replaced by supported reactions of the same terpene type. Before role selection, {rescue['excluded_high_confidence_risk_candidates']} unique candidates were excluded for conservative complete-architecture length, composition, hydrophobicity or residue risks.",
            "",
            f"The final {sequence_integrity['selected_rescue_unique_sequences']} unique UniProt sequences contain {sequence_integrity['high_confidence_sequence_risk']} high-confidence sequence risks and {sequence_integrity['architecture_length_risk']} complete-architecture length risks. Motif absence remains annotation-only because the reviewed reference set does not support using exact motif regexes as a hard activity filter.",
            "",
            "Selection balances an evidence anchor, a named homology candidate, a named predicted candidate and an ESM-C diversity candidate. This campaign is a discovery experiment and is not interpreted as calibrated probability output.",
            "",
            "## Active artifacts",
            "",
            "- `results/terpene_uniprot_controlled_rescue_batch/controlled_rankings.csv`",
            "- `projects/active/terpene_screening/rank_uniprot_rescue.py`",
            "- `results/terpene_uniprot_rescue_campaign/assay_manifest.csv`",
            "- `results/terpene_uniprot_rescue_campaign/assay_results_template.csv`",
            "- `results/terpene_uniprot_rescue_campaign/sequence_deduplicated_constructs.fasta`",
            "",
            "## Final deployment rule",
            "",
            "1. Top-3 remains canonical-only.",
            "2. A contract-supported reaction may append one architecture-compatible UniProt candidate after nine canonical candidates at Top-10.",
            "3. A contract-supported reaction may append two architecture-compatible UniProt candidates after eighteen canonical candidates at Top-20.",
            "4. Contract-unsupported reactions remain canonical-only at every cutoff.",
            "5. A/B/C/D evidence, exact Pfam architecture, contract provenance and historical hub frequency are reported for every UniProt row.",
            "6. Reliability scores trained on the 2,085-candidate canonical universe are not reused after candidate expansion.",
            "7. PF13249-only/PF01397-only/PF03936-only fragments and the 822 domain-only sequences remain outside the active rescue layer.",
        ]
    )

    OUTPUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    summary = {
        "decision": "canonical_prefix_with_uniprot_tail_quota",
        "default_rescue_slots": {"top3": 0, "top10": 1, "top20": 2},
        "candidate_expansion": expansion,
        "embedding": embedding,
        "free_merge_paired_retention": free_paired.to_dict("records"),
        "selected_quota": selected_quota.to_dict("records"),
        "controlled_batch": controlled,
        "controlled_audit": controlled_audit,
        "architecture_contracts": architecture_contracts,
        "wetlab_rescue": rescue,
        "sequence_integrity": sequence_integrity,
        "replacements": replacements.to_dict("records"),
        "report": str(OUTPUT),
    }
    OUTPUT_JSON.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(OUTPUT)
    print(OUTPUT_JSON)


if __name__ == "__main__":
    main()
