from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from projects.active.terpene_screening.common import canonicalize_reaction_smiles  # noqa: E402
from projects.active.terpene_screening.fair_benchmark import (  # noqa: E402
    DEFAULT_BUDGETS,
    DEFAULT_TOP_PERCENTS,
    evaluate_ranking_frame,
    sha256_file,
)
from projects.active.terpene_screening.rank_open_world import (  # noqa: E402
    encode_model_reactions,
    load_feature_schema,
    load_models,
    load_protein_library,
    load_registered_reaction_feature_library,
)

DEFAULT_UNIVERSE = ROOT / "data/catalyst_candidate_universes/general_merged"
DEFAULT_BASE = ROOT / "results/terpene_production_models/marts_adapted_drfp_pu"


def _pick(frame: pd.DataFrame, names: tuple[str, ...], *, required: bool = True) -> str | None:
    for name in names:
        if name in frame.columns:
            return name
    if required:
        raise ValueError(f"None of {names} found in columns {list(frame.columns)}")
    return None


def clean_sequence(value: object) -> str:
    return "".join(str(value or "").split()).upper()


def load_alias_map(universe: Path) -> dict[str, str]:
    frame = pd.read_csv(universe / "protein_metadata.csv", dtype=str).fillna("")
    mapping: dict[str, str] = {}
    for record in frame.to_dict("records"):
        canonical = str(record["protein_id"]).strip()
        values = [canonical, str(record.get("canonical_accession", "")).strip()]
        values.extend(str(record.get("aliases", "")).split(";"))
        for value in values:
            value = value.strip()
            if value:
                mapping.setdefault(value, canonical)
    return mapping


def load_sequence_map(universe: Path) -> dict[str, str]:
    path = universe / "protein_sequences.tsv"
    if not path.exists():
        return {}
    frame = pd.read_csv(path, sep="\t", dtype=str).fillna("")
    mapping: dict[str, str] = {}
    for protein_id, sequence in frame[["protein_id", "sequence"]].itertuples(index=False):
        sequence = clean_sequence(sequence)
        if sequence:
            mapping.setdefault(sequence, str(protein_id))
    return mapping


def build_reaction_lookup(universe: Path) -> tuple[set[str], dict[str, list[str]]]:
    reactions = pd.read_csv(universe / "reactions.csv", dtype=str).fillna("")
    ids = set(reactions["reaction_id"].astype(str))
    by_canonical: dict[str, list[str]] = {}
    for reaction_id, smiles in reactions[["reaction_id", "reaction_smiles"]].itertuples(index=False):
        key = canonicalize_reaction_smiles(str(smiles), remove_stereo=True)
        if key:
            by_canonical.setdefault(key, []).append(str(reaction_id))
    for key in by_canonical:
        by_canonical[key] = sorted(set(by_canonical[key]))
    return ids, by_canonical


def normalize_official_pairs(frame: pd.DataFrame) -> pd.DataFrame:
    protein_col = _pick(frame, ("UniprotID", "uniprotID", "uniprot_id", "enzyme", "Entry", "protein_id"))
    reaction_id_col = _pick(frame, ("reaction_id", "rhea_id", "Rhea ID", "RHEA_ID"), required=False)
    reaction_smiles_col = _pick(frame, ("CANO_RXN_SMILES", "reaction", "reaction_smiles", "smiles_seq"), required=False)
    if reaction_id_col is None and reaction_smiles_col is None:
        raise ValueError("Official pair table needs a reaction ID or reaction SMILES column")
    label_col = _pick(frame, ("Label", "label", "is_positive", "positive", "y_true"))
    sequence_col = _pick(frame, ("sequence", "Sequence", "protein_sequence"), required=False)
    pred_col = _pick(frame, ("pred", "cage_score", "score"), required=False)
    out = pd.DataFrame({
        "official_protein_id": frame[protein_col].astype(str).str.strip(),
        "official_sequence": frame[sequence_col].astype(str).map(clean_sequence) if sequence_col else "",
        "official_reaction_id": frame[reaction_id_col].astype(str).str.strip() if reaction_id_col else "",
        "official_reaction_smiles": frame[reaction_smiles_col].astype(str).str.strip() if reaction_smiles_col else "",
        "label": pd.to_numeric(frame[label_col], errors="coerce").fillna(0).astype(int).clip(0, 1),
    })
    if pred_col:
        out["cage_score"] = pd.to_numeric(frame[pred_col], errors="coerce")
    return out


def map_official_pairs(frame: pd.DataFrame, universe: Path) -> tuple[pd.DataFrame, dict[str, object]]:
    aliases = load_alias_map(universe)
    sequence_map = load_sequence_map(universe)
    reaction_ids, reaction_lookup = build_reaction_lookup(universe)
    data = normalize_official_pairs(frame).copy()
    direct_proteins = data["official_protein_id"].map(aliases).fillna("")
    sequence_proteins = data["official_sequence"].map(sequence_map).fillna("")
    data["protein_id"] = direct_proteins.where(direct_proteins.ne(""), sequence_proteins)
    data["protein_mapping"] = np.select(
        [direct_proteins.ne(""), direct_proteins.eq("") & sequence_proteins.ne("")],
        ["alias_or_direct", "exact_sequence"],
        default="unmapped",
    )

    mapped_reactions: list[str] = []
    mapping_status: list[str] = []
    candidate_counts: list[int] = []
    for official_id, smiles in data[["official_reaction_id", "official_reaction_smiles"]].itertuples(index=False):
        official_id = str(official_id).strip()
        direct_candidates = [official_id]
        if official_id.isdigit():
            direct_candidates.insert(0, f"RHEA:{official_id}")
        direct = next((value for value in direct_candidates if value in reaction_ids), None)
        if direct is not None:
            mapped_reactions.append(direct)
            mapping_status.append("direct_id")
            candidate_counts.append(1)
            continue
        key = canonicalize_reaction_smiles(str(smiles), remove_stereo=True) if str(smiles).strip() else None
        candidates = reaction_lookup.get(key or "", [])
        candidate_counts.append(len(candidates))
        if len(candidates) == 1:
            mapped_reactions.append(candidates[0])
            mapping_status.append("canonical_smiles")
        elif len(candidates) > 1:
            mapped_reactions.append("")
            mapping_status.append("ambiguous_smiles")
        else:
            mapped_reactions.append("")
            mapping_status.append("unmapped")
    data["reaction_id"] = mapped_reactions
    data["reaction_mapping"] = mapping_status
    data["reaction_mapping_candidate_count"] = candidate_counts
    data["mapped"] = data["protein_id"].ne("") & data["reaction_id"].ne("")
    data["pair_key"] = data["protein_id"] + "\x1f" + data["reaction_id"]
    mapped = data[data["mapped"]].copy()
    duplicate_mapped = int(mapped.duplicated(["protein_id", "reaction_id"]).sum())
    audit: dict[str, object] = {
        "input_rows": int(len(data)),
        "positive_rows": int(data["label"].sum()),
        "mapped_rows": int(data["mapped"].sum()),
        "mapped_fraction": float(data["mapped"].mean()) if len(data) else 0.0,
        "mapped_positive_rows": int(data.loc[data["mapped"], "label"].sum()),
        "positive_mapping_fraction": float(data.loc[data["mapped"], "label"].sum() / data["label"].sum()) if data["label"].sum() else 0.0,
        "mapped_unique_proteins": int(mapped["protein_id"].nunique()),
        "mapped_unique_reactions": int(mapped["reaction_id"].nunique()),
        "duplicate_mapped_pairs": duplicate_mapped,
        "protein_mapping_counts": data["protein_mapping"].value_counts().to_dict(),
        "reaction_mapping_counts": data["reaction_mapping"].value_counts().to_dict(),
    }
    return data, audit


def load_base_exposure(base_dir: Path, universe: Path) -> tuple[set[str], set[str], set[tuple[str, str]]]:
    aliases = load_alias_map(universe)
    path = base_dir / "training_pairs.csv"
    frame = pd.read_csv(path, dtype=str).fillna("")
    pcol = _pick(frame, ("protein_id", "Entry", "uniprot_id", "UniprotID"))
    rcol = _pick(frame, ("reaction_id", "rhea_id"))
    proteins: set[str] = set()
    reactions: set[str] = set()
    pairs: set[tuple[str, str]] = set()
    for raw_p, raw_r in frame[[pcol, rcol]].itertuples(index=False):
        p = aliases.get(str(raw_p).strip(), str(raw_p).strip())
        r = str(raw_r).strip()
        if p:
            proteins.add(p)
        if r:
            reactions.add(r)
        if p and r:
            pairs.add((p, r))
    return proteins, reactions, pairs


def exposure_audit(mapped: pd.DataFrame, base_dir: Path, universe: Path) -> tuple[pd.DataFrame, dict[str, object], set[str]]:
    base_p, base_r, base_pairs = load_base_exposure(base_dir, universe)
    positives = mapped[mapped["mapped"] & mapped["label"].eq(1)].copy()
    positives["base_protein_seen"] = positives["protein_id"].isin(base_p)
    positives["base_reaction_seen"] = positives["reaction_id"].isin(base_r)
    positives["base_exact_pair_seen"] = [
        (p, r) in base_pairs for p, r in positives[["protein_id", "reaction_id"]].itertuples(index=False, name=None)
    ]
    strict_queries: set[str] = set()
    for reaction_id, group in positives.groupby("reaction_id", sort=True):
        if not bool(group["base_reaction_seen"].any()) and not bool(group["base_protein_seen"].any()) and not bool(group["base_exact_pair_seen"].any()):
            strict_queries.add(str(reaction_id))
    audit = {
        "positive_rows": int(len(positives)),
        "base_exact_pair_seen_rows": int(positives["base_exact_pair_seen"].sum()),
        "base_protein_seen_rows": int(positives["base_protein_seen"].sum()),
        "base_reaction_seen_rows": int(positives["base_reaction_seen"].sum()),
        "positive_exact_pair_clean_fraction": float((~positives["base_exact_pair_seen"]).mean()) if len(positives) else 0.0,
        "positive_protein_cold_fraction": float((~positives["base_protein_seen"]).mean()) if len(positives) else 0.0,
        "positive_reaction_cold_fraction": float((~positives["base_reaction_seen"]).mean()) if len(positives) else 0.0,
        "strict_double_cold_query_count": len(strict_queries),
        "mapped_positive_query_count": int(positives["reaction_id"].nunique()),
    }
    return positives, audit, strict_queries


def _encode_selected(model: torch.nn.Module, values: np.ndarray, rows: np.ndarray, *, kind: str, device: torch.device) -> torch.Tensor:
    batch = torch.as_tensor(values[rows], dtype=torch.float32, device=device)
    with torch.no_grad():
        if kind == "protein":
            return model.encode_proteins(batch).detach()
        return encode_model_reactions(model, batch, None).detach()


def score_pair_reservoir(mapped: pd.DataFrame, model_dir: Path, universe: Path, device: torch.device) -> pd.DataFrame:
    data = mapped[mapped["mapped"]].drop_duplicates(["protein_id", "reaction_id"]).copy()
    if data.empty:
        raise ValueError("No mapped pairs to score")
    protein_features, protein_ids = load_protein_library(universe / "proteins")
    pindex = {value: i for i, value in enumerate(protein_ids)}
    schema = load_feature_schema(model_dir)
    reaction_features, reaction_ids = load_registered_reaction_feature_library(
        universe / "reaction_features" / "drfp_categorical_v1", schema
    )
    rindex = {value: i for i, value in enumerate(reaction_ids)}
    missing_p = sorted(set(data["protein_id"]) - set(pindex))
    missing_r = sorted(set(data["reaction_id"]) - set(rindex))
    if missing_p or missing_r:
        raise ValueError(f"Mapped pairs absent from feature universe: proteins={missing_p[:5]}, reactions={missing_r[:5]}")

    unique_p = sorted(set(data["protein_id"]))
    unique_r = sorted(set(data["reaction_id"]))
    p_rows = np.asarray([pindex[x] for x in unique_p], dtype=np.int64)
    r_rows = np.asarray([rindex[x] for x in unique_r], dtype=np.int64)
    local_p = {x: i for i, x in enumerate(unique_p)}
    local_r = {x: i for i, x in enumerate(unique_r)}
    models = load_models(model_dir / "models", "production", device)
    member_scores: list[np.ndarray] = []
    pair_p = np.asarray([local_p[x] for x in data["protein_id"]], dtype=np.int64)
    pair_r = np.asarray([local_r[x] for x in data["reaction_id"]], dtype=np.int64)
    for model in models:
        p_emb = _encode_selected(model, protein_features, p_rows, kind="protein", device=device)
        r_emb = _encode_selected(model, reaction_features, r_rows, kind="reaction", device=device)
        with torch.no_grad():
            member = (p_emb[torch.as_tensor(pair_p, device=device)] * r_emb[torch.as_tensor(pair_r, device=device)]).sum(dim=1)
        member_scores.append(member.detach().cpu().numpy())
    stacked = np.stack(member_scores, axis=0)
    data["neural_score"] = stacked.mean(axis=0)
    data["neural_score_std"] = stacked.std(axis=0)
    for index, values in enumerate(stacked):
        data[f"neural_member_{index}_score"] = values
    data["ensemble_members"] = len(models)
    return data


def evaluate_enzymecage_native_r2e(frame: pd.DataFrame, score_col: str) -> dict[str, float]:
    """Reproduce EnzymeCAGE ``evaluate.py`` ranking semantics exactly.

    This intentionally differs from the shared IR evaluator in two places:
    score ties preserve input-file order (Python's stable score-only sort), and
    percentage EF uses a minimum Top-5 panel while the random expectation still
    uses the *original* requested 1%/2% fraction.  Both details are present in
    the authors' evaluator and matter on small or heavily tied candidate pools.
    """
    required = {"reaction_id", "protein_id", "label", score_col}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Native EnzymeCAGE frame missing columns: {sorted(missing)}")
    data = frame.copy().reset_index(drop=True)
    data[score_col] = pd.to_numeric(data[score_col], errors="coerce")
    data["label"] = pd.to_numeric(data["label"], errors="coerce").fillna(0).astype(int).clip(0, 1)
    if data[score_col].isna().any():
        raise ValueError("Native EnzymeCAGE frame contains missing/non-numeric scores")
    # Mirrors filter_duplicates(): keep the first enzyme occurrence per reaction.
    data = data.drop_duplicates(["reaction_id", "protein_id"], keep="first")

    best_ranks: list[int] = []
    dcgs: list[float] = []
    ef1: list[float] = []
    ef2: list[float] = []
    for _, group in data.groupby("reaction_id", sort=False):
        # mergesort is stable, matching sorted(..., key=score, reverse=True).
        ranked = group.sort_values(score_col, ascending=False, kind="mergesort")
        labels = ranked["label"].to_numpy(dtype=np.int8)
        hit = np.flatnonzero(labels > 0) + 1
        best_ranks.append(int(hit[0]) if len(hit) else -1)
        values = labels[: min(10, len(labels))].astype(float)
        discounts = np.log2(np.arange(2, len(values) + 2, dtype=float))
        dcgs.append(float(np.sum((np.power(2.0, values) - 1.0) / discounts)))
        total_active = int(labels.sum())
        for percent, bucket in ((0.01, ef1), (0.02, ef2)):
            if total_active <= 0:
                bucket.append(0.0)
                continue
            topk = max(int(percent * len(labels)), 5)
            active_top = int(labels[:topk].sum())
            # Exact EnzymeCAGE behavior: expectation is based on requested
            # percent, not on the Top-5 floor actually used for selection.
            bucket.append(float(active_top / (total_active * percent)))

    n_queries = len(best_ranks)
    if n_queries == 0:
        raise ValueError("No reaction queries for native EnzymeCAGE evaluation")
    native: dict[str, float] = {
        "top10_dcg": float(np.mean(dcgs)),
        "top1_percent_ef": float(np.mean(ef1)),
        "top2_percent_ef": float(np.mean(ef2)),
    }
    for k in (1, 3, 5, 10):
        native[f"top{k}_sr"] = float(np.mean([0 < rank <= k for rank in best_ranks]))
    return native


def evaluate_scores(frame: pd.DataFrame, score_col: str) -> dict[str, object]:
    r2e = frame[["reaction_id", "protein_id", score_col, "label"]].rename(
        columns={"reaction_id": "query_id", "protein_id": "candidate_id", score_col: "score"}
    )
    e2r = frame[["protein_id", "reaction_id", score_col, "label"]].rename(
        columns={"protein_id": "query_id", "reaction_id": "candidate_id", score_col: "score"}
    )
    r2e_q, r2e_s = evaluate_ranking_frame(r2e, budgets=DEFAULT_BUDGETS, top_percents=DEFAULT_TOP_PERCENTS)
    e2r_q, e2r_s = evaluate_ranking_frame(e2r, budgets=DEFAULT_BUDGETS, top_percents=DEFAULT_TOP_PERCENTS)
    native = evaluate_enzymecage_native_r2e(frame, score_col)
    return {"enzymecage_native_r2e": native, "reaction_to_enzyme": r2e_s, "enzyme_to_reaction": e2r_s}, pd.concat([
        r2e_q.assign(direction="reaction_to_enzyme"), e2r_q.assign(direction="enzyme_to_reaction")
    ], ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Catalyst and EnzymeCAGE on the exact same official pair reservoir with exposure audits.")
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--universe-dir", type=Path, default=DEFAULT_UNIVERSE)
    parser.add_argument("--base-dir", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--strict-double-cold-only", action="store_true", help="Retain all candidates for only those reaction queries whose mapped positive reaction/proteins are unseen by the base checkpoint.")
    parser.add_argument("--require-complete-mapping", action="store_true", help="Hard-fail unless every official candidate row and every positive maps into the Catalyst feature universe.")
    parser.add_argument("--require-zero-base-positive-exposure", action="store_true", help="Hard-fail unless mapped positives have zero protein, reaction and exact-pair exposure in the declared base checkpoint training pairs.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    pairs_path = args.pairs.resolve(); universe = args.universe_dir.resolve(); model_dir = args.model_dir.resolve(); base_dir = args.base_dir.resolve(); output = args.output_dir.resolve()
    raw = pd.read_csv(pairs_path, dtype=str).fillna("")
    mapped, mapping_audit = map_official_pairs(raw, universe)
    positive_exposure, exposure_summary, strict_queries = exposure_audit(mapped, base_dir, universe)
    if args.require_complete_mapping and (
        mapping_audit["mapped_rows"] != mapping_audit["input_rows"]
        or mapping_audit["mapped_positive_rows"] != mapping_audit["positive_rows"]
    ):
        raise ValueError(f"Official benchmark mapping is incomplete: {mapping_audit}")
    if args.require_zero_base_positive_exposure and any(
        int(exposure_summary[key]) > 0
        for key in ("base_exact_pair_seen_rows", "base_protein_seen_rows", "base_reaction_seen_rows")
    ):
        raise ValueError(f"Base checkpoint has positive benchmark exposure: {exposure_summary}")
    scored_input = mapped[mapped["mapped"]].copy()
    if args.strict_double_cold_only:
        scored_input = scored_input[scored_input["reaction_id"].isin(strict_queries)].copy()
    if scored_input.empty:
        raise ValueError("No rows remain after mapping/strict exposure filtering")
    neural = score_pair_reservoir(scored_input, model_dir, universe, torch.device(args.device))
    neural_metrics, neural_query = evaluate_scores(neural, "neural_score")
    cage_metrics = None; cage_query = None
    if "cage_score" in scored_input.columns and scored_input["cage_score"].notna().all():
        cage_frame = scored_input.drop_duplicates(["protein_id", "reaction_id"]).copy()
        cage_metrics, cage_query = evaluate_scores(cage_frame, "cage_score")

    output.mkdir(parents=True, exist_ok=True)
    mapped.to_csv(output / "mapping_audit_rows.csv", index=False)
    positive_exposure.to_csv(output / "positive_exposure_rows.csv", index=False)
    neural.to_csv(output / "pair_scores.csv", index=False)
    neural_query.to_csv(output / "neural_query_metrics.csv", index=False)
    if cage_query is not None:
        cage_query.to_csv(output / "cage_query_metrics.csv", index=False)
    summary = {
        "protocol": "EnzymeCAGE official pair reservoir aligned evaluation",
        "pairs": str(pairs_path),
        "pairs_sha256": sha256_file(pairs_path),
        "model_dir": str(model_dir),
        "base_dir": str(base_dir),
        "universe_dir": str(universe),
        "strict_double_cold_only": bool(args.strict_double_cold_only),
        "mapping_audit": mapping_audit,
        "base_positive_exposure_audit": exposure_summary,
        "scored_rows": int(len(neural)),
        "scored_reactions": int(neural["reaction_id"].nunique()),
        "scored_proteins": int(neural["protein_id"].nunique()),
        "scored_positive_rows": int(neural["label"].sum()),
        "neural_metrics": neural_metrics,
        "cage_metrics": cage_metrics,
        "metric_contract": {
            "author_native": "Top-10 DCG, EF@1%, EF@2%, and reaction-to-enzyme SR@1/3/5/10 reproduce EnzymeCAGE evaluate.py including stable input-order tie handling and its Top-5-floor EF convention.",
            "shared_ir": "Same pair reservoir additionally reports deterministic candidate-ID-tie first-positive MRR, MAP, NDCG, AUROC, precision and positive recall. These common IR fields are intentionally distinct from author-native tie/EF semantics.",
            "strict_filter": "Strict mode filters reaction queries using positive/base exposure only; negative candidate rows are retained exactly as supplied by the official reservoir.",
        },
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
