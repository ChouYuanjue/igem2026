from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from projects.active.terpene_screening.evaluate_enzymecage_official_aligned import (  # noqa: E402
    clean_sequence,
    evaluate_scores,
    load_alias_map,
    load_sequence_map,
)
from projects.active.terpene_screening.fair_benchmark import sha256_file  # noqa: E402
from projects.active.terpene_screening.rank_open_world import (  # noqa: E402
    load_feature_schema,
    load_models,
    load_protein_library,
)

DEFAULT_PAIRS = ROOT / "data/external/enzymecage_current/Enzyme-405.csv"
DEFAULT_UNIVERSE = ROOT / "data/catalyst_candidate_universes/general_merged"
DEFAULT_NEW_PROTEINS = (
    ROOT / "data/external/enzymecage_current/catalyst_features/new_protein_esmc"
)
DEFAULT_QUERY_REACTIONS = (
    ROOT / "data/external/enzymecage_current/catalyst_features/query_reaction_features"
)
DEFAULT_MEMBERSHIP = ROOT / "data/external/enzymecage_current/candidate_membership_2023.csv"
DEFAULT_OUTPUT = ROOT / "results/enzymecage_405_cleanroom_eval"
DEFAULT_AUTHOR_GVP = ROOT / "data/external/enzymecage_current/cage_official_features/gvp_feature/gvp_protein_feature.pt"


def official_query_id(reaction: str) -> str:
    return "E405:" + hashlib.sha1(str(reaction).encode("utf-8")).hexdigest()[:16]


def _normalize_rows(values: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (values / norms).astype(np.float32, copy=False)


def filter_author_valid_pocket_reservoir(frame: pd.DataFrame, gvp_path: Path) -> tuple[pd.DataFrame, dict[str, object]]:
    """Mirror EnzymeCAGE infer.py's valid-GVP candidate filtering.

    This is a comparison reservoir, not a replacement for the full official
    table.  We retain the full-reservoir result separately and make the filter
    auditable because author inference drops candidates without usable pocket
    geometry before ranking.
    """
    features = torch.load(gvp_path, map_location="cpu", weights_only=False)
    if not isinstance(features, dict):
        raise ValueError(f"author GVP feature file must contain a UID mapping: {gvp_path}")
    valid = set(map(str, features.keys()))
    before = frame.copy()
    filtered = before[before["UniprotID"].astype(str).isin(valid)].copy()
    if filtered.empty:
        raise ValueError("author-valid pocket reservoir is empty")
    query_col = "reaction_id" if "reaction_id" in filtered.columns else "CANO_RXN_SMILES"
    label_col = "label" if "label" in filtered.columns else "Label"
    labels = pd.to_numeric(filtered[label_col], errors="raise").astype(int)
    positive_by_query = filtered.assign(_label=labels).groupby(query_col)["_label"].sum()
    audit = {
        "mode": "author_valid_pocket",
        "gvp_path": str(gvp_path.resolve()),
        "gvp_sha256": sha256_file(gvp_path),
        "raw_rows": int(len(before)),
        "filtered_rows": int(len(filtered)),
        "removed_rows": int(len(before) - len(filtered)),
        "raw_candidate_uids": int(before["UniprotID"].astype(str).nunique()),
        "filtered_candidate_uids": int(filtered["UniprotID"].astype(str).nunique()),
        "removed_candidate_uids": int(before["UniprotID"].astype(str).nunique() - filtered["UniprotID"].astype(str).nunique()),
        "filtered_queries": int(filtered[query_col].nunique()),
        "queries_without_positive_after_filter": int((positive_by_query <= 0).sum()),
        "filtered_positive_rows": int(labels.sum()),
    }
    return filtered, audit


def build_official_protein_library(
    pairs: pd.DataFrame,
    universe: Path,
    new_protein_dir: Path,
) -> tuple[np.ndarray, list[str], pd.DataFrame]:
    base_features, base_ids = load_protein_library(universe / "proteins")
    base_index = {value: row for row, value in enumerate(base_ids)}
    aliases = load_alias_map(universe)
    sequence_map = load_sequence_map(universe)
    extra_features, extra_ids = load_protein_library(new_protein_dir)
    extra_index = {value: row for row, value in enumerate(extra_ids)}

    entities = pairs[["UniprotID", "sequence"]].drop_duplicates("UniprotID", keep="first")
    vectors: list[np.ndarray] = []
    ids: list[str] = []
    audit: list[dict[str, object]] = []
    for uid, sequence in entities[["UniprotID", "sequence"]].itertuples(index=False):
        uid = str(uid)
        sequence = clean_sequence(sequence)
        canonical = aliases.get(uid, "")
        source = "direct_or_alias"
        if not canonical:
            canonical = sequence_map.get(sequence, "")
            source = "exact_sequence"
        if canonical and canonical in base_index:
            vector = base_features[base_index[canonical]]
            mapped_id = canonical
        elif uid in extra_index:
            vector = extra_features[extra_index[uid]]
            mapped_id = uid
            source = "new_input_esmc"
        else:
            raise ValueError(f"No ESM-C feature for official candidate {uid}")
        vectors.append(np.asarray(vector, dtype=np.float32))
        ids.append(uid)
        audit.append(
            {
                "protein_id": uid,
                "feature_source": source,
                "feature_entity": mapped_id,
            }
        )
    matrix = _normalize_rows(np.stack(vectors))
    return matrix, ids, pd.DataFrame(audit)


def load_query_reaction_library(feature_dir: Path) -> tuple[np.ndarray, list[str]]:
    entries = pd.read_csv(feature_dir / "entries.csv", dtype={"reaction_id": str}).sort_values("row")
    matrix = np.load(feature_dir / "reaction_feature_matrix.npy").astype(np.float32)
    if len(entries) != len(matrix):
        raise ValueError("query reaction entries/features differ in length")
    return matrix, entries["reaction_id"].astype(str).tolist()


def add_membership_features(frame: pd.DataFrame, path: Path) -> pd.DataFrame:
    membership = pd.read_csv(path, dtype=str).fillna("")
    required = {"UniprotID", "seen_uid_2023", "seen_exact_sequence_2023", "seen_any_2023"}
    if not required <= set(membership.columns):
        raise ValueError(f"membership file missing {sorted(required - set(membership.columns))}")
    for column in ["seen_uid_2023", "seen_exact_sequence_2023", "seen_any_2023"]:
        membership[column] = membership[column].astype(str).str.lower().eq("true")
    data = frame.merge(
        membership[["UniprotID", "seen_uid_2023", "seen_exact_sequence_2023", "seen_any_2023"]],
        on="UniprotID",
        how="left",
        validate="many_to_one",
    )
    if data["seen_any_2023"].isna().any():
        missing = data.loc[data["seen_any_2023"].isna(), "UniprotID"].drop_duplicates().head().tolist()
        raise ValueError(f"membership missing official candidates: {missing}")
    return data


def score_models(
    frame: pd.DataFrame,
    *,
    model_dir: Path,
    protein_features: np.ndarray,
    protein_ids: list[str],
    reaction_features: np.ndarray,
    reaction_ids: list[str],
    device: torch.device,
) -> pd.DataFrame:
    schema = load_feature_schema(model_dir)
    if int(schema.get("protein_feature_dimension") or protein_features.shape[1]) != protein_features.shape[1]:
        raise ValueError("protein feature dimension mismatch")
    if int(schema.get("reaction_feature_dimension") or reaction_features.shape[1]) != reaction_features.shape[1]:
        raise ValueError("reaction feature dimension mismatch")
    pindex = {value: row for row, value in enumerate(protein_ids)}
    rindex = {value: row for row, value in enumerate(reaction_ids)}
    models = load_models(model_dir / "models", "production", device)
    p_tensor = torch.as_tensor(protein_features, dtype=torch.float32, device=device)
    r_tensor = torch.as_tensor(reaction_features, dtype=torch.float32, device=device)
    pair_p = torch.as_tensor([pindex[value] for value in frame["protein_id"]], dtype=torch.long, device=device)
    pair_r = torch.as_tensor([rindex[value] for value in frame["reaction_id"]], dtype=torch.long, device=device)
    member_scores: list[np.ndarray] = []
    with torch.no_grad():
        for index, model in enumerate(models):
            p_emb = model.encode_proteins(p_tensor)
            r_emb = model.encode_reactions(r_tensor)
            score = (p_emb[pair_p] * r_emb[pair_r]).sum(dim=1).cpu().numpy()
            member_scores.append(score)
            frame[f"neural_member_{index}_score"] = score
    stacked = np.stack(member_scores)
    frame["neural_score"] = stacked.mean(axis=0)
    frame["neural_score_std"] = stacked.std(axis=0)
    return frame


def add_protocol_aware_score(frame: pd.DataFrame) -> pd.DataFrame:
    """Fixed lexicographic novelty-first expert; no labels or tuned scalar are used.

    Enzyme-405 is explicitly constructed with test positives absent from the 2023
    snapshot while negatives are sourced from enzymes of similar 2023 reactions.
    This expert exposes that benchmark-construction shortcut rather than silently
    attributing it to biological modeling.  A 4-point bucket offset dominates the
    cosine neural range [-1, 1], so unseen candidates rank ahead of seen candidates;
    the clean neural score orders candidates only within each membership bucket.
    """
    unseen = (~frame["seen_any_2023"].astype(bool)).astype(float)
    frame["protocol_novelty_score"] = 4.0 * unseen + frame["neural_score"].astype(float)
    return frame


def main() -> None:
    parser = argparse.ArgumentParser(description="Exact Enzyme-405 scoring for leakage-clean Catalyst models.")
    parser.add_argument("--pairs", type=Path, default=DEFAULT_PAIRS)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--universe-dir", type=Path, default=DEFAULT_UNIVERSE)
    parser.add_argument("--new-protein-dir", type=Path, default=DEFAULT_NEW_PROTEINS)
    parser.add_argument("--query-reaction-dir", type=Path, default=DEFAULT_QUERY_REACTIONS)
    parser.add_argument("--membership", type=Path, default=DEFAULT_MEMBERSHIP)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--reservoir", choices=["full_official", "author_valid_pocket"], default="full_official")
    parser.add_argument("--author-gvp", type=Path, default=DEFAULT_AUTHOR_GVP)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    source = args.pairs.resolve()
    raw = pd.read_csv(source, dtype=str).fillna("")
    required = {"UniprotID", "sequence", "CANO_RXN_SMILES", "Label"}
    if not required <= set(raw.columns):
        raise ValueError(f"official Enzyme-405 table missing {sorted(required - set(raw.columns))}")
    raw = raw.copy()
    raw["_official_row"] = np.arange(len(raw), dtype=np.int64)
    raw["reaction_id"] = raw["CANO_RXN_SMILES"].map(official_query_id)
    raw["protein_id"] = raw["UniprotID"].astype(str)
    raw["label"] = pd.to_numeric(raw["Label"], errors="raise").astype(int)
    if raw["reaction_id"].nunique() != 295:
        raise ValueError(f"Expected 295 canonical Enzyme-405 queries, found {raw['reaction_id'].nunique()}")
    source_counts = {
        "rows": int(len(raw)),
        "queries": int(raw["reaction_id"].nunique()),
        "candidate_uids": int(raw["protein_id"].nunique()),
        "positive_rows": int(raw["label"].sum()),
    }
    reservoir_audit: dict[str, object] = {"mode": "full_official", **source_counts}
    if args.reservoir == "author_valid_pocket":
        raw, reservoir_audit = filter_author_valid_pocket_reservoir(raw, args.author_gvp.resolve())
        if raw["reaction_id"].nunique() != 295:
            raise ValueError(f"Author-valid pocket filtering removed reaction queries: {raw['reaction_id'].nunique()} remain")
        if raw.groupby("reaction_id")["label"].sum().le(0).any():
            raise ValueError("Author-valid pocket filtering leaves a reaction without any positive")

    protein_features, protein_ids, protein_audit = build_official_protein_library(
        raw, args.universe_dir.resolve(), args.new_protein_dir.resolve()
    )
    reaction_features, reaction_ids = load_query_reaction_library(args.query_reaction_dir.resolve())
    missing_reactions = sorted(set(raw["reaction_id"]) - set(reaction_ids))
    if missing_reactions:
        raise ValueError(f"query feature library misses reactions: {missing_reactions[:5]}")
    frame = add_membership_features(raw, args.membership.resolve())
    frame = score_models(
        frame,
        model_dir=args.model_dir.resolve(),
        protein_features=protein_features,
        protein_ids=protein_ids,
        reaction_features=reaction_features,
        reaction_ids=reaction_ids,
        device=torch.device(args.device),
    )
    frame = add_protocol_aware_score(frame)
    neural_metrics, neural_query = evaluate_scores(frame, "neural_score")
    protocol_metrics, protocol_query = evaluate_scores(frame, "protocol_novelty_score")

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    frame.sort_values("_official_row").to_csv(output / "pair_scores.csv", index=False)
    protein_audit.to_csv(output / "protein_feature_audit.csv", index=False)
    neural_query.to_csv(output / "neural_query_metrics.csv", index=False)
    protocol_query.to_csv(output / "protocol_query_metrics.csv", index=False)
    summary = {
        "protocol": "official Enzyme-405 canonical-reaction reservoir",
        "reservoir_mode": args.reservoir,
        "source_counts_before_reservoir_filter": source_counts,
        "reservoir_audit": reservoir_audit,
        "pairs": str(source),
        "pairs_sha256": sha256_file(source),
        "model_dir": str(args.model_dir.resolve()),
        "model_summary": (
            json.loads((args.model_dir.resolve() / "summary.json").read_text(encoding="utf-8"))
            if (args.model_dir.resolve() / "summary.json").exists()
            else None
        ),
        "rows": int(len(frame)),
        "unique_query_candidate_pairs": int(len(frame.drop_duplicates(["reaction_id", "protein_id"]))),
        "duplicate_query_candidate_rows_removed_by_official_protocol": int(
            len(frame) - len(frame.drop_duplicates(["reaction_id", "protein_id"]))
        ),
        "queries": int(frame["reaction_id"].nunique()),
        "candidate_uids": int(frame["protein_id"].nunique()),
        "positive_rows": int(frame["label"].sum()),
        "protein_feature_coverage": int(len(protein_ids)),
        "reaction_feature_coverage": int(len(reaction_ids)),
        "membership_file": str(args.membership.resolve()),
        "membership_file_sha256": sha256_file(args.membership.resolve()),
        "target_labels_used_for_routing": False,
        "neural_metrics": neural_metrics,
        "protocol_aware_metrics": protocol_metrics,
        "protocol_aware_warning": (
            "The novelty-first expert is a benchmark-construction shortcut: Enzyme-405 explicitly makes positive "
            "enzymes unseen in the 2023 snapshot and draws negatives from enzymes of similar 2023 reactions. "
            "Report it separately from the biology-facing neural result."
        ),
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
