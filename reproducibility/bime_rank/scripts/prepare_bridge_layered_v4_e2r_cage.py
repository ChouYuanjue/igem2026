from __future__ import annotations

import json
import pickle
from pathlib import Path

import pandas as pd
import torch
import yaml

from projects.active.bridge.model.assets import ROOT

CAND = ROOT / "results/bridge_layered_v4_candidates"
SUPPORT = ROOT / "results/bridge_layered_v4_cage_support"
AUTHOR = ROOT / "results/bridge_layered_v4_cage_features/e2r"
FALLBACK = ROOT / "results/bridge_layered_v4_cage_features/e2r_fallback"
REACTION = ROOT / "results/bridge_layered_v4_e2r_reaction_assets"
BASE_CONFIG = ROOT / "results/enzymecage_relation_unseen_reaction_assets_v1/probe.yaml"
BASE_GVP = ROOT / "data/external/enzymecage_current/cage_official_features/gvp_feature/gvp_protein_feature.pt"
BASE_NODE = ROOT / "data/external/enzymecage_current/cage_official_features/esmc600m_max_support_v1/pocket_node_feature/esm_node_feature.pt"
BASE_MEAN = ROOT / "data/external/enzymecage_current/cage_official_features/esmc600m_max_support_v1/protein_level/seq2feature.pkl"
BASE_MAP = ROOT / "data/external/enzymecage_current/cage_official_features/esmc600m_max_support_v1/mapping_audit.csv"
CKPT = ROOT / "external_repos/EnzymeCAGE/checkpoints/pretrain/seed_42"
OUT = ROOT / "results/bridge_layered_v4_e2r_cage"
EXTRA = OUT / "extra_features"


def feature_paths(root: Path) -> tuple[Path, Path, Path]:
    return (
        root / "feature/protein/gvp_feature/gvp_protein_feature.pt",
        root / "feature/protein/ESM-C_600M/pocket_node_feature/esm_node_feature.pt",
        root / "feature/protein/ESM-C_600M/protein_level/seq2feature.pkl",
    )


def load_feature_triplet(root: Path) -> tuple[dict, dict, dict]:
    gvp_path, node_path, mean_path = feature_paths(root)
    missing = [p for p in (gvp_path, node_path, mean_path) if not p.exists()]
    if missing:
        raise FileNotFoundError("missing protein features: " + ", ".join(map(str, missing)))
    gvp = torch.load(gvp_path, map_location="cpu", weights_only=False)
    node = torch.load(node_path, map_location="cpu", weights_only=False)
    with mean_path.open("rb") as handle:
        mean = pickle.load(handle)
    return gvp, node, mean


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    EXTRA.mkdir(parents=True, exist_ok=True)

    author_gvp, author_node, author_mean = load_feature_triplet(AUTHOR)
    fallback_gvp, fallback_node, fallback_mean = load_feature_triplet(FALLBACK)

    extra_gvp = dict(author_gvp)
    extra_gvp.update(fallback_gvp)
    extra_node = dict(author_node)
    extra_node.update(fallback_node)
    extra_mean = dict(author_mean)
    extra_mean.update(fallback_mean)

    extra_gvp_path = EXTRA / "gvp_protein_feature.pt"
    extra_node_path = EXTRA / "esm_node_feature.pt"
    extra_mean_path = EXTRA / "seq2feature.pkl"
    torch.save(extra_gvp, extra_gvp_path)
    torch.save(extra_node, extra_node_path)
    with extra_mean_path.open("wb") as handle:
        pickle.dump(extra_mean, handle)

    existing_frame = pd.read_csv(BASE_MAP, dtype=str).fillna("")
    existing = set(
        existing_frame.loc[
            existing_frame.mapping_ok.astype(str).str.lower().eq("true"), "UniprotID"
        ].astype(str)
    )
    supported_uids = existing | (set(map(str, extra_gvp)) & set(map(str, extra_node)))

    registry = pd.read_csv(
        SUPPORT / "protein_support_registry.csv.gz", dtype=str
    ).fillna("")
    query_registry = registry[
        registry.entity_kind.eq("general_merged")
        & registry.scopes.str.contains("e2r_query", regex=False)
    ][["entity_id", "chosen_uid", "sequence", "support_source"]].drop_duplicates(
        "entity_id"
    )
    query_registry = query_registry.rename(columns={"entity_id": "protein_id"})
    query_registry["protein_supported"] = query_registry.chosen_uid.isin(supported_uids)

    native = pd.read_csv(
        CAND / "e2r_similarity_gate_candidates.csv.gz", dtype=str
    ).fillna("")
    broad = pd.read_csv(
        CAND / "e2r_broad_equal_budget_candidates.csv.gz", dtype=str
    ).fillna("")
    native["native_pool"] = True
    broad["broad_pool"] = True
    native["native_rank"] = pd.to_numeric(
        native["candidate_rank_in_gate"], errors="raise"
    ).astype(int)
    broad["broad_rank"] = pd.to_numeric(broad["broad_rank"], errors="raise").astype(int)

    keys = ["protein_id", "reaction_id"]
    membership = native[
        keys + ["label", "native_pool", "native_rank"]
    ].merge(
        broad[keys + ["label", "broad_pool", "broad_rank", "broad_score"]],
        on=keys,
        how="outer",
        suffixes=("_native", "_broad"),
    )
    membership["native_pool"] = membership.native_pool.fillna(False).astype(bool)
    membership["broad_pool"] = membership.broad_pool.fillna(False).astype(bool)
    membership["label"] = (
        pd.to_numeric(membership.label_native, errors="coerce").fillna(0).astype(int)
        | pd.to_numeric(membership.label_broad, errors="coerce").fillna(0).astype(int)
    ).astype(int)
    membership = membership.drop(columns=["label_native", "label_broad"])

    reaction_registry = pd.read_csv(
        REACTION / "reaction_registry.csv", dtype=str
    ).fillna("")
    reaction_audit = pd.read_csv(
        REACTION / "reaction_support_audit.csv", dtype=str
    ).fillna("")
    reaction_support = reaction_audit[
        ["reaction_id", "supported", "reason"]
    ].drop_duplicates("reaction_id")
    reaction_support["reaction_supported"] = reaction_support.supported.astype(
        str
    ).str.lower().eq("true")
    reaction_support = reaction_support.drop(columns=["supported"])

    membership = membership.merge(
        query_registry,
        on="protein_id",
        how="left",
        validate="many_to_one",
    ).merge(
        reaction_registry,
        on="reaction_id",
        how="left",
        validate="many_to_one",
    ).merge(
        reaction_support,
        on="reaction_id",
        how="left",
        validate="many_to_one",
    )
    membership["protein_supported"] = membership.protein_supported.fillna(False).astype(bool)
    membership["reaction_supported"] = membership.reaction_supported.fillna(False).astype(bool)
    membership["cage_pair_supported"] = (
        membership.protein_supported & membership.reaction_supported
    )
    membership.to_csv(OUT / "pair_membership.csv.gz", index=False)

    pairs = membership[membership.cage_pair_supported].copy()
    pairs["UniprotID"] = pairs.chosen_uid.astype(str)
    pairs["Label"] = pairs.label.astype(int)
    pairs = pairs[
        [
            "protein_id",
            "reaction_id",
            "UniprotID",
            "sequence",
            "CANO_RXN_SMILES",
            "Label",
        ]
    ].drop_duplicates(["protein_id", "reaction_id"])
    pairs.to_csv(OUT / "pairs.csv", index=False)

    config = yaml.safe_load(BASE_CONFIG.read_text())
    config["data_path"] = str((OUT / "pairs.csv").resolve())
    config["protein_gvp_feat"] = str(BASE_GVP.resolve())
    config["esm_node_feature"] = str(BASE_NODE.resolve())
    config["esm_mean_feature"] = str(BASE_MEAN.resolve())
    config["protein_gvp_feat_extra"] = str(extra_gvp_path.resolve())
    config["esm_node_feature_extra"] = str(extra_node_path.resolve())
    config["esm_mean_feature_extra"] = str(extra_mean_path.resolve())
    config["rxn_fp"] = str((REACTION / "feature/reaction/drfp/rxn2fp.pkl").resolve())
    config["mol_conformation"] = str(
        (REACTION / "feature/reaction/molecule_conformation").resolve()
    )
    config["reaction_center"] = str(
        (REACTION / "feature/reaction/reacting_center/reacting_center.pkl").resolve()
    )
    config["ckpt_dir"] = str(CKPT.resolve())
    config["model_list"] = ["epoch_19.pth"]
    config["result_dir"] = str((OUT / "cage_inference").resolve())
    config["batch_size"] = 256
    (OUT / "cage_infer.yaml").write_text(yaml.safe_dump(config, sort_keys=False))

    budgets = pd.read_csv(CAND / "e2r_query_budget.csv", dtype=str).fillna("")
    nonzero = budgets[pd.to_numeric(budgets.effective_budget).gt(0)]
    supported_queries = set(
        query_registry.loc[query_registry.protein_supported, "protein_id"].astype(str)
    )
    pair_supported_queries = set(pairs.protein_id.astype(str))
    summary = {
        "schema": "bridge-layered-v4-e2r-cage-prepare",
        "status": "prepared",
        "mother_queries": int(len(budgets)),
        "nonzero_budget_queries": int(len(nonzero)),
        "protein_supported_queries": int(len(supported_queries)),
        "pair_supported_queries": int(len(pair_supported_queries)),
        "extra_gvp_uids": int(len(extra_gvp)),
        "extra_node_uids": int(len(extra_node)),
        "union_candidate_pairs": int(len(membership)),
        "cage_supported_pairs": int(len(pairs)),
        "unsupported_pairs": int((~membership.cage_pair_supported).sum()),
        "native_candidate_rows": int(membership.native_pool.sum()),
        "broad_candidate_rows": int(membership.broad_pool.sum()),
        "rules": {
            "candidate_generation_labels_used": False,
            "unsupported_queries_retained_in_final_evaluation": True,
            "unsupported_pairs_not_silently_replaced": True,
        },
    }
    (OUT / "prepare_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
