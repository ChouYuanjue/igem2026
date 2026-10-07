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
AUTHOR = ROOT / "results/bridge_layered_v4_cage_features/r2e_union"
FALLBACK = ROOT / "results/bridge_layered_v4_cage_features/r2e_fallback"
REACTION = ROOT / "results/enzymecage_relation_unseen_reaction_assets_v1"
BASE_GVP = ROOT / "data/external/enzymecage_current/cage_official_features/gvp_feature/gvp_protein_feature.pt"
BASE_NODE = ROOT / "data/external/enzymecage_current/cage_official_features/esmc600m_max_support_v1/pocket_node_feature/esm_node_feature.pt"
BASE_MEAN = ROOT / "data/external/enzymecage_current/cage_official_features/esmc600m_max_support_v1/protein_level/seq2feature.pkl"
BASE_MAP = ROOT / "data/external/enzymecage_current/cage_official_features/esmc600m_max_support_v1/mapping_audit.csv"
CKPT = ROOT / "external_repos/EnzymeCAGE/checkpoints/pretrain/seed_42"
OUT = ROOT / "results/bridge_layered_v4_r2e_cage"
EXTRA = OUT / "extra_features"
CAGE_SKIP_MOL = {"[*H2]"}


def feature_paths(root: Path) -> tuple[Path, Path, Path]:
    return (
        root / "feature/protein/gvp_feature/gvp_protein_feature.pt",
        root / "feature/protein/ESM-C_600M/pocket_node_feature/esm_node_feature.pt",
        root / "feature/protein/ESM-C_600M/protein_level/seq2feature.pkl",
    )


def load_triplet(root: Path, *, required: bool) -> tuple[dict, dict, dict]:
    paths = feature_paths(root)
    if not all(p.exists() for p in paths):
        if required:
            missing = [str(p) for p in paths if not p.exists()]
            raise FileNotFoundError("missing feature assets: " + ", ".join(missing))
        return {}, {}, {}
    gvp = torch.load(paths[0], map_location="cpu", weights_only=False)
    node = torch.load(paths[1], map_location="cpu", weights_only=False)
    with paths[2].open("rb") as handle:
        mean = pickle.load(handle)
    return gvp, node, mean


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    EXTRA.mkdir(parents=True, exist_ok=True)

    author_gvp, author_node, author_mean = load_triplet(AUTHOR, required=True)
    fallback_gvp, fallback_node, fallback_mean = load_triplet(FALLBACK, required=False)

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
            existing_frame.mapping_ok.astype(str).str.lower().eq("true"),
            "UniprotID",
        ].astype(str)
    )
    supported_uids = existing | (set(map(str, extra_gvp)) & set(map(str, extra_node)))

    registry = pd.read_csv(
        SUPPORT / "protein_support_registry.csv.gz", dtype=str
    ).fillna("")
    native_registry = registry[
        registry.entity_kind.eq("native_cage_uid")
    ][["entity_id", "chosen_uid", "sequence", "support_source"]].drop_duplicates(
        "chosen_uid"
    )
    native_registry = native_registry.rename(columns={"entity_id": "registry_entity_id"})

    broad_registry = registry[
        registry.entity_kind.eq("general_merged")
        & registry.scopes.str.contains("r2e_broad_candidate", regex=False)
    ][["entity_id", "chosen_uid", "sequence", "support_source"]].drop_duplicates(
        "entity_id"
    )
    broad_registry = broad_registry.rename(columns={"entity_id": "protein_id"})

    reaction_probe = pd.read_csv(
        REACTION / "probe_pairs.csv", dtype=str
    ).fillna("")
    reaction_map = (
        reaction_probe[["reaction_id", "CANO_RXN_SMILES"]]
        .drop_duplicates("reaction_id")
        .set_index("reaction_id")
        .CANO_RXN_SMILES.to_dict()
    )
    probe_config = yaml.safe_load((REACTION / "probe.yaml").read_text())
    mol_graph_path = Path(probe_config["mol_conformation"]) / "mol_graph_dict.pt"
    mol_graphs = torch.load(mol_graph_path, map_location="cpu", weights_only=False)
    mol_graph_keys = set(map(str, mol_graphs))
    graph_bad_reactions: dict[str, list[str]] = {}
    for rid, reaction in reaction_map.items():
        left, right = str(reaction).split(">>")
        molecules = {
            x.replace("*", "C")
            for x in left.split(".") + right.split(".")
            if x and x not in CAGE_SKIP_MOL
        }
        missing = sorted(molecules - mol_graph_keys)
        if missing:
            graph_bad_reactions[str(rid)] = missing
    supported_reactions = set(reaction_map) - set(graph_bad_reactions)
    (OUT / "reaction_graph_preflight.json").write_text(
        json.dumps(
            {
                "checked_reactions": len(reaction_map),
                "unsupported_reactions": len(graph_bad_reactions),
                "details": graph_bad_reactions,
            },
            indent=2,
        ) + "\n"
    )

    native = pd.read_csv(
        CAND / "r2e_native_candidates.csv.gz", dtype=str
    ).fillna("")
    native["route"] = "enzymecage"
    native["logical_candidate_id"] = native.candidate_uid.astype(str)
    native["score_uid"] = native.candidate_uid.astype(str)
    native["target_protein_ids"] = native.candidate_protein_ids.astype(str)
    native["candidate_rank"] = pd.to_numeric(
        native.candidate_rank_in_gate, errors="raise"
    ).astype(int)
    native = native.merge(
        native_registry[["chosen_uid", "sequence", "support_source"]],
        left_on="score_uid",
        right_on="chosen_uid",
        how="left",
        validate="many_to_one",
    )

    broad = pd.read_csv(
        CAND / "r2e_broad_equal_budget_candidates.csv.gz", dtype=str
    ).fillna("")
    broad["route"] = "broad_cage"
    broad["logical_candidate_id"] = broad.protein_id.astype(str)
    broad["target_protein_ids"] = broad.protein_id.astype(str)
    broad["candidate_rank"] = pd.to_numeric(broad.broad_rank, errors="raise").astype(int)
    broad = broad.merge(
        broad_registry,
        on="protein_id",
        how="left",
        validate="many_to_one",
    )
    broad["score_uid"] = broad.chosen_uid.astype(str)

    keep = [
        "route",
        "reaction_id",
        "logical_candidate_id",
        "score_uid",
        "target_protein_ids",
        "candidate_rank",
        "label",
        "sequence",
        "support_source",
    ]
    membership = pd.concat(
        [native[keep], broad[keep]],
        ignore_index=True,
    )
    membership["label"] = pd.to_numeric(
        membership.label, errors="coerce"
    ).fillna(0).astype(int)
    membership["protein_supported"] = membership.score_uid.isin(supported_uids)
    membership["reaction_supported"] = membership.reaction_id.isin(
        supported_reactions
    )
    membership["cage_pair_supported"] = (
        membership.protein_supported & membership.reaction_supported
    )
    membership.to_csv(OUT / "pair_membership.csv.gz", index=False)

    pairs = membership[membership.cage_pair_supported].copy()
    pairs["UniprotID"] = pairs.score_uid.astype(str)
    pairs["CANO_RXN_SMILES"] = pairs.reaction_id.map(reaction_map).fillna("")
    pairs["Label"] = pairs.label.astype(int)
    pairs = (
        pairs[
            [
                "reaction_id",
                "UniprotID",
                "sequence",
                "CANO_RXN_SMILES",
                "Label",
            ]
        ]
        .sort_values(["reaction_id", "UniprotID"], kind="stable")
        .drop_duplicates(["reaction_id", "UniprotID"], keep="first")
    )
    if pairs.CANO_RXN_SMILES.eq("").any():
        raise RuntimeError("supported pair has missing reaction representation")
    pairs.to_csv(OUT / "pairs.csv", index=False)

    config = dict(probe_config)
    config["data_path"] = str((OUT / "pairs.csv").resolve())
    config["protein_gvp_feat"] = str(BASE_GVP.resolve())
    config["esm_node_feature"] = str(BASE_NODE.resolve())
    config["esm_mean_feature"] = str(BASE_MEAN.resolve())
    config["protein_gvp_feat_extra"] = str(extra_gvp_path.resolve())
    config["esm_node_feature_extra"] = str(extra_node_path.resolve())
    config["esm_mean_feature_extra"] = str(extra_mean_path.resolve())
    config["ckpt_dir"] = str(CKPT.resolve())
    config["model_list"] = ["epoch_19.pth"]
    config["result_dir"] = str((OUT / "cage_inference").resolve())
    # 128 was validated through the former failure point with ample 4090 headroom;
    # 256 is unsafe while the production Starase service remains resident.
    config["batch_size"] = 128
    (OUT / "cage_infer.yaml").write_text(yaml.safe_dump(config, sort_keys=False))

    budget = pd.read_csv(CAND / "r2e_query_budget.csv", dtype=str).fillna("")
    summary = {
        "schema": "bridge-layered-v4-r2e-cage-prepare",
        "status": "prepared",
        "mother_queries": int(len(budget)),
        "supported_reaction_queries": int(
            budget.reaction_id.astype(str).isin(supported_reactions).sum()
        ),
        "unsupported_reaction_queries": int(
            (~budget.reaction_id.astype(str).isin(supported_reactions)).sum()
        ),
        "graph_preflight_unsupported_reactions": int(len(graph_bad_reactions)),
        "base_feature_uids": int(len(existing)),
        "author_extra_uids": int(
            len(set(map(str, author_gvp)) & set(map(str, author_node)))
        ),
        "fallback_extra_uids": int(
            len(set(map(str, fallback_gvp)) & set(map(str, fallback_node)))
        ),
        "logical_candidate_rows": int(len(membership)),
        "native_candidate_rows": int(membership.route.eq("enzymecage").sum()),
        "broad_candidate_rows": int(membership.route.eq("broad_cage").sum()),
        "unique_cage_pairs": int(len(pairs)),
        "cage_supported_logical_rows": int(membership.cage_pair_supported.sum()),
        "unsupported_logical_rows": int((~membership.cage_pair_supported).sum()),
        "rules": {
            "candidate_sets_remain_route_specific": True,
            "neural_scores_deduplicated_by_reaction_uid_only": True,
            "unsupported_query_or_pair_retained_as_miss": True,
        },
    }
    (OUT / "prepare_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
