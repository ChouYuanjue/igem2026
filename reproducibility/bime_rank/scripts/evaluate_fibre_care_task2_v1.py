from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import sys
import zipfile
from hashlib import blake2b
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from drfp import DrfpEncoder

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from projects.active.bridge.model.index import DEFAULT_INDEX, FibreCandidateIndex
from projects.active.bridge.runtime.base_model import ModelConfig, TerpeneDualTower
from projects.active.bridge.runtime.cli import encode_reaction_with_audit, load_feature_schema

CARE_ZIP = ROOT / "data/external/care/CARE_datasets.zip"
GENERAL_SEQS = ROOT / "data/catalyst_candidate_universes/general_merged/protein_sequences.tsv"
SCHEMA_SOURCE = ROOT / "results/terpene_production_models/marts_adapted_drfp_pu"
DEFAULT_OUTPUT = ROOT / "results/fibre_care_task2_v1"
SPLITS = ("easy", "medium", "hard")


def install_legacy_drfp_hash() -> None:
    def legacy_hash(shingling):
        vals = [int(blake2b(t, digest_size=4).hexdigest(), 16) for t in shingling]
        return np.asarray(vals, dtype=np.uint32).view(np.int32)
    DrfpEncoder.hash = staticmethod(legacy_hash)


def read_zip_csv(member: str) -> pd.DataFrame:
    with zipfile.ZipFile(CARE_ZIP) as zf:
        with zf.open(member) as fh:
            return pd.read_csv(fh, dtype=str).fillna("")


def read_ec_list() -> list[str]:
    with zipfile.ZipFile(CARE_ZIP) as zf:
        raw = zf.read("CARE_datasets/processed_data/EC_list.txt").decode("utf-8")
    return [x.strip() for x in raw.splitlines() if x.strip()]


def stable_reaction_id(reaction: str) -> str:
    return "CARE_RXN_" + hashlib.sha1(reaction.encode("utf-8")).hexdigest()[:20]


def prepare(output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    install_legacy_drfp_hash()
    schema = load_feature_schema(SCHEMA_SOURCE)
    expected = int(schema["reaction_feature_dimension"])

    tests: dict[str, pd.DataFrame] = {}
    reactions: dict[str, str] = {}
    for split in SPLITS:
        df = read_zip_csv(f"CARE_datasets/splits/task2/{split}_reaction_test.csv")
        if not {"Reaction", "EC number"} <= set(df.columns):
            raise ValueError(f"{split}: CARE Task2 schema changed")
        df = df.copy()
        df["reaction_id"] = df["Reaction"].map(stable_reaction_id)
        tests[split] = df
        reactions.update(dict(zip(df["reaction_id"], df["Reaction"])))
        df[["reaction_id", "Reaction", "EC number"]].to_csv(
            output / f"{split}_test.csv", index=False
        )

    registry = pd.DataFrame(
        sorted(reactions.items()), columns=["reaction_id", "reaction_smiles"]
    )
    features = []
    audits = []
    for i, row in enumerate(registry.itertuples(index=False)):
        feat, audit = encode_reaction_with_audit(
            row.reaction_smiles,
            schema,
            failure_policy="strict",
            cache_dir=None,
        )
        if feat.shape != (expected,):
            raise RuntimeError((row.reaction_id, feat.shape, expected))
        features.append(feat)
        audits.append(
            {
                "row": i,
                "reaction_id": row.reaction_id,
                "drfp_status": audit.drfp_status,
                "fallback_used": bool(audit.fallback_used),
                "warning": audit.warning,
            }
        )
    matrix = np.stack(features).astype(np.float32)
    registry.to_csv(output / "reaction_registry.csv", index=False)
    pd.DataFrame(
        {"row": range(len(registry)), "reaction_id": registry["reaction_id"]}
    ).to_csv(output / "reaction_entries.csv", index=False)
    np.save(output / "reaction_features.npy", matrix)
    audit_frame = pd.DataFrame(audits)
    audit_frame.to_csv(output / "reaction_audit.csv", index=False)
    if audit_frame["fallback_used"].any():
        raise RuntimeError("CARE reaction feature fallback detected")

    care_proteins = read_zip_csv(
        "CARE_datasets/processed_data/protein2EC_clustered50.csv"
    )
    ec_list = read_ec_list()
    general = pd.read_csv(GENERAL_SEQS, sep="\t", dtype=str).fillna("")
    seq_to_general = dict(zip(general["sequence"], general["protein_id"]))
    care_proteins["general_protein_id"] = care_proteins["Sequence"].map(seq_to_general).fillna("")
    missing = care_proteins[care_proteins["general_protein_id"].eq("")][
        ["Sequence"]
    ].drop_duplicates().reset_index(drop=True)
    missing["Entry"] = [
        f"CARE_MISSING_{i:05d}" for i in range(len(missing))
    ]
    missing[["Entry", "Sequence"]].to_csv(
        output / "missing_cluster50_proteins.tsv", sep="\t", index=False
    )
    seq_to_missing = dict(zip(missing["Sequence"], missing["Entry"]))
    care_proteins["missing_entry"] = care_proteins["Sequence"].map(seq_to_missing).fillna("")
    care_proteins.to_csv(output / "protein2EC_clustered50_mapped.csv.gz", index=False, compression="gzip")
    (output / "EC_list.txt").write_text("\n".join(ec_list) + "\n")

    summary = {
        "schema": "fibre-care-task2-prep-v1",
        "care_zip_md5": hashlib.md5(CARE_ZIP.read_bytes()).hexdigest(),
        "test_rows": {s: int(len(tests[s])) for s in SPLITS},
        "unique_test_reactions": int(len(registry)),
        "reaction_feature_dimension": expected,
        "reaction_fallback_count": int(audit_frame["fallback_used"].sum()),
        "care_cluster50_rows": int(len(care_proteins)),
        "care_cluster50_unique_sequences": int(care_proteins["Sequence"].nunique()),
        "care_ec_count": int(len(ec_list)),
        "cluster50_sequences_reused_from_broad": int(
            care_proteins.loc[care_proteins["general_protein_id"].ne(""), "Sequence"].nunique()
        ),
        "cluster50_sequences_needing_embedding": int(len(missing)),
        "ec_with_existing_broad_sequence_support": int(
            care_proteins.loc[
                care_proteins["general_protein_id"].ne(""), "EC number"
            ].nunique()
        ),
    }
    (output / "prepare_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


def load_model(device: torch.device) -> TerpeneDualTower:
    payload = torch.load(DEFAULT_INDEX, map_location="cpu", weights_only=False)
    model = TerpeneDualTower(ModelConfig(**dict(payload["model_config"])))
    model.load_state_dict(payload["model_state_dict"])
    return model.to(device).eval()


def ec_prefix(ec: str, level: int) -> str:
    parts = str(ec).split(".")
    return ".".join(parts[:level])


def metrics_for_ranks(test: pd.DataFrame, rankings: np.ndarray, ec_list: list[str]) -> dict[str, float]:
    ec_array = np.asarray(ec_list, dtype=object)
    true = test["EC number"].astype(str).to_numpy()
    positions = []
    top1 = ec_array[rankings[:, 0]]
    for i, label in enumerate(true):
        where = np.flatnonzero(ec_array[rankings[i]] == label)
        positions.append(int(where[0] + 1) if len(where) else len(ec_list) + 1)
    pos = np.asarray(positions)
    result = {
        "queries": int(len(test)),
        "candidate_ecs": int(len(ec_list)),
        "mrr": float(np.mean(1.0 / pos)),
        "mean_rank": float(pos.mean()),
        "median_rank": float(np.median(pos)),
    }
    for k in (1, 3, 5, 10, 20, 50, 100):
        result[f"hit_at_{k}"] = float(np.mean(pos <= k))
    for level in (4, 3, 2, 1):
        result[f"top1_level_{level}_accuracy"] = float(
            np.mean(
                [ec_prefix(a, level) == ec_prefix(b, level) for a, b in zip(top1, true)]
            )
        )
    return result


def score(output: Path, missing_embedding_dir: Path, device_name: str) -> None:
    device = torch.device(device_name)
    model = load_model(device)
    mapping = pd.read_csv(
        output / "protein2EC_clustered50_mapped.csv.gz", dtype=str
    ).fillna("")
    ec_list = [
        x.strip() for x in (output / "EC_list.txt").read_text().splitlines() if x.strip()
    ]
    if len(ec_list) != 4960:
        raise RuntimeError(f"CARE EC list drift: {len(ec_list)}")

    index = FibreCandidateIndex(device=device_name)
    broad_seq = pd.read_csv(GENERAL_SEQS, sep="\t", dtype=str).fillna("")
    seq_to_pid = dict(zip(broad_seq["sequence"], broad_seq["protein_id"]))

    missing_entries = pd.read_csv(missing_embedding_dir / "entries.csv", dtype=str).fillna("")
    missing_entries["row"] = pd.to_numeric(missing_entries["row"]).astype(int)
    missing_raw = np.load(missing_embedding_dir / "embeddings.npy").astype(np.float32)
    with torch.no_grad():
        missing_z = model.encode_proteins(torch.as_tensor(missing_raw, device=device))
    missing_index = {
        entry: int(row)
        for entry, row in zip(missing_entries["Entry"], missing_entries["row"])
    }

    vectors = []
    for row in mapping.itertuples(index=False):
        if row.general_protein_id:
            vectors.append(index.protein_embeddings[index.protein_index[row.general_protein_id]])
        else:
            vectors.append(missing_z[missing_index[row.missing_entry]])
    protein_z = torch.stack(vectors, dim=0)

    ec_rows: dict[str, list[int]] = {}
    for i, ec in enumerate(mapping["EC number"].astype(str)):
        ec_rows.setdefault(ec, []).append(i)
    centers = []
    zero_ec = []
    for ec in ec_list:
        rows = ec_rows.get(ec, [])
        if not rows:
            centers.append(torch.zeros(protein_z.shape[1], device=device))
            zero_ec.append(ec)
        else:
            center = protein_z[torch.as_tensor(rows, device=device)].mean(dim=0)
            centers.append(center)
    centers_t = torch.stack(centers, dim=0)
    centers_t = torch.nn.functional.normalize(centers_t, dim=1)

    rentries = pd.read_csv(output / "reaction_entries.csv", dtype=str).fillna("")
    rentries["row"] = pd.to_numeric(rentries["row"]).astype(int)
    rmat = np.load(output / "reaction_features.npy").astype(np.float32)
    with torch.no_grad():
        rz = model.encode_reactions(torch.as_tensor(rmat, device=device))
        rz = torch.nn.functional.normalize(rz, dim=1)
        sim = (rz @ centers_t.T).float().cpu().numpy()
    rid_to_row = dict(zip(rentries["reaction_id"], rentries["row"]))

    summaries = {}
    for split in SPLITS:
        test = pd.read_csv(output / f"{split}_test.csv", dtype=str).fillna("")
        score_rows = np.stack([sim[rid_to_row[rid]] for rid in test["reaction_id"]])
        rankings = np.argsort(-score_rows, axis=1, kind="stable")
        metrics = metrics_for_ranks(test, rankings, ec_list)
        summaries[split] = metrics
        top = min(100, len(ec_list))
        out = test[["reaction_id", "Reaction", "EC number"]].copy()
        for j in range(top):
            out[str(j)] = np.asarray(ec_list, dtype=object)[rankings[:, j]]
        out.to_csv(output / f"{split}_rankings_top100.csv.gz", index=False, compression="gzip")
    summary = {
        "schema": "fibre-care-task2-v1",
        "status": "completed",
        "protocol": (
            "CARE Task2 all_ECs reference: protein2EC_clustered50 representations "
            "averaged within each of 4960 ECs; cosine retrieval from reaction representation"
        ),
        "broad_model": str(DEFAULT_INDEX.relative_to(ROOT)),
        "missing_sequences_encoded": int(len(missing_entries)),
        "zero_ec_centers": zero_ec,
        "splits": summaries,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["prepare", "score"])
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--missing-embedding-dir",
        type=Path,
        default=DEFAULT_OUTPUT / "missing_cluster50_esmc",
    )
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    out = args.output_dir.resolve()
    if args.stage == "prepare":
        prepare(out)
    else:
        score(out, args.missing_embedding_dir.resolve(), args.device)


if __name__ == "__main__":
    main()
