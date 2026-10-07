from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import pickle
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from projects.active.bridge.model.assets import ROOT
from projects.active.bridge.pipelines.whole_structure import download_one

SUPPORT = ROOT / "results/bridge_layered_v4_cage_support/protein_support_registry.csv.gz"
AF_CACHE = ROOT / "results/clipzyme_native_extension_v1/structures/af_v6"
E2R_FALLBACK = ROOT / "results/bridge_layered_v4_cage_features/e2r_fallback"
OUT = ROOT / "results/bridge_layered_v4_cage_features/r2e_fallback"
STRUCTURES = OUT / "structures"
R2E_AUTHOR = ROOT / "results/bridge_layered_v4_cage_features/r2e_union"
E2R_AUTHOR = ROOT / "results/bridge_layered_v4_cage_features/e2r"
P2RANK_HOME = ROOT / "data/assets/p2rank/p2rank_2.5.1"
if not P2RANK_HOME.exists():
    P2RANK_HOME = ROOT / "external_repos/EnzymeCAGE/tools/p2rank_2.5.1"
JAVA_HOME = ROOT / "data/assets/java17/jdk-17.0.19+10-jre"
if not JAVA_HOME.exists():
    JAVA_HOME = ROOT / ".runtime/java17/usr/lib/jvm/java-17-openjdk-amd64"

CAGE_SCRIPTS = ROOT / "external_repos/EnzymeCAGE/scripts"
if str(CAGE_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(CAGE_SCRIPTS))

from extract_p2rank_pockets import (
    list_structure_files,
    load_pocket_positions,
    load_prediction_metadata,
    resolve_prediction_file,
    run_p2rank,
    save_pocket_structure,
)


def fallback_rows() -> pd.DataFrame:
    reg = pd.read_csv(SUPPORT, dtype=str).fillna("")
    rows = reg[
        reg.scopes.str.contains("r2e_", regex=False)
        & reg.support_source.eq("fallback_structure")
    ][["entity_kind", "entity_id", "chosen_uid", "sequence", "scopes"]].copy()
    rows = rows[rows.chosen_uid.ne("") & rows.sequence.ne("")].copy()
    rows["sequence_length"] = rows.sequence.astype(str).str.len()
    rows["source_priority"] = rows.entity_kind.map({"general_merged": 0}).fillna(1).astype(int)
    rows = rows.sort_values(
        ["chosen_uid", "source_priority", "sequence_length", "entity_id"],
        ascending=[True, True, False, True],
        kind="stable",
    )
    sequence_counts = rows.groupby("chosen_uid").sequence.nunique()
    conflict_uids = set(sequence_counts[sequence_counts > 1].index.astype(str))
    if conflict_uids:
        OUT.mkdir(parents=True, exist_ok=True)
        audit = rows[rows.chosen_uid.astype(str).isin(conflict_uids)].copy()
        first = rows.drop_duplicates("chosen_uid", keep="first")[["chosen_uid", "sequence"]]
        selected = dict(zip(first.chosen_uid.astype(str), first.sequence.astype(str)))
        audit["selected_for_canonical_structure"] = [
            str(row.sequence) == selected[str(row.chosen_uid)]
            for row in audit.itertuples(index=False)
        ]
        audit.to_csv(OUT / "sequence_conflict_audit.csv", index=False)
    return (
        rows.drop_duplicates("chosen_uid", keep="first")
        .drop(columns=["sequence_length", "source_priority"])
        .reset_index(drop=True)
    )


def reuse_file(source: Path, target: Path) -> None:
    if target.exists() and target.stat().st_size > 0:
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, target)
    except OSError:
        try:
            target.symlink_to(source)
        except OSError:
            shutil.copy2(source, target)


def download(workers: int) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    STRUCTURES.mkdir(parents=True, exist_ok=True)
    rows = fallback_rows()
    rows.rename(columns={"chosen_uid": "UniprotID"}).to_csv(OUT / "input_all.csv", index=False)

    manifest: list[tuple[str, str, str, bool, str]] = []
    pending: list[tuple[str, str, Path]] = []
    e2r_structures = E2R_FALLBACK / "structures"

    for row in rows.itertuples(index=False):
        uid = str(row.chosen_uid)
        target = STRUCTURES / f"{uid}.cif"
        local = AF_CACHE / f"AF-{uid}-F1-model_v6.cif"
        e2r = e2r_structures / f"{uid}.cif"
        if target.exists() and target.stat().st_size > 0:
            manifest.append((uid, "r2e_cache", str(target), True, "cached"))
        elif local.exists() and local.stat().st_size > 0:
            reuse_file(local, target)
            manifest.append((uid, "existing_af_v6", str(target), True, "reused"))
        elif e2r.exists() and e2r.stat().st_size > 0:
            reuse_file(e2r, target)
            manifest.append((uid, "e2r_fallback_reuse", str(target), True, "reused"))
        else:
            url = f"https://alphafold.ebi.ac.uk/files/AF-{uid}-F1-model_v6.cif"
            pending.append((uid, url, target))

    def one(item: tuple[str, str, Path]) -> tuple[str, str, str, bool, str]:
        uid, url, target = item
        path, ok, message = download_one(url, target)
        return uid, "afdb_v6", str(path), bool(ok), str(message)

    if pending:
        with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
            futures = [ex.submit(one, item) for item in pending]
            for future in concurrent.futures.as_completed(futures):
                manifest.append(future.result())

    mf = pd.DataFrame(
        manifest,
        columns=["UniprotID", "source", "structure_path", "ok", "message"],
    ).sort_values("UniprotID", kind="stable")
    mf.to_csv(OUT / "structure_manifest.csv", index=False)
    ok = mf.ok.astype(str).str.lower().eq("true")
    summary = {
        "schema": "bridge-layered-v4-r2e-fallback-structures",
        "fallback_uids": int(len(rows)),
        "structures_ok": int(ok.sum()),
        "structures_failed": int((~ok).sum()),
        "source_counts": {str(k): int(v) for k, v in mf.source.value_counts().items()},
    }
    (OUT / "structure_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))



def extract_one_pocket(args: tuple[str, str, str, str]) -> tuple[bool, dict[str, object]]:
    uid, structure_path_s, raw_output_s, pocket_dir_s = args
    structure_path = Path(structure_path_s)
    raw_output = Path(raw_output_s)
    pocket_dir = Path(pocket_dir_s)
    prediction_dir = raw_output / "predictions"
    if not prediction_dir.exists():
        prediction_dir = raw_output
    try:
        residue_csv = resolve_prediction_file(prediction_dir, uid, "_residues.csv")
        prediction_csv = resolve_prediction_file(prediction_dir, uid, "_predictions.csv")
        pocket_residue_ids, selected_keys = load_pocket_positions(
            residue_csv, structure_path, 1
        )
        pocket_path = pocket_dir / f"{uid}.pdb"
        save_pocket_structure(structure_path, selected_keys, pocket_path)
        row: dict[str, object] = {
            "UniprotID": uid,
            "pocket_residues": ",".join(pocket_residue_ids),
            "structure_path": str(structure_path),
            "pocket_path": str(pocket_path.resolve()),
            "pocket_rank": 1,
        }
        row.update(load_prediction_metadata(prediction_csv, 1))
        return True, row
    except Exception as exc:
        return False, {"UniprotID": uid, "error": str(exc)}


def extract_parallel(workers: int) -> None:
    rows = fallback_rows()
    mf = pd.read_csv(OUT / "structure_manifest.csv", dtype=str).fillna("")
    allowed = set(
        mf.loc[mf.ok.astype(str).str.lower().eq("true"), "UniprotID"].astype(str)
    )
    structures = list_structure_files(STRUCTURES, allowed)
    p2out = OUT / "p2rank"
    raw = p2out / "raw"
    pocket_dir = p2out / "pocket"
    pocket_dir.mkdir(parents=True, exist_ok=True)
    jobs = [
        (uid, str(path), str(raw), str(pocket_dir))
        for uid, path in structures.items()
    ]
    good_rows: list[dict[str, object]] = []
    failed_rows: list[dict[str, object]] = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=max(1, workers)) as ex:
        futures = [ex.submit(extract_one_pocket, job) for job in jobs]
        done = 0
        for future in concurrent.futures.as_completed(futures):
            ok, row = future.result()
            (good_rows if ok else failed_rows).append(row)
            done += 1
            if done % 250 == 0 or done == len(jobs):
                print(f"pocket-extract {done}/{len(jobs)}", flush=True)

    pocket_info = pd.DataFrame(good_rows).sort_values("UniprotID", kind="stable")
    pocket_info.to_csv(p2out / "pocket_info.csv", index=False)
    failed = pd.DataFrame(failed_rows)
    failed.to_csv(p2out / "failed_p2rank_pockets.csv", index=False)

    good = set(pocket_info.UniprotID.astype(str))
    input_frame = rows[rows.chosen_uid.astype(str).isin(allowed)].copy()
    input_frame["UniprotID"] = input_frame.chosen_uid.astype(str)
    input_frame[["UniprotID", "sequence", "entity_kind", "entity_id", "scopes"]].to_csv(
        OUT / "input_structured.csv", index=False
    )
    feature_input = input_frame[input_frame.UniprotID.astype(str).isin(good)].copy()
    feature_input[["UniprotID", "sequence", "entity_kind", "entity_id", "scopes"]].to_csv(
        OUT / "input_feature_supported.csv", index=False
    )
    summary = {
        "schema": "bridge-layered-v4-r2e-fallback-p2rank",
        "structured_uids": int(len(input_frame)),
        "pocket_uids": int(len(good)),
        "pocket_failures": int(len(input_frame) - len(good)),
    }
    (OUT / "p2rank_summary.json").write_text(json.dumps(summary, indent=2) + "\
")
    print(json.dumps(summary, indent=2))

def p2rank(threads: int) -> None:
    rows = fallback_rows()
    mf = pd.read_csv(OUT / "structure_manifest.csv", dtype=str).fillna("")
    allowed = set(
        mf.loc[mf.ok.astype(str).str.lower().eq("true"), "UniprotID"].astype(str)
    )
    input_frame = rows[rows.chosen_uid.astype(str).isin(allowed)].copy()
    input_frame["UniprotID"] = input_frame.chosen_uid.astype(str)
    input_frame[["UniprotID", "sequence", "entity_kind", "entity_id", "scopes"]].to_csv(
        OUT / "input_structured.csv", index=False
    )

    structures = list_structure_files(STRUCTURES, allowed)
    p2out = OUT / "p2rank"
    p2out.mkdir(parents=True, exist_ok=True)
    run_p2rank(
        structures,
        P2RANK_HOME,
        p2out,
        threads=max(1, threads),
        java_home=str(JAVA_HOME),
    )
    extract_parallel(min(max(1, threads), 16))



def seed_esm() -> None:
    input_path = OUT / "input_feature_supported.csv"
    frame = pd.read_csv(input_path, dtype=str).fillna("")
    node_dir = OUT / "feature/protein/ESM-C_600M/node_level"
    mean_path = OUT / "feature/protein/ESM-C_600M/protein_level/seq2feature.pkl"
    node_dir.mkdir(parents=True, exist_ok=True)
    mean_path.parent.mkdir(parents=True, exist_ok=True)

    registry = pd.read_csv(SUPPORT, dtype=str).fillna("")
    author = registry[
        registry.support_source.eq("author_pocket") & registry.sequence.ne("")
    ][["chosen_uid", "sequence"]].drop_duplicates()
    seq_sources: dict[str, list[str]] = {}
    for row in author.itertuples(index=False):
        seq_sources.setdefault(str(row.sequence), []).append(str(row.chosen_uid))

    e2r_input = E2R_FALLBACK / "input_feature_supported.csv"
    e2r_seq = {}
    if e2r_input.exists():
        tmp = pd.read_csv(e2r_input, dtype=str).fillna("")
        e2r_seq = dict(zip(tmp.UniprotID.astype(str), tmp.sequence.astype(str)))

    source_dirs = [
        R2E_AUTHOR / "feature/protein/ESM-C_600M/node_level",
        E2R_AUTHOR / "feature/protein/ESM-C_600M/node_level",
    ]
    e2r_node_dir = E2R_FALLBACK / "feature/protein/ESM-C_600M/node_level"
    reused_records = []
    mean: dict[str, object] = {}
    if mean_path.exists():
        with mean_path.open("rb") as handle:
            mean = pickle.load(handle)

    for row in frame.itertuples(index=False):
        uid = str(row.UniprotID)
        seq = str(row.sequence)
        dst = node_dir / f"{uid}.npz"
        source = None
        source_kind = ""
        if dst.exists():
            source = dst
            source_kind = "existing"
        else:
            same_uid = e2r_node_dir / f"{uid}.npz"
            if same_uid.exists() and e2r_seq.get(uid) == seq:
                source = same_uid
                source_kind = "e2r_fallback_same_uid"
            if source is None:
                for src_uid in seq_sources.get(seq, []):
                    found = None
                    for directory in source_dirs:
                        candidate = directory / f"{src_uid}.npz"
                        if candidate.exists():
                            found = candidate
                            break
                    if found is not None:
                        source = found
                        source_kind = "author_exact_sequence"
                        break
            if source is not None:
                reuse_file(source, dst)
        if dst.exists():
            arr = np.load(dst)["node_feature"]
            expected_nodes = len(seq) + 2  # ESM-C saves BOS + residues + EOS
            if int(arr.shape[0]) != expected_nodes:
                dst.unlink()
                reused_records.append(
                    (uid, source_kind, False, f"length:{arr.shape[0]}!={expected_nodes}")
                )
                continue
            if seq not in mean:
                mean[seq] = torch.as_tensor(arr).mean(dim=0)
            reused_records.append((uid, source_kind, True, ""))

    with mean_path.open("wb") as handle:
        pickle.dump(mean, handle)
    audit = pd.DataFrame(
        reused_records, columns=["UniprotID", "source", "ok", "reason"]
    )
    audit.to_csv(OUT / "esm_seed_audit.csv", index=False)
    summary = {
        "schema": "bridge-layered-v4-r2e-fallback-esm-seed",
        "input_uids": int(len(frame)),
        "seeded_or_existing": int(audit.ok.astype(bool).sum()) if len(audit) else 0,
        "source_counts": {str(k): int(v) for k, v in audit[audit.ok.astype(bool)].source.value_counts().items()} if len(audit) else {},
    }
    (OUT / "esm_seed_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=("download", "p2rank", "extract", "seed-esm"))
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--threads", type=int, default=32)
    args = ap.parse_args()
    if args.action == "download":
        download(args.workers)
    elif args.action == "p2rank":
        p2rank(args.threads)
    elif args.action == "extract":
        extract_parallel(min(max(1, args.threads), 16))
    else:
        seed_esm()


if __name__ == "__main__":
    main()
