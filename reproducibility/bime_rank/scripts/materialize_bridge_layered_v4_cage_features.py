from __future__ import annotations

import argparse
import json
import os
import pickle
import shutil
import sys
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd
import torch

from projects.active.bridge.model.assets import ROOT

SUPPORT = ROOT / "results/bridge_layered_v4_cage_support/protein_support_registry.csv.gz"
ZIP_PATH = ROOT / "data/external/enzymecage_current/authors_drive_current/dataset.zip"
OUT = ROOT / "results/bridge_layered_v4_cage_features"
AUTHOR_POCKET_PREFIX = "dataset/RHEA/2025-02-05/pockets/pocket/"
AUTHOR_POCKET_INFO = "dataset/RHEA/2025-02-05/pockets/pocket_info.csv"
CAGE_ROOT = ROOT / "external_repos/EnzymeCAGE"
FEATURE_MODULE = CAGE_ROOT / "feature"


def scope_rows(scope: str) -> pd.DataFrame:
    registry = pd.read_csv(SUPPORT, dtype=str).fillna("")
    if scope == "e2r":
        rows = registry[registry.scopes.str.contains("e2r_query", regex=False)].copy()
    elif scope == "r2e_broad":
        rows = registry[registry.scopes.str.contains("r2e_broad_candidate", regex=False)].copy()
    elif scope == "r2e_native":
        rows = registry[registry.scopes.str.contains("r2e_native_candidate", regex=False)].copy()
    elif scope == "r2e_union":
        rows = registry[
            registry.scopes.str.contains("r2e_native_candidate", regex=False)
            | registry.scopes.str.contains("r2e_broad_candidate", regex=False)
        ].copy()
    else:
        raise ValueError(scope)
    rows["existing_feature"] = rows.existing_feature.astype(str).str.lower().eq("true")
    rows["author_pocket"] = rows.author_pocket.astype(str).str.lower().eq("true")
    return rows.drop_duplicates("chosen_uid").reset_index(drop=True)


def paths(scope: str) -> dict[str, Path]:
    root = OUT / scope
    feature = root / "feature/protein"
    return {
        "root": root,
        "input": root / "input_author.csv",
        "pocket": root / "pocket",
        "pocket_info": root / "pocket_info.csv",
        "gvp": feature / "gvp_feature/gvp_protein_feature.pt",
        "node_dir": feature / "ESM-C_600M/node_level",
        "mean": feature / "ESM-C_600M/protein_level/seq2feature.pkl",
        "pocket_esm": feature / "ESM-C_600M/pocket_node_feature/esm_node_feature.pt",
        "progress": root / "esm_progress.json",
        "failures": root / "esm_failures.csv",
    }


def prepare(scope: str) -> None:
    p = paths(scope)
    for key in ("root", "pocket", "node_dir"):
        p[key].mkdir(parents=True, exist_ok=True)
    p["gvp"].parent.mkdir(parents=True, exist_ok=True)
    p["mean"].parent.mkdir(parents=True, exist_ok=True)
    p["pocket_esm"].parent.mkdir(parents=True, exist_ok=True)

    rows = scope_rows(scope)
    rows = rows[(~rows.existing_feature) & rows.author_pocket].copy()
    rows = rows.sort_values("chosen_uid", kind="stable").reset_index(drop=True)
    input_frame = pd.DataFrame(
        {
            "UniprotID": rows.chosen_uid.astype(str),
            "sequence": rows.sequence.astype(str),
            "CANO_RXN_SMILES": "CC>>CC",
        }
    )
    input_frame.to_csv(p["input"], index=False)
    wanted = set(input_frame.UniprotID.astype(str))

    extracted = 0
    skipped = 0
    with ZipFile(ZIP_PATH) as zf:
        with zf.open(AUTHOR_POCKET_INFO) as fh:
            pocket_info = pd.read_csv(fh, dtype={"UniprotID": str})
        pocket_info = pocket_info[
            pocket_info.UniprotID.astype(str).isin(wanted)
        ].copy()
        pocket_info.to_csv(p["pocket_info"], index=False)

        for uid in sorted(wanted):
            dst = p["pocket"] / f"{uid}.pdb"
            if dst.exists() and dst.stat().st_size > 0:
                skipped += 1
                continue
            src = f"{AUTHOR_POCKET_PREFIX}{uid}.pdb"
            try:
                with zf.open(src) as fi, dst.open("wb") as fo:
                    shutil.copyfileobj(fi, fo, length=1024 * 1024)
                extracted += 1
            except KeyError:
                if dst.exists():
                    dst.unlink()
                raise RuntimeError(f"author pocket declared but missing in zip: {uid}")

    missing_info = wanted - set(pocket_info.UniprotID.astype(str))
    summary = {
        "scope": scope,
        "author_feature_targets": int(len(input_frame)),
        "pocket_info_rows": int(len(pocket_info)),
        "missing_pocket_info": sorted(missing_info),
        "pockets_extracted_now": int(extracted),
        "pockets_reused": int(skipped),
    }
    (p["root"] / "prepare_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(json.dumps(summary, indent=2))


def gvp(scope: str) -> None:
    p = paths(scope)
    if not p["input"].exists():
        raise FileNotFoundError(p["input"])
    sys.path.insert(0, str(FEATURE_MODULE))
    from gvp_torchdrug_feature import calc_gvp_feature

    calc_gvp_feature(
        str(p["input"]),
        str(p["pocket"]),
        str(p["gvp"]),
        fallback_structure_dir=None,
        align_to_sequence=True,
    )


def esm(scope: str, checkpoint_every: int) -> None:
    p = paths(scope)
    if not p["input"].exists():
        raise FileNotFoundError(p["input"])
    p["node_dir"].mkdir(parents=True, exist_ok=True)
    p["mean"].parent.mkdir(parents=True, exist_ok=True)

    from esm.models.esmc import ESMC
    from esm.sdk.api import ESMProtein, LogitsConfig

    frame = pd.read_csv(p["input"], dtype=str).fillna("").drop_duplicates("UniprotID")
    uid_to_seq = dict(zip(frame.UniprotID.astype(str), frame.sequence.astype(str)))

    if p["mean"].exists():
        with p["mean"].open("rb") as fh:
            seq_to_feature = pickle.load(fh)
    else:
        seq_to_feature = {}

    existing_uids = {path.stem for path in p["node_dir"].glob("*.npz")}

    def link_or_copy(source: Path, target: Path) -> bool:
        uid = target.stem
        if uid in existing_uids:
            return False
        try:
            os.link(source, target)
        except OSError:
            shutil.copy2(source, target)
        existing_uids.add(uid)
        return True

    # Existing node files are exact reusable sequence representations.  For the
    # large R2E union, also reuse the already-materialized E2R query features.
    seq_source: dict[str, Path] = {}
    for uid, seq in uid_to_seq.items():
        if uid in existing_uids:
            seq_source.setdefault(seq, p["node_dir"] / f"{uid}.npz")

    external_reused = 0
    if scope == "r2e_union":
        e2r = paths("e2r")
        if e2r["input"].exists() and e2r["node_dir"].exists():
            e2r_frame = pd.read_csv(
                e2r["input"], dtype=str
            ).fillna("").drop_duplicates("UniprotID")
            e2r_node_uids = {
                path.stem for path in e2r["node_dir"].glob("*.npz")
            }
            for rec in e2r_frame[["UniprotID", "sequence"]].itertuples(index=False):
                if str(rec.UniprotID) in e2r_node_uids:
                    seq_source.setdefault(
                        str(rec.sequence),
                        e2r["node_dir"] / f"{rec.UniprotID}.npz",
                    )
            if e2r["mean"].exists():
                with e2r["mean"].open("rb") as fh:
                    external_mean = pickle.load(fh)
                for seq in set(uid_to_seq.values()) & set(external_mean):
                    seq_to_feature.setdefault(seq, external_mean[seq])

    pending_by_seq: dict[str, list[str]] = {}
    for uid, seq in uid_to_seq.items():
        if uid in existing_uids:
            continue
        target = p["node_dir"] / f"{uid}.npz"
        source = seq_source.get(seq)
        if source is not None:
            if link_or_copy(source, target):
                external_reused += 1
        else:
            pending_by_seq.setdefault(seq, []).append(uid)

    repaired = 0
    for uid, seq in uid_to_seq.items():
        node_path = p["node_dir"] / f"{uid}.npz"
        if uid in existing_uids and seq not in seq_to_feature:
            node = np.load(node_path)["node_feature"]
            seq_to_feature[seq] = torch.as_tensor(node).mean(dim=0)
            repaired += 1
    if repaired or external_reused:
        with p["mean"].open("wb") as fh:
            pickle.dump(seq_to_feature, fh)

    representatives = [
        (seq, uids[0], uids[1:])
        for seq, uids in pending_by_seq.items()
    ]
    already_done = len(set(uid_to_seq) & existing_uids)
    print(json.dumps({
        "scope": scope,
        "total": int(len(frame)),
        "already_done": int(already_done),
        "reused_node_files": int(external_reused),
        "unique_sequences_to_encode": int(len(representatives)),
        "uids_waiting_on_new_sequences": int(sum(1 + len(rest) for _, _, rest in representatives)),
        "repaired_means": int(repaired),
        "mode": "official_single_sequence_semantics_with_exact_sequence_reuse",
    }), flush=True)
    if not representatives:
        return

    print("loading ESM-C 600M once", flush=True)
    model = ESMC.from_pretrained("esmc_600m").to("cuda")
    model.eval()
    print("ESM-C ready", flush=True)

    failures: list[dict[str, str]] = []
    completed = already_done
    attempted_uids = 0
    for i, (seq, uid, duplicates) in enumerate(representatives, 1):
        group = [uid, *duplicates]
        attempted_uids += len(group)
        try:
            protein = ESMProtein(sequence=seq)
            with torch.no_grad():
                protein_tensor = model.encode(protein)
                output = model.logits(
                    protein_tensor,
                    LogitsConfig(sequence=True, return_embeddings=True),
                )
            assert output.embeddings is not None
            node = output.embeddings[0].detach().cpu()
            source = p["node_dir"] / f"{uid}.npz"
            np.savez_compressed(source, node_feature=node)
            existing_uids.add(uid)
            seq_to_feature[seq] = node.mean(axis=0)
            for duplicate_uid in duplicates:
                link_or_copy(
                    source,
                    p["node_dir"] / f"{duplicate_uid}.npz",
                )
            completed += len(group)
        except Exception as exc:
            for failed_uid in group:
                failures.append({
                    "UniprotID": failed_uid,
                    "sequence": seq,
                    "error": repr(exc),
                })

        if i % checkpoint_every == 0 or i == len(representatives):
            with p["mean"].open("wb") as fh:
                pickle.dump(seq_to_feature, fh)
            if failures:
                pd.DataFrame(failures).to_csv(p["failures"], index=False)
            p["progress"].write_text(json.dumps({
                "scope": scope,
                "total": int(len(frame)),
                "completed_node_files": int(completed),
                "unique_sequences_attempted": int(i),
                "uids_attempted_this_run": int(attempted_uids),
                "failures_this_run": int(len(failures)),
                "last_uid": uid,
                "mode": "official_single_sequence_semantics_with_exact_sequence_reuse",
            }, indent=2) + "\n")
            print(
                f"{scope} ESM {completed}/{len(frame)} failures={len(failures)}",
                flush=True,
            )

def pocket(scope: str) -> None:
    p = paths(scope)
    if not p["input"].exists():
        raise FileNotFoundError(p["input"])
    if not p["pocket_info"].exists():
        raise FileNotFoundError(p["pocket_info"])

    sys.path.insert(0, str(CAGE_ROOT))
    sys.path.insert(0, str(FEATURE_MODULE))
    import main as cage_feature

    cage_feature.get_esm_pocket_feature(
        str(p["input"]),
        str(p["pocket_info"]),
        str(p["node_dir"]),
        str(p["pocket_esm"]),
        str(p["pocket"]),
    )
    cage_feature.check_pocket_feature(str(p["gvp"]), str(p["pocket_esm"]))


def audit(scope: str) -> None:
    p = paths(scope)
    frame = pd.read_csv(p["input"], dtype=str).fillna("")
    expected = set(frame.UniprotID.astype(str))
    node = {x.stem for x in p["node_dir"].glob("*.npz")}
    gvp_keys: set[str] = set()
    pocket_keys: set[str] = set()
    if p["gvp"].exists():
        gvp_keys = set(map(str, torch.load(p["gvp"], map_location="cpu", weights_only=False).keys()))
    if p["pocket_esm"].exists():
        pocket_keys = set(
            map(
                str,
                torch.load(
                    p["pocket_esm"], map_location="cpu", weights_only=False
                ).keys(),
            )
        )
    common = expected & node & gvp_keys & pocket_keys
    summary = {
        "scope": scope,
        "expected": len(expected),
        "node_npz": len(expected & node),
        "gvp": len(expected & gvp_keys),
        "pocket_esm": len(expected & pocket_keys),
        "complete_common": len(common),
        "missing_complete": len(expected - common),
    }
    (p["root"] / "feature_audit.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(json.dumps(summary, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=("prepare", "gvp", "esm", "pocket", "audit"))
    ap.add_argument(
        "--scope",
        choices=("e2r", "r2e_broad", "r2e_native", "r2e_union"),
        required=True,
    )
    ap.add_argument("--checkpoint-every", type=int, default=64)
    args = ap.parse_args()

    if args.action == "prepare":
        prepare(args.scope)
    elif args.action == "gvp":
        gvp(args.scope)
    elif args.action == "esm":
        esm(args.scope, max(1, int(args.checkpoint_every)))
    elif args.action == "pocket":
        pocket(args.scope)
    elif args.action == "audit":
        audit(args.scope)


if __name__ == "__main__":
    main()
