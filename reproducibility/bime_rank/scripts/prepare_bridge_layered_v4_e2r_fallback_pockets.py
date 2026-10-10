from __future__ import annotations

import argparse
import concurrent.futures
import os
import shutil
import sys
from pathlib import Path

import pandas as pd

from projects.active.bridge.model.assets import ROOT
from projects.active.bridge.pipelines.whole_structure import download_one

SUPPORT = ROOT / "results/bridge_layered_v4_cage_support/protein_support_registry.csv.gz"
AF_CACHE = ROOT / "results/clipzyme_native_extension_v1/structures/af_v6"
OUT = ROOT / "results/bridge_layered_v4_cage_features/e2r_fallback"
STRUCTURES = OUT / "structures"
MAIN_E2R = ROOT / "results/bridge_layered_v4_cage_features/e2r"
P2RANK_HOME = ROOT / "data/assets/p2rank/p2rank_2.5.1"
if not P2RANK_HOME.exists():
    P2RANK_HOME = ROOT / "external_repos/EnzymeCAGE/tools/p2rank_2.5.1"

CAGE_SCRIPTS = ROOT / "external_repos/EnzymeCAGE/scripts"
if str(CAGE_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(CAGE_SCRIPTS))

from extract_p2rank_pockets import extract_pockets, list_structure_files, run_p2rank


def fallback_rows() -> pd.DataFrame:
    reg = pd.read_csv(SUPPORT, dtype=str).fillna("")
    rows = reg[
        reg.scopes.str.contains("e2r_query", regex=False)
        & reg.support_source.eq("fallback_structure")
    ][["entity_id", "chosen_uid", "sequence"]].copy()
    return rows.drop_duplicates("chosen_uid").reset_index(drop=True)


def download(workers: int) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    STRUCTURES.mkdir(parents=True, exist_ok=True)
    rows = fallback_rows()
    manifest = []
    pending = []

    for row in rows.itertuples(index=False):
        uid = str(row.chosen_uid)
        target = STRUCTURES / f"{uid}.cif"
        cached = AF_CACHE / f"AF-{uid}-F1-model_v6.cif"
        if target.exists() and target.stat().st_size > 0:
            manifest.append((uid, "download_cache", str(target), True, "cached"))
        elif cached.exists() and cached.stat().st_size > 0:
            try:
                os.link(cached, target)
            except OSError:
                shutil.copy2(cached, target)
            manifest.append((uid, "existing_af_v6", str(target), True, "cached"))
        else:
            url = f"https://alphafold.ebi.ac.uk/files/AF-{uid}-F1-model_v6.cif"
            pending.append((uid, url, target))

    def one(item):
        uid, url, target = item
        path, ok, message = download_one(url, target)
        return uid, "afdb_v6", path, ok, message

    if pending:
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
            futures = [ex.submit(one, item) for item in pending]
            for future in concurrent.futures.as_completed(futures):
                manifest.append(future.result())

    mf = pd.DataFrame(
        manifest,
        columns=["chosen_uid", "source", "structure_path", "ok", "message"],
    ).sort_values("chosen_uid", kind="stable")
    mf.to_csv(OUT / "structure_manifest.csv", index=False)
    ok = mf[mf.ok.astype(bool)]
    print({
        "fallback_queries": len(rows),
        "structures_ok": len(ok),
        "structures_failed": len(mf) - len(ok),
        "existing_af_v6": int(mf.source.eq("existing_af_v6").sum()),
        "downloaded_or_cached": int(mf.source.isin(["afdb_v6", "download_cache"]).sum()),
    })


def p2rank(threads: int) -> None:
    rows = fallback_rows()
    mf = pd.read_csv(OUT / "structure_manifest.csv", dtype=str).fillna("")
    ok = mf[mf.ok.astype(str).str.lower().eq("true")]
    allowed = set(ok.chosen_uid.astype(str))
    input_csv = OUT / "input.csv"
    rows[rows.chosen_uid.astype(str).isin(allowed)].rename(
        columns={"chosen_uid": "UniprotID"}
    )[["UniprotID", "sequence"]].to_csv(input_csv, index=False)

    structures = list_structure_files(STRUCTURES, allowed)
    raw = run_p2rank(
        structures,
        P2RANK_HOME,
        OUT,
        threads=threads,
        java_home=None,
    )
    extract_pockets(structures, raw, OUT, pocket_rank=1)


def merge() -> None:
    source_input = pd.read_csv(OUT / "input.csv", dtype=str).fillna("")
    pocket_info = pd.read_csv(OUT / "pocket_info.csv", dtype=str).fillna("")
    good = set(pocket_info.UniprotID.astype(str))
    source_input = source_input[source_input.UniprotID.astype(str).isin(good)]

    main_input = pd.read_csv(MAIN_E2R / "input_author.csv", dtype=str).fillna("")
    if "CANO_RXN_SMILES" not in source_input.columns:
        source_input["CANO_RXN_SMILES"] = "CC>>CC"
    merged_input = pd.concat([main_input, source_input], ignore_index=True)
    merged_input = merged_input.drop_duplicates("UniprotID", keep="first")
    merged_input.to_csv(MAIN_E2R / "input_author.csv", index=False)

    main_info = pd.read_csv(MAIN_E2R / "pocket_info.csv", dtype=str).fillna("")
    merged_info = pd.concat([main_info, pocket_info], ignore_index=True)
    merged_info = merged_info.drop_duplicates("UniprotID", keep="first")
    merged_info.to_csv(MAIN_E2R / "pocket_info.csv", index=False)

    main_pocket = MAIN_E2R / "pocket"
    main_pocket.mkdir(parents=True, exist_ok=True)
    copied = 0
    for uid in sorted(good):
        src = OUT / "pocket" / f"{uid}.pdb"
        dst = main_pocket / f"{uid}.pdb"
        if dst.exists():
            continue
        try:
            os.link(src, dst)
        except OSError:
            shutil.copy2(src, dst)
        copied += 1

    print({
        "fallback_pockets": len(good),
        "merged_input_queries": len(merged_input),
        "merged_pocket_info": len(merged_info),
        "pocket_files_added": copied,
    })


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=("download", "p2rank", "merge"))
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--threads", type=int, default=32)
    args = ap.parse_args()
    if args.action == "download":
        download(max(1, args.workers))
    elif args.action == "p2rank":
        p2rank(max(1, args.threads))
    else:
        merge()


if __name__ == "__main__":
    main()
