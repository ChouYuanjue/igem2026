from __future__ import annotations

import argparse
import contextlib
import json
import pickle
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from Bio import PDB

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_PAIRS = ROOT / "data/external/enzymecage_current/Enzyme-405.csv"
DEFAULT_POCKET_INFO = ROOT / "data/external/enzymecage_current/minimal_405/pockets/pocket_info.csv"
DEFAULT_POCKET_DIR = ROOT / "data/external/enzymecage_current/minimal_405/pockets/pocket"
DEFAULT_GVP = ROOT / "data/external/enzymecage_current/cage_official_features/gvp_feature/gvp_protein_feature.pt"
DEFAULT_REUSE_NODE_DIR = ROOT / "reports/zz_model_workflow_20260817/native_cage/data/feature/protein/ESM-C_600M/node_level"
DEFAULT_OUTPUT = ROOT / "data/external/enzymecage_current/cage_official_features/ESM-C_600M_minimal"
VALID_AA = re.compile(r"^[ACDEFGHIKLMNPQRSTVWYBXZJUO]+$")
THREE_TO_ONE = {
    "ALA": "A", "CYS": "C", "ASP": "D", "GLU": "E", "PHE": "F", "GLY": "G", "HIS": "H",
    "ILE": "I", "LYS": "K", "LEU": "L", "MET": "M", "ASN": "N", "PRO": "P", "GLN": "Q",
    "ARG": "R", "SER": "S", "THR": "T", "VAL": "V", "TRP": "W", "TYR": "Y",
}


def clean_sequence(value: object) -> str:
    return "".join(str(value).upper().split()).rstrip("*")


def parse_pocket_residue_ids(value: object) -> list[int]:
    if pd.isna(value):
        return []
    raw = str(value).strip().strip("[](){}")
    result: list[int] = []
    for token in raw.split(","):
        token = token.strip().strip("'\"")
        if token:
            result.append(int(float(token)))
    return result


def load_pocket_residue_records(path: Path) -> list[tuple[int, str]]:
    structure = PDB.PDBParser(QUIET=True).get_structure("pocket", str(path))
    records: list[tuple[int, str]] = []
    seen: set[tuple[int, str]] = set()
    for residue in structure.get_residues():
        hetero, residue_number, insertion_code = residue.id
        if hetero != " ":
            continue
        aa = THREE_TO_ONE.get(residue.resname)
        if aa is None:
            continue
        key = (int(residue_number), str(insertion_code))
        if key in seen:
            continue
        seen.add(key)
        records.append((int(residue_number), aa))
    return records


def infer_sequence_indices(full_sequence: str, residue_records: list[tuple[int, str]], pocket_ids: list[int]) -> tuple[list[int], dict[str, int]]:
    sequence = clean_sequence(full_sequence)
    if not sequence:
        raise ValueError("empty full sequence")
    id_to_aa: dict[int, str] = {}
    for residue_number, aa in residue_records:
        id_to_aa.setdefault(int(residue_number), aa)
    matched = [(rid, id_to_aa[rid]) for rid in pocket_ids if rid in id_to_aa]
    if not matched:
        raise ValueError("no pocket residue ids can be matched to the pocket PDB")

    direct = all(
        0 <= residue_number - 1 < len(sequence) and sequence[residue_number - 1] == aa
        for residue_number, aa in residue_records
    )
    if direct:
        indices = [rid - 1 for rid in pocket_ids]
        if all(0 <= i < len(sequence) for i in indices):
            return indices, {"offset": 1, "matches": len(matched), "mismatches": 0, "out_of_range": 0, "all_out_of_range": 0}

    aa_positions: dict[str, list[int]] = {}
    for i, aa in enumerate(sequence):
        aa_positions.setdefault(aa, []).append(i)
    offsets = {0, 1}
    for residue_number, aa in matched:
        offsets.update(residue_number - i for i in aa_positions.get(aa, []))
    scored: list[dict[str, int]] = []
    for offset in offsets:
        matches = mismatches = out_of_range = 0
        for residue_number, aa in matched:
            i = residue_number - offset
            if i < 0 or i >= len(sequence):
                out_of_range += 1
            elif sequence[i] == aa:
                matches += 1
            else:
                mismatches += 1
        all_indices = [rid - offset for rid in pocket_ids]
        scored.append({
            "offset": int(offset), "matches": matches, "mismatches": mismatches, "out_of_range": out_of_range,
            "all_out_of_range": sum(i < 0 or i >= len(sequence) for i in all_indices),
        })
    scored.sort(key=lambda x: (-x["matches"], x["mismatches"], x["all_out_of_range"], abs(x["offset"] - 1)))
    best = scored[0]
    if best["matches"] == 0:
        raise ValueError("could not infer a residue-number offset")
    if len(scored) > 1:
        a = (best["matches"], best["mismatches"], best["all_out_of_range"])
        b = (scored[1]["matches"], scored[1]["mismatches"], scored[1]["all_out_of_range"])
        if a == b:
            raise ValueError(f"ambiguous residue-number offset: {best['offset']} vs {scored[1]['offset']}")
    indices = [rid - best["offset"] for rid in pocket_ids]
    if any(i < 0 or i >= len(sequence) for i in indices):
        raise ValueError(f"inferred offset {best['offset']} leaves pocket indices out of range")
    return indices, best


def author_exact_from_full_embedding(full_embedding: np.ndarray, sequence_length: int, sequence_indices: list[int]) -> tuple[np.ndarray, np.ndarray]:
    """Reproduce EnzymeCAGE's published feature code, including BOS/EOS semantics."""
    values = np.asarray(full_embedding, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != 1152:
        raise ValueError(f"unexpected ESM-C embedding shape {values.shape}")
    if values.shape[0] != sequence_length + 2:
        raise ValueError(f"author-exact embedding must have L+2 rows, got {values.shape[0]} for L={sequence_length}")
    # This intentionally mirrors feature/main.py: mean over all L+2 rows and
    # index the L+2 tensor using sequence indices without a +1 BOS correction.
    return values.mean(axis=0, dtype=np.float32), values[np.asarray(sequence_indices, dtype=np.int64)]


def residue_aligned_from_full_embedding(full_embedding: np.ndarray, sequence_length: int, sequence_indices: list[int]) -> tuple[np.ndarray, np.ndarray]:
    """Sensitivity-analysis semantics with BOS/EOS removed and residue indices aligned."""
    values = np.asarray(full_embedding, dtype=np.float32)
    if values.shape != (sequence_length + 2, 1152):
        raise ValueError(f"expected {(sequence_length + 2, 1152)}, got {values.shape}")
    residues = values[1:-1]
    return residues.mean(axis=0, dtype=np.float32), residues[np.asarray(sequence_indices, dtype=np.int64)]


def build_target_table(pairs: Path, pocket_info: Path, pocket_dir: Path, valid_uids: set[str]) -> pd.DataFrame:
    data = pd.read_csv(pairs, dtype=str).fillna("")
    required = {"UniprotID", "sequence"}
    if not required <= set(data.columns):
        raise ValueError(f"pair table missing {sorted(required - set(data.columns))}")
    data = data[["UniprotID", "sequence"]].drop_duplicates("UniprotID", keep="first")
    data["UniprotID"] = data["UniprotID"].astype(str)
    data["sequence"] = data["sequence"].map(clean_sequence)
    data = data[data["UniprotID"].isin(valid_uids)].copy()
    pockets = pd.read_csv(pocket_info, dtype=str).fillna("")[["UniprotID", "pocket_residues"]]
    pockets["UniprotID"] = pockets["UniprotID"].astype(str)
    data = data.merge(pockets, on="UniprotID", how="left", validate="one_to_one")
    data["pdb_path"] = data["UniprotID"].map(lambda u: str((pocket_dir / f"{u}.pdb").resolve()))
    if data["pocket_residues"].eq("").any():
        raise ValueError("one or more valid GVP UIDs lack pocket_residues")
    missing_pdb = data.loc[~data["pdb_path"].map(lambda x: Path(x).exists()), "UniprotID"].tolist()
    if missing_pdb:
        raise ValueError(f"missing pocket PDBs for {len(missing_pdb)} UIDs; sample={missing_pdb[:5]}")
    invalid_seq = data.loc[~data["sequence"].map(lambda s: bool(s) and bool(VALID_AA.match(s))), "UniprotID"].tolist()
    if invalid_seq:
        raise ValueError(f"invalid sequences for {len(invalid_seq)} UIDs; sample={invalid_seq[:5]}")
    return data.sort_values("UniprotID").reset_index(drop=True)


def build_mapping_audit(targets: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for row in targets.itertuples(index=False):
        uid = str(row.UniprotID); sequence = str(row.sequence)
        ids = parse_pocket_residue_ids(row.pocket_residues)
        try:
            records = load_pocket_residue_records(Path(row.pdb_path))
            indices, best = infer_sequence_indices(sequence, records, ids)
            rows.append({
                "UniprotID": uid, "sequence_length": len(sequence), "pocket_nodes": len(ids), "mapping_ok": True,
                "offset": best["offset"], "matches": best["matches"], "mismatches": best["mismatches"],
                "min_sequence_index": min(indices), "max_sequence_index": max(indices), "error": "",
                "sequence_indices": ",".join(map(str, indices)),
            })
        except Exception as exc:
            rows.append({
                "UniprotID": uid, "sequence_length": len(sequence), "pocket_nodes": len(ids), "mapping_ok": False,
                "offset": None, "matches": None, "mismatches": None, "min_sequence_index": None,
                "max_sequence_index": None, "error": repr(exc), "sequence_indices": "",
            })
    return pd.DataFrame(rows)


def batched_full_embeddings(model, sequences: list[str]) -> list[np.ndarray]:
    """Exact ESMC.forward semantics, batched with padding; returns each unpadded L+2 tensor."""
    if not sequences:
        return []
    tokens = model._tokenize(sequences)
    pad = model.tokenizer.pad_token_id
    device = next(model.parameters()).device
    with torch.no_grad(), (torch.autocast(device_type=device.type, dtype=torch.bfloat16) if device.type == "cuda" else contextlib.nullcontext()):
        output = model.forward(sequence_tokens=tokens)
        embeddings = output.embeddings.float()
    result: list[np.ndarray] = []
    for i, sequence in enumerate(sequences):
        expected = len(sequence) + 2
        nonpad = int(tokens[i].ne(pad).sum().item())
        if nonpad != expected:
            raise ValueError(f"token count {nonpad} != L+2 {expected} for sequence length {len(sequence)}")
        result.append(embeddings[i, :expected].cpu().numpy().astype(np.float32, copy=False))
    return result


def reference_full_embedding(model, sequence: str) -> np.ndarray:
    from esm.sdk.api import ESMProtein, LogitsConfig
    with torch.no_grad():
        tensor = model.encode(ESMProtein(sequence=sequence))
        output = model.logits(tensor, LogitsConfig(sequence=True, return_embeddings=True))
    return output.embeddings[0].float().cpu().numpy().astype(np.float32, copy=False)


def build_batches(rows: list[tuple[str, str]], max_batch_tokens: int, max_batch_size: int) -> list[list[tuple[str, str]]]:
    ordered = sorted(rows, key=lambda x: (len(x[1]), x[0]))
    batches: list[list[tuple[str, str]]] = []; current: list[tuple[str, str]] = []; max_len = 0
    for item in ordered:
        proposed_len = max(max_len, len(item[1]) + 2); proposed_n = len(current) + 1
        if current and (proposed_n > max_batch_size or proposed_len * proposed_n > max_batch_tokens):
            batches.append(current); current = []; max_len = 0
        current.append(item); max_len = max(max_len, len(item[1]) + 2)
    if current:
        batches.append(current)
    return batches


def collate(output_dir: Path, targets: pd.DataFrame, semantics: str) -> dict[str, object]:
    per_uid = output_dir / "per_uid"
    mean_by_sequence: dict[str, torch.Tensor] = {}
    pocket_by_uid: dict[str, np.ndarray] = {}
    missing: list[str] = []
    for row in targets.itertuples(index=False):
        path = per_uid / f"{row.UniprotID}.npz"
        if not path.exists():
            missing.append(str(row.UniprotID)); continue
        x = np.load(path)
        mean = np.asarray(x["mean_feature"], dtype=np.float32)
        pocket = np.asarray(x["pocket_node_feature"], dtype=np.float32)
        if mean.shape != (1152,):
            raise ValueError(f"bad mean shape for {row.UniprotID}: {mean.shape}")
        if pocket.ndim != 2 or pocket.shape[1] != 1152:
            raise ValueError(f"bad pocket shape for {row.UniprotID}: {pocket.shape}")
        mean_by_sequence[str(row.sequence)] = torch.from_numpy(mean.copy())
        pocket_by_uid[str(row.UniprotID)] = pocket
    protein_dir = output_dir / "protein_level"; pocket_dir = output_dir / "pocket_node_feature"
    protein_dir.mkdir(parents=True, exist_ok=True); pocket_dir.mkdir(parents=True, exist_ok=True)
    with (protein_dir / "seq2feature.pkl").open("wb") as handle:
        pickle.dump(mean_by_sequence, handle)
    torch.save(pocket_by_uid, pocket_dir / "esm_node_feature.pt")
    return {"semantics": semantics, "mean_sequences": len(mean_by_sequence), "pocket_uids": len(pocket_by_uid), "missing_uids": missing}


def main() -> None:
    parser = argparse.ArgumentParser(description="Resumable minimal ESM-C features for exact EnzymeCAGE-style inference.")
    parser.add_argument("--pairs", type=Path, default=DEFAULT_PAIRS)
    parser.add_argument("--pocket-info", type=Path, default=DEFAULT_POCKET_INFO)
    parser.add_argument("--pocket-dir", type=Path, default=DEFAULT_POCKET_DIR)
    parser.add_argument("--gvp", type=Path, default=DEFAULT_GVP)
    parser.add_argument("--reuse-node-dir", type=Path, default=DEFAULT_REUSE_NODE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--semantics", choices=["author_exact", "residue_aligned"], default="author_exact")
    parser.add_argument("--model", default="esmc_600m")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-batch-tokens", type=int, default=4096)
    parser.add_argument("--max-batch-size", type=int, default=16)
    parser.add_argument("--max-new", type=int)
    parser.add_argument("--mapping-only", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--verify-reference", type=int, default=0, help="Compare this many generated embeddings against encode->logits reference.")
    args = parser.parse_args()

    output = args.output_dir.resolve(); per_uid = output / "per_uid"; per_uid.mkdir(parents=True, exist_ok=True)
    gvp = torch.load(args.gvp.resolve(), map_location="cpu", weights_only=False)
    if not isinstance(gvp, dict):
        raise ValueError("GVP file must be a UID mapping")
    targets = build_target_table(args.pairs.resolve(), args.pocket_info.resolve(), args.pocket_dir.resolve(), set(map(str, gvp)))
    mapping_path = output / "mapping_audit.csv"
    mapping = build_mapping_audit(targets)
    mapping.to_csv(mapping_path, index=False)
    bad = mapping[~mapping["mapping_ok"]].copy()
    good_uids = set(mapping.loc[mapping["mapping_ok"], "UniprotID"].astype(str))
    usable_targets = targets[targets["UniprotID"].astype(str).isin(good_uids)].copy().reset_index(drop=True)
    if args.mapping_only:
        print(json.dumps({
            "targets": len(targets),
            "mapping_ok": int(mapping.mapping_ok.sum()),
            "mapping_failed": int(len(bad)),
            "offset_counts": {str(k): int(v) for k, v in mapping.loc[mapping.mapping_ok, "offset"].value_counts().items()},
            "failure_examples": bad[["UniprotID", "error"]].head(10).to_dict("records"),
        }, indent=2))
        return
    map_index = mapping.set_index("UniprotID")

    pending: list[tuple[str, str]] = []
    reused = 0; stale_reuse = 0
    transformer = author_exact_from_full_embedding if args.semantics == "author_exact" else residue_aligned_from_full_embedding
    for row in usable_targets.itertuples(index=False):
        uid = str(row.UniprotID); sequence = str(row.sequence); dest = per_uid / f"{uid}.npz"
        if dest.exists() and not args.force:
            continue
        indices = [int(x) for x in str(map_index.loc[uid, "sequence_indices"]).split(",") if x != ""]
        reuse = args.reuse_node_dir.resolve() / f"{uid}.npz"
        if reuse.exists() and not args.force:
            full = np.load(reuse)["node_feature"]
            if full.shape == (len(sequence) + 2, 1152):
                mean, pocket = transformer(full, len(sequence), indices)
                np.savez_compressed(dest, mean_feature=mean, pocket_node_feature=pocket, source=np.asarray("reused_author_node_cache"))
                reused += 1
                continue
            stale_reuse += 1
        pending.append((uid, sequence))
    if args.max_new is not None:
        pending = pending[: max(0, args.max_new)]

    model = None
    generated = 0; verified = 0; failures: list[dict[str, object]] = []
    if pending:
        from esm.models.esmc import ESMC
        model = ESMC.from_pretrained(args.model).eval().to(args.device)
    for batch in build_batches(pending, args.max_batch_tokens, args.max_batch_size):
        try:
            assert model is not None
            full_batch = batched_full_embeddings(model, [sequence for _, sequence in batch])
            for (uid, sequence), full in zip(batch, full_batch):
                if verified < args.verify_reference:
                    reference = reference_full_embedding(model, sequence)
                    error = float(np.max(np.abs(full - reference)))
                    if error > 2e-5:
                        raise ValueError(f"batched/reference ESM-C mismatch for {uid}: max_abs={error}")
                    verified += 1
                indices = [int(x) for x in str(map_index.loc[uid, "sequence_indices"]).split(",") if x != ""]
                mean, pocket = transformer(full, len(sequence), indices)
                np.savez_compressed(per_uid / f"{uid}.npz", mean_feature=mean, pocket_node_feature=pocket, source=np.asarray("generated_esmc_600m"))
                generated += 1
        except Exception as exc:
            if len(batch) > 1:
                for item in batch:
                    pending_single = [item]
                    try:
                        assert model is not None
                        full = batched_full_embeddings(model, [item[1]])[0]
                        indices = [int(x) for x in str(map_index.loc[item[0], "sequence_indices"]).split(",") if x != ""]
                        mean, pocket = transformer(full, len(item[1]), indices)
                        np.savez_compressed(per_uid / f"{item[0]}.npz", mean_feature=mean, pocket_node_feature=pocket, source=np.asarray("generated_esmc_600m_single_fallback"))
                        generated += 1
                    except Exception as inner:
                        failures.append({"UniprotID": item[0], "sequence_length": len(item[1]), "error": repr(inner)})
            else:
                failures.append({"UniprotID": batch[0][0], "sequence_length": len(batch[0][1]), "error": repr(exc)})
        pd.DataFrame(failures, columns=["UniprotID", "sequence_length", "error"]).to_csv(output / "failures.csv", index=False)
        (output / "progress.json").write_text(json.dumps({"targets": len(targets), "reused": reused, "stale_reuse": stale_reuse, "generated": generated, "failed": len(failures)}, indent=2), encoding="utf-8")

    collated = collate(output, usable_targets, args.semantics)
    summary = {
        "protocol": "EnzymeCAGE ESM-C 600M minimal reconstruction",
        "semantics": args.semantics,
        "author_exact_note": "author_exact intentionally mirrors original L+2 mean and direct sequence-index lookup into the L+2 tensor",
        "targets": int(len(targets)), "mapping_ok": int(mapping.mapping_ok.sum()), "mapping_failed": int(len(bad)),
        "offset_counts": {str(k): int(v) for k, v in mapping.loc[mapping.mapping_ok, "offset"].value_counts().items()},
        "mapping_mismatch_uids": int((pd.to_numeric(mapping.mismatches, errors="coerce").fillna(0) > 0).sum()),
        "reused_full_node_cache": reused, "stale_reuse_cache": stale_reuse, "generated": generated, "failed": len(failures),
        "verified_against_reference": verified, "collated": collated,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
