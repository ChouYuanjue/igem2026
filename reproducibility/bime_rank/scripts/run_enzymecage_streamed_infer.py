from __future__ import annotations

import argparse
import gc
import os
import pickle as pkl
import sys
from collections import OrderedDict
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch_geometric.loader import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[3]
CAGE = ROOT / "external_repos/EnzymeCAGE"
FEATURE = CAGE / "feature"
for path in (CAGE, FEATURE):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from config import Config
from enzymecage.base import RXN_COL, UID_COL, SEQ_COL
from enzymecage.dataset.geometric import GeometricDataset
from enzymecage.model import EnzymeCAGE
from main import (
    infer_sequence_indices_from_sequence,
    load_pocket_residue_records,
    parse_pocket_residue_ids,
)
from utils import check_files, seed_everything

AUTHOR = ROOT / "results/bridge_layered_v4_cage_features/r2e_union"
FALLBACK = ROOT / "results/bridge_layered_v4_cage_features/r2e_fallback"

AUTHOR_GVP = AUTHOR / "feature/protein/gvp_feature/tmp"
AUTHOR_NODE = AUTHOR / "feature/protein/ESM-C_600M/node_level"
AUTHOR_POCKET_INFO = AUTHOR / "pocket_info.csv"
AUTHOR_POCKET = AUTHOR / "pocket"

FALLBACK_GVP = FALLBACK / "feature/protein/gvp_feature/tmp"
FALLBACK_NODE = FALLBACK / "feature/protein/ESM-C_600M/node_level"
FALLBACK_POCKET_INFO = FALLBACK / "p2rank/pocket_info.csv"
FALLBACK_POCKET = FALLBACK / "p2rank/pocket"

_SHARED_MOL_GRAPH: dict[str, object] | None = None
_SHARED_RXN_FEATURE: dict[str, object] | None = None
_SHARED_REACTION_CENTER: dict[str, object] | None = None
_SHARED_MOL_TO_INDEX: dict[str, object] | None = None
_RXN_GRAPH_CACHE: OrderedDict[str, tuple[object, object]] = OrderedDict()
_RXN_CACHE_MAX = 256
_GVP_NUM_TO_AA = {
    4: "C", 3: "D", 15: "S", 5: "Q", 11: "K", 9: "I",
    14: "P", 16: "T", 13: "F", 0: "A", 7: "G", 8: "H",
    6: "E", 10: "L", 1: "R", 17: "W", 19: "V", 2: "N",
    18: "Y", 12: "M",
}


def normalize_frame(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.fillna("")
    if UID_COL not in frame.columns and "enzyme" in frame.columns:
        frame[UID_COL] = frame["enzyme"]
    if RXN_COL not in frame.columns and "reaction" in frame.columns:
        frame[RXN_COL] = frame["reaction"]
    if "Label" not in frame.columns:
        frame["Label"] = 0
    frame["Label"] = pd.to_numeric(
        frame["Label"], errors="coerce"
    ).fillna(0).astype(float)
    frame[UID_COL] = frame[UID_COL].astype(str)
    frame[RXN_COL] = frame[RXN_COL].astype(str)
    if "reaction_id" in frame.columns:
        frame["reaction_id"] = frame["reaction_id"].astype(str)
    return frame


def infer_sequence_indices_from_gvp(
    full_sequence: str,
    gvp_sequence,
    pocket_residue_ids: list[int],
) -> list[int]:
    """Exact fallback mapping from GVP pocket residue identities.

    Used only when the official PDB-record offset inference fails.  We accept
    a mapping only if one constant offset yields a complete, mismatch-free,
    in-range alignment for every pocket residue, and that exact offset is
    unique.
    """

    full_sequence = str(full_sequence).strip().upper()
    pocket_aa = "".join(
        _GVP_NUM_TO_AA[int(value)]
        for value in torch.as_tensor(gvp_sequence).detach().cpu().tolist()
    )
    if len(pocket_aa) != len(pocket_residue_ids):
        raise ValueError(
            "GVP/pocket residue count mismatch "
            f"{len(pocket_aa)}!={len(pocket_residue_ids)}"
        )

    direct = [int(residue_id) - 1 for residue_id in pocket_residue_ids]
    if (
        all(0 <= idx < len(full_sequence) for idx in direct)
        and all(
            full_sequence[idx] == amino_acid
            for idx, amino_acid in zip(direct, pocket_aa, strict=True)
        )
    ):
        return direct

    aa_positions: dict[str, list[int]] = {}
    for idx, amino_acid in enumerate(full_sequence):
        aa_positions.setdefault(amino_acid, []).append(idx)

    candidate_offsets: set[int] = set()
    for residue_id, amino_acid in zip(
        pocket_residue_ids,
        pocket_aa,
        strict=True,
    ):
        for seq_idx in aa_positions.get(amino_acid, []):
            candidate_offsets.add(int(residue_id) - int(seq_idx))

    exact: list[tuple[int, list[int]]] = []
    for offset in candidate_offsets:
        indices = [
            int(residue_id) - int(offset)
            for residue_id in pocket_residue_ids
        ]
        if not all(0 <= idx < len(full_sequence) for idx in indices):
            continue
        if all(
            full_sequence[idx] == amino_acid
            for idx, amino_acid in zip(indices, pocket_aa, strict=True)
        ):
            exact.append((int(offset), indices))

    if len(exact) != 1:
        offsets = [item[0] for item in exact[:8]]
        raise ValueError(
            "GVP exact offset mapping is not unique: "
            f"matches={len(exact)} offsets={offsets}"
        )
    return exact[0][1]


def pair_keys(frame: pd.DataFrame) -> list[str]:
    return ["reaction_id", UID_COL] if "reaction_id" in frame.columns else [RXN_COL, UID_COL]


def output_paths(model_conf, model_name: str) -> tuple[Path, Path, Path]:
    result_dir = (
        Path(model_conf.result_dir)
        if hasattr(model_conf, "result_dir")
        else Path(model_conf.ckpt_dir)
    )
    result_dir.mkdir(parents=True, exist_ok=True)
    base = Path(model_conf.data_path).stem + "_" + model_name.replace(".pth", ".csv")
    final = result_dir / base
    partial = final.with_name(final.stem + ".partial.csv")
    skipped = final.with_name(final.stem + ".stream_skipped.csv")
    return final, partial, skipped


def load_pocket_maps() -> tuple[dict[str, tuple[str, Path]], set[str]]:
    mapping: dict[str, tuple[str, Path]] = {}
    for info_path, pocket_dir in (
        (AUTHOR_POCKET_INFO, AUTHOR_POCKET),
        (FALLBACK_POCKET_INFO, FALLBACK_POCKET),
    ):
        if not info_path.exists():
            continue
        frame = pd.read_csv(info_path, dtype=str).fillna("")
        for row in frame[[UID_COL, "pocket_residues"]].drop_duplicates(UID_COL).itertuples(index=False):
            uid = str(row[0])
            mapping[uid] = (str(row[1]), pocket_dir / f"{uid}.pdb")
    return mapping, set(mapping)


class FeatureStore:
    def __init__(self, model_conf, cache_size: int):
        print("Loading base CAGE features only...", flush=True)
        self.base_gvp = torch.load(
            model_conf.protein_gvp_feat, map_location="cpu", weights_only=False
        )
        self.base_node = torch.load(
            model_conf.esm_node_feature, map_location="cpu", weights_only=False
        )
        with open(model_conf.esm_mean_feature, "rb") as handle:
            self.base_mean = pkl.load(handle)
        self.pocket_map, self.extra_uids = load_pocket_maps()
        self.cache_size = max(0, int(cache_size))
        self.cache: OrderedDict[str, tuple[object, np.ndarray, torch.Tensor]] = OrderedDict()
        self.load_count = 0

    def _extra_paths(self, uid: str) -> tuple[Path, Path]:
        if uid in self.pocket_map:
            if (AUTHOR_GVP / f"{uid}.pt").exists():
                return AUTHOR_GVP / f"{uid}.pt", AUTHOR_NODE / f"{uid}.npz"
            return FALLBACK_GVP / f"{uid}.pt", FALLBACK_NODE / f"{uid}.npz"
        raise KeyError(uid)

    def has_uid(self, uid: str) -> bool:
        if uid in self.base_gvp and uid in self.base_node:
            return True
        if uid not in self.pocket_map:
            return False
        try:
            gp, npz = self._extra_paths(uid)
        except KeyError:
            return False
        return gp.exists() and npz.exists() and self.pocket_map[uid][1].exists()

    def get(self, uid: str, sequence: str):
        if uid in self.base_gvp and uid in self.base_node:
            mean = self.base_mean.get(sequence)
            if mean is None:
                mean = torch.zeros(self.base_node[uid].shape[1])
            return self.base_gvp[uid], self.base_node[uid], mean

        if uid in self.cache:
            value = self.cache.pop(uid)
            self.cache[uid] = value
            return value

        gp, npz = self._extra_paths(uid)
        gvp_dict = torch.load(gp, map_location="cpu", weights_only=False)
        gvp = gvp_dict[uid]
        with np.load(npz) as z:
            full_node = np.asarray(z["node_feature"])

        pocket_residues, pocket_pdb = self.pocket_map[uid]
        residue_ids = parse_pocket_residue_ids(pocket_residues)
        residue_records = load_pocket_residue_records(str(pocket_pdb))
        try:
            sequence_indices, _ = infer_sequence_indices_from_sequence(
                sequence, residue_records, residue_ids
            )
        except ValueError as pdb_mapping_error:
            try:
                sequence_indices = infer_sequence_indices_from_gvp(
                    sequence,
                    gvp[1],
                    residue_ids,
                )
            except ValueError as gvp_mapping_error:
                raise ValueError(
                    f"PDB mapping failed ({pdb_mapping_error}); "
                    f"GVP mapping failed ({gvp_mapping_error})"
                ) from gvp_mapping_error
        if not sequence_indices:
            raise ValueError(f"{uid}: empty pocket mapping")
        if max(sequence_indices) >= full_node.shape[0]:
            raise ValueError(
                f"{uid}: mapped index {max(sequence_indices)} exceeds "
                f"ESM length {full_node.shape[0]}"
            )
        pocket_node = full_node[sequence_indices]
        if len(gvp[0]) != len(pocket_node):
            raise ValueError(
                f"{uid}: GVP/pocket ESM mismatch {len(gvp[0])}!={len(pocket_node)}"
            )
        mean = torch.as_tensor(full_node).mean(dim=0)
        value = (gvp, pocket_node, mean)
        self.load_count += 1

        if self.cache_size:
            self.cache[uid] = value
            while len(self.cache) > self.cache_size:
                self.cache.popitem(last=False)
        return value

    def chunk_dicts(self, frame: pd.DataFrame):
        gvp: dict[str, object] = {}
        node: dict[str, np.ndarray] = {}
        mean: dict[str, object] = {}
        skipped: list[dict[str, str]] = []
        uid_seq = frame[[UID_COL, "sequence"]].drop_duplicates(UID_COL)
        for row in uid_seq.itertuples(index=False):
            uid, seq = str(row[0]), str(row[1])
            try:
                g, n, m = self.get(uid, seq)
                gvp[uid] = g
                node[uid] = n
                mean[seq] = m
            except Exception as exc:
                skipped.append(
                    {
                        "kind": "uid",
                        "id": uid,
                        "reason": f"feature_load:{type(exc).__name__}:{exc}",
                    }
                )
        return gvp, node, mean, skipped


def install_shared_reaction_graphs(model_conf) -> None:
    global _SHARED_MOL_GRAPH
    global _SHARED_RXN_FEATURE
    global _SHARED_REACTION_CENTER
    global _SHARED_MOL_TO_INDEX
    graph_path = Path(model_conf.mol_conformation) / "mol_graph_dict.pt"
    print(f"Loading shared molecule graph cache: {graph_path}", flush=True)
    _SHARED_MOL_GRAPH = torch.load(
        graph_path, map_location="cpu", weights_only=False
    )
    print(f"Loading shared reaction feature cache: {model_conf.rxn_fp}", flush=True)
    with open(model_conf.rxn_fp, "rb") as handle:
        _SHARED_RXN_FEATURE = pkl.load(handle)
    reaction_center_path = getattr(model_conf, "reaction_center", None)
    if reaction_center_path:
        print(
            f"Loading shared reaction-center cache: {reaction_center_path}",
            flush=True,
        )
        with open(reaction_center_path, "rb") as handle:
            _SHARED_REACTION_CENTER = pkl.load(handle)
    mol_index = pd.read_csv(
        Path(model_conf.mol_conformation) / "mol2id.csv"
    )
    _SHARED_MOL_TO_INDEX = dict(zip(mol_index["SMILES"], mol_index["ID"]))

    def _shared_load_mol_feat(self, mol_sdf_dir, use_cache=True):
        return _SHARED_MOL_GRAPH

    def _shared_init(
        self,
        df_data,
        protein_data,
        rxn_feat_path,
        mol_sdf_dir,
        pocket_node_feature,
        esm_feature_path=None,
        reacting_center_path=None,
        weight_col=None,
    ):
        super(GeometricDataset, self).__init__()
        if isinstance(df_data, str) and os.path.exists(df_data):
            self.df_data = pd.read_csv(df_data)
        elif isinstance(df_data, pd.DataFrame):
            self.df_data = df_data
        else:
            raise ValueError(f"Invalid data input: {type(df_data)}")

        if isinstance(pocket_node_feature, str) and os.path.exists(pocket_node_feature):
            self.pocket_node_feature = torch.load(
                pocket_node_feature,
                map_location="cpu",
                weights_only=False,
            )
        elif isinstance(pocket_node_feature, dict):
            self.pocket_node_feature = pocket_node_feature
        else:
            raise ValueError("Invalid pocket node feature input")
        self.esm_dim = list(self.pocket_node_feature.values())[0].shape[1]

        if isinstance(protein_data, str) and os.path.exists(protein_data):
            self.protein_dict = torch.load(
                protein_data,
                map_location="cpu",
                weights_only=False,
            )
        elif isinstance(protein_data, dict):
            self.protein_dict = protein_data
        else:
            raise ValueError("Invalid protein data input")

        if isinstance(esm_feature_path, str) and os.path.exists(esm_feature_path):
            with open(esm_feature_path, "rb") as handle:
                self.esm_feat_dict = pkl.load(handle)
        elif isinstance(esm_feature_path, dict):
            self.esm_feat_dict = esm_feature_path
        else:
            self.esm_feat_dict = None

        self.rxn_feat_dict = _SHARED_RXN_FEATURE
        self.uniprot_ids = self.df_data[UID_COL].tolist()
        self.rxns = self.df_data[RXN_COL].tolist()
        self.targets = self.df_data["Label"].tolist()
        self.seqs = self.df_data[SEQ_COL].tolist()
        if weight_col is not None:
            if weight_col not in self.df_data.columns:
                raise ValueError(f"Invalid weight column name: {weight_col}")
            self.weights = self.df_data[weight_col].tolist()
        else:
            self.weights = torch.ones(len(self.df_data))
        self.mol_to_index = _SHARED_MOL_TO_INDEX
        self.mol_to_data = _SHARED_MOL_GRAPH
        self.reacting_center_map = (
            _SHARED_REACTION_CENTER if reacting_center_path else None
        )

    original_get = GeometricDataset.get_rxn_graph_data

    def _cached_get(self, rxn: str):
        if rxn in _RXN_GRAPH_CACHE:
            value = _RXN_GRAPH_CACHE.pop(rxn)
            _RXN_GRAPH_CACHE[rxn] = value
            return value
        value = original_get(self, rxn)
        _RXN_GRAPH_CACHE[rxn] = value
        while len(_RXN_GRAPH_CACHE) > _RXN_CACHE_MAX:
            _RXN_GRAPH_CACHE.popitem(last=False)
        return value

    GeometricDataset.__init__ = _shared_init
    GeometricDataset.load_mol_feat = _shared_load_mol_feat
    GeometricDataset.get_rxn_graph_data = _cached_get


def build_model(model_conf, model_name: str, device: str) -> EnzymeCAGE:
    model = EnzymeCAGE(
        use_esm=model_conf.use_esm,
        use_structure=model_conf.use_structure,
        use_drfp=model_conf.use_drfp,
        use_prods_info=model_conf.use_prods_info,
        esm_model=getattr(model_conf, "esm_model", None),
        interaction_method=model_conf.interaction_method,
        rxn_inner_interaction=model_conf.rxn_inner_interaction,
        pocket_inner_interaction=model_conf.pocket_inner_interaction,
        device=device,
    )
    ckpt = Path(model_conf.ckpt_dir) / model_name
    state = torch.load(ckpt, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


@torch.no_grad()
def score_once(
    model,
    model_conf,
    frame: pd.DataFrame,
    feature_store: FeatureStore,
    batch_size: int,
    device: str,
) -> tuple[pd.DataFrame, list[dict[str, str]]]:
    gvp, node, mean, skipped = feature_store.chunk_dicts(frame)
    if skipped:
        bad_uids = {x["id"] for x in skipped if x["kind"] == "uid"}
        frame = frame[~frame[UID_COL].isin(bad_uids)].reset_index(drop=True)
    if frame.empty:
        return pd.DataFrame(), skipped

    dataset = GeometricDataset(
        frame,
        gvp,
        model_conf.rxn_fp,
        model_conf.mol_conformation,
        node,
        mean,
        model_conf.reaction_center,
    )
    dataset_frame = dataset.df_data.reset_index(drop=True)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        follow_batch=[
            "protein",
            "reaction_feature",
            "esm_feature",
            "substrates",
            "products",
        ],
    )

    output: list[pd.DataFrame] = []
    offset = 0
    for batch in loader:
        score = model.forward(batch.to(device))
        if isinstance(score, tuple):
            score = score[0]
        raw = score.detach().reshape(-1)
        if getattr(model, "sigmoid_readout", False):
            probability = raw
            logit = torch.logit(raw.clamp(min=1e-7, max=1 - 1e-7))
        else:
            logit = raw
            probability = torch.sigmoid(raw)
        n = int(raw.numel())
        rows = dataset_frame.iloc[offset : offset + n].copy()
        if len(rows) != n:
            raise RuntimeError(f"batch/data alignment {len(rows)}!={n}")
        rows["pred"] = probability.cpu().numpy()
        rows["pred_logit"] = logit.cpu().numpy()
        output.append(rows)
        offset += n
        del batch, score, raw, probability, logit

    if offset != len(dataset_frame):
        raise RuntimeError(f"chunk alignment {offset}!={len(dataset_frame)}")
    result = pd.concat(output, ignore_index=True) if output else pd.DataFrame()
    del dataset, loader, gvp, node, mean, output
    return result, skipped


def score_resilient(
    model,
    model_conf,
    frame: pd.DataFrame,
    feature_store: FeatureStore,
    batch_size: int,
    min_batch_size: int,
    device: str,
) -> tuple[pd.DataFrame, list[dict[str, str]]]:
    try:
        return score_once(
            model, model_conf, frame, feature_store, batch_size, device
        )
    except torch.OutOfMemoryError as exc:
        torch.cuda.empty_cache()
        gc.collect()
        if batch_size > min_batch_size:
            smaller = max(min_batch_size, batch_size // 2)
            print(
                f"CHUNK_OOM rows={len(frame)} batch={batch_size}; "
                f"retry batch={smaller}",
                flush=True,
            )
            return score_resilient(
                model,
                model_conf,
                frame,
                feature_store,
                smaller,
                min_batch_size,
                device,
            )
        if len(frame) > 1:
            mid = len(frame) // 2
            print(
                f"CHUNK_OOM at min batch; bisect {len(frame)} -> "
                f"{mid}+{len(frame)-mid}",
                flush=True,
            )
            left, skip_left = score_resilient(
                model, model_conf, frame.iloc[:mid].copy(),
                feature_store, batch_size, min_batch_size, device
            )
            right, skip_right = score_resilient(
                model, model_conf, frame.iloc[mid:].copy(),
                feature_store, batch_size, min_batch_size, device
            )
            return pd.concat([left, right], ignore_index=True), skip_left + skip_right
        key = "|".join(map(str, frame[pair_keys(frame)].iloc[0].tolist()))
        return pd.DataFrame(), [
            {"kind": "pair", "id": key, "reason": f"oom:{exc}"}
        ]
    except Exception as exc:
        torch.cuda.empty_cache()
        gc.collect()
        if len(frame) > 1:
            mid = len(frame) // 2
            print(
                f"CHUNK_ERROR {type(exc).__name__}; bisect "
                f"{len(frame)} -> {mid}+{len(frame)-mid}",
                flush=True,
            )
            left, skip_left = score_resilient(
                model, model_conf, frame.iloc[:mid].copy(),
                feature_store, batch_size, min_batch_size, device
            )
            right, skip_right = score_resilient(
                model, model_conf, frame.iloc[mid:].copy(),
                feature_store, batch_size, min_batch_size, device
            )
            return pd.concat([left, right], ignore_index=True), skip_left + skip_right
        key = "|".join(map(str, frame[pair_keys(frame)].iloc[0].tolist()))
        return pd.DataFrame(), [
            {
                "kind": "pair",
                "id": key,
                "reason": f"{type(exc).__name__}:{exc}",
            }
        ]


def append_frame(path: Path, frame: pd.DataFrame) -> None:
    if frame.empty:
        return
    exists = path.exists() and path.stat().st_size > 0
    with path.open("a", encoding="utf-8", newline="") as handle:
        frame.to_csv(handle, index=False, header=not exists)
        handle.flush()
        os.fsync(handle.fileno())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--chunk-rows", type=int, default=4096)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--min-batch-size", type=int, default=32)
    ap.add_argument("--feature-cache-uids", type=int, default=256)
    args = ap.parse_args()

    model_conf = Config(args.config)
    seed = 42 if not hasattr(model_conf, "seed") else model_conf.seed
    seed_everything(seed)
    check_files(model_conf)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model_name = model_conf.model_list[0]
    final, partial, skipped_path = output_paths(model_conf, model_name)
    if final.exists():
        print(f"Final output already exists: {final}", flush=True)
        return

    done: set[tuple[str, str]] = set()
    if partial.exists() and partial.stat().st_size > 0:
        old = pd.read_csv(partial, dtype=str).fillna("")
        keys = pair_keys(old)
        old = old.drop_duplicates(keys, keep="last")
        old.to_csv(partial, index=False)
        done = set(map(tuple, old[keys].itertuples(index=False, name=None)))
        print(f"Resume partial: {len(done)} scored pairs.", flush=True)

    install_shared_reaction_graphs(model_conf)
    features = FeatureStore(model_conf, args.feature_cache_uids)
    model = build_model(model_conf, model_name, device)

    skipped_records: list[dict[str, str]] = []
    input_rows = 0
    scored_now = 0
    chunk_index = 0
    for raw in pd.read_csv(
        model_conf.data_path,
        dtype=str,
        chunksize=max(1, args.chunk_rows),
    ):
        chunk_index += 1
        frame = normalize_frame(raw)
        input_rows += len(frame)
        keys = pair_keys(frame)
        if done:
            tuples = list(map(tuple, frame[keys].itertuples(index=False, name=None)))
            mask = [key not in done for key in tuples]
            frame = frame.loc[mask].reset_index(drop=True)
        if frame.empty:
            continue

        result, skipped = score_resilient(
            model,
            model_conf,
            frame,
            features,
            max(1, args.batch_size),
            max(1, args.min_batch_size),
            device,
        )
        append_frame(partial, result)
        if not result.empty:
            result_keys = list(
                map(tuple, result[keys].astype(str).itertuples(index=False, name=None))
            )
            done.update(result_keys)
            scored_now += len(result_keys)
        if skipped:
            skipped_records.extend(skipped)
            pd.DataFrame(
                skipped_records, columns=["kind", "id", "reason"]
            ).to_csv(skipped_path, index=False)

        print(
            f"STREAM_CHECKPOINT chunk={chunk_index} input_seen={input_rows} "
            f"persisted={len(done)} scored_this_run={scored_now} "
            f"skipped={len(skipped_records)} feature_cache={len(features.cache)}",
            flush=True,
        )
        gc.collect()
        torch.cuda.empty_cache()

    complete = pd.read_csv(partial, dtype=str).fillna("")
    keys = pair_keys(complete)
    complete = complete.drop_duplicates(keys, keep="last")
    if skipped_records:
        print(
            f"STREAM_FINISH scored={len(complete)} runtime_skipped={len(skipped_records)}",
            flush=True,
        )
    else:
        print(f"STREAM_FINISH scored={len(complete)} runtime_skipped=0", flush=True)

    tmp = final.with_suffix(final.suffix + ".tmp")
    complete.to_csv(tmp, index=False)
    os.replace(tmp, final)
    print(f"Save complete pred result to: {final}", flush=True)


if __name__ == "__main__":
    main()
