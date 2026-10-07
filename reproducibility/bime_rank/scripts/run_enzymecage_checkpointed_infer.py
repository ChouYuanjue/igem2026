from __future__ import annotations

import argparse
import os
import pickle as pkl
import sys
from pathlib import Path

import pandas as pd
import torch
from torch_geometric.loader import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[3]
CAGE = ROOT / "external_repos/EnzymeCAGE"
if str(CAGE) not in sys.path:
    sys.path.insert(0, str(CAGE))

from config import Config
from enzymecage.base import RXN_COL, UID_COL
from enzymecage.dataset.geometric import GeometricDataset, load_geometric_dataset
from enzymecage.model import EnzymeCAGE
from infer import reuse_identical_sequence_structure_features
from utils import check_files, seed_everything

_ORIGINAL_GET_RXN_GRAPH_DATA = GeometricDataset.get_rxn_graph_data
_RXN_CACHE: dict[str, object] = {}
_CACHE_STATS = {"hits": 0, "misses": 0}


def _cached_get_rxn_graph_data(self: GeometricDataset, rxn: str):
    if rxn in _RXN_CACHE:
        _CACHE_STATS["hits"] += 1
        return _RXN_CACHE[rxn]
    value = _ORIGINAL_GET_RXN_GRAPH_DATA(self, rxn)
    _RXN_CACHE[rxn] = value
    _CACHE_STATS["misses"] += 1
    return value


GeometricDataset.get_rxn_graph_data = _cached_get_rxn_graph_data


def normalize_frame(path: str) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype=str).fillna("")
    if UID_COL not in frame.columns and "enzyme" in frame.columns:
        frame[UID_COL] = frame["enzyme"]
    if RXN_COL not in frame.columns and "reaction" in frame.columns:
        frame[RXN_COL] = frame["reaction"]
    if "Label" not in frame.columns:
        frame["Label"] = 0
    frame[UID_COL] = frame[UID_COL].astype(str)
    frame[RXN_COL] = frame[RXN_COL].astype(str)
    return frame


def load_features(model_conf, frame: pd.DataFrame):
    gvp = torch.load(model_conf.protein_gvp_feat, map_location="cpu", weights_only=False)
    if hasattr(model_conf, "protein_gvp_feat_extra"):
        extra = torch.load(model_conf.protein_gvp_feat_extra, map_location="cpu", weights_only=False)
        gvp.update(extra)
        print(f"Loaded {len(extra)} extra GVP features.", flush=True)

    node = torch.load(model_conf.esm_node_feature, map_location="cpu", weights_only=False)
    if hasattr(model_conf, "esm_node_feature_extra"):
        extra = torch.load(model_conf.esm_node_feature_extra, map_location="cpu", weights_only=False)
        node.update(extra)
        print(f"Loaded {len(extra)} extra pocket ESM features.", flush=True)

    if hasattr(model_conf, "esm_mean_feature_extra"):
        with open(model_conf.esm_mean_feature, "rb") as handle:
            mean = pkl.load(handle)
        with open(model_conf.esm_mean_feature_extra, "rb") as handle:
            extra = pkl.load(handle)
        mean.update(extra)
        print(f"Loaded {len(extra)} extra protein-level ESM features.", flush=True)
    else:
        mean = model_conf.esm_mean_feature

    reused = reuse_identical_sequence_structure_features(frame, gvp, node)
    if reused:
        print(f"Reused structure features for {len(reused)} UIDs.", flush=True)
    return gvp, node, mean


def preflight(frame: pd.DataFrame, model_conf, gvp: dict, node: dict):
    skipped: list[dict[str, str]] = []

    valid_uid: set[str] = set()
    for uid in sorted(set(frame[UID_COL])):
        if uid not in gvp or uid not in node:
            skipped.append({"kind": "uid", "id": uid, "reason": "missing_gvp_or_node_feature"})
            continue
        try:
            n_gvp = len(gvp[uid][0])
            n_node = len(node[uid])
        except Exception as exc:
            skipped.append({"kind": "uid", "id": uid, "reason": f"feature_shape_error:{exc}"})
            continue
        if n_gvp != n_node:
            skipped.append(
                {"kind": "uid", "id": uid, "reason": f"gvp_node_mismatch:{n_gvp}!={n_node}"}
            )
            continue
        valid_uid.add(uid)

    with open(model_conf.rxn_fp, "rb") as handle:
        rxn_fp = pkl.load(handle)
    graph_path = Path(model_conf.mol_conformation) / "mol_graph_dict.pt"
    graph_dict = torch.load(graph_path, map_location="cpu", weights_only=False)
    graph_keys = set(map(str, graph_dict))
    skip_mol = {"[*H2]"}

    valid_rxn: set[str] = set()
    rxn_to_smiles = (
        frame[[RXN_COL, "CANO_RXN_SMILES"]]
        .drop_duplicates(RXN_COL)
        .set_index(RXN_COL)
        .CANO_RXN_SMILES.to_dict()
    )
    for rid, reaction in rxn_to_smiles.items():
        if reaction not in rxn_fp:
            skipped.append({"kind": "reaction", "id": rid, "reason": "missing_reaction_fingerprint"})
            continue
        try:
            left, right = str(reaction).split(">>")
            molecules = {
                x.replace("*", "C")
                for x in left.split(".") + right.split(".")
                if x and x not in skip_mol
            }
        except Exception as exc:
            skipped.append({"kind": "reaction", "id": rid, "reason": f"reaction_parse_error:{exc}"})
            continue
        missing = sorted(molecules - graph_keys)
        if missing:
            skipped.append(
                {
                    "kind": "reaction",
                    "id": rid,
                    "reason": "missing_mol_graph:" + "|".join(missing[:3]),
                }
            )
            continue
        valid_rxn.add(rid)

    filtered = frame[
        frame[UID_COL].isin(valid_uid) & frame[RXN_COL].isin(valid_rxn)
    ].copy()
    skip_frame = pd.DataFrame(skipped, columns=["kind", "id", "reason"])
    return filtered, skip_frame


def output_paths(model_conf, model_name: str) -> tuple[Path, Path, Path]:
    result_dir = Path(model_conf.result_dir) if hasattr(model_conf, "result_dir") else Path(model_conf.ckpt_dir)
    result_dir.mkdir(parents=True, exist_ok=True)
    base = Path(model_conf.data_path).stem + "_" + model_name.replace(".pth", ".csv")
    final = result_dir / base
    partial = final.with_name(final.stem + ".partial.csv")
    skipped = final.with_name(final.stem + ".skipped.csv")
    return final, partial, skipped


def reconcile_partial(partial: Path, frame: pd.DataFrame) -> tuple[pd.DataFrame, set[tuple[str, str]]]:
    keys = [RXN_COL, UID_COL]
    if not partial.exists() or partial.stat().st_size == 0:
        return pd.DataFrame(), set()
    old = pd.read_csv(partial, dtype=str).fillna("")
    if not set(keys + ["pred", "pred_logit"]).issubset(old.columns):
        bad = partial.with_suffix(partial.suffix + ".invalid")
        partial.replace(bad)
        print(f"Moved invalid partial checkpoint to {bad}", flush=True)
        return pd.DataFrame(), set()
    valid_keys = frame[keys].drop_duplicates()
    old = old.merge(valid_keys, on=keys, how="inner", validate="many_to_one")
    old = old.drop_duplicates(keys, keep="last")
    old.to_csv(partial, index=False)
    done = set(map(tuple, old[keys].itertuples(index=False, name=None)))
    print(f"Resume checkpoint: {len(done)} scored pairs already on disk.", flush=True)
    return old, done


def append_checkpoint(partial: Path, chunks: list[pd.DataFrame]) -> int:
    if not chunks:
        return 0
    frame = pd.concat(chunks, ignore_index=True)
    exists = partial.exists() and partial.stat().st_size > 0
    with partial.open("a", encoding="utf-8", newline="") as handle:
        frame.to_csv(handle, index=False, header=not exists)
        handle.flush()
        os.fsync(handle.fileno())
    chunks.clear()
    return len(frame)


def run_model(model_conf, model_name: str, frame: pd.DataFrame, gvp, node, mean, checkpoint_batches: int):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    final, partial, skipped_path = output_paths(model_conf, model_name)

    filtered, skipped = preflight(frame, model_conf, gvp, node)
    skipped.to_csv(skipped_path, index=False)
    print(
        f"Preflight kept {len(filtered)}/{len(frame)} pairs; "
        f"skipped items={len(skipped)}.",
        flush=True,
    )

    _, done = reconcile_partial(partial, filtered)
    if done:
        key_series = list(zip(filtered[RXN_COL], filtered[UID_COL]))
        mask = [key not in done for key in key_series]
        remaining = filtered.loc[mask].reset_index(drop=True)
    else:
        remaining = filtered.reset_index(drop=True)

    if final.exists():
        existing_final = pd.read_csv(final, dtype=str).fillna("")
        expected = set(map(tuple, filtered[[RXN_COL, UID_COL]].itertuples(index=False, name=None)))
        actual = set(map(tuple, existing_final[[RXN_COL, UID_COL]].itertuples(index=False, name=None)))
        if actual == expected:
            print(f"Final output already complete: {final}", flush=True)
            return
        final.unlink()

    if remaining.empty:
        complete = pd.read_csv(partial, dtype=str).fillna("")
        complete = complete.drop_duplicates([RXN_COL, UID_COL], keep="last")
        complete.to_csv(final, index=False)
        print(f"Recovered complete output from checkpoint: {final}", flush=True)
        return

    dataset = load_geometric_dataset(
        remaining,
        gvp,
        model_conf.rxn_fp,
        model_conf.mol_conformation,
        node,
        mean,
        model_conf.reaction_center,
    )
    dataset_frame = dataset.df_data.reset_index(drop=True)
    follow_batch = ["protein", "reaction_feature", "esm_feature", "substrates", "products"]
    loader = DataLoader(
        dataset,
        batch_size=model_conf.batch_size,
        shuffle=False,
        follow_batch=follow_batch,
    )

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
    ckpt_path = Path(model_conf.ckpt_dir) / model_name
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    offset = 0
    chunks: list[pd.DataFrame] = []
    persisted = len(done)
    try:
        for batch_index, batch in enumerate(tqdm(loader, total=len(loader)), 1):
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
                raise RuntimeError(f"batch/data alignment failure at batch {batch_index}: {len(rows)} != {n}")
            rows["pred"] = probability.cpu().numpy()
            rows["pred_logit"] = logit.cpu().numpy()
            chunks.append(rows)
            offset += n

            if batch_index % checkpoint_batches == 0:
                persisted += append_checkpoint(partial, chunks)
                print(
                    f"BRIDGE_V4_CHECKPOINT batch={batch_index}/{len(loader)} "
                    f"persisted={persisted}/{len(filtered)}",
                    flush=True,
                )
    except Exception:
        persisted += append_checkpoint(partial, chunks)
        print(
            f"BRIDGE_V4_CHECKPOINT_ON_ERROR persisted={persisted}/{len(filtered)}",
            flush=True,
        )
        raise

    persisted += append_checkpoint(partial, chunks)
    complete = pd.read_csv(partial, dtype=str).fillna("")
    complete = complete.drop_duplicates([RXN_COL, UID_COL], keep="last")
    expected = set(map(tuple, filtered[[RXN_COL, UID_COL]].itertuples(index=False, name=None)))
    actual = set(map(tuple, complete[[RXN_COL, UID_COL]].itertuples(index=False, name=None)))
    if actual != expected:
        missing = len(expected - actual)
        extra = len(actual - expected)
        raise RuntimeError(f"checkpoint coverage mismatch after scoring: missing={missing}, extra={extra}")

    tmp = final.with_suffix(final.suffix + ".tmp")
    complete.to_csv(tmp, index=False)
    os.replace(tmp, final)
    print(f"Save complete pred result to: {final}", flush=True)
    print(
        "BRIDGE_V4_RXN_GRAPH_CACHE",
        {**_CACHE_STATS, "unique_reactions_cached": _CACHE_STATS["misses"]},
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint-batches", type=int, default=16)
    args = parser.parse_args()

    model_conf = Config(args.config)
    seed = 42 if not hasattr(model_conf, "seed") else model_conf.seed
    seed_everything(seed)
    check_files(model_conf)

    frame = normalize_frame(model_conf.data_path)
    gvp, node, mean = load_features(model_conf, frame)
    for model_name in model_conf.model_list:
        run_model(
            model_conf,
            model_name,
            frame,
            gvp,
            node,
            mean,
            max(1, args.checkpoint_batches),
        )


if __name__ == "__main__":
    main()
