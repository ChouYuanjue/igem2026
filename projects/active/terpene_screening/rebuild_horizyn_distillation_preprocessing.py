from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[3]
HORIZYN_ROOT = ROOT / "external/horizyn"
for path in (ROOT, HORIZYN_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from horizyn.chemistry.standardizer import Standardizer  # noqa: E402

DEFAULT_TRAIN = HORIZYN_ROOT / "data/sota/train_rxns.csv"
DEFAULT_TEST = HORIZYN_ROOT / "data/sota/test_rxns.csv"
DEFAULT_CONFIG = HORIZYN_ROOT / "configs/sota.yaml"
DEFAULT_MARTS = ROOT / "data/terpene_marts_adaptation/reaction_entities.csv"
DEFAULT_DATA_OUTPUT = ROOT / "data/terpene_horizyn_adapter_v2"
DEFAULT_OVERLAP_OUTPUT = ROOT / "results/terpene_horizyn_reaction_overlap.csv"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build_standardizer(config_path: Path) -> Standardizer:
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    data = dict(payload.get("data") or {})
    if not bool(data.get("standardize_reactions", True)):
        raise ValueError("Horizyn SOTA contract requires reaction standardization")
    return Standardizer(
        standardize_hypervalent=bool(data.get("standardize_hypervalent", True)),
        standardize_remove_hs=bool(data.get("standardize_remove_hs", True)),
        standardize_kekulize=bool(data.get("standardize_kekulize", False)),
        standardize_uncharge=bool(data.get("standardize_uncharge", True)),
        standardize_metals=bool(data.get("standardize_metals", True)),
    )


def standardize_table(path: Path, standardizer: Standardizer, required: list[str]) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype=str).fillna("")
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"{path} is missing columns: {missing}")
    output = frame.copy()
    values: list[str] = []
    for index, reaction in enumerate(frame["reaction_smiles"].astype(str)):
        try:
            values.append(standardizer.standardize_reaction(reaction))
        except Exception as exc:
            raise ValueError(f"reaction standardization failed at row {index} in {path}") from exc
    output["standardized_reaction"] = values
    return output


def unique_lookup(frame: pd.DataFrame, label: str) -> dict[str, str]:
    grouped = frame.groupby("standardized_reaction", sort=False)["reaction_id"].apply(list)
    result: dict[str, str] = {}
    for standardized, ids in grouped.items():
        ordered = [str(value) for value in ids]
        # The historical overlap table only records one ID per matching MARTS
        # reaction. Fail closed if a future input makes a matched standardized
        # reaction ambiguous rather than silently changing serialization semantics.
        if len(ordered) == 1:
            result[str(standardized)] = ordered[0]
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rebuild the exact Horizyn/MARTS standardization and overlap inputs used by the released reaction distiller."
    )
    parser.add_argument("--train-reactions", type=Path, default=DEFAULT_TRAIN)
    parser.add_argument("--test-reactions", type=Path, default=DEFAULT_TEST)
    parser.add_argument("--horizyn-config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--marts-reactions", type=Path, default=DEFAULT_MARTS)
    parser.add_argument("--data-output-dir", type=Path, default=DEFAULT_DATA_OUTPUT)
    parser.add_argument("--overlap-output", type=Path, default=DEFAULT_OVERLAP_OUTPUT)
    args = parser.parse_args()

    standardizer = build_standardizer(args.horizyn_config.resolve())
    train = standardize_table(
        args.train_reactions.resolve(), standardizer, ["rs_id", "reaction_id", "reaction_smiles"]
    )
    test = standardize_table(
        args.test_reactions.resolve(), standardizer, ["rs_id", "reaction_id", "reaction_smiles"]
    )
    marts = standardize_table(
        args.marts_reactions.resolve(),
        standardizer,
        ["reaction_id", "reaction_signature", "reaction_smiles", "reaction_seen", "cluster_id"],
    )

    train_lookup = unique_lookup(train, "train")
    test_lookup = unique_lookup(test, "test")
    train_counts = train["standardized_reaction"].value_counts()
    test_counts = test["standardized_reaction"].value_counts()

    overlap = marts.copy()
    overlap["horizyn_train_matches"] = overlap["standardized_reaction"].map(train_lookup).fillna("")
    overlap["horizyn_test_matches"] = overlap["standardized_reaction"].map(test_lookup).fillna("")
    # Preserve the historical one-to-one overlap semantics. If a MARTS reaction
    # hits a duplicated standardized Horizyn reaction, the old table would be
    # ambiguous and cannot be reproduced safely.
    ambiguous_train = overlap["standardized_reaction"].map(train_counts).fillna(0).astype(int) > 1
    ambiguous_test = overlap["standardized_reaction"].map(test_counts).fillna(0).astype(int) > 1
    if ambiguous_train.any() or ambiguous_test.any():
        rows = overlap.loc[ambiguous_train | ambiguous_test, "reaction_id"].astype(str).tolist()
        raise ValueError(f"ambiguous standardized Horizyn matches for MARTS reactions: {rows[:10]}")
    overlap["in_horizyn_train"] = overlap["horizyn_train_matches"].ne("")
    overlap["in_horizyn_test"] = overlap["horizyn_test_matches"].ne("")

    data_output = args.data_output_dir.resolve()
    overlap_output = args.overlap_output.resolve()
    data_output.mkdir(parents=True, exist_ok=True)
    overlap_output.parent.mkdir(parents=True, exist_ok=True)
    outputs = {
        "train_standardized_reactions": data_output / "train_standardized_reactions.csv",
        "test_standardized_reactions": data_output / "test_standardized_reactions.csv",
        "marts_standardized_reactions": data_output / "marts_standardized_reactions.csv",
        "reaction_overlap": overlap_output,
    }
    train.to_csv(outputs["train_standardized_reactions"], index=False)
    test.to_csv(outputs["test_standardized_reactions"], index=False)
    marts.to_csv(outputs["marts_standardized_reactions"], index=False)
    overlap.to_csv(outputs["reaction_overlap"], index=False)

    result = {
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "marts_rows": int(len(marts)),
        "marts_train_overlap": int(overlap["in_horizyn_train"].sum()),
        "marts_test_overlap": int(overlap["in_horizyn_test"].sum()),
        "outputs": {
            key: {"path": str(path), "sha256": sha256(path)} for key, path in outputs.items()
        },
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
