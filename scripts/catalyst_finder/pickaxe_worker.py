#!/usr/bin/env python3
"""Isolated MINE/Pickaxe one-generation expansion worker.

This module is intentionally launched as a subprocess. It never becomes an import
requirement of the Catalyst Finder web service. The vendored upstream source and the
small runtime-only dependency site are prepended only inside this process.
"""
from __future__ import annotations

import contextlib
import io
import json
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
VENDOR = ROOT / "external_repos/route_design/MINE-Database"
SITE = ROOT / "results/catalyst_finder_runtime/route_design/pickaxe_site"
RULES = VENDOR / "minedatabase/data/metacyc_rules/metacyc_generalized_rules.tsv"
COFACTORS = VENDOR / "minedatabase/data/metacyc_rules/metacyc_coreactants.tsv"
WORK = ROOT / "results/catalyst_finder_runtime/route_design/pickaxe_worker"


def _prepare_imports() -> None:
    sys.path.insert(0, str(VENDOR))
    sys.path.insert(0, str(SITE))
    # Upstream imports Draw at module import even when no image is requested. The host
    # lacks X11 libraries, so provide a nonvisual stub rather than patching upstream.
    stub = types.ModuleType("rdkit.Chem.Draw")

    def disabled(*_args, **_kwargs):
        raise RuntimeError("drawing is disabled in the isolated route worker")

    class Draw2DStub:
        def __getattr__(self, _name):
            return disabled

    stub.MolToFile = disabled
    stub.rdMolDraw2D = Draw2DStub()
    sys.modules["rdkit.Chem.Draw"] = stub


def expand(start_smiles: str) -> dict:
    _prepare_imports()
    from rdkit import Chem, RDLogger
    from minedatabase.pickaxe import Pickaxe

    RDLogger.DisableLog("rdApp.warning")
    mol = Chem.MolFromSmiles(start_smiles)
    if mol is None:
        raise ValueError("invalid starting SMILES")
    canonical_start = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
    WORK.mkdir(parents=True, exist_ok=True)
    start_file = WORK / "start.csv"
    start_file.write_text(f'id,smiles\nSTART,"{canonical_start}"\n', encoding="utf-8")

    captured = io.StringIO()
    with contextlib.redirect_stdout(captured), contextlib.redirect_stderr(captured):
        px = Pickaxe(
            rule_list=str(RULES),
            coreactant_list=str(COFACTORS),
            errors=False,
            quiet=True,
            react_targets=False,
            filter_after_final_gen=False,
        )
        px.load_compound_set(str(start_file))
        start_ids = {
            cid for cid, row in px.compounds.items()
            if row.get("Type") == "Starting Compound"
        }
        px.transform_all(processes=1, generations=1)

    predictions = []
    seen = set()
    for reaction in px.reactions.values():
        reactant_ids = {cid for _stoich, cid in reaction.get("Reactants", [])}
        if not (reactant_ids & start_ids):
            continue
        operators = sorted(str(x) for x in (reaction.get("Operators") or []))
        reaction_smiles = str(reaction.get("SMILES_rxn") or "")
        for _stoich, cid in reaction.get("Products", []):
            row = px.compounds.get(cid) or {}
            if row.get("Type") == "Coreactant":
                continue
            raw_smiles = str(row.get("SMILES") or "")
            pmol = Chem.MolFromSmiles(raw_smiles)
            if pmol is None:
                continue
            product_smiles = Chem.MolToSmiles(pmol, canonical=True, isomericSmiles=True)
            if product_smiles == canonical_start:
                continue
            key = (product_smiles, tuple(operators))
            if key in seen:
                continue
            seen.add(key)
            predictions.append({
                "product_smiles": product_smiles,
                "rules": operators,
                "reaction_smiles": reaction_smiles,
            })
    return {
        "engine": "MINE/Pickaxe",
        "generation": 1,
        "operators": len(px.operators),
        "generated_compounds": sum(1 for row in px.compounds.values() if row.get("Generation") == 1),
        "generated_reactions": len(px.reactions),
        "predictions": predictions,
    }


def main() -> None:
    payload = json.loads(sys.stdin.read() or "{}")
    result = expand(str(payload.get("start_smiles") or ""))
    sys.stdout.write(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
