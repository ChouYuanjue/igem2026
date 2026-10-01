from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import lmdb
import pandas as pd
import torch
from rdkit import Chem
from tqdm import tqdm
from unimol_tools import UniMolRepr

ROOT = Path(__file__).resolve().parents[4]
DEFAULT_REACTIONS = ROOT / 'data/catalyst_candidate_universes/general_merged/reactions.csv'
DEFAULT_OUTPUT = ROOT / 'data/catalyst_candidate_universes/general_merged/reaction_features/eram_unimol_v1'

# Adapted directly from YuanshengH/Dual-Enzy dataprocess.py (MIT):
# UniMolRepr(data_type='molecule', remove_hs=False, use_gpu=True), then concatenate
# the CLS representation with atomic representations for each unique molecule.


def split_reaction(smiles: str) -> tuple[list[str], list[str]]:
    left, right = str(smiles).split('>>', 1)
    return [x for x in left.split('.') if x], [x for x in right.split('.') if x]


def molecule_supported(smiles: str) -> bool:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return False
    return Chem.AddHs(mol).GetNumAtoms() <= 256


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--reactions', type=Path, default=DEFAULT_REACTIONS)
    ap.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    ap.add_argument('--batch-size', type=int, default=64)
    args = ap.parse_args()

    reactions = pd.read_csv(args.reactions, dtype=str).fillna('')
    if reactions.reaction_id.duplicated().any():
        raise ValueError('reaction_id must be unique')
    parsed: list[tuple[list[str], list[str]]] = []
    all_smiles: set[str] = set()
    for value in reactions.reaction_smiles.astype(str):
        reactants, products = split_reaction(value)
        parsed.append((reactants, products))
        all_smiles.update(reactants)
        all_smiles.update(products)

    smiles = sorted(all_smiles)
    ids = {value: i for i, value in enumerate(smiles)}
    supported = {value: molecule_supported(value) for value in smiles}
    args.output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({
        'compound_id': range(len(smiles)),
        'smiles': smiles,
        'supported': [supported[x] for x in smiles],
    }).to_csv(args.output / 'compounds.csv', index=False)

    reaction_rows = []
    for row, (reactants, products) in zip(reactions.itertuples(index=False), parsed, strict=True):
        ok = all(supported[x] for x in reactants + products)
        reaction_rows.append({
            'reaction_id': str(row.reaction_id),
            'reactant_ids': json.dumps([ids[x] for x in reactants]),
            'product_ids': json.dumps([ids[x] for x in products]),
            'eram_available': ok,
        })
    pd.DataFrame(reaction_rows).to_csv(args.output / 'reactions.csv', index=False)

    env = lmdb.open(str(args.output / 'molecules.lmdb'), map_size=64 * 1024**3)
    with env.begin() as txn:
        missing = [s for s in smiles if supported[s] and txn.get(str(ids[s]).encode()) is None]
    if missing:
        model = UniMolRepr(data_type='molecule', remove_hs=False, use_gpu=True)
        for start in tqdm(range(0, len(missing), args.batch_size), desc='UniMol'):
            batch = missing[start:start + args.batch_size]
            reprs = model.get_repr(batch, return_atomic_reprs=True)
            cls_repr = torch.as_tensor(reprs['cls_repr'])
            with env.begin(write=True) as txn:
                for j, value in enumerate(batch):
                    atom_repr = torch.as_tensor(reprs['atomic_reprs'][j])
                    token_repr = torch.cat([cls_repr[j].unsqueeze(0), atom_repr], dim=0).float()
                    txn.put(str(ids[value]).encode(), pickle.dumps(token_repr, protocol=pickle.HIGHEST_PROTOCOL))
    env.sync(); env.close()

    manifest = {
        'schema': 'fibre-eram-unimol-v1',
        'source_reactions': str(args.reactions.relative_to(ROOT)),
        'reaction_count': int(len(reactions)),
        'compound_count': int(len(smiles)),
        'supported_compounds': int(sum(supported.values())),
        'supported_reactions': int(sum(x['eram_available'] for x in reaction_rows)),
        'upstream': 'https://github.com/YuanshengH/Dual-Enzy',
        'upstream_preprocess': 'dataprocess.py',
        'representation': 'UniMol CLS + atomic representations, remove_hs=False',
    }
    (args.output / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
