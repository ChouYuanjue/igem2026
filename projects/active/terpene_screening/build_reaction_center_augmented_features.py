from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_BASE = ROOT / "data/catalyst_candidate_universes/general_merged/reaction_features/drfp_categorical_rdkitplus_v1"
DEFAULT_MAPPING = ROOT / "data/external/rxnmapper_current/general_merged_v1/mapped_reactions.csv"
DEFAULT_OUTPUT = ROOT / "data/catalyst_candidate_universes/general_merged/reaction_features/drfp_categorical_rdkitplus_center_v1"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _atom_map(mol: Chem.Mol) -> dict[int, int]:
    out: dict[int, int] = {}
    for atom in mol.GetAtoms():
        if not atom.HasProp("molAtomMapNumber"):
            continue
        value = int(atom.GetProp("molAtomMapNumber"))
        if value > 0:
            out[value] = int(atom.GetIdx())
    return out


def _bond_map(mol: Chem.Mol) -> dict[tuple[int, int], float]:
    out: dict[tuple[int, int], float] = {}
    for bond in mol.GetBonds():
        a = bond.GetBeginAtom(); b = bond.GetEndAtom()
        if not a.HasProp("molAtomMapNumber") or not b.HasProp("molAtomMapNumber"):
            continue
        ma = int(a.GetProp("molAtomMapNumber")); mb = int(b.GetProp("molAtomMapNumber"))
        if ma <= 0 or mb <= 0:
            continue
        out[tuple(sorted((ma, mb)))] = float(bond.GetBondTypeAsDouble())
    return out


def _atom_state(atom: Chem.Atom) -> tuple[int, int, int, int, int]:
    return (
        int(atom.GetAtomicNum()),
        int(atom.GetFormalCharge()),
        int(atom.GetTotalNumHs()),
        int(atom.GetIsAromatic()),
        int(atom.GetChiralTag()),
    )


def _token_bin(token: str, size: int) -> int:
    digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") % size


def reaction_center_features(
    mapped_reaction: str,
    *,
    center_fp_size: int = 512,
    token_dim: int = 256,
    radius: int = 2,
) -> tuple[np.ndarray, dict[str, object]]:
    """Deterministic reaction-center features from an atom-mapped reaction.

    No learned reaction/protein association model is used. The two Morgan blocks
    are centered only on atoms whose bonds or local atom state change. The final
    block hashes explicit bond/atom transitions into a fixed binary vocabulary.
    """
    if center_fp_size <= 0 or token_dim <= 0 or radius <= 0:
        raise ValueError("center_fp_size, token_dim and radius must be positive")
    parts = str(mapped_reaction).split(">>")
    if len(parts) != 2:
        raise ValueError("mapped reaction must contain exactly one >>")
    reactant = Chem.MolFromSmiles(parts[0])
    product = Chem.MolFromSmiles(parts[1])
    if reactant is None or product is None:
        raise ValueError("RDKit could not parse mapped reaction")

    rmap = _atom_map(reactant); pmap = _atom_map(product)
    rbonds = _bond_map(reactant); pbonds = _bond_map(product)
    changed_maps: set[int] = set()
    transition_tokens: list[str] = []
    changed_bonds = 0
    for pair in sorted(set(rbonds) | set(pbonds)):
        old = float(rbonds.get(pair, 0.0)); new = float(pbonds.get(pair, 0.0))
        if old == new:
            continue
        changed_bonds += 1
        changed_maps.update(pair)
        atoms = []
        for map_id in pair:
            if map_id in pmap:
                atoms.append(product.GetAtomWithIdx(pmap[map_id]).GetAtomicNum())
            elif map_id in rmap:
                atoms.append(reactant.GetAtomWithIdx(rmap[map_id]).GetAtomicNum())
            else:
                atoms.append(0)
        transition_tokens.append(
            f"bond:{min(atoms)}-{max(atoms)}:{old:g}>{new:g}"
        )

    changed_atom_states = 0
    for map_id in sorted(set(rmap) & set(pmap)):
        old = _atom_state(reactant.GetAtomWithIdx(rmap[map_id]))
        new = _atom_state(product.GetAtomWithIdx(pmap[map_id]))
        if old == new:
            continue
        changed_atom_states += 1
        changed_maps.add(map_id)
        transition_tokens.append(f"atom:{old}>{new}")

    generator = rdFingerprintGenerator.GetMorganGenerator(
        radius=radius, fpSize=center_fp_size, includeChirality=True
    )
    before_centers = [rmap[x] for x in sorted(changed_maps) if x in rmap]
    after_centers = [pmap[x] for x in sorted(changed_maps) if x in pmap]
    before = np.zeros(center_fp_size, dtype=np.float32)
    after = np.zeros(center_fp_size, dtype=np.float32)
    if before_centers:
        before[:] = generator.GetFingerprintAsNumPy(reactant, fromAtoms=before_centers)
    if after_centers:
        after[:] = generator.GetFingerprintAsNumPy(product, fromAtoms=after_centers)
    tokens = np.zeros(token_dim, dtype=np.float32)
    for token in transition_tokens:
        tokens[_token_bin(token, token_dim)] = 1.0
    feature = np.concatenate([before, after, tokens]).astype(np.float32, copy=False)
    audit = {
        "mapped_reactant_atoms": int(len(rmap)),
        "mapped_product_atoms": int(len(pmap)),
        "changed_map_count": int(len(changed_maps)),
        "changed_bond_count": int(changed_bonds),
        "changed_atom_state_count": int(changed_atom_states),
        "transition_token_count": int(len(transition_tokens)),
        "feature_nonzero": int(np.count_nonzero(feature)),
    }
    return feature, audit


def main() -> None:
    parser = argparse.ArgumentParser(description="Append deterministic atom-mapped reaction-center features to RDKit+ Catalyst reaction features.")
    parser.add_argument("--base-feature-dir", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--mapping-csv", type=Path, default=DEFAULT_MAPPING)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--center-fp-size", type=int, default=512)
    parser.add_argument("--token-dim", type=int, default=256)
    parser.add_argument("--radius", type=int, default=2)
    args = parser.parse_args()

    base_dir=args.base_feature_dir.resolve(); mapping_path=args.mapping_csv.resolve(); out=args.output_dir.resolve()
    out.mkdir(parents=True,exist_ok=True)
    entries=pd.read_csv(base_dir/'entries.csv',dtype={'reaction_id':str}).sort_values('row').reset_index(drop=True)
    base=np.load(base_dir/'reaction_feature_matrix.npy').astype(np.float32)
    if len(entries)!=len(base): raise ValueError('base entries/features differ in row count')
    mapping=pd.read_csv(mapping_path,dtype=str).fillna('')
    required={'reaction_id','mapped_rxn','success'}
    if not required <= set(mapping.columns): raise ValueError(f'mapping file missing {sorted(required-set(mapping.columns))}')
    mapping=mapping.drop_duplicates('reaction_id',keep='last').set_index('reaction_id')
    extension_dim=2*args.center_fp_size+args.token_dim
    center=np.zeros((len(entries),extension_dim),dtype=np.float32)
    audit=[]
    for row,rid in enumerate(entries.reaction_id.astype(str)):
        status='zero_fallback'; warning='missing_mapping'; detail={}
        if rid in mapping.index and str(mapping.at[rid,'success']).lower()=='true':
            try:
                vector,detail=reaction_center_features(
                    str(mapping.at[rid,'mapped_rxn']),center_fp_size=args.center_fp_size,
                    token_dim=args.token_dim,radius=args.radius,
                )
                center[row]=vector; status='valid'; warning=''
            except Exception as exc:
                warning=f'{type(exc).__name__}:{exc}'[:1000]
        audit.append({'row':row,'reaction_id':rid,'status':status,'warning':warning,**detail})
    augmented=np.concatenate([base,center],axis=1).astype(np.float32,copy=False)
    np.save(out/'reaction_feature_matrix.npy',augmented)
    entries.to_csv(out/'entries.csv',index=False)
    pd.DataFrame(audit).to_csv(out/'audit.csv',index=False)

    schema=json.loads((base_dir/'feature_schema.json').read_text(encoding='utf-8'))
    schema['reaction_feature_dimension']=int(augmented.shape[1])
    schema['reaction_feature_mode_extension']='append_atom_mapped_reaction_center_v1'
    schema['reaction_center_extension']={
        'dimension':extension_dim,'center_fp_size_each_side':args.center_fp_size,
        'token_dim':args.token_dim,'radius':args.radius,'include_chirality':True,
        'mapping_role':'chemical preprocessing only; no enzyme-reaction association scores',
    }
    (out/'feature_schema.json').write_text(json.dumps(schema,indent=2),encoding='utf-8')
    base_manifest=json.loads((base_dir/'manifest.json').read_text(encoding='utf-8'))
    valid=sum(x['status']=='valid' for x in audit)
    nonempty=sum(x.get('feature_nonzero',0)>0 for x in audit if x['status']=='valid')
    manifest={
        'version':'general-merged-reaction-features-rdkitplus-center-v1',
        'base_feature_dir':str(base_dir),'base_feature_manifest_sha256':sha256_file(base_dir/'manifest.json'),
        'mapping_csv':str(mapping_path),'mapping_csv_sha256':sha256_file(mapping_path),
        'feature_dimension':int(augmented.shape[1]),'base_dimension':int(base.shape[1]),
        'reaction_center_dimension':extension_dim,
        'reaction_center_settings':{
            'center_fp_size_each_side':args.center_fp_size,'center_morgan_radius':args.radius,
            'center_morgan_include_chirality':True,'transition_token_dim':args.token_dim,
            'changed_atom_definition':'bond-order/presence changes plus atomic-number/formal-charge/total-H/aromatic/chiral-state changes',
            'token_hash':'blake2b-64 modulo token_dim; binary occupancy',
        },
        'mapping_provenance':'precomputed RXNMapper general_merged_v1 used only for atom correspondence; no learned association score or enzyme label enters features',
        'reaction_count':int(len(entries)),'valid_reaction_center':int(valid),
        'zero_fallback_reaction_center':int(len(entries)-valid),'nonempty_reaction_center':int(nonempty),
    }
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print(json.dumps(manifest,indent=2))

if __name__=='__main__':
    main()
