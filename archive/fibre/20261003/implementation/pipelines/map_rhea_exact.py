from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from rdkit import Chem

ROOT=Path(__file__).resolve().parents[4]
RHEA=ROOT/"results/fibre_rhea_mapping_v1"
MARTS=ROOT/"data/terpene_marts/marts_reactions.tsv"


def canonical_molecule(smiles: str) -> str | None:
    mol=Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Chem.MolToSmiles(
        mol,canonical=True,isomericSmiles=False
    )


def canonical_side(side: str) -> str | None:
    values=[]
    for token in side.split("."):
        value=canonical_molecule(token)
        if value is None:
            return None
        values.append(value)
    return ".".join(sorted(values))


def canonical_reaction(reaction_smiles: str) -> str | None:
    if not reaction_smiles or ">>" not in reaction_smiles:
        return None
    left,right=reaction_smiles.split(">>",1)
    left_key=canonical_side(left)
    right_key=canonical_side(right)
    if left_key is None or right_key is None:
        return None
    return f"{left_key}>>{right_key}"


def main() -> None:
    smiles=pd.read_csv(
        RHEA/"rhea-reaction-smiles.tsv",
        sep="\t",header=None,
        names=["rhea_directed_id","reaction_smiles"],
        dtype=str,
    ).fillna("")
    directions=pd.read_csv(
        RHEA/"rhea-directions.tsv",sep="\t",dtype=str
    ).fillna("")
    marts=pd.read_csv(MARTS,sep="\t",dtype=str).fillna("")

    directed={}
    for row in directions.itertuples(index=False):
        directed[str(row.RHEA_ID_LR)]=(str(row.RHEA_ID_MASTER),"left_to_right")
        directed[str(row.RHEA_ID_RL)]=(str(row.RHEA_ID_MASTER),"right_to_left")

    index={}
    rhea_unparseable=0
    for row in smiles.itertuples(index=False):
        key=canonical_reaction(str(row.reaction_smiles))
        if key is None:
            rhea_unparseable+=1
            continue
        master,direction=directed.get(
            str(row.rhea_directed_id),("", "unknown")
        )
        index.setdefault(key,[]).append(
            (
                str(row.rhea_directed_id),
                master,
                direction,
            )
        )

    rows=[]
    marts_unparseable=0
    for row in marts.itertuples(index=False):
        key=canonical_reaction(str(row.reaction_signature))
        if key is None:
            marts_unparseable+=1
            rows.append({
                "reaction_signature":str(row.reaction_signature),
                "rhea_directed_ids":"",
                "rhea_master_ids":"",
                "rhea_directions":"",
                "exact_match_count":0,
            })
            continue
        hits=index.get(key,[])
        rows.append({
            "reaction_signature":str(row.reaction_signature),
            "rhea_directed_ids":"|".join(sorted({x[0] for x in hits})),
            "rhea_master_ids":"|".join(sorted({x[1] for x in hits if x[1]})),
            "rhea_directions":"|".join(sorted({x[2] for x in hits})),
            "exact_match_count":len({x[0] for x in hits}),
        })

    mapped=pd.DataFrame(rows)
    mapped.to_csv(RHEA/"marts_exact_rhea_mapping.csv",index=False)
    summary={
        "schema":"fibre-rhea-exact-mapping-v1",
        "rhea_directed_rows":int(len(smiles)),
        "rhea_unparseable":int(rhea_unparseable),
        "marts_reaction_rows":int(len(marts)),
        "marts_unparseable_or_empty":int(marts_unparseable),
        "marts_exact_directed_matches":int(
            mapped.exact_match_count.gt(0).sum()
        ),
        "marts_exact_directed_match_fraction":float(
            mapped.exact_match_count.gt(0).mean()
        ),
        "marts_unique_single_matches":int(
            mapped.exact_match_count.eq(1).sum()
        ),
        "marts_multiple_exact_matches":int(
            mapped.exact_match_count.gt(1).sum()
        ),
        "policy":(
            "RDKit non-isomeric canonical structure identity on directed "
            "reaction sides only; no name matching, hierarchy expansion, "
            "participant dropping or direction guessing."
        ),
    }
    (RHEA/"summary.json").write_text(
        json.dumps(summary,indent=2)+"\n"
    )
    print(json.dumps(summary,indent=2))


if __name__=="__main__":
    main()
