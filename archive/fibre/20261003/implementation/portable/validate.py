from __future__ import annotations

import argparse
import json
from pathlib import Path

from .contracts import load_and_validate


def validate_dataset(
    proteins: str | Path,
    reactions: str | Path,
    pairs: str | Path,
) -> dict:
    t=load_and_validate(proteins,reactions,pairs)
    return {
        "schema":"fibre-portable-dataset-validation-v1",
        "status":"valid",
        "protein_count":int(len(t.proteins)),
        "reaction_count":int(len(t.reactions)),
        "positive_pair_count":int(len(t.pairs)),
        "positive_protein_support_count":int(t.pairs.protein_id.nunique()),
        "positive_reaction_support_count":int(t.pairs.reaction_id.nunique()),
        "negative_pair_requirement":"none",
        "missing_semantics":"unknown/unobserved is not negative",
    }


def main() -> None:
    ap=argparse.ArgumentParser(description="Validate generic FIBRE dataset tables.")
    ap.add_argument("--proteins",type=Path,required=True)
    ap.add_argument("--reactions",type=Path,required=True)
    ap.add_argument("--pairs",type=Path,required=True)
    ap.add_argument("--output",type=Path)
    a=ap.parse_args()
    report=validate_dataset(a.proteins,a.reactions,a.pairs)
    text=json.dumps(report,indent=2)+"\n"
    if a.output:
        a.output.parent.mkdir(parents=True,exist_ok=True)
        a.output.write_text(text)
    print(text,end="")


if __name__=="__main__":
    main()
