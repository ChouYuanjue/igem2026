from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from projects.active.fibre.runtime.cli import (
    encode_external_enzymes_with_audit,
    encode_reaction,
    load_feature_schema,
    load_models,
)


ROOT=Path(__file__).resolve().parents[4]
DEFAULT_MODEL=ROOT/"results/terpene_production_models/marts_adapted_drfp_pu"
DEFAULT_PROTEINS=ROOT/"data/terpene_marts_adaptation/protein_entities.csv"
DEFAULT_REACTIONS=ROOT/"data/terpene_marts_adaptation/reaction_entities.csv"
DEFAULT_PROTEIN_FEATURES=ROOT/"data/terpene_marts_adaptation/protein_features.npy"
DEFAULT_REACTION_FEATURES=ROOT/"data/terpene_marts_adaptation/reaction_features.npy"
DEFAULT_OUTPUT=ROOT/"results/fibre_application/tps_adapted_coordinate"


def portable_path(path: str | Path) -> str:
    p=Path(path).resolve()
    try:
        return str(p.relative_to(ROOT))
    except ValueError:
        return str(p)


def sha256_file(path: str | Path) -> str:
    h=hashlib.sha256()
    with Path(path).open("rb") as fh:
        for block in iter(lambda:fh.read(1<<20),b""):
            h.update(block)
    return h.hexdigest()


def _normalize(x: np.ndarray) -> np.ndarray:
    x=np.asarray(x,dtype=np.float32)
    norm=np.linalg.norm(x,axis=1,keepdims=True)
    if np.any(norm[:,0]<=1e-12):
        raise ValueError("TPS-adapted coordinate contains a zero vector")
    return (x/norm).astype(np.float32)


def _encode(
    model,
    values: np.ndarray,
    kind: str,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    rows=[]
    model.eval()
    with torch.no_grad():
        for start in range(0,len(values),batch_size):
            # numpy memmaps are read-only; copy the bounded batch before
            # handing it to torch so runtime projection never relies on undefined
            # writes through a non-writable view.
            batch=torch.tensor(
                np.asarray(values[start:start+batch_size],dtype=np.float32).copy(),
                dtype=torch.float32,
                device=device,
            )
            if kind=="protein":
                z=model.encode_proteins(batch)
            elif kind=="reaction":
                z=model.encode_reactions(batch)
            else:
                raise ValueError(kind)
            rows.append(z.float().cpu().numpy())
    return _normalize(np.concatenate(rows,axis=0))


def _write_csv(
    ids: list[str],
    id_column: str,
    matrix: np.ndarray,
    path: Path,
) -> None:
    frame=pd.DataFrame(
        np.asarray(matrix,dtype=np.float32),
        columns=[f"tps_{i}" for i in range(matrix.shape[1])],
    )
    frame.insert(0,id_column,ids)
    path.parent.mkdir(parents=True,exist_ok=True)
    frame.to_csv(path,index=False)


class TPSAdaptedCoordinateProjector:
    """Lazy application-time projector for the frozen TPS pair-trained towers."""

    def __init__(
        self,
        *,
        model_dir: str | Path=DEFAULT_MODEL,
        device: str | None=None,
        batch_size: int=512,
    ) -> None:
        self.model_dir=Path(model_dir).resolve()
        self.device=torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.batch_size=max(1,int(batch_size))
        self.schema=load_feature_schema(self.model_dir)
        self.models=load_models(
            self.model_dir/"models","production",self.device
        )
        if not self.models:
            raise ValueError("TPS specialization has no production checkpoints")

    def project_protein_features(self, values: np.ndarray) -> np.ndarray:
        x=np.asarray(values,dtype=np.float32)
        if x.ndim==1:
            x=x.reshape(1,-1)
        expected=int(self.schema["protein_feature_dimension"])
        if x.ndim!=2 or x.shape[1]!=expected:
            raise ValueError(
                f"TPS protein input dimension must be {expected}, got {x.shape}"
            )
        seeds=[
            _encode(model,x,"protein",self.device,self.batch_size)
            for model in self.models
        ]
        return _normalize(np.concatenate(seeds,axis=1))

    def project_reaction_features(self, values: np.ndarray) -> np.ndarray:
        x=np.asarray(values,dtype=np.float32)
        if x.ndim==1:
            x=x.reshape(1,-1)
        expected=int(self.schema["reaction_feature_dimension"])
        if x.ndim!=2 or x.shape[1]!=expected:
            raise ValueError(
                f"TPS reaction input dimension must be {expected}, got {x.shape}"
            )
        seeds=[
            _encode(model,x,"reaction",self.device,self.batch_size)
            for model in self.models
        ]
        return _normalize(np.concatenate(seeds,axis=1))

    def reaction_smiles(self, reaction_smiles: str) -> np.ndarray:
        x=np.asarray(
            encode_reaction(str(reaction_smiles),self.schema),dtype=np.float32
        )
        return self.project_reaction_features(x)[0]

    def protein_sequence(self, sequence: str) -> tuple[np.ndarray,dict]:
        frame=pd.DataFrame([{
            "enzyme_id":"application_query",
            "sequence":"".join(str(sequence).upper().split()).rstrip("*"),
        }])
        matrix,audits=encode_external_enzymes_with_audit(
            frame,str(self.device),"esmc_600m"
        )
        vector=self.project_protein_features(matrix)[0]
        audit=audits[0].__dict__ if audits else {}
        return vector,audit


def build_tps_adapted_coordinate(
    *,
    model_dir: str | Path=DEFAULT_MODEL,
    proteins_path: str | Path=DEFAULT_PROTEINS,
    reactions_path: str | Path=DEFAULT_REACTIONS,
    protein_features_path: str | Path=DEFAULT_PROTEIN_FEATURES,
    reaction_features_path: str | Path=DEFAULT_REACTION_FEATURES,
    output_dir: str | Path=DEFAULT_OUTPUT,
    device: str="cpu",
    batch_size: int=512,
) -> dict:
    """Project the current canonical TPS molecular states through legacy TPS towers.

    The source weights were trained with pair supervision on the historical
    current+MARTS union. We keep that learned domain prior, but do not revive
    its direct protein-reaction cosine score. Instead, each tower defines an
    optional coordinate on the *current canonical* protein/reaction factors.

    The three independently trained seed spaces are never averaged as raw
    vectors: seed outputs are L2-normalized, concatenated, then normalized.
    Cosine in the 768-D concatenation is therefore the mean within-seed cosine
    and does not assume seed latent bases are aligned.
    """
    model_dir=Path(model_dir).resolve()
    proteins_path=Path(proteins_path).resolve()
    reactions_path=Path(reactions_path).resolve()
    protein_features_path=Path(protein_features_path).resolve()
    reaction_features_path=Path(reaction_features_path).resolve()
    output=Path(output_dir).resolve()
    if batch_size<=0:
        raise ValueError("batch_size must be positive")

    proteins=pd.read_csv(proteins_path,dtype=str).fillna("")
    reactions=pd.read_csv(reactions_path,dtype=str).fillna("")
    if "protein_id" not in proteins or proteins.protein_id.duplicated().any():
        raise ValueError("canonical proteins require unique protein_id")
    if "reaction_id" not in reactions or reactions.reaction_id.duplicated().any():
        raise ValueError("canonical reactions require unique reaction_id")

    protein_x=np.load(protein_features_path,mmap_mode="r")
    reaction_x=np.load(reaction_features_path,mmap_mode="r")
    if protein_x.ndim!=2 or len(protein_x)!=len(proteins):
        raise ValueError("canonical protein feature matrix is not row-aligned")
    if reaction_x.ndim!=2 or len(reaction_x)!=len(reactions):
        raise ValueError("canonical reaction feature matrix is not row-aligned")

    schema=json.loads((model_dir/"feature_schema.json").read_text(encoding="utf-8"))
    if int(schema["protein_feature_dimension"])!=protein_x.shape[1]:
        raise ValueError("canonical protein feature dimension differs from TPS tower input")
    if int(schema["reaction_feature_dimension"])!=reaction_x.shape[1]:
        raise ValueError("canonical reaction feature dimension differs from TPS tower input")

    source_summary=json.loads((model_dir/"summary.json").read_text(encoding="utf-8"))
    torch_device=torch.device(device)
    models=load_models(model_dir/"models","production",torch_device)
    paths=sorted((model_dir/"models").glob("production_seed*.pt"))
    if not models or len(models)!=len(paths):
        raise ValueError("TPS specialization production checkpoint set is incomplete")

    protein_seed=[]; reaction_seed=[]; checkpoints=[]
    for model,path in zip(models,paths,strict=True):
        protein_seed.append(
            _encode(model,np.asarray(protein_x,dtype=np.float32),"protein",torch_device,batch_size)
        )
        reaction_seed.append(
            _encode(model,np.asarray(reaction_x,dtype=np.float32),"reaction",torch_device,batch_size)
        )
        checkpoints.append({
            "path":str(path.relative_to(ROOT)),
            "sha256":sha256_file(path),
        })

    protein_coordinate=_normalize(np.concatenate(protein_seed,axis=1))
    reaction_coordinate=_normalize(np.concatenate(reaction_seed,axis=1))
    output.mkdir(parents=True,exist_ok=True)
    protein_csv=output/"protein_tps_adapted.csv"
    reaction_csv=output/"reaction_tps_adapted.csv"
    _write_csv(
        proteins.protein_id.astype(str).tolist(),
        "protein_id",
        protein_coordinate,
        protein_csv,
    )
    _write_csv(
        reactions.reaction_id.astype(str).tolist(),
        "reaction_id",
        reaction_coordinate,
        reaction_csv,
    )

    manifest={
        "schema":"fibre-application-tps-adapted-coordinate-v2",
        "release_profile":"starase-application",
        "release_scope":"application_only",
        "scientific_role":(
            "application-only pair-supervised TPS-domain coordinate on the current "
            "canonical FIBRE factors, used only to refine candidates within a primary "
            "FIBRE numerical level; legacy cross-factor scoring is not used"
        ),
        "source_model_family":"marts_adapted_drfp_pu",
        "source_training":{
            "current_proteins":int(source_summary["n_current_proteins"]),
            "external_proteins":int(source_summary["n_external_proteins"]),
            "current_reactions":int(source_summary["n_current_reactions"]),
            "external_reactions":int(source_summary["n_external_reactions"]),
            "training_pairs":int(source_summary["n_training_pairs"]),
            "pair_supervised":True,
        },
        "canonical_projection":{
            "protein_states":len(proteins),
            "reaction_states":len(reactions),
            "protein_input_dimension":int(protein_x.shape[1]),
            "reaction_input_dimension":int(reaction_x.shape[1]),
            "entity_policy":(
                "project the current deduplicated molecular-state universe directly "
                "through the learned towers; historical alias/training registry rows "
                "do not become new FIBRE entities"
            ),
        },
        "allowed_in_method_package":False,
        "allowed_in_frozen_fibre_benchmark_claims":False,
        "direct_legacy_pair_score_used":False,
        "ensemble":{
            "seed_count":len(models),
            "per_seed_dimension":int(protein_seed[0].shape[1]),
            "output_dimension":int(protein_coordinate.shape[1]),
            "aggregation":"L2-normalize each seed output -> concatenate seeds -> L2-normalize",
            "reason":(
                "preserves mean within-seed cosine without assuming independently "
                "trained latent bases are aligned"
            ),
        },
        "input_sha256":{
            "canonical_proteins":sha256_file(proteins_path),
            "canonical_reactions":sha256_file(reactions_path),
            "canonical_protein_features":sha256_file(protein_features_path),
            "canonical_reaction_features":sha256_file(reaction_features_path),
            "source_training_pairs":sha256_file(model_dir/"training_pairs.csv"),
            "source_feature_schema":sha256_file(model_dir/"feature_schema.json"),
        },
        "checkpoints":checkpoints,
        "outputs":{
            "protein_feature_view":portable_path(protein_csv),
            "reaction_feature_view":portable_path(reaction_csv),
        },
        "evaluation_policy":(
            "Full-data pair-supervised application coordinates are not benchmark "
            "evidence. Reported benchmark metrics must come from fibre-reproduction."
        ),
    }
    (output/"manifest.json").write_text(
        json.dumps(manifest,indent=2)+"\n",encoding="utf-8"
    )
    return manifest


def main() -> None:
    ap=argparse.ArgumentParser(
        description=(
            "Project current canonical TPS molecular states through the full-data "
            "pair-trained TPS towers as application-only FIBRE factor coordinates."
        )
    )
    ap.add_argument("--model-dir",type=Path,default=DEFAULT_MODEL)
    ap.add_argument("--proteins",type=Path,default=DEFAULT_PROTEINS)
    ap.add_argument("--reactions",type=Path,default=DEFAULT_REACTIONS)
    ap.add_argument("--protein-features",type=Path,default=DEFAULT_PROTEIN_FEATURES)
    ap.add_argument("--reaction-features",type=Path,default=DEFAULT_REACTION_FEATURES)
    ap.add_argument("--output-dir",type=Path,default=DEFAULT_OUTPUT)
    ap.add_argument("--device",default="cpu")
    ap.add_argument("--batch-size",type=int,default=512)
    args=ap.parse_args()
    print(json.dumps(build_tps_adapted_coordinate(
        model_dir=args.model_dir,
        proteins_path=args.proteins,
        reactions_path=args.reactions,
        protein_features_path=args.protein_features,
        reaction_features_path=args.reaction_features,
        output_dir=args.output_dir,
        device=args.device,
        batch_size=args.batch_size,
    ),indent=2))


if __name__=="__main__":
    main()
