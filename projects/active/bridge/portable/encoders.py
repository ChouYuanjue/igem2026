from __future__ import annotations

import contextlib
import hashlib
import re
from pathlib import Path

import numpy as np
import torch
from rdkit import Chem


VALID_AA=re.compile(r"^[ACDEFGHIKLMNPQRSTVWYBXZJUO]+$")


def clean_sequence(value: object) -> str:
    return "".join(str(value).upper().split()).rstrip("*")


def canonical_or_raw_reaction(raw: object) -> str:
    """Canonicalize both reaction sides without application/runtime dependencies."""
    text=str(raw).strip()
    if not text or ">>" not in text:
        return text
    parts=text.split(">")
    if len(parts)<3:
        return text
    sides=[]
    for raw_side in (parts[0],parts[-1]):
        pieces=[x.strip() for x in raw_side.split(".") if x.strip()]
        if not pieces:
            return text
        converted=[]
        for smiles in pieces:
            mol=Chem.MolFromSmiles(smiles)
            if mol is None:
                return text
            Chem.RemoveStereochemistry(mol)
            converted.append(Chem.MolToSmiles(mol))
        sides.append(".".join(sorted(converted)))
    return ">>".join(sides)


def _digest(model_name: str, sequence: str) -> str:
    return hashlib.sha256(
        f"portable-esmc-mean-v1\0{model_name}\0{sequence}".encode()
    ).hexdigest()


def _load_cached(cache_dir: Path | None, model_name: str, sequence: str) -> np.ndarray | None:
    if cache_dir is None:
        return None
    path=cache_dir/f"{_digest(model_name,sequence)}.npy"
    if not path.is_file():
        return None
    x=np.asarray(np.load(path),dtype=np.float32)
    return x if x.ndim==1 and np.all(np.isfinite(x)) else None


def _save_cached(cache_dir: Path | None, model_name: str, sequence: str, vector: np.ndarray) -> None:
    if cache_dir is None:
        return
    cache_dir.mkdir(parents=True,exist_ok=True)
    path=cache_dir/f"{_digest(model_name,sequence)}.npy"
    tmp=path.with_suffix(".npy.tmp")
    with tmp.open("wb") as fh:
        np.save(fh,np.asarray(vector,dtype=np.float32))
    tmp.replace(path)


def _batched_mean_embeddings(model, sequences: list[str]) -> np.ndarray:
    if not sequences:
        return np.empty((0,0),dtype=np.float32)
    tokens=model._tokenize(sequences)
    pad=model.tokenizer.pad_token_id
    bos=model.tokenizer.bos_token_id
    eos=model.tokenizer.eos_token_id
    if pad is None:
        raise ValueError("ESM-C tokenizer has no pad token")
    dev=next(model.parameters()).device
    autocast=(
        torch.autocast(device_type=dev.type,dtype=torch.bfloat16)
        if dev.type=="cuda"
        else contextlib.nullcontext()
    )
    with torch.no_grad(),autocast:
        emb=model.embed(tokens)
        emb,_,_=model.transformer(emb,sequence_id=tokens.eq(pad))
    mask=tokens.ne(pad)
    if bos is not None:
        mask &= tokens.ne(bos)
    if eos is not None:
        mask &= tokens.ne(eos)
    denom=mask.sum(dim=1,keepdim=True).clamp_min(1)
    pooled=(emb.float()*mask.unsqueeze(-1)).sum(dim=1)/denom
    return pooled.cpu().numpy().astype(np.float32)


def build_length_batches(
    items: list[tuple[str,str]],
    max_batch_tokens: int,
    max_batch_size: int,
) -> list[list[tuple[str,str]]]:
    if max_batch_tokens<=0 or max_batch_size<=0:
        raise ValueError("batch token and size limits must be positive")
    ordered=sorted(items,key=lambda x:(len(x[1]),x[0]))
    batches=[]; current=[]; current_max=0
    for item in ordered:
        proposed=max(current_max,len(item[1])+2)
        size=len(current)+1
        if current and (size>max_batch_size or proposed*size>max_batch_tokens):
            batches.append(current); current=[]; current_max=0
        current.append(item); current_max=max(current_max,len(item[1])+2)
    if current:
        batches.append(current)
    return batches


def encode_esmc_mean(
    sequences: list[str],
    *,
    model_name: str="esmc_600m",
    device: str="cpu",
    cache_dir: str | Path | None=None,
    max_batch_tokens: int=2048,
    max_batch_size: int=16,
) -> tuple[np.ndarray,list[dict[str,object]]]:
    """Portable ESM-C mean embedding with deterministic length batching/cache."""
    cleaned=[clean_sequence(x) for x in sequences]
    for i,seq in enumerate(cleaned):
        if not seq or not VALID_AA.fullmatch(seq):
            raise ValueError(f"invalid amino-acid sequence at row {i}")
    cache=Path(cache_dir) if cache_dir is not None else None
    output: list[np.ndarray | None]=[None]*len(cleaned)
    audits=[]
    pending=[]
    for i,seq in enumerate(cleaned):
        cached=_load_cached(cache,model_name,seq)
        if cached is not None:
            output[i]=cached
            audits.append({"row":i,"status":"cache_hit","length":len(seq)})
        else:
            pending.append((i,seq))
    if pending:
        from esm.models.esmc import ESMC
        model=ESMC.from_pretrained(model_name).eval().to(device)
        for batch in build_length_batches(
            pending,max_batch_tokens,max_batch_size
        ):
            vectors=_batched_mean_embeddings(model,[x[1] for x in batch])
            if len(vectors)!=len(batch):
                raise RuntimeError("ESM-C batch output row mismatch")
            for (i,seq),vector in zip(batch,vectors,strict=True):
                output[i]=vector
                _save_cached(cache,model_name,seq,vector)
                audits.append({"row":i,"status":"encoded","length":len(seq)})
    if any(x is None for x in output):
        raise RuntimeError("portable ESM-C encoder left unresolved rows")
    matrix=np.stack([np.asarray(x,dtype=np.float32) for x in output])
    norm=np.linalg.norm(matrix,axis=1,keepdims=True)
    if np.any(norm[:,0]<=1e-12):
        raise ValueError("ESM-C returned a zero feature vector")
    return (matrix/norm).astype(np.float32),sorted(audits,key=lambda x:int(x["row"]))
