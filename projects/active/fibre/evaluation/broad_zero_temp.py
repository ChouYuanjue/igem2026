from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from projects.active.fibre.geometry.broad_section import (
    group_reactions_by_protein_support,
    section_from_group_distances,
)
from projects.active.fibre.geometry.multiscale import resolution_product_cross_energy
from projects.active.fibre.runtime.cli import load_protein_library
from projects.active.fibre.runtime.ranking_metrics import (
    candidate_ranking_context,
    positive_ranks,
    summarize_query_metrics,
)

ROOT=Path(__file__).resolve().parents[4]
DEV_ROOT=ROOT/"results/comprehensive_enzgfm_center_top1_v1/dev"
PROTEINS=ROOT/"data/catalyst_candidate_universes/general_merged/proteins"
REACTIONS=ROOT/"data/catalyst_candidate_universes/general_merged/reaction_features/drfp_categorical_rdkitplus_center_v1"
OUT=ROOT/"results/fibre_broad_zero_temp_dev_v1"
V8_QUERY_METRICS=ROOT/"results/geometric_product_flow_clean_dev_v8/development_oof_query_metrics.csv"
V8_RESULT=ROOT/"results/geometric_product_flow_clean_dev_v8/development_result.json"
BOOTSTRAP_REPLICATES=20_000
BOOTSTRAP_SEED=20260919


def sha256(path: Path) -> str:
    h=hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda:handle.read(1<<20),b""):
            h.update(chunk)
    return h.hexdigest()


def fold_pairs(fold: int) -> tuple[pd.DataFrame,pd.DataFrame]:
    d=DEV_ROOT/"baseline_base"/f"fold{fold}"
    train=pd.read_csv(d/"training_pairs.csv",dtype=str).fillna("").drop_duplicates(
        ["protein_id","reaction_id"]
    )
    dev=pd.read_csv(d/"dev_pairs.csv",dtype=str).fillna("").drop_duplicates(
        ["protein_id","reaction_id"]
    )
    return train,dev


def build_support(fold: int, candidate_ids: list[str]):
    train,_=fold_pairs(fold)
    pindex={x:i for i,x in enumerate(candidate_ids)}
    missing=sorted(set(train.protein_id.astype(str))-set(pindex))
    if missing:
        raise RuntimeError(f"train proteins missing from candidate library: {missing[:10]}")
    train_reactions=sorted(train.reaction_id.astype(str).unique())
    rlocal={x:i for i,x in enumerate(train_reactions)}
    pairs=np.asarray([
        (rlocal[str(x.reaction_id)],pindex[str(x.protein_id)])
        for x in train.itertuples(index=False)
    ],dtype=np.int64)
    groups=group_reactions_by_protein_support(
        pairs,reaction_support_size=len(train_reactions)
    )
    support_global=np.unique(pairs[:,1])
    spos={int(g):i for i,g in enumerate(support_global)}
    membership_support=[]
    membership_group=[]
    group_ptr=[0]
    group_protein_global=[]
    for gi,members in enumerate(groups.protein_groups):
        ordered=np.asarray(sorted(map(int,members)),dtype=np.int64)
        group_protein_global.extend(ordered.tolist())
        membership_support.extend([spos[int(g)] for g in ordered])
        membership_group.extend([gi]*len(ordered))
        group_ptr.append(len(group_protein_global))
    return {
        "train":train,
        "train_reactions":train_reactions,
        "pairs":pairs,
        "groups":groups,
        "support_global":support_global,
        "membership_support":np.asarray(membership_support,dtype=np.int64),
        "membership_group":np.asarray(membership_group,dtype=np.int64),
        "group_ptr":np.asarray(group_ptr,dtype=np.int64),
        "group_protein_global":np.asarray(group_protein_global,dtype=np.int64),
    }


def grouped_distance_block(
    candidate_block: torch.Tensor,
    support: torch.Tensor,
    membership_support: torch.Tensor,
    membership_group: torch.Tensor,
    group_count: int,
) -> torch.Tensor:
    similarity=candidate_block@support.T
    member_similarity=similarity.index_select(1,membership_support)
    out=torch.full(
        (candidate_block.shape[0],int(group_count)),
        -torch.inf,
        dtype=similarity.dtype,
        device=similarity.device,
    )
    out.scatter_reduce_(
        1,
        membership_group.unsqueeze(0).expand(candidate_block.shape[0],-1),
        member_similarity,
        reduce="amax",
        include_self=True,
    )
    if not bool(torch.isfinite(out).all()):
        raise RuntimeError("support group without finite protein similarity")
    return torch.clamp(2.0-2.0*out,min=0.0)


def prepare_protein_cache(fold: int, device_name: str, batch_size: int) -> None:
    t0=time.time()
    matrix,candidate_ids=load_protein_library(PROTEINS)
    support=build_support(fold,candidate_ids)
    fold_out=OUT/f"fold{fold}"
    fold_out.mkdir(parents=True,exist_ok=True)

    device=torch.device(device_name)
    if device.type=="cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if device.type=="cuda":
        torch.backends.cuda.matmul.allow_tf32=False
    support_tensor=torch.from_numpy(
        np.asarray(matrix[support["support_global"]],dtype=np.float32)
    ).to(device)
    ms=torch.from_numpy(support["membership_support"]).to(device)
    mg=torch.from_numpy(support["membership_group"]).to(device)
    group_count=support["groups"].group_count

    # Real-data parity before creating the multi-GB cache.
    parity_n=min(16,len(candidate_ids))
    q=torch.from_numpy(np.asarray(matrix[:parity_n],dtype=np.float32)).to(device)
    with torch.no_grad():
        gpu=grouped_distance_block(q,support_tensor,ms,mg,group_count).cpu().numpy()
    dense_similarity=np.asarray(matrix[:parity_n],dtype=np.float32) @ np.asarray(
        matrix[support["support_global"]],dtype=np.float32
    ).T
    dense_group=np.empty_like(gpu)
    for gi in range(group_count):
        a,b=map(int,support["group_ptr"][gi:gi+2])
        globals_=support["group_protein_global"][a:b]
        cols=np.asarray([np.searchsorted(support["support_global"],g) for g in globals_],dtype=int)
        dense_group[:,gi]=np.maximum(0.0,2.0-2.0*np.max(dense_similarity[:,cols],axis=1))
    parity_max=float(np.max(np.abs(gpu-dense_group)))
    if parity_max>5e-5:
        raise RuntimeError(f"GPU grouped distance parity failed: max_abs={parity_max}")

    shape=(len(candidate_ids),group_count)
    cache_path=fold_out/"candidate_to_protein_support_group_sq.npy"
    mmap=np.lib.format.open_memmap(cache_path,mode="w+",dtype=np.float32,shape=shape)
    marginal=np.empty(len(candidate_ids),dtype=np.float32)
    with torch.no_grad():
        for start in range(0,len(candidate_ids),int(batch_size)):
            stop=min(start+int(batch_size),len(candidate_ids))
            qb=torch.from_numpy(
                np.asarray(matrix[start:stop],dtype=np.float32)
            ).to(device)
            dist=grouped_distance_block(
                qb,support_tensor,ms,mg,group_count
            ).cpu().numpy()
            mmap[start:stop]=dist
            marginal[start:stop]=np.min(dist,axis=1)
            if start==0 or stop==len(candidate_ids) or (start//int(batch_size))%10==0:
                print(
                    f"protein-group-cache fold={fold} {stop}/{len(candidate_ids)}",
                    flush=True,
                )
    mmap.flush()
    np.save(fold_out/"candidate_protein_marginal_sq.npy",marginal)
    np.save(fold_out/"reaction_to_group.npy",support["groups"].reaction_to_group)
    np.save(fold_out/"support_protein_global_rows.npy",support["support_global"])
    np.save(fold_out/"group_ptr.npy",support["group_ptr"])
    np.save(fold_out/"group_protein_global_rows.npy",support["group_protein_global"])
    pd.DataFrame({
        "row":np.arange(len(support["train_reactions"]),dtype=int),
        "reaction_id":support["train_reactions"],
        "group":support["groups"].reaction_to_group,
    }).to_csv(fold_out/"train_reactions.csv",index=False)
    manifest={
        "version":"fibre-broad-zero-temp-protein-cache-v1",
        "fold":int(fold),
        "candidate_count":len(candidate_ids),
        "protein_dimension":int(matrix.shape[1]),
        "train_positive_pairs":int(len(support["pairs"])),
        "train_reaction_support":int(len(support["train_reactions"])),
        "train_protein_support":int(len(support["support_global"])),
        "exact_protein_support_groups":int(group_count),
        "cache_shape":list(shape),
        "cache_dtype":"float32",
        "protein_distance":"squared chordal distance on L2-normalized ESM-C",
        "grouping":"reactions with exactly identical positive protein-support sets; lossless for min-plus section",
        "gpu_dense_parity_candidates":int(parity_n),
        "gpu_dense_parity_max_abs":parity_max,
        "candidate_library":str(PROTEINS),
        "candidate_entries_sha256":sha256(PROTEINS/"entries.csv"),
        "elapsed_seconds":time.time()-t0,
    }
    (fold_out/"protein_cache_manifest.json").write_text(
        json.dumps(manifest,indent=2)+chr(10)
    )
    print(json.dumps(manifest,indent=2),flush=True)


def reaction_view_blocks():
    matrix=np.load(REACTIONS/"reaction_feature_matrix.npy",mmap_mode="r")
    entries=pd.read_csv(REACTIONS/"entries.csv",dtype=str).fillna("")
    manifest=json.loads((REACTIONS/"manifest.json").read_text())
    center_dim=int(manifest["reaction_center_dimension"])
    base_dim=int(manifest["base_dimension"])
    rdkit_manifest=json.loads((Path(manifest["base_feature_dir"])/"manifest.json").read_text())
    pre_rdkit_dim=int(rdkit_manifest["base_dimension"])
    rdkit_dim=int(rdkit_manifest["rdkitplus_dimension"])
    cat_manifest=json.loads((Path(rdkit_manifest["base_feature_dir"])/"manifest.json").read_text())
    drfp_dim=int(cat_manifest["contract"]["drfp_dimension"])
    if base_dim-pre_rdkit_dim!=rdkit_dim or matrix.shape[1]-base_dim!=center_dim:
        raise RuntimeError("reaction feature block geometry drifted")
    views=[
        np.asarray(matrix[:,:drfp_dim],dtype=np.float32),
        np.asarray(matrix[:,pre_rdkit_dim:base_dim],dtype=np.float32),
        np.asarray(matrix[:,base_dim:base_dim+center_dim],dtype=np.float32),
    ]
    ids=entries.reaction_id.astype(str).tolist()
    return views,ids,["drfp_global","rdkitplus_whole","mapped_center"]


def binary_jaccard_cross_matrix_gpu(
    query: np.ndarray,
    reference: np.ndarray,
    *,
    device: torch.device,
    query_batch_size: int=128,
) -> tuple[np.ndarray,np.ndarray,np.ndarray]:
    r=(np.asarray(reference)>0).astype(np.float32,copy=False)
    q=(np.asarray(query)>0).astype(np.float32,copy=False)
    rmass=r.sum(axis=1).astype(np.float32)
    qmass=q.sum(axis=1).astype(np.float32)
    r_available=rmass>0
    q_available=qmass>0
    rt=torch.from_numpy(r).to(device)
    rm=torch.from_numpy(rmass).to(device)
    out=np.empty((len(q),len(r)),dtype=np.float32)
    with torch.no_grad():
        for start in range(0,len(q),int(query_batch_size)):
            stop=min(start+int(query_batch_size),len(q))
            qt=torch.from_numpy(q[start:stop]).to(device)
            qm=torch.from_numpy(qmass[start:stop]).to(device)
            inter=qt@rt.T
            union=qm[:,None]+rm[None,:]-inter
            sim=torch.where(union>0,inter/union,torch.zeros_like(union))
            out[start:stop]=(1.0-torch.clamp(sim,0.0,1.0)).cpu().numpy()
    return out,q_available,r_available


def evaluate_r2e(
    fold: int,
    device_name: str,
    *,
    query_batch_size: int = 8,
    group_batch_size: int = 128,
) -> None:
    t0=time.time()
    fold_out=OUT/f"fold{fold}"
    cache_np=np.load(
        fold_out/"candidate_to_protein_support_group_sq.npy",mmap_mode="c"
    )
    reaction_to_group=np.load(fold_out/"reaction_to_group.npy")
    train_reactions=pd.read_csv(
        fold_out/"train_reactions.csv",dtype=str
    ).reaction_id.astype(str).tolist()

    matrix,candidate_ids=load_protein_library(PROTEINS)
    del matrix
    _,dev=fold_pairs(fold)
    query_ids=sorted(dev.reaction_id.astype(str).unique())
    positives=dev.groupby("reaction_id").protein_id.apply(
        lambda x:set(map(str,x))
    ).to_dict()
    candidate_index,lexical=candidate_ranking_context(candidate_ids)

    views,reaction_ids,view_names=reaction_view_blocks()
    rindex={x:i for i,x in enumerate(reaction_ids)}
    missing=sorted((set(train_reactions)|set(query_ids))-set(rindex))
    if missing:
        raise RuntimeError(f"reaction library missing benchmark reactions: {missing[:10]}")
    tr=np.asarray([rindex[x] for x in train_reactions],dtype=np.int64)
    qr=np.asarray([rindex[x] for x in query_ids],dtype=np.int64)
    device=torch.device(device_name)
    if device.type=="cuda":
        torch.backends.cuda.matmul.allow_tf32=False

    cross=[];qavail=[];ravail=[]
    for name,view in zip(view_names,views):
        d,qa,ra=binary_jaccard_cross_matrix_gpu(
            view[qr],view[tr],device=device
        )
        cross.append(d);qavail.append(qa);ravail.append(ra)
        print(name,"cross",d.shape,"available",int(qa.sum()),int(ra.sum()),flush=True)

    # Build every query-to-support-group reaction anchor first. A group is absent
    # for a query only when every reaction in that exact protein-support group is
    # unavailable under the query's observed reaction coordinates.
    group_count=int(cache_np.shape[1])
    group_anchor=np.full((len(query_ids),group_count),np.inf,dtype=np.float32)
    query_marginal=np.full(len(query_ids),np.inf,dtype=np.float32)
    diag_info=[]
    for qi,qid in enumerate(query_ids):
        energy,info=resolution_product_cross_energy(
            [d[qi] for d in cross],
            [bool(a[qi]) for a in qavail],
            ravail,
        )
        finite=np.isfinite(energy)
        if not np.any(finite):
            raise RuntimeError(f"{qid}: no finite reaction support")
        np.minimum.at(group_anchor[qi],reaction_to_group[finite],energy[finite].astype(np.float32))
        query_marginal[qi]=float(np.min(energy[finite]))
        diag_info.append((info,int(finite.sum()),int(np.isfinite(group_anchor[qi]).sum())))

    # Keep the complete protein-side grouped distance cache on the GPU. The
    # candidate marginal is a property of Omega_E and therefore never depends on
    # reaction-query availability.
    cache=torch.from_numpy(np.asarray(cache_np,dtype=np.float32)).to(device)
    candidate_marginal=torch.min(cache,dim=1).values
    anchors=torch.from_numpy(group_anchor).to(device)
    qm=torch.from_numpy(query_marginal).to(device)

    records=[];diag=[]
    with torch.no_grad():
        for q0 in range(0,len(query_ids),int(query_batch_size)):
            q1=min(q0+int(query_batch_size),len(query_ids))
            batch=q1-q0
            joint=torch.full(
                (batch,cache.shape[0]),torch.inf,dtype=cache.dtype,device=device
            )
            for g0 in range(0,group_count,int(group_batch_size)):
                g1=min(g0+int(group_batch_size),group_count)
                a=anchors[q0:q1,g0:g1]
                finite=torch.isfinite(a)
                if not bool(finite.any()):
                    continue
                # [B,C,G] temporary is bounded by query_batch_size*candidate_count*
                # group_batch_size and remains entirely on device.
                local=cache[:,g0:g1].T.unsqueeze(0) + a.unsqueeze(2)
                local=torch.where(
                    finite.unsqueeze(2),local,
                    torch.full_like(local,torch.inf)
                )
                block_min=torch.min(local,dim=1).values
                joint=torch.minimum(joint,block_min)
            defect=torch.clamp(
                joint-qm[q0:q1,None]-candidate_marginal[None,:],
                min=0.0,
            ).cpu().numpy()
            for local_i,qi in enumerate(range(q0,q1)):
                qid=query_ids[qi]
                scores=-np.asarray(defect[local_i],dtype=np.float64)
                ranks,_=positive_ranks(
                    scores,candidate_ids,positives[qid],
                    candidate_index=candidate_index,lexical_order=lexical,
                )
                from projects.active.fibre.runtime.ranking_metrics import evaluate_full_candidate_ranks
                records.append({
                    "fold":fold,"query_id":qid,
                    **evaluate_full_candidate_ranks(ranks,len(candidate_ids))
                })
                info,finite_count,finite_groups=diag_info[qi]
                diag.append({
                    "fold":fold,"query_id":qid,
                    "finite_reaction_support":finite_count,
                    "finite_support_groups":finite_groups,
                    "query_marginal_sq":float(query_marginal[qi]),
                    "view_resolution":json.dumps(info["view_resolution"]),
                    "view_scales":json.dumps(info["view_scales"]),
                    "best_defect":float(np.min(defect[local_i])),
                    "zero_defect_candidates":int(np.sum(defect[local_i]==0)),
                })
            print(f"zero-temp fold={fold} {q1}/{len(query_ids)}",flush=True)

    frame=pd.DataFrame(records)
    diagnostics=pd.DataFrame(diag)
    frame.to_csv(fold_out/"r2e_query_metrics.csv",index=False)
    diagnostics.to_csv(fold_out/"r2e_diagnostics.csv",index=False)
    summary=summarize_query_metrics(frame)
    result={
        "version":"fibre-broad-zero-temp-r2e-dev-v1",
        "fold":int(fold),
        "method":"exact zero-temperature FIBRE correspondence section",
        "protein_geometry":"squared chordal distance on complete L2-normalized ESM-C",
        "reaction_geometry":"v8 perplexity-contraction cross energy on DRFP global + RDKitPlus whole + mapped reaction center",
        "positive_measure":"fold training positives only",
        "candidate_count":len(candidate_ids),
        "query_count":len(query_ids),
        "train_reaction_support":len(train_reactions),
        "exact_protein_support_groups":group_count,
        "candidate_marginal_scope":"all fold training positive proteins; independent of reaction-query availability",
        "external_metrics_used":False,
        "metrics":summary,
        "elapsed_seconds":time.time()-t0,
    }
    (fold_out/"r2e_summary.json").write_text(json.dumps(result,indent=2)+chr(10))
    print(json.dumps(result,indent=2),flush=True)


def _paired_bootstrap(delta: np.ndarray) -> dict[str, object]:
    delta=np.asarray(delta,dtype=np.float64).reshape(-1)
    rng=np.random.default_rng(BOOTSTRAP_SEED)
    n=len(delta)
    samples=np.empty(BOOTSTRAP_REPLICATES,dtype=np.float64)
    for i in range(BOOTSTRAP_REPLICATES):
        samples[i]=float(np.mean(delta[rng.integers(0,n,size=n)]))
    eps=1e-15
    return {
        "delta":float(np.mean(delta)),
        "ci95":[
            float(np.quantile(samples,.025)),
            float(np.quantile(samples,.975)),
        ],
        "p_delta_gt_0":float(np.mean(samples>0)),
        "improve":int(np.sum(delta>eps)),
        "tie":int(np.sum(np.abs(delta)<=eps)),
        "worse":int(np.sum(delta<-eps)),
    }


def summarize_development() -> None:
    zero=pd.concat([
        pd.read_csv(OUT/f"fold{fold}"/"r2e_query_metrics.csv")
        for fold in range(3)
    ],ignore_index=True).sort_values(["fold","query_id"]).reset_index(drop=True)
    v8=pd.read_csv(V8_QUERY_METRICS).sort_values(
        ["fold","query_id"]
    ).reset_index(drop=True)
    if len(zero)!=1903 or len(v8)!=1903:
        raise RuntimeError(
            f"matched development query count drift: zero={len(zero)} v8={len(v8)}"
        )
    if not zero[["fold","query_id"]].equals(v8[["fold","query_id"]]):
        raise RuntimeError("zero-temperature and v8 query keys are not identical")
    if not np.all(zero.candidate_count.to_numpy()==185918):
        raise RuntimeError("zero-temperature candidate support drifted")
    if not np.all(v8.candidate_count.to_numpy()==185918):
        raise RuntimeError("v8 candidate support drifted")

    metric_columns={
        "mrr":"reciprocal_rank",
        "map":"average_precision",
        "macro_roc_auc":"roc_auc",
        "ndcg_at_10":"ndcg_at_10",
        "hit_at_10":"hit_at_10",
        "hit_at_20":"hit_at_20",
        "hit_at_50":"hit_at_50",
    }
    zero_metrics={
        name:float(zero[column].mean())
        for name,column in metric_columns.items()
    }
    zero_metrics["median_best_positive_rank"]=float(
        zero.best_positive_rank.median()
    )
    v8_metrics={
        name:float(v8[column].mean())
        for name,column in metric_columns.items()
    }
    v8_metrics["median_best_positive_rank"]=float(
        v8.best_positive_rank.median()
    )
    paired={
        name:_paired_bootstrap(
            zero[column].to_numpy(dtype=np.float64)
            -v8[column].to_numpy(dtype=np.float64)
        )
        for name,column in metric_columns.items()
    }
    folds=[]
    for fold in range(3):
        z=zero[zero.fold.eq(fold)]
        v=v8[v8.fold.eq(fold)]
        folds.append({
            "fold":int(fold),
            "queries":int(len(z)),
            "zero_temperature":{
                name:float(z[column].mean())
                for name,column in metric_columns.items()
            }|{
                "median_best_positive_rank":float(
                    z.best_positive_rank.median()
                )
            },
            "v8":{
                name:float(v[column].mean())
                for name,column in metric_columns.items()
            }|{
                "median_best_positive_rank":float(
                    v.best_positive_rank.median()
                )
            },
        })

    v8_result=json.loads(V8_RESULT.read_text())
    result={
        "version":"fibre-broad-zero-temp-development-v1",
        "status":"rejected_internal_promotion",
        "comparison":"exact zero-temperature FIBRE versus frozen broad product-flow v8 on identical clean-development queries and full candidate support",
        "protocol":{
            "folds":[0,1,2],
            "queries":1903,
            "candidate_count":185918,
            "selection_scope":"internal clean-development only",
            "external_metrics_used":False,
            "positive_measure_source":"fold training positives only",
            "protein_geometry":"squared chordal distance on complete L2-normalized ESM-C",
            "reaction_geometry":"v8 perplexity-contraction cross energy on DRFP global + RDKitPlus whole + mapped reaction center",
            "operator":"exact zero-temperature correspondence defect from grouped support distances",
        },
        "zero_temperature":zero_metrics,
        "v8":v8_metrics,
        "v8_declared_method":v8_result.get("method"),
        "paired_zero_vs_v8":paired,
        "folds":folds,
        "promotion":{
            "promoted":False,
            "reason":"all three fold MRR values and pooled MRR/MAP/NDCG@10/Hit@10/20/50 are below v8; paired pooled intervals are negative for the principal retrieval metrics",
            "canonical_broad_realization":"frozen product-flow v8",
            "research_asset_role":"exact scalable correspondence primitive and negative operator-promotion evidence",
            "next_constraint":"do not repair with a hybrid score or direction-specific gate; the remaining gap is the broad nonlinear variational extension rather than dense-field feasibility",
        },
        "source_hashes":{
            "v8_query_metrics_sha256":sha256(V8_QUERY_METRICS),
            "v8_result_sha256":sha256(V8_RESULT),
        },
    }
    (OUT/"development_result.json").write_text(
        json.dumps(result,indent=2)+chr(10)
    )
    print(json.dumps(result,indent=2),flush=True)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("stage",choices=("prepare-protein","evaluate-r2e","summarize"))
    ap.add_argument("--fold",type=int,choices=(0,1,2),required=False)
    ap.add_argument("--device",default="cuda")
    ap.add_argument("--candidate-batch-size",type=int,default=1024)
    ap.add_argument("--query-batch-size",type=int,default=8)
    ap.add_argument("--group-batch-size",type=int,default=128)
    args=ap.parse_args()
    if args.stage=="summarize":
        summarize_development()
        return
    if args.fold is None:
        ap.error("--fold is required for prepare-protein/evaluate-r2e")
    if args.stage=="prepare-protein":
        prepare_protein_cache(args.fold,args.device,args.candidate_batch_size)
    else:
        evaluate_r2e(
            args.fold,args.device,
            query_batch_size=args.query_batch_size,
            group_batch_size=args.group_batch_size,
        )


if __name__=="__main__":
    main()
