from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse.csgraph import shortest_path

from projects.active.fibre.evaluation.strict_inductive import (
    CACHE, POCKET, MOTIF, STRUCT, BUD,
    bools, chordal_full, coordinate_reference_and_cross,
    diffusion_reference_and_cross, factor_all_to_reference,
    build_protein_factor_inputs, build_reaction_factor_inputs,
    materialize_views, length_graph_from_affinity,
    score_r2e, score_e2r,
)
from projects.active.fibre.geometry.multiscale import (
    information_product_affinity,
    intrinsic_view_information,
)
from projects.active.fibre.geometry.extension import (
    attach_information_product_query,
    query_geodesic_to_reference,
)
from projects.active.fibre.geometry.correspondence import (
    reaction_to_protein_section_from_support_distances,
    protein_to_reaction_section_from_support_distances,
)
from projects.active.fibre.geometry.stratified import stratified_resolution
from projects.active.fibre.runtime.base_model import rank_metrics

ROOT=Path(__file__).resolve().parents[4]
RLOCAL=ROOT/"data/terpene_multiresolution_reaction_geometry_v2"
OUT=ROOT/"results/fibre_stratified_correspondence_strict_inductive_v1"
BOOTSTRAP_REPLICATES=20_000
BOOTSTRAP_SEED=20260919


def build_local_reaction_inputs():
    return [
        (
            "center_transition_wasserstein","coordinate",
            np.load(RLOCAL/"center_transition_wasserstein_distance.npy").astype(np.float64),
            np.load(RLOCAL/"center_transition_wasserstein_available.npy").astype(bool),
        ),
        (
            "center_transition_token","coordinate",
            np.load(RLOCAL/"center_token_jaccard_distance.npy").astype(np.float64),
            np.load(RLOCAL/"center_token_jaccard_available.npy").astype(bool),
        ),
    ]


def build_local_protein_inputs(include_motifs: bool = True):
    views=[]
    p=np.load(POCKET/"embeddings.npy",mmap_mode="r")
    pa=np.load(POCKET/"available.npy").astype(bool)
    views.append(("pocket_local_esmc","coordinate",chordal_full(p,pa),pa))
    if include_motifs:
        for name in ["typeI_aspartate","nse_dte","dxdd","qw"]:
            e=np.load(MOTIF/f"{name}_embeddings.npy",mmap_mode="r")
            a=np.load(MOTIF/f"{name}_available.npy").astype(bool)
            views.append((name,"coordinate",chordal_full(e,a),a))
    for name in ["pocket_3di","pocket_ot"]:
        s=np.load(STRUCT/name/"similarity.npy",mmap_mode="r")
        a=np.load(STRUCT/name/"available.npy").astype(bool)
        views.append((name,"diffusion",s,a))
    return views


def materialize_named_views(raw_views, ref):
    out=[]
    for name,kind,data,avail in raw_views:
        if kind=="coordinate":
            x=coordinate_reference_and_cross(data,avail,ref)
        else:
            x=diffusion_reference_and_cross(data,avail,ref)
        out.append((name,*x))
    return out


def factor_all_to_reference_information(ref, named_view_data):
    ref_dist=[x[1] for x in named_view_data]
    ref_avail=[x[2] for x in named_view_data]
    cross=[x[3] for x in named_view_data]
    qall=[x[4] for x in named_view_data]

    affinity,diag=information_product_affinity(ref_dist,ref_avail)
    length_graph,ell=length_graph_from_affinity(affinity)
    infos=[];scales=[]
    for d,a in zip(ref_dist,ref_avail):
        info,sigma=intrinsic_view_information(d,a)
        infos.append(info);scales.append(sigma)

    dref=shortest_path(length_graph,directed=False,unweighted=False)
    if not np.all(np.isfinite(dref)):
        raise RuntimeError("local reference atlas disconnected")

    n=cross[0].shape[0]
    out=np.full((n,len(ref)),np.inf,dtype=np.float32)
    refpos={int(g):i for i,g in enumerate(ref)}
    for g,local in refpos.items():
        out[g]=np.asarray(dref[local]/ell,dtype=np.float32)

    unit_graph=length_graph.copy()
    unit_graph.data/=ell
    for i in range(n):
        if i in refpos:
            continue
        qdist=[np.asarray(x[i],dtype=np.float64) for x in cross]
        qavail=[bool(x[i]) for x in qall]
        if not any(qavail):
            continue
        try:
            att=attach_information_product_query(
                qdist,qavail,ref_avail,scales,infos
            )
            from dataclasses import replace
            att=replace(att,edge_lengths=att.edge_lengths/ell)
            out[i]=query_geodesic_to_reference(unit_graph,att).astype(np.float32)
        except ValueError:
            continue

    return out,{
        "reference_count":int(len(ref)),
        "external_count":int(n-len(ref)),
        "graph_k":int(diag["graph_k"]),
        "characteristic_length":float(ell),
        "query_available_by_view":{
            name:np.asarray(q,dtype=bool)
            for (name,*_),q in zip(named_view_data,qall)
        },
        "reference_available_by_view":{
            name:np.asarray(a,dtype=bool)
            for (name,*_),a in zip(named_view_data,ref_avail)
        },
    }


def metrics_from_order(order, ids, positives):
    ordered=[ids[int(i)] for i in order]
    positive_ranks=[i+1 for i,x in enumerate(ordered) if x in positives]
    if not positive_ranks:
        return {"best_positive_rank":float("inf"),"reciprocal_rank":0.0}
    best=min(positive_ranks)
    return {"best_positive_rank":float(best),"reciprocal_rank":1.0/float(best)}


def paired_summary(q):
    rng=np.random.default_rng(BOOTSTRAP_SEED)
    out={}
    for direction,g in q.groupby("direction"):
        a=g[g.method.eq("coarse")][
            ["split_id","query_id","reciprocal_rank","best_positive_rank"]
        ].rename(columns={"reciprocal_rank":"rr_base","best_positive_rank":"rank_base"})
        b=g[g.method.eq("stratified")][
            ["split_id","query_id","reciprocal_rank","best_positive_rank"]
        ].rename(columns={"reciprocal_rank":"rr_refined","best_positive_rank":"rank_refined"})
        z=a.merge(b,on=["split_id","query_id"],validate="one_to_one")
        delta=(z.rr_refined-z.rr_base).to_numpy(float)
        samples=np.empty(BOOTSTRAP_REPLICATES)
        n=len(delta)
        for i in range(BOOTSTRAP_REPLICATES):
            samples[i]=float(np.mean(delta[rng.integers(0,n,size=n)]))
        out[direction]={
            "queries":int(n),
            "mean_rr_delta":float(np.mean(delta)),
            "rr_improve_tie_worse":[
                int(np.sum(delta>0)),int(np.sum(delta==0)),int(np.sum(delta<0))
            ],
            "bootstrap_95ci":[
                float(np.quantile(samples,.025)),float(np.quantile(samples,.975))
            ],
            "bootstrap_p_delta_gt_0":float(np.mean(samples>0)),
        }
    return out


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--protein-mode",choices=("structure","structure_and_motifs"),default="structure_and_motifs")
    ap.add_argument("--output",type=Path,default=OUT)
    args=ap.parse_args()
    out_dir=Path(args.output)
    t0=time.time()
    proteins=pd.read_csv(CACHE/"protein_entities.csv",dtype=str).fillna("")
    reactions=pd.read_csv(CACHE/"reaction_entities.csv",dtype=str).fillna("")
    pairs=pd.read_csv(CACHE/"marts_pair_folds.csv",dtype=str).fillna("")
    pairs[["protein_fold","reaction_fold"]]=pairs[
        ["protein_fold","reaction_fold"]
    ].astype(int)
    pairs["protein_seen"]=bools(pairs.protein_seen)
    pairs["reaction_seen"]=bools(pairs.reaction_seen)
    pids=proteins.protein_id.astype(str).tolist()
    rids=reactions.reaction_id.astype(str).tolist()
    pi={x:i for i,x in enumerate(pids)}
    ri={x:i for i,x in enumerate(rids)}
    pfold={str(k):int(v) for k,v in pairs[["Entry","protein_fold"]].drop_duplicates().itertuples(index=False)}
    rfold={str(k):int(v) for k,v in pairs[["rhea_id","reaction_fold"]].drop_duplicates().itertuples(index=False)}

    print("precomputing coarse and catalytic-local raw views",flush=True)
    coarse_p_raw=build_protein_factor_inputs()
    coarse_r_raw=build_reaction_factor_inputs(reactions)
    local_p_raw=build_local_protein_inputs(include_motifs=args.protein_mode=="structure_and_motifs")
    local_r_raw=build_local_reaction_inputs()

    # Nodes with no local coordinate are excluded from local reference atlases.
    local_p_any=np.logical_or.reduce([np.asarray(v[3],bool) for v in local_p_raw])
    local_r_any=np.logical_or.reduce([np.asarray(v[3],bool) for v in local_r_raw])

    rows=[];split_rows=[]
    for pf in range(5):
      pref=np.array([i for i,pid in enumerate(pids) if pfold.get(pid)!=pf],dtype=int)
      cp,cpinfo=factor_all_to_reference(pref,materialize_views(coarse_p_raw,pref))
      lpref=np.array([
          i for i,pid in enumerate(pids)
          if pfold.get(pid)!=pf and local_p_any[i]
      ],dtype=int)
      lp,lpinfo=factor_all_to_reference_information(
          lpref,materialize_named_views(local_p_raw,lpref)
      )
      pocket3di_q=np.asarray(
          lpinfo["query_available_by_view"]["pocket_3di"],dtype=bool
      )

      for rf in range(5):
        if not (pf==4 or rf==4):
            continue
        rref=np.array([i for i,rid in enumerate(rids) if rfold.get(rid)!=rf],dtype=int)
        cr,crinfo=factor_all_to_reference(rref,materialize_views(coarse_r_raw,rref))
        lrref=np.array([
            i for i,rid in enumerate(rids)
            if rfold.get(rid)!=rf and local_r_any[i]
        ],dtype=int)
        lr,lrinfo=factor_all_to_reference_information(
            lrref,materialize_named_views(local_r_raw,lrref)
        )

        sid=f"p{pf}_r{rf}"
        train=pairs[
            pairs.protein_fold.ne(pf)&pairs.reaction_fold.ne(rf)
        ].drop_duplicates(["rhea_id","Entry"])
        test=pairs[
            pairs.protein_fold.eq(pf)&pairs.reaction_fold.eq(rf)
            & ~pairs.protein_seen & ~pairs.reaction_seen
        ].drop_duplicates(["rhea_id","Entry"])

        pmap={int(g):i for i,g in enumerate(pref)}
        rmap={int(g):i for i,g in enumerate(rref)}
        Dr2=np.asarray(cr,dtype=np.float64)**2
        Dp2=np.asarray(cp,dtype=np.float64)**2

        lpmap={int(g):i for i,g in enumerate(lpref)}
        lrmap={int(g):i for i,g in enumerate(lrref)}

        # Local positive support is structure-qualified on the protein side.
        local_train=train[
            train.Entry.map(pi).map(
                lambda g: int(g) in lpmap and bool(
                    lpinfo["reference_available_by_view"]["pocket_3di"][lpmap[int(g)]]
                )
            )
        ].copy()
        if local_train.empty:
            raise RuntimeError(f"{sid}: no structure-qualified local positive support")

        r_support_global=sorted({ri[str(x)] for x in local_train.rhea_id})
        e_support_global=sorted({pi[str(x)] for x in local_train.Entry})
        r_support_pos={g:i for i,g in enumerate(r_support_global)}
        e_support_pos={g:i for i,g in enumerate(e_support_global)}
        support_pairs=np.asarray([
            (r_support_pos[ri[str(x.rhea_id)]],e_support_pos[pi[str(x.Entry)]])
            for x in local_train.itertuples(index=False)
        ],dtype=int)
        r_support_ref=np.asarray([lrmap[g] for g in r_support_global],dtype=int)
        e_support_ref=np.asarray([lpmap[g] for g in e_support_global],dtype=int)

        reaction_local_available=np.all(np.isfinite(lr[:,r_support_ref]),axis=1)
        order_bearing_protein=(
            pocket3di_q
            & np.all(np.isfinite(lp[:,e_support_ref]),axis=1)
        )

        for rid,g in test.groupby("rhea_id",sort=True):
            qg=ri[rid]
            coarse_score=score_r2e(
                qg,train,Dr2,Dp2,rmap,pmap,ri,pi,len(pids)
            )
            coarse_defect=-np.asarray(coarse_score,dtype=np.float64)
            baseline=np.argsort(coarse_defect,kind="stable")

            local_defect=np.full(len(pids),np.nan,dtype=np.float64)
            candidate_idx=np.flatnonzero(order_bearing_protein)
            if reaction_local_available[qg] and len(candidate_idx):
                sec=reaction_to_protein_section_from_support_distances(
                    np.square(lr[qg,r_support_ref]),
                    np.square(lp[np.ix_(candidate_idx,e_support_ref)]),
                    support_pairs,
                    query_index=qg,
                    candidate_indices=candidate_idx,
                )
                local_defect[candidate_idx]=sec.defect
                resolution=stratified_resolution(
                    coarse_defect,local_defect,order_bearing_protein
                )
                refined=resolution.display_order()
                refined_levels=len(resolution.locally_refined_coarse_levels)
            else:
                refined=baseline.copy(); refined_levels=0
            positives=set(g.Entry.astype(str))
            for method,order in (("coarse",baseline),("stratified",refined)):
                m=metrics_from_order(order,pids,positives)
                rows.append({
                    "split_id":sid,"direction":"reaction_to_enzyme",
                    "query_id":rid,"method":method,**m,
                    "local_refinement_applied":bool(refined_levels),
                })

        for pid,g in test.groupby("Entry",sort=True):
            qg=pi[pid]
            coarse_score=score_e2r(
                qg,train,Dr2,Dp2,rmap,pmap,ri,pi,len(rids)
            )
            coarse_defect=-np.asarray(coarse_score,dtype=np.float64)
            baseline=np.argsort(coarse_defect,kind="stable")

            if order_bearing_protein[qg]:
                local_defect=np.full(len(rids),np.nan,dtype=np.float64)
                candidate_idx=np.flatnonzero(reaction_local_available)
                sec=protein_to_reaction_section_from_support_distances(
                    np.square(lp[qg,e_support_ref]),
                    np.square(lr[np.ix_(candidate_idx,r_support_ref)]),
                    support_pairs,
                    query_index=qg,
                    candidate_indices=candidate_idx,
                )
                local_defect[candidate_idx]=sec.defect
                resolution=stratified_resolution(
                    coarse_defect,local_defect,reaction_local_available
                )
                refined=resolution.display_order()
                refined_levels=len(resolution.locally_refined_coarse_levels)
            else:
                refined=baseline.copy();refined_levels=0
            positives=set(g.rhea_id.astype(str))
            for method,order in (("coarse",baseline),("stratified",refined)):
                m=metrics_from_order(order,rids,positives)
                rows.append({
                    "split_id":sid,"direction":"enzyme_to_reaction",
                    "query_id":pid,"method":method,**m,
                    "local_refinement_applied":bool(refined_levels),
                })

        split_rows.append({
            "split_id":sid,
            "train_pairs":int(len(train)),
            "local_train_pairs":int(len(local_train)),
            "test_pairs":int(len(test)),
            "protein_overlap":len(set(train.Entry)&set(test.Entry)),
            "reaction_overlap":len(set(train.rhea_id)&set(test.rhea_id)),
            "coarse_protein_reference_count":int(cpinfo["reference_count"]),
            "coarse_reaction_reference_count":int(crinfo["reference_count"]),
            "local_protein_reference_count":int(lpinfo["reference_count"]),
            "local_reaction_reference_count":int(lrinfo["reference_count"]),
        })
        print(sid,"done",flush=True)

    q=pd.DataFrame(rows);sp=pd.DataFrame(split_rows)
    metrics=[]
    for (method,direction),g in q.groupby(["method","direction"]):
        metrics.append({
            "method":method,"direction":direction,"n_queries":int(len(g)),
            "mrr":float(g.reciprocal_rank.mean()),
            "median_rank":float(g.best_positive_rank.median()),
            "local_refinement_applied_fraction":float(g.local_refinement_applied.mean()),
        })
    paired=paired_summary(q)

    out_dir.mkdir(parents=True,exist_ok=True)
    q.to_csv(out_dir/"query_metrics.csv",index=False)
    sp.to_csv(out_dir/"split_summary.csv",index=False)
    pd.DataFrame(metrics).to_csv(out_dir/"metrics.csv",index=False)
    summary={
        "version":"fibre-stratified-correspondence-strict-inductive-v1",
        "protocol":"for each held-out factor fold, rebuild coarse and catalytic-local reference atlases without held-out entities; attach each excluded entity using only query-to-reference molecular observations; apply structure-qualified local sublevels only inside coarse numerical levels",
        "coarse_operator":"zero-temperature FIBRE correspondence defect",
        "local_operator":"same zero-temperature FIBRE correspondence defect",
        "local_reaction_views":["center_transition_wasserstein","center_transition_token"],
        "protein_mode":args.protein_mode,
        "local_protein_views":[x[0] for x in local_p_raw],
        "order_bearing_qualification":"query/candidate protein must have a pocket-3Di relational observation against the reference-only atlas; motif/sequence-only local coordinates are non-order-bearing",
        "qualification_rule_selected_on":"internal development analysis; no external-retention labels",
        "all_train_test_entity_overlaps_zero":bool(
            ((sp.protein_overlap==0)&(sp.reaction_overlap==0)).all()
        ),
        "paired":paired,
        "metrics":metrics,
        "elapsed_seconds":time.time()-t0,
    }
    (out_dir/"summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    print(json.dumps(summary,indent=2),flush=True)


if __name__=="__main__":
    main()
