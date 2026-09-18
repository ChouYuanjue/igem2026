from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from Bio import Align
from Bio.Align import substitution_matrices

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from reproducibility.bime_rank.scripts import run_bime_r2e_clipzyme_expert_v1 as clip
from reproducibility.bime_rank.scripts import run_r2e_lambdarank_fusion_v1 as base
from reproducibility.bime_rank.scripts.extract_esmc_motif_context_embeddings import select_motif_contexts
from projects.active.terpene_screening.runtime.reaction_features import (
    _atom_map, _atom_state, _bond_map,
)
from rdkit import Chem
from projects.active.terpene_screening.evidence.mechanism import family_motif_observables
from projects.active.terpene_screening.geometry.multiscale import (
    _perplexity_contraction_from_energy,
    _self_tuning_scale,
    binary_jaccard_cross_distance,
)
from projects.active.terpene_screening.runtime.cli import (
    load_feature_schema,
    load_registered_reaction_feature_library,
)

SEQUENCES = ROOT / "data/catalyst_candidate_universes/general_merged/protein_sequences.tsv"
METADATA = ROOT / "data/catalyst_candidate_universes/general_merged/protein_metadata.csv"
REACTION_META = ROOT / "data/catalyst_candidate_universes/general_merged/reactions.csv"
REACTION_MAPPING = ROOT / "data/external/rxnmapper_current/general_merged_v1/mapped_reactions.csv"
PROTEIN_ENTRIES = ROOT / "data/catalyst_candidate_universes/general_merged/proteins/entries.csv"
PROTEIN_EMBEDDINGS = ROOT / "data/catalyst_candidate_universes/general_merged/proteins/embeddings.npy"



def _atom_state_label(state: tuple[int, int, int, int, int]) -> dict[str, object]:
    atomic_num, charge, hydrogens, aromatic, chiral = state
    symbol = Chem.GetPeriodicTable().GetElementSymbol(int(atomic_num)) if atomic_num > 0 else "?"
    return {
        "element": symbol,
        "formal_charge": int(charge),
        "total_h": int(hydrogens),
        "aromatic": bool(aromatic),
        "chiral_tag": int(chiral),
    }


def _bond_order_label(value: float) -> str:
    labels = {0.0: "absent", 1.0: "single", 1.5: "aromatic", 2.0: "double", 3.0: "triple"}
    return labels.get(float(value), f"order_{float(value):g}")


def _reaction_center_edit_summary(mapped_reaction: str) -> dict[str, object]:
    """Human-readable view of the exact reaction-center definition used by g_R.

    This reports graph edits, not a transition-state mechanism. Atom-map indices
    are correspondence labels only and never become ranking features.
    """
    parts = str(mapped_reaction).split(">>")
    if len(parts) != 2:
        return {"available": False, "reason": "invalid_mapped_reaction"}
    reactant = Chem.MolFromSmiles(parts[0]); product = Chem.MolFromSmiles(parts[1])
    if reactant is None or product is None:
        return {"available": False, "reason": "rdkit_parse_failure"}
    rmap = _atom_map(reactant); pmap = _atom_map(product)
    rbonds = _bond_map(reactant); pbonds = _bond_map(product)
    bond_changes=[]; signatures=[]; changed_maps=set()
    for pair in sorted(set(rbonds) | set(pbonds)):
        old=float(rbonds.get(pair,0.0)); new=float(pbonds.get(pair,0.0))
        if old == new: continue
        changed_maps.update(pair)
        symbols=[]
        for mid in pair:
            if mid in pmap: atom=product.GetAtomWithIdx(pmap[mid])
            elif mid in rmap: atom=reactant.GetAtomWithIdx(rmap[mid])
            else: atom=None
            symbols.append(atom.GetSymbol() if atom is not None else "?")
        sig=f"bond:{'-'.join(sorted(symbols))}:{_bond_order_label(old)}>{_bond_order_label(new)}"
        signatures.append(sig)
        bond_changes.append({
            "mapped_atoms":[int(pair[0]),int(pair[1])],
            "elements":symbols,
            "before":_bond_order_label(old),
            "after":_bond_order_label(new),
            "signature":sig,
        })
    atom_changes=[]
    for mid in sorted(set(rmap) & set(pmap)):
        old=_atom_state(reactant.GetAtomWithIdx(rmap[mid])); new=_atom_state(product.GetAtomWithIdx(pmap[mid]))
        if old == new: continue
        changed_maps.add(mid)
        old_l=_atom_state_label(old); new_l=_atom_state_label(new)
        sig=f"atom:{old_l['element']}:{old_l['formal_charge']}>{new_l['formal_charge']}:H{old_l['total_h']}>{new_l['total_h']}:arom{int(old_l['aromatic'])}>{int(new_l['aromatic'])}"
        signatures.append(sig)
        atom_changes.append({"mapped_atom":int(mid),"before":old_l,"after":new_l,"signature":sig})
    return {
        "available": True,
        "interpretation": "atom-mapped local graph edits used by the reaction-center coordinate; not a transition-state mechanism",
        "mapped_reactant_atoms":int(len(rmap)),
        "mapped_product_atoms":int(len(pmap)),
        "changed_mapped_atoms":sorted(map(int,changed_maps)),
        "bond_changes":bond_changes,
        "atom_state_changes":atom_changes,
        "transition_signatures":sorted(set(signatures)),
    }

def _reaction_blocks(features: np.ndarray) -> list[np.ndarray]:
    manifest = json.loads((base.REACTIONS / "manifest.json").read_text())
    center_dim = int(manifest["reaction_center_dimension"])
    base_dim = int(manifest["base_dimension"])
    rdkit_manifest = json.loads((Path(manifest["base_feature_dir"]) / "manifest.json").read_text())
    pre_rdkit_dim = int(rdkit_manifest["base_dimension"])
    rdkit_dim = int(rdkit_manifest["rdkitplus_dimension"])
    categorical_manifest = json.loads((Path(rdkit_manifest["base_feature_dir"]) / "manifest.json").read_text())
    drfp_dim = int(categorical_manifest["contract"]["drfp_dimension"])
    if base_dim - pre_rdkit_dim != rdkit_dim or features.shape[1] - base_dim != center_dim:
        raise RuntimeError("reaction feature block dimensions drifted")
    return [
        np.asarray(features[:, :drfp_dim], dtype=np.float32),
        np.asarray(features[:, pre_rdkit_dim:base_dim], dtype=np.float32),
        np.asarray(features[:, base_dim:base_dim + center_dim], dtype=np.float32),
    ]


def _reaction_local_energy(
    query_row: int,
    train_rows: np.ndarray,
    views: list[np.ndarray],
    *,
    epsilon: float = 1e-8,
) -> tuple[np.ndarray, dict[str, object]]:
    weighted_sum = np.zeros(len(train_rows), dtype=np.float64)
    precision_sum = np.zeros(len(train_rows), dtype=np.float64)
    fallback_sum = np.zeros(len(train_rows), dtype=np.float64)
    fallback_count = np.zeros(len(train_rows), dtype=np.int16)
    similarities: list[np.ndarray] = []
    resolutions: list[float] = []
    scales: list[float | None] = []
    for view in views:
        d, q_ok, r_ok = binary_jaccard_cross_distance(view[query_row], view[train_rows])
        valid = np.asarray(r_ok, dtype=bool) & np.isfinite(d)
        similarities.append(np.where(valid, 1.0 - d, np.nan))
        if (not q_ok) or not np.any(valid):
            resolutions.append(0.0); scales.append(None); continue
        support = int(valid.sum())
        kk = min(support, max(1, int(math.ceil(math.sqrt(support)))))
        vals = d[valid]
        sigma = float(np.partition(vals, kk - 1)[kk - 1])
        positive = vals[np.isfinite(vals) & (vals > epsilon)]
        if (not np.isfinite(sigma)) or sigma <= epsilon:
            sigma = float(np.median(positive)) if len(positive) else 1.0
        e = np.square(d[valid]) / max(sigma * sigma, epsilon)
        resolution = _perplexity_contraction_from_energy(e)
        resolutions.append(float(resolution)); scales.append(float(sigma))
        fallback_sum[valid] += e; fallback_count[valid] += 1
        if resolution > epsilon:
            weighted_sum[valid] += resolution * e
            precision_sum[valid] += resolution
    effective = np.full(len(train_rows), np.inf, dtype=np.float64)
    active = precision_sum > epsilon
    effective[active] = weighted_sum[active] / precision_sum[active]
    fallback = (~active) & (fallback_count > 0)
    effective[fallback] = fallback_sum[fallback] / fallback_count[fallback]
    return effective, {
        "view_names": ["drfp_global", "rdkitplus_whole", "mapped_center"],
        "view_resolution": resolutions,
        "view_scales": scales,
        "view_similarity": similarities,
    }


def _combined_protein_similarity(
    rows_a: np.ndarray,
    rows_b: np.ndarray,
    esmc: np.ndarray,
    *,
    clip_pt: torch.Tensor,
    clip_lookup: np.ndarray,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with torch.no_grad():
        a = F.normalize(torch.as_tensor(np.asarray(esmc[rows_a], dtype=np.float32).copy(), device=device), dim=1)
        b = F.normalize(torch.as_tensor(np.asarray(esmc[rows_b], dtype=np.float32).copy(), device=device), dim=1)
        esmc_sim = (a @ b.T).cpu().numpy().astype(np.float64)
    combined = esmc_sim.copy()
    structural = np.full_like(combined, np.nan, dtype=np.float64)
    apos = clip_lookup[rows_a]; bpos = clip_lookup[rows_b]
    ai = np.flatnonzero(apos >= 0); bj = np.flatnonzero(bpos >= 0)
    if len(ai) and len(bj):
        with torch.no_grad():
            aa = F.normalize(clip_pt[torch.as_tensor(apos[ai], dtype=torch.long, device=device)].float(), dim=1)
            bb = F.normalize(clip_pt[torch.as_tensor(bpos[bj], dtype=torch.long, device=device)].float(), dim=1)
            ss = (aa @ bb.T).cpu().numpy().astype(np.float64)
        structural[np.ix_(ai, bj)] = ss
        combined[np.ix_(ai, bj)] = 0.5 * (combined[np.ix_(ai, bj)] + ss)
    return combined, esmc_sim, structural


def _protein_local_energy(
    candidate_row: int,
    source_rows: np.ndarray,
    esmc: np.ndarray,
    *,
    clip_pt: torch.Tensor,
    clip_lookup: np.ndarray,
    device: torch.device,
    epsilon: float = 1e-8,
) -> tuple[np.ndarray, dict[str, object]]:
    source_rows = np.asarray(source_rows, dtype=np.int64)
    source_sim, _, _ = _combined_protein_similarity(
        source_rows, source_rows, esmc, clip_pt=clip_pt, clip_lookup=clip_lookup, device=device
    )
    source_dist = np.sqrt(np.maximum(2.0 - 2.0 * np.clip(source_sim, -1.0, 1.0), 0.0))
    source_sigma = _self_tuning_scale(source_dist, np.ones(len(source_rows), dtype=bool), epsilon=epsilon)
    cross_sim, esmc_sim, structural = _combined_protein_similarity(
        np.asarray([candidate_row]), source_rows, esmc,
        clip_pt=clip_pt, clip_lookup=clip_lookup, device=device,
    )
    d = np.sqrt(np.maximum(2.0 - 2.0 * np.clip(cross_sim[0], -1.0, 1.0), 0.0))
    kk = min(len(d), max(1, int(math.ceil(math.sqrt(len(d))))))
    candidate_sigma = float(np.partition(d, kk - 1)[kk - 1])
    positive = d[np.isfinite(d) & (d > epsilon)]
    if (not np.isfinite(candidate_sigma)) or candidate_sigma <= epsilon:
        candidate_sigma = float(np.median(positive)) if len(positive) else 1.0
    denom = candidate_sigma * source_sigma
    energy = np.divide(
        np.square(d), denom,
        out=np.full_like(d, np.inf, dtype=np.float64),
        where=np.isfinite(denom) & (denom > epsilon),
    )
    return energy, {
        "combined_similarity": cross_sim[0],
        "esmc_similarity": esmc_sim[0],
        "clip_similarity": structural[0],
        "candidate_scale": candidate_sigma,
    }


def _alignment_identity(a: str, b: str) -> dict[str, float | int]:
    """Report both full-length and local BLOSUM62 homology observables.

    The full-length value is conservative for proteins with different domain
    architectures. The local value and two coverages expose whether a strong
    homologous domain exists without turning homology into a ranking feature.
    """
    matrix = substitution_matrices.load("BLOSUM62")
    global_aligner = Align.PairwiseAligner()
    global_aligner.mode = "global"
    global_aligner.substitution_matrix = matrix
    global_aligner.open_gap_score = -10.0
    global_aligner.extend_gap_score = -0.5
    galn = global_aligner.align(a, b)[0]
    gcounts = galn.counts()
    glength = int(galn.length)

    local_aligner = Align.PairwiseAligner()
    local_aligner.mode = "local"
    local_aligner.substitution_matrix = matrix
    local_aligner.open_gap_score = -10.0
    local_aligner.extend_gap_score = -0.5
    laln = local_aligner.align(a, b)[0]
    lcounts = laln.counts()
    llength = int(laln.length)
    aligned_a = int(sum(int(end - start) for start, end in laln.aligned[0]))
    aligned_b = int(sum(int(end - start) for start, end in laln.aligned[1]))
    return {
        "global_alignment_length": glength,
        "global_identities": int(gcounts.identities),
        "global_sequence_identity": float(gcounts.identities / max(glength, 1)),
        "local_alignment_length": llength,
        "local_identities": int(lcounts.identities),
        "local_sequence_identity": float(lcounts.identities / max(llength, 1)),
        "candidate_local_coverage": float(aligned_a / max(len(a), 1)),
        "source_local_coverage": float(aligned_b / max(len(b), 1)),
        "local_alignment_score": float(laln.score),
    }


TPS_PFAMS = {"PF01397", "PF03936", "PF13243", "PF13249", "PF19086"}


def _tps_domain_scope(pfam: str, domain_family: str) -> bool:
    domains = {x.strip() for x in str(pfam).split(";") if x.strip()}
    if domains & TPS_PFAMS:
        return True
    family = str(domain_family).lower()
    return any(token in family for token in ("tps", "terpene", "triterpene", "bacterial_classi", "plant_like_classi", "osc"))


def _motif_summary(sequence: str, *, pfam: str = "", domain_family: str = "") -> dict[str, object]:
    """Interpret TPS motifs only inside a TPS-supported domain context.

    Short patterns such as DDxxD or QW occur by chance in unrelated proteins.
    Reporting them as catalytic evidence outside a TPS family would be biologically
    misleading, even though they are harmless to ranking. Domain scope therefore
    controls *interpretation*, not scoring.
    """
    scoped = _tps_domain_scope(pfam, domain_family)
    if not scoped:
        return {
            "interpretation_scope": "not_interpreted_without_TPS_domain_evidence",
            "tps_domain_supported": False,
        }
    c = select_motif_contexts(sequence)
    return {
        "interpretation_scope": "TPS_domain_supported",
        "tps_domain_supported": True,
        "ddxxd": bool(c["ddxxd"]["all_positions"]),
        "nse_or_dte": bool(c["nse_dte"]["all_positions"]),
        "dxdd": bool(c["dxdd"]["all_positions"]),
        "qw_count": int(len(c["qw"]["all_positions"])),
        "classI_pair": bool(c["classI_pair"]["present"]),
        "classI_pair_distance": int(c["classI_pair"]["pair_distance"]),
    }



def _family_specific_observables(meta: dict[str, object], sequence: str) -> dict[str, object]:
    """Expose TPS mechanism observables only for an annotated TPS family.

    This delegates to the evidence-only family-aware scanner used by the
    application mechanism sheet. It is intentionally separate from historical
    motif descriptor assets so known variants such as geosmin DDHFLE are not
    incorrectly treated as absent. No motif changes ranking.
    """
    family = str(meta.get("domain_family", "") or "")
    pfam = str(meta.get("pfam", "") or "")
    if not _tps_domain_scope(pfam, family):
        return {
            "scope": "not_applicable_without_tps_family_annotation",
            "domain_family": family,
            "pfam": pfam,
        }
    out = family_motif_observables(sequence, family)
    out["pfam"] = pfam
    return out

def build_witness(
    fold: int,
    query_reaction: str,
    candidate_protein: str,
    *,
    top_witnesses: int = 5,
    device_name: str | None = None,
) -> dict[str, object]:
    device = torch.device(device_name or ("cuda" if torch.cuda.is_available() else "cpu"))
    entries = pd.read_csv(PROTEIN_ENTRIES, dtype=str)
    ids = entries.Entry.astype(str).tolist()
    pindex = {p: i for i, p in enumerate(ids)}
    if candidate_protein not in pindex:
        raise KeyError(f"candidate protein not in canonical universe: {candidate_protein}")
    schema = load_feature_schema(base.PRIMARY_MODELS / f"fold{fold}")
    reaction_features, reaction_ids = load_registered_reaction_feature_library(base.REACTIONS, schema)
    rindex = {r: i for i, r in enumerate(reaction_ids)}
    if query_reaction not in rindex:
        raise KeyError(f"query reaction not registered: {query_reaction}")
    train = pd.read_csv(base.DEV_ROOT / "baseline_base" / f"fold{fold}" / "training_pairs.csv", dtype=str).fillna("")
    train = train[train.reaction_id.isin(rindex) & train.protein_id.isin(pindex)].copy()
    train_reactions = sorted(train.reaction_id.unique())
    train_rows = np.asarray([rindex[x] for x in train_reactions], dtype=np.int64)
    renergy, rinfo = _reaction_local_energy(rindex[query_reaction], train_rows, _reaction_blocks(reaction_features))
    finite = np.flatnonzero(np.isfinite(renergy))
    reaction_k = min(len(finite), max(1, int(math.ceil(math.sqrt(len(finite))))))
    chosen = finite[np.argpartition(renergy[finite], reaction_k - 1)[:reaction_k]]
    chosen = chosen[np.argsort(renergy[chosen], kind="stable")]
    chosen_reactions = {train_reactions[int(i)]: float(renergy[int(i)]) for i in chosen}
    local_pairs = train[train.reaction_id.isin(chosen_reactions)].drop_duplicates(["reaction_id", "protein_id"]).copy()
    source_ids = sorted(local_pairs.protein_id.unique())
    source_rows = np.asarray([pindex[x] for x in source_ids], dtype=np.int64)
    source_lookup = {p: i for i, p in enumerate(source_ids)}
    esmc = np.load(PROTEIN_EMBEDDINGS, mmap_mode="r")
    clip_pt, _, clip_lookup, _, _ = clip._load_clip_assets(ids, device)
    penergy, pinfo = _protein_local_energy(
        pindex[candidate_protein], source_rows, esmc,
        clip_pt=clip_pt, clip_lookup=clip_lookup, device=device,
    )
    finite_reaction_energy = renergy[np.isfinite(renergy)]
    reaction_nearest_energy = float(np.min(finite_reaction_energy))
    reaction_chart_boundary_energy = float(np.max(renergy[chosen]))
    reaction_locality_ratio = float(
        reaction_nearest_energy / max(reaction_chart_boundary_energy, 1e-12)
    )
    finite_protein_energy = penergy[np.isfinite(penergy)]
    protein_nearest_energy = float(np.min(finite_protein_energy))
    protein_k = min(
        len(finite_protein_energy),
        max(1, int(math.ceil(math.sqrt(len(finite_protein_energy))))),
    )
    protein_chart_boundary_energy = float(
        np.partition(finite_protein_energy, protein_k - 1)[protein_k - 1]
    )
    protein_locality_ratio = float(
        protein_nearest_energy / max(protein_chart_boundary_energy, 1e-12)
    )
    local_pairs["reaction_energy"] = local_pairs.reaction_id.map(chosen_reactions).astype(float)
    local_pairs["protein_energy"] = local_pairs.protein_id.map(lambda x: float(penergy[source_lookup[x]]))
    local_pairs["product_support_energy"] = local_pairs.reaction_energy + local_pairs.protein_energy
    local_pairs = local_pairs.sort_values(
        ["product_support_energy", "reaction_energy", "protein_energy", "reaction_id", "protein_id"],
        kind="stable",
    )

    seq = pd.read_csv(SEQUENCES, sep="\t", dtype=str).fillna("").set_index("protein_id")["sequence"].to_dict()
    meta = pd.read_csv(METADATA, dtype=str).fillna("").set_index("protein_id")
    rxmeta = pd.read_csv(REACTION_META, dtype=str).fillna("").set_index("reaction_id")
    mapping_frame = pd.read_csv(REACTION_MAPPING, dtype=str).fillna("")
    mapped_rxn = dict(zip(mapping_frame.reaction_id.astype(str), mapping_frame.mapped_rxn.astype(str)))
    query_center_edit = _reaction_center_edit_summary(mapped_rxn.get(query_reaction, ""))
    candidate_sequence = seq[candidate_protein]
    candidate_meta = meta.loc[candidate_protein].to_dict() if candidate_protein in meta.index else {}
    selected = []
    train_pos = {r: i for i, r in enumerate(train_reactions)}
    for row in local_pairs.head(top_witnesses).itertuples(index=False):
        sid = str(row.protein_id); srid = str(row.reaction_id)
        si = source_lookup[sid]; ri = train_pos[srid]
        source_sequence = seq[sid]
        source_meta = meta.loc[sid].to_dict() if sid in meta.index else {}
        selected.append({
            "observed_positive_pair": {"reaction_id": srid, "protein_id": sid},
            "same_protein_as_prediction": bool(sid == candidate_protein),
            "product_support_energy": float(row.product_support_energy),
            "reaction_energy": float(row.reaction_energy),
            "protein_energy": float(row.protein_energy),
            "reaction_similarity": {
                name: (None if not np.isfinite(rinfo["view_similarity"][j][ri]) else float(rinfo["view_similarity"][j][ri]))
                for j, name in enumerate(rinfo["view_names"])
            },
            "protein_similarity": {
                "esmc_cosine": float(pinfo["esmc_similarity"][si]),
                "clip_cosine": None if not np.isfinite(pinfo["clip_similarity"][si]) else float(pinfo["clip_similarity"][si]),
                "canonical_direct_sum_cosine": float(pinfo["combined_similarity"][si]),
                **_alignment_identity(candidate_sequence, source_sequence),
            },
            "source_protein_annotation": {
                "pfam": source_meta.get("pfam", ""),
                "domain_family": source_meta.get("domain_family", ""),
                "source_layer": source_meta.get("source_layer", ""),
                "evidence_scope": source_meta.get("evidence_scope", ""),
                "family_specific_observables": _family_specific_observables(source_meta, source_sequence),
            },
            "source_reaction_center_edit": (
                lambda source_edit: {
                    **source_edit,
                    "shared_transition_signatures_with_query": sorted(
                        set(query_center_edit.get("transition_signatures", []))
                        & set(source_edit.get("transition_signatures", []))
                    ),
                }
            )(_reaction_center_edit_summary(mapped_rxn.get(srid, ""))),
            "source_reaction_smiles": rxmeta.loc[srid, "reaction_smiles"] if srid in rxmeta.index else "",
        })
    consensus = {
        "witness_count": int(len(selected)),
        "unique_source_reactions": int(len({w["observed_positive_pair"]["reaction_id"] for w in selected})),
        "unique_source_proteins": int(len({w["observed_positive_pair"]["protein_id"] for w in selected})),
        "same_protein_precedent_count": int(sum(w.get("same_protein_as_prediction", False) for w in selected)),
        "support_provenance": (
            "same_protein_precedent_present"
            if any(w.get("same_protein_as_prediction", False) for w in selected)
            else "distinct_protein_precedents_only"
        ),
        "median_product_support_energy": float(np.median([w["product_support_energy"] for w in selected])) if selected else None,
        "median_sequence_identity": float(np.median([w["protein_similarity"]["global_sequence_identity"] for w in selected])) if selected else None,
        "median_esmc_cosine": float(np.median([w["protein_similarity"]["esmc_cosine"] for w in selected])) if selected else None,
        "structure_observed_witnesses": int(sum(w["protein_similarity"]["clip_cosine"] is not None for w in selected)),
        "median_clip_cosine_when_observed": (
            float(np.median([w["protein_similarity"]["clip_cosine"] for w in selected if w["protein_similarity"]["clip_cosine"] is not None]))
            if any(w["protein_similarity"]["clip_cosine"] is not None for w in selected) else None
        ),
        "median_reaction_similarity": {
            name: (
                float(np.median([w["reaction_similarity"][name] for w in selected if w["reaction_similarity"][name] is not None]))
                if any(w["reaction_similarity"][name] is not None for w in selected) else None
            )
            for name in ("drfp_global", "rdkitplus_whole", "mapped_center")
        },
        "interpretation": "descriptive consistency across the geometrically nearest known positive precedents; no threshold and no effect on ranking",
    }

    return {
        "schema": "product-manifold-biological-witness-v2",
        "role": "post-hoc evidence witness on the same geometry; never used to change ranking",
        "fold": int(fold),
        "query_reaction": query_reaction,
        "query_reaction_smiles": rxmeta.loc[query_reaction, "reaction_smiles"] if query_reaction in rxmeta.index else "",
        "query_reaction_center_edit": query_center_edit,
        "candidate_protein": candidate_protein,
        "candidate_annotation": {
            "pfam": candidate_meta.get("pfam", ""),
            "domain_family": candidate_meta.get("domain_family", ""),
            "source_layer": candidate_meta.get("source_layer", ""),
            "evidence_scope": candidate_meta.get("evidence_scope", ""),
            "sequence_length": int(len(candidate_sequence)),
            "family_specific_observables": _family_specific_observables(candidate_meta, candidate_sequence),
        },
        "applicability_coordinates": {
            "interpretation": "continuous geometry only; smaller locality ratios mean denser support and no threshold changes ranking",
            "reaction_axis": {
                "nearest_support_energy": reaction_nearest_energy,
                "sqrtN_chart_boundary_energy": reaction_chart_boundary_energy,
                "nearest_to_boundary_ratio": reaction_locality_ratio,
            },
            "protein_axis": {
                "nearest_support_energy": protein_nearest_energy,
                "sqrtN_chart_boundary_energy": protein_chart_boundary_energy,
                "nearest_to_boundary_ratio": protein_locality_ratio,
                "reference_proteins": int(len(finite_protein_energy)),
            },
        },
        "reaction_geometry": {
            "local_reference_reactions": int(reaction_k),
            "view_resolution": rinfo["view_resolution"],
            "view_scales": rinfo["view_scales"],
        },
        "source_positive_pairs_considered": int(len(local_pairs)),
        "witness_consensus": consensus,
        "witnesses": selected,
        "labels_used_for_witness_selection": False,
        "ranking_modified": False,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, required=True, choices=(0, 1, 2))
    ap.add_argument("--reaction", required=True)
    ap.add_argument("--protein", required=True)
    ap.add_argument("--top-witnesses", type=int, default=5)
    ap.add_argument("--device", default=None)
    ap.add_argument("--output", type=Path, default=None)
    args = ap.parse_args()
    result = build_witness(args.fold, args.reaction, args.protein, top_witnesses=args.top_witnesses, device_name=args.device)
    text = json.dumps(result, indent=2, ensure_ascii=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
    print(text, end="")


if __name__ == "__main__":
    main()
