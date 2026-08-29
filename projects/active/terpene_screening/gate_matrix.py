from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import rdFingerprintGenerator

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from projects.active.terpene_screening.common import (  # noqa: E402
    SOURCE_FILES,
    TERPENE_DATA_DIR,
    TERPENE_RESULTS_DIR,
    canonicalize_reaction_smiles,
    coerce_text,
    identify_terpene_columns,
    parse_uniprot_id,
    read_table,
    safe_json_dump,
    write_table,
)


GATE_MATRIX_RESULTS_DIR = PROJECT_ROOT / "results" / "terpene_gate_matrix"
GATE_MATRIX_DATA_DIR = PROJECT_ROOT / "data" / "terpene_gate_matrix"
DEFAULT_POSITIVE_PATH = SOURCE_FILES["positive_labels"]
DEFAULT_CANDIDATE_PATH = SOURCE_FILES["candidate_enzymes"]
DEFAULT_CAGE_SCORE_PATH = TERPENE_RESULTS_DIR / "all_rhea_gate" / "all_pair_scores.csv"


MORGAN_RADIUS = 2
MORGAN_BITS = 2048
KMER_SIZE = 3


@dataclass(frozen=True)
class GateSpec:
    gate_id: str
    method: str
    topk_reactions: int | None = None
    topn_candidates: int | None = None
    max_candidates: int | None = None
    score_mode: str = "balanced"


DEFAULT_GATE_SPECS: list[GateSpec] = [
    GateSpec("rxn_balanced_top5", "reaction_similarity", topk_reactions=5, score_mode="balanced"),
    GateSpec("rxn_balanced_top10", "reaction_similarity", topk_reactions=10, score_mode="balanced"),
    GateSpec("rxn_balanced_top20", "reaction_similarity", topk_reactions=20, score_mode="balanced"),
    GateSpec("rxn_balanced_top50", "reaction_similarity", topk_reactions=50, score_mode="balanced"),
    GateSpec("rxn_product_top20", "reaction_similarity", topk_reactions=20, score_mode="product"),
    GateSpec("rxn_substrate_top20", "reaction_similarity", topk_reactions=20, score_mode="substrate"),
    GateSpec("precursor_exact_top200", "precursor_exact", max_candidates=200),
    GateSpec("product_skeleton_top200", "product_skeleton", max_candidates=200),
    GateSpec("seq_kmer_top50", "sequence_kmer", topk_reactions=20, topn_candidates=50),
    GateSpec("mechanism_precursor_motif_top200", "mechanism_precursor_motif", max_candidates=200),
    GateSpec("recall_union_core", "recall_union", max_candidates=300),
    GateSpec("weighted_top50", "weighted", topn_candidates=50),
    GateSpec("weighted_top100", "weighted", topn_candidates=100),
]

RERANKERS = ["gate_score", "reaction_similarity", "sequence_kmer", "motif", "fusion", "cage_if_available"]


RDLogger.DisableLog("rdApp.*")
_generator = rdFingerprintGenerator.GetMorganGenerator(radius=MORGAN_RADIUS, fpSize=MORGAN_BITS)


def stable_random_score(*parts: str) -> float:
    payload = "||".join(parts).encode("utf-8")
    digest = hashlib.sha256(payload).hexdigest()[:16]
    return int(digest, 16) / float(16**16 - 1)


def mol_from_smiles(smiles: str) -> Chem.Mol | None:
    smiles = coerce_text(smiles)
    if not smiles:
        return None
    try:
        return Chem.MolFromSmiles(smiles)
    except Exception:
        return None


def mol_fp(smiles: str):
    mol = mol_from_smiles(smiles)
    if mol is None:
        return None
    try:
        return _generator.GetFingerprint(mol)
    except Exception:
        return None


def tanimoto(fp_a: Any, fp_b: Any) -> float:
    if fp_a is None or fp_b is None:
        return 0.0
    try:
        return float(DataStructs.TanimotoSimilarity(fp_a, fp_b))
    except Exception:
        return 0.0


def split_reaction_smiles(reaction_smiles: str) -> tuple[list[str], list[str]]:
    text = coerce_text(reaction_smiles)
    if ">>" not in text:
        return [], []
    left, right = text.split(">>", 1)
    reactants = [part.strip() for part in left.split(".") if part.strip()]
    products = [part.strip() for part in right.split(".") if part.strip()]
    return reactants, products


def best_match_similarity(a_fps: list[Any], b_fps: list[Any]) -> float:
    if not a_fps or not b_fps:
        return 0.0
    forward = [max(tanimoto(fp_a, fp_b) for fp_b in b_fps) for fp_a in a_fps]
    backward = [max(tanimoto(fp_b, fp_a) for fp_a in a_fps) for fp_b in b_fps]
    values = forward + backward
    return float(sum(values) / len(values)) if values else 0.0


def canonical_or_raw_reaction(raw_rxn: str) -> str:
    raw_rxn = coerce_text(raw_rxn)
    return canonicalize_reaction_smiles(raw_rxn) or raw_rxn


def carbon_count(smiles: str) -> int:
    mol = mol_from_smiles(smiles)
    if mol is None:
        return 0
    return int(sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() == 6))


def oxygen_count(smiles: str) -> int:
    mol = mol_from_smiles(smiles)
    if mol is None:
        return 0
    return int(sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() == 8))


def phosphorus_count(smiles: str) -> int:
    mol = mol_from_smiles(smiles)
    if mol is None:
        return 0
    return int(sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() == 15))


def ring_count(smiles: str) -> int:
    mol = mol_from_smiles(smiles)
    if mol is None:
        return 0
    try:
        return int(mol.GetRingInfo().NumRings())
    except Exception:
        return 0


def largest_organic_component(parts: Iterable[str]) -> str:
    best = ""
    best_key = (-1, -1, "")
    for part in parts:
        c = carbon_count(part)
        if c == 0:
            continue
        key = (c, len(part), part)
        if key > best_key:
            best_key = key
            best = part
    return best


def precursor_class_from_reaction(cano_rxn: str) -> str:
    reactants, _ = split_reaction_smiles(cano_rxn)
    candidates: list[int] = []
    for smi in reactants:
        c = carbon_count(smi)
        p = phosphorus_count(smi)
        if p >= 1 and c >= 5:
            candidates.append(c)
    if not candidates:
        largest = largest_organic_component(reactants)
        c = carbon_count(largest)
    else:
        c = max(candidates)
    if c <= 0:
        return "unknown"
    if c <= 6:
        return "C5_IPP_DMAPP_like"
    if c <= 12:
        return "C10_GPP_like"
    if c <= 17:
        return "C15_FPP_like"
    if c <= 22:
        return "C20_GGPP_like"
    if c <= 27:
        return "C25_GFPP_like"
    return f"C{c}_long_prenyl_like"


def product_skeleton_class(cano_rxn: str) -> str:
    _, products = split_reaction_smiles(cano_rxn)
    product = largest_organic_component(products)
    if not product:
        return "unknown"
    c = carbon_count(product)
    rings = ring_count(product)
    oxy = oxygen_count(product)
    if c <= 0:
        return "unknown"
    c_bucket = "C5" if c <= 6 else "C10" if c <= 12 else "C15" if c <= 17 else "C20" if c <= 22 else "C25" if c <= 27 else "C_long"
    ring_bucket = "acyclic" if rings == 0 else "mono" if rings == 1 else "bicyclic" if rings == 2 else "polycyclic"
    oxy_bucket = "hydrocarbon" if oxy == 0 else "oxygenated" if oxy <= 2 else "highly_oxygenated"
    return f"{c_bucket}_{ring_bucket}_{oxy_bucket}"


def product_carbon_skeleton_signature(cano_rxn: str) -> str:
    """Return a canonical carbon-connectivity signature for the main product.

    The signature keeps only direct carbon-carbon edges and converts every edge
    to a single bond. Heteroatoms, bond order, charge, isotope and stereochemistry
    are intentionally ignored. This is a supervision label for TPS cyclization
    topology, not a chemically complete product representation.
    """
    _, products = split_reaction_smiles(cano_rxn)
    product = largest_organic_component(products)
    molecule = mol_from_smiles(product)
    if molecule is None:
        return "unknown"
    editable = Chem.RWMol()
    original_to_carbon: dict[int, int] = {}
    for atom in molecule.GetAtoms():
        if atom.GetAtomicNum() != 6:
            continue
        carbon = Chem.Atom(6)
        carbon.SetNoImplicit(False)
        original_to_carbon[atom.GetIdx()] = editable.AddAtom(carbon)
    if not original_to_carbon:
        return "unknown"
    for bond in molecule.GetBonds():
        begin = original_to_carbon.get(bond.GetBeginAtomIdx())
        end = original_to_carbon.get(bond.GetEndAtomIdx())
        if begin is not None and end is not None and editable.GetBondBetweenAtoms(begin, end) is None:
            editable.AddBond(begin, end, Chem.BondType.SINGLE)
    skeleton = editable.GetMol()
    try:
        Chem.SanitizeMol(skeleton)
    except Exception:
        try:
            skeleton.UpdatePropertyCache(strict=False)
        except Exception:
            return "unknown"
    fragments = Chem.GetMolFrags(skeleton, asMols=True, sanitizeFrags=False)
    signatures: list[tuple[int, str]] = []
    for fragment in fragments:
        try:
            fragment.UpdatePropertyCache(strict=False)
            smiles = Chem.MolToSmiles(
                fragment,
                canonical=True,
                isomericSmiles=False,
                kekuleSmiles=False,
            )
        except Exception:
            continue
        if smiles:
            signatures.append((fragment.GetNumAtoms(), smiles))
    if not signatures:
        return "unknown"
    signatures.sort(key=lambda item: (-item[0], item[1]))
    return ".".join(smiles for _, smiles in signatures)


def has_class_i_motif(sequence: str) -> bool:
    seq = coerce_text(sequence).upper()
    # Aspartate-rich DDxxD/DDxxxD-like motif plus a loose NSE/DTE-like motif.
    has_ddxxd = bool(re.search(r"D[D/E][A-Z]{2,4}D", seq))
    has_nse_dte = bool(re.search(r"N[DST][A-Z]{2,4}[ST][A-Z]{2,3}E", seq)) or bool(
        re.search(r"DTE[A-Z]{2,4}E", seq)
    )
    return has_ddxxd and has_nse_dte


def has_class_ii_motif(sequence: str) -> bool:
    seq = coerce_text(sequence).upper()
    return bool(re.search(r"D[A-Z]DD", seq))


def motif_score(sequence: str) -> float:
    class_i = has_class_i_motif(sequence)
    class_ii = has_class_ii_motif(sequence)
    if class_i and class_ii:
        return 1.0
    if class_i:
        return 0.8
    if class_ii:
        return 0.5
    return 0.0


def kmers(sequence: str, k: int = KMER_SIZE) -> set[str]:
    seq = re.sub(r"[^A-Z]", "", coerce_text(sequence).upper())
    if len(seq) < k:
        return set()
    return {seq[i : i + k] for i in range(0, len(seq) - k + 1)}


def jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


class TerpeneGateMatrix:
    def __init__(
        self,
        positive_path: Path,
        candidate_path: Path,
        cage_score_path: Path | None = None,
        exclude_same_reaction_smiles: bool = True,
    ) -> None:
        self.positive_path = positive_path
        self.candidate_path = candidate_path
        self.cage_score_path = cage_score_path
        self.exclude_same_reaction_smiles = exclude_same_reaction_smiles
        self.candidates = self._load_candidates(candidate_path)
        self.positive_rows = self._load_positive_rows(positive_path)
        self.reactions = self._build_reactions()
        self.candidate_ids = set(self.candidates["uniprot_id"].astype(str))
        self.known_positive_ids = set(self.positive_rows["uniprot_id"].astype(str))
        self.true_map = (
            self.positive_rows.groupby("rhea_id")["uniprot_id"].apply(lambda s: set(s.astype(str))).to_dict()
        )
        self.reaction_to_positive_rows = {
            rid: group.copy() for rid, group in self.positive_rows.groupby("rhea_id", sort=False)
        }
        self.candidate_seq = self.candidates.set_index("uniprot_id")["sequence"].astype(str).to_dict()
        self.positive_seq_lookup = self.positive_rows.drop_duplicates("uniprot_id").set_index("uniprot_id")["sequence"].astype(str).to_dict()
        self.candidate_kmers = {uid: kmers(seq) for uid, seq in self.candidate_seq.items()}
        self.candidate_kmer_sizes = {uid: len(kset) for uid, kset in self.candidate_kmers.items()}
        self.kmer_to_candidate_ids: dict[str, set[str]] = defaultdict(set)
        for uid, kset in self.candidate_kmers.items():
            for kmer in kset:
                self.kmer_to_candidate_ids[kmer].add(uid)
        self.candidate_motif_scores = {uid: motif_score(seq) for uid, seq in self.candidate_seq.items()}
        self.reaction_features = self._build_reaction_features()
        self.cage_scores = self._load_cage_scores(cage_score_path) if cage_score_path else {}
        self._reaction_sim_cache: dict[tuple[str, str], dict[str, float]] = {}
        self._seed_cache: dict[tuple[str, str, int | None], list[tuple[str, float]]] = {}
        self._sequence_seed_score_cache: dict[tuple[str, tuple[str, ...], int | None], dict[str, dict[str, float]]] = {}
        self._precursor_gate_cache: dict[str, dict[str, dict[str, float]]] = {}
        self._product_skeleton_gate_cache: dict[str, dict[str, dict[str, float]]] = {}
        self._mechanism_gate_cache: dict[str, dict[str, dict[str, float]]] = {}
        self._weighted_gate_cache: dict[str, dict[str, dict[str, float]]] = {}
        self._built_gate_cache: dict[tuple[str, GateSpec], dict[str, dict[str, float]]] = {}

    def _load_candidates(self, candidate_path: Path) -> pd.DataFrame:
        raw_df = read_table(candidate_path)
        cols = identify_terpene_columns(raw_df)
        id_col = cols["uniprot_id"]["column"] or cols["enzyme_id"]["column"]
        seq_col = cols["sequence"]["column"]
        if id_col is None or seq_col is None:
            raise ValueError(f"Could not identify candidate ID/sequence columns in {candidate_path}")
        rows: list[dict[str, str]] = []
        seen: set[str] = set()
        for _, row in raw_df.iterrows():
            uid = parse_uniprot_id(row.get(id_col))
            seq = coerce_text(row.get(seq_col))
            if not uid or not seq or uid in seen:
                continue
            seen.add(uid)
            rows.append({"enzyme_id": uid, "uniprot_id": uid, "sequence": seq})
        return pd.DataFrame(rows).sort_values("uniprot_id", kind="mergesort").reset_index(drop=True)

    def _load_positive_rows(self, positive_path: Path) -> pd.DataFrame:
        raw_df = read_table(positive_path)
        if "label" in raw_df.columns:
            raw_df = raw_df[raw_df["label"].astype(str).isin({"1", "1.0", "True", "true"})].copy()
        elif "Label" in raw_df.columns:
            raw_df = raw_df[raw_df["Label"].astype(str).isin({"1", "1.0", "True", "true"})].copy()
        cols = identify_terpene_columns(raw_df)
        id_col = cols["uniprot_id"]["column"] or cols["enzyme_id"]["column"]
        seq_col = cols["sequence"]["column"]
        rhea_col = cols["rhea_id"]["column"]
        rxn_col = cols["reaction_smiles"]["column"]
        ec_col = "EC number" if "EC number" in raw_df.columns else None
        if id_col is None or seq_col is None or rhea_col is None or rxn_col is None:
            raise ValueError(f"Could not identify positive-label columns in {positive_path}")
        rows: list[dict[str, str]] = []
        seen: set[tuple[str, str]] = set()
        for _, row in raw_df.iterrows():
            uid = parse_uniprot_id(row.get(id_col))
            rhea_id = coerce_text(row.get(rhea_col))
            raw_rxn = coerce_text(row.get(rxn_col))
            seq = coerce_text(row.get(seq_col))
            if not uid or not rhea_id:
                continue
            key = (rhea_id, uid)
            if key in seen:
                continue
            seen.add(key)
            cano = canonical_or_raw_reaction(raw_rxn)
            rows.append(
                {
                    "rhea_id": rhea_id,
                    "uniprot_id": uid,
                    "enzyme_id": uid,
                    "sequence": seq,
                    "reaction_smiles": raw_rxn,
                    "CANO_RXN_SMILES": cano,
                    "ec_number": coerce_text(row.get(ec_col)) if ec_col else "",
                }
            )
        return pd.DataFrame(rows).sort_values(["rhea_id", "uniprot_id"], kind="mergesort").reset_index(drop=True)

    def _build_reactions(self) -> pd.DataFrame:
        rows: list[dict[str, Any]] = []
        for rhea_id, group in self.positive_rows.groupby("rhea_id", sort=True):
            cano_values = [coerce_text(v) for v in group["CANO_RXN_SMILES"].tolist() if coerce_text(v)]
            cano = cano_values[0] if cano_values else ""
            rows.append(
                {
                    "reaction_id": rhea_id,
                    "rhea_id": rhea_id,
                    "CANO_RXN_SMILES": cano,
                    "n_true_enzymes": int(group["uniprot_id"].nunique()),
                    "n_known_positive_in_candidate_universe": int(
                        len(set(group["uniprot_id"].astype(str)) & set(self.candidates["uniprot_id"].astype(str)))
                    ),
                }
            )
        return pd.DataFrame(rows).sort_values("rhea_id", kind="mergesort").reset_index(drop=True)

    def _build_reaction_features(self) -> dict[str, dict[str, Any]]:
        features: dict[str, dict[str, Any]] = {}
        for _, row in self.reactions.iterrows():
            rhea_id = coerce_text(row["rhea_id"])
            cano = coerce_text(row["CANO_RXN_SMILES"])
            reactants, products = split_reaction_smiles(cano)
            features[rhea_id] = {
                "rhea_id": rhea_id,
                "CANO_RXN_SMILES": cano,
                "has_reaction_smiles": bool(cano),
                "reactants": reactants,
                "products": products,
                "reactant_fps": [fp for fp in (mol_fp(s) for s in reactants) if fp is not None],
                "product_fps": [fp for fp in (mol_fp(s) for s in products) if fp is not None],
                "precursor_class": precursor_class_from_reaction(cano),
                "product_skeleton_class": product_skeleton_class(cano),
            }
        return features

    def _load_cage_scores(self, cage_score_path: Path | None) -> dict[tuple[str, str], float]:
        if cage_score_path is None or not cage_score_path.exists():
            return {}
        df = pd.read_csv(cage_score_path)
        if "cage_score" not in df.columns and "pred" in df.columns:
            df = df.rename(columns={"pred": "cage_score"})
        if "reaction_id" not in df.columns or "uniprot_id" not in df.columns or "cage_score" not in df.columns:
            return {}
        scores: dict[tuple[str, str], float] = {}
        for _, row in df.iterrows():
            rid = coerce_text(row.get("reaction_id")) or coerce_text(row.get("rhea_id"))
            uid = coerce_text(row.get("uniprot_id")) or coerce_text(row.get("UniprotID"))
            if not rid or not uid:
                continue
            try:
                scores[(rid, uid)] = float(row.get("cage_score"))
            except Exception:
                continue
        return scores

    def reaction_similarity(self, target_rhea: str, seed_rhea: str) -> dict[str, float]:
        key = (target_rhea, seed_rhea)
        if key in self._reaction_sim_cache:
            return self._reaction_sim_cache[key]
        ft = self.reaction_features.get(target_rhea, {})
        fs = self.reaction_features.get(seed_rhea, {})
        substrate = best_match_similarity(ft.get("reactant_fps", []), fs.get("reactant_fps", []))
        product = best_match_similarity(ft.get("product_fps", []), fs.get("product_fps", []))
        precursor_bonus = 1.0 if ft.get("precursor_class") == fs.get("precursor_class") and ft.get("precursor_class") != "unknown" else 0.0
        skeleton_bonus = (
            1.0
            if ft.get("product_skeleton_class") == fs.get("product_skeleton_class")
            and ft.get("product_skeleton_class") != "unknown"
            else 0.0
        )
        balanced = 0.4 * substrate + 0.4 * product + 0.1 * precursor_bonus + 0.1 * skeleton_bonus
        payload = {
            "substrate": substrate,
            "product": product,
            "balanced": balanced,
            "precursor_bonus": precursor_bonus,
            "skeleton_bonus": skeleton_bonus,
        }
        self._reaction_sim_cache[key] = payload
        return payload

    def seed_reactions(self, target_rhea: str, mode: str = "balanced", topk: int | None = None) -> list[tuple[str, float]]:
        cache_key = (target_rhea, mode, topk)
        if cache_key in self._seed_cache:
            return self._seed_cache[cache_key]
        full_cache_key = (target_rhea, mode, None)
        if topk is not None and full_cache_key in self._seed_cache:
            rows = self._seed_cache[full_cache_key][:topk]
            self._seed_cache[cache_key] = rows
            return rows
        target_cano = coerce_text(self.reaction_features[target_rhea]["CANO_RXN_SMILES"])
        rows: list[tuple[str, float]] = []
        if not target_cano:
            self._seed_cache[full_cache_key] = rows
            self._seed_cache[cache_key] = rows
            return rows
        for seed_rhea in self.reactions["rhea_id"].astype(str).tolist():
            if seed_rhea == target_rhea:
                continue
            seed_cano = coerce_text(self.reaction_features[seed_rhea]["CANO_RXN_SMILES"])
            if self.exclude_same_reaction_smiles and target_cano and seed_cano and target_cano == seed_cano:
                continue
            sim = self.reaction_similarity(target_rhea, seed_rhea)
            score = float(sim.get(mode, sim.get("balanced", 0.0)))
            rows.append((seed_rhea, score))
        rows.sort(key=lambda item: (-item[1], item[0]))
        self._seed_cache[full_cache_key] = rows
        if topk is not None:
            rows = rows[:topk]
        self._seed_cache[cache_key] = rows
        return rows

    def enzymes_from_seed_reactions(self, seed_rows: list[tuple[str, float]]) -> dict[str, dict[str, float]]:
        enzyme_scores: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
        for seed_rhea, seed_score in seed_rows:
            for uid in self.true_map.get(seed_rhea, set()):
                if uid not in self.candidate_ids:
                    continue
                enzyme_scores[uid]["reaction_similarity"] = max(
                    enzyme_scores[uid]["reaction_similarity"], float(seed_score)
                )
        return {uid: dict(scores) for uid, scores in enzyme_scores.items()}

    def sequence_seed_scores(self, target_rhea: str, seed_rows: list[tuple[str, float]], topn: int | None = None) -> dict[str, dict[str, float]]:
        seed_ids: list[str] = []
        for seed_rhea, _ in seed_rows:
            seed_ids.extend(sorted(self.true_map.get(seed_rhea, set())))
        seed_ids = [uid for uid in dict.fromkeys(seed_ids) if uid in self.candidate_ids or uid in self.known_positive_ids]
        cache_key = (target_rhea, tuple(seed_ids), topn)
        if cache_key in self._sequence_seed_score_cache:
            return self._sequence_seed_score_cache[cache_key]

        best_scores: dict[str, float] = defaultdict(float)
        for seed_uid in seed_ids:
            seed_seq = self.candidate_seq.get(seed_uid) or self.positive_seq_lookup.get(seed_uid, "")
            seed_kset = kmers(seed_seq)
            if not seed_kset:
                continue
            intersection_counts: dict[str, int] = defaultdict(int)
            for kmer in seed_kset:
                for candidate_uid in self.kmer_to_candidate_ids.get(kmer, set()):
                    intersection_counts[candidate_uid] += 1
            seed_size = len(seed_kset)
            for candidate_uid, intersection in intersection_counts.items():
                candidate_size = self.candidate_kmer_sizes.get(candidate_uid, 0)
                denom = candidate_size + seed_size - intersection
                if denom <= 0:
                    continue
                score = intersection / denom
                if score > best_scores[candidate_uid]:
                    best_scores[candidate_uid] = score

        scores = sorted(best_scores.items(), key=lambda item: (-item[1], item[0]))
        if topn is not None:
            scores = scores[:topn]
        result = {uid: {"sequence_kmer": score} for uid, score in scores if score > 0 or topn is not None}
        self._sequence_seed_score_cache[cache_key] = result
        return result

    def precursor_gate_scores(self, target_rhea: str, max_candidates: int | None = None) -> dict[str, dict[str, float]]:
        if target_rhea not in self._precursor_gate_cache:
            target_class = self.reaction_features[target_rhea]["precursor_class"]
            rows: list[tuple[str, float]] = []
            for seed_rhea in self.reactions["rhea_id"].astype(str).tolist():
                if seed_rhea == target_rhea:
                    continue
                if target_class == "unknown" or self.reaction_features[seed_rhea]["precursor_class"] != target_class:
                    continue
                sim = self.reaction_similarity(target_rhea, seed_rhea)["balanced"]
                for uid in self.true_map.get(seed_rhea, set()):
                    if uid in self.candidate_ids:
                        rows.append((uid, sim))
            self._precursor_gate_cache[target_rhea] = self._dedupe_score_rows(rows, "precursor_match", max_candidates=None)
        result = self._precursor_gate_cache[target_rhea]
        if max_candidates is None:
            return result
        return dict(list(result.items())[:max_candidates])

    def product_skeleton_gate_scores(self, target_rhea: str, max_candidates: int | None = None) -> dict[str, dict[str, float]]:
        if target_rhea not in self._product_skeleton_gate_cache:
            target_class = self.reaction_features[target_rhea]["product_skeleton_class"]
            rows: list[tuple[str, float]] = []
            for seed_rhea in self.reactions["rhea_id"].astype(str).tolist():
                if seed_rhea == target_rhea:
                    continue
                if target_class == "unknown" or self.reaction_features[seed_rhea]["product_skeleton_class"] != target_class:
                    continue
                sim = self.reaction_similarity(target_rhea, seed_rhea)["balanced"]
                for uid in self.true_map.get(seed_rhea, set()):
                    if uid in self.candidate_ids:
                        rows.append((uid, sim))
            self._product_skeleton_gate_cache[target_rhea] = self._dedupe_score_rows(rows, "product_skeleton", max_candidates=None)
        result = self._product_skeleton_gate_cache[target_rhea]
        if max_candidates is None:
            return result
        return dict(list(result.items())[:max_candidates])

    def _dedupe_score_rows(
        self,
        rows: list[tuple[str, float]],
        score_name: str,
        max_candidates: int | None = None,
    ) -> dict[str, dict[str, float]]:
        best: dict[str, float] = defaultdict(float)
        for uid, score in rows:
            best[uid] = max(best[uid], float(score))
        ordered = sorted(best.items(), key=lambda item: (-item[1], item[0]))
        if max_candidates is not None:
            ordered = ordered[:max_candidates]
        return {uid: {score_name: score} for uid, score in ordered}

    def mechanism_precursor_motif_scores(self, target_rhea: str, max_candidates: int | None = None) -> dict[str, dict[str, float]]:
        if target_rhea not in self._mechanism_gate_cache:
            base = self.precursor_gate_scores(target_rhea, max_candidates=None)
            rows: list[tuple[str, float]] = []
            for uid, scores in base.items():
                m = self.candidate_motif_scores.get(uid, 0.0)
                if m <= 0:
                    continue
                score = 0.7 * float(scores.get("precursor_match", 0.0)) + 0.3 * m
                rows.append((uid, score))
            self._mechanism_gate_cache[target_rhea] = self._dedupe_score_rows(rows, "mechanism", max_candidates=None)
        result = self._mechanism_gate_cache[target_rhea]
        if max_candidates is None:
            return result
        return dict(list(result.items())[:max_candidates])

    def weighted_candidate_scores(self, target_rhea: str, topn: int | None = None) -> dict[str, dict[str, float]]:
        if target_rhea not in self._weighted_gate_cache:
            seed20 = self.seed_reactions(target_rhea, "balanced", 20)
            rxn_scores = self.enzymes_from_seed_reactions(seed20)
            seq_scores = self.sequence_seed_scores(target_rhea, seed20, topn=None)
            prec_scores = self.precursor_gate_scores(target_rhea, max_candidates=None)
            skel_scores = self.product_skeleton_gate_scores(target_rhea, max_candidates=None)
            rows: list[tuple[str, dict[str, float]]] = []
            for uid in sorted(self.candidate_ids):
                rxn = float(rxn_scores.get(uid, {}).get("reaction_similarity", 0.0))
                seq = float(seq_scores.get(uid, {}).get("sequence_kmer", 0.0))
                prec = 1.0 if uid in prec_scores else 0.0
                skel = 1.0 if uid in skel_scores else 0.0
                motif = self.candidate_motif_scores.get(uid, 0.0)
                score = 0.40 * rxn + 0.25 * seq + 0.15 * prec + 0.10 * skel + 0.10 * motif
                if score <= 0:
                    continue
                rows.append(
                    (
                        uid,
                        {
                            "weighted": score,
                            "reaction_similarity": rxn,
                            "sequence_kmer": seq,
                            "precursor_match": prec,
                            "product_skeleton": skel,
                            "motif": motif,
                        },
                    )
                )
            rows.sort(key=lambda item: (-item[1]["weighted"], item[0]))
            self._weighted_gate_cache[target_rhea] = {uid: scores for uid, scores in rows}
        result = self._weighted_gate_cache[target_rhea]
        if topn is None:
            return result
        return dict(list(result.items())[:topn])

    @staticmethod
    def merge_score_dicts(*dicts: dict[str, dict[str, float]]) -> dict[str, dict[str, float]]:
        merged: dict[str, dict[str, float]] = {}
        for dct in dicts:
            for uid, scores in dct.items():
                if uid not in merged:
                    merged[uid] = {}
                for key, value in scores.items():
                    merged[uid][key] = max(float(value), float(merged[uid].get(key, 0.0)))
        return merged

    def build_gate_for_reaction(self, target_rhea: str, spec: GateSpec) -> dict[str, dict[str, float]]:
        cache_key = (target_rhea, spec)
        if cache_key in self._built_gate_cache:
            return self._built_gate_cache[cache_key]
        if spec.method == "reaction_similarity":
            seeds = self.seed_reactions(target_rhea, spec.score_mode, spec.topk_reactions)
            result = self.enzymes_from_seed_reactions(seeds)
        elif spec.method == "precursor_exact":
            result = self.precursor_gate_scores(target_rhea, max_candidates=spec.max_candidates)
        elif spec.method == "product_skeleton":
            result = self.product_skeleton_gate_scores(target_rhea, max_candidates=spec.max_candidates)
        elif spec.method == "sequence_kmer":
            seeds = self.seed_reactions(target_rhea, "balanced", spec.topk_reactions or 20)
            result = self.sequence_seed_scores(target_rhea, seeds, topn=spec.topn_candidates)
        elif spec.method == "mechanism_precursor_motif":
            result = self.mechanism_precursor_motif_scores(target_rhea, max_candidates=spec.max_candidates)
        elif spec.method == "recall_union":
            rxn = self.build_gate_for_reaction(target_rhea, GateSpec("tmp", "reaction_similarity", topk_reactions=20))
            prod = self.build_gate_for_reaction(
                target_rhea, GateSpec("tmp", "reaction_similarity", topk_reactions=20, score_mode="product")
            )
            prec = self.precursor_gate_scores(target_rhea, max_candidates=200)
            seq = self.sequence_seed_scores(target_rhea, self.seed_reactions(target_rhea, "balanced", 20), topn=50)
            mech = self.mechanism_precursor_motif_scores(target_rhea, max_candidates=200)
            merged = self.merge_score_dicts(rxn, prod, prec, seq, mech)
            rows = []
            for uid, scores in merged.items():
                union_score = max(scores.values()) if scores else 0.0
                scores["union"] = union_score
                rows.append((uid, scores))
            rows.sort(key=lambda item: (-item[1].get("union", 0.0), item[0]))
            if spec.max_candidates is not None:
                rows = rows[: spec.max_candidates]
            result = {uid: scores for uid, scores in rows}
        elif spec.method == "weighted":
            result = self.weighted_candidate_scores(target_rhea, topn=spec.topn_candidates)
        else:
            raise ValueError(f"Unknown gate method: {spec.method}")
        self._built_gate_cache[cache_key] = result
        return result

    def candidate_rerank_score(self, rhea_id: str, uid: str, component_scores: dict[str, float], reranker: str) -> float:
        if reranker == "gate_score":
            candidates = [
                component_scores.get("weighted", 0.0),
                component_scores.get("union", 0.0),
                component_scores.get("reaction_similarity", 0.0),
                component_scores.get("sequence_kmer", 0.0),
                component_scores.get("precursor_match", 0.0),
                component_scores.get("product_skeleton", 0.0),
                component_scores.get("mechanism", 0.0),
            ]
            return float(max(candidates))
        if reranker == "reaction_similarity":
            return float(component_scores.get("reaction_similarity", 0.0))
        if reranker == "sequence_kmer":
            if "sequence_kmer" in component_scores:
                return float(component_scores.get("sequence_kmer", 0.0))
            seeds = self.seed_reactions(rhea_id, "balanced", 20)
            seq_score = self.sequence_seed_scores(rhea_id, seeds, topn=None).get(uid, {}).get("sequence_kmer", 0.0)
            return float(seq_score)
        if reranker == "motif":
            return float(self.candidate_motif_scores.get(uid, 0.0))
        if reranker == "fusion":
            rxn = float(component_scores.get("reaction_similarity", 0.0))
            seq = float(component_scores.get("sequence_kmer", 0.0))
            if seq == 0.0:
                seq = float(self.candidate_rerank_score(rhea_id, uid, component_scores, "sequence_kmer"))
            prec = float(component_scores.get("precursor_match", 0.0))
            skel = float(component_scores.get("product_skeleton", 0.0))
            motif = float(self.candidate_motif_scores.get(uid, 0.0))
            cage = float(self.cage_scores.get((rhea_id, uid), 0.0))
            return 0.35 * rxn + 0.25 * seq + 0.15 * prec + 0.10 * skel + 0.10 * motif + 0.05 * cage
        if reranker == "cage_if_available":
            return float(self.cage_scores.get((rhea_id, uid), float("nan")))
        if reranker == "random":
            return stable_random_score(rhea_id, uid)
        raise ValueError(f"Unknown reranker: {reranker}")

    def evaluate_ranked_candidates(
        self,
        rhea_id: str,
        candidate_scores: dict[str, dict[str, float]],
        reranker: str,
    ) -> dict[str, Any]:
        true = self.true_map.get(rhea_id, set()) & self.candidate_ids
        rows: list[tuple[str, float]] = []
        for uid, scores in candidate_scores.items():
            score = self.candidate_rerank_score(rhea_id, uid, scores, reranker)
            if reranker == "cage_if_available" and math.isnan(score):
                continue
            rows.append((uid, score))
        rows.sort(key=lambda item: (-item[1], item[0]))
        best_rank: int | None = None
        best_uid = ""
        best_score: float | None = None
        for idx, (uid, score) in enumerate(rows, start=1):
            if uid in true:
                best_rank = idx
                best_uid = uid
                best_score = float(score)
                break
        return {
            "n_ranked_candidates": len(rows),
            "best_positive_rank": best_rank,
            "best_positive_enzyme_id": best_uid,
            "best_positive_score": best_score,
            "top1_hit": bool(best_rank is not None and best_rank <= 1),
            "top3_hit": bool(best_rank is not None and best_rank <= 3),
            "top5_hit": bool(best_rank is not None and best_rank <= 5),
            "top10_hit": bool(best_rank is not None and best_rank <= 10),
            "reciprocal_rank": 1.0 / best_rank if best_rank else 0.0,
            "top10_enzyme_ids": json.dumps([uid for uid, _ in rows[:10]], ensure_ascii=False),
            "top10_scores": json.dumps([float(score) for _, score in rows[:10]], ensure_ascii=False),
        }

    def run(
        self,
        gate_specs: list[GateSpec],
        rerankers: list[str],
        max_reactions: int | None = None,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        reaction_ids = self.reactions["rhea_id"].astype(str).tolist()
        if max_reactions is not None:
            reaction_ids = reaction_ids[:max_reactions]

        candidate_rows: list[dict[str, Any]] = []
        gate_reaction_rows: list[dict[str, Any]] = []
        rerank_reaction_rows: list[dict[str, Any]] = []

        for spec in gate_specs:
            for rhea_id in reaction_ids:
                candidates = self.build_gate_for_reaction(rhea_id, spec)
                true = self.true_map.get(rhea_id, set()) & self.candidate_ids
                hit_set = set(candidates) & true
                ranked_by_gate = sorted(
                    candidates.items(),
                    key=lambda item: (-self.candidate_rerank_score(rhea_id, item[0], item[1], "gate_score"), item[0]),
                )
                for rank, (uid, scores) in enumerate(ranked_by_gate, start=1):
                    candidate_rows.append(
                        {
                            "gate_id": spec.gate_id,
                            "reaction_id": rhea_id,
                            "rhea_id": rhea_id,
                            "enzyme_id": uid,
                            "uniprot_id": uid,
                            "gate_rank": rank,
                            "gate_score": self.candidate_rerank_score(rhea_id, uid, scores, "gate_score"),
                            "reaction_similarity": scores.get("reaction_similarity", 0.0),
                            "sequence_kmer": scores.get("sequence_kmer", 0.0),
                            "precursor_match": scores.get("precursor_match", 0.0),
                            "product_skeleton": scores.get("product_skeleton", 0.0),
                            "mechanism": scores.get("mechanism", 0.0),
                            "motif_score": self.candidate_motif_scores.get(uid, 0.0),
                            "label": 1 if uid in true else 0,
                        }
                    )
                gate_reaction_rows.append(
                    {
                        "gate_id": spec.gate_id,
                        "method": spec.method,
                        "reaction_id": rhea_id,
                        "rhea_id": rhea_id,
                        "n_candidates": len(candidates),
                        "n_true_enzymes": len(true),
                        "n_positive_candidates": len(hit_set),
                        "gate_hit": bool(hit_set),
                        "positive_coverage": len(hit_set) / len(true) if true else None,
                        "has_reaction_smiles": self.reaction_features[rhea_id].get("has_reaction_smiles", False),
                        "precursor_class": self.reaction_features[rhea_id]["precursor_class"],
                        "product_skeleton_class": self.reaction_features[rhea_id]["product_skeleton_class"],
                    }
                )
                for reranker in rerankers:
                    metrics = self.evaluate_ranked_candidates(rhea_id, candidates, reranker)
                    rerank_reaction_rows.append(
                        {
                            "gate_id": spec.gate_id,
                            "reranker": reranker,
                            "reaction_id": rhea_id,
                            "rhea_id": rhea_id,
                            **metrics,
                        }
                    )

        candidates_df = pd.DataFrame(candidate_rows)
        gate_reaction_df = pd.DataFrame(gate_reaction_rows)
        rerank_reaction_df = pd.DataFrame(rerank_reaction_rows)
        gate_metrics_df = self.aggregate_gate_metrics(gate_reaction_df)
        rerank_metrics_df = self.aggregate_rerank_metrics(rerank_reaction_df)
        return candidates_df, gate_reaction_df, gate_metrics_df, rerank_metrics_df

    @staticmethod
    def aggregate_gate_metrics(gate_reaction_df: pd.DataFrame) -> pd.DataFrame:
        rows: list[dict[str, Any]] = []
        for gate_id, group in gate_reaction_df.groupby("gate_id", sort=False):
            positives = group[group["n_true_enzymes"].astype(int) > 0].copy()
            total_true = float(positives["n_true_enzymes"].sum()) if not positives.empty else 0.0
            total_hit = float(positives["n_positive_candidates"].sum()) if not positives.empty else 0.0
            rows.append(
                {
                    "gate_id": gate_id,
                    "n_reactions": int(len(group)),
                    "mean_pool_size": float(group["n_candidates"].mean()) if len(group) else 0.0,
                    "median_pool_size": float(group["n_candidates"].median()) if len(group) else 0.0,
                    "max_pool_size": int(group["n_candidates"].max()) if len(group) else 0,
                    "empty_pool_rate": float((group["n_candidates"] == 0).mean()) if len(group) else 0.0,
                    "reaction_hit_rate": float(positives["gate_hit"].astype(bool).mean()) if len(positives) else 0.0,
                    "micro_positive_coverage": total_hit / total_true if total_true else 0.0,
                    "macro_positive_coverage": float(positives["positive_coverage"].dropna().mean()) if len(positives) else 0.0,
                    "positive_candidates_total": int(positives["n_positive_candidates"].sum()) if len(positives) else 0,
                    "candidate_pairs_total": int(group["n_candidates"].sum()) if len(group) else 0,
                }
            )
        return pd.DataFrame(rows).sort_values(
            ["micro_positive_coverage", "reaction_hit_rate", "mean_pool_size"], ascending=[False, False, True]
        )

    @staticmethod
    def aggregate_rerank_metrics(rerank_reaction_df: pd.DataFrame) -> pd.DataFrame:
        rows: list[dict[str, Any]] = []
        for (gate_id, reranker), group in rerank_reaction_df.groupby(["gate_id", "reranker"], sort=False):
            eligible = group[group["n_ranked_candidates"] > 0].copy()
            if eligible.empty:
                eligible = group.copy()
            ranks = eligible["best_positive_rank"].dropna().astype(float)
            rows.append(
                {
                    "gate_id": gate_id,
                    "reranker": reranker,
                    "n_reactions": int(len(group)),
                    "n_ranked_reactions": int((group["n_ranked_candidates"] > 0).sum()),
                    "top1_recall": float(eligible["top1_hit"].astype(bool).mean()) if len(eligible) else 0.0,
                    "top3_recall": float(eligible["top3_hit"].astype(bool).mean()) if len(eligible) else 0.0,
                    "top5_recall": float(eligible["top5_hit"].astype(bool).mean()) if len(eligible) else 0.0,
                    "top10_recall": float(eligible["top10_hit"].astype(bool).mean()) if len(eligible) else 0.0,
                    "mean_reciprocal_rank": float(eligible["reciprocal_rank"].astype(float).mean()) if len(eligible) else 0.0,
                    "median_best_positive_rank": float(ranks.median()) if len(ranks) else None,
                    "mean_ranked_pool_size": float(eligible["n_ranked_candidates"].mean()) if len(eligible) else 0.0,
                }
            )
        return pd.DataFrame(rows).sort_values(
            ["top10_recall", "mean_reciprocal_rank", "mean_ranked_pool_size"], ascending=[False, False, True]
        )


def parse_gate_ids(gate_ids: str | None) -> list[GateSpec]:
    if not gate_ids:
        return DEFAULT_GATE_SPECS
    wanted = {item.strip() for item in gate_ids.split(",") if item.strip()}
    selected = [spec for spec in DEFAULT_GATE_SPECS if spec.gate_id in wanted]
    missing = wanted - {spec.gate_id for spec in selected}
    if missing:
        raise ValueError(f"Unknown gate IDs: {sorted(missing)}")
    return selected


def dataframe_to_simple_markdown(df: pd.DataFrame, max_rows: int = 50) -> str:
    if df.empty:
        return "_No rows._"
    view = df.head(max_rows).copy()
    columns = [str(col) for col in view.columns]
    rows = [columns, ["---" for _ in columns]]
    for _, row in view.iterrows():
        rows.append([coerce_text(row.get(col, "")) for col in view.columns])
    escaped_rows = []
    for row in rows:
        escaped_rows.append([cell.replace("|", "\\|").replace("\n", " ") for cell in row])
    return "\n".join("| " + " | ".join(row) + " |" for row in escaped_rows)


def write_matrix_report(
    path: Path,
    gate_metrics_df: pd.DataFrame,
    rerank_metrics_df: pd.DataFrame,
    summary: dict[str, Any],
) -> None:
    lines: list[str] = [
        "# Terpene all-Rhea gate matrix",
        "",
        "This report compares candidate-pool construction methods for all known terpene Rhea reactions.",
        "The pipeline intentionally avoids constructing the full reaction × enzyme pair matrix; each gate emits only its own candidate pool.",
        "",
        "## Run summary",
        "",
    ]
    for key, value in summary.items():
        if key.endswith("_path"):
            continue
        lines.append(f"- `{key}`: `{value}`")
    lines.extend([
        "",
        "## Gate-only metrics",
        "",
        dataframe_to_simple_markdown(gate_metrics_df),
        "",
        "## Gate + reranker metrics",
        "",
        dataframe_to_simple_markdown(rerank_metrics_df),
        "",
    ])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build and evaluate all-Rhea terpene synthase gate/rerank matrices without full-pair CAGE expansion.")
    parser.add_argument("--positive_path", default=str(DEFAULT_POSITIVE_PATH))
    parser.add_argument("--candidate_path", default=str(DEFAULT_CANDIDATE_PATH))
    parser.add_argument("--output_dir", default=str(GATE_MATRIX_RESULTS_DIR))
    parser.add_argument("--data_output_dir", default=str(GATE_MATRIX_DATA_DIR))
    parser.add_argument("--cage_score_path", default=str(DEFAULT_CAGE_SCORE_PATH))
    parser.add_argument("--no_cage_scores", action="store_true")
    parser.add_argument("--gate_ids", default="", help="Comma-separated subset of built-in gate IDs.")
    parser.add_argument("--rerankers", default=",".join(RERANKERS), help="Comma-separated rerankers.")
    parser.add_argument("--max_reactions", type=int, default=None, help="DEBUG ONLY: limit reactions for smoke tests; formal experiments must omit this and use all known Rhea reactions.")
    parser.add_argument("--allow_same_reaction_smiles", action="store_true", help="Allow seed reactions with identical canonical reaction SMILES. Default excludes them.")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    data_output_dir = Path(args.data_output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    data_output_dir.mkdir(parents=True, exist_ok=True)

    gate_specs = parse_gate_ids(args.gate_ids)
    rerankers = [item.strip() for item in args.rerankers.split(",") if item.strip()]
    cage_path = None if args.no_cage_scores else Path(args.cage_score_path)

    matrix = TerpeneGateMatrix(
        positive_path=Path(args.positive_path),
        candidate_path=Path(args.candidate_path),
        cage_score_path=cage_path,
        exclude_same_reaction_smiles=not args.allow_same_reaction_smiles,
    )
    candidates_df, gate_reaction_df, gate_metrics_df, rerank_metrics_df = matrix.run(
        gate_specs=gate_specs,
        rerankers=rerankers,
        max_reactions=args.max_reactions,
    )

    write_table(candidates_df, data_output_dir / "gate_candidate_pools.csv", sep=",")
    write_table(gate_reaction_df, output_dir / "gate_reaction_level.csv", sep=",")
    write_table(gate_metrics_df, output_dir / "gate_metrics.csv", sep=",")
    write_table(rerank_metrics_df, output_dir / "rerank_metrics.csv", sep=",")

    summary = {
        "n_known_rhea_reactions_total": int(len(matrix.reactions)),
        "n_evaluated_reactions": int(args.max_reactions or len(matrix.reactions)),
        "debug_subset": bool(args.max_reactions is not None),
        "n_candidate_enzymes": int(len(matrix.candidates)),
        "n_positive_pairs": int(len(matrix.positive_rows)),
        "n_gate_specs": int(len(gate_specs)),
        "gate_ids": [spec.gate_id for spec in gate_specs],
        "rerankers": rerankers,
        "exclude_same_reaction_smiles": not args.allow_same_reaction_smiles,
        "cage_scores_loaded": int(len(matrix.cage_scores)),
        "candidate_pools_path": str((data_output_dir / "gate_candidate_pools.csv").resolve()),
        "gate_metrics_path": str((output_dir / "gate_metrics.csv").resolve()),
        "rerank_metrics_path": str((output_dir / "rerank_metrics.csv").resolve()),
    }
    safe_json_dump(summary, output_dir / "run_summary.json")
    write_matrix_report(output_dir / "gate_matrix_report.md", gate_metrics_df, rerank_metrics_df, summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
