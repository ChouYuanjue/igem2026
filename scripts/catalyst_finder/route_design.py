from __future__ import annotations

import csv
import gzip
import hashlib
import json
import math
import pickle
import re
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import requests
import networkx as nx
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import rdFingerprintGenerator

RHEA_FTP = "https://ftp.expasy.org/databases/rhea/tsv"
ASSETS = {
    "reaction_smiles": f"{RHEA_FTP}/rhea-reaction-smiles.tsv",
    "chebi_smiles": f"{RHEA_FTP}/rhea-chebi-smiles.tsv",
    "directions": f"{RHEA_FTP}/rhea-directions.tsv",
    "sprot": f"{RHEA_FTP}/rhea2uniprot_sprot.tsv",
    "names": f"{RHEA_FTP}/chebiId_name.tsv",
}

# Small, ubiquitous cofactors should not become the "main carbon path" merely because
# they appear in many reactions. They remain part of the underlying Rhea reaction and
# can still be inspected later through the Rhea equation.
CURRENCY_CHEBI = {
    "CHEBI:15377",  # water
    "CHEBI:15378",  # proton
    "CHEBI:15379",  # oxygen
    "CHEBI:16240",  # hydrogen peroxide
    "CHEBI:16526",  # carbon dioxide
    "CHEBI:28938",  # ammonium
    "CHEBI:43474",  # phosphate
    "CHEBI:33019",  # diphosphate / pyrophosphate
    "CHEBI:456216", # ADP
    "CHEBI:30616",  # ATP
    "CHEBI:57945",  # NADH
    "CHEBI:57540",  # NAD(+)
    "CHEBI:57783",  # NADPH
    "CHEBI:58349",  # NADP(+)
    "CHEBI:57692",  # CoA
}

ALIASES = {
    "gpp": "(2E)-geranyl diphosphate",
    "fpp": "(2E,6E)-farnesyl diphosphate",
    "ggpp": "(2E,6E,10E)-geranylgeranyl diphosphate",
    "ipp": "isopentenyl diphosphate",
    "dmapp": "dimethylallyl diphosphate",
}

_morgan = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=1024)
RDLogger.DisableLog("rdApp.warning")
RDLogger.DisableLog("rdApp.error")


def _norm_name(value: str) -> str:
    text = str(value or "").strip().casefold()
    text = text.replace("β", "beta").replace("α", "alpha").replace("γ", "gamma")
    text = text.replace("–", "-").replace("—", "-")
    text = re.sub(r"\s+", " ", text)
    return text


def _biochemical_name_variants(value: str) -> set[str]:
    """Generate conservative biochemical naming equivalents without assigning IDs.

    These are nomenclature transformations (stereochemical decoration, aromatic
    positional prefixes, and acid/conjugate-base suffixes), not a compound lookup
    table. The final identifier always comes from the official Rhea/ChEBI index.
    """
    base = _norm_name(value)
    if not base:
        return set()
    variants = {base}

    # Rhea often stores a specific stereochemical form while users give the common
    # parent name. Keep both forms rather than discarding stereochemistry globally.
    stripped = re.sub(r"^\((?:[0-9,]*[erzs]|[erzs])\)-?", "", base, flags=re.I).strip()
    if stripped:
        variants.add(stripped)

    def positional(text: str) -> str:
        mappings = (
            (r"^(?:para|p)[- ]", "4-"),
            (r"^(?:meta|m)[- ]", "3-"),
            (r"^(?:ortho|o)[- ]", "2-"),
        )
        result = text
        for pattern, replacement in mappings:
            result = re.sub(pattern, replacement, result, flags=re.I)
        return result

    for current in list(variants):
        positioned = positional(current)
        if positioned:
            variants.add(positioned)

    # Common biochemical databases frequently index the dominant carboxylate form
    # rather than the neutral acid name. This suffix rule covers broad chemistry
    # such as lactic acid/lactate and cinnamic acid/cinnamate without named aliases.
    for current in list(variants):
        if current.endswith("ic acid") and len(current) > len("ic acid"):
            variants.add(current[: -len("ic acid")] + "ate")
        if current.endswith("ous acid") and len(current) > len("ous acid"):
            variants.add(current[: -len("ous acid")] + "ite")
        if current.endswith("ate") and len(current) > 4:
            variants.add(current[:-3] + "ic acid")
        if current.endswith("ite") and len(current) > 4:
            variants.add(current[:-3] + "ous acid")

    return {_norm_name(value) for value in variants if _norm_name(value)}


def _canonical_smiles(value: str) -> str | None:
    try:
        mol = Chem.MolFromSmiles(str(value or ""), sanitize=True)
    except Exception:
        return None
    if mol is None:
        return None
    try:
        return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
    except Exception:
        return None


def _connectivity_key(smiles: str) -> str:
    mol = Chem.MolFromSmiles(str(smiles or ""))
    if mol is None:
        return ""
    try:
        return Chem.MolToInchiKey(mol).split("-", 1)[0]
    except Exception:
        return ""


def _mol_features(smiles: str) -> tuple[int, Any] | None:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    heavy = int(mol.GetNumHeavyAtoms())
    fp = _morgan.GetFingerprint(mol)
    return heavy, fp


def _pair_score(a: tuple[int, Any], b: tuple[int, Any]) -> float:
    ha, fpa = a
    hb, fpb = b
    if ha <= 0 or hb <= 0:
        return 0.0
    tanimoto = float(DataStructs.TanimotoSimilarity(fpa, fpb))
    atom_ratio = min(ha, hb) / max(ha, hb)
    return 0.72 * tanimoto + 0.28 * atom_ratio


class RouteDesignError(RuntimeError):
    pass


class RheaRouteDesigner:
    """Broad known-biochemistry route search backed by the full Rhea release.

    The route graph is intentionally separate from the Catalyst Finder model catalog.
    Rhea contributes known biochemical reactions; project-local model coverage is added
    only as a ranking feature. The graph reduces each hyper-reaction to likely main
    substrate/product transformations using structure conservation, while the full Rhea
    identifier is retained for downstream verification.
    """

    def __init__(self, root: Path, *, user_agent: str, cache_root: Path) -> None:
        self.root = Path(root)
        self.cache_root = Path(cache_root) / "route_design" / "rhea"
        self.cache_root.mkdir(parents=True, exist_ok=True)
        self.index_path = self.cache_root / "rhea_route_index.pkl.gz"
        self.connectivity_path = self.cache_root / "rhea_connectivity.pkl.gz"
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": user_agent})
        self._index: dict[str, Any] | None = None
        self._reaction_smiles_by_id: dict[str, str] | None = None
        self._smiles_to_chebi_exact: dict[str, list[str]] | None = None
        self._known_uniprot_by_rhea: dict[str, tuple[str, ...]] | None = None
        self._known_rhea_by_uniprot: dict[str, tuple[str, ...]] | None = None
        self._compound_alias_to_ids: dict[str, tuple[str, ...]] | None = None
        self.pickaxe_worker = self.root / "scripts/catalyst_finder/pickaxe_worker.py"
        self.pickaxe_vendor = self.root / "external_repos/route_design/MINE-Database"
        self.pickaxe_site = self.root / "results/catalyst_finder_runtime/route_design/pickaxe_site"

    def _asset_path(self, key: str) -> Path:
        return self.cache_root / Path(ASSETS[key]).name

    def _download_asset(self, key: str) -> Path:
        path = self._asset_path(key)
        if path.exists() and path.stat().st_size > 100:
            return path
        tmp = path.with_suffix(path.suffix + ".tmp")
        try:
            with self.session.get(ASSETS[key], stream=True, timeout=45) as response:
                response.raise_for_status()
                with tmp.open("wb") as handle:
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            handle.write(chunk)
            tmp.replace(path)
        except requests.RequestException as exc:
            tmp.unlink(missing_ok=True)
            raise RouteDesignError(f"Rhea 路线数据下载失败: {key}: {exc}") from exc
        return path

    def _ensure_sprot_association_maps(self) -> None:
        if self._known_uniprot_by_rhea is not None and self._known_rhea_by_uniprot is not None:
            return
        path = self._download_asset("sprot")
        by_rhea: dict[str, list[str]] = defaultdict(list)
        by_uniprot: dict[str, list[str]] = defaultdict(list)
        seen_rhea: dict[str, set[str]] = defaultdict(set)
        seen_uniprot: dict[str, set[str]] = defaultdict(set)
        with path.open(encoding="utf-8", errors="replace", newline="") as handle:
            for row in csv.DictReader(handle, delimiter="\t"):
                accession = str(row.get("ID") or "").strip()
                directed = str(row.get("RHEA_ID") or "").strip()
                master = str(row.get("MASTER_ID") or "").strip()
                if not accession:
                    continue
                for key in {directed, master}:
                    if key and accession not in seen_rhea[key]:
                        seen_rhea[key].add(accession)
                        by_rhea[key].append(accession)
                canonical_rhea = master or directed
                if canonical_rhea and canonical_rhea not in seen_uniprot[accession]:
                    seen_uniprot[accession].add(canonical_rhea)
                    by_uniprot[accession].append(canonical_rhea)
        self._known_uniprot_by_rhea = {key: tuple(values) for key, values in by_rhea.items()}
        self._known_rhea_by_uniprot = {key: tuple(values) for key, values in by_uniprot.items()}

    def known_uniprot_ids(self, rhea_id: str) -> list[str]:
        """Swiss-Prot accessions recorded by the official Rhea mapping."""
        self._ensure_sprot_association_maps()
        match = re.search(r"(\d{5})", str(rhea_id or ""))
        if not match or self._known_uniprot_by_rhea is None:
            return []
        return list(self._known_uniprot_by_rhea.get(match.group(1), ()))

    def known_rhea_ids(self, uniprot_id: str) -> list[str]:
        """Canonical master Rhea IDs recorded for one Swiss-Prot accession."""
        self._ensure_sprot_association_maps()
        accession = str(uniprot_id or "").strip().upper()
        if not accession or self._known_rhea_by_uniprot is None:
            return []
        return [f"RHEA:{value}" for value in self._known_rhea_by_uniprot.get(accession, ())]

    def ensure_index(self) -> dict[str, Any]:
        if self._index is not None:
            return self._index
        if self.index_path.exists():
            try:
                with gzip.open(self.index_path, "rb") as handle:
                    self._index = pickle.load(handle)
                return self._index
            except Exception:
                self.index_path.unlink(missing_ok=True)
        for key in ASSETS:
            self._download_asset(key)
        self._index = self._build_index()
        tmp = self.index_path.with_suffix(".tmp.gz")
        with gzip.open(tmp, "wb", compresslevel=4) as handle:
            pickle.dump(self._index, handle, protocol=pickle.HIGHEST_PROTOCOL)
        tmp.replace(self.index_path)
        return self._index

    def _build_index(self) -> dict[str, Any]:
        names: dict[str, str] = {}
        name_to_ids: dict[str, list[str]] = defaultdict(list)
        with self._asset_path("names").open(encoding="utf-8", errors="replace") as handle:
            for line in handle:
                parts = line.rstrip("\n\r").split("\t", 1)
                if len(parts) != 2:
                    continue
                cid, name = parts[0].strip(), parts[1].strip()
                if not cid or not name:
                    continue
                names[cid] = name
                name_to_ids[_norm_name(name)].append(cid)

        chebi_smiles: dict[str, str] = {}
        smiles_to_chebi: dict[str, list[str]] = defaultdict(list)
        features: dict[str, tuple[int, Any]] = {}
        with self._asset_path("chebi_smiles").open(encoding="utf-8", errors="replace") as handle:
            for line in handle:
                parts = line.rstrip("\n\r").split("\t", 1)
                if len(parts) != 2:
                    continue
                cid, raw = parts[0].strip(), parts[1].strip()
                can = _canonical_smiles(raw)
                if not cid or not can:
                    continue
                feat = _mol_features(can)
                if feat is None:
                    continue
                chebi_smiles[cid] = can
                features[cid] = feat
                smiles_to_chebi[can].append(cid)

        directed_map: dict[str, tuple[str, str]] = {}
        with self._asset_path("directions").open(encoding="utf-8", errors="replace") as handle:
            rows = csv.DictReader(handle, delimiter="\t")
            for row in rows:
                master = str(row.get("RHEA_ID_MASTER") or "").strip()
                if not master:
                    continue
                lr = str(row.get("RHEA_ID_LR") or "").strip()
                rl = str(row.get("RHEA_ID_RL") or "").strip()
                bi = str(row.get("RHEA_ID_BI") or "").strip()
                directed_map[master] = (master, "UN")
                if lr:
                    directed_map[lr] = (master, "LR")
                if rl:
                    directed_map[rl] = (master, "RL")
                if bi:
                    directed_map[bi] = (master, "BI")

        enzyme_sets: dict[tuple[str, str], set[str]] = defaultdict(set)
        with self._asset_path("sprot").open(encoding="utf-8", errors="replace") as handle:
            rows = csv.DictReader(handle, delimiter="\t")
            for row in rows:
                master = str(row.get("MASTER_ID") or row.get("RHEA_ID") or "").strip()
                direction = str(row.get("DIRECTION") or "UN").strip().upper() or "UN"
                accession = str(row.get("ID") or "").strip()
                if master and accession:
                    enzyme_sets[(master, direction)].add(accession)
                    enzyme_sets[(master, "ALL")].add(accession)
        enzyme_counts = {key: len(values) for key, values in enzyme_sets.items()}

        adjacency: dict[str, list[dict[str, Any]]] = defaultdict(list)
        reverse: dict[str, list[dict[str, Any]]] = defaultdict(list)
        edge_best: dict[tuple[str, str, str, str], dict[str, Any]] = {}
        reaction_rows = mapped_rows = 0

        with self._asset_path("reaction_smiles").open(encoding="utf-8", errors="replace") as handle:
            for line in handle:
                reaction_rows += 1
                parts = line.rstrip("\n\r").split("\t", 1)
                if len(parts) != 2 or ">>" not in parts[1]:
                    continue
                directed_id, reaction_smiles = parts[0].strip(), parts[1].strip()
                master, orientation = directed_map.get(directed_id, (directed_id, "UN"))
                left_raw, right_raw = reaction_smiles.split(">>", 1)

                def map_side(blob: str) -> list[str]:
                    found: list[str] = []
                    for component in blob.split("."):
                        can = _canonical_smiles(component)
                        if not can:
                            continue
                        ids = smiles_to_chebi.get(can) or []
                        for cid in ids:
                            feat = features.get(cid)
                            if feat and feat[0] >= 4 and cid not in CURRENCY_CHEBI:
                                found.append(cid)
                                break
                    return list(dict.fromkeys(found))

                left = map_side(left_raw)
                right = map_side(right_raw)
                if not left or not right:
                    continue

                pair_scores: dict[tuple[str, str], float] = {}
                for src in left:
                    for dst in right:
                        if src == dst:
                            continue
                        score = _pair_score(features[src], features[dst])
                        if score >= 0.12:
                            pair_scores[(src, dst)] = score
                if not pair_scores:
                    continue

                # Hyper-reactions often contain cofactors and several organic products.
                # Retain the best structural continuation for each participant in both
                # directions, rather than connecting every molecule to every molecule.
                chosen: set[tuple[str, str]] = set()
                for src in left:
                    candidates = [(score, dst) for (s, dst), score in pair_scores.items() if s == src]
                    if candidates:
                        best = max(candidates)[0]
                        chosen.update((src, dst) for score, dst in candidates if score >= best - 0.08)
                for dst in right:
                    candidates = [(score, src) for (src, d), score in pair_scores.items() if d == dst]
                    if candidates:
                        best = max(candidates)[0]
                        chosen.update((src, dst) for score, src in candidates if score >= best - 0.08)

                direction_count = enzyme_counts.get((master, orientation), 0)
                all_count = enzyme_counts.get((master, "ALL"), 0)
                unknown_count = enzyme_counts.get((master, "UN"), 0)
                if direction_count > 0:
                    direction_support = 1.0
                elif unknown_count > 0:
                    direction_support = 0.62
                elif all_count > 0:
                    direction_support = 0.48
                else:
                    direction_support = 0.24

                for src, dst in chosen:
                    score = float(pair_scores[(src, dst)])
                    edge = {
                        "source": src,
                        "target": dst,
                        "rhea_id": f"RHEA:{master}",
                        "directed_rhea_id": f"RHEA:{directed_id}",
                        "orientation": "reverse" if orientation == "RL" else "forward",
                        "direction_code": orientation,
                        "transformation_score": round(score, 5),
                        "swissprot_count": int(all_count),
                        "direction_swissprot_count": int(direction_count),
                        "direction_support": float(direction_support),
                    }
                    key = (src, dst, master, orientation)
                    current = edge_best.get(key)
                    if current is None or edge["transformation_score"] > current["transformation_score"]:
                        edge_best[key] = edge
                mapped_rows += 1

        for edge in edge_best.values():
            adjacency[edge["source"]].append(edge)
            reverse_edge = dict(edge)
            reverse_edge["source"], reverse_edge["target"] = edge["target"], edge["source"]
            reverse[edge["target"]].append(reverse_edge)

        # Add stable short aliases without changing the source-of-truth names.
        for alias, canonical_name in ALIASES.items():
            ids = name_to_ids.get(_norm_name(canonical_name), [])
            if ids:
                name_to_ids[_norm_name(alias)].extend(ids)

        return {
            "built_at": time.time(),
            "names": names,
            "name_to_ids": dict(name_to_ids),
            "chebi_smiles": chebi_smiles,
            "adjacency": dict(adjacency),
            "reverse": dict(reverse),
            "enzyme_counts": enzyme_counts,
            "stats": {
                "reaction_smiles_rows": reaction_rows,
                "mapped_reaction_rows": mapped_rows,
                "compound_names": len(names),
                "structured_compounds": len(chebi_smiles),
                "route_edges": len(edge_best),
                "route_nodes": len(set(adjacency) | {e["target"] for rows in adjacency.values() for e in rows}),
            },
        }

    def _load_exact_stoichiometry_maps(self) -> tuple[dict[str, str], dict[str, list[str]]]:
        if self._reaction_smiles_by_id is not None and self._smiles_to_chebi_exact is not None:
            return self._reaction_smiles_by_id, self._smiles_to_chebi_exact
        self._download_asset("reaction_smiles")
        self._download_asset("chebi_smiles")
        reaction_map: dict[str, str] = {}
        with self._asset_path("reaction_smiles").open(encoding="utf-8", errors="replace") as handle:
            for line in handle:
                parts = line.rstrip("\n\r").split("\t", 1)
                if len(parts) == 2 and ">>" in parts[1]:
                    reaction_map[parts[0].strip()] = parts[1].strip()
        smiles_map: dict[str, list[str]] = defaultdict(list)
        with self._asset_path("chebi_smiles").open(encoding="utf-8", errors="replace") as handle:
            for line in handle:
                parts = line.rstrip("\n\r").split("\t", 1)
                if len(parts) != 2:
                    continue
                cid, raw = parts[0].strip(), parts[1].strip()
                canonical = _canonical_smiles(raw)
                if cid and canonical:
                    smiles_map[canonical].append(cid)
        self._reaction_smiles_by_id = reaction_map
        self._smiles_to_chebi_exact = dict(smiles_map)
        return self._reaction_smiles_by_id, self._smiles_to_chebi_exact

    def reaction_stoichiometry(self, directed_rhea_id: str) -> dict[str, Any]:
        """Return full verified Rhea stoichiometry mapped back to ChEBI candidates.

        Route search uses a main-chain projection of a hyper-reaction. Thermodynamics
        and FBA must instead consume the complete reaction. This helper reconstructs
        all participants from the official directed reaction-SMILES file and requires
        an exact structure match to the official Rhea ChEBI-SMILES table.
        """
        raw_id = str(directed_rhea_id or "").upper().replace("RHEA:", "").strip()
        reaction_map, smiles_map = self._load_exact_stoichiometry_maps()
        reaction_smiles = reaction_map.get(raw_id)
        if not reaction_smiles:
            return {"status": "missing_reaction", "directed_rhea_id": f"RHEA:{raw_id}" if raw_id else "", "participants": []}
        left_raw, right_raw = reaction_smiles.split(">>", 1)
        participants: list[dict[str, Any]] = []
        unmapped: list[str] = []
        for coefficient, blob in ((-1.0, left_raw), (1.0, right_raw)):
            grouped: dict[str, dict[str, Any]] = {}
            for component in blob.split("."):
                canonical = _canonical_smiles(component)
                if not canonical:
                    unmapped.append(component)
                    continue
                candidates = list(smiles_map.get(canonical) or [])
                key = canonical
                row = grouped.get(key)
                if row is None:
                    row = {
                        "coefficient": 0.0,
                        "smiles": canonical,
                        "chebi_candidates": candidates,
                    }
                    grouped[key] = row
                row["coefficient"] += coefficient
                if not candidates:
                    unmapped.append(component)
            participants.extend(row for row in grouped.values() if abs(float(row["coefficient"])) > 1e-12)
        return {
            "status": "complete" if participants and not unmapped and all(row.get("chebi_candidates") for row in participants) else "partial",
            "directed_rhea_id": f"RHEA:{raw_id}",
            "reaction_smiles": reaction_smiles,
            "participants": participants,
            "unmapped_components": list(dict.fromkeys(unmapped)),
        }

    def enrich_route_stoichiometry(self, route: dict[str, Any]) -> dict[str, Any]:
        output = dict(route)
        steps = []
        complete = True
        for step in route.get("steps", []):
            item = dict(step)
            directed = str(item.get("directed_rhea_id") or "")
            if item.get("evidence_type") == "predicted_pickaxe" or not directed:
                complete = False
                item["full_stoichiometry"] = {"status": "predicted_or_unavailable", "participants": []}
            else:
                stoich = self.reaction_stoichiometry(directed)
                item["full_stoichiometry"] = stoich
                complete = complete and stoich.get("status") == "complete"
            steps.append(item)
        output["steps"] = steps
        output["full_stoichiometry_complete"] = bool(steps) and complete
        return output

    def resolve_compound(self, terms: Iterable[str], *, limit: int = 5) -> list[dict[str, str]]:
        index = self.ensure_index()
        names: dict[str, str] = index["names"]
        name_to_ids: dict[str, list[str]] = index["name_to_ids"]
        chebi_smiles: dict[str, str] = index["chebi_smiles"]
        output: list[dict[str, str]] = []
        seen: set[str] = set()
        requested = [str(x or "").strip() for x in terms if str(x or "").strip()]

        if self._compound_alias_to_ids is None:
            alias_map: dict[str, list[str]] = defaultdict(list)
            for name_key, values in name_to_ids.items():
                for variant in _biochemical_name_variants(name_key):
                    for cid in values:
                        if cid not in alias_map[variant]:
                            alias_map[variant].append(cid)
            self._compound_alias_to_ids = {key: tuple(values) for key, values in alias_map.items()}

        for term in requested:
            m = re.search(r"CHEBI\s*:\s*(\d+)", term, re.I)
            ids = [f"CHEBI:{m.group(1)}"] if m else []
            if not ids:
                for key in _biochemical_name_variants(term):
                    for cid in name_to_ids.get(key, []):
                        if cid not in ids:
                            ids.append(cid)
            if not ids:
                for key in _biochemical_name_variants(term):
                    for cid in (self._compound_alias_to_ids or {}).get(key, ()):
                        if cid not in ids:
                            ids.append(cid)
            if not ids:
                q_variants = _biochemical_name_variants(term)
                # Last-resort lexical retrieval generates candidates only; it never
                # assigns an identifier outside the official ChEBI name index.
                hits = []
                for name_key, values in name_to_ids.items():
                    name_variants = _biochemical_name_variants(name_key)
                    if any(q and len(q) >= 4 and q in candidate for q in q_variants for candidate in name_variants):
                        hits.extend(values)
                        if len(hits) >= limit * 3:
                            break
                ids = hits
            for cid in ids:
                if cid in seen or cid not in chebi_smiles:
                    continue
                seen.add(cid)
                output.append({"chebi_id": cid, "name": names.get(cid, cid), "smiles": chebi_smiles[cid]})
                if len(output) >= limit:
                    return output
        return output

    def ecoli_start_pool(self) -> set[str]:
        index = self.ensure_index()
        structured = set(index["chebi_smiles"])
        real_pool = self.root / "results/catalyst_finder_runtime/route_feasibility/iML1515_cytosol_chebi.txt"
        if real_pool.exists():
            ids = {line.strip().upper() for line in real_pool.read_text(encoding="utf-8", errors="replace").splitlines() if line.strip()}
            starts = (ids & structured) - CURRENCY_CHEBI
            if starts:
                return starts
        # Fallback only for deployments where the optional COBRApy runtime has not yet
        # been prepared. It preserves route discovery but is not considered host-FBA evidence.
        smiles_to_chebi = {smi: cid for cid, smi in index["chebi_smiles"].items()}
        path = self.root / "external_repos/route_design/MINE-Database/example_data/iML1515_ecoli_GEM.csv"
        if not path.exists():
            return set()
        starts: set[str] = set()
        with path.open(encoding="utf-8", errors="replace") as handle:
            rows = csv.DictReader(handle)
            for row in rows:
                can = _canonical_smiles(str(row.get("smiles") or ""))
                if can and can in smiles_to_chebi:
                    starts.add(smiles_to_chebi[can])
        return starts - CURRENCY_CHEBI

    @staticmethod
    def _edge_cost(edge: dict[str, Any], *, local_reactions: set[str], priority: str) -> float:
        transform = float(edge.get("transformation_score") or 0.0)
        enzyme_n = int(edge.get("swissprot_count") or 0)
        direction = float(edge.get("direction_support") or 0.0)
        local = 1.0 if edge.get("rhea_id") in local_reactions else 0.0
        enzyme_score = min(1.0, math.log1p(enzyme_n) / math.log(21.0))
        if priority == "short":
            weights = (0.20, 0.12, 0.10, 0.05)
        elif priority == "enzyme_available":
            weights = (0.18, 0.42, 0.14, 0.08)
        elif priority == "project_covered":
            weights = (0.18, 0.22, 0.12, 0.30)
        else:
            weights = (0.28, 0.30, 0.16, 0.12)
        bonus = weights[0] * transform + weights[1] * enzyme_score + weights[2] * direction + weights[3] * local
        # Each step always costs at least 0.30; therefore a longer path must earn
        # meaningful evidence advantages to outrank a short one.
        return max(0.30, 1.15 - bonus)

    def _search(
        self,
        starts: set[str],
        target: str,
        *,
        max_steps: int,
        limit: int,
        local_reactions: set[str],
        priority: str,
    ) -> list[list[dict[str, Any]]]:
        index = self.ensure_index()
        adjacency: dict[str, list[dict[str, Any]]] = index["adjacency"]

        # Collapse parallel Rhea transformations between the same compound pair to
        # the lowest-cost edge for this ranking objective, then use NetworkX's
        # mature k-shortest-simple-path implementation. The full Rhea ID remains on
        # each selected edge for downstream verification.
        graph = nx.DiGraph()
        best_edge: dict[tuple[str, str], dict[str, Any]] = {}
        for src, rows in adjacency.items():
            for edge in rows:
                dst = str(edge["target"])
                cost = self._edge_cost(edge, local_reactions=local_reactions, priority=priority)
                key = (src, dst)
                current = best_edge.get(key)
                if current is None or cost < float(current["_route_cost"]):
                    chosen = dict(edge)
                    chosen["_route_cost"] = float(cost)
                    best_edge[key] = chosen
        for (src, dst), edge in best_edge.items():
            graph.add_edge(src, dst, weight=float(edge["_route_cost"]))

        super_source = "__CATALYST_FINDER_ROUTE_SOURCE__"
        graph.add_node(super_source)
        for start in starts:
            if start in graph and start != target:
                graph.add_edge(super_source, start, weight=0.0)
        if target not in graph or graph.out_degree(super_source) == 0:
            return []

        results: list[list[dict[str, Any]]] = []
        seen: set[tuple[tuple[str, str, str], ...]] = set()
        try:
            generator = nx.shortest_simple_paths(graph, super_source, target, weight="weight")
            inspected = 0
            for node_path in generator:
                inspected += 1
                if inspected > max(400, limit * 80):
                    break
                nodes = [node for node in node_path if node != super_source]
                if len(nodes) < 2 or len(nodes) - 1 > max_steps:
                    continue
                edges: list[dict[str, Any]] = []
                valid = True
                for src, dst in zip(nodes, nodes[1:]):
                    edge = best_edge.get((src, dst))
                    if edge is None:
                        valid = False
                        break
                    clean = dict(edge)
                    clean.pop("_route_cost", None)
                    edges.append(clean)
                if not valid or not edges:
                    continue
                signature = tuple((str(e["rhea_id"]), str(e["source"]), str(e["target"])) for e in edges)
                if signature in seen:
                    continue
                seen.add(signature)
                results.append(edges)
                if len(results) >= limit * 4:
                    break
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return []
        return results

    def pickaxe_available(self) -> bool:
        return (
            self.pickaxe_worker.is_file()
            and self.pickaxe_vendor.is_dir()
            and (self.pickaxe_site / "libsbml").exists()
            and (self.pickaxe_site / "lxml").exists()
        )

    def _run_pickaxe(self, start_smiles: str, *, timeout: int = 28) -> dict[str, Any]:
        if not self.pickaxe_available():
            raise RouteDesignError("隔离的 MINE/Pickaxe worker 尚未准备好。")
        try:
            completed = subprocess.run(
                [sys.executable, str(self.pickaxe_worker)],
                input=json.dumps({"start_smiles": start_smiles}, ensure_ascii=False),
                text=True,
                capture_output=True,
                cwd=str(self.root),
                timeout=timeout,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise RouteDesignError("MINE/Pickaxe 预测扩展超过安全时间限制，已停止本次探索。") from exc
        if completed.returncode != 0:
            detail = (completed.stderr or completed.stdout or "worker failed")[-1200:]
            raise RouteDesignError(f"MINE/Pickaxe 预测扩展失败: {detail}")
        try:
            data = json.loads(completed.stdout)
        except json.JSONDecodeError as exc:
            raise RouteDesignError("MINE/Pickaxe worker 没有返回有效 JSON。") from exc
        if not isinstance(data, dict):
            raise RouteDesignError("MINE/Pickaxe worker 返回格式不正确。")
        return data

    def _connectivity_to_chebi(self) -> dict[str, list[str]]:
        if self.connectivity_path.exists():
            try:
                with gzip.open(self.connectivity_path, "rb") as handle:
                    data = pickle.load(handle)
                if isinstance(data, dict):
                    return data
            except Exception:
                self.connectivity_path.unlink(missing_ok=True)
        index = self.ensure_index()
        mapping: dict[str, list[str]] = defaultdict(list)
        for cid, smi in index["chebi_smiles"].items():
            key = _connectivity_key(smi)
            if key:
                mapping[key].append(cid)
        output = dict(mapping)
        tmp = self.connectivity_path.with_suffix(".tmp.gz")
        with gzip.open(tmp, "wb", compresslevel=4) as handle:
            pickle.dump(output, handle, protocol=pickle.HIGHEST_PROTOCOL)
        tmp.replace(self.connectivity_path)
        return output

    def explore_predicted_bridges(
        self,
        *,
        source_chebi_id: str,
        target_chebi_id: str,
        max_steps: int,
        limit: int,
        priority: str,
        local_reaction_ids: Iterable[str] = (),
    ) -> dict[str, Any]:
        index = self.ensure_index()
        chebi_smiles: dict[str, str] = index["chebi_smiles"]
        names: dict[str, str] = index["names"]
        source_smiles = chebi_smiles.get(source_chebi_id)
        target_smiles = chebi_smiles.get(target_chebi_id)
        if not source_smiles or not target_smiles:
            raise RouteDesignError("预测探索需要已经核对并具有结构的起始前体和目标产物。")

        worker = self._run_pickaxe(source_smiles)
        connectivity_to_chebi = self._connectivity_to_chebi()

        known_direct = {
            str(edge["target"])
            for edge in index["adjacency"].get(source_chebi_id, [])
        }
        target_key = _connectivity_key(target_smiles)
        bridges: dict[str, dict[str, Any]] = {}
        duplicate_known = 0
        for prediction in worker.get("predictions", []):
            product_smiles = _canonical_smiles(str(prediction.get("product_smiles") or ""))
            if not product_smiles:
                continue
            pmol = Chem.MolFromSmiles(product_smiles)
            if pmol is None or sum(1 for atom in pmol.GetAtoms() if atom.GetAtomicNum() == 6) < 2:
                continue
            key = _connectivity_key(product_smiles)
            matched = list(connectivity_to_chebi.get(key, []))
            if key == target_key and target_chebi_id not in matched:
                matched.insert(0, target_chebi_id)
            for cid in matched:
                if cid == source_chebi_id:
                    continue
                if cid in known_direct:
                    duplicate_known += 1
                    continue
                current = bridges.get(cid)
                record = {
                    "source": source_chebi_id,
                    "target": cid,
                    "source_name": names.get(source_chebi_id, source_chebi_id),
                    "target_name": names.get(cid, cid),
                    "product_smiles": product_smiles,
                    "prediction_rules": list(prediction.get("rules") or []),
                    "reaction_smiles": str(prediction.get("reaction_smiles") or ""),
                    "evidence_type": "predicted_pickaxe",
                }
                if current is None or len(record["prediction_rules"]) < len(current["prediction_rules"]):
                    bridges[cid] = record

        if not bridges:
            return {
                "status": "completed",
                "worker": {k: worker.get(k) for k in ("engine", "generation", "operators", "generated_compounds", "generated_reactions")},
                "mapped_bridge_count": 0,
                "known_duplicate_count": duplicate_known,
                "routes": [],
            }

        local = {str(x) for x in local_reaction_ids}
        bridge_ids = set(bridges)
        raw_suffixes: list[list[dict[str, Any]]] = []
        remaining = bridge_ids - {target_chebi_id}
        if remaining and max_steps > 1:
            # Multi-source suffix search must look substantially deeper than the final
            # exploratory Top-K: short suffixes that loop back through the original
            # source are rejected below and can otherwise occupy the whole K-path budget.
            suffix_search_limit = min(80, max(60, limit * 12))
            raw_suffixes = self._search(
                remaining, target_chebi_id, max_steps=max_steps - 1,
                limit=suffix_search_limit, local_reactions=local, priority=priority,
            )

        candidates: list[dict[str, Any]] = []
        suffix_sets = ([[]] if target_chebi_id in bridge_ids else []) + raw_suffixes
        for suffix in suffix_sets:
            suffix_nodes = {str(edge.get("source") or "") for edge in suffix} | {str(edge.get("target") or "") for edge in suffix}
            if source_chebi_id in suffix_nodes:
                continue
            bridge_target = target_chebi_id if not suffix else str(suffix[0]["source"])
            bridge = bridges.get(bridge_target)
            if not bridge:
                continue
            steps = [{
                "step_index": 1,
                "source": source_chebi_id,
                "target": bridge_target,
                "source_name": bridge["source_name"],
                "target_name": bridge["target_name"],
                "rhea_id": None,
                "orientation": "predicted",
                "swissprot_count": 0,
                "local_model_ready": False,
                "evidence_type": "predicted_pickaxe",
                "prediction_rules": bridge["prediction_rules"],
                "reaction_smiles": bridge["reaction_smiles"],
            }]
            for i, edge in enumerate(suffix, start=2):
                steps.append({
                    "step_index": i, **edge,
                    "source_name": names.get(edge["source"], edge["source"]),
                    "target_name": names.get(edge["target"], edge["target"]),
                    "local_model_ready": edge.get("rhea_id") in local,
                    "evidence_type": "known_rhea",
                })
            nodes = [source_chebi_id, bridge_target] + [str(e["target"]) for e in suffix]
            step_count = len(steps)
            known_steps = max(0, step_count - 1)
            enzyme_scores = [
                min(1.0, math.log1p(int(e.get("swissprot_count") or 0)) / math.log(21.0))
                for e in suffix
            ]
            suffix_availability = sum(enzyme_scores) / known_steps if known_steps else 0.0
            length_score = 1.0 if step_count == 1 else max(0.0, 1.0 - (step_count - 1) / max(1, max_steps))
            # Predicted bridges are intentionally penalized relative to fully known Rhea
            # routes. Their score ranks only within the exploratory section.
            score = 100.0 * (0.40 * length_score + 0.25 * suffix_availability + 0.15 * (known_steps / step_count) + 0.20 * 0.35)
            signature = bridge_target + "|" + "|".join(str(e.get("rhea_id") or "") for e in suffix)
            candidates.append({
                "route_id": "PX-" + hashlib.sha256(signature.encode()).hexdigest()[:10],
                "route_type": "predicted_bridge",
                "score": round(score, 2),
                "compound_ids": nodes,
                "compound_names": [names.get(cid, cid) for cid in nodes],
                "steps": steps,
                "metrics": {
                    "step_count": step_count,
                    "known_rhea_steps": known_steps,
                    "predicted_steps": 1,
                    "enzyme_availability_known_suffix": round(suffix_availability, 4),
                },
                "evidence_note": "第一步由隔离的 MINE/Pickaxe + MetaCyc rule 预测；后续步骤（如有）来自 Rhea。预测步骤不是数据库事实，必须独立验证。",
            })
        candidates.sort(key=lambda r: (-float(r["score"]), int(r["metrics"]["step_count"]), r["route_id"]))
        candidates = candidates[:limit]
        for rank, route in enumerate(candidates, start=1):
            route["rank"] = rank
        return {
            "status": "completed",
            "worker": {k: worker.get(k) for k in ("engine", "generation", "operators", "generated_compounds", "generated_reactions")},
            "mapped_bridge_count": len(bridges),
            "known_duplicate_count": duplicate_known,
            "routes": candidates,
        }

    def design(
        self,
        *,
        source_terms: list[str],
        target_terms: list[str],
        host: str = "",
        max_steps: int = 6,
        limit: int = 10,
        candidate_limit: int | None = None,
        priority: str = "balanced",
        local_reaction_ids: Iterable[str] = (),
    ) -> dict[str, Any]:
        max_steps = max(1, min(int(max_steps or 6), 8))
        limit = max(1, min(int(limit or 10), 20))
        internal_limit = max(limit, min(int(candidate_limit or limit), 80))
        local_reactions = {str(x) for x in local_reaction_ids}
        target_candidates = self.resolve_compound(target_terms, limit=5)
        if not target_candidates:
            raise RouteDesignError("没有在 Rhea 参与物中核对到目标化合物；请尝试标准英文名称或 ChEBI ID。")
        target = target_candidates[0]

        source_candidates = self.resolve_compound(source_terms, limit=8) if source_terms else []
        starts = {row["chebi_id"] for row in source_candidates}
        start_mode = "explicit"
        host_norm = _norm_name(host)
        if not starts and ("coli" in host_norm or "大肠杆菌" in host_norm or "escherichia" in host_norm):
            starts = self.ecoli_start_pool()
            start_mode = "ecoli_iML1515_pool"
        if not starts:
            raise RouteDesignError("路线推荐需要一个起始前体，或者明确宿主（目前可直接识别 E. coli / 大肠杆菌的 iML1515 代谢物池）。")

        raw_paths = self._search(
            starts,
            target["chebi_id"],
            max_steps=max_steps,
            limit=internal_limit,
            local_reactions=local_reactions,
            priority=priority,
        )
        names: dict[str, str] = self.ensure_index()["names"]
        routes: list[dict[str, Any]] = []
        for edges in raw_paths:
            if not edges:
                continue
            step_count = len(edges)
            swiss = [int(e.get("swissprot_count") or 0) for e in edges]
            enzyme_scores = [min(1.0, math.log1p(n) / math.log(21.0)) for n in swiss]
            transform_scores = [float(e.get("transformation_score") or 0.0) for e in edges]
            dir_scores = [float(e.get("direction_support") or 0.0) for e in edges]
            local_flags = [1.0 if e.get("rhea_id") in local_reactions else 0.0 for e in edges]
            length_score = 1.0 if step_count == 1 else max(0.0, 1.0 - (step_count - 1) / max(1, max_steps))
            metrics = {
                "step_count": step_count,
                "length_score": round(length_score, 4),
                "enzyme_availability": round(sum(enzyme_scores) / step_count, 4),
                "transformation_continuity": round(sum(transform_scores) / step_count, 4),
                "direction_evidence": round(sum(dir_scores) / step_count, 4),
                "project_model_coverage": round(sum(local_flags) / step_count, 4),
                "min_swissprot_count": min(swiss) if swiss else 0,
            }
            if priority == "short":
                total = 0.70 * metrics["length_score"] + 0.10 * metrics["enzyme_availability"] + 0.08 * metrics["transformation_continuity"] + 0.07 * metrics["direction_evidence"] + 0.05 * metrics["project_model_coverage"]
            elif priority == "enzyme_available":
                total = 0.18 * metrics["length_score"] + 0.46 * metrics["enzyme_availability"] + 0.14 * metrics["transformation_continuity"] + 0.12 * metrics["direction_evidence"] + 0.10 * metrics["project_model_coverage"]
            elif priority == "project_covered":
                total = 0.18 * metrics["length_score"] + 0.24 * metrics["enzyme_availability"] + 0.12 * metrics["transformation_continuity"] + 0.10 * metrics["direction_evidence"] + 0.36 * metrics["project_model_coverage"]
            else:
                total = 0.28 * metrics["length_score"] + 0.30 * metrics["enzyme_availability"] + 0.20 * metrics["transformation_continuity"] + 0.12 * metrics["direction_evidence"] + 0.10 * metrics["project_model_coverage"]
            nodes = [edges[0]["source"]] + [e["target"] for e in edges]
            signature = "|".join(str(e["rhea_id"]) + ":" + str(e["orientation"]) for e in edges)
            route_id = "RR-" + hashlib.sha256(signature.encode("utf-8")).hexdigest()[:10]
            steps = []
            for i, edge in enumerate(edges, start=1):
                steps.append({
                    "step_index": i,
                    **edge,
                    "source_name": names.get(edge["source"], edge["source"]),
                    "target_name": names.get(edge["target"], edge["target"]),
                    "local_model_ready": edge.get("rhea_id") in local_reactions,
                })
            routes.append({
                "route_id": route_id,
                "route_type": "known_rhea",
                "score": round(100.0 * total, 2),
                "base_route_score": round(100.0 * total, 2),
                "metrics": metrics,
                "compound_ids": nodes,
                "compound_names": [names.get(cid, cid) for cid in nodes],
                "steps": steps,
                "thermodynamics": {"status": "not_computed", "note": "当前排名没有把反应方向伪装成 ΔG；热力学需独立计算后再加入。"},
                "evidence_note": "所有步骤来自 Rhea 已收录反应；路线层的主底物/产物连接由结构连续性筛选得到，最终仍应逐步核对完整 Rhea 方程。",
            })
        routes.sort(key=lambda row: (-float(row["score"]), int(row["metrics"]["step_count"]), row["route_id"]))
        routes = routes[:internal_limit]
        for rank, route in enumerate(routes, start=1):
            route["base_rank"] = rank
            route["rank"] = rank
        return {
            "engine": "rhea_full_graph_v1",
            "source_mode": start_mode,
            "source_candidates": source_candidates,
            "target_candidates": target_candidates,
            "selected_target": target,
            "host": host,
            "priority": priority,
            "max_steps": max_steps,
            "route_count": len(routes),
            "requested_route_count": limit,
            "candidate_limit": internal_limit,
            "routes": routes,
            "graph_stats": dict(self.ensure_index()["stats"]),
        }
