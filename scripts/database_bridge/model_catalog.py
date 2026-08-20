"""Read-only browser projection of the deployed terpene model data universe."""
from __future__ import annotations

import csv
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


class ModelDataCatalog:
    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        production = self.root / "results/terpene_production_models/marts_adapted_drfp_pu"
        self.paths = {
            "proteins": production / "protein_registry.csv",
            "reactions": production / "reaction_registry.csv",
            "pairs": production / "training_pairs.csv",
            "marts_enzymes": self.root / "data/terpene_marts/marts_enzymes.tsv",
            "marts_reactions": self.root / "data/terpene_marts/marts_reactions.tsv",
            "current_sequences": self.root / "data/terpene/all_seq_terpene_synthase.tsv",
        }
        missing = [str(path.relative_to(self.root)) for path in self.paths.values() if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"Missing model catalog files: {', '.join(missing)}")
        protein_meta = self._protein_metadata()
        reaction_meta = self._reaction_metadata()
        self.proteins = self._load_proteins(protein_meta)
        self.reactions = self._load_reactions(reaction_meta)
        self.protein_by_id = {item["id"]: item for item in self.proteins}
        self.reaction_by_id = {item["id"]: item for item in self.reactions}
        self.pairs = self._load_pairs()
        self.pairs_by_protein: dict[str, list[dict[str, Any]]] = defaultdict(list)
        self.pairs_by_reaction: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for pair in self.pairs:
            self.pairs_by_protein[pair["protein_id"]].append(pair)
            self.pairs_by_reaction[pair["reaction_id"]].append(pair)

    @staticmethod
    def _rows(path: Path, delimiter: str = ",") -> list[dict[str, str]]:
        with path.open(encoding="utf-8", newline="") as handle:
            return list(csv.DictReader(handle, delimiter=delimiter))

    def _protein_metadata(self) -> dict[str, dict[str, Any]]:
        sequence_map = {row.get("Entry", "").strip(): (row.get("Sequence") or "").strip() for row in self._rows(self.paths["current_sequences"], "\t")}
        metadata: dict[str, dict[str, Any]] = {}
        for row in self._rows(self.paths["marts_enzymes"], "\t"):
            ids = {value.strip() for value in [row.get("enzyme_id", ""), row.get("uniprot_id", ""), row.get("genbank_id", "")] if value and value.strip()}
            sequence = (row.get("sequence") or "").strip()
            payload = {
                "name": (row.get("enzyme_name") or "").strip() or None,
                "uniprot_id": (row.get("uniprot_id") or "").strip() or None,
                "genbank_id": (row.get("genbank_id") or "").strip() or None,
                "species": (row.get("species") or "").strip() or None,
                "kingdom": (row.get("kingdom") or "").strip() or None,
                "terpene_type": (row.get("terpene_type") or "unknown").strip() or "unknown",
                "tps_class": (row.get("tps_class") or "unknown").strip() or "unknown",
                "sequence_length": len(sequence) if sequence else None,
            }
            for identifier in ids:
                metadata[identifier] = payload
        for identifier, sequence in sequence_map.items():
            metadata.setdefault(identifier, {})["sequence_length"] = len(sequence) if sequence else None
        return metadata

    def _reaction_metadata(self) -> dict[str, dict[str, Any]]:
        metadata: dict[str, dict[str, Any]] = {}
        for row in self._rows(self.paths["marts_reactions"], "\t"):
            signature = (row.get("reaction_signature") or "").strip()
            canonical = (row.get("canonical_reaction") or "").strip()
            substrate = (row.get("substrate_name") or "").strip()
            product = (row.get("product_name") or "").strip()
            payload = {
                "name": f"{substrate or '?'} → {product or '?'}",
                "substrate_name": substrate or None,
                "product_name": product or None,
                "terpene_type": (row.get("terpene_type") or "unknown").strip() or "unknown",
                "has_mechanism": bool((row.get("mechanism_marts_id") or "").strip() not in {"", "no_mechanism"}),
                "mechanism_id": (row.get("mechanism_marts_id") or "").strip() or None,
            }
            if signature:
                metadata[signature] = payload
            if canonical:
                metadata[canonical] = payload
        return metadata

    def _load_proteins(self, metadata: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
        items = []
        for row in self._rows(self.paths["proteins"]):
            pid = (row.get("protein_id") or "").strip()
            if not pid:
                continue
            source = (row.get("source") or "unknown").strip()
            meta = metadata.get(pid, {})
            items.append({
                "id": pid, "kind": "protein", "name": meta.get("name") or pid,
                "uniprot_id": meta.get("uniprot_id"), "genbank_id": meta.get("genbank_id"),
                "species": meta.get("species"), "kingdom": meta.get("kingdom"),
                "terpene_type": meta.get("terpene_type", "unknown"), "tps_class": meta.get("tps_class", "unknown"),
                "sequence_length": meta.get("sequence_length"), "source": source,
                "seen": source == "current", "registered": source != "current",
                "source_file": str(self.paths["proteins"].relative_to(self.root)),
            })
        return items

    def _load_reactions(self, metadata: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
        items = []
        for row in self._rows(self.paths["reactions"]):
            rid = (row.get("reaction_id") or "").strip()
            if not rid:
                continue
            source = (row.get("source") or "unknown").strip()
            signature = (row.get("reaction_signature") or "").strip()
            smiles = (row.get("reaction_smiles") or "").strip()
            meta = metadata.get(signature) or metadata.get(smiles) or {}
            items.append({
                "id": rid, "kind": "reaction", "name": meta.get("name") or rid,
                "substrate_name": meta.get("substrate_name"), "product_name": meta.get("product_name"),
                "reaction_smiles": smiles, "terpene_type": meta.get("terpene_type", "unknown"),
                "has_mechanism": bool(meta.get("has_mechanism")), "mechanism_id": meta.get("mechanism_id"),
                "source": source, "seen": source == "current", "registered": source != "current",
                "source_file": str(self.paths["reactions"].relative_to(self.root)),
            })
        return items

    def _load_pairs(self) -> list[dict[str, Any]]:
        items = []
        for index, row in enumerate(self._rows(self.paths["pairs"])):
            pid, rid = (row.get("Entry") or "").strip(), (row.get("rhea_id") or "").strip()
            if not pid or not rid:
                continue
            protein = self.protein_by_id.get(pid, {"name": pid, "source": "unknown", "terpene_type": "unknown"})
            reaction = self.reaction_by_id.get(rid, {"name": rid, "source": "unknown", "terpene_type": "unknown"})
            p_seen, r_seen = protein.get("source") == "current", reaction.get("source") == "current"
            category = f"{'seen' if p_seen else 'unseen'}_enzyme_{'seen' if r_seen else 'unseen'}_reaction"
            items.append({
                "id": f"PAIR:{index:05d}", "protein_id": pid, "reaction_id": rid,
                "protein_name": protein.get("name", pid), "reaction_name": reaction.get("name", rid),
                "species": protein.get("species"), "kingdom": protein.get("kingdom"),
                "terpene_type": reaction.get("terpene_type") if reaction.get("terpene_type") != "unknown" else protein.get("terpene_type", "unknown"),
                "tps_class": protein.get("tps_class"), "open_world_category": category,
                "protein_seen": p_seen, "reaction_seen": r_seen,
                "has_mechanism": bool(reaction.get("has_mechanism")),
                "association_source": (row.get("source") or "unknown").strip(),
                "source_file": str(self.paths["pairs"].relative_to(self.root)),
            })
        return items

    def summary(self) -> dict[str, Any]:
        buckets = lambda values: [{"label": key, "count": count} for key, count in Counter(values).most_common()]
        return {
            "proteins": len(self.proteins), "reactions": len(self.reactions), "associations": len(self.pairs),
            "registered_proteins": sum(item["registered"] for item in self.proteins),
            "registered_reactions": sum(item["registered"] for item in self.reactions),
            "mechanism_reactions": sum(item["has_mechanism"] for item in self.reactions),
            "seen_proteins": sum(item["seen"] for item in self.proteins),
            "seen_reactions": sum(item["seen"] for item in self.reactions),
            "terpene_types": buckets(item["terpene_type"] for item in self.pairs),
            "open_world_categories": buckets(item["open_world_category"] for item in self.pairs),
            "source_files": [str(self.paths[key].relative_to(self.root)) for key in ("proteins", "reactions", "pairs", "marts_enzymes", "marts_reactions")],
            "read_only": True, "catalog_contract": "deployed-candidate-universe-v1",
        }

    def search(self, query: str = "", kind: str = "all", limit: int = 40) -> dict[str, Any]:
        query_l = query.strip().lower(); limit = max(1, min(int(limit), 100)); items: list[dict[str, Any]] = []
        if kind in {"all", "protein"}: items.extend(item for item in self.proteins if not query_l or query_l in _blob(item))
        if kind in {"all", "reaction"}: items.extend(item for item in self.reactions if not query_l or query_l in _blob(item))
        if kind in {"all", "association"}: items.extend({**item, "kind": "association", "name": f"{item['protein_name']} → {item['reaction_name']}"} for item in self.pairs if not query_l or query_l in _blob(item))
        items.sort(key=lambda item: (_rank(item, query_l), item.get("kind", ""), item.get("name", "")))
        return {"query": query, "kind": kind, "items": items[:limit], "limit": limit, "total_returned": min(len(items), limit)}

    def graph(self, query: str = "", limit: int = 36, focus_id: str | None = None) -> dict[str, Any]:
        limit = max(4, min(int(limit), 80)); query_l = query.strip().lower()
        if focus_id:
            selected = self.pairs_by_protein.get(focus_id, []) + self.pairs_by_reaction.get(focus_id, [])
        elif query_l:
            selected = [item for item in self.pairs if query_l in _blob(item)]
        else:
            selected = _balanced(self.pairs, limit)
        selected = selected[:limit]
        protein_ids = list(dict.fromkeys(item["protein_id"] for item in selected))
        reaction_ids = list(dict.fromkeys(item["reaction_id"] for item in selected))
        nodes = [{**self.protein_by_id[pid], "degree": sum(item["protein_id"] == pid for item in selected)} for pid in protein_ids if pid in self.protein_by_id]
        nodes += [{**self.reaction_by_id[rid], "degree": sum(item["reaction_id"] == rid for item in selected)} for rid in reaction_ids if rid in self.reaction_by_id]
        if focus_id and not nodes:
            record = self.protein_by_id.get(focus_id) or self.reaction_by_id.get(focus_id)
            if record:
                nodes = [{**record, "degree": 0}]
        return {"query": query, "focus_id": focus_id, "nodes": nodes, "edges": selected, "node_count": len(nodes), "edge_count": len(selected), "total_associations": len(self.pairs), "truncated": len(selected) >= limit, "read_only": True}

    def entity(self, kind: str, identifier: str) -> dict[str, Any] | None:
        if kind == "protein": record, pairs = self.protein_by_id.get(identifier), self.pairs_by_protein.get(identifier, [])
        elif kind == "reaction": record, pairs = self.reaction_by_id.get(identifier), self.pairs_by_reaction.get(identifier, [])
        else: return None
        return None if record is None else {**record, "associations": pairs[:100], "association_count": len(pairs), "read_only": True}


def _blob(item: dict[str, Any]) -> str:
    return " ".join(str(value) for value in item.values() if value is not None).lower()


def _rank(item: dict[str, Any], query: str) -> int:
    if not query: return 2
    identifier, name = str(item.get("id", "")).lower(), str(item.get("name", "")).lower()
    if identifier == query or name == query: return 0
    if identifier.startswith(query) or name.startswith(query): return 1
    return 2


def _balanced(pairs: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for pair in pairs: groups[(pair["terpene_type"], pair["open_world_category"])].append(pair)
    selected, buckets, index = [], [values for _, values in sorted(groups.items())], 0
    while buckets and len(selected) < limit:
        next_buckets = []
        for values in buckets:
            if index < len(values):
                selected.append(values[index]); next_buckets.append(values)
                if len(selected) >= limit: break
        buckets, index = next_buckets, index + 1
    return selected
