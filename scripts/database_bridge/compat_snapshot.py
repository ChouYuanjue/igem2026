"""Read-only compatibility payloads for the pinned igem_database frontend.

The snapshot exists only so the already-built upstream map/search/detail UI can be
shown when no live database API is configured. It deliberately does not implement
unfinished account, persistence, homology-job, or download-generation features.
"""

from __future__ import annotations

from typing import Any

COMPOUNDS: list[dict[str, Any]] = [
    {"compoundId": "CHEBI:15422", "name": "Geranyl diphosphate", "chebiId": "CHEBI:15422", "formula": "C10H20O7P2", "averageMass": 314.21, "smiles": "CC(=CCCOP(=O)(O)OP(=O)(O)O)C", "description": "Monoterpene precursor represented in the upstream database demo."},
    {"compoundId": "CHEBI:17580", "name": "Linalool", "chebiId": "CHEBI:17580", "formula": "C10H18O", "averageMass": 154.25, "description": "Acyclic monoterpene alcohol."},
    {"compoundId": "CHEBI:10280", "name": "Limonene", "chebiId": "CHEBI:10280", "formula": "C10H16", "averageMass": 136.24, "description": "Cyclic monoterpene product."},
    {"compoundId": "CHEBI:17292", "name": "Farnesyl diphosphate", "chebiId": "CHEBI:17292", "formula": "C15H28O7P2", "averageMass": 382.33, "description": "Sesquiterpene precursor."},
    {"compoundId": "CHEBI:6467", "name": "Germacrene D", "chebiId": "CHEBI:6467", "formula": "C15H24", "averageMass": 204.36, "description": "Sesquiterpene hydrocarbon product."},
    {"compoundId": "CHEBI:28036", "name": "Nerolidol", "chebiId": "CHEBI:28036", "formula": "C15H26O", "averageMass": 222.37, "description": "Acyclic sesquiterpene alcohol."},
    {"compoundId": "CHEBI:17221", "name": "Myrcene", "chebiId": "CHEBI:17221", "formula": "C10H16", "averageMass": 136.24, "description": "Acyclic monoterpene hydrocarbon."},
    {"compoundId": "CHEBI:17115", "name": "Alpha-pinene", "chebiId": "CHEBI:17115", "formula": "C10H16", "averageMass": 136.24, "description": "Bicyclic monoterpene product."},
    {"compoundId": "CHEBI:15377", "name": "Water", "chebiId": "CHEBI:15377", "formula": "H2O", "averageMass": 18.02, "description": "Reaction participant."},
]


def enzyme_card(edge_id: str, enzyme_id: str, name: str, uniprot_id: str, reaction_id: str, equation: str, organism: str, ec: str = "4.2.3.-", source: str = "swiss_prot") -> dict[str, Any]:
    return {
        "edgeId": edge_id,
        "enzymeId": enzyme_id,
        "primaryName": name,
        "uniprotId": uniprot_id,
        "databaseCode": f"TA-{enzyme_id.replace(':', '-')}",
        "organismName": organism,
        "ecNumber": ec,
        "reactionId": reaction_id,
        "reactionEquation": equation,
        "reactionDirection": "forward",
        "sourceType": source,
        "reviewStatus": "official" if source == "swiss_prot" else "reviewed",
    }

EDGES: list[dict[str, Any]] = [
    {"edgeId": "EDGE-GPP-LIN", "edgeGroupId": "GROUP-GPP-LIN", "reactionId": "RHEA:24464", "enzymeId": "ENZ:Q9ZSY2", "sourceCompoundId": "CHEBI:15422", "targetCompoundId": "CHEBI:17580", "label": "Q9ZSY2", "direction": "forward", "sourceType": "swiss_prot", "reviewStatus": "official", "card": enzyme_card("EDGE-GPP-LIN", "ENZ:Q9ZSY2", "Linalool synthase", "Q9ZSY2", "RHEA:24464", "GPP → Linalool", "Arabidopsis thaliana", "4.2.3.26")},
    {"edgeId": "EDGE-GPP-LIM", "edgeGroupId": "GROUP-GPP-LIM", "reactionId": "RHEA:32731", "enzymeId": "ENZ:Q40322", "sourceCompoundId": "CHEBI:15422", "targetCompoundId": "CHEBI:10280", "label": "Q40322", "direction": "forward", "sourceType": "swiss_prot", "reviewStatus": "official", "card": enzyme_card("EDGE-GPP-LIM", "ENZ:Q40322", "Limonene synthase", "Q40322", "RHEA:32731", "GPP → Limonene", "Mentha spicata")},
    {"edgeId": "EDGE-FPP-GER", "edgeGroupId": "GROUP-FPP-GER", "reactionId": "RHEA:68824", "enzymeId": "ENZ:GDS001", "sourceCompoundId": "CHEBI:17292", "targetCompoundId": "CHEBI:6467", "label": "GDS001", "direction": "forward", "sourceType": "manual_literature", "reviewStatus": "reviewed", "card": enzyme_card("EDGE-FPP-GER", "ENZ:GDS001", "Germacrene D synthase", "", "RHEA:68824", "FPP → Germacrene D", "Solidago canadensis", source="manual_literature")},
    {"edgeId": "EDGE-FPP-NER", "edgeGroupId": "GROUP-FPP-NER", "reactionId": "RHEA:27530", "enzymeId": "ENZ:NES001", "sourceCompoundId": "CHEBI:17292", "targetCompoundId": "CHEBI:28036", "label": "NES001", "direction": "forward", "sourceType": "manual_literature", "reviewStatus": "reviewed", "card": enzyme_card("EDGE-FPP-NER", "ENZ:NES001", "Nerolidol synthase", "", "RHEA:27530", "FPP → Nerolidol", "Zea mays", source="manual_literature")},
    {"edgeId": "EDGE-GPP-MYR", "edgeGroupId": "GROUP-GPP-MYR", "reactionId": "RHEA:33991", "enzymeId": "ENZ:MYS001", "sourceCompoundId": "CHEBI:15422", "targetCompoundId": "CHEBI:17221", "label": "MYS001", "direction": "forward", "sourceType": "ai_literature", "reviewStatus": "reviewed", "card": enzyme_card("EDGE-GPP-MYR", "ENZ:MYS001", "Myrcene synthase", "", "RHEA:33991", "GPP → Myrcene", "Ocimum basilicum", source="ai_literature")},
    {"edgeId": "EDGE-GPP-PIN-A", "edgeGroupId": "GROUP-GPP-PIN", "reactionId": "RHEA:31807", "enzymeId": "ENZ:PINS001", "sourceCompoundId": "CHEBI:15422", "targetCompoundId": "CHEBI:17115", "label": "PINS001", "direction": "forward", "sourceType": "swiss_prot", "reviewStatus": "official", "card": enzyme_card("EDGE-GPP-PIN-A", "ENZ:PINS001", "Alpha-pinene synthase", "P0C565", "RHEA:31807", "GPP → Alpha-pinene", "Abies grandis")},
    {"edgeId": "EDGE-GPP-PIN-B", "edgeGroupId": "GROUP-GPP-PIN", "reactionId": "RHEA:31831", "enzymeId": "ENZ:PINS002", "sourceCompoundId": "CHEBI:15422", "targetCompoundId": "CHEBI:17115", "label": "PINS002", "direction": "forward", "sourceType": "ai_literature", "reviewStatus": "pending", "card": enzyme_card("EDGE-GPP-PIN-B", "ENZ:PINS002", "Alpha-pinene synthase candidate", "", "RHEA:31831", "GPP → Alpha-pinene", "Pinus taeda", source="ai_literature")},
]

EDGE_GROUPS = [
    {"edgeGroupId": "GROUP-GPP-PIN", "sourceCompoundId": "CHEBI:15422", "targetCompoundId": "CHEBI:17115", "label": "enzyme*2", "count": 2, "edgeIds": ["EDGE-GPP-PIN-A", "EDGE-GPP-PIN-B"]}
]


def graph_payload() -> dict[str, Any]:
    grouped_ids = {edge_id for group in EDGE_GROUPS for edge_id in group["edgeIds"]}
    return {"nodes": COMPOUNDS, "edges": [edge for edge in EDGES if edge["edgeId"] not in grouped_ids], "edgeGroups": EDGE_GROUPS}


def compound(compound_id: str) -> dict[str, Any] | None:
    return next((item for item in COMPOUNDS if item["compoundId"] == compound_id), None)


def edge_group(group_id: str) -> list[dict[str, Any]]:
    return [edge for edge in EDGES if edge.get("edgeGroupId") == group_id]


def enzyme_detail(enzyme_id: str) -> dict[str, Any] | None:
    matches = [edge for edge in EDGES if edge["enzymeId"] == enzyme_id]
    if not matches:
        return None
    card = matches[0]["card"]
    reactions = []
    for edge in matches:
        reactions.append({
            "reactionId": edge["reactionId"],
            "rheaId": edge["reactionId"],
            "rheaUrl": f"https://www.rhea-db.org/rhea/{edge['reactionId'].split(':')[-1]}",
            "equation": edge["card"]["reactionEquation"],
            "direction": edge["direction"],
            "ecNumber": edge["card"]["ecNumber"],
            "smiles": None,
            "atomMapImageUrl": None,
            "substrates": [compound(edge["sourceCompoundId"])],
            "products": [compound(edge["targetCompoundId"])],
            "sourceType": edge["sourceType"],
            "reviewStatus": edge["reviewStatus"],
        })
    return {
        "enzymeId": enzyme_id,
        "databaseCode": card["databaseCode"],
        "primaryName": card["primaryName"],
        "secondaryNames": [],
        "uniprotId": card["uniprotId"] or None,
        "uniprotUrl": f"https://www.uniprot.org/uniprotkb/{card['uniprotId']}" if card["uniprotId"] else None,
        "organismName": card["organismName"],
        "sequence": None,
        "length": None,
        "mass": None,
        "gene": None,
        "sequenceLinks": [],
        "reactions": reactions,
        "evidence": [],
        "links": [],
    }


def search_entries(query: str) -> list[dict[str, Any]]:
    normalized = query.strip().lower()
    cards = [edge["card"] for edge in EDGES]
    if not normalized:
        return cards
    return [card for card in cards if normalized in " ".join(str(value) for value in card.values()).lower()]


def pathway(start_id: str, end_id: str) -> list[dict[str, Any]]:
    direct = [edge for edge in EDGES if edge["sourceCompoundId"] == start_id and edge["targetCompoundId"] == end_id]
    if not direct:
        return []
    edge = direct[0]
    return [{
        "pathwayId": f"PATH:{start_id}:{end_id}",
        "summary": f"{start_id} -> {end_id}",
        "compoundIds": [start_id, end_id],
        "edgeIds": [edge["edgeId"]],
        "edgeGroupIds": [edge["edgeGroupId"]] if edge.get("edgeGroupId") else [],
        "stepCount": 1,
        "score": None,
        "graph": {"nodes": [compound(start_id), compound(end_id)], "edges": direct, "edgeGroups": []},
    }]
