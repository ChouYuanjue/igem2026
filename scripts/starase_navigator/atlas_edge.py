from __future__ import annotations

import os
import time
from typing import Any
from urllib.parse import quote

import requests


DEFAULT_EDGE_BACKEND = "http://127.0.0.1:8000/database/api/v1"


class AtlasEdgeClient:
    """Read-only client for the locally deployed Atlas EDGE backend.

    COMPASS never calls the public tunnel for EDGE data. All requests stay on the
    deployment host and go directly to the FastAPI listener.
    """

    def __init__(self, base_url: str | None = None, *, timeout_seconds: float = 4.0) -> None:
        self.base_url = str(
            base_url or os.environ.get("ATLAS_EDGE_BACKEND_URL") or DEFAULT_EDGE_BACKEND
        ).rstrip("/")
        self.timeout_seconds = float(timeout_seconds)
        self.session = requests.Session()
        self._detail_cache: dict[str, tuple[float, dict[str, Any]]] = {}
        self._cache_ttl_seconds = 60.0

    def _cached_detail(self, key: str) -> dict[str, Any] | None:
        row = self._detail_cache.get(key)
        if row is None:
            return None
        cached_at, payload = row
        if time.monotonic() - cached_at > self._cache_ttl_seconds:
            self._detail_cache.pop(key, None)
            return None
        return payload

    def _store_detail(self, key: str, payload: dict[str, Any]) -> dict[str, Any]:
        self._detail_cache[key] = (time.monotonic(), payload)
        return payload

    def _get(self, path: str, *, params: dict[str, Any] | None = None) -> dict[str, Any]:
        response = self.session.get(
            f"{self.base_url}/{path.lstrip('/')}",
            params=params,
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, dict) or payload.get("success") is False:
            raise RuntimeError(f"Atlas EDGE request failed for {path}")
        data = payload.get("data")
        return data if isinstance(data, dict) else {"items": data if isinstance(data, list) else []}

    def reaction_proteins(self, reaction_id: str, *, limit: int = 2000) -> list[dict[str, Any]]:
        reaction_id = str(reaction_id or "").strip().upper()
        if not reaction_id:
            return []
        data = self._get(
            "search/entries",
            params={
                "q": reaction_id,
                "input_type": "rhea_id",
                "page": 1,
                "page_size": max(1, min(int(limit), 2000)),
            },
        )
        items = []
        for row in data.get("items") or []:
            if not isinstance(row, dict):
                continue
            if str(row.get("reactionId") or "").strip().upper() != reaction_id:
                continue
            accession = str(row.get("uniprotId") or "").strip().upper()
            if not accession:
                continue
            items.append(
                {
                    "protein_id": accession,
                    "enzyme_id": str(row.get("enzymeId") or "").strip(),
                    "name": str(row.get("primaryName") or "").strip() or accession,
                    "organism": str(row.get("organismName") or "").strip() or None,
                    "source_type": str(row.get("sourceType") or "").strip() or None,
                    "review_status": str(row.get("reviewStatus") or "").strip() or None,
                    "reaction_id": reaction_id,
                }
            )
        return items

    def protein_detail(self, accession: str) -> dict[str, Any] | None:
        accession = str(accession or "").strip().upper()
        if not accession:
            return None
        cache_key = f"protein:{accession}"
        cached = self._cached_detail(cache_key)
        if cached is not None:
            return cached
        data = self._get(
            "search/entries",
            params={
                "q": accession,
                "input_type": "uniprot_id",
                "page": 1,
                "page_size": 8,
            },
        )
        enzyme_id = ""
        for row in data.get("items") or []:
            if not isinstance(row, dict):
                continue
            if str(row.get("uniprotId") or "").strip().upper() == accession:
                enzyme_id = str(row.get("enzymeId") or "").strip()
                if enzyme_id:
                    break
        if not enzyme_id:
            return None
        detail = self._get(f"enzymes/{quote(enzyme_id, safe='')}")
        if str(detail.get("uniprotId") or "").strip().upper() != accession:
            return None
        return self._store_detail(cache_key, detail)

    def protein_reactions(self, accession: str) -> list[dict[str, Any]]:
        detail = self.protein_detail(accession)
        if not detail:
            return []
        source_type = str(detail.get("sourceType") or "").strip() or None
        review_status = str(detail.get("reviewStatus") or "").strip() or None
        enzyme_id = str(detail.get("enzymeId") or "").strip()
        rows: list[dict[str, Any]] = []
        seen: set[str] = set()
        for reaction in detail.get("reactions") or []:
            if not isinstance(reaction, dict):
                continue
            reaction_id = str(reaction.get("rheaId") or reaction.get("reactionId") or "").strip().upper()
            if not reaction_id or reaction_id in seen:
                continue
            seen.add(reaction_id)
            rows.append(
                {
                    "reaction_id": reaction_id,
                    "enzyme_id": enzyme_id,
                    "protein_id": str(detail.get("uniprotId") or accession).strip().upper(),
                    "protein_name": str(detail.get("primaryName") or accession).strip(),
                    "organism": str(detail.get("organismName") or "").strip() or None,
                    "source_type": source_type,
                    "review_status": review_status,
                    "equation": str(reaction.get("equation") or "").strip() or None,
                    "reaction_smiles": str(reaction.get("smiles") or "").strip() or None,
                }
            )
        return rows

    def protein_literature(self, accession: str) -> list[dict[str, Any]]:
        detail = self.protein_detail(accession)
        if not detail:
            return []
        items: list[dict[str, Any]] = []
        seen: set[str] = set()
        for row in detail.get("evidence") or []:
            if not isinstance(row, dict):
                continue
            pmid = str(row.get("pubmedId") or "").strip()
            doi = str(row.get("doi") or "").strip()
            title = str(row.get("title") or "").strip()
            authors = str(row.get("authors") or "").strip()
            if not (pmid or doi or title):
                continue
            key = (
                f"pmid:{pmid}"
                if pmid
                else f"doi:{doi.casefold()}"
                if doi
                else f"meta:{title.casefold()}:{authors.casefold()}"
            )
            if key in seen:
                continue
            seen.add(key)
            url = str(row.get("url") or "").strip()
            if not url and pmid:
                url = f"https://pubmed.ncbi.nlm.nih.gov/{quote(pmid, safe='')}/"
            elif not url and doi:
                url = f"https://doi.org/{quote(doi, safe='/()')}"
            reference_type = str(row.get("referenceType") or "").strip()
            positions = str(row.get("positions") or "").strip()
            items.append(
                {
                    "id": f"MED:{pmid}" if pmid else f"DOI:{doi}" if doi else key,
                    "pmid": pmid or None,
                    "doi": doi or None,
                    "title": title or str(row.get("sourceDescription") or "Atlas EDGE evidence").strip(),
                    "authors": authors or None,
                    "journal": str(row.get("journal") or "").strip() or None,
                    "year": row.get("publicationYear"),
                    "url": url or None,
                    "publication_types": [reference_type] if reference_type else [],
                    "annotation_context": [positions] if positions else [],
                    "review_status": str(row.get("reviewStatus") or "").strip() or None,
                    "source": "Atlas EDGE",
                    "provider": "Atlas EDGE",
                    "content_basis": "database_linked_reference_metadata",
                }
            )
        return items

    def reaction_preview(self, reaction_id: str) -> dict[str, Any] | None:
        reaction_id = str(reaction_id or "").strip().upper()
        if not reaction_id:
            return None
        cache_key = f"reaction:{reaction_id}"
        detail = self._cached_detail(cache_key)
        try:
            if detail is None:
                detail = self._store_detail(cache_key, self._get(f"reactions/{quote(reaction_id, safe='')}"))
        except (requests.RequestException, RuntimeError):
            return None
        resolved = str(detail.get("rheaId") or detail.get("reactionId") or "").strip().upper()
        if resolved != reaction_id:
            return None

        def compact_compounds(values: Any) -> list[dict[str, Any]]:
            compact: list[dict[str, Any]] = []
            for row in values or []:
                if not isinstance(row, dict):
                    continue
                compound_id = str(row.get("chebiId") or row.get("compoundId") or "").strip()
                if not compound_id:
                    continue
                compact.append(
                    {
                        "compound_id": compound_id,
                        "name": str(row.get("name") or compound_id).strip(),
                        "formula": str(row.get("formula") or "").strip() or None,
                        "average_mass": row.get("averageMass"),
                        "inchi_key": str(row.get("inchiKey") or "").strip() or None,
                    }
                )
            return compact

        return {
            "reaction_id": reaction_id,
            "equation": str(detail.get("equation") or "").strip() or None,
            "direction": str(detail.get("direction") or "").strip() or None,
            "ec_number": str(detail.get("ecNumber") or "").strip() or None,
            "source_type": str(detail.get("sourceType") or "").strip() or None,
            "review_status": str(detail.get("reviewStatus") or "").strip() or None,
            "substrates": compact_compounds(detail.get("substrates")),
            "products": compact_compounds(detail.get("products")),
        }

    def compound_detail(self, compound_id: str) -> dict[str, Any] | None:
        compound_id = str(compound_id or "").strip().upper()
        if not compound_id:
            return None
        cache_key = f"compound:{compound_id}"
        cached = self._cached_detail(cache_key)
        if cached is not None:
            return cached
        try:
            detail = self._get(f"compounds/{quote(compound_id, safe='')}/card")
        except (requests.RequestException, RuntimeError):
            return None
        resolved = str(detail.get("chebiId") or detail.get("compoundId") or "").strip().upper()
        if resolved != compound_id:
            return None
        return self._store_detail(cache_key, detail)

    def compound_structure(self, compound_id: str) -> tuple[bytes, str] | None:
        compound_id = str(compound_id or "").strip().upper()
        if not compound_id:
            return None
        response = self.session.get(
            f"{self.base_url}/assets/compounds/{quote(compound_id, safe='')}/structure.svg",
            timeout=self.timeout_seconds,
        )
        if response.status_code == 404:
            return None
        response.raise_for_status()
        content_type = str(response.headers.get("content-type") or "image/svg+xml").split(";", 1)[0]
        if not content_type.startswith("image/") or not response.content:
            return None
        return response.content, content_type
