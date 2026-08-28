from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any
from urllib.parse import quote

import requests

from scripts.database_bridge.model_catalog import ModelDataCatalog

UNIPROT_SEARCH_URL = "https://rest.uniprot.org/uniprotkb/search"
UNIPROT_ENTRY_URL = "https://rest.uniprot.org/uniprotkb/{accession}.json"
UNIPROT_WEB_BASE = "https://www.uniprot.org/uniprotkb/"
ACCESSION_RE = re.compile(r"^[A-Z0-9]{6}(?:[A-Z0-9]{4})?$", re.IGNORECASE)


@dataclass(frozen=True)
class ProteinCandidate:
    identifier: str
    accession: str | None
    name: str
    organism: str | None
    gene_names: list[str]
    reviewed: bool | None
    length: int | None
    source: str
    model_ready: bool
    local_id: str | None
    score: float
    url: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.identifier,
            "accession": self.accession,
            "name": self.name,
            "organism": self.organism,
            "gene_names": self.gene_names,
            "reviewed": self.reviewed,
            "length": self.length,
            "source": self.source,
            "model_ready": self.model_ready,
            "local_id": self.local_id,
            "score": round(float(self.score), 4),
            "url": self.url,
        }


class UniProtClient:
    def __init__(self, *, user_agent: str) -> None:
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": user_agent})

    @staticmethod
    def _quote(value: str) -> str:
        return f'"{str(value).replace(chr(34), "").strip()}"'

    def search(
        self,
        *,
        protein_terms: list[str],
        organism_terms: list[str],
        gene_terms: list[str],
        accession_terms: list[str],
        limit: int = 8,
        reviewed_only: bool = False,
    ) -> list[dict[str, Any]]:
        clauses: list[str] = []
        if reviewed_only:
            clauses.append("reviewed:true")
        for accession in accession_terms[:2]:
            accession = accession.strip()
            if ACCESSION_RE.fullmatch(accession):
                clauses.append(f"accession:{accession.upper()}")
        text_terms = [value for value in protein_terms[:3] if str(value).strip()]
        gene_terms = [value for value in gene_terms[:2] if str(value).strip()]
        organism_terms = [value for value in organism_terms[:2] if str(value).strip()]
        if text_terms:
            clauses.extend(self._quote(value) for value in text_terms)
        if gene_terms:
            clauses.extend(self._quote(value) for value in gene_terms)
        if organism_terms:
            clauses.extend(self._quote(value) for value in organism_terms)
        if not clauses:
            return []
        query = " AND ".join(clauses)
        response = self.session.get(
            UNIPROT_SEARCH_URL,
            params={
                "query": query,
                "format": "json",
                "fields": "accession,id,protein_name,organism_name,gene_names,reviewed,length",
                "size": max(1, min(int(limit), 50)),
            },
            timeout=25,
        )
        response.raise_for_status()
        payload = response.json()
        results: list[dict[str, Any]] = []
        for row in payload.get("results", []) or []:
            accession = str(row.get("primaryAccession") or "").strip()
            if not accession:
                continue
            protein_description = row.get("proteinDescription") or {}
            recommended = protein_description.get("recommendedName") or {}
            full_name = ((recommended.get("fullName") or {}).get("value") or "").strip()
            if not full_name:
                submitted = protein_description.get("submissionNames") or []
                if submitted:
                    full_name = str(((submitted[0].get("fullName") or {}).get("value") or "")).strip()
            organism = str((row.get("organism") or {}).get("scientificName") or "").strip() or None
            genes: list[str] = []
            for gene in row.get("genes", []) or []:
                name = str(((gene.get("geneName") or {}).get("value") or "")).strip()
                if name and name not in genes:
                    genes.append(name)
            entry_type = str(row.get("entryType") or "")
            entry_type_lower = entry_type.lower()
            reviewed = False if "unreviewed" in entry_type_lower else True if "reviewed" in entry_type_lower else None
            sequence = row.get("sequence") or {}
            length = sequence.get("length")
            try:
                length = int(length) if length is not None else None
            except (TypeError, ValueError):
                length = None
            results.append({
                "accession": accession,
                "name": full_name or str(row.get("uniProtkbId") or accession),
                "organism": organism,
                "gene_names": genes,
                "reviewed": reviewed,
                "length": length,
            })
        return results

    def exact(self, accession: str) -> dict[str, Any]:
        accession = str(accession or "").strip().upper()
        response = self.session.get(UNIPROT_ENTRY_URL.format(accession=quote(accession, safe="")), timeout=25)
        response.raise_for_status()
        row = response.json()
        protein_description = row.get("proteinDescription") or {}
        recommended = protein_description.get("recommendedName") or {}
        full_name = str(((recommended.get("fullName") or {}).get("value") or "")).strip()
        if not full_name:
            submitted = protein_description.get("submissionNames") or []
            if submitted:
                full_name = str(((submitted[0].get("fullName") or {}).get("value") or "")).strip()
        organism = str((row.get("organism") or {}).get("scientificName") or "").strip() or None
        sequence_payload = row.get("sequence") or {}
        sequence = str(sequence_payload.get("value") or "").strip().upper()
        genes: list[str] = []
        for gene in row.get("genes", []) or []:
            name = str(((gene.get("geneName") or {}).get("value") or "")).strip()
            if name and name not in genes:
                genes.append(name)
        return {
            "accession": str(row.get("primaryAccession") or accession).strip(),
            "name": full_name or accession,
            "organism": organism,
            "gene_names": genes,
            "sequence": sequence,
            "length": len(sequence) if sequence else None,
        }


class ProteinResolver:
    def __init__(self, catalog: ModelDataCatalog, *, user_agent: str) -> None:
        self.catalog = catalog
        self.uniprot = UniProtClient(user_agent=user_agent)
        self._local_aliases: dict[str, str] = {}
        for row in catalog.proteins:
            canonical = str(row.get("id") or "").strip()
            for value in [canonical, row.get("uniprot_id"), row.get("genbank_id")]:
                alias = str(value or "").strip()
                if alias:
                    self._local_aliases.setdefault(alias.casefold(), canonical)

    def canonical_local_id(self, identifier: str) -> str | None:
        return self._local_aliases.get(str(identifier or "").strip().casefold())

    def local_record(self, identifier: str) -> dict[str, Any] | None:
        canonical = self.canonical_local_id(identifier)
        return self.catalog.protein_by_id.get(canonical) if canonical else None

    def _local_score(
        self,
        row: dict[str, Any],
        *,
        protein_terms: list[str],
        organism_terms: list[str],
        gene_terms: list[str],
        accession_terms: list[str],
    ) -> float:
        identifier = str(row.get("id") or "")
        accession = str(row.get("uniprot_id") or "")
        name = str(row.get("name") or "")
        organism = str(row.get("species") or "")
        blob = " ".join([identifier, accession, name, organism, str(row.get("genbank_id") or "")]).casefold()
        score = 0.0
        for term in accession_terms:
            term = term.strip().casefold()
            if term and term in {identifier.casefold(), accession.casefold()}:
                score += 16.0
        for term in protein_terms:
            term = term.strip().casefold()
            if not term:
                continue
            if term == name.casefold():
                score += 10.0
            elif term in name.casefold():
                score += 6.0
            elif term in blob:
                score += 2.5
        for term in organism_terms:
            term = term.strip().casefold()
            if not term:
                continue
            if term == organism.casefold():
                score += 7.0
            elif term in organism.casefold():
                score += 4.0
        for term in gene_terms:
            term = term.strip().casefold()
            if term and term in blob:
                score += 2.0
        # `seen` is only a tie-breaker after a real textual/accession match.
        # Applying it to a zero-score row makes every seen local protein eligible,
        # which can outrank an exact external UniProt accession after the local-model
        # readiness bonus is added by `search()`.
        if score > 0 and row.get("seen"):
            score += 0.08
        return score

    def search(
        self,
        *,
        protein_terms: list[str],
        organism_terms: list[str] | None = None,
        gene_terms: list[str] | None = None,
        accession_terms: list[str] | None = None,
        limit: int = 8,
        include_uniprot: bool = True,
    ) -> list[ProteinCandidate]:
        organism_terms = [str(x).strip() for x in (organism_terms or []) if str(x).strip()]
        gene_terms = [str(x).strip() for x in (gene_terms or []) if str(x).strip()]
        accession_terms = [str(x).strip() for x in (accession_terms or []) if str(x).strip()]
        protein_terms = [str(x).strip() for x in (protein_terms or []) if str(x).strip()]
        candidates: dict[str, ProteinCandidate] = {}
        for row in self.catalog.proteins:
            score = self._local_score(
                row,
                protein_terms=protein_terms,
                organism_terms=organism_terms,
                gene_terms=gene_terms,
                accession_terms=accession_terms,
            )
            if score <= 0:
                continue
            local_id = str(row["id"])
            accession = str(row.get("uniprot_id") or "").strip() or (local_id if ACCESSION_RE.fullmatch(local_id) else None)
            key = (accession or local_id).casefold()
            candidates[key] = ProteinCandidate(
                identifier=local_id,
                accession=accession,
                name=str(row.get("name") or local_id),
                organism=str(row.get("species") or "").strip() or None,
                gene_names=[],
                reviewed=None,
                length=row.get("sequence_length"),
                source="model_catalog",
                model_ready=True,
                local_id=local_id,
                score=score + 25.0,
                url=f"{UNIPROT_WEB_BASE}{quote(accession or local_id, safe='')}",
            )

        if include_uniprot:
            try:
                remote = self.uniprot.search(
                    protein_terms=protein_terms,
                    organism_terms=organism_terms,
                    gene_terms=gene_terms,
                    accession_terms=accession_terms,
                    limit=max(4, min(limit, 10)),
                )
            except requests.RequestException:
                remote = []
            for index, row in enumerate(remote):
                accession = str(row["accession"])
                local_id = self.canonical_local_id(accession)
                key = accession.casefold()
                # Preserve a strong local match, but fill missing display metadata from UniProt.
                if key in candidates:
                    continue
                score = 12.0 - min(index, 8) * 0.45
                name_blob = str(row.get("name") or "").casefold()
                organism_blob = str(row.get("organism") or "").casefold()
                for term in protein_terms:
                    if term.casefold() in name_blob:
                        score += 3.5
                for term in organism_terms:
                    if term.casefold() in organism_blob:
                        score += 3.0
                if row.get("reviewed"):
                    score += 0.35
                candidates[key] = ProteinCandidate(
                    identifier=local_id or accession,
                    accession=accession,
                    name=str(row.get("name") or accession),
                    organism=row.get("organism"),
                    gene_names=list(row.get("gene_names") or []),
                    reviewed=row.get("reviewed"),
                    length=row.get("length"),
                    source="uniprot",
                    model_ready=bool(local_id),
                    local_id=local_id,
                    score=score,
                    url=f"{UNIPROT_WEB_BASE}{quote(accession, safe='')}",
                )

        ranked = sorted(
            candidates.values(),
            key=lambda row: (-row.score, not row.model_ready, row.name.casefold(), row.identifier.casefold()),
        )
        return ranked[: max(1, min(int(limit), 12))]

    def search_class_members(
        self,
        *,
        protein_terms: list[str],
        organism_terms: list[str] | None = None,
        gene_terms: list[str] | None = None,
        limit: int = 40,
    ) -> list[ProteinCandidate]:
        """Build an auditable search-derived cohort for a functional enzyme class.

        Unlike ``search()``, this method does not promote one model-ready local hit
        as a representative of the whole class. It unions several concise UniProt
        searches plus matching local records and returns the cohort itself.
        """
        organism_terms = [str(x).strip() for x in (organism_terms or []) if str(x).strip()]
        gene_terms = [str(x).strip() for x in (gene_terms or []) if str(x).strip()]
        terms = [str(x).strip() for x in protein_terms if str(x).strip()][:5]
        max_results = max(1, min(int(limit), 50))
        candidates: dict[str, ProteinCandidate] = {}

        for row in self.catalog.proteins:
            score = self._local_score(
                row,
                protein_terms=terms,
                organism_terms=organism_terms,
                gene_terms=gene_terms,
                accession_terms=[],
            )
            if score <= 0:
                continue
            local_id = str(row["id"])
            accession = str(row.get("uniprot_id") or "").strip() or (local_id if ACCESSION_RE.fullmatch(local_id) else None)
            key = (accession or local_id).casefold()
            candidates[key] = ProteinCandidate(
                identifier=local_id,
                accession=accession,
                name=str(row.get("name") or local_id),
                organism=str(row.get("species") or "").strip() or None,
                gene_names=[],
                reviewed=None,
                length=row.get("sequence_length"),
                source="model_catalog",
                model_ready=True,
                local_id=local_id,
                score=score,
                url=f"{UNIPROT_WEB_BASE}{quote(accession or local_id, safe='')}",
            )

        query_groups = [[term] for term in terms]
        if len(terms) > 1:
            query_groups.insert(0, terms[:3])
        for reviewed_only in (True, False):
            if not reviewed_only and len(candidates) >= max_results:
                break
            for query_index, query_terms in enumerate(query_groups[:6]):
                try:
                    remote = self.uniprot.search(
                        protein_terms=query_terms,
                        organism_terms=organism_terms,
                        gene_terms=gene_terms,
                        accession_terms=[],
                        limit=max_results,
                        reviewed_only=reviewed_only,
                    )
                except requests.RequestException:
                    continue
                for index, row in enumerate(remote):
                    accession = str(row.get("accession") or "").strip()
                    if not accession:
                        continue
                    local_id = self.canonical_local_id(accession)
                    key = accession.casefold()
                    score = 42.0 if reviewed_only else 24.0
                    score -= query_index * 1.5 + min(index, 30) * 0.08
                    existing = candidates.get(key)
                    candidate = ProteinCandidate(
                        identifier=local_id or accession,
                        accession=accession,
                        name=str(row.get("name") or accession),
                        organism=row.get("organism"),
                        gene_names=list(row.get("gene_names") or []),
                        reviewed=row.get("reviewed"),
                        length=row.get("length"),
                        source="uniprot",
                        model_ready=bool(local_id),
                        local_id=local_id,
                        score=score,
                        url=f"{UNIPROT_WEB_BASE}{quote(accession, safe='')}",
                    )
                    if existing is None or candidate.score > existing.score:
                        candidates[key] = candidate

        ranked = sorted(
            candidates.values(),
            key=lambda row: (-row.score, row.name.casefold(), row.identifier.casefold()),
        )
        return ranked[:max_results]

    def exact_or_search(self, text: str, *, limit: int = 8) -> list[ProteinCandidate]:
        value = str(text or "").strip()
        local_id = self.canonical_local_id(value)
        if local_id:
            row = self.catalog.protein_by_id[local_id]
            accession = str(row.get("uniprot_id") or "").strip() or (local_id if ACCESSION_RE.fullmatch(local_id) else None)
            return [ProteinCandidate(
                identifier=local_id,
                accession=accession,
                name=str(row.get("name") or local_id),
                organism=str(row.get("species") or "").strip() or None,
                gene_names=[],
                reviewed=None,
                length=row.get("sequence_length"),
                source="model_catalog",
                model_ready=True,
                local_id=local_id,
                score=100.0,
                url=f"{UNIPROT_WEB_BASE}{quote(accession or local_id, safe='')}",
            )]
        if ACCESSION_RE.fullmatch(value):
            try:
                exact = self.uniprot.exact(value)
            except requests.RequestException:
                return []
            accession = str(exact["accession"])
            local_id = self.canonical_local_id(accession)
            return [ProteinCandidate(
                identifier=local_id or accession,
                accession=accession,
                name=str(exact.get("name") or accession),
                organism=exact.get("organism"),
                gene_names=list(exact.get("gene_names") or []),
                reviewed=None,
                length=exact.get("length"),
                source="uniprot",
                model_ready=bool(local_id),
                local_id=local_id,
                score=90.0,
                url=f"{UNIPROT_WEB_BASE}{quote(accession, safe='')}",
            )]
        return []

    def sequence_for(self, identifier: str) -> tuple[str, str]:
        local_id = self.canonical_local_id(identifier)
        if local_id:
            # Current model catalog may omit current sequences in protein_registry, so use
            # UniProt when an accession is available; external registered rows are also
            # available from UniProt by accession in the normal user-facing flow.
            row = self.catalog.protein_by_id[local_id]
            accession = str(row.get("uniprot_id") or "").strip() or (local_id if ACCESSION_RE.fullmatch(local_id) else "")
            if not accession:
                raise KeyError(f"No resolvable UniProt accession for {local_id}")
            exact = self.uniprot.exact(accession)
            return local_id, str(exact.get("sequence") or "")
        exact = self.uniprot.exact(identifier)
        return str(exact["accession"]), str(exact.get("sequence") or "")


def compact_query_terms(payload: dict[str, Any]) -> dict[str, list[str]]:
    def clean(key: str, limit: int) -> list[str]:
        raw = payload.get(key) or []
        if not isinstance(raw, list):
            return []
        result: list[str] = []
        seen: set[str] = set()
        for item in raw:
            value = str(item or "").strip()
            marker = value.casefold()
            if value and marker not in seen:
                seen.add(marker)
                result.append(value[:180])
        return result[:limit]

    return {
        "protein_terms": clean("protein_terms", 6),
        "organism_terms": clean("organism_terms", 4),
        "gene_terms": clean("gene_terms", 4),
        "accession_terms": clean("accession_terms", 4),
    }
