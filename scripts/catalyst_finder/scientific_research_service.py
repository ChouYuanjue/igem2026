from __future__ import annotations

import concurrent.futures
import html
import re
import time
import xml.etree.ElementTree as ET
from typing import Any
from urllib.parse import quote

import requests

from scripts.catalyst_finder.formatting import probable_uniprot
from scripts.catalyst_finder.model_validation_snapshot import AUDIT_CONTEXT, PROJECT_ALIGNED_EXTERNAL_RELATION_AUDIT


class ScientificResearchService:
    """Build an on-demand research workspace around a verified protein or reaction.

    The service deliberately does not mirror whole external databases locally. It uses
    official online APIs for current annotations/literature and keeps the neural model
    as a parallel research lens over the same verified entity.
    """

    def __init__(
        self,
        *,
        evidence: Any,
        evidence_queries: Any,
        proteins: Any,
        rhea: Any,
        route_designer: Any,
        model_gateway: Any,
        catalog: Any,
        user_agent: str,
        deepseek: Any | None = None,
        retrieval_service: Any | None = None,
    ) -> None:
        self.evidence = evidence
        self.evidence_queries = evidence_queries
        self.proteins = proteins
        self.rhea = rhea
        self.route_designer = route_designer
        self.model_gateway = model_gateway
        self.retrieval_service = retrieval_service
        self.catalog = catalog
        self.deepseek = deepseek
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": user_agent})
        self.timeout = 14

    @staticmethod
    def _name_from_uniprot(row: dict[str, Any]) -> str:
        desc = row.get("proteinDescription") or {}
        rec = desc.get("recommendedName") or {}
        name = str(((rec.get("fullName") or {}).get("value") or "")).strip()
        if name:
            return name
        submitted = desc.get("submissionNames") or []
        if submitted:
            return str((((submitted[0] or {}).get("fullName") or {}).get("value") or "")).strip()
        return str(row.get("uniProtkbId") or row.get("primaryAccession") or "")

    @staticmethod
    def _comment_text(comment: dict[str, Any]) -> list[str]:
        texts: list[str] = []
        for part in comment.get("texts") or []:
            value = str((part or {}).get("value") or "").strip()
            if value and value not in texts:
                texts.append(value)
        return texts

    @staticmethod
    def _plain_text(value: Any) -> str:
        text = html.unescape(str(value or ""))
        text = re.sub(r"<[^>]+>", " ", text)
        return re.sub(r"\s+", " ", text).strip()

    @staticmethod
    def _xref_url(database: str, identifier: str) -> str | None:
        db = str(database or "").strip()
        xid = str(identifier or "").strip()
        if not db or not xid:
            return None
        patterns = {
            "PDB": "https://www.rcsb.org/structure/{id}",
            "AlphaFoldDB": "https://alphafold.ebi.ac.uk/entry/{id}",
            "InterPro": "https://www.ebi.ac.uk/interpro/entry/InterPro/{id}/",
            "Pfam": "https://www.ebi.ac.uk/interpro/entry/pfam/{id}/",
            "BRENDA": "https://www.brenda-enzymes.org/enzyme.php?ecno={id}",
            "KEGG": "https://www.genome.jp/entry/{id}",
            "Reactome": "https://reactome.org/content/detail/{id}",
            "BioCyc": "https://biocyc.org/gene?orgid=META&id={id}",
            "GeneID": "https://www.ncbi.nlm.nih.gov/gene/{id}",
            "ChEMBL": "https://www.ebi.ac.uk/chembl/explore/target/{id}",
            "DrugBank": "https://go.drugbank.com/unearth/q?searcher=targets&query={id}",
            "GO": "https://www.ebi.ac.uk/QuickGO/term/{id}",
        }
        pattern = patterns.get(db)
        return pattern.format(id=quote(xid, safe="")) if pattern else None

    @staticmethod
    def _citation_refs(row: dict[str, Any]) -> tuple[list[str], dict[str, dict[str, Any]]]:
        pmids: list[str] = []
        metadata: dict[str, dict[str, Any]] = {}
        for ref in row.get("references") or []:
            if not isinstance(ref, dict):
                continue
            citation = ref.get("citation") or {}
            pmid = ""
            doi = ""
            for cross in citation.get("citationCrossReferences") or []:
                if not isinstance(cross, dict):
                    continue
                db = str(cross.get("database") or "")
                if db == "PubMed":
                    pmid = str(cross.get("id") or "").strip()
                elif db == "DOI":
                    doi = str(cross.get("id") or "").strip()
            if not pmid:
                continue
            if pmid not in pmids:
                pmids.append(pmid)
            metadata[pmid] = {
                "title": str(citation.get("title") or ""),
                "authors": ", ".join(str(x) for x in citation.get("authors") or []),
                "journal": str(citation.get("journal") or ""),
                "year": str(citation.get("publicationDate") or ""),
                "doi": doi,
                "annotation_context": [str(x) for x in ref.get("referencePositions") or [] if str(x).strip()],
            }
        return pmids, metadata

    def _uniprot_panel(self, accession: str) -> dict[str, Any]:
        started = time.time()
        url = f"https://rest.uniprot.org/uniprotkb/{quote(accession, safe='')}.json"
        response = self.session.get(url, timeout=self.timeout)
        response.raise_for_status()
        row = response.json()
        name = self._name_from_uniprot(row)
        organism = str((row.get("organism") or {}).get("scientificName") or "").strip()
        genes: list[str] = []
        for gene in row.get("genes") or []:
            value = str((((gene or {}).get("geneName") or {}).get("value") or "")).strip()
            if value and value not in genes:
                genes.append(value)

        facts: list[dict[str, Any]] = [
            {"label": "Protein", "value": name},
            {"label": "Organism", "value": organism},
            {"label": "Gene", "value": ", ".join(genes)},
            {"label": "Entry", "value": str(row.get("entryType") or "")},
            {"label": "Annotation score", "value": row.get("annotationScore")},
            {"label": "Protein existence", "value": str(row.get("proteinExistence") or "")},
        ]
        comments: dict[str, list[str]] = {}
        catalytic: list[dict[str, Any]] = []
        cofactors: list[str] = []
        for comment in row.get("comments") or []:
            if not isinstance(comment, dict):
                continue
            ctype = str(comment.get("commentType") or "").strip()
            if ctype == "CATALYTIC ACTIVITY":
                reaction = comment.get("reaction") or {}
                rhea_ids = [
                    str(x.get("id") or "")
                    for x in reaction.get("reactionCrossReferences") or []
                    if isinstance(x, dict) and str(x.get("database") or "") == "Rhea" and str(x.get("id") or "")
                ]
                catalytic.append({
                    "reaction": str(reaction.get("name") or ""),
                    "ec_number": str(reaction.get("ecNumber") or ""),
                    "rhea_ids": rhea_ids,
                })
            elif ctype == "COFACTOR":
                for item in comment.get("cofactors") or []:
                    value = str((item or {}).get("name") or "").strip()
                    if value and value not in cofactors:
                        cofactors.append(value)
            else:
                values = self._comment_text(comment)
                if values:
                    comments.setdefault(ctype, []).extend(v for v in values if v not in comments.get(ctype, []))

        wanted_dbs = {
            "PDB", "AlphaFoldDB", "InterPro", "Pfam", "BRENDA", "KEGG", "Reactome",
            "BioCyc", "GeneID", "ChEMBL", "DrugBank", "GO",
        }
        xrefs: dict[str, list[str]] = {}
        xref_items: list[dict[str, Any]] = []
        for xref in row.get("uniProtKBCrossReferences") or []:
            if not isinstance(xref, dict):
                continue
            db = str(xref.get("database") or "")
            xid = str(xref.get("id") or "")
            if db in wanted_dbs and xid:
                bucket = xrefs.setdefault(db, [])
                if xid not in bucket:
                    bucket.append(xid)
                    xref_items.append({
                        "database": db,
                        "id": xid,
                        "url": self._xref_url(db, xid),
                    })

        pmids, curated_reference_metadata = self._citation_refs(row)

        return {
            "id": "uniprot",
            "title": "UniProtKB",
            "status": "ok",
            "url": f"https://www.uniprot.org/uniprotkb/{quote(accession, safe='')}",
            "facts": [item for item in facts if item.get("value") not in {None, ""}],
            "catalytic_activities": catalytic,
            "cofactors": cofactors,
            "annotations": {key: value for key, value in comments.items() if value},
            "cross_references": xrefs,
            "cross_reference_items": xref_items,
            "publication_ids": pmids,
            "curated_reference_metadata": curated_reference_metadata,
            "record": {"accession": accession, "name": name, "organism": organism, "genes": genes},
            "latency_ms": round((time.time() - started) * 1000, 1),
        }

    def protein_detail(self, accession: str) -> dict[str, Any]:
        """Return bounded substantive UniProt evidence for one verified accession.

        This is the detail-level counterpart to the research workspace: callers that
        inspect or compare a specific protein can reason over actual function, catalytic
        activity, cofactors and selected annotations instead of identity metadata alone.
        """
        panel = self._uniprot_panel(str(accession or "").strip())
        return {
            "record": dict(panel.get("record") or {}),
            "facts": list(panel.get("facts") or [])[:12],
            "catalytic_activities": list(panel.get("catalytic_activities") or [])[:12],
            "cofactors": list(panel.get("cofactors") or [])[:12],
            "annotations": {str(key): list(value)[:5] for key, value in (panel.get("annotations") or {}).items() if isinstance(value, list)},
            "cross_references": {str(key): list(value)[:12] for key, value in (panel.get("cross_references") or {}).items() if isinstance(value, list)},
            "url": panel.get("url"),
        }

    def _structure_panel(self, accession: str, *, uniprot_panel: dict[str, Any]) -> dict[str, Any]:
        started = time.time()
        xrefs = uniprot_panel.get("cross_references") if isinstance(uniprot_panel, dict) else {}
        xrefs = xrefs if isinstance(xrefs, dict) else {}
        pdb_ids = list(dict.fromkeys(str(x).strip().upper() for x in xrefs.get("PDB") or [] if str(x).strip()))
        alphafold_ids = list(dict.fromkeys(str(x).strip() for x in xrefs.get("AlphaFoldDB") or [] if str(x).strip()))
        # AlphaFold API is accession-addressable even when a UniProt cross-reference is
        # temporarily absent, so try the verified accession as a bounded fallback.
        if not alphafold_ids and probable_uniprot(accession):
            alphafold_ids = [accession]

        items: list[dict[str, Any]] = [
            {
                "id": pdb_id, "name": pdb_id, "type": "experimental_structure",
                "source": "RCSB PDB", "url": f"https://www.rcsb.org/structure/{quote(pdb_id, safe='')}",
                "detail_status": "identity_only",
            }
            for pdb_id in pdb_ids
        ]

        def fetch_pdb(pdb_id: str) -> dict[str, Any] | None:
            response = self.session.get(f"https://data.rcsb.org/rest/v1/core/entry/{quote(pdb_id, safe='')}", timeout=self.timeout)
            if response.status_code == 404:
                return None
            response.raise_for_status()
            row = response.json()
            info = row.get("rcsb_entry_info") or {}
            resolutions = [float(x) for x in info.get("resolution_combined") or [] if x is not None]
            method = str(info.get("experimental_method") or "").strip()
            if not method:
                methods = [str((x or {}).get("method") or "").strip() for x in row.get("exptl") or []]
                method = ", ".join(x for x in methods if x)
            return {
                "id": pdb_id,
                "name": self._plain_text((row.get("struct") or {}).get("title") or pdb_id),
                "type": "experimental_structure",
                "source": "RCSB PDB",
                "method": method or None,
                "resolution_angstrom": min(resolutions) if resolutions else None,
                "released": str((row.get("rcsb_accession_info") or {}).get("initial_release_date") or "")[:10] or None,
                "url": f"https://www.rcsb.org/structure/{quote(pdb_id, safe='')}",
            }

        def fetch_alphafold(model_id: str) -> dict[str, Any] | None:
            query_id = accession if probable_uniprot(accession) else model_id
            response = self.session.get(f"https://alphafold.ebi.ac.uk/api/prediction/{quote(query_id, safe='')}", timeout=self.timeout)
            if response.status_code == 404:
                return None
            response.raise_for_status()
            payload = response.json()
            rows = payload if isinstance(payload, list) else [payload]
            rows = [row for row in rows if isinstance(row, dict)]
            if not rows:
                return None
            row = max(rows, key=lambda value: int(value.get("latestVersion") or 0))
            entity_id = str(row.get("modelEntityId") or model_id or query_id)
            metric = row.get("globalMetricValue")
            return {
                "id": entity_id,
                "name": f"AlphaFold model for {accession}",
                "type": "predicted_structure",
                "source": "AlphaFold DB",
                "method": str(row.get("toolUsed") or "AlphaFold"),
                "global_plddt": float(metric) if metric is not None else None,
                "version": int(row.get("latestVersion") or 0) or None,
                "created": str(row.get("modelCreatedDate") or "")[:10] or None,
                "url": f"https://alphafold.ebi.ac.uk/entry/{quote(accession, safe='')}",
            }

        # Prefetch rich metadata for the first visible page only. Every PDB identity is
        # still present in ``items`` and therefore browsable through local pagination.
        prefetch_pdb_ids = pdb_ids[:10]
        enriched_by_id: dict[str, dict[str, Any]] = {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(8, max(1, len(prefetch_pdb_ids) + min(1, len(alphafold_ids))))) as pool:
            futures = [(pdb_id, pool.submit(fetch_pdb, pdb_id)) for pdb_id in prefetch_pdb_ids]
            for pdb_id, future in futures:
                try:
                    row = future.result()
                except Exception:
                    row = None
                if isinstance(row, dict):
                    enriched_by_id[pdb_id] = row
            if alphafold_ids:
                try:
                    af_row = pool.submit(fetch_alphafold, alphafold_ids[0]).result()
                except Exception:
                    af_row = None
                if isinstance(af_row, dict):
                    items.append(af_row)
        if enriched_by_id:
            items = [enriched_by_id.get(str(row.get("id") or ""), row) for row in items]
        experimental = len(pdb_ids)
        predicted = sum(str(row.get("type") or "") == "predicted_structure" for row in items)
        return {
            "id": "structures",
            "title": "Structures",
            "status": "ok",
            "url": f"https://www.rcsb.org/search?request={{}}",
            "count": len(items),
            "facts": [
                {"label": "Experimental structures", "value": experimental},
                {"label": "Predicted models", "value": predicted},
            ],
            "items": items,
            "latency_ms": round((time.time() - started) * 1000, 1),
        }

    def _interpro_panel(self, accession: str) -> dict[str, Any]:
        started = time.time()
        base_url = f"https://www.ebi.ac.uk/interpro/api/entry/interpro/protein/uniprot/{quote(accession, safe='')}"
        next_url: str | None = base_url
        first = True
        entries: list[dict[str, Any]] = []
        seen_ids: set[str] = set()
        total_count = 0
        while next_url:
            response = self.session.get(next_url, params={"page_size": 50} if first else None, timeout=self.timeout)
            response.raise_for_status()
            payload = response.json()
            if first:
                total_count = int(payload.get("count") or 0)
            for row in payload.get("results") or []:
                meta = (row or {}).get("metadata") or {}
                acc = str(meta.get("accession") or "").strip()
                if not acc or acc in seen_ids:
                    continue
                seen_ids.add(acc)
                member = meta.get("member_databases") or {}
                members: list[str] = []
                for db, values in member.items():
                    if isinstance(values, dict):
                        members.extend(f"{db}:{key}" for key in values)
                entries.append({
                    "id": acc, "name": str(meta.get("name") or acc),
                    "type": str(meta.get("type") or ""), "member_entries": members,
                    "url": f"https://www.ebi.ac.uk/interpro/entry/InterPro/{acc}/",
                })
            candidate_next = str(payload.get("next") or "").strip()
            if not candidate_next or candidate_next == next_url:
                break
            next_url = candidate_next
            first = False
        return {
            "id": "interpro", "title": "InterPro", "status": "ok",
            "url": f"https://www.ebi.ac.uk/interpro/protein/UniProt/{quote(accession, safe='')}/",
            "count": total_count or len(entries), "items": entries,
            "pagination": {"mode": "local", "page_size": 10, "has_more": False},
            "latency_ms": round((time.time() - started) * 1000, 1),
        }

    def _europe_pmc_item(self, row: dict[str, Any]) -> dict[str, Any] | None:
        if not isinstance(row, dict):
            return None
        source = str(row.get("source") or "")
        article_id = str(row.get("id") or row.get("pmid") or "")
        if not article_id:
            return None
        pub_types = [
            self._plain_text(value)
            for value in ((row.get("pubTypeList") or {}).get("pubType") or [])
            if self._plain_text(value)
        ]
        corrections = []
        for relation in ((row.get("commentCorrectionList") or {}).get("commentCorrection") or []):
            if not isinstance(relation, dict):
                continue
            related_id = str(relation.get("id") or "").strip()
            if not related_id:
                continue
            corrections.append({
                "id": related_id,
                "source": str(relation.get("source") or "MED"),
                "type": self._plain_text(relation.get("type") or ""),
                "reference": self._plain_text(relation.get("reference") or ""),
            })
        return {
            "id": article_id,
            "source": source,
            "provider": "europe_pmc",
            "pmid": str(row.get("pmid") or "") or None,
            "pmcid": str(row.get("pmcid") or "") or None,
            "doi": str(row.get("doi") or "") or None,
            "title": self._plain_text(row.get("title") or article_id),
            "authors": self._plain_text(row.get("authorString") or ""),
            "journal": self._plain_text(row.get("journalTitle") or ((row.get("journalInfo") or {}).get("journal") or {}).get("medlineAbbreviation") or ""),
            "year": str(row.get("pubYear") or ((row.get("journalInfo") or {}).get("yearOfPublication") or "")),
            "abstract": self._plain_text(row.get("abstractText") or ""),
            "cited_by": int(row.get("citedByCount") or 0),
            "open_access": str(row.get("isOpenAccess") or "").upper() == "Y",
            "publication_types": pub_types,
            "corrections": corrections,
            "url": f"https://europepmc.org/article/{quote(source, safe='')}/{quote(article_id, safe='')}",
        }

    def _literature_panel(
        self,
        query: str,
        *,
        limit: int,
        cursor_mark: str = "*",
    ) -> dict[str, Any]:
        started = time.time()
        url = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
        page_size = max(1, min(int(limit or 10), 100))
        params = {"query": query, "format": "json", "resultType": "core", "pageSize": page_size}
        if cursor_mark:
            params["cursorMark"] = str(cursor_mark)
        response = self.session.get(url, params=params, timeout=self.timeout)
        response.raise_for_status()
        payload = response.json()
        items = [
            item
            for row in ((payload.get("resultList") or {}).get("result") or [])
            if (item := self._europe_pmc_item(row)) is not None
        ]
        hit_count = int(payload.get("hitCount") or len(items))
        next_cursor = str(payload.get("nextCursorMark") or "")
        return {
            "id": "literature_europe_pmc",
            "entity_kind": "literature",
            "provider": "europe_pmc",
            "title": "Europe PMC",
            "status": "ok",
            "url": f"https://europepmc.org/search?query={quote(query)}",
            "query": query,
            "count": hit_count,
            "items": items,
            "pagination": {
                "mode": "remote" if hit_count > len(items) else "local",
                "provider": "europe_pmc",
                "page_size": page_size,
                "cursor": str(cursor_mark or "*"),
                "next_cursor": next_cursor,
                "has_more": bool(next_cursor and hit_count > len(items)),
            },
            "latency_ms": round((time.time() - started) * 1000, 1),
        }

    @staticmethod
    def _openalex_abstract(inverted: Any) -> str:
        if not isinstance(inverted, dict):
            return ""
        positioned: list[tuple[int, str]] = []
        for token, positions in inverted.items():
            if not isinstance(positions, list):
                continue
            for position in positions:
                try:
                    positioned.append((int(position), str(token)))
                except (TypeError, ValueError):
                    continue
        positioned.sort(key=lambda item: item[0])
        return " ".join(token for _position, token in positioned).strip()

    def _openalex_item(self, row: dict[str, Any]) -> dict[str, Any] | None:
        if not isinstance(row, dict):
            return None
        raw_id = str(row.get("id") or "").strip()
        work_id = raw_id.rsplit("/", 1)[-1] if raw_id else ""
        if not work_id:
            return None
        ids = row.get("ids") if isinstance(row.get("ids"), dict) else {}
        pmid_url = str(ids.get("pmid") or "")
        pmid_match = re.search(r"(\d{5,})$", pmid_url)
        pmid = pmid_match.group(1) if pmid_match else None
        doi = str(row.get("doi") or ids.get("doi") or "").strip()
        doi = re.sub(r"(?i)^https?://(?:dx\.)?doi\.org/", "", doi).strip() or None
        authors: list[str] = []
        for authorship in row.get("authorships") or []:
            if not isinstance(authorship, dict):
                continue
            author = authorship.get("author") if isinstance(authorship.get("author"), dict) else {}
            name = str(author.get("display_name") or authorship.get("raw_author_name") or "").strip()
            if name and name not in authors:
                authors.append(name)
        primary = row.get("primary_location") if isinstance(row.get("primary_location"), dict) else {}
        source = primary.get("source") if isinstance(primary.get("source"), dict) else {}
        landing = str(primary.get("landing_page_url") or raw_id).strip()
        work_type = str(row.get("type") or "").strip()
        return {
            "id": f"OPENALEX:{work_id}",
            "openalex_id": work_id,
            "source": "OPENALEX",
            "provider": "openalex",
            "pmid": pmid,
            "doi": doi,
            "title": self._plain_text(row.get("display_name") or row.get("title") or work_id),
            "authors": ", ".join(authors),
            "journal": self._plain_text(source.get("display_name") or primary.get("raw_source_name") or ""),
            "year": str(row.get("publication_year") or ""),
            "abstract": self._openalex_abstract(row.get("abstract_inverted_index")),
            "cited_by": int(row.get("cited_by_count") or 0),
            "open_access": bool((row.get("open_access") or {}).get("is_oa")),
            "publication_types": [work_type] if work_type else [],
            "indexed_in": [str(value) for value in row.get("indexed_in") or [] if str(value).strip()],
            "url": landing or raw_id,
        }

    def _openalex_panel(self, query: str, *, page_size: int = 10, cursor: str = "*") -> dict[str, Any]:
        query = str(query or "").strip()
        if not query or len(query) > 1200:
            raise ValueError("OpenAlex literature query is empty or too long")
        size = max(1, min(int(page_size or 10), 25))
        response = self.session.get(
            "https://api.openalex.org/works",
            params={"search": query, "per-page": size, "cursor": cursor or "*"},
            timeout=self.timeout,
        )
        response.raise_for_status()
        payload = response.json()
        meta = payload.get("meta") if isinstance(payload.get("meta"), dict) else {}
        items = [
            item
            for row in payload.get("results") or []
            if (item := self._openalex_item(row)) is not None
        ]
        next_cursor = str(meta.get("next_cursor") or "").strip()
        return {
            "id": "literature_openalex",
            "entity_kind": "literature",
            "provider": "openalex",
            "title": "OpenAlex",
            "status": "ok",
            "url": "https://openalex.org/",
            "query": query,
            "count": int(meta.get("count") or len(items)),
            "items": items,
            "curated_by": "broad_scholarly_search",
            "pagination": {
                "mode": "remote", "provider": "openalex", "page_size": size,
                "cursor": cursor or "*", "next_cursor": next_cursor,
                "has_more": bool(next_cursor),
            },
        }

    def resolve_literature(self, text: str, *, limit: int = 6) -> list[dict[str, Any]]:
        """Resolve PMID/PMCID/DOI/title text to current Europe PMC records."""
        text = str(text or "").strip()
        if not text:
            return []
        med = re.search(r"(?:MED|PMID)\s*[:#]?\s*(\d{5,10})", text, re.I)
        pmc = re.search(r"\b(PMC\d+)\b", text, re.I)
        doi = re.search(r'\b(10\.\d{4,9}/[^\s<>"）)]+)', text, re.I)
        if med:
            query = f"EXT_ID:{med.group(1)}"
        elif pmc:
            query = f"PMCID:{pmc.group(1).upper()}"
        elif doi:
            query = f'DOI:"{doi.group(1).rstrip(".,;:。，；：")}"'
        else:
            query = text
        panel = self._literature_panel(query, limit=max(1, min(int(limit or 6), 20)))
        return [dict(row) for row in panel.get("items") or [] if isinstance(row, dict)][: max(1, min(int(limit or 6), 20))]

    def literature_page(
        self, query: str, *, cursor_mark: str = "*", page_size: int = 10, provider: str = "europe_pmc",
    ) -> dict[str, Any]:
        query = str(query or "").strip()
        if not query or len(query) > 6000:
            raise ValueError("literature query is empty or too long")
        provider_key = str(provider or "europe_pmc").strip().lower()
        if provider_key == "openalex":
            return self._openalex_panel(query, page_size=page_size, cursor=cursor_mark or "*")
        if provider_key != "europe_pmc":
            raise ValueError(f"unsupported literature provider: {provider_key}")
        return self._literature_panel(query, limit=max(1, min(int(page_size or 10), 20)), cursor_mark=cursor_mark or "*")

    def _literature_full_text_sections(self, pmcid: str) -> list[dict[str, str]]:
        pmcid = str(pmcid or "").strip()
        if not pmcid:
            return []
        try:
            response = self.session.get(
                f"https://www.ebi.ac.uk/europepmc/webservices/rest/{quote(pmcid, safe='')}/fullTextXML",
                timeout=self.timeout,
            )
            if response.status_code == 404:
                return []
            response.raise_for_status()
            root = ET.fromstring(response.text)
        except Exception:
            return []
        sections: list[dict[str, str]] = []
        candidates = root.findall(".//body/sec")
        if not candidates:
            candidates = root.findall(".//sec")
        for sec in candidates:
            title_node = sec.find("title")
            title = self._plain_text(" ".join(title_node.itertext()) if title_node is not None else "")
            paragraphs = []
            for paragraph in sec.findall("./p"):
                text = self._plain_text(" ".join(paragraph.itertext()))
                if text:
                    paragraphs.append(text)
                if sum(len(value) for value in paragraphs) >= 2200:
                    break
            text = " ".join(paragraphs).strip()[:2600]
            if text:
                sections.append({"title": title or "Section", "text": text})
            if len(sections) >= 8:
                break
        return sections

    def literature_detail(self, row: dict[str, Any]) -> dict[str, Any]:
        """Enrich one already verified literature row with live, auditable content.

        Bibliographic correction/erratum links are followed generically so the agent can
        distinguish a correction notice from the scientific article it refers to.
        """
        base = dict(row or {})
        discovery_provider = str(base.get("provider") or "").strip().lower()
        pmid = str(base.get("pmid") or "").strip()
        doi = str(base.get("doi") or "").strip()
        raw_id = str(base.get("id") or "").strip()
        article_id = pmid or (raw_id.split(":")[-1] if raw_id and not raw_id.upper().startswith("OPENALEX:") else "")
        lookup_query = f"EXT_ID:{article_id}" if article_id else f'DOI:"{doi}"' if doi else ""
        if lookup_query:
            try:
                exact = self._literature_panel(lookup_query, limit=3)
                match = next((
                    item for item in exact.get("items") or []
                    if (article_id and str(item.get("pmid") or item.get("id") or "") == article_id)
                    or (doi and str(item.get("doi") or "").casefold() == doi.casefold())
                ), None)
                if isinstance(match, dict):
                    merged = dict(base)
                    merged.update({key: value for key, value in match.items() if value not in (None, "", [], {})})
                    base = merged
            except Exception:
                pass
        evidence_providers = []
        for provider_name in [discovery_provider, str(base.get("provider") or "").strip().lower()]:
            if provider_name and provider_name not in evidence_providers:
                evidence_providers.append(provider_name)
        if discovery_provider:
            base["provider"] = discovery_provider
        base["evidence_providers"] = evidence_providers
        pmcid = str(base.get("pmcid") or "").strip()
        full_text_sections = self._literature_full_text_sections(pmcid) if pmcid else []
        if full_text_sections:
            base["full_text_sections"] = full_text_sections
        related_publications = []
        for relation in list(base.get("corrections") or [])[:4]:
            if not isinstance(relation, dict):
                continue
            related_id = str(relation.get("id") or "").strip()
            if not related_id or related_id == article_id:
                continue
            related = dict(relation)
            try:
                panel = self._literature_panel(f"EXT_ID:{related_id}", limit=3)
                match = next((item for item in panel.get("items") or [] if str(item.get("pmid") or item.get("id") or "") == related_id), None)
                if isinstance(match, dict):
                    related.update({
                        key: match.get(key)
                        for key in ("pmid", "pmcid", "doi", "title", "authors", "journal", "year", "abstract", "publication_types", "url")
                        if match.get(key) not in (None, "", [], {})
                    })
            except Exception:
                pass
            related_publications.append(related)
        if related_publications:
            base["related_publications"] = related_publications
        if full_text_sections:
            basis = "full_text"
        elif str(base.get("abstract") or "").strip():
            basis = "abstract"
        elif any(str(item.get("abstract") or "").strip() for item in related_publications):
            basis = "bibliographic_relation+linked_article_abstract"
        else:
            basis = "metadata_only"
        base["content_basis"] = basis
        return base

    def _literature_panel_for_pmids(
        self, pmids: list[str], *, limit: int, curated_by: str,
        metadata: dict[str, dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        unique = list(dict.fromkeys(str(x).strip() for x in pmids if str(x).strip()))
        if not unique:
            return {
                "id": "literature_curated", "entity_kind": "literature", "provider": "europe_pmc",
                "title": "Database-linked references · Europe PMC", "status": "ok",
                "count": 0, "items": [], "curated_by": curated_by,
                "pagination": {"mode": "local", "provider": "europe_pmc", "page_size": 10, "has_more": False},
            }
        by_id: dict[str, dict[str, Any]] = {}
        # Europe PMC accepts large boolean queries, but batching keeps URL/body sizes
        # bounded while preserving every finite curated reference supplied by UniProt/Rhea.
        for offset in range(0, len(unique), 80):
            batch = unique[offset:offset + 80]
            query = " OR ".join(f"EXT_ID:{pmid}" for pmid in batch)
            panel = self._literature_panel(query, limit=len(batch))
            for row in panel.get("items") or []:
                if not isinstance(row, dict):
                    continue
                pid = str(row.get("pmid") or row.get("id") or "").strip()
                if pid:
                    by_id[pid] = dict(row)
        ordered: list[dict[str, Any]] = []
        meta_map = metadata or {}
        for pmid in unique:
            row = by_id.get(pmid)
            if row is None:
                continue
            extra = meta_map.get(pmid) or {}
            if extra:
                row = {**row, "annotation_context": list(extra.get("annotation_context") or []), "curated_metadata": dict(extra)}
            ordered.append(row)
        return {
            "id": "literature_curated", "entity_kind": "literature", "provider": "europe_pmc",
            "title": "Database-linked references · Europe PMC", "status": "ok",
            "url": "https://europepmc.org/", "count": len(ordered), "items": ordered,
            "curated_reference_count": len(unique),
            "missing_reference_ids": [pmid for pmid in unique if pmid not in by_id],
            "curated_by": curated_by,
            "pagination": {"mode": "local", "provider": "europe_pmc", "page_size": 10, "has_more": False},
        }

    def _rhea_pubmed_ids(self, reaction_id: str) -> list[str]:
        response = self.session.get(
            "https://www.rhea-db.org/rhea/",
            params={"query": f"rhea:{reaction_id.split(':')[-1]}", "columns": "rhea-id,pubmed", "format": "tsv", "limit": 5},
            timeout=self.timeout,
        )
        response.raise_for_status()
        lines = [line for line in response.text.splitlines() if line.strip()]
        if len(lines) < 2:
            return []
        header = lines[0].split("\t")
        try:
            pub_index = header.index("PubMed")
        except ValueError:
            return []
        ids: list[str] = []
        for line in lines[1:]:
            parts = line.split("\t")
            if pub_index >= len(parts):
                continue
            for value in parts[pub_index].split(";"):
                pid = value.strip()
                if pid and pid not in ids:
                    ids.append(pid)
        return ids

    @staticmethod
    def _source_error(source_id: str, title: str, exc: Exception) -> dict[str, Any]:
        return {
            "id": source_id,
            "title": title,
            "status": "unavailable",
            "error": f"{type(exc).__name__}: {exc}"[:240],
        }

    def _model_domain(self, entity_kind: str, identifier: str, *, precomputed: bool) -> dict[str, Any]:
        identifier = str(identifier or "").strip()
        if entity_kind == "reaction":
            project_aligned = identifier in self.catalog.reaction_by_id
        else:
            try:
                project_aligned = bool(self.proteins.canonical_local_id(identifier))
            except Exception:
                project_aligned = identifier in self.catalog.protein_by_id
        if project_aligned:
            direction = "reaction_to_enzyme" if entity_kind == "reaction" else "enzyme_to_reaction"
            audit = dict(PROJECT_ALIGNED_EXTERNAL_RELATION_AUDIT.get(direction) or {})
            if audit:
                audit["context"] = dict(AUDIT_CONTEXT)
            return {
                "status": "project_aligned",
                "label_en": "Project-aligned model domain",
                "label_zh": "领域内模型扩展",
                "interpretation_en": "Use the model actively for prioritization; retrospective recovery on this exact target is shown when recorded relationships are available.",
                "interpretation_zh": "可把模型作为主动的优先级工具；如果这个目标有已记录关系，会同时显示本次查询的回顾性回收情况。",
                "retrospective_audit": audit or None,
            }
        if precomputed:
            return {
                "status": "expanded_universe_exploratory",
                "label_en": "Expanded-universe exploratory model view",
                "label_zh": "扩展候选域 · 探索性模型视角",
                "interpretation_en": "This target is in the expanded candidate universe. Use target-specific known-relation recovery as the local model check and the frontier as an experimental priority list.",
                "interpretation_zh": "这个目标位于扩展候选域；以本目标的已知关系回收作为局部模型检查，模型前沿用于安排后续实验优先级。",
            }
        return {
            "status": "open_world_exploratory",
            "label_en": "Open-world exploratory model view",
            "label_zh": "开放世界 · 探索性模型视角",
            "interpretation_en": "The query is encoded at request time. Use the model for hypothesis generation, not as database evidence.",
            "interpretation_zh": "这个查询在请求时现场编码；模型用于生成和排序假设，不作为数据库事实证据。",
        }

    @staticmethod
    def _conditioning_plan(eligible_known: list[str], *, max_seeds: int = 5) -> tuple[list[str], str | None]:
        unique = list(dict.fromkeys(str(value).strip() for value in eligible_known if str(value).strip()))
        if not unique:
            return [], None
        if len(unique) == 1:
            return unique, None
        holdout = unique[-1]
        seeds = unique[:-1][:max_seeds]
        return seeds, holdout

    @staticmethod
    def _recovery_payload(
        *,
        ranked: dict[str, dict[str, Any]],
        holdout_id: str | None,
        seed_ids: list[str],
    ) -> dict[str, Any]:
        if holdout_id:
            row = ranked.get(holdout_id)
            items = []
            if row is not None:
                items.append({
                    "id": holdout_id,
                    "rank": int(row.get("rank") or 0),
                    "score": float(row.get("score") or 0.0),
                })
            return {
                "mode": "leave_one_out",
                "eligible_recorded": 1,
                "recovered": 1 if row is not None else 0,
                "rate": 1.0 if row is not None else 0.0,
                "items": items,
                "holdout_id": holdout_id,
                "seed_count": len(seed_ids),
            }
        return {
            "mode": "seeded_no_holdout" if seed_ids else "no_recorded_anchor",
            "eligible_recorded": 0,
            "recovered": 0,
            "rate": None,
            "items": [],
            "holdout_id": None,
            "seed_count": len(seed_ids),
        }

    def _model_lens_protein(self, accession: str, *, known_result: dict[str, Any]) -> dict[str, Any]:
        started = time.time()
        canonical = self.evidence.canonical_protein_id(accession)
        model_ready = bool(self.evidence.is_candidate_protein(canonical))
        known_items = [
            row for row in (known_result.get("known_associations") or {}).get("items") or []
            if isinstance(row, dict)
        ]
        known_ids = list(dict.fromkeys(
            str(row.get("candidate_id") or "").strip()
            for row in known_items
            if str(row.get("candidate_id") or "").strip()
        ))
        eligible_known = [rid for rid in known_ids if self.evidence.is_candidate_reaction(rid)]
        seed_ids, holdout_id = self._conditioning_plan(eligible_known)

        # Recovery audit and candidate frontier are intentionally separate tasks.
        # The audit may condition on known associations, but the user-facing frontier
        # must use the same production zero-shot candidate path as a normal confirmed
        # ranking request; otherwise the workspace silently answers a different task.
        audit_payload: dict[str, Any]
        if model_ready:
            audit_payload = {
                "enzyme_id": canonical, "top_k": 20, "candidate_universe": "general_merged",
                "ranking_objective": "top20", "reliability_policy": "annotate",
            }
        else:
            exact = self.proteins.uniprot.exact(accession)
            sequence = str(exact.get("sequence") or "").strip()
            if not sequence:
                return {
                    "status": "unsupported", "reason": "protein sequence unavailable",
                    "latency_ms": round((time.time() - started) * 1000, 1),
                }
            audit_payload = {
                "query_id": accession, "enzyme_sequence": sequence, "protein_input_policy": "warn",
                "top_k": 20, "candidate_universe": "general_merged",
                "ranking_objective": "top20", "reliability_policy": "annotate",
            }
        if seed_ids:
            audit_payload.update({
                "known_reaction_ids": seed_ids, "retrieval_mode": "hybrid",
                "hybrid_direct_weight": 0.5,
            })
        audit_raw = self.model_gateway.rank("rank-reactions", audit_payload)
        audit_rows = [row for row in audit_raw.get("candidates") or [] if isinstance(row, dict)]
        audit_ranked = {str(row.get("candidate_id") or ""): row for row in audit_rows}
        recovery = self._recovery_payload(ranked=audit_ranked, holdout_id=holdout_id, seed_ids=seed_ids)
        audit_query = dict(audit_raw.get("query") or {})

        if self.retrieval_service is not None:
            frontier_raw = self.retrieval_service.rank_reactions(
                canonical if model_ready else accession,
                user_text="", route_mode="default", conversation_context={}, ui_language="en",
            )
            frontier_rows = [row for row in frontier_raw.get("candidates") or [] if isinstance(row, dict)]
            frontier_ranking = dict(frontier_raw.get("ranking") or {})
        else:
            # Compatibility fallback for isolated unit construction. It is deliberately
            # unconditioned Top-10 and never reuses the seeded audit ranking.
            fallback_payload = dict(audit_payload)
            fallback_payload.pop("known_reaction_ids", None)
            fallback_payload.pop("retrieval_mode", None)
            fallback_payload.pop("hybrid_direct_weight", None)
            fallback_payload["top_k"] = 10
            fallback_payload["ranking_objective"] = "top10"
            frontier_raw_engine = self.model_gateway.rank("rank-reactions", fallback_payload)
            frontier_rows = [row for row in frontier_raw_engine.get("candidates") or [] if isinstance(row, dict) and str(row.get("candidate_id") or "") not in set(known_ids)]
            frontier_query = dict(frontier_raw_engine.get("query") or {})
            frontier_ranking = {
                "top_k": 10, "route_id": frontier_query.get("route_id"),
                "score_source": frontier_query.get("score_source"),
                "candidate_universe": frontier_query.get("candidate_universe") or "general_merged",
                "candidate_universe_size": frontier_query.get("candidate_universe_size"),
                "scope": frontier_query.get("scope"), "shot_mode": frontier_query.get("shot_mode"),
            }

        frontier: list[dict[str, Any]] = []
        known_set = set(known_ids)
        for row in frontier_rows:
            rid = str(row.get("candidate_id") or "").strip()
            if not rid or rid in known_set:
                continue
            meta = self.evidence.reaction_metadata(rid) or self.catalog.reaction_by_id.get(rid, {}) or {}
            frontier.append({
                "rank": int(row.get("rank") or len(frontier) + 1),
                "candidate_id": rid, "score": float(row.get("score") or 0.0),
                "model_support_index": row.get("model_support_index"),
                "model_support_tier": row.get("model_support_tier"),
                "name": str(meta.get("name") or "") or None,
                "substrate_name": str(meta.get("substrate_name") or "") or None,
                "product_name": str(meta.get("product_name") or "") or None,
                "url": f"https://www.rhea-db.org/rhea/{rid.split(':',1)[1]}" if rid.startswith("RHEA:") else None,
            })
            if len(frontier) >= 5:
                break
        return {
            "status": "ok", "direction": "enzyme_to_reaction", "top_k": int(frontier_ranking.get("top_k") or 10),
            "mode": "production_frontier_with_separate_recovery_audit",
            "evidence_conditioned": False, "seed_count": len(seed_ids), "seed_ids": seed_ids,
            "hybrid_direct_weight": None, "query_in_precomputed_model_universe": model_ready,
            "domain": self._model_domain("protein", accession, precomputed=model_ready),
            "recorded_recovery": recovery, "frontier": frontier,
            "score_source": str(frontier_ranking.get("score_source") or ""),
            "route_id": str(frontier_ranking.get("route_id") or ""),
            "frontier_ranking": frontier_ranking,
            "recovery_audit": {
                "conditioned": bool(seed_ids), "top_k": 20,
                "score_source": str(audit_query.get("score_source") or ""),
                "route_id": str(audit_query.get("route_id") or ""),
                "seed_ids": seed_ids, "holdout_id": holdout_id,
            },
            "latency_ms": round((time.time() - started) * 1000, 1),
        }

    def _model_lens_reaction(self, reaction_id: str, *, known_result: dict[str, Any]) -> dict[str, Any]:
        started = time.time()
        known_items = [row for row in (known_result.get("known_associations") or {}).get("items") or [] if isinstance(row, dict)]
        known_ids = list(dict.fromkeys(str(row.get("candidate_id") or "").strip() for row in known_items if str(row.get("candidate_id") or "").strip()))
        canonical_known = list(dict.fromkeys(self.evidence.canonical_protein_id(pid) for pid in known_ids))
        eligible_known = [pid for pid in canonical_known if self.evidence.is_candidate_protein(pid)]
        seed_ids, holdout_id = self._conditioning_plan(eligible_known)
        model_ready = bool(self.evidence.is_candidate_reaction(reaction_id))

        if model_ready:
            audit_payload: dict[str, Any] = {
                "reaction_id": reaction_id, "top_k": 20, "candidate_universe": "general_merged",
                "ranking_objective": "top20", "reliability_policy": "annotate", "enzyme_taxonomy_scope": "all",
            }
        else:
            smiles = self.rhea.reaction_smiles(reaction_id, orientation="forward")
            audit_payload = {
                "query_id": reaction_id, "reaction_smiles": str(smiles.get("reaction_smiles") or ""),
                "reaction_feature_policy": "warn", "top_k": 20, "candidate_universe": "general_merged",
                "ranking_objective": "top20", "reliability_policy": "annotate", "enzyme_taxonomy_scope": "all",
            }
        if seed_ids:
            audit_payload.update({"known_enzyme_ids": seed_ids, "retrieval_mode": "hybrid", "hybrid_direct_weight": 0.5})
        audit_raw = self.model_gateway.rank("rank-enzymes", audit_payload)
        audit_rows = [row for row in audit_raw.get("candidates") or [] if isinstance(row, dict)]
        audit_ranked = {str(row.get("candidate_id") or ""): row for row in audit_rows}
        recovery = self._recovery_payload(ranked=audit_ranked, holdout_id=holdout_id, seed_ids=seed_ids)
        audit_query = dict(audit_raw.get("query") or {})

        if self.retrieval_service is not None:
            frontier_raw = self.retrieval_service.rank(
                reaction_id, user_text="", route_mode="default", top_k=10,
                conversation_context={}, ui_language="en",
            )
            frontier_rows = [row for row in frontier_raw.get("candidates") or [] if isinstance(row, dict)]
            frontier_ranking = dict(frontier_raw.get("ranking") or {})
        else:
            fallback_payload = dict(audit_payload)
            fallback_payload.pop("known_enzyme_ids", None); fallback_payload.pop("retrieval_mode", None); fallback_payload.pop("hybrid_direct_weight", None)
            fallback_payload["top_k"] = 10; fallback_payload["ranking_objective"] = "top10"
            raw_engine = self.model_gateway.rank("rank-enzymes", fallback_payload)
            known_set_fallback = set(canonical_known) | set(known_ids)
            frontier_rows = [row for row in raw_engine.get("candidates") or [] if isinstance(row, dict) and str(row.get("candidate_id") or "") not in known_set_fallback]
            frontier_query = dict(raw_engine.get("query") or {})
            frontier_ranking = {
                "top_k": 10, "route_id": frontier_query.get("route_id"), "score_source": frontier_query.get("score_source"),
                "candidate_universe": frontier_query.get("candidate_universe") or "general_merged",
                "candidate_universe_size": frontier_query.get("candidate_universe_size"),
                "scope": frontier_query.get("scope"), "shot_mode": frontier_query.get("shot_mode"),
            }
        known_set = set(canonical_known) | set(known_ids)
        frontier: list[dict[str, Any]] = []
        for row in frontier_rows:
            pid = str(row.get("candidate_id") or "").strip()
            if not pid or pid in known_set: continue
            meta = self.catalog.protein_by_id.get(pid, {}) or self.evidence.protein_metadata(pid) or {}
            accession = str(meta.get("uniprot_id") or "").strip() or (pid if probable_uniprot(pid) else "")
            frontier.append({
                "rank": int(row.get("rank") or len(frontier)+1), "candidate_id": pid, "score": float(row.get("score") or 0.0),
                "model_support_index": row.get("model_support_index"),
                "model_support_tier": row.get("model_support_tier"),
                "name": str(meta.get("name") or "") or None, "species": str(meta.get("species") or "") or None,
                "url": f"https://www.uniprot.org/uniprotkb/{quote(accession, safe='')}" if accession else None,
            })
            if len(frontier) >= 5: break
        return {
            "status": "ok", "direction": "reaction_to_enzyme", "top_k": int(frontier_ranking.get("top_k") or 10),
            "mode": "production_frontier_with_separate_recovery_audit", "evidence_conditioned": False,
            "seed_count": len(seed_ids), "seed_ids": seed_ids, "hybrid_direct_weight": None,
            "query_in_precomputed_model_universe": model_ready, "domain": self._model_domain("reaction", reaction_id, precomputed=model_ready),
            "recorded_recovery": recovery, "frontier": frontier, "score_source": str(frontier_ranking.get("score_source") or ""),
            "route_id": str(frontier_ranking.get("route_id") or ""), "frontier_ranking": frontier_ranking,
            "recovery_audit": {
                "conditioned": bool(seed_ids), "top_k": 20, "score_source": str(audit_query.get("score_source") or ""),
                "route_id": str(audit_query.get("route_id") or ""), "seed_ids": seed_ids, "holdout_id": holdout_id,
            },
            "latency_ms": round((time.time() - started) * 1000, 1),
        }

    @staticmethod
    def _workspace_route_view(
        *,
        entity_kind: str,
        entity_id: str,
        selected_sections: list[str],
        source_panels: list[dict[str, Any]],
        known_count: int,
        model_lens: dict[str, Any],
        opportunities_count: int = 0,
        ui_language: str,
    ) -> dict[str, Any]:
        zh = str(ui_language or "").lower().startswith("zh")
        selected = set(selected_sections)
        available = [row for row in source_panels if isinstance(row, dict) and row.get("status") == "ok"]
        unavailable = [row for row in source_panels if isinstance(row, dict) and row.get("status") != "ok"]
        model_ok = str(model_lens.get("status") or "") == "ok"
        recovery = model_lens.get("recorded_recovery") if isinstance(model_lens.get("recorded_recovery"), dict) else {}
        frontier_count = len(model_lens.get("frontier") or []) if model_ok else 0
        seed_count = int(model_lens.get("seed_count") or 0) if model_ok else 0
        if model_ok and seed_count and int(recovery.get("eligible_recorded") or 0):
            recovery_metric = (
                f"联合排序 · {seed_count} 个已知锚点 · 留出回收 {int(recovery.get('recovered') or 0)}/1"
                if zh else f"hybrid · {seed_count} anchors · holdout {int(recovery.get('recovered') or 0)}/1"
            )
        elif model_ok and seed_count:
            recovery_metric = f"联合排序 · {seed_count} 个已知锚点" if zh else f"hybrid · {seed_count} known anchors"
        elif model_ok and int(recovery.get("eligible_recorded") or 0):
            recovery_metric = (
                f"Top-{int(model_lens.get('top_k') or 20)} · 已知回收 {int(recovery.get('recovered') or 0)}/{int(recovery.get('eligible_recorded') or 0)}"
                if zh else f"Top-{int(model_lens.get('top_k') or 20)} · {int(recovery.get('recovered') or 0)}/{int(recovery.get('eligible_recorded') or 0)} known recovered"
            )
        else:
            recovery_metric = f"Top-{int(model_lens.get('top_k') or 20)}" if model_ok else ("未运行" if zh else "not run")
        nodes = [{
            "id": "research-entity",
            "title": "研究对象" if zh else "Research entity",
            "kind": "input",
            "metric": entity_id,
        }]
        section_titles = {
            "annotations": ("数据库注释", "Database annotations"),
            "structures": ("结构信息", "Structures"),
            "literature": ("关联文献", "Literature"),
            "recorded_relations": ("已记录关系", "Recorded relationships"),
            "model": ("模型视角", "Model lens"),
            "next_steps": ("下一步", "Next steps"),
        }
        for section in selected_sections:
            if section == "recorded_relations":
                metric = f"{known_count} 条" if zh else f"{known_count} recorded"
                kind = "evidence"
            elif section == "model":
                metric = recovery_metric
                kind = "model"
            elif section == "next_steps":
                metric = f"{int(opportunities_count)} 条动态建议" if zh else f"{int(opportunities_count)} contextual suggestions"
                kind = "output"
            else:
                section_panels = [row for row in source_panels if str(row.get("section") or "") == section]
                ok = sum(1 for row in section_panels if row.get("status") == "ok")
                metric = f"{ok}/{len(section_panels)} 可用" if zh else f"{ok}/{len(section_panels)} available"
                kind = "evidence"
            zh_title, en_title = section_titles.get(section, (section, section))
            nodes.append({"id": f"research-{section}", "title": zh_title if zh else en_title, "kind": kind, "metric": metric})
        edges = [{"from": nodes[i]["id"], "to": nodes[i+1]["id"]} for i in range(len(nodes)-1)]
        return {
            "direction": "research_workspace",
            "route_id": "research-workspace-v2",
            "base_route_id": "research-workspace-v2",
            "active_overlays": [],
            "title": "本轮科研组合" if zh else "Research composition",
            "summary": "只执行并组合本轮实际请求的科研模块。" if zh else "Only the research modules requested in this turn are executed and composed.",
            "selected_sections": list(selected_sections),
            "source_status": {"available": len(available), "unavailable": len(unavailable)},
            "nodes": nodes,
            "edges": edges,
        }

    def _opportunities(
        self,
        known_result: dict[str, Any],
        model_lens: dict[str, Any],
        *,
        entity_kind: str,
        entity_id: str,
        selected_sections: list[str],
        source_panels: list[dict[str, Any]],
        ui_language: str,
    ) -> list[dict[str, Any]]:
        if self.deepseek is None or not hasattr(self.deepseek, "suggest_next_steps"):
            return []
        known_count = int((known_result.get("known_associations") or {}).get("count") or 0)
        frontier = list(model_lens.get("frontier") or []) if model_lens.get("status") == "ok" else []
        panel_context = []
        for panel in source_panels:
            if not isinstance(panel, dict):
                continue
            items = []
            for row in list(panel.get("items") or [])[:4]:
                if not isinstance(row, dict):
                    continue
                items.append({
                    key: row.get(key)
                    for key in ("id", "title", "name", "source", "method", "year")
                    if row.get(key) not in (None, "")
                })
            panel_context.append({
                "section": panel.get("section"),
                "source": panel.get("title") or panel.get("id"),
                "status": panel.get("status"),
                "count": panel.get("count", len(panel.get("items") or [])),
                "items": items,
            })
        context = {
            "answer_mode": "research_workspace",
            "entity": {"kind": entity_kind, "id": entity_id},
            "selected_sections": list(selected_sections),
            "recorded_association_count": known_count,
            "model_frontier": [
                {
                    "id": row.get("candidate_id"),
                    "name": row.get("name") or row.get("substrate_name") or row.get("product_name"),
                    "score": row.get("score"),
                }
                for row in frontier[:5]
                if isinstance(row, dict)
            ],
            "source_panels": panel_context,
        }
        rows = self.deepseek.suggest_next_steps(
            result_context=context,
            session_facts={},
            tool_catalog=[],
            ui_language=ui_language,
            limit=4,
        )
        return [
            {
                "kind": "model_generated_next_step",
                "priority": row.get("priority") or "medium",
                "title": row.get("title") or row.get("prompt") or "",
                "reason": row.get("reason") or "",
                "prompt": row.get("prompt") or "",
            }
            for row in rows
            if isinstance(row, dict) and str(row.get("prompt") or row.get("title") or "").strip()
        ][:4]

    @staticmethod
    def _normalize_sections(sections: list[str] | tuple[str, ...] | None) -> list[str]:
        allowed = ("annotations", "structures", "literature", "recorded_relations", "model", "next_steps")
        values = list(sections or ("recorded_relations", "model"))
        result: list[str] = []
        for value in values:
            key = str(value or "").strip()
            if key in allowed and key not in result:
                result.append(key)
        return result or ["recorded_relations", "model"]

    @staticmethod
    def _tag_panel(panel: dict[str, Any], section: str) -> dict[str, Any]:
        tagged = dict(panel or {})
        tagged["section"] = section
        return tagged

    def protein_workspace(
        self, accession: str, *, ui_language: str = "en", literature_limit: int = 10,
        sections: list[str] | tuple[str, ...] | None = None, primary_section: str | None = None,
    ) -> dict[str, Any]:
        accession = str(accession or "").strip().upper()
        selected = self._normalize_sections(sections)
        primary = str(primary_section or "").strip()
        if primary not in selected:
            primary = selected[0]
        need_known = any(section in selected for section in ("recorded_relations", "model", "next_steps"))
        known = self.evidence_queries.lookup_protein_reactions(accession, ui_language=ui_language) if need_known else {"known_associations": {"count": 0, "items": []}}
        known_payload = known.get("known_associations") or {}
        local_meta = self.evidence.protein_metadata(accession) or {}
        entity_name = str(local_meta.get("name") or accession)
        entity_organism = str(local_meta.get("species") or local_meta.get("organism") or "") or None

        panels: list[dict[str, Any]] = []
        uniprot_panel: dict[str, Any] | None = None
        # UniProt is a necessary dependency for annotations, structure xrefs, and curated literature.
        # It is not fetched for a relations+model-only workspace.
        if any(section in selected for section in ("annotations", "structures", "literature")):
            try:
                uniprot_panel = self._uniprot_panel(accession)
                record = uniprot_panel.get("record") or {}
                entity_name = str(record.get("name") or entity_name)
                entity_organism = str(record.get("organism") or entity_organism or "") or None
            except Exception as exc:
                uniprot_panel = self._source_error("uniprot", "UniProtKB", exc)

        if "annotations" in selected:
            panels.append(self._tag_panel(uniprot_panel or self._source_error("uniprot", "UniProtKB", RuntimeError("not available")), "annotations"))
            try:
                panels.append(self._tag_panel(self._interpro_panel(accession), "annotations"))
            except Exception as exc:
                panels.append(self._tag_panel(self._source_error("interpro", "InterPro", exc), "annotations"))

        if "structures" in selected:
            try:
                panels.append(self._tag_panel(self._structure_panel(accession, uniprot_panel=uniprot_panel or {}), "structures"))
            except Exception as exc:
                panels.append(self._tag_panel(self._source_error("structures", "Structures", exc), "structures"))

        if "literature" in selected:
            curated_ids = list((uniprot_panel or {}).get("publication_ids") or [])
            curated_meta = dict((uniprot_panel or {}).get("curated_reference_metadata") or {})
            if curated_ids:
                try:
                    panels.append(self._tag_panel(self._literature_panel_for_pmids(
                        curated_ids, limit=literature_limit, curated_by="UniProtKB", metadata=curated_meta,
                    ), "literature"))
                except Exception as exc:
                    panels.append(self._tag_panel(self._source_error(
                        "literature_curated", "Database-linked references · Europe PMC", exc,
                    ), "literature"))
            europe_query = f'"{accession}" OR "{entity_name}"'
            try:
                broad = self._literature_panel(europe_query, limit=literature_limit)
                broad["id"] = "literature_europe_pmc"
                broad["title"] = "Broad biomedical search · Europe PMC"
                broad["curated_by"] = "broad_biomedical_search"
                panels.append(self._tag_panel(broad, "literature"))
            except Exception as exc:
                panels.append(self._tag_panel(self._source_error(
                    "literature_europe_pmc", "Broad biomedical search · Europe PMC", exc,
                ), "literature"))
            try:
                panels.append(self._tag_panel(
                    self._openalex_panel(accession, page_size=literature_limit), "literature",
                ))
            except Exception as exc:
                panels.append(self._tag_panel(self._source_error("literature_openalex", "OpenAlex", exc), "literature"))

        if "model" in selected:
            try:
                model = self._model_lens_protein(accession, known_result=known)
            except Exception as exc:
                model = self._source_error("model", "Model lens", exc)
        else:
            model = {"status": "not_requested"}

        opportunities = self._opportunities(
            known, model, entity_kind="protein", entity_id=accession, selected_sections=selected,
            source_panels=panels, ui_language=ui_language,
        ) if "next_steps" in selected else []
        visible_known = known_payload if "recorded_relations" in selected else None
        return {
            "answer_mode": "research_workspace",
            "workspace_kind": "protein",
            "title": "Scientific research workspace" if not str(ui_language).lower().startswith("zh") else "科研工作区",
            "selected_sections": selected,
            "primary_section": primary,
            "entity": {"kind": "protein", "id": accession, "name": entity_name, "subtitle": entity_organism, "url": f"https://www.uniprot.org/uniprotkb/{quote(accession, safe='')}"},
            "known_associations": visible_known,
            "source_panels": panels,
            "model_lens": model if "model" in selected else None,
            "opportunities": opportunities,
            "route_view": self._workspace_route_view(
                entity_kind="protein", entity_id=accession, selected_sections=selected, source_panels=panels,
                known_count=int(known_payload.get("count") or 0), model_lens=model,
                opportunities_count=len(opportunities), ui_language=ui_language,
            ),
            "score_note": "模型检索分数用于当前候选集合中的相对优先级。" if str(ui_language).lower().startswith("zh") else "Model retrieval scores are relative priorities within the current candidate set.",
        }

    def reaction_workspace(
        self, reaction_id: str, *, ui_language: str = "en", literature_limit: int = 10,
        sections: list[str] | tuple[str, ...] | None = None, primary_section: str | None = None,
    ) -> dict[str, Any]:
        reaction_id = str(reaction_id or "").strip().upper()
        selected = self._normalize_sections(sections)
        primary = str(primary_section or "").strip()
        if primary not in selected:
            primary = selected[0]
        need_known = any(section in selected for section in ("recorded_relations", "model", "next_steps"))
        known = self.evidence_queries.lookup_reaction_proteins(reaction_id, ui_language=ui_language) if need_known else {"known_associations": {"count": 0, "items": []}}
        known_payload = known.get("known_associations") or {}
        local_meta = self.evidence.reaction_metadata(reaction_id) or {}
        equation = str(local_meta.get("equation") or local_meta.get("name") or reaction_id)
        reaction_url = f"https://www.rhea-db.org/rhea/{reaction_id.split(':')[-1]}"
        panels: list[dict[str, Any]] = []
        reaction = None

        # Rhea details are fetched only for annotations or literature.
        if any(section in selected for section in ("annotations", "literature")):
            try:
                reaction = self.rhea.exact(reaction_id)
                equation = str(reaction.equation or equation)
                reaction_url = str(reaction.url or reaction_url)
            except Exception:
                reaction = None

        if "annotations" in selected:
            chebi_names = list(getattr(reaction, "chebi_names", None) or [])
            chebi_ids = list(getattr(reaction, "chebi_ids", None) or [])
            try:
                reaction_smiles = str(self.rhea.reaction_smiles(reaction_id, orientation="forward").get("reaction_smiles") or "")
            except Exception:
                reaction_smiles = str(local_meta.get("reaction_smiles") or "")
            try:
                official_uniprot_ids = list(self.route_designer.known_uniprot_ids(reaction_id))
            except Exception:
                official_uniprot_ids = []
            participants = []
            for index, chebi_id in enumerate(chebi_ids):
                name = chebi_names[index] if index < len(chebi_names) else chebi_id
                participants.append({"id": chebi_id, "name": name, "url": f"https://www.ebi.ac.uk/chebi/searchId.do?chebiId={quote(str(chebi_id), safe=':')}"})
            rhea_panel = {
                "id": "rhea", "title": "Rhea", "status": "ok", "url": reaction_url,
                "facts": [
                    {"label": "Reaction", "value": equation},
                    {"label": "Rhea ID", "value": reaction_id},
                    {"label": "Swiss-Prot mapping", "value": len(official_uniprot_ids)},
                    {"label": "Rhea enzyme count", "value": getattr(reaction, "enzyme_count", None)},
                ],
                "reaction_smiles": reaction_smiles or None, "participants": participants,
                "official_uniprot_ids": official_uniprot_ids,
                "official_uniprot_items": [
                    {"id": protein_id, "url": f"https://www.uniprot.org/uniprotkb/{quote(str(protein_id), safe='')}"}
                    for protein_id in official_uniprot_ids
                ],
                "known_protein_count": int(known_payload.get("count") or 0),
            }
            panels.append(self._tag_panel(rhea_panel, "annotations"))

        if "structures" in selected:
            panels.append(self._tag_panel({
                "id": "structures", "title": "Structures", "status": "not_applicable",
                "note": "A reaction has no single protein structure; select a concrete enzyme to inspect PDB/AlphaFold records.",
                "items": [],
            }, "structures"))

        if "literature" in selected:
            try:
                rhea_pmids = self._rhea_pubmed_ids(reaction_id)
            except Exception:
                rhea_pmids = []
            if rhea_pmids:
                try:
                    panels.append(self._tag_panel(self._literature_panel_for_pmids(
                        rhea_pmids, limit=literature_limit, curated_by="Rhea",
                    ), "literature"))
                except Exception as exc:
                    panels.append(self._tag_panel(self._source_error(
                        "literature_curated", "Database-linked references · Europe PMC", exc,
                    ), "literature"))
            chebi_names = list(getattr(reaction, "chebi_names", None) or [])
            terms = [reaction_id] + [name for name in chebi_names if len(str(name).strip()) >= 4][:3]
            europe_query = " OR ".join(
                f'"{str(term).replace(chr(34), "").strip()}"'
                for term in terms if str(term).strip()
            ) or reaction_id
            try:
                broad = self._literature_panel(europe_query, limit=literature_limit)
                broad["id"] = "literature_europe_pmc"
                broad["title"] = "Broad biomedical search · Europe PMC"
                broad["curated_by"] = "broad_biomedical_search"
                panels.append(self._tag_panel(broad, "literature"))
            except Exception as exc:
                panels.append(self._tag_panel(self._source_error(
                    "literature_europe_pmc", "Broad biomedical search · Europe PMC", exc,
                ), "literature"))
            openalex_query = " ".join(
                [reaction_id] + [str(name).strip() for name in chebi_names[:2] if str(name).strip()]
            )
            try:
                panels.append(self._tag_panel(
                    self._openalex_panel(openalex_query, page_size=literature_limit), "literature",
                ))
            except Exception as exc:
                panels.append(self._tag_panel(self._source_error("literature_openalex", "OpenAlex", exc), "literature"))

        if "model" in selected:
            try:
                model = self._model_lens_reaction(reaction_id, known_result=known)
            except Exception as exc:
                model = self._source_error("model", "Model lens", exc)
        else:
            model = {"status": "not_requested"}
        opportunities = self._opportunities(
            known, model, entity_kind="reaction", entity_id=reaction_id, selected_sections=selected,
            source_panels=panels, ui_language=ui_language,
        ) if "next_steps" in selected else []
        visible_known = known_payload if "recorded_relations" in selected else None
        return {
            "answer_mode": "research_workspace",
            "workspace_kind": "reaction",
            "title": "Scientific research workspace" if not str(ui_language).lower().startswith("zh") else "科研工作区",
            "selected_sections": selected,
            "primary_section": primary,
            "entity": {"kind": "reaction", "id": reaction_id, "name": equation, "subtitle": None, "url": reaction_url},
            "known_associations": visible_known,
            "source_panels": panels,
            "model_lens": model if "model" in selected else None,
            "opportunities": opportunities,
            "route_view": self._workspace_route_view(
                entity_kind="reaction", entity_id=reaction_id, selected_sections=selected, source_panels=panels,
                known_count=int(known_payload.get("count") or 0), model_lens=model,
                opportunities_count=len(opportunities), ui_language=ui_language,
            ),
            "score_note": "模型检索分数用于当前候选集合中的相对优先级。" if str(ui_language).lower().startswith("zh") else "Model retrieval scores are relative priorities within the current candidate set.",
        }
