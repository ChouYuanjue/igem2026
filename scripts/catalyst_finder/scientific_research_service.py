from __future__ import annotations

import concurrent.futures
import html
import re
import time
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
    ) -> None:
        self.evidence = evidence
        self.evidence_queries = evidence_queries
        self.proteins = proteins
        self.rhea = rhea
        self.route_designer = route_designer
        self.model_gateway = model_gateway
        self.catalog = catalog
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
            {"label": "Gene", "value": ", ".join(genes[:8])},
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
                if xid not in bucket and len(bucket) < 12:
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
            "catalytic_activities": catalytic[:12],
            "cofactors": cofactors[:12],
            "annotations": {key: value[:5] for key, value in comments.items() if key in {"FUNCTION", "ACTIVITY REGULATION", "SUBCELLULAR LOCATION", "PH DEPENDENCE", "TEMPERATURE DEPENDENCE"}},
            "cross_references": xrefs,
            "cross_reference_items": xref_items[:48],
            "publication_ids": pmids[:40],
            "curated_reference_metadata": curated_reference_metadata,
            "record": {"accession": accession, "name": name, "organism": organism, "genes": genes},
            "latency_ms": round((time.time() - started) * 1000, 1),
        }

    def _structure_panel(self, accession: str, *, uniprot_panel: dict[str, Any]) -> dict[str, Any]:
        started = time.time()
        xrefs = uniprot_panel.get("cross_references") if isinstance(uniprot_panel, dict) else {}
        xrefs = xrefs if isinstance(xrefs, dict) else {}
        pdb_ids = [str(x).strip().upper() for x in xrefs.get("PDB") or [] if str(x).strip()][:6]
        alphafold_ids = [str(x).strip() for x in xrefs.get("AlphaFoldDB") or [] if str(x).strip()][:2]
        # AlphaFold API is accession-addressable even when a UniProt cross-reference is
        # temporarily absent, so try the verified accession as a bounded fallback.
        if not alphafold_ids and probable_uniprot(accession):
            alphafold_ids = [accession]

        items: list[dict[str, Any]] = []

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

        futures: list[Any] = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(8, max(1, len(pdb_ids) + len(alphafold_ids)))) as pool:
            futures.extend(pool.submit(fetch_pdb, pdb_id) for pdb_id in pdb_ids)
            futures.extend(pool.submit(fetch_alphafold, model_id) for model_id in alphafold_ids[:1])
            for future in futures:
                try:
                    row = future.result()
                except Exception:
                    row = None
                if isinstance(row, dict):
                    items.append(row)
        experimental = sum(str(row.get("type") or "") == "experimental_structure" for row in items)
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
        url = f"https://www.ebi.ac.uk/interpro/api/entry/interpro/protein/uniprot/{quote(accession, safe='')}"
        response = self.session.get(url, params={"page_size": 20}, timeout=self.timeout)
        response.raise_for_status()
        payload = response.json()
        entries: list[dict[str, Any]] = []
        for row in payload.get("results") or []:
            meta = (row or {}).get("metadata") or {}
            acc = str(meta.get("accession") or "").strip()
            if not acc:
                continue
            member = meta.get("member_databases") or {}
            members: list[str] = []
            for db, values in member.items():
                if isinstance(values, dict):
                    members.extend(f"{db}:{key}" for key in list(values)[:4])
            entries.append({
                "id": acc,
                "name": str(meta.get("name") or acc),
                "type": str(meta.get("type") or ""),
                "member_entries": members[:8],
                "url": f"https://www.ebi.ac.uk/interpro/entry/InterPro/{acc}/",
            })
        return {
            "id": "interpro",
            "title": "InterPro",
            "status": "ok",
            "url": f"https://www.ebi.ac.uk/interpro/protein/UniProt/{quote(accession, safe='')}/",
            "count": int(payload.get("count") or len(entries)),
            "items": entries[:12],
            "latency_ms": round((time.time() - started) * 1000, 1),
        }

    def _literature_panel(self, query: str, *, limit: int) -> dict[str, Any]:
        started = time.time()
        url = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
        response = self.session.get(
            url,
            params={"query": query, "format": "json", "resultType": "core", "pageSize": max(1, min(limit, 8))},
            timeout=self.timeout,
        )
        response.raise_for_status()
        payload = response.json()
        items: list[dict[str, Any]] = []
        for row in ((payload.get("resultList") or {}).get("result") or []):
            if not isinstance(row, dict):
                continue
            source = str(row.get("source") or "")
            article_id = str(row.get("id") or row.get("pmid") or "")
            if not article_id:
                continue
            items.append({
                "id": article_id,
                "source": source,
                "pmid": str(row.get("pmid") or "") or None,
                "pmcid": str(row.get("pmcid") or "") or None,
                "doi": str(row.get("doi") or "") or None,
                "title": self._plain_text(row.get("title") or article_id),
                "authors": self._plain_text(row.get("authorString") or ""),
                "journal": self._plain_text(row.get("journalTitle") or ""),
                "year": str(row.get("pubYear") or ""),
                "abstract": self._plain_text(row.get("abstractText") or ""),
                "cited_by": int(row.get("citedByCount") or 0),
                "open_access": str(row.get("isOpenAccess") or "").upper() == "Y",
                "url": f"https://europepmc.org/article/{quote(source, safe='')}/{quote(article_id, safe='')}",
            })
        return {
            "id": "literature",
            "title": "Europe PMC",
            "status": "ok",
            "url": f"https://europepmc.org/search?query={quote(query)}",
            "query": query,
            "count": int(payload.get("hitCount") or len(items)),
            "items": items,
            "latency_ms": round((time.time() - started) * 1000, 1),
        }

    def _literature_panel_for_pmids(
        self,
        pmids: list[str],
        *,
        limit: int,
        curated_by: str,
        metadata: dict[str, dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        unique = list(dict.fromkeys(str(x).strip() for x in pmids if str(x).strip()))[:40]
        if not unique:
            return {"id": "literature", "title": "Europe PMC", "status": "ok", "count": 0, "items": [], "curated_by": curated_by}
        query = " OR ".join(f"EXT_ID:{pid}" for pid in unique)
        panel = self._literature_panel(query, limit=min(max(limit, 1), 8))
        panel["curated_by"] = curated_by
        panel["curated_reference_count"] = len(unique)
        context = metadata or {}
        for row in panel.get("items") or []:
            pmid = str(row.get("pmid") or row.get("id") or "")
            extra = context.get(pmid) or {}
            if extra:
                row["annotation_context"] = list(extra.get("annotation_context") or [])
                for key in ("doi", "journal", "year", "authors", "title"):
                    if not row.get(key) and extra.get(key):
                        row[key] = extra[key]
        # Keep the source's curated ordering when possible; it is often more useful than
        # Europe PMC's default relevance ordering for an exact-ID disjunction.
        order = {pid: index for index, pid in enumerate(unique)}
        panel["items"] = sorted(panel.get("items") or [], key=lambda row: order.get(str(row.get("pmid") or row.get("id") or ""), 10**6))[:limit]
        panel["count"] = len(unique)
        return panel

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

        if model_ready:
            payload: dict[str, Any] = {
                "enzyme_id": canonical,
                "top_k": 20,
                "candidate_universe": "general_merged",
                "ranking_objective": "top20",
                "reliability_policy": "annotate",
            }
        else:
            exact = self.proteins.uniprot.exact(accession)
            sequence = str(exact.get("sequence") or "").strip()
            if not sequence:
                return {
                    "status": "unsupported",
                    "reason": "protein sequence unavailable",
                    "latency_ms": round((time.time() - started) * 1000, 1),
                }
            payload = {
                "query_id": accession,
                "enzyme_sequence": sequence,
                "protein_input_policy": "warn",
                "top_k": 20,
                "candidate_universe": "general_merged",
                "ranking_objective": "top20",
                "reliability_policy": "annotate",
            }
        if seed_ids:
            payload.update({
                "known_reaction_ids": seed_ids,
                "retrieval_mode": "hybrid",
                "hybrid_direct_weight": 0.5,
            })

        raw = self.model_gateway.rank("rank-reactions", payload)
        rows = [row for row in raw.get("candidates") or [] if isinstance(row, dict)]
        ranked = {str(row.get("candidate_id") or ""): row for row in rows}
        recovery = self._recovery_payload(ranked=ranked, holdout_id=holdout_id, seed_ids=seed_ids)
        known_set = set(known_ids)
        frontier: list[dict[str, Any]] = []
        for row in rows:
            rid = str(row.get("candidate_id") or "").strip()
            if not rid or rid in known_set:
                continue
            meta = self.evidence.reaction_metadata(rid) or self.catalog.reaction_by_id.get(rid, {}) or {}
            frontier.append({
                "rank": int(row.get("rank") or len(frontier) + 1),
                "candidate_id": rid,
                "score": float(row.get("score") or 0.0),
                "name": str(meta.get("name") or "") or None,
                "substrate_name": str(meta.get("substrate_name") or "") or None,
                "product_name": str(meta.get("product_name") or "") or None,
                "url": f"https://www.rhea-db.org/rhea/{rid.split(':',1)[1]}" if rid.startswith("RHEA:") else None,
            })
            if len(frontier) >= 5:
                break
        query = raw.get("query") or {}
        return {
            "status": "ok",
            "direction": "enzyme_to_reaction",
            "top_k": 20,
            "mode": "evidence_conditioned_hybrid" if seed_ids else "direct_model",
            "evidence_conditioned": bool(seed_ids),
            "seed_count": len(seed_ids),
            "seed_ids": seed_ids,
            "hybrid_direct_weight": 0.5 if seed_ids else None,
            "query_in_precomputed_model_universe": model_ready,
            "domain": self._model_domain("protein", accession, precomputed=model_ready),
            "recorded_recovery": recovery,
            "frontier": frontier,
            "score_source": str(query.get("score_source") or ""),
            "route_id": str(query.get("route_id") or ""),
            "latency_ms": round((time.time() - started) * 1000, 1),
        }

    def _model_lens_reaction(self, reaction_id: str, *, known_result: dict[str, Any]) -> dict[str, Any]:
        started = time.time()
        known_items = [
            row for row in (known_result.get("known_associations") or {}).get("items") or []
            if isinstance(row, dict)
        ]
        known_ids = list(dict.fromkeys(
            str(row.get("candidate_id") or "").strip()
            for row in known_items
            if str(row.get("candidate_id") or "").strip()
        ))
        canonical_known = list(dict.fromkeys(self.evidence.canonical_protein_id(pid) for pid in known_ids))
        eligible_known = [pid for pid in canonical_known if self.evidence.is_candidate_protein(pid)]
        seed_ids, holdout_id = self._conditioning_plan(eligible_known)
        model_ready = bool(self.evidence.is_candidate_reaction(reaction_id))

        if model_ready:
            payload: dict[str, Any] = {
                "reaction_id": reaction_id,
                "top_k": 20,
                "candidate_universe": "general_merged",
                "ranking_objective": "top20",
                "reliability_policy": "annotate",
                "enzyme_taxonomy_scope": "all",
            }
        else:
            smiles = self.rhea.reaction_smiles(reaction_id, orientation="forward")
            payload = {
                "query_id": reaction_id,
                "reaction_smiles": str(smiles.get("reaction_smiles") or ""),
                "reaction_feature_policy": "warn",
                "top_k": 20,
                "candidate_universe": "general_merged",
                "ranking_objective": "top20",
                "reliability_policy": "annotate",
                "enzyme_taxonomy_scope": "all",
            }
        if seed_ids:
            payload.update({
                "known_enzyme_ids": seed_ids,
                "retrieval_mode": "hybrid",
                "hybrid_direct_weight": 0.5,
            })

        raw = self.model_gateway.rank("rank-enzymes", payload)
        rows = [row for row in raw.get("candidates") or [] if isinstance(row, dict)]
        ranked = {str(row.get("candidate_id") or ""): row for row in rows}
        recovery = self._recovery_payload(ranked=ranked, holdout_id=holdout_id, seed_ids=seed_ids)
        known_set = set(canonical_known) | set(known_ids)
        frontier: list[dict[str, Any]] = []
        for row in rows:
            pid = str(row.get("candidate_id") or "").strip()
            if not pid or pid in known_set:
                continue
            meta = self.catalog.protein_by_id.get(pid, {}) or self.evidence.protein_metadata(pid) or {}
            accession = str(meta.get("uniprot_id") or "").strip() or (pid if probable_uniprot(pid) else "")
            frontier.append({
                "rank": int(row.get("rank") or len(frontier) + 1),
                "candidate_id": pid,
                "score": float(row.get("score") or 0.0),
                "name": str(meta.get("name") or "") or None,
                "species": str(meta.get("species") or "") or None,
                "url": f"https://www.uniprot.org/uniprotkb/{quote(accession, safe='')}" if accession else None,
            })
            if len(frontier) >= 5:
                break
        query = raw.get("query") or {}
        return {
            "status": "ok",
            "direction": "reaction_to_enzyme",
            "top_k": 20,
            "mode": "evidence_conditioned_hybrid" if seed_ids else "direct_model",
            "evidence_conditioned": bool(seed_ids),
            "seed_count": len(seed_ids),
            "seed_ids": seed_ids,
            "hybrid_direct_weight": 0.5 if seed_ids else None,
            "query_in_precomputed_model_universe": model_ready,
            "domain": self._model_domain("reaction", reaction_id, precomputed=model_ready),
            "recorded_recovery": recovery,
            "frontier": frontier,
            "score_source": str(query.get("score_source") or ""),
            "route_id": str(query.get("route_id") or ""),
            "latency_ms": round((time.time() - started) * 1000, 1),
        }

    @staticmethod
    def _workspace_route_view(
        *,
        entity_kind: str,
        entity_id: str,
        source_panels: list[dict[str, Any]],
        known_count: int,
        model_lens: dict[str, Any],
        ui_language: str,
    ) -> dict[str, Any]:
        zh = str(ui_language or "").lower().startswith("zh")
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
            recovery_metric = (f"Top-{int(model_lens.get('top_k') or 20)} 模型排序" if zh else f"Top-{int(model_lens.get('top_k') or 20)} model lens") if model_ok else ("模型暂不可用" if zh else "model unavailable")
        nodes = [
            {
                "id": "research-entity",
                "title": "锁定研究对象" if zh else "Verify the research entity",
                "subtitle": entity_kind,
                "kind": "input",
                "metric": entity_id,
                "detail": "后续资料、数据库关系与模型检索都绑定到同一个已核对实体。" if zh else "All later evidence and model work is tied to this same verified entity.",
            },
            {
                "id": "research-live-sources",
                "title": "按需汇集外部资料" if zh else "Query live research sources",
                "subtitle": " · ".join(str(row.get("title") or row.get("id") or "") for row in available),
                "kind": "evidence",
                "metric": f"{len(available)} available · {len(unavailable)} unavailable",
                "detail": "实时读取可用的官方数据库与文献接口，不依赖本地整库镜像。" if zh else "Read current official database and literature services on demand instead of depending on full local mirrors.",
            },
            {
                "id": "research-recorded-relations",
                "title": "整理已记录关系" if zh else "Assemble recorded relationships",
                "subtitle": "Rhea / UniProt / integrated evidence",
                "kind": "evidence",
                "metric": f"{known_count} recorded associations",
                "detail": "把数据库关系作为可核对的事实层，保留来源和外部记录入口。" if zh else "Keep database relationships as the auditable factual layer with source provenance.",
            },
            {
                "id": "research-model-lens",
                "title": "让模型检验已知并向外扩展" if zh else "Use the model as a research lens",
                "subtitle": "已知关系回看 + 新关联扩展" if zh else "retrospective recovery + frontier",
                "kind": "model",
                "metric": recovery_metric,
                "detail": (
                    "已记录关系直接作为模型锚点，与当前目标的模型分数共同决定新关联优先级；有足够锚点时同时留出一条已知关系做局部回看。"
                    if zh and seed_count else
                    "Recorded relationships directly anchor a hybrid model ranking; when enough anchors exist, one known relation is held out as a local check."
                    if seed_count else
                    "同一目标上查看模型对已知关系的回看与新关联优先级。"
                    if zh else
                    "On the same target, inspect known-relation recovery and unrecorded priorities."
                ),
            },
            {
                "id": "research-next-tests",
                "title": "形成下一步验证短名单" if zh else "Create the next validation shortlist",
                "subtitle": "evidence gap → testable priorities",
                "kind": "output",
                "metric": f"{frontier_count} 个新关联候选" if zh else f"{frontier_count} frontier candidates",
                "detail": "把资料空白、模型排序和现有证据放在一起决定下一步查什么、测什么。" if zh else "Combine evidence gaps and model priorities into concrete next research or experimental checks.",
            },
        ]
        return {
            "direction": "research_workspace",
            "route_id": "research-workspace-v1",
            "base_route_id": "research-workspace-v1",
            "active_overlays": [],
            "title": "科研资料与模型联合工作流" if zh else "Integrated research evidence and model workflow",
            "summary": "从同一个已核对目标出发，按需汇集资料、核对已知关系，并用模型检验已知与提出下一步实验优先级。" if zh else "Start from one verified target, gather live research evidence, verify recorded relations, then use the model both retrospectively and prospectively.",
            "nodes": nodes,
            "edges": [
                {"from": "research-entity", "to": "research-live-sources"},
                {"from": "research-live-sources", "to": "research-recorded-relations"},
                {"from": "research-recorded-relations", "to": "research-model-lens"},
                {"from": "research-model-lens", "to": "research-next-tests"},
            ],
        }

    @staticmethod
    def _opportunities(known_result: dict[str, Any], model_lens: dict[str, Any], *, entity_kind: str) -> list[dict[str, Any]]:
        known_count = int((known_result.get("known_associations") or {}).get("count") or 0)
        frontier_count = len(model_lens.get("frontier") or []) if model_lens.get("status") == "ok" else 0
        recovery = model_lens.get("recorded_recovery") if isinstance(model_lens.get("recorded_recovery"), dict) else {}
        items: list[dict[str, Any]] = []
        if frontier_count:
            items.append({
                "kind": "model_frontier",
                "priority": "high",
                "title": "Explore the model frontier",
                "reason": f"{frontier_count} high-priority unrecorded associations are already available from the same verified query.",
            })
        if known_count == 0:
            items.append({
                "kind": "evidence_gap",
                "priority": "high",
                "title": "Treat this as an evidence gap",
                "reason": "No recorded association was found in the integrated evidence layer; literature and model results become the natural next checks.",
            })
        elif recovery.get("eligible_recorded"):
            items.append({
                "kind": "retrospective_model_check",
                "priority": "medium",
                "title": "Use known associations as an internal model check",
                "reason": f"The Top-{model_lens.get('top_k', 20)} model list recovered {recovery.get('recovered', 0)}/{recovery.get('eligible_recorded', 0)} recorded associations that fall inside the active model universe.",
            })
        if entity_kind == "protein":
            items.append({"kind": "experimental_context", "priority": "medium", "title": "Check domains, literature and assay context", "reason": "Protein annotations and literature often determine whether a model-ranked reaction is experimentally plausible."})
        else:
            items.append({"kind": "candidate_validation", "priority": "medium", "title": "Compare recorded and model-ranked enzymes", "reason": "The recorded set supplies anchors; the frontier is useful for broader screening or sequence-diverse follow-up."})
        return items[:4]

    def protein_workspace(self, accession: str, *, ui_language: str = "en", literature_limit: int = 6, include_model: bool = True) -> dict[str, Any]:
        accession = str(accession or "").strip().upper()
        known = self.evidence_queries.lookup_protein_reactions(accession, ui_language=ui_language)
        try:
            exact = self.proteins.uniprot.exact(accession)
        except Exception:
            exact = {"accession": accession, "name": accession, "organism": None}
        tasks: dict[str, Any] = {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
            futures = {
                "uniprot": pool.submit(self._uniprot_panel, accession),
                "interpro": pool.submit(self._interpro_panel, accession),
            }
            if include_model:
                futures["model"] = pool.submit(self._model_lens_protein, accession, known_result=known)
            for key, future in futures.items():
                try:
                    tasks[key] = future.result()
                except Exception as exc:
                    title = {"uniprot": "UniProtKB", "interpro": "InterPro", "model": "Model lens"}[key]
                    tasks[key] = self._source_error(key, title, exc)
        uniprot_panel = tasks.get("uniprot") if isinstance(tasks.get("uniprot"), dict) else {}
        curated_ids = list(uniprot_panel.get("publication_ids") or [])
        curated_meta = dict(uniprot_panel.get("curated_reference_metadata") or {})
        fallback_name = str((uniprot_panel.get("record") or {}).get("name") or exact.get("name") or accession)
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            structure_future = pool.submit(self._structure_panel, accession, uniprot_panel=uniprot_panel)
            if curated_ids:
                literature_future = pool.submit(self._literature_panel_for_pmids, curated_ids, limit=literature_limit, curated_by="UniProtKB", metadata=curated_meta)
            else:
                literature_future = pool.submit(self._literature_panel, f'"{accession}" OR "{fallback_name}"', limit=literature_limit)
            try:
                tasks["structures"] = structure_future.result()
            except Exception as exc:
                tasks["structures"] = self._source_error("structures", "Structures", exc)
            try:
                tasks["literature"] = literature_future.result()
                if not curated_ids and isinstance(tasks["literature"], dict):
                    tasks["literature"]["curated_by"] = "keyword_fallback"
            except Exception as exc:
                tasks["literature"] = self._source_error("literature", "Europe PMC", exc)
        model = tasks.pop("model", {"status": "disabled"})
        panels = [tasks[key] for key in ("uniprot", "interpro", "structures", "literature")]
        name = str((tasks.get("uniprot") or {}).get("record", {}).get("name") or exact.get("name") or accession)
        organism = str((tasks.get("uniprot") or {}).get("record", {}).get("organism") or exact.get("organism") or "") or None
        known_payload = known.get("known_associations") or {}
        return {
            "answer_mode": "research_workspace",
            "workspace_kind": "protein",
            "title": "Scientific research workspace" if not str(ui_language).lower().startswith("zh") else "科研资料工作区",
            "entity": {"kind": "protein", "id": accession, "name": name, "subtitle": organism, "url": f"https://www.uniprot.org/uniprotkb/{quote(accession, safe='')}"},
            "known_associations": known_payload,
            "source_panels": panels,
            "model_lens": model,
            "opportunities": self._opportunities(known, model, entity_kind="protein"),
            "route_view": self._workspace_route_view(
                entity_kind="protein", entity_id=accession, source_panels=panels,
                known_count=int(known_payload.get("count") or 0), model_lens=model, ui_language=ui_language,
            ),
            "score_note": "模型检索分数用于当前候选集合中的相对优先级；数据库关系仍由证据源确定。" if str(ui_language).lower().startswith("zh") else "Model retrieval scores rank priorities within the current candidate set; database relationships come from evidence sources.",
        }

    def reaction_workspace(self, reaction_id: str, *, ui_language: str = "en", literature_limit: int = 6, include_model: bool = True) -> dict[str, Any]:
        reaction_id = str(reaction_id or "").strip().upper()
        known = self.evidence_queries.lookup_reaction_proteins(reaction_id, ui_language=ui_language)
        reaction = None
        try:
            reaction = self.rhea.exact(reaction_id)
            equation = str(reaction.equation or reaction_id)
            reaction_url = str(reaction.url or f"https://www.rhea-db.org/rhea/{reaction_id.split(':')[-1]}")
        except Exception:
            equation = reaction_id
            reaction_url = f"https://www.rhea-db.org/rhea/{reaction_id.split(':')[-1]}"
        chebi_names = list(getattr(reaction, "chebi_names", None) or [])
        chebi_ids = list(getattr(reaction, "chebi_ids", None) or [])
        try:
            reaction_smiles = str(self.rhea.reaction_smiles(reaction_id, orientation="forward").get("reaction_smiles") or "")
        except Exception:
            reaction_smiles = ""
        try:
            official_uniprot_ids = list(self.route_designer.known_uniprot_ids(reaction_id))
        except Exception:
            official_uniprot_ids = []
        literature_terms = [reaction_id] + [name for name in chebi_names if len(str(name).strip()) >= 4][:3]
        literature_query = " OR ".join(f'"{str(term).replace(chr(34), "").strip()}"' for term in literature_terms if str(term).strip()) or reaction_id
        tasks: dict[str, Any] = {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            futures = {}
            if include_model:
                futures["model"] = pool.submit(self._model_lens_reaction, reaction_id, known_result=known)
            futures["rhea_pubmed"] = pool.submit(self._rhea_pubmed_ids, reaction_id)
            for key, future in futures.items():
                try:
                    tasks[key] = future.result()
                except Exception as exc:
                    tasks[key] = [] if key == "rhea_pubmed" else self._source_error("model", "Model lens", exc)
        try:
            rhea_pmids = list(tasks.pop("rhea_pubmed", []) or [])
            if rhea_pmids:
                tasks["literature"] = self._literature_panel_for_pmids(rhea_pmids, limit=literature_limit, curated_by="Rhea")
            else:
                tasks["literature"] = self._literature_panel(literature_query, limit=literature_limit)
                tasks["literature"]["curated_by"] = "keyword_fallback"
        except Exception as exc:
            tasks["literature"] = self._source_error("literature", "Europe PMC", exc)
        model = tasks.pop("model", {"status": "disabled"})
        participants = []
        for index, chebi_id in enumerate(chebi_ids):
            name = chebi_names[index] if index < len(chebi_names) else chebi_id
            participants.append({
                "id": chebi_id,
                "name": name,
                "url": f"https://www.ebi.ac.uk/chebi/searchId.do?chebiId={quote(str(chebi_id), safe=':')}",
            })
        rhea_panel = {
            "id": "rhea",
            "title": "Rhea",
            "status": "ok",
            "url": reaction_url,
            "facts": [
                {"label": "Reaction", "value": equation},
                {"label": "Rhea ID", "value": reaction_id},
                {"label": "Swiss-Prot mapping", "value": len(official_uniprot_ids)},
                {"label": "Rhea enzyme count", "value": getattr(reaction, "enzyme_count", None)},
            ],
            "reaction_smiles": reaction_smiles or None,
            "participants": participants[:16],
            "official_uniprot_ids": official_uniprot_ids[:20],
            "known_protein_count": int((known.get("known_associations") or {}).get("count") or 0),
        }
        panels = [rhea_panel, tasks["literature"]]
        known_payload = known.get("known_associations") or {}
        return {
            "answer_mode": "research_workspace",
            "workspace_kind": "reaction",
            "title": "Scientific research workspace" if not str(ui_language).lower().startswith("zh") else "科研资料工作区",
            "entity": {"kind": "reaction", "id": reaction_id, "name": equation, "subtitle": None, "url": reaction_url},
            "known_associations": known_payload,
            "source_panels": panels,
            "model_lens": model,
            "opportunities": self._opportunities(known, model, entity_kind="reaction"),
            "route_view": self._workspace_route_view(
                entity_kind="reaction", entity_id=reaction_id, source_panels=panels,
                known_count=int(known_payload.get("count") or 0), model_lens=model, ui_language=ui_language,
            ),
            "score_note": "模型检索分数用于当前候选集合中的相对优先级；数据库关系仍由证据源确定。" if str(ui_language).lower().startswith("zh") else "Model retrieval scores rank priorities within the current candidate set; database relationships come from evidence sources.",
        }
