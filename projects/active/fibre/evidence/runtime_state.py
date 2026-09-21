from __future__ import annotations

from collections import Counter,defaultdict
import json
from pathlib import Path
import re
from typing import Any

from projects.active.fibre.evidence.assay_context import (
    AssayObservation,
    assay_observation_from_dict,
)


_RHEA_RE=re.compile(r"RHEA:\d+")
_COFACTOR_RE=re.compile(r"Name=([^;]+)")


class EnzymologyEvidenceIndex:
    """Small read-only application index over scoped catalytic evidence.

    It deliberately separates protein-level annotation, reaction-level mechanism
    observations, and pair-specific assay/corroboration. Nothing here changes a
    FIBRE score.
    """

    def __init__(
        self,
        *,
        pair_states: str | Path,
        protein_annotations: str | Path,
        assay_observations: str | Path | None = None,
        cross_source_pairs: str | Path | None = None,
    ) -> None:
        self.proteins: dict[str,dict[str,Any]]={}
        self.reactions: dict[str,dict[str,Any]]={}
        self.pairs: dict[tuple[str,str],dict[str,Any]]={}
        self.assays: dict[tuple[str,str],list[dict[str,Any]]]=defaultdict(list)
        self.cross_source: dict[tuple[str,str],dict[str,Any]]={}

        self._load_proteins(Path(protein_annotations))
        self._load_pairs(Path(pair_states))
        if assay_observations is not None and Path(assay_observations).is_file():
            for row in self._jsonl(Path(assay_observations)):
                self.assays[(str(row["enzyme_id"]),str(row["reaction_id"]))].append(row)
        if cross_source_pairs is not None and Path(cross_source_pairs).is_file():
            import pandas as pd
            frame=pd.read_csv(cross_source_pairs,dtype=str).fillna("")
            for row in frame.to_dict("records"):
                self.cross_source[(str(row["enzyme_id"]),str(row["reaction_id"]))]=row

    @staticmethod
    def _jsonl(path: Path):
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    yield json.loads(line)

    def _load_proteins(self,path: Path) -> None:
        work: dict[str,dict[str,Any]]={}
        for row in self._jsonl(path):
            pid=str(row.get("canonical_subject_id") or "")
            if not pid:
                continue
            bucket=work.setdefault(pid,{
                "predicates":Counter(),"cofactors":set(),"rhea":set(),
                "experimental_tokens":set(),"inferred_tokens":set(),"other_tokens":set(),
                "source_subject_ids":set(),
            })
            pred=str(row.get("predicate") or "")
            value=str(row.get("value") or "")
            bucket["predicates"][pred]+=1
            bucket["source_subject_ids"].add(str(row.get("source_subject_id") or ""))
            if pred=="uniprot:Cofactor":
                bucket["cofactors"].update(x.strip() for x in _COFACTOR_RE.findall(value) if x.strip())
            if pred=="uniprot:CatalyticActivity":
                bucket["rhea"].update(_RHEA_RE.findall(value))
            for token in row.get("evidence") or []:
                token=str(token)
                if token.startswith("ECO:0000269|"):
                    bucket["experimental_tokens"].add(token)
                elif token.startswith(("ECO:0000250|","ECO:0000255|","ECO:0000256|")):
                    bucket["inferred_tokens"].add(token)
                elif token:
                    bucket["other_tokens"].add(token)
        for pid,b in work.items():
            self.proteins[pid]={
                "available":True,
                "source":"UniProtKB",
                "source_subject_ids":sorted(x for x in b["source_subject_ids"] if x),
                "catalytic_activity_annotation_count":int(b["predicates"]["uniprot:CatalyticActivity"]),
                "cofactor_annotation_count":int(b["predicates"]["uniprot:Cofactor"]),
                "active_site_annotation_count":int(b["predicates"]["uniprot:ActiveSite"]),
                "binding_site_annotation_count":int(b["predicates"]["uniprot:BindingSite"]),
                "catalytic_rhea_ids":sorted(b["rhea"]),
                "cofactors":sorted(b["cofactors"]),
                "experimental_evidence_token_count":len(b["experimental_tokens"]),
                "inferred_evidence_token_count":len(b["inferred_tokens"]),
                "other_evidence_token_count":len(b["other_tokens"]),
                "scope":"protein_entity_annotation_not_pair_assay",
            }

    def _load_pairs(self,path: Path) -> None:
        reaction_work: dict[str,dict[str,Any]]={}
        for row in self._jsonl(path):
            enzyme=str(row["enzyme_id"]); reaction=str(row["reaction_id"])
            pair={
                "available":True,
                "mechanism_ids":list(row.get("mechanism_ids") or []),
                "mechanism_step_count":int(row.get("mechanism_step_count") or 0),
                "mechanism_reaction_types":list(row.get("mechanism_reaction_types") or []),
                "mechanism_source_evidence":list(row.get("mechanism_source_evidence") or []),
                "assay_observation_ids":list(row.get("pair_assay_observation_ids") or []),
                "scope":"accepted_pair_specific_state",
            }
            self.pairs[(enzyme,reaction)]=pair
            rw=reaction_work.setdefault(reaction,{
                "mechanisms":set(),"types":set(),"evidence":set(),"pair_count":0,
            })
            rw["mechanisms"].update(pair["mechanism_ids"])
            rw["types"].update(pair["mechanism_reaction_types"])
            rw["evidence"].update(pair["mechanism_source_evidence"])
            rw["pair_count"]+=1
        for rid,b in reaction_work.items():
            self.reactions[rid]={
                "available":True,
                "mechanism_ids":sorted(b["mechanisms"]),
                "mechanism_reaction_types":sorted(b["types"]),
                "mechanism_source_evidence":sorted(b["evidence"]),
                "known_pair_count":int(b["pair_count"]),
                "scope":"reaction_state_aggregated_from_accepted_pair_mechanism_records",
            }

    def assay_observations(
        self,protein_id: str,reaction_id: str
    ) -> tuple[AssayObservation,...]:
        key=(str(protein_id),str(reaction_id))
        return tuple(
            assay_observation_from_dict(row)
            for row in self.assays.get(key,[])
        )

    def state(self,protein_id: str,reaction_id: str) -> dict[str,Any]:
        key=(str(protein_id),str(reaction_id))
        pair=dict(self.pairs.get(key) or {
            "available":False,
            "scope":"no_pair_specific_state_observed",
        })
        if self.assays.get(key):
            pair["assay_context_observations"]=[
                {
                    "observation_id":str(row.get("observation_id") or ""),
                    "outcome":str(row.get("outcome") or ""),
                    "context":dict(row.get("context") or {}),
                    "source_uri":row.get("source_uri"),
                }
                for row in self.assays[key]
            ]
        cross=self.cross_source.get(key)
        if cross is not None:
            pair["cross_source_corroboration"]={
                "status":str(cross.get("status") or ""),
                "overlap_rhea_ids":[x for x in str(cross.get("overlap_rhea_ids") or "").split("|") if x],
                "independent_experimental_support":str(
                    cross.get("independent_experimental_support") or ""
                ).strip().lower() in {"true","1","yes"},
                "ranking_effect":False,
            }
        protein=dict(self.proteins.get(str(protein_id)) or {
                "available":False,
                "scope":"protein_state_unresolved",
            })
        reaction=dict(self.reactions.get(str(reaction_id)) or {
                "available":False,
                "scope":"reaction_mechanism_state_unresolved",
            })
        components=[]
        if pair.get("mechanism_ids"): components.append("pair_mechanism")
        if protein.get("catalytic_activity_annotation_count"): components.append("protein_catalytic_activity")
        if protein.get("cofactor_annotation_count"): components.append("protein_cofactor")
        if protein.get("active_site_annotation_count"): components.append("protein_active_site")
        if protein.get("binding_site_annotation_count"): components.append("protein_binding_site")
        if pair.get("assay_context_observations"): components.append("pair_assay_context")
        if pair.get("cross_source_corroboration"): components.append("cross_source_corroboration")
        return {
            "schema":"fibre-enzymology-state-v1",
            "observed_components":components,
            "protein":protein,
            "reaction":reaction,
            "pair":pair,
            "ranking_effect":False,
            "interpretation":(
                "scoped enzymology evidence attached to the same candidate; missing "
                "evidence is unresolved and does not change the FIBRE score"
            ),
        }

    def status(self) -> dict[str,Any]:
        negative_outcomes={"inactive","below_detection","no_conversion"}
        negative_assays=sum(
            1
            for rows in self.assays.values()
            for row in rows
            if str(row.get("outcome") or "") in negative_outcomes
        )
        return {
            "schema":"fibre-enzymology-state-index-v1",
            "protein_states":len(self.proteins),
            "reaction_states":len(self.reactions),
            "pair_states":len(self.pairs),
            "pair_assay_states":len(self.assays),
            "pair_specific_negative_assay_observations":int(negative_assays),
            "cross_source_pair_rows":len(self.cross_source),
            "ranking_effect":False,
            "conditional_eligibility_censor_supported":True,
            "current_dataset_censor_can_fire":bool(negative_assays),
        }
