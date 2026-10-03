from __future__ import annotations

from dataclasses import asdict, dataclass
import re
from typing import Any, Iterable

from projects.active.fibre.evidence.assay_context import AssayObservation


_RHEA_RE=re.compile(r"RHEA:\d+")
_COFACTOR_RE=re.compile(r"Name=([^;]+)")


@dataclass(frozen=True)
class CatalyticStateEvidence:
    """Sparse enzymology state attached to one canonical enzyme-reaction pair.

    The object records what is observed and at which scope. It deliberately has
    no scalar quality/compatibility score.
    """

    enzyme_id: str
    reaction_id: str
    mechanism_ids: tuple[str,...]
    mechanism_step_count: int
    mechanism_reaction_types: tuple[str,...]
    mechanism_source_evidence: tuple[str,...]
    protein_catalytic_rhea_ids: tuple[str,...]
    protein_cofactors: tuple[str,...]
    protein_active_site_annotations: int
    protein_binding_site_annotations: int
    protein_evidence_tokens: tuple[str,...]
    pair_assay_observation_ids: tuple[str,...]

    @property
    def observed_components(self) -> tuple[str,...]:
        out=[]
        if self.mechanism_ids: out.append("pair_mechanism")
        if self.protein_catalytic_rhea_ids: out.append("protein_catalytic_activity")
        if self.protein_cofactors: out.append("protein_cofactor")
        if self.protein_active_site_annotations: out.append("protein_active_site")
        if self.protein_binding_site_annotations: out.append("protein_binding_site")
        if self.pair_assay_observation_ids: out.append("pair_assay_context")
        return tuple(out)

    def to_dict(self) -> dict[str,Any]:
        row=asdict(self)
        row["observed_components"]=list(self.observed_components)
        row["semantics"]={
            "pair_mechanism":"pair-associated mechanism evidence when present",
            "protein_annotations":"entity-level UniProt evidence; never promoted to pair-specific assay evidence",
            "pair_assay_context":"source-bound pair-specific experimental context",
            "ranking_effect":"none unless a separately validated FIBRE coordinate/constraint grants ordering authority",
        }
        return row


def _property_value(row: dict[str,Any],term: str) -> list[str]:
    out=[]
    for prop in row.get("properties") or []:
        if str(prop.get("term") or "")==term:
            value=str(prop.get("value") or "").strip()
            if value:
                out.append(value)
    return out


def build_pair_catalytic_states(
    correspondence_rows: Iterable[dict[str,Any]],
    mechanism_step_rows: Iterable[dict[str,Any]],
    protein_annotation_rows: Iterable[dict[str,Any]],
    assay_observations: Iterable[AssayObservation]=(),
) -> list[CatalyticStateEvidence]:
    pairs={}
    for row in correspondence_rows:
        if str(row.get("association") or "")!="reported_positive":
            continue
        enzyme=str(row.get("canonical_enzyme_id") or "")
        reaction=str(row.get("canonical_reaction_id") or "")
        if not (enzyme and reaction):
            continue
        key=(enzyme,reaction)
        record=pairs.setdefault(key,{"mechanisms":set()})
        record["mechanisms"].update(_property_value(row,"fibre:MARTS.mechanism_id"))

    mechanism_steps={}
    for row in mechanism_step_rows:
        mid=str(row.get("mechanism_id") or "")
        if not mid:
            continue
        bucket=mechanism_steps.setdefault(mid,{"count":0,"types":set(),"evidence":set()})
        bucket["count"]+=1
        rtype=str(row.get("reaction_type") or "").strip()
        if rtype: bucket["types"].add(rtype)
        evidence=str(row.get("source_evidence") or "").strip()
        if evidence: bucket["evidence"].add(evidence)

    proteins={}
    for row in protein_annotation_rows:
        pid=str(row.get("canonical_subject_id") or "")
        if not pid:
            continue
        p=proteins.setdefault(pid,{
            "rhea":set(),"cofactors":set(),"active":0,"binding":0,"evidence":set(),
        })
        pred=str(row.get("predicate") or "")
        value=str(row.get("value") or "")
        p["evidence"].update(str(x) for x in row.get("evidence") or [] if str(x))
        if pred=="uniprot:CatalyticActivity":
            p["rhea"].update(_RHEA_RE.findall(value))
        elif pred=="uniprot:Cofactor":
            p["cofactors"].update(
                x.strip() for x in _COFACTOR_RE.findall(value) if x.strip()
            )
        elif pred=="uniprot:ActiveSite":
            p["active"]+=1
        elif pred=="uniprot:BindingSite":
            p["binding"]+=1

    assays={}
    for row in assay_observations:
        assays.setdefault((row.enzyme_id,row.reaction_id),set()).add(row.observation_id)

    out=[]
    for enzyme,reaction in sorted(pairs):
        mechanisms=tuple(sorted(pairs[(enzyme,reaction)]["mechanisms"]))
        step_count=0; reaction_types=set(); mechanism_evidence=set()
        for mid in mechanisms:
            bucket=mechanism_steps.get(mid,{})
            step_count+=int(bucket.get("count") or 0)
            reaction_types.update(bucket.get("types") or set())
            mechanism_evidence.update(bucket.get("evidence") or set())
        p=proteins.get(enzyme,{})
        out.append(CatalyticStateEvidence(
            enzyme_id=enzyme,
            reaction_id=reaction,
            mechanism_ids=mechanisms,
            mechanism_step_count=step_count,
            mechanism_reaction_types=tuple(sorted(reaction_types)),
            mechanism_source_evidence=tuple(sorted(mechanism_evidence)),
            protein_catalytic_rhea_ids=tuple(sorted(p.get("rhea") or set())),
            protein_cofactors=tuple(sorted(p.get("cofactors") or set())),
            protein_active_site_annotations=int(p.get("active") or 0),
            protein_binding_site_annotations=int(p.get("binding") or 0),
            protein_evidence_tokens=tuple(sorted(p.get("evidence") or set())),
            pair_assay_observation_ids=tuple(sorted(assays.get((enzyme,reaction),set()))),
        ))
    return out
