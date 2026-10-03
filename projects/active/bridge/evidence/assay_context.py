from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import re
from typing import Any, Iterable


_SUPPORTING_OUTCOMES={"reported_positive","measured_active","quantitative_activity","conversion_observed"}
_CONTRADICTING_OUTCOMES={"inactive","below_detection","no_conversion"}

_METAL_ALIASES={
    "magnesium":"Mg2+","mg2+":"Mg2+","mg(2+)":"Mg2+","mgcl2":"Mg2+",
    "manganese":"Mn2+","mn2+":"Mn2+","mn(2+)":"Mn2+","mncl2":"Mn2+",
    "zinc":"Zn2+","zn2+":"Zn2+","zn(2+)":"Zn2+","zncl2":"Zn2+",
    "iron(ii)":"Fe2+","fe2+":"Fe2+","fe(2+)":"Fe2+","fecl2":"Fe2+",
    "calcium":"Ca2+","ca2+":"Ca2+","ca(2+)":"Ca2+","cacl2":"Ca2+",
    "cobalt":"Co2+","co2+":"Co2+","co(2+)":"Co2+","cocl2":"Co2+",
    "nickel":"Ni2+","ni2+":"Ni2+","ni(2+)":"Ni2+","nicl2":"Ni2+",
}


@dataclass(frozen=True)
class NumericInterval:
    """Closed interval for one explicitly reported assay quantity."""

    lower: float
    upper: float
    unit: str

    def __post_init__(self) -> None:
        lo=float(self.lower); hi=float(self.upper)
        if lo>hi:
            raise ValueError("interval lower must not exceed upper")
        object.__setattr__(self,"lower",lo)
        object.__setattr__(self,"upper",hi)
        object.__setattr__(self,"unit",str(self.unit))

    @classmethod
    def point(cls,value: float,unit: str) -> "NumericInterval":
        x=float(value)
        return cls(x,x,unit)

    def contains(self,value: float) -> bool:
        x=float(value)
        return self.lower <= x <= self.upper

    def overlaps(self,other: "NumericInterval") -> bool:
        if self.unit != other.unit:
            return False
        return max(self.lower,other.lower) <= min(self.upper,other.upper)

    def to_dict(self) -> dict[str,Any]:
        return asdict(self)


@dataclass(frozen=True)
class AssayContext:
    """Sparse experimental context.

    Only dimensions explicitly observed for one assay belong here. Missing
    dimensions stay absent; they are never filled from enzyme-level annotations.
    """

    ph: NumericInterval | None = None
    temperature_c: NumericInterval | None = None
    cofactors: tuple[str,...] = ()
    substrate_concentration: tuple[NumericInterval,...] = ()
    enzyme_concentration: tuple[NumericInterval,...] = ()
    incubation_time_s: tuple[NumericInterval,...] = ()
    raw_conditions: tuple[str,...] = ()

    @property
    def observed_dimensions(self) -> tuple[str,...]:
        out=[]
        if self.ph is not None: out.append("ph")
        if self.temperature_c is not None: out.append("temperature")
        if self.cofactors: out.append("cofactors")
        if self.substrate_concentration: out.append("substrate_concentration")
        if self.enzyme_concentration: out.append("enzyme_concentration")
        if self.incubation_time_s: out.append("incubation_time")
        return tuple(out)

    def to_dict(self) -> dict[str,Any]:
        return {
            "ph":None if self.ph is None else self.ph.to_dict(),
            "temperature_c":None if self.temperature_c is None else self.temperature_c.to_dict(),
            "cofactors":list(self.cofactors),
            "substrate_concentration":[x.to_dict() for x in self.substrate_concentration],
            "enzyme_concentration":[x.to_dict() for x in self.enzyme_concentration],
            "incubation_time_s":[x.to_dict() for x in self.incubation_time_s],
            "raw_conditions":list(self.raw_conditions),
            "observed_dimensions":list(self.observed_dimensions),
        }


@dataclass(frozen=True)
class AssayObservation:
    observation_id: str
    enzyme_id: str
    reaction_id: str
    context: AssayContext
    outcome: str
    source_scope: str
    source_uri: str | None
    source_record: str
    evidence_texts: tuple[str,...]
    target_binding_status: str

    def __post_init__(self) -> None:
        if self.outcome not in (_SUPPORTING_OUTCOMES|_CONTRADICTING_OUTCOMES|{"unknown"}):
            raise ValueError(f"unsupported assay outcome: {self.outcome}")
        if self.source_scope not in {"pair_assay","enzyme_annotation","reaction_annotation"}:
            raise ValueError(f"unsupported source scope: {self.source_scope}")

    @property
    def is_pair_specific(self) -> bool:
        return self.source_scope=="pair_assay"

    def to_dict(self) -> dict[str,Any]:
        return {
            "observation_id":self.observation_id,
            "enzyme_id":self.enzyme_id,
            "reaction_id":self.reaction_id,
            "context":self.context.to_dict(),
            "outcome":self.outcome,
            "source_scope":self.source_scope,
            "source_uri":self.source_uri,
            "source_record":self.source_record,
            "evidence_texts":list(self.evidence_texts),
            "target_binding_status":self.target_binding_status,
        }


@dataclass(frozen=True)
class ContextAssessment:
    """Non-scalar relation between a requested condition and assay evidence."""

    status: str
    matched_support_observations: tuple[str,...]
    matched_contradiction_observations: tuple[str,...]
    relevant_observation_count: int
    requested_dimensions: tuple[str,...]
    interpretation: str

    def to_dict(self) -> dict[str,Any]:
        return asdict(self)


def assay_observation_from_dict(row: dict[str,Any]) -> AssayObservation:
    def interval(value: Any) -> NumericInterval | None:
        if not isinstance(value,dict):
            return None
        return NumericInterval(
            float(value["lower"]),float(value["upper"]),str(value["unit"])
        )

    context=dict(row.get("context") or {})
    return AssayObservation(
        observation_id=str(row.get("observation_id") or ""),
        enzyme_id=str(row.get("enzyme_id") or ""),
        reaction_id=str(row.get("reaction_id") or ""),
        context=AssayContext(
            ph=interval(context.get("ph")),
            temperature_c=interval(context.get("temperature_c")),
            cofactors=tuple(str(x) for x in context.get("cofactors") or [] if str(x)),
            substrate_concentration=tuple(
                x for x in (interval(v) for v in context.get("substrate_concentration") or []) if x is not None
            ),
            enzyme_concentration=tuple(
                x for x in (interval(v) for v in context.get("enzyme_concentration") or []) if x is not None
            ),
            incubation_time_s=tuple(
                x for x in (interval(v) for v in context.get("incubation_time_s") or []) if x is not None
            ),
            raw_conditions=tuple(str(x) for x in context.get("raw_conditions") or [] if str(x)),
        ),
        outcome=str(row.get("outcome") or "unknown"),
        source_scope=str(row.get("source_scope") or "pair_assay"),
        source_uri=row.get("source_uri"),
        source_record=str(row.get("source_record") or ""),
        evidence_texts=tuple(str(x) for x in row.get("evidence_texts") or [] if str(x)),
        target_binding_status=str(row.get("target_binding_status") or ""),
    )


def context_from_target_conditions(value: dict[str,Any] | None) -> AssayContext:
    row=dict(value or {})
    ph=row.get("ph")
    temperature=row.get("temperature_c")
    raw_cofactors=[str(x).strip() for x in row.get("cofactors") or [] if str(x).strip()]
    cofactors=[]
    for raw in raw_cofactors:
        normalized=_normalized_cofactors(raw)
        if normalized:
            cofactors.extend(normalized)
        else:
            cofactors.append(raw)
    return AssayContext(
        ph=None if ph is None else NumericInterval.point(float(ph),"pH"),
        temperature_c=(
            None if temperature is None
            else NumericInterval.point(float(temperature),"°C")
        ),
        cofactors=tuple(sorted(set(cofactors))),
    )


def marts_target_id(enzyme_id: str,reaction_signature: str) -> str:
    digest=hashlib.sha256(str(reaction_signature).encode("utf-8")).hexdigest()[:12]
    return f"marts:{str(enzyme_id)}:{digest}"


def _normalized_cofactors(text: str) -> tuple[str,...]:
    low=str(text).lower().replace(" ","")
    found=set()
    for token,label in _METAL_ALIASES.items():
        if token in low:
            found.add(label)
    return tuple(sorted(found))


def _quantity(fact: dict[str,Any],unit_names: set[str]) -> list[float]:
    values=[]
    for q in fact.get("explicit_quantities") or []:
        unit=str(q.get("unit") or "").strip()
        if unit not in unit_names:
            continue
        try:
            values.append(float(str(q.get("value") or "").replace("−","-").replace("–","-")))
        except ValueError:
            continue
    return values


def _seconds(value: float,unit: str) -> float | None:
    u=str(unit).strip().lower()
    if u in {"s","sec","secs","second","seconds"}: return float(value)
    if u in {"min","mins","minute","minutes"}: return float(value)*60.0
    if u in {"h","hr","hrs","hour","hours"}: return float(value)*3600.0
    return None


def context_from_source_bound_facts(facts: Iterable[dict[str,Any]]) -> AssayContext:
    ph=None
    temp=None
    cofactors=set()
    substrate=[]
    enzyme=[]
    time=[]
    raw=[]
    for fact in facts:
        kind=str(fact.get("type") or "")
        text=str(fact.get("evidence_text") or "").strip()
        if text:
            raw.append(text)
        if kind in {"pH","buffer"}:
            values=_quantity(fact,{"pH"})
            if values:
                ph=NumericInterval.point(values[0],"pH")
        if kind=="temperature":
            values=_quantity(fact,{"°C","° C","degC","degree C","degrees C"})
            if values:
                temp=NumericInterval.point(values[0],"°C")
        if kind=="metal_or_cofactor":
            cofactors.update(_normalized_cofactors(text))
        target=substrate if kind=="substrate_concentration" else enzyme if kind=="enzyme_concentration" else None
        if target is not None:
            for q in fact.get("explicit_quantities") or []:
                try:
                    value=float(q.get("value"))
                except (TypeError,ValueError):
                    continue
                unit=str(q.get("unit") or "")
                if unit in {"M","mM","µM","μM","uM","nM"}:
                    target.append(NumericInterval.point(value,unit))
        if kind=="incubation_time":
            for q in fact.get("explicit_quantities") or []:
                try:
                    value=float(q.get("value"))
                except (TypeError,ValueError):
                    continue
                sec=_seconds(value,str(q.get("unit") or ""))
                if sec is not None:
                    time.append(NumericInterval.point(sec,"s"))
    return AssayContext(
        ph=ph,
        temperature_c=temp,
        cofactors=tuple(sorted(cofactors)),
        substrate_concentration=tuple(substrate),
        enzyme_concentration=tuple(enzyme),
        incubation_time_s=tuple(time),
        raw_conditions=tuple(dict.fromkeys(raw)),
    )


def _dimension_matches(observed: AssayContext,target: AssayContext) -> bool:
    requested=target.observed_dimensions
    if not requested:
        return False
    if target.ph is not None:
        if observed.ph is None or not observed.ph.overlaps(target.ph):
            return False
    if target.temperature_c is not None:
        if observed.temperature_c is None or not observed.temperature_c.overlaps(target.temperature_c):
            return False
    if target.cofactors:
        if not observed.cofactors:
            return False
        if not set(target.cofactors).issubset(set(observed.cofactors)):
            return False
    # Concentration/time constraints are intentionally not compared across units
    # here. They remain first-class observations until a unit-normalization
    # contract is explicitly enabled.
    if target.substrate_concentration or target.enzyme_concentration or target.incubation_time_s:
        return False
    return True


def assess_pair_context(
    observations: Iterable[AssayObservation],
    target: AssayContext,
    *,
    enzyme_id: str,
    reaction_id: str,
) -> ContextAssessment:
    relevant=[
        x for x in observations
        if x.enzyme_id==str(enzyme_id)
        and x.reaction_id==str(reaction_id)
        and x.is_pair_specific
    ]
    support=[]
    contradiction=[]
    for obs in relevant:
        if not _dimension_matches(obs.context,target):
            continue
        if obs.outcome in _SUPPORTING_OUTCOMES:
            support.append(obs.observation_id)
        elif obs.outcome in _CONTRADICTING_OUTCOMES:
            contradiction.append(obs.observation_id)
    if support and contradiction:
        status="conflicting"
    elif contradiction:
        status="contradicted"
    elif support:
        status="supported"
    else:
        status="unresolved"
    return ContextAssessment(
        status=status,
        matched_support_observations=tuple(sorted(set(support))),
        matched_contradiction_observations=tuple(sorted(set(contradiction))),
        relevant_observation_count=len(relevant),
        requested_dimensions=target.observed_dimensions,
        interpretation=(
            "pair-specific assay constraint; absence or condition mismatch remains unresolved, "
            "and only an explicitly matched inactive/below-detection assay can contradict"
        ),
    )


def build_source_bound_assay_observations(
    extraction_rows: Iterable[dict[str,Any]],
    target_map: dict[str,tuple[str,str]],
) -> list[AssayObservation]:
    """Materialize only strongly scoped publication assay contexts.

    Conservative promotion gate:
      * paragraph must be classified as catalytic_assay;
      * source span and experiment scope must be resolved;
      * exactly one explicit target descriptor must be linked;
      * the target must map to one current canonical pair.

    This intentionally leaves broad/mixed paragraphs as candidate evidence only.
    """

    out=[]
    for row in extraction_rows:
        if str(row.get("status") or "")!="ok":
            continue
        if str(row.get("paragraph_role") or "")!="catalytic_assay":
            continue
        by_target: dict[str,list[dict[str,Any]]]={}
        for fact in row.get("facts") or []:
            if not bool(fact.get("source_span_verified")) or not bool(fact.get("scope_resolved")):
                continue
            targets=list(fact.get("target_ids") or [])
            if len(targets)!=1:
                continue
            target=str(targets[0])
            if target not in target_map:
                continue
            by_target.setdefault(target,[]).append(fact)
        for target,facts in sorted(by_target.items()):
            enzyme_id,reaction_id=target_map[target]
            outcome="reported_positive"
            if any(str(f.get("type") or "")=="inactive_or_detection_limit" for f in facts):
                # A negative observation is only promoted if the extraction was
                # explicitly typed as inactive/detection-limit and scoped to this pair.
                outcome="below_detection"
            context=context_from_source_bound_facts(facts)
            if not context.observed_dimensions:
                continue
            payload="\x1f".join([
                str(row.get("source_key") or ""),
                target,
                outcome,
                json.dumps(context.to_dict(),sort_keys=True,ensure_ascii=False),
            ])
            oid="assay_ctx_"+hashlib.sha256(payload.encode("utf-8")).hexdigest()[:20]
            out.append(AssayObservation(
                observation_id=oid,
                enzyme_id=str(enzyme_id),
                reaction_id=str(reaction_id),
                context=context,
                outcome=outcome,
                source_scope="pair_assay",
                source_uri=str(row.get("source_url") or "") or None,
                source_record=str(row.get("source_key") or ""),
                evidence_texts=tuple(dict.fromkeys(
                    str(f.get("evidence_text") or "").strip()
                    for f in facts if str(f.get("evidence_text") or "").strip()
                )),
                target_binding_status="source_span_and_scope_resolved_single_explicit_target",
            ))
    return out
