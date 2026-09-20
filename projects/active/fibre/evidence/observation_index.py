"""Standards-aligned sparse biological observations for FIBRE.

This module deliberately does *not* define an enzymology ontology or an evidence
ranking.  It is a thin index over source observations:

- entity/reaction identity,
- sparse typed properties,
- per-property/source provenance,
- an optional mapping into the current FIBRE atlas.

Community terms are reused only when the source semantics match.  Missing
properties remain absent rather than being imputed or converted into negatives.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from typing import Any, Iterable

import pandas as pd


@dataclass(frozen=True)
class PropertyObservation:
    term: str
    value: str | float | int | bool
    source_field: str
    unit: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SourceProvenance:
    dataset: str
    source_row: int
    source_uri: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class EntityAnnotationObservation:
    subject_kind: str
    source_subject_id: str
    canonical_subject_id: str | None
    predicate: str
    value: str
    evidence: tuple[str, ...]
    provenance: SourceProvenance

    def to_dict(self) -> dict[str, Any]:
        row=asdict(self)
        row["provenance"]=self.provenance.to_dict()
        return row


@dataclass(frozen=True)
class MechanismStepObservation:
    mechanism_id: str
    step_index: int
    reaction_type: str
    substrate_name: str
    substrate_smiles: str
    product_name: str
    product_smiles: str
    step_reaction_signature: str
    source_evidence: str | None
    provenance: SourceProvenance

    def to_dict(self) -> dict[str, Any]:
        row=asdict(self)
        row["provenance"]=self.provenance.to_dict()
        return row


@dataclass(frozen=True)
class CorrespondenceObservation:
    observation_id: str
    enzyme_source_id: str
    reaction_source_signature: str
    canonical_enzyme_id: str | None
    canonical_reaction_id: str | None
    association: str
    properties: tuple[PropertyObservation, ...]
    provenance: tuple[SourceProvenance, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "observation_id": self.observation_id,
            "enzyme_source_id": self.enzyme_source_id,
            "reaction_source_signature": self.reaction_source_signature,
            "canonical_enzyme_id": self.canonical_enzyme_id,
            "canonical_reaction_id": self.canonical_reaction_id,
            "association": self.association,
            "properties": [x.to_dict() for x in self.properties],
            "provenance": [x.to_dict() for x in self.provenance],
        }


def _stable_id(source_row: int, row: pd.Series) -> str:
    payload="\x1f".join(
        [
            str(source_row),
            str(row.get("enzyme_id", "")),
            str(row.get("reaction_signature", "")),
            str(row.get("publication", "")),
        ]
    )
    return "marts_obs_" + hashlib.sha256(payload.encode("utf-8")).hexdigest()[:20]


def _add(
    out: list[PropertyObservation],
    *,
    term: str,
    value: Any,
    source_field: str,
    unit: str | None = None,
) -> None:
    if value is None:
        return
    if isinstance(value, float) and pd.isna(value):
        return
    if isinstance(value, str):
        value=value.strip()
        if not value:
            return
    out.append(
        PropertyObservation(
            term=term,
            value=value,
            source_field=source_field,
            unit=unit,
        )
    )


def build_marts_observations(
    raw_pairs: pd.DataFrame,
    protein_entities: pd.DataFrame,
    reaction_entities: pd.DataFrame,
) -> list[CorrespondenceObservation]:
    """Project MARTS source rows into sparse standard-aligned observations.

    MARTS currently provides literature-backed enzyme/reaction associations and
    molecular identity/context metadata, but it does not materialise assay pH,
    temperature, kinetic constants, detection limits, or concentration/time
    courses.  Those fields are therefore *absent*, not filled with defaults.

    The current canonical FIBRE ids are attached only when exact sequence and
    exact reaction-signature mappings exist.
    """
    raw=raw_pairs.fillna("").copy()
    proteins=protein_entities.fillna("").copy()
    reactions=reaction_entities.fillna("").copy()

    seq_to_pid: dict[str, str] = {}
    for row in proteins.itertuples(index=False):
        seq=str(row.sequence)
        if seq in seq_to_pid and seq_to_pid[seq] != str(row.protein_id):
            raise ValueError("canonical protein sequence mapping is ambiguous")
        seq_to_pid[seq]=str(row.protein_id)

    sig_to_rid: dict[str, str] = {}
    for row in reactions.itertuples(index=False):
        sig=str(row.reaction_signature)
        if sig in sig_to_rid and sig_to_rid[sig] != str(row.reaction_id):
            raise ValueError("canonical reaction signature mapping is ambiguous")
        sig_to_rid[sig]=str(row.reaction_id)

    observations: list[CorrespondenceObservation] = []
    for source_row, row in raw.iterrows():
        props: list[PropertyObservation] = []

        # EnzymeML terms: semantics match exactly.
        _add(props,term="enzymeml:Protein.name",value=row.get("enzyme_name"),source_field="enzyme_name")
        _add(props,term="enzymeml:Protein.sequence",value=row.get("sequence"),source_field="sequence")
        if str(row.get("uniprot_id","")).strip():
            _add(
                props,
                term="enzymeml:Protein.references",
                value=f"https://www.uniprot.org/uniprotkb/{str(row['uniprot_id']).strip()}",
                source_field="uniprot_id",
            )

        # STRENDA distinguishes source organism from production/expression host.
        _add(
            props,
            term="strenda:Biocatalyst.origin_organism",
            value=row.get("species"),
            source_field="species",
        )

        # EnzymeML models participants structurally as ReactionElement -> SmallMolecule.
        # MARTS is flat, so these convenience fields remain explicit FIBRE
        # extensions rather than inventing nonexistent EnzymeML scalar slots.
        _add(props,term="fibre:reaction.reactant_name",value=row.get("substrate_name"),source_field="substrate_name")
        _add(props,term="fibre:reaction.reactant_smiles",value=row.get("substrate_smiles"),source_field="substrate_smiles")
        _add(props,term="fibre:reaction.product_name",value=row.get("product_name"),source_field="product_name")
        _add(props,term="fibre:reaction.product_smiles",value=row.get("product_smiles"),source_field="product_smiles")

        # FIBRE/MARTS-specific annotations stay explicitly project-namespaced.
        _add(props,term="fibre:MARTS.terpene_type",value=row.get("terpene_type"),source_field="terpene_type")
        _add(props,term="fibre:MARTS.tps_class",value=row.get("tps_class"),source_field="tps_class")
        mechanism_id=str(row.get("mechanism_marts_id","")).strip()
        if mechanism_id and mechanism_id != "no_mechanism":
            _add(props,term="fibre:MARTS.mechanism_id",value=mechanism_id,source_field="mechanism_marts_id")

        publication=str(row.get("publication","")).strip() or None
        observations.append(
            CorrespondenceObservation(
                observation_id=_stable_id(int(source_row),row),
                enzyme_source_id=str(row.get("enzyme_id","")),
                reaction_source_signature=str(row.get("reaction_signature","")),
                canonical_enzyme_id=seq_to_pid.get(str(row.get("sequence",""))),
                canonical_reaction_id=sig_to_rid.get(str(row.get("reaction_signature",""))),
                # This means only that MARTS reports the pair as a positive
                # association.  It does not pretend a quantitative assay was
                # materialised in the current dataset.
                association="reported_positive",
                properties=tuple(props),
                provenance=(
                    SourceProvenance(
                        dataset="MARTS",
                        source_row=int(source_row),
                        source_uri=publication,
                    ),
                ),
            )
        )
    return observations


def build_uniprot_entity_annotations(
    uniprot_table: pd.DataFrame,
    protein_entities: pd.DataFrame,
    extra_subject_map: dict[str, str] | None = None,
) -> list[EntityAnnotationObservation]:
    """Map structured UniProt annotations to current proteins where possible.

    Raw UniProt annotation text and ECO/source tokens are preserved.  This
    function does not infer a quality score from reviewed status or ECO codes.
    """
    import re

    alias_to_pid: dict[str, str] = {}
    for row in protein_entities.fillna("").itertuples(index=False):
        for token in re.split(r"[;,| ]+",str(row.aliases)):
            token=token.strip()
            if not token:
                continue
            previous=alias_to_pid.get(token)
            if previous is not None and previous != str(row.protein_id):
                raise ValueError(f"protein alias maps to multiple canonical ids: {token}")
            alias_to_pid[token]=str(row.protein_id)
    for token,pid in (extra_subject_map or {}).items():
        previous=alias_to_pid.get(str(token))
        if previous is not None and previous != str(pid):
            raise ValueError(f"explicit subject map conflicts with protein alias: {token}")
        alias_to_pid[str(token)]=str(pid)

    field_map={
        "Catalytic activity":"uniprot:CatalyticActivity",
        "Cofactor":"uniprot:Cofactor",
        "Active site":"uniprot:ActiveSite",
        "Binding site":"uniprot:BindingSite",
        "EC number":"uniprot:ECNumber",
        "Reviewed":"uniprot:ReviewedStatus",
    }
    evidence_re=re.compile(r"ECO:\d+\|[^\"},;\s]+")
    out=[]
    for source_row,row in uniprot_table.fillna("").iterrows():
        accession=str(row.get("Entry","")).strip()
        if not accession:
            continue
        canonical=alias_to_pid.get(accession)
        uri=f"https://www.uniprot.org/uniprotkb/{accession}/entry"
        for field,predicate in field_map.items():
            value=str(row.get(field,"")).strip()
            if not value:
                continue
            evidence=tuple(sorted(set(evidence_re.findall(value))))
            out.append(
                EntityAnnotationObservation(
                    subject_kind="protein",
                    source_subject_id=accession,
                    canonical_subject_id=canonical,
                    predicate=predicate,
                    value=value,
                    evidence=evidence,
                    provenance=SourceProvenance(
                        dataset="UniProtKB",
                        source_row=int(source_row),
                        source_uri=uri,
                    ),
                )
            )
    return out


def build_marts_mechanism_steps(
    mechanism_steps: pd.DataFrame,
) -> list[MechanismStepObservation]:
    """Preserve MARTS step-level mechanism observations without re-scoring them."""
    rows=[]
    for source_row,row in mechanism_steps.fillna("").iterrows():
        mechanism_id=str(row.get("Mechanism_marts_id","")).strip()
        if not mechanism_id or mechanism_id == "no_mechanism":
            continue
        evidence=str(row.get("Evidence","")).strip() or None
        publication=str(row.get("Publication","")).strip() or None
        step_index_raw=str(row.get("step_index","")).strip()
        try:
            step_index=int(step_index_raw)
        except ValueError as exc:
            raise ValueError(f"invalid MARTS mechanism step_index: {step_index_raw!r}") from exc
        rows.append(
            MechanismStepObservation(
                mechanism_id=mechanism_id,
                step_index=step_index,
                reaction_type=str(row.get("Reaction_type","")).strip(),
                substrate_name=str(row.get("Substrate_name","")).strip(),
                substrate_smiles=str(row.get("Substrate_smiles","")).strip(),
                product_name=str(row.get("Product_name","")).strip(),
                product_smiles=str(row.get("Product_smiles","")).strip(),
                step_reaction_signature=str(row.get("step_reaction_signature","")).strip(),
                source_evidence=evidence,
                provenance=SourceProvenance(
                    dataset="MARTS mechanism steps",
                    source_row=int(source_row),
                    source_uri=publication,
                ),
            )
        )
    return rows


def canonical_positive_pairs(
    observations: Iterable[CorrespondenceObservation],
) -> set[tuple[str, str]]:
    """Pair-level projection used by the current canonical FIBRE field."""
    return {
        (x.canonical_enzyme_id,x.canonical_reaction_id)
        for x in observations
        if x.association == "reported_positive"
        and x.canonical_enzyme_id is not None
        and x.canonical_reaction_id is not None
    }


def observation_property_terms(
    observations: Iterable[CorrespondenceObservation],
) -> set[str]:
    return {p.term for x in observations for p in x.properties}


def dumps_observation(observation: CorrespondenceObservation) -> str:
    return json.dumps(observation.to_dict(),ensure_ascii=False,sort_keys=True)
