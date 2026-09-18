from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd
from projects.active.terpene_screening.prepare_marts_dataset import reaction_signature

ROOT = Path(__file__).resolve().parents[2]
PROTEIN_ENTITIES = ROOT / "data/terpene_marts_adaptation/protein_entities.csv"
REACTION_ENTITIES = ROOT / "data/terpene_marts_adaptation/reaction_entities.csv"
PROTEIN_COVERAGE = ROOT / "data/terpene_multiresolution_protein_geometry_v4/coverage.csv"
REACTION_COVERAGE = ROOT / "data/terpene_multiresolution_reaction_geometry_v1/coverage.csv"
REACTION_CENTER_COVERAGE = ROOT / "data/terpene_multiresolution_reaction_geometry_v2/coverage.csv"
DEPLOYMENT_V2_REACTIONS = ROOT / "data/terpene_correspondence_deployment_atlas_v2/reaction_entities.csv"


def _split_aliases(value: str) -> list[str]:
    raw = str(value or "").strip()
    if not raw:
        return []
    for sep in (";", ",", "|"):
        raw = raw.replace(sep, " ")
    return [item.strip() for item in raw.split() if item.strip()]


class ObservationInventory:
    """Read-only lookup of measurements materialised in the versioned MARTS atlas."""

    def __init__(self) -> None:
        proteins = pd.read_csv(PROTEIN_ENTITIES, dtype=str).fillna("")
        reactions = pd.read_csv(REACTION_ENTITIES, dtype=str).fillna("")
        pcov = pd.read_csv(PROTEIN_COVERAGE)
        rcov = pd.read_csv(REACTION_COVERAGE)
        rcenter = pd.read_csv(REACTION_CENTER_COVERAGE)

        self._protein_alias: dict[str, str] = {}
        for row in proteins.itertuples(index=False):
            pid = str(row.protein_id)
            for alias in [pid, *_split_aliases(getattr(row, "aliases", ""))]:
                self._protein_alias.setdefault(alias.casefold(), pid)

        self._reaction_alias: dict[str, str] = {}
        for row in reactions.itertuples(index=False):
            rid = str(row.reaction_id)
            self._reaction_alias.setdefault(rid.casefold(), rid)
            sig = str(getattr(row, "reaction_signature", "") or "").strip()
            smiles = str(getattr(row, "reaction_smiles", "") or "").strip()
            if sig:
                self._reaction_alias.setdefault(sig.casefold(), rid)
            if smiles:
                self._reaction_alias.setdefault(smiles.casefold(), rid)

        if DEPLOYMENT_V2_REACTIONS.is_file():
            deployed = pd.read_csv(DEPLOYMENT_V2_REACTIONS, dtype=str).fillna("")
            for row in deployed.itertuples(index=False):
                rid = str(row.reaction_id)
                aliases = [str(row.primary_alias), *_split_aliases(str(row.aliases))]
                for alias in aliases:
                    if alias:
                        self._reaction_alias.setdefault(alias.casefold(), rid)

        self._protein_measurements: dict[str, set[str]] = {}
        for row in pcov.itertuples(index=False):
            pid = str(row.protein_id)
            m = {"protein_sequence"}
            if bool(row.global_esmc_chordal):
                m.add("global_esmc")
            motif_cols = (
                "typeI_aspartate_esmc_chordal",
                "nse_dte_esmc_chordal",
                "dxdd_esmc_chordal",
                "qw_esmc_chordal",
            )
            if any(bool(getattr(row, col)) for col in motif_cols):
                m.add("family_motif_context")
            if bool(row.whole_3di_diffusion):
                m.update({"resolved_structure", "whole_3di"})
            if bool(row.pocket_local_esmc_chordal):
                m.update({"pocket_detection", "pocket_local_esmc"})
            if bool(row.pocket_3di_diffusion):
                m.update({"resolved_structure", "pocket_detection", "pocket_3di"})
            if bool(row.pocket_ot_diffusion):
                m.update({"resolved_structure", "pocket_detection", "pocket_ot"})
            self._protein_measurements[pid] = m

        center_map = {
            str(row.reaction_id): bool(getattr(row, "center_transition_wasserstein", True))
            for row in rcenter.itertuples(index=False)
        }
        self._reaction_measurements: dict[str, set[str]] = {}
        for row in rcov.itertuples(index=False):
            rid = str(row.reaction_id)
            m = {"reaction_structure"}
            if bool(row.drfp):
                m.add("drfp")
            if bool(row.reactant) and bool(row.product):
                m.add("reactant_product_neighbourhood")
            if center_map.get(rid, False):
                m.update({"atom_mapping", "reaction_center_transition"})
            self._reaction_measurements[rid] = m

    def canonical_protein_id(self, identifier: str) -> str | None:
        return self._protein_alias.get(str(identifier or "").strip().casefold())

    def canonical_reaction_id(self, identifier: str) -> str | None:
        raw = str(identifier or "").strip()
        direct = self._reaction_alias.get(raw.casefold())
        if direct is not None:
            return direct
        if ">>" in raw:
            signature = reaction_signature(raw)
            if signature:
                return self._reaction_alias.get(signature.casefold())
        return None

    def protein_measurements(self, identifier: str) -> set[str]:
        pid = self.canonical_protein_id(identifier)
        return set(self._protein_measurements.get(pid or "", set()))

    def reaction_measurements(self, identifier: str) -> set[str]:
        rid = self.canonical_reaction_id(identifier)
        return set(self._reaction_measurements.get(rid or "", set()))

    def is_reference_protein(self, identifier: str) -> bool:
        pid = self.canonical_protein_id(identifier)
        return bool(pid and pid in self._protein_measurements)

    def is_reference_reaction(self, identifier: str) -> bool:
        rid = self.canonical_reaction_id(identifier)
        return bool(rid and rid in self._reaction_measurements)

    @staticmethod
    def _summarize_candidates(
        identifiers: list[str],
        *,
        canonicalize,
        measurements_for,
        measurement_order: tuple[str, ...],
        entity_kind: str,
        factor_measurements: tuple[str, ...] = (),
        evidence_only_measurements: tuple[str, ...] = (),
        prerequisite_measurements: tuple[str, ...] = (),
    ) -> dict[str, object]:
        unique: list[str] = []
        seen: set[str] = set()
        for value in identifiers:
            value = str(value or "").strip()
            if value and value not in seen:
                unique.append(value)
                seen.add(value)
        rows: list[dict[str, object]] = []
        counts = {name: 0 for name in measurement_order}
        resolved = 0
        for value in unique:
            canonical = canonicalize(value)
            measurements = measurements_for(value) if canonical else set()
            if canonical:
                resolved += 1
            ordered = [name for name in measurement_order if name in measurements]
            for name in ordered:
                counts[name] += 1
            rows.append({
                "candidate_id": value,
                "canonical_id": canonical,
                "reference_entity": bool(canonical),
                "measurements": ordered,
            })
        return {
            "entity_kind": entity_kind,
            "candidate_count": len(unique),
            "reference_resolved_count": resolved,
            "measurement_counts": counts,
            "candidates": rows,
            "factor_measurements": list(factor_measurements),
            "evidence_only_measurements": list(evidence_only_measurements),
            "prerequisite_measurements": list(prerequisite_measurements),
            "policy": (
                "cached reference measurements are reused in the factor geometry at zero query-time acquisition cost; "
                "a missing view remains unobserved rather than becoming negative evidence"
            ),
        }

    def summarize_protein_candidates(self, identifiers: list[str]) -> dict[str, object]:
        return self._summarize_candidates(
            identifiers,
            canonicalize=self.canonical_protein_id,
            measurements_for=self.protein_measurements,
            measurement_order=(
                "protein_sequence", "global_esmc", "family_motif_context",
                "resolved_structure", "whole_3di", "pocket_detection",
                "pocket_local_esmc", "pocket_3di", "pocket_ot",
            ),
            entity_kind="protein",
            factor_measurements=(
                "global_esmc", "family_motif_context", "whole_3di",
                "pocket_local_esmc", "pocket_3di", "pocket_ot",
            ),
            prerequisite_measurements=("protein_sequence", "resolved_structure", "pocket_detection"),
        )

    def summarize_reaction_candidates(self, identifiers: list[str]) -> dict[str, object]:
        return self._summarize_candidates(
            identifiers,
            canonicalize=self.canonical_reaction_id,
            measurements_for=self.reaction_measurements,
            measurement_order=(
                "reaction_structure", "drfp", "reactant_product_neighbourhood",
                "atom_mapping", "reaction_center_transition",
            ),
            entity_kind="reaction",
            factor_measurements=("drfp", "reactant_product_neighbourhood"),
            evidence_only_measurements=("reaction_center_transition",),
            prerequisite_measurements=("reaction_structure", "atom_mapping"),
        )
