from __future__ import annotations

from dataclasses import asdict, replace
import json
import os
import threading
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from drfp import DrfpEncoder
from scipy.sparse import load_npz

from projects.active.fibre.runtime.reaction_similarity import (
    best_match_similarity,
    reaction_features,
)
from projects.active.fibre.runtime.reaction_encoding import canonical_or_raw_reaction
from projects.active.fibre.geometry.extension import (
    attach_query_to_reference,
    query_geodesic_to_reference,
)
from projects.active.fibre.runtime.entities import reaction_signature
from projects.active.fibre.runtime.cli import encode_external_enzymes_with_audit
from projects.active.fibre.geometry.structure import query_structural_view
from projects.active.fibre.geometry.levelset import numerical_level_tolerance, stable_level_ids
from projects.active.fibre.geometry.correspondence import correspondence_state
from projects.active.fibre.geometry.stability import (
    section_update_influence,
    seed_influence,
)
from projects.active.fibre.geometry.stratified import (
    consensus_stratified_resolution,
    mechanistic_chart_resolution,
)
from projects.active.fibre.geometry.partial_relation import (
    partial_correspondence_relation,
)
from projects.active.fibre.geometry.biological_relation import (
    biological_correspondence_relation,
)
from projects.active.fibre.geometry.foundation import product_support_diagnostics
from projects.active.fibre.evidence.assay_context import (
    assess_pair_context,
    context_from_target_conditions,
)
from projects.active.fibre.evidence.runtime_state import EnzymologyEvidenceIndex
from projects.active.fibre.portable.reference_query import PortableReferenceBundle
from projects.active.fibre.application.tps_adapted_coordinate import (
    DEFAULT_MODEL as TPS_SOURCE_MODEL,
    DEFAULT_PROTEIN_FEATURES as TPS_CANONICAL_PROTEIN_FEATURES,
    DEFAULT_REACTION_FEATURES as TPS_CANONICAL_REACTION_FEATURES,
    TPSAdaptedCoordinateProjector,
)
from projects.active.fibre.application.build_full_data import (
    CANONICAL as APPLICATION_CANONICAL,
    PROTEIN_GEOMETRY as APPLICATION_PROTEIN_GEOMETRY,
    REACTION_GEOMETRY as APPLICATION_REACTION_GEOMETRY,
    sha256_file as application_sha256_file,
    tree_sha256 as application_tree_sha256,
)

ROOT = Path(__file__).resolve().parents[3]
ATLAS = ROOT / 'data/terpene_correspondence_deployment_atlas_v2'
OMEGA = ROOT / 'data/terpene_marts_adaptation/marts_pair_folds.csv'
STRUCTURAL_BASIS = ROOT / 'data/terpene_correspondence_structural_basis_v1'
CATALYTIC_CONSENSUS = ROOT / 'data/terpene_catalytic_consensus_geometry_v1'
MECHANISTIC_CHART = ROOT / 'data/terpene_mechanistic_chart_geometry_v1'
PROTEIN_GEOMETRY = ROOT / 'data/terpene_multiresolution_protein_geometry_v4'
STRUCTURAL_WORK_ROOT = ROOT / 'results/starase_navigator_runtime/tmp'
APPLICATION_ROOT = ROOT / 'results/fibre_application/full_data'
TPS_DOMAIN_REFERENCE = APPLICATION_ROOT / 'tps_domain_reference'
TPS_APPLICATION_MANIFEST = APPLICATION_ROOT / 'manifest.json'
CATALYTIC_STATE_INDEX = ROOT / 'results/fibre_catalytic_state_v1/pair_states.jsonl'
PROTEIN_STATE_INDEX = ROOT / 'results/fibre_uniprot_state_v1/protein_annotations.jsonl'
ASSAY_STATE_INDEX = ROOT / 'results/fibre_assay_context_v1/assay_observations.jsonl'
CROSS_SOURCE_STATE_INDEX = ROOT / 'results/fibre_cross_source_catalytic_evidence_v1/pair_evidence.csv'


class CorrespondenceGeometryService:
    """Read-only online realization of the canonical FIBRE correspondence field over the MARTS reference atlas.

    Reference geometry and the production positive registry are loaded once. A
    registered query reads its exact reference geodesic row. A genuinely external
    query is attached to the immutable atlas from query-side molecular observations;
    the reference graph is never rebuilt in an HTTP request.
    """

    def __init__(self, atlas: Path = ATLAS, omega: Path = OMEGA) -> None:
        self.atlas = Path(atlas)
        self.omega_path = Path(omega)
        self.manifest = json.loads((self.atlas / 'manifest.json').read_text())
        self.version = str(self.manifest['version'])
        self.reaction_ell = float(self.manifest['factor_characteristic_lengths']['reaction'])
        self.protein_ell = float(self.manifest['factor_characteristic_lengths']['protein'])

        self.proteins = pd.read_csv(self.atlas / 'protein_entities.csv', dtype=str).fillna('')
        self.reactions = pd.read_csv(self.atlas / 'reaction_entities.csv', dtype=str).fillna('')
        protein_order = pd.read_csv(self.atlas / 'protein_ids.csv', dtype={'protein_id': str})
        protein_order['row'] = pd.to_numeric(protein_order['row']).astype(int)
        reaction_order = pd.read_csv(self.atlas / 'reaction_ids.csv', dtype={'reaction_id': str})
        reaction_order['row'] = pd.to_numeric(reaction_order['row']).astype(int)
        self.protein_ids = protein_order.sort_values('row').protein_id.astype(str).tolist()
        self.reaction_ids = reaction_order.sort_values('row').reaction_id.astype(str).tolist()
        if self.protein_ids != self.proteins.protein_id.astype(str).tolist():
            raise RuntimeError('deployment protein entity order mismatch')
        if self.reaction_ids != self.reactions.reaction_id.astype(str).tolist():
            raise RuntimeError('deployment reaction entity order mismatch')
        self.pi = {value: index for index, value in enumerate(self.protein_ids)}
        self.ri = {value: index for index, value in enumerate(self.reaction_ids)}

        self.protein_primary = []
        self.protein_alias_to_internal: dict[str, str] = {}
        self.sequence_to_internal: dict[str, str] = {}
        for row in self.proteins.itertuples(index=False):
            internal = str(row.protein_id)
            aliases = [value.strip() for value in str(row.aliases).replace(',', ';').split(';') if value.strip()]
            primary = aliases[0] if aliases else internal
            self.protein_primary.append(primary)
            for value in [internal, *aliases]:
                self.protein_alias_to_internal.setdefault(value.casefold(), internal)
            sequence = ''.join(str(row.sequence).split()).upper()
            if sequence:
                self.sequence_to_internal.setdefault(sequence, internal)
        self.protein_primary = np.asarray(self.protein_primary, dtype=object)

        self.reaction_primary = self.reactions.primary_alias.astype(str).to_numpy(dtype=object)
        self.reaction_alias_to_internal: dict[str, str] = {}
        self.signature_to_internal: dict[str, str] = {}
        for row in self.reactions.itertuples(index=False):
            internal = str(row.reaction_id)
            aliases = [value.strip() for value in str(row.aliases).split(';') if value.strip()]
            for value in [internal, str(row.primary_alias), *aliases]:
                self.reaction_alias_to_internal.setdefault(value.casefold(), internal)
            signature = str(row.reaction_signature)
            if signature:
                self.signature_to_internal.setdefault(signature, internal)

        self.Dp = np.load(self.atlas / 'protein_geodesic.npy', mmap_mode='r')
        self.Dr = np.load(self.atlas / 'reaction_geodesic.npy', mmap_mode='r')
        self.Dp2 = np.square(np.asarray(self.Dp, dtype=np.float64))
        self.Dr2 = np.square(np.asarray(self.Dr, dtype=np.float64))
        self.protein_graph = load_npz(self.atlas / 'protein_unit_length_graph.npz').astype(np.float64)
        self.reaction_graph = load_npz(self.atlas / 'reaction_unit_length_graph.npz').astype(np.float64)

        self.protein_global = np.load(self.atlas / 'protein_global_esmc_normalized.npy', mmap_mode='r')
        self.protein_global_available = np.load(self.atlas / 'protein_global_esmc_available.npy').astype(bool)
        self.protein_global_scale = np.load(self.atlas / 'protein_global_esmc_scale.npy').astype(np.float64)
        self.structural_basis = STRUCTURAL_BASIS if (STRUCTURAL_BASIS / 'manifest.json').is_file() else None
        self.protein_whole_3di_available = None
        self.protein_whole_3di_scale = None
        if self.structural_basis is not None:
            self.protein_whole_3di_available = np.load(self.structural_basis / 'whole_3di/available.npy').astype(bool)
            self.protein_whole_3di_scale = np.load(self.structural_basis / 'whole_3di/scale.npy').astype(np.float64)
        self.reaction_drfp = np.load(self.atlas / 'reaction_drfp_normalized.npy', mmap_mode='r')
        self.reaction_drfp_available = np.load(self.atlas / 'reaction_drfp_available.npy').astype(bool)
        self.reaction_drfp_scale = np.load(self.atlas / 'reaction_drfp_scale.npy').astype(np.float64)

        self.reference_reaction_features = [reaction_features(value) for value in self.reactions.reaction_smiles.astype(str)]
        self.side_basis: dict[str, dict[str, np.ndarray]] = {}
        for name in ('reactant', 'product'):
            self.side_basis[name] = {
                'weighted_transition': np.load(self.atlas / f'reaction_{name}_weighted_transition.npy', mmap_mode='r'),
                'weighted_norm': np.load(self.atlas / f'reaction_{name}_weighted_norm.npy').astype(np.float64),
                'stationary': np.load(self.atlas / f'reaction_{name}_stationary.npy').astype(np.float64),
                'available': np.load(self.atlas / f'reaction_{name}_available.npy').astype(bool),
                'scale': np.load(self.atlas / f'reaction_{name}_scale.npy').astype(np.float64),
            }

        pairs = pd.read_csv(self.omega_path, dtype=str).fillna('').drop_duplicates(['rhea_id', 'Entry'])
        mapped = [
            (self.ri[str(row.rhea_id)], self.pi[str(row.Entry)])
            for row in pairs.itertuples(index=False)
            if str(row.rhea_id) in self.ri and str(row.Entry) in self.pi
        ]
        if not mapped:
            raise RuntimeError('production correspondence registry is empty')
        self.positive_pairs = np.asarray(mapped, dtype=np.int64)
        self.r_support = np.unique(self.positive_pairs[:, 0])
        self.e_support = np.unique(self.positive_pairs[:, 1])
        self.reaction_marginal_sq = np.min(self.Dr2[:, self.r_support], axis=1)
        self.protein_marginal_sq = np.min(self.Dp2[:, self.e_support], axis=1)
        self._protein_encode_lock = threading.RLock()
        self._seed_stability_lock = threading.RLock()
        self._seed_reference_state = None
        self._application_lock = threading.RLock()
        self.application_manifest: dict[str, Any] | None = None
        self.application_load_error: str | None = None
        self.tps_domain_bundle: PortableReferenceBundle | None = None
        self._tps_projector: TPSAdaptedCoordinateProjector | None = None
        self._load_stratified_geometry()
        self._load_application_profile()
        self._load_enzymology_evidence()

    def _load_enzymology_evidence(self) -> None:
        self.enzymology_evidence: EnzymologyEvidenceIndex | None = None
        self.enzymology_evidence_error: str | None = None
        required=(CATALYTIC_STATE_INDEX,PROTEIN_STATE_INDEX)
        if not all(path.is_file() for path in required):
            self.enzymology_evidence_error='enzymology_evidence_assets_missing'
            return
        try:
            self.enzymology_evidence=EnzymologyEvidenceIndex(
                pair_states=CATALYTIC_STATE_INDEX,
                protein_annotations=PROTEIN_STATE_INDEX,
                assay_observations=ASSAY_STATE_INDEX,
                cross_source_pairs=CROSS_SOURCE_STATE_INDEX,
            )
        except Exception as exc:
            self.enzymology_evidence_error=f'{type(exc).__name__}: {exc}'
            self.enzymology_evidence=None

    def enzymology_evidence_status(self) -> dict[str,Any]:
        if self.enzymology_evidence is None:
            return {
                'schema':'fibre-enzymology-state-index-v1',
                'status':'unavailable',
                'load_error':self.enzymology_evidence_error,
                'ranking_effect':False,
            }
        return {
            'status':'ready',
            **self.enzymology_evidence.status(),
        }

    def _enzymology_state(self,protein_id: str,reaction_id: str) -> dict[str,Any]:
        if self.enzymology_evidence is None:
            return {
                'schema':'fibre-enzymology-state-v1',
                'status':'unavailable',
                'ranking_effect':False,
                'load_error':self.enzymology_evidence_error,
            }
        return {
            'status':'ready',
            **self.enzymology_evidence.state(protein_id,reaction_id),
        }

    def _assay_context_censor(
        self,
        direction: str,
        query_meta: dict[str, Any],
        target_conditions: dict[str, Any] | None,
        eligible: np.ndarray,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Apply only exact, pair-specific matched assay contradictions as censoring.

        This is an eligibility constraint, not another ranking coordinate. Missing
        assays, mismatched conditions, supporting assays and conflicting evidence
        never become negative evidence.
        """
        mask=np.zeros(len(eligible),dtype=bool)
        target=context_from_target_conditions(target_conditions)
        info: dict[str,Any]={
            'schema':'fibre-assay-context-censor-v1',
            'status':'not_requested',
            'requested_dimensions':list(target.observed_dimensions),
            'role':'eligibility_constraint_only',
            'score_mutated':False,
            'policy':(
                'exclude only an exact canonical pair with pair-specific '
                'inactive/below-detection/no-conversion evidence matching every '
                'requested assay dimension; missing, mismatched, supporting or '
                'conflicting evidence is never treated as a negative'
            ),
            'excluded_candidate_count':0,
            'excluded_canonical_candidate_ids':[],
        }
        if not target.observed_dimensions:
            return mask,info
        if self.enzymology_evidence is None:
            info['status']='evidence_unavailable'
            return mask,info
        if not bool(query_meta.get('query_is_reference_entity')):
            info['status']='external_query_has_no_exact_pair_assay_identity'
            return mask,info
        canonical=str(query_meta.get('canonical_query_id') or '')
        if not canonical:
            info['status']='canonical_query_identity_unavailable'
            return mask,info

        statuses={'supported':0,'contradicted':0,'conflicting':0,'unresolved':0}
        eligible_arr=np.asarray(eligible,dtype=bool)
        for index in np.flatnonzero(eligible_arr):
            if direction=='reaction_to_enzyme':
                protein_id=self.protein_ids[int(index)]
                reaction_id=canonical
                candidate_id=protein_id
            elif direction=='enzyme_to_reaction':
                protein_id=canonical
                reaction_id=self.reaction_ids[int(index)]
                candidate_id=reaction_id
            else:
                raise ValueError(f'unsupported FIBRE direction: {direction}')
            assessment=assess_pair_context(
                self.enzymology_evidence.assay_observations(
                    protein_id,reaction_id
                ),
                target,
                enzyme_id=protein_id,
                reaction_id=reaction_id,
            )
            statuses[assessment.status]=statuses.get(assessment.status,0)+1
            if assessment.status=='contradicted':
                mask[int(index)]=True

        excluded=np.flatnonzero(mask)
        ids=(
            self.protein_ids if direction=='reaction_to_enzyme'
            else self.reaction_ids
        )
        info.update({
            'status':'applied' if len(excluded) else 'evaluated_no_exclusion',
            'eligible_candidate_count_before_censor':int(np.sum(eligible_arr)),
            'context_status_counts':{k:int(v) for k,v in statuses.items()},
            'excluded_candidate_count':int(len(excluded)),
            'excluded_canonical_candidate_ids':[
                str(ids[int(i)]) for i in excluded
            ],
        })
        return mask,info

    def _load_stratified_geometry(self) -> None:
        """Load optional finer FIBRE resolutions without changing coarse rank.

        The service must remain usable when catalytic-local assets are absent.
        A loaded local geometry is therefore additive scientific resolution,
        not a startup dependency of the canonical coarse correspondence route.
        """
        self.stratified_manifest: dict[str, Any] | None = None
        self.stratified_load_error: str | None = None
        self.local_global_rows = np.zeros(0, dtype=np.int64)
        self.local_common = np.zeros(len(self.protein_ids), dtype=bool)
        self.local_global_to_row: dict[int, int] = {}
        self.local_states: dict[str, Any] = {}
        self.local_coordinates: tuple[str, ...] = ()
        self.mechanistic_manifest: dict[str, Any] | None = None
        self.mechanistic_load_error: str | None = None
        self.mechanistic_coordinates: tuple[str, ...] = ()
        self.mechanistic_states: dict[str, Any] = {}
        self.mechanistic_global_rows: dict[str, np.ndarray] = {}
        self.mechanistic_global_to_row: dict[str, dict[int, int]] = {}
        self.mechanistic_applicability: dict[str, np.ndarray] = {}
        self.mechanistic_availability: dict[str, np.ndarray] = {}

        manifest_path = CATALYTIC_CONSENSUS / 'manifest.json'
        if not manifest_path.is_file():
            return
        try:
            manifest = json.loads(manifest_path.read_text())
            local_reactions = pd.read_csv(
                CATALYTIC_CONSENSUS / 'reaction_ids.csv', dtype=str
            ).fillna('')
            reaction_column = (
                'reaction_id' if 'reaction_id' in local_reactions.columns
                else local_reactions.columns[-1]
            )
            if local_reactions[reaction_column].astype(str).tolist() != self.reaction_ids:
                raise RuntimeError('catalytic consensus reaction order mismatch')

            local_proteins = pd.read_csv(
                CATALYTIC_CONSENSUS / 'protein_ids.csv', dtype=str
            ).fillna('')
            protein_column = (
                'protein_id' if 'protein_id' in local_proteins.columns
                else local_proteins.columns[-1]
            )
            rows = np.load(
                CATALYTIC_CONSENSUS / 'protein_global_rows.npy'
            ).astype(np.int64)
            local_ids = local_proteins[protein_column].astype(str).tolist()
            if len(rows) != len(local_ids):
                raise RuntimeError('catalytic consensus protein mapping length mismatch')
            if [self.protein_ids[int(i)] for i in rows] != local_ids:
                raise RuntimeError('catalytic consensus protein order mismatch')

            coords = tuple(str(x) for x in manifest.get('protein_coordinates', []))
            if not coords:
                raise RuntimeError('catalytic consensus has no protein coordinates')
            reaction_geo = np.load(
                CATALYTIC_CONSENSUS / 'reaction_local_geodesic.npy', mmap_mode='r'
            )
            if reaction_geo.shape != (len(self.reaction_ids), len(self.reaction_ids)):
                raise RuntimeError('catalytic consensus reaction geodesic shape mismatch')
            reaction_sq = np.square(np.asarray(reaction_geo, dtype=np.float64))

            self.local_global_rows = rows
            self.local_common[rows] = True
            self.local_global_to_row = {
                int(global_index): local_index
                for local_index, global_index in enumerate(rows)
            }
            local_pairs = np.asarray([
                (int(rr), self.local_global_to_row[int(ee)])
                for rr, ee in self.positive_pairs
                if int(ee) in self.local_global_to_row
            ], dtype=np.int64)
            if not len(local_pairs):
                raise RuntimeError('catalytic consensus positive support is empty')

            for coord in coords:
                geo = np.load(
                    CATALYTIC_CONSENSUS / f'protein_{coord}_geodesic.npy',
                    mmap_mode='r',
                )
                if geo.shape != (len(rows), len(rows)):
                    raise RuntimeError(f'catalytic consensus {coord} geodesic shape mismatch')
                protein_sq = np.square(np.asarray(geo, dtype=np.float64))
                self.local_states[coord] = correspondence_state(
                    reaction_sq, protein_sq, local_pairs
                )

            if (PROTEIN_GEOMETRY / 'protein_ids.csv').is_file():
                pg_ids = pd.read_csv(
                    PROTEIN_GEOMETRY / 'protein_ids.csv', dtype=str
                ).fillna('')
                pg_col = (
                    'protein_id' if 'protein_id' in pg_ids.columns
                    else pg_ids.columns[-1]
                )
                if pg_ids[pg_col].astype(str).tolist() != self.protein_ids:
                    raise RuntimeError('protein geometry order mismatch for mechanistic coordinates')
                for name in ('typeI_aspartate', 'nse_dte', 'dxdd', 'qw'):
                    apath = PROTEIN_GEOMETRY / f'{name}_available.npy'
                    if apath.is_file():
                        values = np.load(apath).astype(bool)
                        if len(values) != len(self.protein_ids):
                            raise RuntimeError(f'{name} availability length mismatch')
                        self.mechanistic_availability[name] = values

            self.local_coordinates = coords
            self.stratified_manifest = manifest

            mechanistic_manifest_path = MECHANISTIC_CHART / 'manifest.json'
            if mechanistic_manifest_path.is_file():
                try:
                    mmanifest = json.loads(mechanistic_manifest_path.read_text())
                    mids = pd.read_csv(
                        MECHANISTIC_CHART / 'protein_ids.csv', dtype=str
                    ).fillna('')
                    mid_col = (
                        'protein_id' if 'protein_id' in mids.columns
                        else mids.columns[-1]
                    )
                    if mids[mid_col].astype(str).tolist() != self.protein_ids:
                        raise RuntimeError('mechanistic chart protein order mismatch')
                    mrids = pd.read_csv(
                        MECHANISTIC_CHART / 'reaction_ids.csv', dtype=str
                    ).fillna('')
                    mrid_col = (
                        'reaction_id' if 'reaction_id' in mrids.columns
                        else mrids.columns[-1]
                    )
                    if mrids[mrid_col].astype(str).tolist() != self.reaction_ids:
                        raise RuntimeError('mechanistic chart reaction order mismatch')
                    mcoords = tuple(
                        str(x) for x in mmanifest.get('protein_coordinates', [])
                    )
                    if not mcoords:
                        raise RuntimeError('mechanistic chart has no coordinates')
                    for name in mcoords:
                        mrows = np.load(
                            MECHANISTIC_CHART / f'protein_{name}_global_rows.npy'
                        ).astype(np.int64)
                        applicable = np.load(
                            MECHANISTIC_CHART / f'protein_{name}_applicable.npy'
                        ).astype(bool)
                        available = np.load(
                            MECHANISTIC_CHART / f'protein_{name}_available.npy'
                        ).astype(bool)
                        if len(applicable) != len(self.protein_ids) or len(available) != len(self.protein_ids):
                            raise RuntimeError(f'{name} mechanistic mask length mismatch')
                        if np.any(available & ~applicable):
                            raise RuntimeError(f'{name} observed outside applicable chart')
                        if not np.array_equal(np.flatnonzero(available), mrows):
                            raise RuntimeError(f'{name} mechanistic row mapping mismatch')
                        geo = np.load(
                            MECHANISTIC_CHART / f'protein_{name}_geodesic.npy',
                            mmap_mode='r',
                        )
                        if geo.shape != (len(mrows), len(mrows)):
                            raise RuntimeError(f'{name} mechanistic geodesic shape mismatch')
                        mapping = {
                            int(global_index): local_index
                            for local_index, global_index in enumerate(mrows)
                        }
                        mpairs = np.asarray([
                            (int(rr), mapping[int(ee)])
                            for rr, ee in self.positive_pairs
                            if int(ee) in mapping
                        ], dtype=np.int64)
                        if not len(mpairs):
                            raise RuntimeError(f'{name} mechanistic positive support is empty')
                        self.mechanistic_states[name] = correspondence_state(
                            reaction_sq,
                            np.square(np.asarray(geo, dtype=np.float64)),
                            mpairs,
                        )
                        self.mechanistic_global_rows[name] = mrows
                        self.mechanistic_global_to_row[name] = mapping
                        self.mechanistic_applicability[name] = applicable
                        self.mechanistic_availability[name] = available
                    self.mechanistic_coordinates = mcoords
                    self.mechanistic_manifest = mmanifest
                except Exception as exc:
                    self.mechanistic_load_error = f'{type(exc).__name__}: {exc}'
                    self.mechanistic_manifest = None
                    self.mechanistic_coordinates = ()
                    self.mechanistic_states = {}
                    self.mechanistic_global_rows = {}
                    self.mechanistic_global_to_row = {}
                    self.mechanistic_applicability = {}
        except Exception as exc:
            self.stratified_load_error = f'{type(exc).__name__}: {exc}'
            self.stratified_manifest = None
            self.local_states = {}
            self.local_coordinates = ()

    def _verify_application_profile_integrity(self, manifest: dict[str, Any]) -> None:
        source_expected = dict(manifest.get("source_input_sha256") or {})
        source_paths = {
            "canonical_protein_entities": APPLICATION_CANONICAL/"protein_entities.csv",
            "canonical_reaction_entities": APPLICATION_CANONICAL/"reaction_entities.csv",
            "accepted_pair_source": APPLICATION_CANONICAL/"marts_pair_folds.csv",
            "primary_protein_affinity": APPLICATION_PROTEIN_GEOMETRY/"partial_pullback_affinity.npz",
            "primary_reaction_affinity": APPLICATION_REACTION_GEOMETRY/"partial_pullback_affinity.npz",
            "tps_canonical_protein_features": TPS_CANONICAL_PROTEIN_FEATURES,
            "tps_canonical_reaction_features": TPS_CANONICAL_REACTION_FEATURES,
            "tps_training_pairs": TPS_SOURCE_MODEL/"training_pairs.csv",
            "tps_feature_schema": TPS_SOURCE_MODEL/"feature_schema.json",
        }
        missing_source = sorted(set(source_paths)-set(source_expected))
        if missing_source:
            raise RuntimeError(
                "application source integrity manifest is incomplete: "
                + ", ".join(missing_source)
            )
        for key,path in source_paths.items():
            if not path.is_file():
                raise RuntimeError(f"application source missing: {path}")
            actual=application_sha256_file(path)
            if actual != str(source_expected[key]):
                raise RuntimeError(
                    f"application source hash mismatch for {key}: {actual}"
                )

        checkpoint_rows=list(manifest.get("tps_source_checkpoints") or [])
        if not checkpoint_rows:
            raise RuntimeError("application TPS source checkpoint manifest is empty")
        for row in checkpoint_rows:
            raw=Path(str(row.get("path") or ""))
            path=raw if raw.is_absolute() else ROOT/raw
            expected=str(row.get("sha256") or "")
            if not path.is_file() or not expected:
                raise RuntimeError(f"application TPS checkpoint missing/incomplete: {raw}")
            actual=application_sha256_file(path)
            if actual != expected:
                raise RuntimeError(
                    f"application TPS checkpoint hash mismatch: {raw}"
                )

        atlas_inputs=dict(self.manifest.get("input_sha256") or {})
        expected_protein_manifest=str(atlas_inputs.get("protein_geometry_manifest") or "")
        expected_reaction_manifest=str(atlas_inputs.get("reaction_geometry_manifest") or "")
        actual_protein_manifest=application_sha256_file(
            APPLICATION_PROTEIN_GEOMETRY/"manifest.json"
        )
        actual_reaction_manifest=application_sha256_file(
            APPLICATION_REACTION_GEOMETRY/"manifest.json"
        )
        if actual_protein_manifest != expected_protein_manifest:
            raise RuntimeError(
                "online deployment atlas is not tied to the promoted protein geometry"
            )
        if actual_reaction_manifest != expected_reaction_manifest:
            raise RuntimeError(
                "online deployment atlas is not tied to the promoted reaction geometry"
            )

        evidence_manifest=dict(manifest.get("enzymology_evidence") or {})
        source_rows=dict(evidence_manifest.get("sources") or {})
        generated_rows=dict(evidence_manifest.get("generated") or {})
        required_sources={
            "observation_index","uniprot_state","publication_context","rhea_mapping"
        }
        required_generated=set(
            evidence_manifest.get("required_generated_for_full_information_runtime") or []
        )
        expected_generated={
            "assay_context","catalytic_state","cross_source_catalytic_evidence"
        }
        if not required_sources.issubset(source_rows):
            raise RuntimeError(
                "application enzymology source integrity manifest is incomplete: "
                + ", ".join(sorted(required_sources-set(source_rows)))
            )
        if required_generated != expected_generated:
            raise RuntimeError(
                "application enzymology generated-asset contract mismatch: "
                + repr(sorted(required_generated))
            )
        if not expected_generated.issubset(generated_rows):
            raise RuntimeError(
                "application enzymology generated integrity manifest is incomplete: "
                + ", ".join(sorted(expected_generated-set(generated_rows)))
            )
        for section,rows in (("source",source_rows),("generated",generated_rows)):
            for key,row in rows.items():
                raw=Path(str((row or {}).get("path") or ""))
                path=raw if raw.is_absolute() else ROOT/raw
                expected=str((row or {}).get("tree_sha256") or "")
                if not path.is_dir() or len(expected)!=64:
                    raise RuntimeError(
                        f"application enzymology {section} tree missing/incomplete: {key}"
                    )
                actual=application_tree_sha256(path)
                if actual != expected:
                    raise RuntimeError(
                        f"application enzymology {section} tree hash mismatch for {key}: {actual}"
                    )

        generated_expected=dict(manifest.get("generated_tree_sha256") or {})
        generated_paths={
            "primary_global_reference": APPLICATION_ROOT/"primary_global_reference",
            "tps_adapted_coordinate": APPLICATION_ROOT/"tps_adapted_coordinate",
            "tps_domain_reference": TPS_DOMAIN_REFERENCE,
        }
        missing_generated=sorted(set(generated_paths)-set(generated_expected))
        if missing_generated:
            raise RuntimeError(
                "application generated-tree integrity manifest is incomplete: "
                + ", ".join(missing_generated)
            )
        for key,path in generated_paths.items():
            if not path.is_dir():
                raise RuntimeError(f"application generated tree missing: {path}")
            actual=application_tree_sha256(path)
            if actual != str(generated_expected[key]):
                raise RuntimeError(
                    f"application generated-tree hash mismatch for {key}: {actual}"
                )

    def _load_application_profile(self) -> None:
        """Load the generated all-data application refinement without making it a benchmark dependency."""
        self.application_manifest = None
        self.application_load_error = None
        self.application_integrity_verified = False
        self.application_primary_runtime_verified = False
        self.application_integrity_policy = "manifest_only"
        self.tps_domain_bundle = None
        manifest_path = TPS_APPLICATION_MANIFEST
        if not manifest_path.is_file():
            self.application_load_error = "full_data_application_manifest_missing"
            return
        try:
            manifest = json.loads(manifest_path.read_text())
            if str(manifest.get("release_profile") or "") != "starase-application":
                raise RuntimeError("application manifest release_profile mismatch")
            if bool(manifest.get("benchmark_claims_allowed")):
                raise RuntimeError("application manifest must forbid benchmark claims")
            strict_integrity = str(
                os.environ.get(
                    "STARASE_NAVIGATOR_VERIFY_APPLICATION_INTEGRITY",
                    os.environ.get("STARASE_NAVIGATOR_REQUIRE_APPLICATION_PROFILE",""),
                )
            ).strip().lower() in {"1","true","yes","on"}
            if strict_integrity:
                self._verify_application_profile_integrity(manifest)
                self.application_integrity_verified = True
                self.application_primary_runtime_verified = True
                self.application_integrity_policy = "strict_sha256"
            bundle = PortableReferenceBundle(TPS_DOMAIN_REFERENCE)
            if bundle.pids != self.protein_ids:
                raise RuntimeError("TPS-domain application protein order mismatch")
            if bundle.rids != self.reaction_ids:
                raise RuntimeError("TPS-domain application reaction order mismatch")
            self.application_manifest = manifest
            self.tps_domain_bundle = bundle
        except Exception as exc:
            self.application_load_error = f"{type(exc).__name__}: {exc}"
            self.application_manifest = None
            self.tps_domain_bundle = None

    def _tps_projector_runtime(self) -> TPSAdaptedCoordinateProjector:
        with self._application_lock:
            if self._tps_projector is None:
                self._tps_projector = TPSAdaptedCoordinateProjector()
            return self._tps_projector

    def application_profile_status(self) -> dict[str, Any]:
        manifest = self.application_manifest or {}
        canonical = manifest.get("canonical_universe") or {}
        return {
            "release_profile": "starase-application",
            "status": "ready" if self.tps_domain_bundle is not None else "degraded",
            "manifest": str(TPS_APPLICATION_MANIFEST.relative_to(ROOT)),
            "load_error": self.application_load_error,
            "integrity_policy": self.application_integrity_policy,
            "integrity_verified": bool(self.application_integrity_verified),
            "primary_runtime_verified_against_promoted_geometry": bool(
                self.application_primary_runtime_verified
            ),
            "protein_states": canonical.get("protein_states"),
            "reaction_states": canonical.get("reaction_states"),
            "accepted_positive_pairs": canonical.get("accepted_positive_pairs"),
            "primary_resolution": "canonical_fibre_correspondence",
            "within_level_refinement": (
                "tps_pair_supervised_fibre_coordinate"
                if self.tps_domain_bundle is not None else "unavailable"
            ),
            "benchmark_claims_allowed": False,
        }

    def _tps_domain_defect(
        self,
        direction: str,
        payload: dict[str, Any],
        meta: dict[str, Any],
    ) -> tuple[np.ndarray | None, dict[str, Any]]:
        bundle = self.tps_domain_bundle
        if bundle is None:
            return None, {
                "status": "unavailable",
                "reason": self.application_load_error or "application_bundle_unavailable",
            }
        try:
            canonical = str(meta.get("canonical_query_id") or "")
            if bool(meta.get("query_is_reference_entity")) and canonical:
                defect = (
                    bundle.enzyme_defect(canonical)
                    if direction == "reaction_to_enzyme"
                    else bundle.reaction_defect(canonical)
                )
                mode = "reference_entity"
            elif direction == "reaction_to_enzyme":
                reaction_smiles = str(
                    meta.get("_canonical_reaction_smiles")
                    or payload.get("reaction_smiles")
                    or ""
                ).strip()
                if not reaction_smiles:
                    raise ValueError("external TPS-domain reaction refinement needs reaction_smiles")
                vector = self._tps_projector_runtime().reaction_smiles(reaction_smiles)
                defect = bundle.enzyme_defect_from_reaction_views({"global": vector})
                mode = "out_of_sample_reaction"
            elif direction == "enzyme_to_reaction":
                embedding = meta.get("_global_esmc_embedding")
                if embedding is not None:
                    vector = self._tps_projector_runtime().project_protein_features(
                        np.asarray(embedding,dtype=np.float32)
                    )[0]
                    projector_audit = {"source": "reuse_primary_esmc_embedding"}
                else:
                    sequence = str(payload.get("enzyme_sequence") or "").strip()
                    if not sequence:
                        raise ValueError("external TPS-domain protein refinement needs sequence")
                    vector, projector_audit = self._tps_projector_runtime().protein_sequence(sequence)
                defect = bundle.reaction_defect_from_protein_views({"global": vector})
                mode = "out_of_sample_protein"
            else:
                raise ValueError(f"unsupported application direction: {direction}")
            defect = np.asarray(defect,dtype=np.float64)
            expected = len(self.protein_ids) if direction == "reaction_to_enzyme" else len(self.reaction_ids)
            if defect.shape != (expected,):
                raise RuntimeError(
                    f"TPS-domain defect shape mismatch: {defect.shape} != {(expected,)}"
                )
            return defect, {
                "status": "ready",
                "mode": mode,
                "coordinate": "tps_pair_supervised_application_coordinate",
                "pair_supervised": True,
                "legacy_cross_factor_score_used": False,
                "projector_audit": locals().get("projector_audit", {}),
            }
        except Exception as exc:
            return None, {
                "status": "failed",
                "reason": f"{type(exc).__name__}: {exc}",
            }

    @staticmethod
    def _application_refined_order(
        scores: np.ndarray,
        ids: np.ndarray | list[str],
        eligible: np.ndarray,
        top_k: int,
        coarse_levels: np.ndarray,
        tps_defect: np.ndarray | None,
    ) -> np.ndarray:
        if tps_defect is None:
            return CorrespondenceGeometryService._rank_order(
                scores,np.asarray(ids,dtype=object),eligible,top_k
            )
        idx=np.flatnonzero(np.asarray(eligible,dtype=bool))
        if not len(idx):
            return idx
        defect=np.asarray(tps_defect,dtype=np.float64)
        if defect.shape!=np.asarray(scores).shape:
            raise ValueError("application refinement defect shape mismatch")
        levels=np.asarray(coarse_levels,dtype=np.int64)
        stable_ids=np.asarray(ids,dtype=str)
        secondary=np.where(np.isfinite(defect),defect,np.inf)
        order=np.lexsort((stable_ids[idx],secondary[idx],levels[idx]))
        return idx[order[:max(0,int(top_k))]]

    def contains_protein(self, value: str) -> bool:
        return str(value or '').strip().casefold() in self.protein_alias_to_internal

    def contains_reaction(self, value: str) -> bool:
        return str(value or '').strip().casefold() in self.reaction_alias_to_internal

    def _protein_internal(self, value: str) -> str | None:
        return self.protein_alias_to_internal.get(str(value or '').strip().casefold())

    def _reaction_internal(self, value: str) -> str | None:
        return self.reaction_alias_to_internal.get(str(value or '').strip().casefold())

    @staticmethod
    def _chordal_cross(query: np.ndarray, reference: np.ndarray) -> np.ndarray:
        q = np.asarray(query, dtype=np.float64).reshape(-1)
        norm = float(np.linalg.norm(q))
        if norm <= 1e-12:
            return np.full(len(reference), np.inf, dtype=np.float64)
        q = q / norm
        sim = np.clip(np.asarray(reference, dtype=np.float64) @ q, -1.0, 1.0)
        return np.sqrt(np.maximum(2.0 - 2.0 * sim, 0.0))

    def _side_raw_affinity(self, query: dict[str, object], key: str) -> np.ndarray:
        values = np.zeros(len(self.reference_reaction_features), dtype=np.float64)
        if not query[key]:
            return values
        for index, reference in enumerate(self.reference_reaction_features):
            if not reference[key]:
                continue
            forward = float(best_match_similarity(query[key], reference[key]))
            backward = float(best_match_similarity(reference[key], query[key]))
            values[index] = 0.5 * (forward + backward)
        return values

    @staticmethod
    def _diffusion_cross(raw_affinity: np.ndarray, basis: dict[str, np.ndarray]) -> tuple[np.ndarray, bool]:
        a = np.asarray(raw_affinity, dtype=np.float64).reshape(-1)
        available = np.asarray(basis['available'], dtype=bool)
        a = np.where(available, np.maximum(a, 0.0), 0.0)
        degree = float(a.sum())
        if degree <= 1e-12:
            return np.full(len(a), np.inf, dtype=np.float64), False
        p = a / degree
        pi = np.asarray(basis['stationary'], dtype=np.float64)
        weighted = p / np.sqrt(np.maximum(pi, 1e-12))
        qnorm = float(weighted @ weighted)
        ref_w = np.asarray(basis['weighted_transition'], dtype=np.float64)
        ref_norm = np.asarray(basis['weighted_norm'], dtype=np.float64)
        d2 = np.maximum(qnorm + ref_norm - 2.0 * (ref_w @ weighted), 0.0)
        d = np.sqrt(d2)
        d[~available] = np.inf
        return d, True

    def reaction_distances(self, *, reaction_id: str = '', reaction_smiles: str = '') -> tuple[np.ndarray, dict[str, Any]]:
        internal = self._reaction_internal(reaction_id)
        supplied_smiles = str(reaction_smiles or '').strip()
        if internal is None and supplied_smiles:
            signature = reaction_signature(supplied_smiles)
            internal = self.signature_to_internal.get(signature)
        if internal is not None:
            return np.asarray(self.Dr[self.ri[internal]], dtype=np.float64), {
                'query_is_reference_entity': True,
                'canonical_query_id': internal,
                'executed_measurements': [],
            }
        if not supplied_smiles:
            raise ValueError('external reaction query requires reaction_smiles')

        canonical = canonical_or_raw_reaction(supplied_smiles)
        encoded = np.asarray(DrfpEncoder.encode([canonical])[0], dtype=np.float64)
        drfp_distance = self._chordal_cross(encoded, self.reaction_drfp)
        qfeat = reaction_features(canonical)
        react_distance, react_ok = self._diffusion_cross(
            self._side_raw_affinity(qfeat, 'reactant_fps'), self.side_basis['reactant']
        )
        product_distance, product_ok = self._diffusion_cross(
            self._side_raw_affinity(qfeat, 'product_fps'), self.side_basis['product']
        )
        attachment = attach_query_to_reference(
            [drfp_distance, react_distance, product_distance],
            [bool(np.any(np.isfinite(drfp_distance))), react_ok, product_ok],
            [self.reaction_drfp_available, self.side_basis['reactant']['available'], self.side_basis['product']['available']],
            [self.reaction_drfp_scale, self.side_basis['reactant']['scale'], self.side_basis['product']['scale']],
        )
        attachment = replace(attachment, edge_lengths=attachment.edge_lengths / self.reaction_ell)
        return query_geodesic_to_reference(self.reaction_graph, attachment), {
            'query_is_reference_entity': False,
            'canonical_query_id': None,
            'executed_measurements': ['drfp', 'reactant_product_neighbourhood'],
            'attachment_count': int(len(attachment.reference_indices)),
            '_canonical_reaction_smiles': canonical,
        }

    def _protein_oos_distances_from_embedding(
        self,
        embedding: np.ndarray,
        *,
        whole_3di_distance: np.ndarray | None = None,
    ) -> tuple[np.ndarray, int]:
        global_distance = self._chordal_cross(embedding, self.protein_global)
        distances = [global_distance]
        query_available = [True]
        reference_available = [self.protein_global_available]
        reference_scales = [self.protein_global_scale]
        if (
            whole_3di_distance is not None
            and self.protein_whole_3di_available is not None
            and self.protein_whole_3di_scale is not None
        ):
            distances.append(np.asarray(whole_3di_distance, dtype=np.float64))
            query_available.append(bool(np.any(np.isfinite(whole_3di_distance))))
            reference_available.append(self.protein_whole_3di_available)
            reference_scales.append(self.protein_whole_3di_scale)
        attachment = attach_query_to_reference(
            distances, query_available, reference_available, reference_scales
        )
        attachment = replace(attachment, edge_lengths=attachment.edge_lengths / self.protein_ell)
        return query_geodesic_to_reference(self.protein_graph, attachment), int(len(attachment.reference_indices))

    def protein_distances(
        self,
        *,
        enzyme_id: str = '',
        enzyme_sequence: str = '',
        protein_structure_path: str = '',
    ) -> tuple[np.ndarray, dict[str, Any]]:
        internal = self._protein_internal(enzyme_id)
        sequence = ''.join(str(enzyme_sequence or '').split()).upper()
        if internal is None and sequence:
            internal = self.sequence_to_internal.get(sequence)
        if internal is not None:
            return np.asarray(self.Dp[self.pi[internal]], dtype=np.float64), {
                'query_is_reference_entity': True,
                'canonical_query_id': internal,
                'executed_measurements': [],
                'failed_measurements': {},
            }
        if not sequence:
            raise ValueError('external protein query requires enzyme_sequence')
        frame = pd.DataFrame([{'enzyme_id': str(enzyme_id or 'external_query'), 'sequence': sequence}])
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        with self._protein_encode_lock:
            matrix, audits = encode_external_enzymes_with_audit(frame, device, 'esmc_600m')

        executed = ['global_esmc']
        failed: dict[str, str] = {}
        whole_distance: np.ndarray | None = None
        structure_path = Path(str(protein_structure_path)).resolve() if str(protein_structure_path).strip() else None
        structure_meta: dict[str, Any] = {}
        if structure_path is not None and structure_path.is_file() and self.structural_basis is not None:
            executed.append('resolved_structure')
            try:
                structural = query_structural_view(
                    structure_path,
                    view='whole_3di',
                    basis_root=self.structural_basis,
                    work_root=STRUCTURAL_WORK_ROOT,
                )
                if structural.observed:
                    whole_distance = structural.diffusion_distance
                    executed.append('whole_3di')
                    structure_meta = {
                        'whole_3di_hit_count': int(structural.hit_count),
                        'whole_3di_query_self_bits': float(structural.query_self_bits),
                        'structure_path': str(structure_path),
                    }
                else:
                    failed['whole_3di'] = 'no structural neighbours were observed in the frozen reference basis'
            except Exception as exc:
                failed['whole_3di'] = f'{type(exc).__name__}: {exc}'

        geodesic, attachment_count = self._protein_oos_distances_from_embedding(
            matrix[0], whole_3di_distance=whole_distance
        )
        return geodesic, {
            'query_is_reference_entity': False,
            'canonical_query_id': None,
            'executed_measurements': executed,
            'failed_measurements': failed,
            'attachment_count': attachment_count,
            'protein_input_audit': audits[0].__dict__ if audits else {},
            '_global_esmc_embedding': np.asarray(matrix[0],dtype=np.float32),
            **structure_meta,
        }

    def _map_protein_indices(self, values: list[str] | tuple[str, ...] | None) -> tuple[list[int], list[str]]:
        found: list[int] = []
        missing: list[str] = []
        for raw in values or []:
            internal = self._protein_internal(str(raw))
            if internal is None:
                missing.append(str(raw))
            else:
                index = self.pi[internal]
                if index not in found:
                    found.append(index)
        return found, missing

    def _map_reaction_indices(self, values: list[str] | tuple[str, ...] | None) -> tuple[list[int], list[str]]:
        found: list[int] = []
        missing: list[str] = []
        for raw in values or []:
            internal = self._reaction_internal(str(raw))
            if internal is None:
                missing.append(str(raw))
            else:
                index = self.ri[internal]
                if index not in found:
                    found.append(index)
        return found, missing

    @staticmethod
    def _rank_order(scores: np.ndarray, ids: np.ndarray, eligible: np.ndarray, top_k: int) -> np.ndarray:
        selected = np.flatnonzero(eligible)
        order = np.lexsort((ids[selected].astype(str), -scores[selected]))
        return selected[order[:max(0, int(top_k))]]


    @staticmethod
    def _geometric_uncertainty(
        scores: np.ndarray,
        eligible: np.ndarray,
        query_marginal_sq: float,
        candidate_marginal_sq: np.ndarray,
    ) -> dict[str, Any]:
        """Threshold-free ambiguity/support diagnostics for the scored fibre.

        This metadata never changes candidate ordering. It exposes the numerical
        level set of the best eligible defect together with intrinsic distances
        to the current positive marginal support. Values are geometry
        diagnostics, not calibrated probabilities or OOD classes.
        """
        score=np.asarray(scores,dtype=np.float64).reshape(-1)
        mask=np.asarray(eligible,dtype=bool).reshape(-1)
        marginal=np.asarray(candidate_marginal_sq,dtype=np.float64).reshape(-1)
        if not (len(score)==len(mask)==len(marginal)):
            raise ValueError("uncertainty arrays must have equal length")
        selected=np.flatnonzero(mask & np.isfinite(score) & np.isfinite(marginal))
        if not len(selected):
            return {
                'schema':'fibre-geometric-uncertainty-v1',
                'status':'no_eligible_candidates',
                'calibrated_probability':False,
            }
        defect=-score[selected]
        tol=numerical_level_tolerance(defect)
        best=float(np.min(defect))
        top=np.abs(defect-best)<=tol
        above=defect>best+tol
        support=np.sqrt(np.maximum(marginal[selected][top],0.0))
        return {
            'schema':'fibre-geometric-uncertainty-v1',
            'status':'available',
            'calibrated_probability':False,
            'interpretation':'intrinsic support and numerical level-set ambiguity; not a probability or OOD threshold',
            'query_support_distance':float(np.sqrt(max(float(query_marginal_sq),0.0))),
            'best_defect':best,
            'best_level_size':int(np.sum(top)),
            'best_level_fraction':float(np.mean(top)),
            'next_level_gap':(
                float(np.min(defect[above])-best) if np.any(above) else None
            ),
            'best_level_candidate_support_distance_min':float(np.min(support)),
            'best_level_candidate_support_distance_median':float(np.median(support)),
            'numerical_level_tolerance':float(tol),
        }

    def _mechanistic_coordinates_for_protein(self, protein_index: int) -> list[str]:
        index=int(protein_index)
        return [
            name
            for name, available in self.mechanistic_availability.items()
            if 0 <= index < len(available) and bool(available[index])
        ]

    def _mechanistic_chart_for_protein(self, protein_index: int) -> list[str]:
        index=int(protein_index)
        return [
            name
            for name, applicable in self.mechanistic_applicability.items()
            if 0 <= index < len(applicable) and bool(applicable[index])
        ]

    def _mechanistic_names_from_mask(self, mask: int) -> list[str]:
        value=int(mask)
        return [
            name
            for bit,name in enumerate(self.mechanistic_coordinates)
            if value & (1 << bit)
        ]

    def _mechanistic_relation(
        self,
        direction: str,
        canonical: str,
        catalytic_resolution: Any,
    ) -> tuple[np.ndarray,np.ndarray,dict[str,Any]]:
        n=len(catalytic_resolution.coarse_level)
        chart=np.zeros(n,dtype=np.int64)
        strata=np.full(n,-1,dtype=np.int64)
        meta: dict[str,Any]={
            'mechanistic_status':'unavailable',
            'mechanistic_order_bearing':False,
            'mechanistic_coordinates':list(self.mechanistic_coordinates),
            'mechanistic_relation':'family-applicable motif charts with Pareto FIBRE refinement inside observed catalytic parents',
        }
        if self.mechanistic_manifest is None or not self.mechanistic_states:
            if self.mechanistic_load_error:
                meta['mechanistic_load_error']=self.mechanistic_load_error
            return chart,strata,meta

        try:
            if direction=='reaction_to_enzyme':
                r=self.ri[canonical]
                m=len(self.mechanistic_coordinates)
                defect=np.full((m,len(self.protein_ids)),np.nan,dtype=np.float64)
                applicable=np.zeros_like(defect,dtype=bool)
                available=np.zeros_like(defect,dtype=bool)
                for c,name in enumerate(self.mechanistic_coordinates):
                    rows=self.mechanistic_global_rows[name]
                    defect[c,rows]=self.mechanistic_states[name].defect[r]
                    applicable[c]=self.mechanistic_applicability[name]
                    available[c]=self.mechanistic_availability[name]
            elif direction=='enzyme_to_reaction':
                e=self.pi[canonical]
                m=len(self.mechanistic_coordinates)
                defect=np.full((m,len(self.reaction_ids)),np.nan,dtype=np.float64)
                applicable=np.zeros_like(defect,dtype=bool)
                available=np.zeros_like(defect,dtype=bool)
                meta['query_mechanistic_chart']=self._mechanistic_chart_for_protein(e)
                meta['query_mechanistic_coordinates']=self._mechanistic_coordinates_for_protein(e)
                for c,name in enumerate(self.mechanistic_coordinates):
                    is_applicable=bool(self.mechanistic_applicability[name][e])
                    applicable[c,:]=is_applicable
                    local=self.mechanistic_global_to_row[name].get(int(e))
                    if local is None:
                        continue
                    defect[c]=self.mechanistic_states[name].defect[:,local]
                    available[c,:]=True
            else:
                raise ValueError(f'unsupported FIBRE direction: {direction}')

            resolution=mechanistic_chart_resolution(
                catalytic_resolution.coarse_level,
                catalytic_resolution.catalytic_stratum,
                catalytic_resolution.observed_coarse_levels,
                defect,
                applicable,
                available,
            )
            chart=np.asarray(resolution.mechanistic_chart,dtype=np.int64)
            strata=np.asarray(resolution.mechanistic_stratum,dtype=np.int64)
            meta.update({
                'mechanistic_status':'available_non_order_bearing',
                'mechanistic_observed_parent_chart_count':int(
                    len(resolution.observed_parent_charts)
                ),
                'mechanistic_refined_parent_chart_count':int(
                    len(resolution.refined_parent_charts)
                ),
                'mechanistic_stratum_candidate_count':int(
                    resolution.refined_candidate_count
                ),
            })
        except Exception as exc:
            meta['mechanistic_status']='resolution_failed'
            meta['mechanistic_error']=f'{type(exc).__name__}: {exc}'
        return chart,strata,meta

    def _returned_set_biological_relation(
        self,
        direction: str,
        scores: np.ndarray,
        query_meta: dict[str, Any],
        applied_seed_count: int,
        returned_indices: np.ndarray,
        candidate_support_distance: np.ndarray,
        target_conditions: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[int, dict[str, Any]], dict[str, Any]]:
        """Return the non-order-bearing FIBRE relation on returned candidates.

        The offline full-atlas/strict-inductive evaluators define the scientific
        relation. Online we evaluate exactly the same fixed-coordinate Pareto
        rule only on the already selected candidate set so the explanation is
        cheap and cannot affect which candidates were selected.
        """
        idx=np.asarray(returned_indices,dtype=np.int64).reshape(-1)
        front=np.full(len(scores),-1,dtype=np.int64)
        complete=np.zeros(len(scores),dtype=bool)
        to_top=np.full(len(scores),'not_in_returned_set',dtype=object)
        candidate_states: dict[int,dict[str,Any]]={}
        support=np.asarray(candidate_support_distance,dtype=np.float64).reshape(-1)
        if len(support)!=len(scores):
            raise ValueError('candidate_support_distance must align with score vector')
        info: dict[str,Any]={
            'schema':'fibre-biological-relation-v2',
            'status':'global_only',
            'order_bearing':False,
            'canonical_rank_unchanged':True,
            'relation_scope':'returned_candidate_set',
            'front_semantics':'relative_to_returned_candidate_set_only',
            'coordinate_family':['global_correspondence',*list(self.local_coordinates)],
            'comparison_rule':'Pareto minimization on a fixed coordinate family; no weights and no lexicographic priority',
            'missing_policy':'missing any declared coordinate leaves that candidate unordered in this relation',
            'strict_inductive_validation':'audited_non_order_bearing_relation',
            'compatibility_note':'legacy stratified_correspondence remains available but is not the primary scientific relation',
        }
        if not len(idx):
            info['status']='empty_returned_set'
            return front,complete,to_top,candidate_states,info
        if self.stratified_manifest is None or not self.local_states:
            info['status']='local_geometry_unavailable'
            return front,complete,to_top,candidate_states,info
        if int(applied_seed_count)>0:
            info['status']='relation_not_projected_through_dynamic_positive_update'
            return front,complete,to_top,candidate_states,info
        if not bool(query_meta.get('query_is_reference_entity')):
            info['status']='relation_unavailable_for_external_query'
            return front,complete,to_top,candidate_states,info

        canonical=str(query_meta.get('canonical_query_id') or '')
        global_defect=-np.asarray(scores,dtype=np.float64).reshape(-1)
        local=np.full((len(self.local_coordinates),len(idx)),np.nan,dtype=np.float64)
        local_available=np.zeros_like(local,dtype=bool)
        try:
            if direction=='reaction_to_enzyme':
                r=self.ri[canonical]
                for j,gidx in enumerate(idx):
                    local_row=self.local_global_to_row.get(int(gidx))
                    if local_row is None:
                        continue
                    for c,coord in enumerate(self.local_coordinates):
                        local[c,j]=float(self.local_states[coord].defect[r,local_row])
                        local_available[c,j]=True
            elif direction=='enzyme_to_reaction':
                e=self.pi[canonical]
                if not bool(self.local_common[e]):
                    info['status']='reference_query_without_complete_pocket_consensus'
                    return front,complete,to_top,candidate_states,info
                local_row=self.local_global_to_row[e]
                for c,coord in enumerate(self.local_coordinates):
                    local[c,:]=np.asarray(
                        self.local_states[coord].defect[idx,local_row],
                        dtype=np.float64,
                    )
                    local_available[c,:]=True
            else:
                raise ValueError(f'unsupported FIBRE direction: {direction}')

            defects=np.vstack([global_defect[idx][None,:],local])
            available=np.vstack([
                np.ones((1,len(idx)),dtype=bool),
                local_available,
            ])
            relation=partial_correspondence_relation(
                defects,available,coordinate_names=(
                    'global_correspondence',*self.local_coordinates
                ),
            )
        except Exception as exc:
            info['status']='relation_failed'
            info['error']=f'{type(exc).__name__}: {exc}'
            return front,complete,to_top,candidate_states,info

        context_target=context_from_target_conditions(target_conditions)
        context_rows=[] if context_target.observed_dimensions else None
        catalytic_components=[]
        if self.enzymology_evidence is not None:
            for gidx in idx:
                if direction=='reaction_to_enzyme':
                    protein_id=self.protein_ids[int(gidx)]
                    reaction_id=canonical
                else:
                    protein_id=canonical
                    reaction_id=self.reaction_ids[int(gidx)]
                state=self.enzymology_evidence.state(protein_id,reaction_id)
                catalytic_components.append(tuple(state.get('observed_components') or ()))
                if context_rows is not None:
                    context_rows.append(assess_pair_context(
                        self.enzymology_evidence.assay_observations(protein_id,reaction_id),
                        context_target,
                        enzyme_id=protein_id,
                        reaction_id=reaction_id,
                    ))
        else:
            catalytic_components=[() for _ in idx]
            if context_rows is not None:
                context_rows=[
                    assess_pair_context(
                        (),context_target,enzyme_id='',reaction_id=''
                    )
                    for _ in idx
                ]

        biological=biological_correspondence_relation(
            relation,
            support[idx],
            context=context_rows,
            catalytic_state_components=catalytic_components,
        )
        front[idx]=relation.front
        complete[idx]=relation.complete
        top_local=0
        for j,gidx in enumerate(idx):
            to_top[int(gidx)]=biological.relation(j,top_local)
            candidate_states[int(gidx)]=biological.candidate_state(j).to_dict()
        info.update({
            'status':'available_non_order_bearing',
            'complete_candidate_count':int(relation.complete_count),
            'returned_candidate_count':int(len(idx)),
            'complete_candidate_fraction':float(
                relation.complete_count/len(idx)
            ),
            'pareto_front_count':int(relation.front_count),
            'pareto_front_sizes':[int(x) for x in relation.front_sizes],
            'dominance_pair_count':int(np.sum(relation.dominance)),
            'biological_state':biological.summary(),
            'requested_assay_dimensions':list(context_target.observed_dimensions),
            'context_constraint_authority':(
                'matched pair-specific inactive/below-detection assay only'
            ),
            'support_role':'applicability_only_not_ordering',
        })
        return front,complete,to_top,candidate_states,info

    def _stratified_section(
        self,
        direction: str,
        scores: np.ndarray,
        query_meta: dict[str, Any],
        applied_seed_count: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
        """Return rank-preserving three-resolution FIBRE metadata.

        Global correspondence defines the canonical total rank. Catalytic
        pocket strata and family-aware mechanistic charts are finer relations
        inside that object and cannot reorder the deployed candidate list.
        """
        defect=-np.asarray(scores,dtype=np.float64).reshape(-1)
        coarse_level,_tol=stable_level_ids(defect)
        catalytic=np.full(len(defect),-1,dtype=np.int64)
        mechanistic_chart=np.zeros(len(defect),dtype=np.int64)
        mechanistic_stratum=np.full(len(defect),-1,dtype=np.int64)
        info: dict[str,Any]={
            'schema':'fibre-stratified-section-v2',
            'status':'coarse_only',
            'total_rank_source':'coarse_global_correspondence',
            'catalytic_strata_order_bearing':False,
            'mechanistic_strata_order_bearing':False,
            'promotion_status':'not_promoted_strict_inductive_non_degradation_gate_failed',
            'local_coordinates':list(self.local_coordinates),
            'mechanistic_coordinate_role':'third-resolution family-aware mechanism chart; never a scalar bonus',
            'coarse_level_count':int(np.max(coarse_level)+1) if len(coarse_level) else 0,
            'observed_coarse_levels':[],
        }
        if self.stratified_manifest is None or not self.local_states:
            info['status']='local_geometry_unavailable'
            if self.stratified_load_error:
                info['load_error']=self.stratified_load_error
            return (
                coarse_level,catalytic,
                mechanistic_chart,mechanistic_stratum,info
            )
        if int(applied_seed_count)>0:
            info['status']='local_resolution_not_projected_through_dynamic_positive_update'
            return (
                coarse_level,catalytic,
                mechanistic_chart,mechanistic_stratum,info
            )
        if not bool(query_meta.get('query_is_reference_entity')):
            info['status']='local_resolution_unavailable_for_external_query'
            return (
                coarse_level,catalytic,
                mechanistic_chart,mechanistic_stratum,info
            )

        canonical=str(query_meta.get('canonical_query_id') or '')
        try:
            if direction=='reaction_to_enzyme':
                r=self.ri[canonical]
                local=np.full(
                    (len(self.local_coordinates),len(self.protein_ids)),
                    np.nan,dtype=np.float64,
                )
                for c,coord in enumerate(self.local_coordinates):
                    local[c,self.local_global_rows]=self.local_states[coord].defect[r]
                available=np.repeat(
                    self.local_common[None,:],len(self.local_coordinates),axis=0
                )
                resolution=consensus_stratified_resolution(
                    defect,local,available
                )
            elif direction=='enzyme_to_reaction':
                e=self.pi[canonical]
                info['query_mechanistic_coordinates']=self._mechanistic_coordinates_for_protein(e)
                info['query_mechanistic_chart']=self._mechanistic_chart_for_protein(e)
                if not bool(self.local_common[e]):
                    info['status']='reference_query_without_complete_pocket_consensus'
                    return (
                        coarse_level,catalytic,
                        mechanistic_chart,mechanistic_stratum,info
                    )
                le=self.local_global_to_row[e]
                local=np.vstack([
                    self.local_states[coord].defect[:,le]
                    for coord in self.local_coordinates
                ])
                available=np.ones_like(local,dtype=bool)
                resolution=consensus_stratified_resolution(
                    defect,local,available
                )
            else:
                raise ValueError(f'unsupported FIBRE direction: {direction}')
        except Exception as exc:
            info['status']='local_resolution_failed'
            info['error']=f'{type(exc).__name__}: {exc}'
            return (
                coarse_level,catalytic,
                mechanistic_chart,mechanistic_stratum,info
            )

        catalytic=np.asarray(resolution.catalytic_stratum,dtype=np.int64)
        info.update({
            'status':'available_non_order_bearing',
            'observed_coarse_levels':[int(x) for x in resolution.observed_coarse_levels],
            'observed_coarse_level_count':int(len(resolution.observed_coarse_levels)),
            'refined_coarse_level_count':int(len(resolution.refined_coarse_levels)),
            'catalytic_stratum_candidate_count':int(np.sum(catalytic>=0)),
            'local_relation':'Pareto consensus across independent pocket-local ESM-C, pocket-3Di and pocket-OT correspondence coordinates',
        })
        mechanistic_chart,mechanistic_stratum,mmeta=self._mechanistic_relation(
            direction,canonical,resolution
        )
        info.update(mmeta)
        return (
            coarse_level,catalytic,
            mechanistic_chart,mechanistic_stratum,info
        )

    def _seed_reference_correspondence_state(self):
        if self._seed_reference_state is None:
            with self._seed_stability_lock:
                if self._seed_reference_state is None:
                    self._seed_reference_state = correspondence_state(
                        self.Dr2, self.Dp2, self.positive_pairs
                    )
        return self._seed_reference_state

    def _registered_seed_influence(
        self,
        direction: str,
        canonical_query_id: str | None,
        requested_seed_ids: list[str] | tuple[str, ...] | None,
    ) -> list[dict[str, Any]]:
        canonical=str(canonical_query_id or '')
        requested=[str(x) for x in (requested_seed_ids or [])]
        if not canonical or not requested:
            return []
        pairs: list[tuple[str,str,int,int]]=[]
        if direction == 'reaction_to_enzyme':
            if canonical not in self.ri:
                return []
            rr=self.ri[canonical]
            for raw in requested:
                internal=self._protein_internal(raw)
                if internal is not None:
                    pairs.append((raw,internal,rr,self.pi[internal]))
        elif direction == 'enzyme_to_reaction':
            if canonical not in self.pi:
                return []
            ee=self.pi[canonical]
            for raw in requested:
                internal=self._reaction_internal(raw)
                if internal is not None:
                    pairs.append((raw,internal,self.ri[internal],ee))
        else:
            raise ValueError(f'unsupported FIBRE direction: {direction}')
        if not pairs:
            return []
        state=self._seed_reference_correspondence_state()
        out=[]
        for raw,internal,rr,ee in pairs:
            influence=seed_influence(
                state,self.Dr2,self.Dp2,self.positive_pairs,rr,ee
            )
            row=asdict(influence)
            row.update({
                'requested_seed_id':raw,
                'canonical_seed_id':internal,
                'canonical_query_id':canonical,
                'reference_relation':'canonical_omega_before_request',
            })
            out.append(row)
        return out

    def _seed_update_stability(
        self,
        direction: str,
        payload: dict[str, Any],
        meta: dict[str, Any],
        applied_seed_count: int,
        before_defect: np.ndarray,
        after_defect: np.ndarray,
    ) -> dict[str, Any]:
        count=int(applied_seed_count)
        base={
            'schema':'fibre-seed-update-stability-v1',
            'status':'not_applied' if count == 0 else 'applied_exact',
            'seed_count':count,
            'update_rule':'verified positives enter Omega by exact pointwise minima',
            'verified_seed_weight_policy':'exact_observation_no_downweighting',
            'interpretation':'descriptive stability provenance; never a confidence score or seed gate',
        }
        if count == 0:
            return base
        base['query_section_influence']=asdict(
            section_update_influence(before_defect,after_defect)
        )
        seed_key=(
            'known_enzyme_ids'
            if direction == 'reaction_to_enzyme'
            else 'known_reaction_ids'
        )
        registered=self._registered_seed_influence(
            direction,
            meta.get('canonical_query_id'),
            payload.get(seed_key),
        )
        base['registered_seed_influence']=registered
        base['registered_seed_influence_count']=len(registered)
        base['global_influence_scope']=(
            'one canonical seed at a time relative to canonical Omega; '
            'combined multi-seed effect is represented by query_section_influence'
        )
        return base

    def _protein_seed_distance_sq(
        self,
        values: list[str] | tuple[str, ...] | None,
        external_csv: str | Path | None = None,
    ) -> tuple[np.ndarray | None, list[str], int]:
        requested = [str(value) for value in (values or [])]
        mapped, missing = self._map_protein_indices(requested)
        distance_vectors: list[np.ndarray] = [self.Dp2[:, index] for index in mapped]
        external_by_id: dict[str, str] = {}
        if external_csv:
            path = Path(external_csv)
            if path.is_file():
                frame = pd.read_csv(path, dtype=str).fillna('')
                id_col = 'enzyme_id' if 'enzyme_id' in frame.columns else 'Entry' if 'Entry' in frame.columns else None
                seq_col = 'sequence' if 'sequence' in frame.columns else 'Sequence' if 'Sequence' in frame.columns else None
                if id_col and seq_col:
                    external_by_id = {
                        str(row[id_col]): ''.join(str(row[seq_col]).split()).upper()
                        for _, row in frame.iterrows()
                        if str(row[id_col]).strip() and str(row[seq_col]).strip()
                    }

        still_missing: list[str] = []
        batch_ids: list[str] = []
        batch_sequences: list[str] = []
        for value in missing:
            sequence = external_by_id.get(value, '')
            if not sequence:
                still_missing.append(value)
                continue
            reference_internal = self.sequence_to_internal.get(sequence)
            if reference_internal is not None:
                distance_vectors.append(self.Dp2[:, self.pi[reference_internal]])
                continue
            batch_ids.append(value)
            batch_sequences.append(sequence)

        if batch_sequences:
            frame = pd.DataFrame({'enzyme_id': batch_ids, 'sequence': batch_sequences})
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            with self._protein_encode_lock:
                matrix, audits = encode_external_enzymes_with_audit(frame, device, 'esmc_600m')
            if len(matrix) != len(batch_sequences):
                raise RuntimeError('external positive-seed encoder returned an unexpected batch size')
            for embedding in matrix:
                geodesic, _attachment_count = self._protein_oos_distances_from_embedding(embedding)
                distance_vectors.append(np.square(geodesic))

        if not distance_vectors:
            return None, still_missing, 0
        return np.min(np.stack(distance_vectors, axis=1), axis=1), still_missing, len(distance_vectors)

    def rank_enzymes(self, payload: dict[str, Any]) -> dict[str, Any]:
        dq, meta = self.reaction_distances(
            reaction_id=str(payload.get('reaction_id') or ''),
            reaction_smiles=str(payload.get('reaction_smiles') or ''),
        )
        dq2 = np.square(dq)
        joint = np.full(len(self.protein_ids), np.inf, dtype=np.float64)
        for rr, ee in self.positive_pairs:
            joint = np.minimum(joint, dq2[rr] + self.Dp2[:, ee])
        mr = float(np.min(dq2[self.r_support]))
        me = self.protein_marginal_sq.copy()
        before_seed_defect=np.maximum(joint - mr - me, 0.0)

        seed_distance, missing_seeds, applied_seed_count = self._protein_seed_distance_sq(
            payload.get('known_enzyme_ids'), payload.get('external_enzymes_csv')
        )
        if seed_distance is not None:
            joint = np.minimum(joint, seed_distance)
            mr = 0.0
            me = np.minimum(me, seed_distance)
        after_seed_defect=np.maximum(joint - mr - me, 0.0)
        scores = -after_seed_defect
        seed_stability=self._seed_update_stability(
            'reaction_to_enzyme',payload,meta,applied_seed_count,
            before_seed_defect,after_seed_defect,
        )

        eligible = np.ones(len(scores), dtype=bool)
        candidate_indices, missing_candidates = self._map_protein_indices(payload.get('candidate_ids'))
        if payload.get('candidate_ids') is not None:
            eligible[:] = False
            eligible[candidate_indices] = True
        mask_indices, missing_masks = self._map_protein_indices(payload.get('mask_enzyme_ids'))
        if mask_indices:
            eligible[mask_indices] = False
        context_excluded, assay_context_constraint = self._assay_context_censor(
            'reaction_to_enzyme', meta, payload.get('target_conditions'), eligible
        )
        eligible[context_excluded] = False
        uncertainty = self._geometric_uncertainty(scores, eligible, mr, me)
        (
            coarse_levels,
            catalytic_strata,
            mechanistic_charts,
            mechanistic_strata,
            stratified,
        ) = self._stratified_section(
            'reaction_to_enzyme', scores, meta, applied_seed_count
        )
        tps_domain_defect, application_refinement = self._tps_domain_defect(
            'reaction_to_enzyme',payload,meta
        )
        order = self._application_refined_order(
            scores,self.protein_primary,eligible,int(payload.get('top_k') or 10),
            coarse_levels,tps_domain_defect,
        )
        (
            biological_front,
            biological_complete,
            biological_to_top,
            biological_candidate_states,
            biological_relation,
        ) = self._returned_set_biological_relation(
            'reaction_to_enzyme',scores,meta,applied_seed_count,order,
            np.sqrt(np.maximum(me,0.0)),
            target_conditions=payload.get('target_conditions'),
        )
        best = uncertainty.get('best_defect')
        tol = float(uncertainty.get('numerical_level_tolerance') or 0.0)
        candidates = [
            {
                'rank': rank,
                'candidate_id': str(self.protein_primary[index]),
                'canonical_candidate_id': self.protein_ids[index],
                'score': float(scores[index]),
                'correspondence_defect': float(-scores[index]),
                'support_applicability': {
                    'canonical_candidate_support_distance': float(
                        np.sqrt(max(float(self.protein_marginal_sq[index]),0.0))
                    ),
                    'current_candidate_support_distance': float(
                        np.sqrt(max(float(me[index]),0.0))
                    ),
                    'query_support_distance': float(np.sqrt(max(float(mr),0.0))),
                    'nearest_joint_positive_distance': product_support_diagnostics(
                        float(-scores[index]),float(mr),float(me[index])
                    ).joint_precedent_distance,
                    'interpretation': (
                        'intrinsic distance to accepted positive support; '
                        'not an activity probability and not an ordering bonus'
                    ),
                },
                'enzymology_state': self._enzymology_state(
                    self.protein_ids[index],
                    str(meta.get('canonical_query_id') or ''),
                ),
                'in_best_numerical_level': bool(
                    best is not None and abs(float(-scores[index]) - float(best)) <= tol
                ),
                'fibre_relation': {
                    'pareto_front': (
                        int(biological_front[index])
                        if int(biological_front[index]) >= 0 else None
                    ),
                    'coordinate_complete': bool(biological_complete[index]),
                    'relation_to_display_rank_1': str(biological_to_top[index]),
                    'biological_state': dict(biological_candidate_states.get(int(index)) or {}),
                    'order_bearing': False,
                },
                'fibre_resolution': {
                    'coarse_level': int(coarse_levels[index]),
                    'catalytic_stratum': (
                        int(catalytic_strata[index])
                        if int(catalytic_strata[index]) >= 0 else None
                    ),
                    'catalytic_observed': bool(
                        int(coarse_levels[index]) in (
                            stratified.get('observed_coarse_levels') or []
                        )
                    ),
                    'mechanistic_chart': self._mechanistic_names_from_mask(
                        int(mechanistic_charts[index])
                    ),
                    'mechanistic_stratum': (
                        int(mechanistic_strata[index])
                        if int(mechanistic_strata[index]) >= 0 else None
                    ),
                    'mechanistic_coordinates': self._mechanistic_coordinates_for_protein(index),
                },
                'selection_source': 'fibre',
                'application_refinement': {
                    'profile': 'starase-application',
                    'tps_domain_defect': (
                        float(tps_domain_defect[index])
                        if tps_domain_defect is not None
                        and np.isfinite(tps_domain_defect[index])
                        else None
                    ),
                    'within_primary_level_order_bearing': bool(
                        tps_domain_defect is not None
                    ),
                    'pair_supervised': bool(tps_domain_defect is not None),
                    'benchmark_evidence': False,
                },
                'evidence_passport': {},
            }
            for rank, index in enumerate(order, start=1)
        ]
        query = self._query_metadata('reaction_to_enzyme', payload, meta, len(scores), missing_seeds, missing_candidates, missing_masks, applied_seed_count)
        query['geometric_uncertainty'] = uncertainty
        query['assay_context_constraint'] = assay_context_constraint
        query['enzymology_evidence_index'] = self.enzymology_evidence_status()
        query['seed_update_stability'] = seed_stability
        query['biological_relation'] = biological_relation
        query['stratified_correspondence'] = stratified
        query['application_profile'] = {
            **self.application_profile_status(),
            'tps_domain_refinement': application_refinement,
            'ordering_policy': (
                'primary FIBRE numerical level -> TPS-domain FIBRE defect -> stable id'
            ),
        }
        return {
            'query': query,
            'candidates': candidates,
        }

    def rank_reactions(self, payload: dict[str, Any]) -> dict[str, Any]:
        dq, meta = self.protein_distances(
            enzyme_id=str(payload.get('enzyme_id') or ''),
            enzyme_sequence=str(payload.get('enzyme_sequence') or ''),
            protein_structure_path=str(payload.get('protein_structure_path') or ''),
        )
        dq2 = np.square(dq)
        joint = np.full(len(self.reaction_ids), np.inf, dtype=np.float64)
        for rr, ee in self.positive_pairs:
            joint = np.minimum(joint, self.Dr2[:, rr] + dq2[ee])
        me = float(np.min(dq2[self.e_support]))
        mr = self.reaction_marginal_sq.copy()
        before_seed_defect=np.maximum(joint - me - mr, 0.0)

        seed_indices, missing_seeds = self._map_reaction_indices(payload.get('known_reaction_ids'))
        applied_seed_count = len(seed_indices)
        if seed_indices:
            seed_distance = np.min(self.Dr2[:, seed_indices], axis=1)
            joint = np.minimum(joint, seed_distance)
            me = 0.0
            mr = np.minimum(mr, seed_distance)
        after_seed_defect=np.maximum(joint - me - mr, 0.0)
        scores = -after_seed_defect
        seed_stability=self._seed_update_stability(
            'enzyme_to_reaction',payload,meta,applied_seed_count,
            before_seed_defect,after_seed_defect,
        )

        eligible = np.ones(len(scores), dtype=bool)
        candidate_indices, missing_candidates = self._map_reaction_indices(payload.get('candidate_ids'))
        if payload.get('candidate_ids') is not None:
            eligible[:] = False
            eligible[candidate_indices] = True
        mask_indices, missing_masks = self._map_reaction_indices(payload.get('mask_reaction_ids'))
        if mask_indices:
            eligible[mask_indices] = False
        context_excluded, assay_context_constraint = self._assay_context_censor(
            'enzyme_to_reaction', meta, payload.get('target_conditions'), eligible
        )
        eligible[context_excluded] = False
        uncertainty = self._geometric_uncertainty(scores, eligible, me, mr)
        (
            coarse_levels,
            catalytic_strata,
            mechanistic_charts,
            mechanistic_strata,
            stratified,
        ) = self._stratified_section(
            'enzyme_to_reaction', scores, meta, applied_seed_count
        )
        tps_domain_defect, application_refinement = self._tps_domain_defect(
            'enzyme_to_reaction',payload,meta
        )
        order = self._application_refined_order(
            scores,self.reaction_primary,eligible,int(payload.get('top_k') or 10),
            coarse_levels,tps_domain_defect,
        )
        (
            biological_front,
            biological_complete,
            biological_to_top,
            biological_candidate_states,
            biological_relation,
        ) = self._returned_set_biological_relation(
            'enzyme_to_reaction',scores,meta,applied_seed_count,order,
            np.sqrt(np.maximum(mr,0.0)),
            target_conditions=payload.get('target_conditions'),
        )
        best = uncertainty.get('best_defect')
        tol = float(uncertainty.get('numerical_level_tolerance') or 0.0)
        candidates = []
        for rank, index in enumerate(order, start=1):
            row = self.reactions.iloc[index]
            candidates.append({
                'rank': rank,
                'candidate_id': str(row.primary_alias),
                'canonical_candidate_id': self.reaction_ids[index],
                'reaction_aliases': [value for value in str(row.aliases).split(';') if value],
                'score': float(scores[index]),
                'correspondence_defect': float(-scores[index]),
                'support_applicability': {
                    'canonical_candidate_support_distance': float(
                        np.sqrt(max(float(self.reaction_marginal_sq[index]),0.0))
                    ),
                    'current_candidate_support_distance': float(
                        np.sqrt(max(float(mr[index]),0.0))
                    ),
                    'query_support_distance': float(np.sqrt(max(float(me),0.0))),
                    'nearest_joint_positive_distance': product_support_diagnostics(
                        float(-scores[index]),float(mr[index]),float(me)
                    ).joint_precedent_distance,
                    'interpretation': (
                        'intrinsic distance to accepted positive support; '
                        'not an activity probability and not an ordering bonus'
                    ),
                },
                'enzymology_state': self._enzymology_state(
                    str(meta.get('canonical_query_id') or ''),
                    self.reaction_ids[index],
                ),
                'in_best_numerical_level': bool(
                    best is not None and abs(float(-scores[index]) - float(best)) <= tol
                ),
                'fibre_relation': {
                    'pareto_front': (
                        int(biological_front[index])
                        if int(biological_front[index]) >= 0 else None
                    ),
                    'coordinate_complete': bool(biological_complete[index]),
                    'relation_to_display_rank_1': str(biological_to_top[index]),
                    'biological_state': dict(biological_candidate_states.get(int(index)) or {}),
                    'order_bearing': False,
                },
                'fibre_resolution': {
                    'coarse_level': int(coarse_levels[index]),
                    'catalytic_stratum': (
                        int(catalytic_strata[index])
                        if int(catalytic_strata[index]) >= 0 else None
                    ),
                    'catalytic_observed': bool(
                        int(coarse_levels[index]) in (
                            stratified.get('observed_coarse_levels') or []
                        )
                    ),
                    'mechanistic_chart': self._mechanistic_names_from_mask(
                        int(mechanistic_charts[index])
                    ),
                    'mechanistic_stratum': (
                        int(mechanistic_strata[index])
                        if int(mechanistic_strata[index]) >= 0 else None
                    ),
                    'mechanistic_coordinates': list(
                        stratified.get('query_mechanistic_coordinates') or []
                    ),
                },
                'selection_source': 'fibre',
                'application_refinement': {
                    'profile': 'starase-application',
                    'tps_domain_defect': (
                        float(tps_domain_defect[index])
                        if tps_domain_defect is not None
                        and np.isfinite(tps_domain_defect[index])
                        else None
                    ),
                    'within_primary_level_order_bearing': bool(
                        tps_domain_defect is not None
                    ),
                    'pair_supervised': bool(tps_domain_defect is not None),
                    'benchmark_evidence': False,
                },
                'evidence_passport': {},
            })
        query = self._query_metadata('enzyme_to_reaction', payload, meta, len(scores), missing_seeds, missing_candidates, missing_masks, applied_seed_count)
        query['geometric_uncertainty'] = uncertainty
        query['assay_context_constraint'] = assay_context_constraint
        query['enzymology_evidence_index'] = self.enzymology_evidence_status()
        query['seed_update_stability'] = seed_stability
        query['biological_relation'] = biological_relation
        query['stratified_correspondence'] = stratified
        query['application_profile'] = {
            **self.application_profile_status(),
            'tps_domain_refinement': application_refinement,
            'ordering_policy': (
                'primary FIBRE numerical level -> TPS-domain FIBRE defect -> stable id'
            ),
        }
        return {
            'query': query,
            'candidates': candidates,
        }

    def _query_metadata(
        self,
        direction: str,
        payload: dict[str, Any],
        meta: dict[str, Any],
        universe_size: int,
        missing_seeds: list[str],
        missing_candidates: list[str],
        missing_masks: list[str],
        applied_seed_count: int,
    ) -> dict[str, Any]:
        seed_key = 'known_enzyme_ids' if direction == 'reaction_to_enzyme' else 'known_reaction_ids'
        has_seed = int(applied_seed_count) > 0
        return {
            'query_id': str(payload.get('query_id') or payload.get('reaction_id') or payload.get('enzyme_id') or ''),
            'direction': direction,
            'ranking_objective': str(payload.get('ranking_objective') or 'top10'),
            'route_id': f'fibre-{"r2e" if direction == "reaction_to_enzyme" else "e2r"}-v1' + ('+fewshot' if has_seed else ''),
            'route_version': self.version,
            'candidate_universe_version': self.version,
            'candidate_universe_size': int(universe_size),
            'model_bundle_version': self.version,
            'registry_version': 'fibre-reference-omega',
            'score_source': 'correspondence_geometry',
            'query_is_current_entity': bool(meta.get('query_is_reference_entity')),
            'scope': 'current' if meta.get('query_is_reference_entity') else 'external',
            'shot_mode': 'few_shot' if has_seed else 'zero_shot',
            'seed_context_applied': has_seed,
            'seed_context_seed_count': int(applied_seed_count),
            'empirical_reliability_status': 'not_calibrated_correspondence_geometry',
            'empirical_reliability_tier': 'uncalibrated',
            'reliability_recommendation': 'inspect_geometric_witness_and_biological_evidence',
            'evidence_passport': {},
            'conformal_retrieval_set': {
                'status': 'not_calibrated_correspondence_geometry',
                'guarantee_scope': 'none',
                'recommendation': 'inspect_geometric_witness_and_biological_evidence',
            },
            'observation_execution': {
                'executed_measurements': list(meta.get('executed_measurements') or []),
                'failed_measurements': dict(meta.get('failed_measurements') or {}),
                'query_is_reference_entity': bool(meta.get('query_is_reference_entity')),
                'attachment_count': meta.get('attachment_count'),
            },
            'canonical_query_id': meta.get('canonical_query_id'),
            'candidate_subset_missing_ids': missing_candidates,
            'seed_missing_ids': missing_seeds,
            'mask_missing_ids': missing_masks,
        }

    def rank(self, command: str, payload: dict[str, Any]) -> dict[str, Any]:
        if command == 'rank-enzymes':
            return self.rank_enzymes(payload)
        if command == 'rank-reactions':
            return self.rank_reactions(payload)
        raise ValueError(f'unsupported correspondence command: {command}')
