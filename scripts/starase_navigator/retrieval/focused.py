from __future__ import annotations

from dataclasses import asdict, replace
import json
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

ROOT = Path(__file__).resolve().parents[3]
ATLAS = ROOT / 'data/terpene_correspondence_deployment_atlas_v2'
OMEGA = ROOT / 'data/terpene_marts_adaptation/marts_pair_folds.csv'
STRUCTURAL_BASIS = ROOT / 'data/terpene_correspondence_structural_basis_v1'
CATALYTIC_CONSENSUS = ROOT / 'data/terpene_catalytic_consensus_geometry_v1'
MECHANISTIC_CHART = ROOT / 'data/terpene_mechanistic_chart_geometry_v1'
PROTEIN_GEOMETRY = ROOT / 'data/terpene_multiresolution_protein_geometry_v4'
STRUCTURAL_WORK_ROOT = ROOT / 'results/starase_navigator_runtime/tmp'


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
        self._load_stratified_geometry()

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
        order = self._rank_order(scores, self.protein_primary, eligible, int(payload.get('top_k') or 10))
        best = uncertainty.get('best_defect')
        tol = float(uncertainty.get('numerical_level_tolerance') or 0.0)
        candidates = [
            {
                'rank': rank,
                'candidate_id': str(self.protein_primary[index]),
                'canonical_candidate_id': self.protein_ids[index],
                'score': float(scores[index]),
                'correspondence_defect': float(-scores[index]),
                'in_best_numerical_level': bool(
                    best is not None and abs(float(-scores[index]) - float(best)) <= tol
                ),
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
                'evidence_passport': {},
            }
            for rank, index in enumerate(order, start=1)
        ]
        query = self._query_metadata('reaction_to_enzyme', payload, meta, len(scores), missing_seeds, missing_candidates, missing_masks, applied_seed_count)
        query['geometric_uncertainty'] = uncertainty
        query['seed_update_stability'] = seed_stability
        query['stratified_correspondence'] = stratified
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
        order = self._rank_order(scores, self.reaction_primary, eligible, int(payload.get('top_k') or 10))
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
                'in_best_numerical_level': bool(
                    best is not None and abs(float(-scores[index]) - float(best)) <= tol
                ),
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
                'evidence_passport': {},
            })
        query = self._query_metadata('enzyme_to_reaction', payload, meta, len(scores), missing_seeds, missing_candidates, missing_masks, applied_seed_count)
        query['geometric_uncertainty'] = uncertainty
        query['seed_update_stability'] = seed_stability
        query['stratified_correspondence'] = stratified
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
