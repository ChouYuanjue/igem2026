"""Inductive enzyme-reaction evidence over documented biochemical associations.

For a query protein unobserved in the training association graph:
* A protein-latent neighborhood of previously observed enzymes supplies
  real documented reaction links, with per-protein degree normalization.
* A reaction-latent neighborhood supplies similarity-weighted documented
  enzyme prototypes to zero-degree reaction nodes.
Both contributions are missing-neutral when graph evidence is absent.

All embeddings, observed edges and calibration weights are frozen. This
class neither sees heldout labels nor changes the original candidate pool.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import torch
from scipy.sparse import csr_matrix


@dataclass(frozen=True)
class InductiveRelationConfig:
    observed_coefficient: float
    inferred_coefficient: float
    nearest_proteins: int = 16
    nearest_reactions: int = 8
    reaction_neighbor_pool: int = 64
    protein_attention_temperature: float = 40.0
    reaction_attention_temperature: float = 12.0
    prior_normalizer: float = 0.5
    reaction_min_cosine: float = 0.35


class InductiveE2RRelation:
    """Two-sided observed-positive reaction evidence for E2R only."""

    def __init__(self, runtime: object, config: InductiveRelationConfig):
        self.rt = runtime
        self.cfg = config
        self.nr = len(runtime.index.reaction_ids)
        self.np = len(runtime.index.protein_ids)
        if not (config.observed_coefficient >= 0 and config.inferred_coefficient >= 0):
            raise ValueError("Inductive evidence coefficients must be nonnegative")
        pr = runtime.train.protein_id.map(runtime.index.protein_index).to_numpy(np.int64)
        rr = runtime.train.reaction_id.map(runtime.index.reaction_index).to_numpy(np.int64)
        if np.any(pr < 0) or np.any(rr < 0):
            raise ValueError("Documented catalytic graph has unknown entity")
        graph = csr_matrix(
            (np.ones(len(pr), np.float32), (pr, rr)), shape=(self.np, self.nr)
        )
        graph.sort_indices()
        if graph.nnz != len(pr):
            raise RuntimeError("Observed training graph has duplicates; validate edge counts")
        self.graph = graph
        self.protein_degree = np.diff(graph.indptr)
        self.reaction_degree = np.bincount(rr, minlength=self.nr)
        self.reaction_seen = self.reaction_degree > 0
        self.protein_seen = self.protein_degree > 0
        self.reaction_prior = self.reaction_degree / float(len(pr))
        if not np.array_equal(self.reaction_seen, runtime.rmask.cpu().numpy()):
            raise RuntimeError("Reaction graph availability differs from canonical runtime")
        if not np.array_equal(self.protein_seen, runtime.pmask.cpu().numpy()):
            raise RuntimeError("Protein graph availability differs from canonical runtime")

        device = runtime.index.device
        self.seen_protein_rows = np.flatnonzero(self.protein_seen)
        self.seen_protein_embeddings = torch.nn.functional.normalize(
            runtime.index.protein_embeddings.index_select(
                0, torch.as_tensor(self.seen_protein_rows, device=device)
            ), dim=1
        )
        self.seen_reaction_rows = np.flatnonzero(self.reaction_seen)
        self.seen_reaction_prototypes = runtime.rproto.index_select(
            0, torch.as_tensor(self.seen_reaction_rows, device=device)
        )
        # New/cold reaction prototypes and confidence are computed entirely
        # from reaction feature vectors plus the verified training graph.
        self.inferred_proto = torch.zeros_like(runtime.rproto)
        confidence = np.zeros(self.nr, np.float32)
        reaction_embedding = runtime.index.reaction_embeddings
        with torch.no_grad():
            for start in range(0, self.nr, 256):
                stop = min(start + 256, self.nr)
                score = (reaction_embedding[start:stop] @ reaction_embedding.T).float()
                diag = torch.arange(start, stop, device=device)
                score[torch.arange(stop-start, device=device), diag] = -1.0
                values, indices = torch.topk(
                    score, min(config.reaction_neighbor_pool, self.nr - 1),
                    dim=1, sorted=True
                )
                vv = values.cpu().numpy()
                jj = indices.cpu().numpy()
                for local in range(stop-start):
                    row = start + local
                    if self.reaction_seen[row]:
                        continue
                    available = self.reaction_seen[jj[local]]
                    pick = np.flatnonzero(available)[:config.nearest_reactions]
                    if len(pick) == 0:
                        continue
                    ids = jj[local, pick]
                    similarities = np.maximum(vv[local, pick], 0.0)
                    weights = np.exp(
                        config.reaction_attention_temperature * (similarities - 1.0)
                    )
                    if float(weights.sum()) < 1e-8:
                        continue
                    pool = runtime.rproto.index_select(
                        0, torch.as_tensor(ids, dtype=torch.long, device=device)
                    )
                    ws = torch.as_tensor(
                        weights, dtype=pool.dtype, device=device
                    )
                    combined = (pool * ws[:, None]).sum(0) / ws.sum()
                    norm = torch.linalg.vector_norm(combined)
                    if norm < 1e-8:
                        continue
                    self.inferred_proto[row] = combined / norm
                    confidence[row] = float(similarities[0])
        self.cold_trust = (
            np.maximum(
                (confidence - config.reaction_min_cosine)
                / (1.0 - config.reaction_min_cosine), 0.0
            ) ** 3
        ).astype(np.float64)
        self.cold_trust[self.reaction_seen] = 0.0
        self.core_scale = float(runtime.e2r_core_cal["scale"])
        if not self.core_scale > 0.0:
            raise RuntimeError("Invalid frozen E2R calibration scale")

    @torch.no_grad()
    def correction(
        self, protein_id: str, candidate_rows: np.ndarray,
        broad_std: float
    ) -> np.ndarray:
        """Additive calibrated context over the frozen Broad top-1000 head.

        Returned values are model scores, not probabilities.
        """
        top = np.asarray(candidate_rows, dtype=np.int64)
        out = np.zeros(len(top), np.float64)
        pid = self.rt.index.protein_index[str(protein_id)]
        if self.protein_seen[pid]:
            return out
        device = self.rt.index.device
        query = torch.nn.functional.normalize(
            self.rt.index.protein_embeddings[pid], dim=0
        )
        sim = self.seen_protein_embeddings @ query
        nearest_scores, nearest_index = torch.topk(
            sim, min(self.cfg.nearest_proteins, len(sim)), sorted=True
        )
        vv = nearest_scores.float().cpu().numpy().astype(np.float64)
        protein_rows = self.seen_protein_rows[nearest_index.cpu().numpy()]
        attention = np.exp(
            self.cfg.protein_attention_temperature * (vv - vv[0])
        )
        attention /= attention.sum()
        observed = np.zeros(self.nr, np.float64)
        for p, strength in zip(protein_rows, attention):
            degree = int(self.protein_degree[p])
            if degree == 0:
                continue
            neighbors = self.graph.indices[self.graph.indptr[p]:self.graph.indptr[p+1]]
            np.add.at(observed, neighbors, float(strength) / degree)

        observed_mask = self.reaction_seen[top]
        if np.any(observed_mask) and self.cfg.observed_coefficient > 0:
            ix = top[observed_mask]
            background = np.maximum(
                self.cfg.prior_normalizer * self.reaction_prior[ix], 1e-8
            )
            out[observed_mask] += (
                self.cfg.observed_coefficient
                * np.log1p(observed[ix] / background)
            )

        cold_mask = ~observed_mask
        if np.any(cold_mask) and self.cfg.inferred_coefficient > 0:
            ix = top[cold_mask]
            # A query-conditioned z-score relative to the population of
            # real, historically linked enzyme prototypes.
            reference = self.seen_reaction_prototypes @ query
            mu = reference.mean()
            sigma = reference.std(unbiased=False).clamp_min(1e-6)
            predicted = (
                self.inferred_proto.index_select(
                    0, torch.as_tensor(ix, dtype=torch.long, device=device)
                ) @ query
            )
            z = ((predicted - mu) / sigma).float().cpu().numpy()
            out[cold_mask] += (
                self.cfg.inferred_coefficient * z
                * (float(broad_std) / self.core_scale)
                * self.cold_trust[ix]
            )

        if not np.all(np.isfinite(out)):
            raise RuntimeError("Non-finite inductive relation evidence")
        return out
