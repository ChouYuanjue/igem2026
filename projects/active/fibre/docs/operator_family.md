# FIBRE operator family

FIBRE has one scientific object: a compatibility field on the product of a
reaction factor and a protein factor.  Different numerical realizations are
allowed only when their relationship to that object is explicit.

## Zero-temperature canonical operator

For sparse verified positives Omega subset M_R x M_E,

J_0(r,e) = min_(ri,ei in Omega) [d_R(r,ri)^2 + d_E(e,ei)^2],

m_R,0(r) = min_(ri in Omega_R) d_R(r,ri)^2,

m_E,0(e) = min_(ei in Omega_E) d_E(e,ei)^2.

The canonical focused FIBRE defect is

Delta_0(r,e) = J_0(r,e) - m_R,0(r) - m_E,0(e),

with compatibility F_0 = -max(Delta_0,0) at numerical precision.

R2E and E2R are sections of exactly this field.  The full matrix is only a
reference implementation; exact sections may be evaluated from distances to
positive marginal supports without materializing M_R x M_E.

## Positive-temperature kernel family

For tau > 0 define

Z_Omega,tau(r,e)
  = sum_(ri,ei in Omega) exp(-(d_R(r,ri)^2+d_E(e,ei)^2)/tau),

Z_R,tau(r)
  = sum_(ri in Omega_R) exp(-d_R(r,ri)^2/tau),

Z_E,tau(e)
  = sum_(ei in Omega_E) exp(-d_E(e,ei)^2/tau).

With C_tau = -tau log Z,

Delta_tau = C_Omega,tau - C_R,tau - C_E,tau
          = tau log(Z_R,tau Z_E,tau / Z_Omega,tau).

By the log-sum-exp / tropical limit,

lim_(tau -> 0+) Delta_tau = Delta_0

before the zero-temperature numerical non-negativity clamp.

This identity is the mathematical bridge between kernel/heat descriptions and
the min-plus correspondence defect.  It does **not** make an arbitrary
finite-temperature realization canonical.  The previous intrinsic-unit
tau=1 diagnostic regressed both directions and remains rejected evidence.

## Broad-domain relationship

The frozen broad Rhea benchmark uses a smooth variational/transport solver on
the same product geometry and positive empirical relation.  It is not currently
claimed to be numerically identical to Delta_tau for any tau.

A broad realization may be promoted to the FIBRE mainline only after all of the
following are shown under the matched clean-development protocol:

1. its operator is derived from this product-field family rather than attached
   as a direction-specific expert;
2. R2E and E2R are sections of the same scalar object;
3. the focused zero-temperature limit is recovered on a domain where exact
   sections are feasible;
4. full candidate support is preserved;
5. the broad implementation is computable from streamed/query-to-support
   geometry rather than a dense product field;
6. no spent external-retention labels are used for operator selection.

Until that gate is met, broad v8 is a frozen matched-protocol reference and the
zero-temperature correspondence defect is the canonical FIBRE operator on the
focused atlas.

## Numerical invariants

- Positive-pair duplicates are one observation, not extra entropy mass.
- Missing molecular views are absence of measurement, never negative evidence.
- Factor exchange transposes the field.
- Exact section evaluation must equal the corresponding row/column of the full
  reference state element-for-element.
- Candidate batching and support batching may alter memory use only.
- Numerical level sets are defined separately from the operator; candidate
  identifiers never define scientific score differences.
