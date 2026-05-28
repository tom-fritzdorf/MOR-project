"""Parameter sampling, snapshot collection, supremizer enrichment, and POD.

This module is intentionally split into small focused classes so that downstream
tasks can pick what they need:

* :class:`ParameterSampler` — deterministic train/test parameter grids.
  Reused by Task 2 (PODNN training data) and Task 4 (PINN sample design).

* :class:`SnapshotCollector` — runs the FOM solver over the parameter set
  with warm-started continuation and reports timing info.

* :class:`SupremizerEnricher` — solves the velocity-Laplacian supremizer
  problem for each pressure snapshot and concatenates the result to the
  velocity snapshot matrix (inf-sup stability of the reduced pair).

* :class:`PODBasis` — energy-truncated POD in the mass-weighted inner product,
  with ``save`` / ``load`` for re-use across tasks.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import scipy.linalg as la
import scipy.sparse as sp

from dolfin import (
    Function, TrialFunction, TestFunction, Constant,
    assemble, solve, inner, grad, div, dx,
)

try:
    from tqdm import tqdm
except ImportError:                                  # pragma: no cover
    def tqdm(iterable, **kwargs):
        return iterable

from .config import Config
from .problem import NavierStokesProblem
from .fom import FOMSolver, FOMResult


# ==========================================================================
# Parameter sampling
# ==========================================================================
class ParameterSampler:
    """Generates the train and test parameter sets used throughout Tasks 1-4."""

    def __init__(self, config: Config) -> None:
        self.config = config

    def train(self) -> np.ndarray:
        """Uniform tensor grid of shape ``(n_train_per_dim**2, 2)``."""
        c = self.config
        mu0 = np.linspace(c.mu0_range[0], c.mu0_range[1], c.n_train_per_dim)
        mu1 = np.linspace(c.mu1_range[0], c.mu1_range[1], c.n_train_per_dim)
        MM0, MM1 = np.meshgrid(mu0, mu1, indexing="ij")
        return np.stack([MM0.ravel(), MM1.ravel()], axis=1)

    def test(self, train: Optional[np.ndarray] = None) -> np.ndarray:
        """Random samples in the same box, disjoint from the training grid."""
        c = self.config
        if train is None:
            train = self.train()
        rng = np.random.default_rng(c.seed)
        out: List[np.ndarray] = []
        while len(out) < c.n_test:
            cand = np.array([rng.uniform(*c.mu0_range), rng.uniform(*c.mu1_range)])
            if np.min(np.linalg.norm(train - cand, axis=1)) > 1e-6:
                out.append(cand)
        return np.asarray(out)

    def both(self) -> Tuple[np.ndarray, np.ndarray]:
        train = self.train()
        return train, self.test(train)


# ==========================================================================
# Snapshot collection
# ==========================================================================
@dataclass
class SnapshotData:
    """Container for collected FOM snapshots."""
    S_u: np.ndarray                    # (N_u, M)   velocity snapshots
    S_p: np.ndarray                    # (N_p, M)   pressure snapshots
    times: np.ndarray                  # (M,)       FOM solve times
    iters: np.ndarray                  # (M,)       Newton iteration counts
    params: np.ndarray                 # (M, 2)     parameter values
    label: str = "snapshots"

    def save(self, path: Path) -> None:
        np.savez_compressed(
            path,
            S_u=self.S_u, S_p=self.S_p,
            times=self.times, iters=self.iters,
            params=self.params, label=np.array(self.label),
        )

    @classmethod
    def load(cls, path: Path) -> "SnapshotData":
        d = np.load(path, allow_pickle=False)
        return cls(
            S_u=d["S_u"], S_p=d["S_p"],
            times=d["times"], iters=d["iters"],
            params=d["params"], label=str(d["label"]),
        )


class SnapshotCollector:
    """Runs the FOM over a batch of parameters with warm-started continuation."""

    def __init__(self, problem: NavierStokesProblem, fom: Optional[FOMSolver] = None) -> None:
        self.problem = problem
        self.fom = fom or FOMSolver(problem)

    def collect(self, params: np.ndarray, label: str = "train",
                store_solutions: bool = False
                ) -> SnapshotData:
        """Solve the FOM at every parameter in ``params``.

        We sort by descending ``mu0`` (most diffusive first) and warm-start
        from the previous solution, which is critical at high Reynolds.
        """
        N_u, N_p = self.problem.N_u, self.problem.N_p
        M = params.shape[0]
        S_u = np.zeros((N_u, M))
        S_p = np.zeros((N_p, M))
        times = np.zeros(M)
        iters = np.zeros(M, dtype=int)

        order = np.argsort(-params[:, 0])
        w_prev = None
        for idx in tqdm(order, desc=f"FOM [{label}]"):
            mu0, mu1 = params[idx]
            try:
                res = self.fom.solve(mu0, mu1, w_init=w_prev)
            except Exception as exc:                  # pragma: no cover
                print(f"  [warn] warm-start failed at mu=({mu0:.3g},{mu1:.3g}): "
                      f"{exc}; retrying from zero")
                res = self.fom.solve(mu0, mu1, w_init=None)
            S_u[:, idx] = res.u
            S_p[:, idx] = res.p
            times[idx] = res.solve_time
            iters[idx] = res.n_iter
            w_prev = res.w

        return SnapshotData(S_u=S_u, S_p=S_p, times=times, iters=iters,
                            params=params.copy(), label=label)


# ==========================================================================
# Supremizer enrichment
# ==========================================================================
class SupremizerEnricher:
    """Velocity-Laplacian supremizer for each pressure snapshot.

    Solves :math:`(\\nabla s, \\nabla v) = -(p_h, \\nabla\\cdot v)` for every
    pressure snapshot, with zero Dirichlet BCs on the velocity space.
    """

    def __init__(self, problem: NavierStokesProblem) -> None:
        self.problem = problem
        # Pre-assemble (and pre-apply BCs to) the velocity Laplacian once.
        s_tr = TrialFunction(problem.V)
        self._v_te = TestFunction(problem.V)
        a = inner(grad(s_tr), grad(self._v_te)) * dx
        self._bc = problem.velocity_zero_bc()
        self._A = assemble(a)
        self._bc.apply(self._A)

    def compute(self, S_p: np.ndarray) -> np.ndarray:
        """Return the supremizer snapshot matrix ``S_sup`` of shape (N_u, M)."""
        M = S_p.shape[1]
        S_sup = np.zeros((self.problem.N_u, M))
        p_h = Function(self.problem.Q)
        s = Function(self.problem.V)
        for j in tqdm(range(M), desc="Supremizers"):
            p_h.vector().set_local(S_p[:, j])
            p_h.vector().apply("insert")
            L = -p_h * div(self._v_te) * dx
            b = assemble(L)
            self._bc.apply(b)
            solve(self._A, s.vector(), b)
            S_sup[:, j] = s.vector().get_local()
        return S_sup

    @staticmethod
    def enrich(S_u: np.ndarray, S_sup: np.ndarray) -> np.ndarray:
        """Concatenate primary velocity snapshots with supremizer snapshots (unscaled).

        Kept for completeness; the preferred path is
        :meth:`PODBasis.from_velocity_and_supremizers`, which performs separate
        POD on each block before concatenating the *bases*. Joint-SVD on
        concatenated snapshots tends to mix velocity and supremizer directions
        unfavourably.
        """
        return np.concatenate([S_u, S_sup], axis=1)


# ==========================================================================
# POD basis
# ==========================================================================
@dataclass
class PODBasis:
    """Proper Orthogonal Decomposition in an FE mass-weighted inner product.

    The modes are :math:`M`-orthonormal, i.e. :math:`\\Phi^T M \\Phi = I`.
    We diagonalise the snapshot correlation matrix :math:`C = S^T M S`, which
    sidesteps the explicit (and large) :math:`N \\times N` SVD.
    """
    Phi: np.ndarray                    # (N, r) basis
    sigmas: np.ndarray                 # (K,) singular values (sqrt eigenvalues)
    cum_energy: np.ndarray             # (K,) cumulative energy fractions
    r: int                             # number of retained modes
    name: str = "u"
    energy_threshold: float = 0.9999
    r_primary: int = 0                 # primary modes (velocity) when supremizer-enriched
    r_sup: int = 0                     # supremizer modes appended after the primary ones

    # ---- factory ----
    @classmethod
    def from_snapshots(cls, S: np.ndarray, M_inner: sp.csr_matrix,
                       energy_threshold: float = 0.9999,
                       name: str = "u") -> "PODBasis":
        # Correlation matrix in the M-inner product
        MS = M_inner @ S
        C = S.T @ MS
        C = 0.5 * (C + C.T)
        eigvals, eigvecs = la.eigh(C)
        # Sort descending and clamp tiny negatives
        idx = np.argsort(eigvals)[::-1]
        eigvals = np.maximum(eigvals[idx], 0.0)
        eigvecs = eigvecs[:, idx]
        sigmas = np.sqrt(eigvals)

        total = float(np.sum(eigvals))
        if total <= 0:
            raise RuntimeError("Snapshot energy is zero; nothing to compress.")
        cum = np.cumsum(eigvals) / total

        # Energy-based truncation
        r = int(np.searchsorted(cum, energy_threshold) + 1)
        r = min(r, len(sigmas))
        # Drop modes with vanishing singular value (numerical safety)
        nonzero = int(np.sum(sigmas > 1e-12 * (sigmas[0] if sigmas[0] > 0 else 1)))
        r = min(r, max(nonzero, 1))

        # Reconstruct modes: Phi[:, k] = S V[:, k] / sigma_k, then M-Gram-Schmidt
        Phi = np.zeros((S.shape[0], r))
        for k in range(r):
            Phi[:, k] = (S @ eigvecs[:, k]) / max(sigmas[k], 1e-300)
        Phi = _m_gram_schmidt(Phi, M_inner)

        return cls(Phi=Phi, sigmas=sigmas, cum_energy=cum, r=r,
                   name=name, energy_threshold=energy_threshold,
                   r_primary=r, r_sup=0)

    @classmethod
    def from_velocity_and_supremizers(
        cls,
        S_u: np.ndarray,
        S_sup: np.ndarray,
        M_inner: sp.csr_matrix,
        energy_threshold: float = 0.9999,
        name: str = "u",
    ) -> "PODBasis":
        """Textbook supremizer-enriched velocity POD (Ballarin-Rozza style).

        Runs an independent energy-truncated POD on the velocity snapshots and
        on the supremizer snapshots, then M-orthonormalises the concatenation
        ``[Phi_u^primary | Phi_u^sup]``. Storing the primary block first keeps
        the truncation API intuitive (drop supremizers last).
        """
        pod_primary = cls.from_snapshots(S_u, M_inner,
                                         energy_threshold=energy_threshold,
                                         name=name)
        pod_sup = cls.from_snapshots(S_sup, M_inner,
                                     energy_threshold=energy_threshold,
                                     name=f"{name}_sup")
        Phi = np.concatenate([pod_primary.Phi, pod_sup.Phi], axis=1)
        Phi = _m_gram_schmidt(Phi, M_inner)
        # Some columns may collapse to zero after orthogonalisation; drop them.
        norms = np.linalg.norm(Phi, axis=0)
        keep = norms > 1e-12
        Phi = Phi[:, keep]
        r_primary = int(keep[: pod_primary.r].sum())
        r_sup = int(keep[pod_primary.r:].sum())
        return cls(
            Phi=Phi,
            sigmas=pod_primary.sigmas,
            cum_energy=pod_primary.cum_energy,
            r=Phi.shape[1],
            name=name,
            energy_threshold=energy_threshold,
            r_primary=r_primary,
            r_sup=r_sup,
        )

    # ---- queries ----
    def truncate(self, r: int) -> "PODBasis":
        """Return a copy keeping only the first ``r`` modes.

        Primary modes come first, supremizer modes last; truncating below
        ``r_primary`` drops supremizers entirely.
        """
        r = max(1, min(r, self.Phi.shape[1]))
        new_primary = min(r, self.r_primary or r)
        new_sup = max(0, r - new_primary)
        return PODBasis(
            Phi=self.Phi[:, :r].copy(),
            sigmas=self.sigmas.copy(),
            cum_energy=self.cum_energy.copy(),
            r=r, name=self.name,
            energy_threshold=self.energy_threshold,
            r_primary=new_primary,
            r_sup=new_sup,
        )

    def modes_for_energy(self, level: float) -> int:
        return int(np.searchsorted(self.cum_energy, level) + 1)

    # ---- I/O ----
    def save(self, path: Path) -> None:
        np.savez_compressed(
            path,
            Phi=self.Phi, sigmas=self.sigmas, cum_energy=self.cum_energy,
            r=np.array(self.r), name=np.array(self.name),
            energy_threshold=np.array(self.energy_threshold),
            r_primary=np.array(self.r_primary),
            r_sup=np.array(self.r_sup),
        )

    @classmethod
    def load(cls, path: Path) -> "PODBasis":
        d = np.load(path, allow_pickle=False)
        keys = set(d.files)
        r_primary = int(d["r_primary"]) if "r_primary" in keys else int(d["r"])
        r_sup = int(d["r_sup"]) if "r_sup" in keys else 0
        return cls(
            Phi=d["Phi"], sigmas=d["sigmas"], cum_energy=d["cum_energy"],
            r=int(d["r"]), name=str(d["name"]),
            energy_threshold=float(d["energy_threshold"]),
            r_primary=r_primary, r_sup=r_sup,
        )


def _m_gram_schmidt(Phi: np.ndarray, M: sp.csr_matrix) -> np.ndarray:
    """Modified Gram-Schmidt in the M-induced inner product."""
    n = Phi.shape[1]
    out = np.zeros_like(Phi)
    for k in range(n):
        v = Phi[:, k].copy()
        for j in range(k):
            coeff = float(out[:, j] @ (M @ v))
            v -= coeff * out[:, j]
        norm = float(np.sqrt(max(v @ (M @ v), 0.0)))
        if norm < 1e-300:
            continue
        out[:, k] = v / norm
    return out
