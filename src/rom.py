"""Reduced-Order Model (ROM) operators and solver.

The reduced saddle-point system in the POD bases is

.. math::

    \\mu_0 \\tilde A\\,a + \\tilde C(a, a) - \\tilde B^T b = \\tilde f(\\mu_1)
    \\qquad
    \\tilde B\\,a = 0,

where ``a`` and ``b`` are velocity / pressure reduced coefficients.

:class:`ROMOperators` precomputes the parameter-independent operators once
(:math:`\\tilde A, \\tilde B, \\tilde C`) and exposes ``assemble_forcing(mu1)``
for the only :math:`\\mu_1`-dependent piece. The convection tensor is built
via :math:`r_u` FEM assemblies (one per velocity POD mode) — *not*
:math:`r_u^3` — by reusing the linear FEM convection matrix with a fixed
advecting velocity.

:class:`ROMSolver` runs Newton on the reduced system and exposes
``reconstruct`` to map back to FE DOFs.
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
    Function, TrialFunction, TestFunction,
    assemble, inner, dot, grad, dx,
)

try:
    from tqdm import tqdm
except ImportError:                                  # pragma: no cover
    def tqdm(iterable, **kwargs):
        return iterable

from .problem import NavierStokesProblem, fenics_matrix_to_csr
from .pod import PODBasis


# ==========================================================================
# Reduced operators
# ==========================================================================
@dataclass
class ROMOperators:
    """Galerkin-projected operators for the steady NS reduced system.

    Attributes
    ----------
    A_r : (r_u, r_u) ndarray         reduced velocity stiffness  (mu0-scalable)
    B_r : (r_p, r_u) ndarray         reduced divergence
    C_r : (r_u, r_u, r_u) ndarray    reduced convection trilinear form
    """
    A_r: np.ndarray
    B_r: np.ndarray
    C_r: np.ndarray
    r_u: int
    r_p: int
    assembly_time: float = 0.0

    # ---- factory ----
    @classmethod
    def assemble(cls, problem: NavierStokesProblem,
                 Phi_u: np.ndarray, Phi_p: np.ndarray) -> "ROMOperators":
        t0 = time.time()
        A_r = Phi_u.T @ (problem.A_h @ Phi_u)
        B_r = Phi_p.T @ (problem.B_h @ Phi_u)
        C_r = _assemble_convection_tensor(problem, Phi_u)
        t1 = time.time()
        return cls(
            A_r=np.asarray(A_r), B_r=np.asarray(B_r), C_r=C_r,
            r_u=Phi_u.shape[1], r_p=Phi_p.shape[1],
            assembly_time=t1 - t0,
        )

    # ---- truncation (for the error-vs-modes sweep) ----
    def truncate(self, r_u: int, r_p: int) -> "ROMOperators":
        r_u = max(1, min(r_u, self.r_u))
        r_p = max(1, min(r_p, self.r_p))
        return ROMOperators(
            A_r=self.A_r[:r_u, :r_u].copy(),
            B_r=self.B_r[:r_p, :r_u].copy(),
            C_r=self.C_r[:r_u, :r_u, :r_u].copy(),
            r_u=r_u, r_p=r_p, assembly_time=0.0,
        )

    # ---- I/O ----
    def save(self, path: Path) -> None:
        np.savez_compressed(
            path, A_r=self.A_r, B_r=self.B_r, C_r=self.C_r,
            r_u=np.array(self.r_u), r_p=np.array(self.r_p),
            assembly_time=np.array(self.assembly_time),
        )

    @classmethod
    def load(cls, path: Path) -> "ROMOperators":
        d = np.load(path, allow_pickle=False)
        return cls(
            A_r=d["A_r"], B_r=d["B_r"], C_r=d["C_r"],
            r_u=int(d["r_u"]), r_p=int(d["r_p"]),
            assembly_time=float(d["assembly_time"]),
        )


def _assemble_convection_tensor(problem: NavierStokesProblem,
                                Phi_u: np.ndarray) -> np.ndarray:
    """Build ``C_r`` such that ``sum_{j,k} C_r[i,j,k] a_j a_k`` equals
    \int (u_r \cdot \nabla u_r) \cdot \phi_i^{POD} \, dx.
    """
    r = Phi_u.shape[1]
    V = problem.V
    C_r = np.zeros((r, r, r))
    
    u_known = Function(V)
    u_tr = TrialFunction(V)
    v_te = TestFunction(V)
    
    for k in tqdm(range(r), desc="Convection tensor"):
        # Set known function to the k-th basis mode
        u_known.vector().set_local(Phi_u[:, k])
        u_known.vector().apply("insert")
        
        # M_k corresponds to advecting *by* phi_k
        # inner(dot(grad(u_tr), u_known), v_te) -> (phi_k . grad(phi_j)) . phi_i
        M_k_form = inner(dot(grad(u_tr), u_known), v_te) * dx
        M_k = fenics_matrix_to_csr(assemble(M_k_form))
        
        # C_r[:, j, k] -> i is rows of Phi_u.T, j is cols of Phi_u
        C_r[:, :, k] = Phi_u.T @ (M_k @ Phi_u)
        
    return C_r


# ==========================================================================
# ROM solver
# ==========================================================================
@dataclass
class ROMResult:
    a: np.ndarray                  # velocity reduced coefficients (r_u,)
    b: np.ndarray                  # pressure reduced coefficients (r_p,)
    solve_time: float
    n_iter: int
    residuals: List[float] = field(default_factory=list)


class ROMSolver:
    """Newton solver for the projected steady Navier-Stokes saddle-point system."""

    def __init__(self, problem: NavierStokesProblem,
                 operators: ROMOperators,
                 phi_u: np.ndarray,
                 phi_p: np.ndarray) -> None:
        self.problem = problem
        self.operators = operators
        self.Phi_u = phi_u
        self.Phi_p = phi_p
        self.cfg = problem.config

    # ---- forcing (mu1-dependent piece) ----
    def assemble_forcing(self, mu1: float) -> np.ndarray:
        f_h = self.problem.assemble_forcing_vector(mu1)
        return self.Phi_u.T @ f_h

    # ---- solve ----
    def solve(self, mu0: float, mu1: float,
              a_init: Optional[np.ndarray] = None,
              b_init: Optional[np.ndarray] = None,
              tol: Optional[float] = None,
              max_iter: Optional[int] = None) -> ROMResult:
        tol = tol if tol is not None else self.cfg.rom_tol
        max_iter = max_iter if max_iter is not None else self.cfg.rom_max_iter

        op = self.operators
        a = np.zeros(op.r_u) if a_init is None else a_init.copy()
        b = np.zeros(op.r_p) if b_init is None else b_init.copy()
        f_r = self.assemble_forcing(mu1)

        residuals: List[float] = []
        t0 = time.time()
        res0 = None
        n_iter = 0
        for it in range(max_iter):
            # Residual
            R_conv = np.einsum("ijk,j,k->i", op.C_r, a, a)
            R_u = mu0 * (op.A_r @ a) + R_conv - op.B_r.T @ b - f_r
            R_p = op.B_r @ a
            R = np.concatenate([R_u, R_p])
            rn = float(np.linalg.norm(R))
            residuals.append(rn)
            if res0 is None:
                res0 = max(rn, 1e-300)
            if rn < tol or rn / res0 < tol * 1e2:
                n_iter = it
                break
            # Jacobian (saddle-point block form)
            J_conv = (np.einsum("ilk,k->il", op.C_r, a)
                      + np.einsum("ikl,k->il", op.C_r, a))
            J_uu = mu0 * op.A_r + J_conv
            J = np.block([
                [J_uu, -op.B_r.T],
                [op.B_r, np.zeros((op.r_p, op.r_p))],
            ])
            delta = la.solve(J, -R)
            a += delta[: op.r_u]
            b += delta[op.r_u:]
            n_iter = it + 1
        t_solve = time.time() - t0

        return ROMResult(a=a, b=b, solve_time=t_solve, n_iter=n_iter,
                         residuals=residuals)

    # ---- reconstruction ----
    def reconstruct(self, a: np.ndarray, b: np.ndarray
                    ) -> Tuple[np.ndarray, np.ndarray]:
        """Map reduced coefficients back to FE DOF vectors.

        No gauge correction is applied: snapshots are pinned at ``(0, 0)``,
        so the POD pressure subspace inherits that gauge and the reconstruction
        is already comparable to the FOM pressure.
        """
        return self.Phi_u @ a, self.Phi_p @ b

    # ---- batch evaluation (used by Task 2 to convert NN predictions) ----
    def reconstruct_batch(self, coeffs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """``coeffs`` of shape ``(r_u + r_p, M)`` -> (u_recon, p_recon) FE matrices."""
        a = coeffs[: self.operators.r_u, :]
        b = coeffs[self.operators.r_u:, :]
        return self.Phi_u @ a, self.Phi_p @ b
