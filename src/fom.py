"""Full-Order Model (FOM) solver for the steady Navier-Stokes problem.

:class:`FOMSolver` wraps the FEniCS nonlinear solve. It supports

* a fast path using ``NonlinearVariationalSolver`` for production snapshots, and
* a tracked path that implements its own Newton loop in order to record the
  per-iteration residual norm (used by Plot 9 and any convergence diagnostic).

Warm starting via ``w_init`` is essential for the high-Reynolds (low ``mu0``)
samples; pair it with a continuation order in :class:`SnapshotCollector`.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from dolfin import (
    Function, TestFunctions, Constant, DirichletBC,
    assemble, assemble_system, solve, split, derivative,
    NonlinearVariationalProblem, NonlinearVariationalSolver,
    inner, dot, grad, div, dx,
)

from .problem import NavierStokesProblem


@dataclass
class FOMResult:
    """Outcome of a single FOM solve."""
    u: np.ndarray              # velocity DOF vector (size N_u)
    p: np.ndarray              # pressure DOF vector (size N_p)
    solve_time: float
    n_iter: int
    w: Function                # full mixed-space solution (used to warm-start)
    residuals: Optional[List[float]] = None


class FOMSolver:
    """Newton solver for the parametric steady Navier-Stokes problem."""

    def __init__(self, problem: NavierStokesProblem) -> None:
        self.problem = problem
        self.cfg = problem.config

    # ---------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------
    def solve(self, mu0: float, mu1: float,
              w_init: Optional[Function] = None,
              track_residuals: bool = False,
              abs_tol: Optional[float] = None,
              rel_tol: Optional[float] = None,
              max_iter: Optional[int] = None) -> FOMResult:
        """Solve the FOM at parameter ``(mu0, mu1)``.

        Parameters
        ----------
        w_init : Function or None
            Initial guess on ``W``. If ``None``, the zero function is used.
        track_residuals : bool
            If ``True`` runs a custom Newton loop and records the residual norm
            after each linear solve.
        """
        abs_tol = abs_tol if abs_tol is not None else self.cfg.fom_abs_tol
        rel_tol = rel_tol if rel_tol is not None else self.cfg.fom_rel_tol
        max_iter = max_iter if max_iter is not None else self.cfg.fom_max_iter

        W, V, Q = self.problem.W, self.problem.V, self.problem.Q
        w = Function(W) if w_init is None else w_init.copy(deepcopy=True)
        bcs = self.problem.boundary_conditions()
        f = self.problem.forcing(mu1)

        u_split, p_split = split(w)
        v_te, q_te = TestFunctions(W)
        F = (
            Constant(mu0) * inner(grad(u_split), grad(v_te)) * dx
            + inner(dot(grad(u_split), u_split), v_te) * dx
            - p_split * div(v_te) * dx
            - q_te * div(u_split) * dx
            - inner(f, v_te) * dx
        )
        J = derivative(F, w)

        residuals: Optional[List[float]]
        t0 = time.time()
        if track_residuals:
            residuals, n_iter = self._newton_with_residuals(
                F, J, w, bcs, abs_tol, rel_tol, max_iter,
            )
        else:
            residuals = None
            n_iter = self._builtin_newton(F, J, w, bcs,
                                          abs_tol, rel_tol, max_iter)
        t_solve = time.time() - t0

        # Split into standalone V and Q solutions to get clean DOF vectors.
        from dolfin import assign
        u_fn = Function(V)
        p_fn = Function(Q)
        assign(u_fn, w.sub(0))
        assign(p_fn, w.sub(1))

        return FOMResult(
            u=u_fn.vector().get_local().copy(),
            p=p_fn.vector().get_local().copy(),
            solve_time=t_solve,
            n_iter=int(n_iter),
            w=w,
            residuals=residuals,
        )

    # ---------------------------------------------------------------
    # Internal Newton paths
    # ---------------------------------------------------------------
    def _builtin_newton(self, F, J, w, bcs, abs_tol, rel_tol, max_iter) -> int:
        problem = NonlinearVariationalProblem(F, w, bcs, J)
        solver = NonlinearVariationalSolver(problem)
        prm = solver.parameters["newton_solver"]
        prm["absolute_tolerance"] = abs_tol
        prm["relative_tolerance"] = rel_tol
        prm["maximum_iterations"] = max_iter
        prm["error_on_nonconvergence"] = False
        prm["report"] = False
        prm["relaxation_parameter"] = 1.0
        try:
            prm["linear_solver"] = "mumps"
        except Exception:
            prm["linear_solver"] = "default"
        n_iter, _ = solver.solve()
        return int(n_iter)

    def _newton_with_residuals(self, F, J, w, bcs,
                               abs_tol, rel_tol, max_iter
                               ) -> Tuple[List[float], int]:
        for bc in bcs:
            bc.apply(w.vector())
        bcs_hom = [DirichletBC(bc) for bc in bcs]
        for bc in bcs_hom:
            bc.homogenize()

        du = Function(self.problem.W)
        residuals: List[float] = []
        res0 = None
        n_iter = 0
        for it in range(max_iter):
            A, b = assemble_system(J, -F, bcs_hom)
            rn = b.norm("l2")
            residuals.append(rn)
            if res0 is None:
                res0 = max(rn, 1e-300)
            if rn < abs_tol or rn / res0 < rel_tol:
                n_iter = it + 1
                break
            solve(A, du.vector(), b)
            w.vector().axpy(1.0, du.vector())
            n_iter = it + 1
        return residuals, n_iter
