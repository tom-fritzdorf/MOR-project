"""Error metrics in FEM norms.

All norms use the FEM mass / stiffness matrices, never raw numpy norms on
DOF vectors (which ignore mesh geometry). The same class is reused by Task 2
to score neural-network predictions and by Task 3 to compare approaches.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import scipy.sparse as sp

from .problem import NavierStokesProblem


@dataclass
class ErrorReport:
    """Per-sample errors plus aggregated statistics over a test set."""
    rel_l2_u: np.ndarray
    rel_l2_p: np.ndarray
    rel_h1_u: np.ndarray

    def summary(self) -> dict:
        return dict(
            mean_l2_u=float(self.rel_l2_u.mean()),
            max_l2_u=float(self.rel_l2_u.max()),
            mean_l2_p=float(self.rel_l2_p.mean()),
            max_l2_p=float(self.rel_l2_p.max()),
            mean_h1_u=float(self.rel_h1_u.mean()),
            max_h1_u=float(self.rel_h1_u.max()),
        )


class ErrorAnalyzer:
    """FEM-norm error helpers."""

    def __init__(self, problem: NavierStokesProblem) -> None:
        self.problem = problem
        self.Mu: sp.csr_matrix = problem.Mu
        self.Mp: sp.csr_matrix = problem.Mp
        self.Ku: sp.csr_matrix = problem.Ku

    # ---- scalar norms ----
    def l2_velocity(self, vec: np.ndarray) -> float:
        return float(np.sqrt(max(vec @ (self.Mu @ vec), 0.0)))

    def h1_seminorm_velocity(self, vec: np.ndarray) -> float:
        return float(np.sqrt(max(vec @ (self.Ku @ vec), 0.0)))

    def l2_pressure(self, vec: np.ndarray) -> float:
        return float(np.sqrt(max(vec @ (self.Mp @ vec), 0.0)))

    # ---- single-sample relative errors ----
    def relative_errors(self, u_ref: np.ndarray, p_ref: np.ndarray,
                        u_approx: np.ndarray, p_approx: np.ndarray
                        ) -> Tuple[float, float, float]:
        du = u_ref - u_approx
        dp = p_ref - p_approx
        eps = 1e-300
        nu_l2 = self.l2_velocity(u_ref)
        nu_h1 = self.h1_seminorm_velocity(u_ref)
        np_l2 = self.l2_pressure(p_ref)
        return (
            self.l2_velocity(du) / max(nu_l2, eps),
            self.l2_pressure(dp) / max(np_l2, eps),
            self.h1_seminorm_velocity(du) / max(nu_h1, eps),
        )

    # ---- batched ----
    def batch_report(self, U_ref: np.ndarray, P_ref: np.ndarray,
                     U_approx: np.ndarray, P_approx: np.ndarray
                     ) -> ErrorReport:
        """Vectorised computation over a test set (columns = samples)."""
        M = U_ref.shape[1]
        e_u = np.zeros(M); e_p = np.zeros(M); e_h1 = np.zeros(M)
        for i in range(M):
            eu, ep, eh1 = self.relative_errors(
                U_ref[:, i], P_ref[:, i], U_approx[:, i], P_approx[:, i],
            )
            e_u[i] = eu; e_p[i] = ep; e_h1[i] = eh1
        return ErrorReport(rel_l2_u=e_u, rel_l2_p=e_p, rel_h1_u=e_h1)
