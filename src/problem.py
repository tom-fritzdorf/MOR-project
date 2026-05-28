"""The steady Navier-Stokes problem.

:class:`NavierStokesProblem` owns the FE-level data:

* mesh and Taylor-Hood mixed function space ``W = V x Q``,
* the parametric forcing :math:`f(x; \\mu_1)`,
* boundary conditions (no-slip on :math:`\\partial\\Omega` plus pressure pin at
  :math:`(0, 0)`),
* mass and stiffness matrices ``Mu, Mp, Ku`` (used for FEM norms and the
  M-orthonormal POD),
* the FE divergence operator ``B_h``.

Downstream tasks reuse this class to rebuild the same FE infrastructure
without re-deriving any forms.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import scipy.sparse as sp

from dolfin import (
    UnitSquareMesh, FunctionSpace, VectorElement, FiniteElement, MixedElement,
    Function, TrialFunction, TestFunction, Constant, DirichletBC, UserExpression,
    assemble, inner, dot, grad, div, dx, near, as_backend_type,
    parameters,
)

try:
    from dolfin import set_log_level, LogLevel
    set_log_level(LogLevel.ERROR)
except Exception:
    try:
        from dolfin import set_log_level
        set_log_level(30)
    except Exception:
        pass

# Form-compiler optimisations (safe to apply globally).
parameters["form_compiler"]["optimize"] = True
parameters["form_compiler"]["cpp_optimize"] = True
parameters["form_compiler"]["representation"] = "uflacs"

from .config import Config


# --------------------------------------------------------------------------
# Forcing expression
# --------------------------------------------------------------------------
class ForcingExpression(UserExpression):
    """Parametric source term :math:`f(x; \\mu_1)` from the problem statement."""

    def __init__(self, mu1: float, **kwargs) -> None:
        super().__init__(**kwargs)
        self.mu1 = float(mu1)

    def eval(self, values, x):
        # f(x; mu1) = mu1 * (sin(pi*x0)*sin(pi*x1), cos(pi*x0)*cos(pi*x1))
        # Affinely parameterised in mu1 → low-dimensional solution manifold.
        m1 = self.mu1
        pi = np.pi
        x0, x1 = x[0], x[1]
        values[0] = m1 * np.sin(pi * x0) * np.sin(pi * x1)
        values[1] = m1 * np.cos(pi * x0) * np.cos(pi * x1)

    def value_shape(self):
        return (2,)


# --------------------------------------------------------------------------
# Helpers (DOF/matrix conversion + subdomain predicates)
# --------------------------------------------------------------------------
def fenics_matrix_to_csr(A) -> sp.csr_matrix:
    """Convert a FEniCS PETSc-backed matrix to scipy CSR (zero-copy when possible)."""
    petsc_mat = as_backend_type(A).mat()
    indptr, indices, data = petsc_mat.getValuesCSR()
    return sp.csr_matrix((data, indices, indptr), shape=petsc_mat.size)


def _all_boundary(x, on_boundary):
    return on_boundary


def _origin(x, on_boundary):
    return near(x[0], 0.0, 1e-12) and near(x[1], 0.0, 1e-12)


# --------------------------------------------------------------------------
# NavierStokesProblem
# --------------------------------------------------------------------------
class NavierStokesProblem:
    """Encapsulates the FE infrastructure for the steady Navier-Stokes BVP.

    Parameters
    ----------
    config : Config
        Project configuration. Only the FE-related fields are used here.

    Attributes
    ----------
    mesh : dolfin.Mesh
    W, V, Q : dolfin.FunctionSpace
        Mixed Taylor-Hood and standalone velocity / pressure spaces.
    N_u, N_p : int
        DOF counts of V and Q.
    Mu, Mp, Ku : scipy.sparse.csr_matrix
        Velocity-L2, pressure-L2 and velocity-H1-seminorm mass/stiffness matrices.
    A_h : scipy.sparse.csr_matrix
        Velocity Laplacian-like stiffness ``(grad u, grad v)``.
    B_h : scipy.sparse.csr_matrix
        Divergence operator ``(q, div u)`` of shape (N_p, N_u).
    """

    def __init__(self, config: Config) -> None:
        self.config = config
        self.mesh = UnitSquareMesh(config.mesh_n, config.mesh_n)

        P2 = VectorElement("CG", self.mesh.ufl_cell(), config.velocity_degree)
        P1 = FiniteElement("CG", self.mesh.ufl_cell(), config.pressure_degree)
        TH = MixedElement([P2, P1])

        self.W = FunctionSpace(self.mesh, TH)
        self.V = FunctionSpace(self.mesh, P2)
        self.Q = FunctionSpace(self.mesh, P1)

        self.N_u = self.V.dim()
        self.N_p = self.Q.dim()

        self.Mu, self.Mp, self.Ku = self._assemble_mass_stiffness()
        self.A_h, self.B_h = self._assemble_fe_operators()

    # ----- BCs -----
    def boundary_conditions(self):
        """Return Dirichlet BCs on the mixed space ``W``."""
        bc_u = DirichletBC(self.W.sub(0), Constant((0.0, 0.0)), _all_boundary)
        bc_p = DirichletBC(self.W.sub(1), Constant(0.0), _origin, "pointwise")
        return [bc_u, bc_p]

    def velocity_zero_bc(self):
        """Zero Dirichlet BC on the standalone velocity space (used for supremizer)."""
        return DirichletBC(self.V, Constant((0.0, 0.0)), _all_boundary)

    # ----- forcing -----
    def forcing(self, mu1: float, degree: int = 4) -> ForcingExpression:
        return ForcingExpression(mu1, degree=degree)

    def assemble_forcing_vector(self, mu1: float) -> np.ndarray:
        """Assemble the FEM forcing vector ``F_h(\\mu_1)`` on V (with zero BCs)."""
        v_te = TestFunction(self.V)
        f = self.forcing(mu1)
        L = inner(f, v_te) * dx
        bvec = assemble(L)
        self.velocity_zero_bc().apply(bvec)
        return bvec.get_local().copy()

    # ----- matrices -----
    def _assemble_mass_stiffness(self) -> Tuple[sp.csr_matrix, sp.csr_matrix, sp.csr_matrix]:
        u_tr, v_te = TrialFunction(self.V), TestFunction(self.V)
        Mu = fenics_matrix_to_csr(assemble(inner(u_tr, v_te) * dx))
        Ku = fenics_matrix_to_csr(assemble(inner(grad(u_tr), grad(v_te)) * dx))
        p_tr, q_te = TrialFunction(self.Q), TestFunction(self.Q)
        Mp = fenics_matrix_to_csr(assemble(p_tr * q_te * dx))
        return Mu, Mp, Ku

    def _assemble_fe_operators(self) -> Tuple[sp.csr_matrix, sp.csr_matrix]:
        u_tr, v_te = TrialFunction(self.V), TestFunction(self.V)
        q_te = TestFunction(self.Q)
        A_h = fenics_matrix_to_csr(assemble(inner(grad(u_tr), grad(v_te)) * dx))
        B_h = fenics_matrix_to_csr(assemble(q_te * div(u_tr) * dx))
        return A_h, B_h

    # ----- DOF-vector / FEniCS-Function helpers -----
    def velocity_function(self, vec: np.ndarray = None) -> Function:
        f = Function(self.V)
        if vec is not None:
            f.vector().set_local(np.asarray(vec, dtype=float))
            f.vector().apply("insert")
        return f

    def pressure_function(self, vec: np.ndarray = None) -> Function:
        f = Function(self.Q)
        if vec is not None:
            f.vector().set_local(np.asarray(vec, dtype=float))
            f.vector().apply("insert")
        return f

    # ----- convenience -----
    def fe_info(self) -> dict:
        """Lightweight metadata dump for downstream tasks."""
        c = self.config
        return dict(
            mesh_n=c.mesh_n,
            velocity_degree=c.velocity_degree,
            pressure_degree=c.pressure_degree,
            N_u=self.N_u, N_p=self.N_p,
            mu0_range=c.mu0_range, mu1_range=c.mu1_range,
            n_train_per_dim=c.n_train_per_dim, n_test=c.n_test, seed=c.seed,
            energy_threshold=c.energy_threshold,
        )
