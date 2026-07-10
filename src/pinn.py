"""Physics-Informed Neural Network (PINN) for the parametric steady
Navier-Stokes problem (Task 4, ``project2026.pdf``).

Loss (verbatim from the problem statement): ``MSE = MSE_b^mu + lambda*MSE_p^mu``

* ``MSE_b^mu`` -- boundary term, ``(1/N_b) sum |w~(x_b,mu_b) - u(x_b,mu_b)|^2``
  over ``dOmega x P``. Since ``u = 0`` on ``dOmega`` (the problem's own
  Dirichlet BC), this reduces to a zero-velocity penalty at boundary points.
  The pressure pin ``p(0,0)=0`` (also required by the problem, and enforced
  for the FOM/ROM via a pointwise ``DirichletBC`` in
  :meth:`NavierStokesProblem.boundary_conditions`) is folded into the same
  term as a second zero-target penalty.
* ``MSE_p^mu`` -- physics term, ``(1/N_p) sum |R(w~(x_p,mu_p))|^2`` over
  ``Omega x P``, where ``R`` is the steady Navier-Stokes residual
  (momentum + continuity).

No autodiff library is available in ``mor_env`` (no PyTorch/JAX/TensorFlow,
no pip -- see :mod:`src.podnn`'s header for the same constraint). ``R`` needs
first and second spatial derivatives of the network output; rather than
hand-building a second-order forward-mode AD engine plus a custom backprop
pass through it, spatial derivatives are approximated with a 5-point central
finite-difference stencil (center + ``x +/- h*e_i``). This turns ``R`` into a
*linear combination of ordinary network forward passes*, so the
weight-gradient of the loss is then exact, standard reverse-mode backprop --
literally :class:`src.podnn.FeedForwardNet`'s existing ``forward``/
``_backward`` reused as-is. This is a disclosed, controlled simplification
(O(h^2) truncation error, h ~ 1e-3) -- not a hidden shortcut. Note only the
*diagonal* second partials (``d^2/dx0^2``, ``d^2/dx1^2``) are ever needed for
the Laplacian in ``R``; no mixed partial ``d^2/dx0 dx1`` appears anywhere.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from .podnn import FeedForwardNet


# ==========================================================================
# Analytic forcing term (plain numpy -- mirrors problem.py's _F1_CPP/_F2_CPP)
# ==========================================================================
def forcing_numpy(x0: np.ndarray, x1: np.ndarray, mu1: np.ndarray
                  ) -> Tuple[np.ndarray, np.ndarray]:
    """Vectorized numpy evaluation of f(x; mu1), identical formula to
    ``problem.py``'s JIT-compiled C++ expression (no FEM call -- this is
    just the analytic source term evaluated at arbitrary points)."""
    pi = np.pi
    f1 = (-(mu1 ** 3 * pi ** 2 * np.cos(mu1 ** 2 * pi * x0) - mu1 ** 2 * pi ** 2)
          * np.sin(mu1 * pi * x1) * np.cos(mu1 * pi * x1)
          + mu1 * pi * np.cos(mu1 * pi * x0) * np.cos(mu1 * pi * x1))
    f2 = (-(-mu1 ** 3 * pi ** 2 * np.cos(mu1 ** 2 * pi * x1) + mu1 ** 2 * pi ** 2)
          * np.sin(mu1 * pi * x0) * np.cos(mu1 * pi * x0)
          - mu1 * pi * np.sin(mu1 * pi * x0) * np.sin(mu1 * pi * x1))
    return f1, f2


# ==========================================================================
# Collocation sampling
# ==========================================================================
@dataclass
class Collocation:
    """One epoch's worth of sample points, all as raw (unnormalized) values."""
    x_int: np.ndarray      # (N_int, 2) interior (x0,x1)
    mu_int: np.ndarray     # (N_int, 2) interior (mu0,mu1)
    x_bnd: np.ndarray      # (N_bnd, 2) boundary (x0,x1)
    mu_bnd: np.ndarray     # (N_bnd, 2) boundary (mu0,mu1)
    mu_pin: np.ndarray     # (N_pin, 2) pressure-pin (mu0,mu1), x fixed at (0,0)


def sample_collocation_points(n_interior: int, n_boundary: int, n_pin: int,
                              mu0_range: Tuple[float, float],
                              mu1_range: Tuple[float, float],
                              rng: np.random.Generator) -> Collocation:
    """Interior points uniform in the unit square x P; boundary points
    uniform on the 4 edges of dOmega x P; pin points at (0,0) x P."""
    x_int = rng.uniform(0.0, 1.0, size=(n_interior, 2))
    mu_int = np.column_stack([
        rng.uniform(*mu0_range, size=n_interior),
        rng.uniform(*mu1_range, size=n_interior),
    ])

    edges = rng.integers(0, 4, size=n_boundary)
    t = rng.uniform(0.0, 1.0, size=n_boundary)
    x_bnd = np.zeros((n_boundary, 2))
    x_bnd[edges == 0] = np.column_stack([np.zeros((edges == 0).sum()), t[edges == 0]])
    x_bnd[edges == 1] = np.column_stack([np.ones((edges == 1).sum()), t[edges == 1]])
    x_bnd[edges == 2] = np.column_stack([t[edges == 2], np.zeros((edges == 2).sum())])
    x_bnd[edges == 3] = np.column_stack([t[edges == 3], np.ones((edges == 3).sum())])
    mu_bnd = np.column_stack([
        rng.uniform(*mu0_range, size=n_boundary),
        rng.uniform(*mu1_range, size=n_boundary),
    ])

    mu_pin = np.column_stack([
        rng.uniform(*mu0_range, size=n_pin),
        rng.uniform(*mu1_range, size=n_pin),
    ])

    return Collocation(x_int, mu_int, x_bnd, mu_bnd, mu_pin)


# ==========================================================================
# PINN model
# ==========================================================================
class PINNModel:
    """w~(x, mu) -> (u1, u2, p), trained by minimizing the PINN loss above.

    Input feature vector is ``[x0, x1, mu0_n, mu1_n]`` where ``mu0_n, mu1_n``
    are mu0/mu1 affinely mapped to [-1, 1] (spatial coordinates are already
    well-scaled in [0,1] and are left untouched -- passing them through a
    generic standardizer would complicate the finite-difference stencil with
    an extra chain-rule scale factor for no benefit). No output scaling is
    used: unlike PODNN's supervised coefficient regression, PINN training has
    no known target magnitudes to standardize against.
    """

    def __init__(self, mu0_range: Tuple[float, float], mu1_range: Tuple[float, float],
                hidden_sizes: Optional[List[int]] = None, seed: int = 42) -> None:
        self.mu0_range = mu0_range
        self.mu1_range = mu1_range
        self._mu0_mid = 0.5 * (mu0_range[0] + mu0_range[1])
        self._mu0_half = 0.5 * (mu0_range[1] - mu0_range[0])
        self._mu1_mid = 0.5 * (mu1_range[0] + mu1_range[1])
        self._mu1_half = 0.5 * (mu1_range[1] - mu1_range[0])

        hidden_sizes = hidden_sizes if hidden_sizes is not None else [64, 64, 64]
        self.layer_sizes = [4] + list(hidden_sizes) + [3]
        self.net = FeedForwardNet(self.layer_sizes, seed=seed)

    # ---- mu normalization ----
    def _norm_mu(self, mu: np.ndarray) -> np.ndarray:
        mu0n = (mu[:, 0] - self._mu0_mid) / self._mu0_half
        mu1n = (mu[:, 1] - self._mu1_mid) / self._mu1_half
        return np.column_stack([mu0n, mu1n])

    def _features(self, x: np.ndarray, mu: np.ndarray) -> np.ndarray:
        return np.column_stack([x, self._norm_mu(mu)])

    # ---- inference ----
    def predict(self, x: np.ndarray, mu: np.ndarray) -> np.ndarray:
        """``x`` (N,2), ``mu`` (N,2) -> (N,3) = (u1,u2,p)."""
        return self.net.predict(self._features(x, mu))

    # ---- physics-residual forward+backward (5-point FD stencil) ----
    def _physics_forward_backward(self, x_int: np.ndarray, mu_int: np.ndarray,
                                  h: float
                                  ) -> Tuple[float, List[np.ndarray], List[np.ndarray]]:
        n = x_int.shape[0]
        mu0 = mu_int[:, 0]
        mu1 = mu_int[:, 1]

        Xc = self._features(x_int, mu_int)
        Xp0 = Xc.copy(); Xp0[:, 0] += h
        Xm0 = Xc.copy(); Xm0[:, 0] -= h
        Xp1 = Xc.copy(); Xp1[:, 1] += h
        Xm1 = Xc.copy(); Xm1[:, 1] -= h

        Yc, Ac, Pc = self.net.forward(Xc)
        Yp0, Ap0, Pp0 = self.net.forward(Xp0)
        Ym0, Am0, Pm0 = self.net.forward(Xm0)
        Yp1, Ap1, Pp1 = self.net.forward(Xp1)
        Ym1, Am1, Pm1 = self.net.forward(Xm1)

        u1c, u2c = Yc[:, 0], Yc[:, 1]

        du1dx0 = (Yp0[:, 0] - Ym0[:, 0]) / (2 * h)
        du1dx1 = (Yp1[:, 0] - Ym1[:, 0]) / (2 * h)
        du2dx0 = (Yp0[:, 1] - Ym0[:, 1]) / (2 * h)
        du2dx1 = (Yp1[:, 1] - Ym1[:, 1]) / (2 * h)
        d2u1dx0 = (Yp0[:, 0] - 2 * u1c + Ym0[:, 0]) / h ** 2
        d2u1dx1 = (Yp1[:, 0] - 2 * u1c + Ym1[:, 0]) / h ** 2
        d2u2dx0 = (Yp0[:, 1] - 2 * u2c + Ym0[:, 1]) / h ** 2
        d2u2dx1 = (Yp1[:, 1] - 2 * u2c + Ym1[:, 1]) / h ** 2
        dpdx0 = (Yp0[:, 2] - Ym0[:, 2]) / (2 * h)
        dpdx1 = (Yp1[:, 2] - Ym1[:, 2]) / (2 * h)

        lap_u1 = d2u1dx0 + d2u1dx1
        lap_u2 = d2u2dx0 + d2u2dx1
        conv1 = du1dx0 * u1c + du1dx1 * u2c
        conv2 = du2dx0 * u1c + du2dx1 * u2c

        f1, f2 = forcing_numpy(x_int[:, 0], x_int[:, 1], mu1)
        R1 = -mu0 * lap_u1 + conv1 + dpdx0 - f1
        R2 = -mu0 * lap_u2 + conv2 + dpdx1 - f2
        R3 = du1dx0 + du2dx1

        loss_p = float(np.mean(R1 ** 2 + R2 ** 2 + R3 ** 2))

        # Normalise by the mean squared forcing to keep MSE_p O(1) regardless
        # of the forcing magnitude (which scales as mu1^3*pi^2 ~ O(100-300)).
        # We compute the per-sample forcing scale and use its mean as divisor.
        f_scale = float(np.mean(f1 ** 2 + f2 ** 2)) + 1e-8
        loss_p_norm = float(np.mean(R1 ** 2 + R2 ** 2 + R3 ** 2) / f_scale)

        # Adjoint: use normalized loss gradient (divide by f_scale)
        dR1 = 2.0 * R1 / (n * f_scale)
        dR2 = 2.0 * R2 / (n * f_scale)
        dR3 = 2.0 * R3 / (n * f_scale)

        d_lap_u1 = dR1 * (-mu0)
        d_conv1 = dR1.copy()
        d_dpdx0 = dR1.copy()
        d_lap_u2 = dR2 * (-mu0)
        d_conv2 = dR2.copy()
        d_dpdx1 = dR2.copy()

        d_du1dx0 = dR3.copy()
        d_du2dx1 = dR3.copy()
        d_du1dx1 = np.zeros(n)
        d_du2dx0 = np.zeros(n)
        d_u1c = np.zeros(n)
        d_u2c = np.zeros(n)

        # conv1 = du1dx0*u1c + du1dx1*u2c
        d_du1dx0 += d_conv1 * u1c
        d_u1c += d_conv1 * du1dx0
        d_du1dx1 += d_conv1 * u2c
        d_u2c += d_conv1 * du1dx1
        # conv2 = du2dx0*u1c + du2dx1*u2c
        d_du2dx0 += d_conv2 * u1c
        d_u1c += d_conv2 * du2dx0
        d_du2dx1 += d_conv2 * u2c
        d_u2c += d_conv2 * du2dx1

        d_d2u1dx0 = d_lap_u1.copy()
        d_d2u1dx1 = d_lap_u1.copy()
        d_d2u2dx0 = d_lap_u2.copy()
        d_d2u2dx1 = d_lap_u2.copy()

        dYc = np.zeros_like(Yc)
        dYp0 = np.zeros_like(Yp0)
        dYm0 = np.zeros_like(Ym0)
        dYp1 = np.zeros_like(Yp1)
        dYm1 = np.zeros_like(Ym1)

        # first derivatives
        dYp0[:, 0] += d_du1dx0 / (2 * h); dYm0[:, 0] -= d_du1dx0 / (2 * h)
        dYp1[:, 0] += d_du1dx1 / (2 * h); dYm1[:, 0] -= d_du1dx1 / (2 * h)
        dYp0[:, 1] += d_du2dx0 / (2 * h); dYm0[:, 1] -= d_du2dx0 / (2 * h)
        dYp1[:, 1] += d_du2dx1 / (2 * h); dYm1[:, 1] -= d_du2dx1 / (2 * h)
        dYp0[:, 2] += d_dpdx0 / (2 * h); dYm0[:, 2] -= d_dpdx0 / (2 * h)
        dYp1[:, 2] += d_dpdx1 / (2 * h); dYm1[:, 2] -= d_dpdx1 / (2 * h)

        # second derivatives
        dYp0[:, 0] += d_d2u1dx0 / h ** 2
        dYm0[:, 0] += d_d2u1dx0 / h ** 2
        d_u1c += -2.0 * d_d2u1dx0 / h ** 2
        dYp1[:, 0] += d_d2u1dx1 / h ** 2
        dYm1[:, 0] += d_d2u1dx1 / h ** 2
        d_u1c += -2.0 * d_d2u1dx1 / h ** 2
        dYp0[:, 1] += d_d2u2dx0 / h ** 2
        dYm0[:, 1] += d_d2u2dx0 / h ** 2
        d_u2c += -2.0 * d_d2u2dx0 / h ** 2
        dYp1[:, 1] += d_d2u2dx1 / h ** 2
        dYm1[:, 1] += d_d2u2dx1 / h ** 2
        d_u2c += -2.0 * d_d2u2dx1 / h ** 2

        dYc[:, 0] += d_u1c
        dYc[:, 1] += d_u2c

        dWc, dbc = self.net._backward(Ac, Pc, dYc)
        dWp0, dbp0 = self.net._backward(Ap0, Pp0, dYp0)
        dWm0, dbm0 = self.net._backward(Am0, Pm0, dYm0)
        dWp1, dbp1 = self.net._backward(Ap1, Pp1, dYp1)
        dWm1, dbm1 = self.net._backward(Am1, Pm1, dYm1)

        dW = [a + b + c + d + e for a, b, c, d, e in
              zip(dWc, dWp0, dWm0, dWp1, dWm1)]
        db = [a + b + c + d + e for a, b, c, d, e in
              zip(dbc, dbp0, dbm0, dbp1, dbm1)]
        return loss_p_norm, dW, db

    # ---- boundary loss forward+backward ----
    def _boundary_forward_backward(self, x_bnd: np.ndarray, mu_bnd: np.ndarray,
                                   mu_pin: np.ndarray
                                   ) -> Tuple[float, List[np.ndarray], List[np.ndarray]]:
        n_b = x_bnd.shape[0]
        n_pin = mu_pin.shape[0]

        Xb = self._features(x_bnd, mu_bnd)
        Yb, Ab, Pb = self.net.forward(Xb)
        diff_u = Yb[:, 0:2]
        loss_vel = float(np.mean(np.sum(diff_u ** 2, axis=1)))
        dYb = np.zeros_like(Yb)
        dYb[:, 0:2] = 2.0 * diff_u / n_b

        x_pin = np.zeros((n_pin, 2))
        Xpin = self._features(x_pin, mu_pin)
        Ypin, Apin, Ppin = self.net.forward(Xpin)
        loss_pin = float(np.mean(Ypin[:, 2] ** 2))
        dYpin = np.zeros_like(Ypin)
        dYpin[:, 2] = 2.0 * Ypin[:, 2] / n_pin

        dWb, dbb = self.net._backward(Ab, Pb, dYb)
        dWpin, dbpin = self.net._backward(Apin, Ppin, dYpin)
        dW = [a + b for a, b in zip(dWb, dWpin)]
        db = [a + b for a, b in zip(dbb, dbpin)]
        return loss_vel + loss_pin, dW, db

    # ---- one training epoch ----
    def train_step(self, coll: Collocation, lr: float, lambda_p: float,
                   h: float = 1e-3, lambda_b: float = 1.0
                   ) -> Tuple[float, float, float]:
        """One Adam step.

        ``lambda_b`` up-weights the boundary term relative to the (normalised)
        physics term.  Set ``lambda_b > 1`` when the boundary loss is
        competing with a large physics residual (e.g., during Phase 2 after a
        boundary-only warmup).
        """
        loss_p, dWp, dbp = self._physics_forward_backward(coll.x_int, coll.mu_int, h)
        loss_b, dWb, dbb = self._boundary_forward_backward(
            coll.x_bnd, coll.mu_bnd, coll.mu_pin)

        dW = [lambda_b * wb + lambda_p * wp for wb, wp in zip(dWb, dWp)]
        db = [lambda_b * bb_ + lambda_p * bp for bb_, bp in zip(dbb, dbp)]
        self.net._adam_step(dW, db, lr)

        total = lambda_b * loss_b + lambda_p * loss_p
        return total, loss_b, loss_p

    def train(self, n_interior: int, n_boundary: int, n_pin: int,
             epochs: int, lr: float = 1e-3, lambda_p: float = 1.0,
             lambda_b: float = 1.0,
             h: float = 1e-3, seed: int = 42, verbose: bool = False,
             log_every: int = 100
             ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        rng = np.random.default_rng(seed)
        hist_total = np.zeros(epochs)
        hist_b = np.zeros(epochs)
        hist_p = np.zeros(epochs)
        for ep in range(epochs):
            coll = sample_collocation_points(
                n_interior, n_boundary, n_pin,
                self.mu0_range, self.mu1_range, rng)
            total, lb, lp = self.train_step(coll, lr, lambda_p, h, lambda_b=lambda_b)
            hist_total[ep] = total
            hist_b[ep] = lb
            hist_p[ep] = lp
            if verbose and ep % log_every == 0:
                print(f"  epoch {ep:5d}  total {total:.3e}  "
                      f"MSE_b {lb:.3e}  MSE_p {lp:.3e}")
        return hist_total, hist_b, hist_p

    # ---- dynamic weight balancing training ----
    def train_step_dynamic(self, coll: Collocation, lr: float, lambda_p: float,
                           lambda_b: float, h: float = 1e-3, alpha: float = 0.9
                           ) -> Tuple[float, float, float, float]:
        """One Adam step using Learning Rate Annealing for PINNs (Wang et al., 2021).
        
        Returns total_loss, loss_b, loss_p, and the updated lambda_b.
        """
        loss_p, dWp, dbp = self._physics_forward_backward(coll.x_int, coll.mu_int, h)
        loss_b, dWb, dbb = self._boundary_forward_backward(
            coll.x_bnd, coll.mu_bnd, coll.mu_pin)

        # Extract the gradients of the loss with respect to the last shared hidden layer
        # In our architecture, the last layer's weights are dWp[-1] and dWb[-1]
        grad_p_last = dWp[-1]
        grad_b_last = dWb[-1]

        # Compute max physics gradient magnitude and mean boundary gradient magnitude
        max_grad_p = float(np.max(np.abs(grad_p_last)))
        mean_grad_b = float(np.mean(np.abs(grad_b_last)))

        # Compute adaptive weight lambda_hat
        if mean_grad_b < 1e-12:
            lambda_hat = 1e4
        else:
            lambda_hat = max_grad_p / mean_grad_b
            
        # Optional: clip lambda_hat to avoid exploding weights from extreme batches
        lambda_hat = min(lambda_hat, 1e4)

        # Update lambda_b using exponential moving average
        new_lambda_b = alpha * lambda_b + (1.0 - alpha) * lambda_hat

        # Apply gradients using the UPDATED lambda_b
        dW = [new_lambda_b * wb + lambda_p * wp for wb, wp in zip(dWb, dWp)]
        db = [new_lambda_b * bb_ + lambda_p * bp for bb_, bp in zip(dbb, dbp)]
        self.net._adam_step(dW, db, lr)

        total = new_lambda_b * loss_b + lambda_p * loss_p
        return total, loss_b, loss_p, new_lambda_b

    def train_dynamic(self, n_interior: int, n_boundary: int, n_pin: int,
                      epochs: int, lr: float = 1e-3, lambda_p: float = 1.0,
                      lambda_b_init: float = 1.0, alpha: float = 0.9,
                      h: float = 1e-3, seed: int = 42, verbose: bool = False,
                      log_every: int = 100
                      ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        rng = np.random.default_rng(seed)
        hist_total = np.zeros(epochs)
        hist_b = np.zeros(epochs)
        hist_p = np.zeros(epochs)
        hist_lambda_b = np.zeros(epochs)
        
        current_lambda_b = lambda_b_init
        
        for ep in range(epochs):
            coll = sample_collocation_points(
                n_interior, n_boundary, n_pin,
                self.mu0_range, self.mu1_range, rng)
            total, lb, lp, next_lambda_b = self.train_step_dynamic(
                coll, lr, lambda_p, current_lambda_b, h, alpha)
                
            hist_total[ep] = total
            hist_b[ep] = lb
            hist_p[ep] = lp
            hist_lambda_b[ep] = next_lambda_b
            
            current_lambda_b = next_lambda_b
            
            if verbose and ep % log_every == 0:
                print(f"  epoch {ep:5d}  total {total:.3e}  "
                      f"MSE_b {lb:.3e}  MSE_p {lp:.3e}  lambda_b {current_lambda_b:.3f}")
                      
        return hist_total, hist_b, hist_p, hist_lambda_b

    # ---- I/O ----
    def save(self, path: Path) -> None:
        weights = self.net.get_weights()
        n = len(self.net.W)
        np.savez_compressed(
            path,
            layer_sizes=np.array(self.layer_sizes),
            n_weight_layers=np.array(n),
            **{f"W{i}": weights[i] for i in range(n)},
            **{f"b{i}": weights[n + i] for i in range(n)},
            mu0_range=np.array(self.mu0_range),
            mu1_range=np.array(self.mu1_range),
        )

    @classmethod
    def load(cls, path: Path) -> "PINNModel":
        d = np.load(path, allow_pickle=False)
        layer_sizes = [int(v) for v in d["layer_sizes"]]
        n = int(d["n_weight_layers"])
        mu0_range = tuple(d["mu0_range"])
        mu1_range = tuple(d["mu1_range"])
        model = cls(mu0_range, mu1_range, hidden_sizes=layer_sizes[1:-1])
        weights = [d[f"W{i}"] for i in range(n)] + [d[f"b{i}"] for i in range(n)]
        model.net.set_weights(weights)
        return model
