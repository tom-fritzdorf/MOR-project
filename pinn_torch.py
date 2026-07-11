#!/usr/bin/env python3
"""
PINN for the parametric steady Navier-Stokes problem -- PyTorch re-implementation
following the professor's lab recipe:
  * exact derivatives via torch.autograd (no finite differences),
  * HARD boundary conditions  u_i = phi(x) * N_i(x,mu),  phi = 16 x0(1-x0)x1(1-x1)
    (zero on dOmega by construction; the Dirichlet loss is dropped -- exactly the
    trick taught in PINN_Es1/Es2),
  * Adam warm-up  ->  L-BFGS (strong Wolfe),
  * optional random Fourier features on x to beat spectral bias (the forcing has
    ~mu1^2*pi spatial frequency).

Runs in an isolated torch env; evaluation uses FE matrices exported to
data/pinn_fe_export.npz (no FEniCS needed here).

Usage:
    python pinn_torch.py --mode single --mu0 1.0 --mu1 1.5
    python pinn_torch.py --mode parametric
"""
import argparse, time
from pathlib import Path
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn

torch.set_default_dtype(torch.float64)
DATA = Path(__file__).resolve().parent / "data"
PI = np.pi


# --------------------------------------------------------------------------
# Data / FE export
# --------------------------------------------------------------------------
def load_fe():
    d = np.load(DATA / "pinn_fe_export.npz")
    def spmat(k):
        return sp.coo_matrix((d[f"{k}_data"], (d[f"{k}_row"], d[f"{k}_col"])),
                             shape=tuple(d[f"{k}_shape"])).tocsr()
    fe = dict(N_u=int(d["N_u"]), N_p=int(d["N_p"]),
              pd0=d["parent_dofs_0"], pd1=d["parent_dofs_1"],
              xu=d["dof_coords_u"], xp=d["dof_coords_p"],
              Mu=spmat("Mu"), Mp=spmat("Mp"), Ku=spmat("Ku"),
              mu0_range=d["mu0_range"], mu1_range=d["mu1_range"])
    return fe


def rel_errors(fe, U, P, u_true, p_true):
    """M-weighted relative errors for one sample (numpy vectors)."""
    Mu, Mp, Ku = fe["Mu"], fe["Mp"], fe["Ku"]
    du, dp = U - u_true, P - p_true
    l2u = np.sqrt(max(du @ (Mu @ du), 0) / (u_true @ (Mu @ u_true)))
    l2p = np.sqrt(max(dp @ (Mp @ dp), 0) / (p_true @ (Mp @ p_true)))
    Hu = Mu + Ku
    h1u = np.sqrt(max(du @ (Hu @ du), 0) / (u_true @ (Hu @ u_true)))
    return l2u, l2p, h1u


# --------------------------------------------------------------------------
# Network with hard BC + optional Fourier features
# --------------------------------------------------------------------------
class PINN(nn.Module):
    def __init__(self, mu0_range, mu1_range, hidden=(64, 64, 64, 64),
                 n_fourier=24, sigma=2.0, seed=0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.n_fourier = n_fourier
        if n_fourier > 0:
            B = torch.randn(2, n_fourier, generator=g) * sigma
            self.register_buffer("B", B)
            in_dim = 2 + 2 * n_fourier + 2
        else:
            self.B = None
            in_dim = 4
        self.mu0_mid = 0.5 * (mu0_range[0] + mu0_range[1])
        self.mu0_half = 0.5 * (mu0_range[1] - mu0_range[0])
        self.mu1_mid = 0.5 * (mu1_range[0] + mu1_range[1])
        self.mu1_half = 0.5 * (mu1_range[1] - mu1_range[0])
        layers, d = [], in_dim
        for h in hidden:
            layers += [nn.Linear(d, h), nn.Tanh()]; d = h
        layers += [nn.Linear(d, 3)]
        self.net = nn.Sequential(*layers)
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight); nn.init.zeros_(m.bias)

    def _features(self, x0, x1, mu0, mu1):
        mu0n = (mu0 - self.mu0_mid) / self.mu0_half
        mu1n = (mu1 - self.mu1_mid) / self.mu1_half
        if self.B is None:
            return torch.cat([x0, x1, mu0n, mu1n], dim=1)
        x = torch.cat([x0, x1], dim=1)
        proj = 2 * PI * (x @ self.B)
        return torch.cat([x0, x1, torch.sin(proj), torch.cos(proj), mu0n, mu1n], dim=1)

    def raw(self, x0, x1, mu0, mu1):
        return self.net(self._features(x0, x1, mu0, mu1))

    def fields(self, x0, x1, mu0, mu1):
        out = self.raw(x0, x1, mu0, mu1)
        phi = (16.0 * x0 * (1 - x0) * x1 * (1 - x1))          # hard no-slip BC
        u1 = phi * out[:, 0:1]
        u2 = phi * out[:, 1:2]
        p = out[:, 2:3]
        return u1, u2, p


def forcing(x0, x1, mu1):
    f1 = (-(mu1**3 * PI**2 * torch.cos(mu1**2 * PI * x0) - mu1**2 * PI**2)
          * torch.sin(mu1 * PI * x1) * torch.cos(mu1 * PI * x1)
          + mu1 * PI * torch.cos(mu1 * PI * x0) * torch.cos(mu1 * PI * x1))
    f2 = (-(-mu1**3 * PI**2 * torch.cos(mu1**2 * PI * x1) + mu1**2 * PI**2)
          * torch.sin(mu1 * PI * x0) * torch.cos(mu1 * PI * x0)
          - mu1 * PI * torch.sin(mu1 * PI * x0) * torch.sin(mu1 * PI * x1))
    return f1, f2


def grad(y, x):
    return torch.autograd.grad(y, x, torch.ones_like(y), create_graph=True)[0]


def residual(model, x0, x1, mu0, mu1):
    u1, u2, p = model.fields(x0, x1, mu0, mu1)
    u1_x0 = grad(u1, x0); u1_x1 = grad(u1, x1)
    u2_x0 = grad(u2, x0); u2_x1 = grad(u2, x1)
    p_x0 = grad(p, x0);   p_x1 = grad(p, x1)
    lap_u1 = grad(u1_x0, x0) + grad(u1_x1, x1)
    lap_u2 = grad(u2_x0, x0) + grad(u2_x1, x1)
    f1, f2 = forcing(x0, x1, mu1)
    R1 = -mu0 * lap_u1 + (u1_x0 * u1 + u1_x1 * u2) + p_x0 - f1
    R2 = -mu0 * lap_u2 + (u2_x0 * u1 + u2_x1 * u2) + p_x1 - f2
    R3 = u1_x0 + u2_x1
    # normalise momentum residual by the local forcing scale (per-mu balance)
    scale = (f1.detach()**2 + f2.detach()**2).mean() + 1e-8
    return (R1**2 + R2**2).mean() / scale + (R3**2).mean()


def sample(n, mu0_range, mu1_range, fixed_mu=None, dev="cpu"):
    x0 = torch.rand(n, 1, device=dev, requires_grad=True)
    x1 = torch.rand(n, 1, device=dev, requires_grad=True)
    if fixed_mu is None:
        mu0 = torch.rand(n, 1, device=dev) * (mu0_range[1]-mu0_range[0]) + mu0_range[0]
        mu1 = torch.rand(n, 1, device=dev) * (mu1_range[1]-mu1_range[0]) + mu1_range[0]
    else:
        mu0 = torch.full((n, 1), fixed_mu[0], device=dev)
        mu1 = torch.full((n, 1), fixed_mu[1], device=dev)
    return x0, x1, mu0, mu1


# --------------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------------
@torch.no_grad()
def reconstruct(model, fe, mu):
    xu = torch.tensor(fe["xu"]); xp = torch.tensor(fe["xp"])
    mu0 = torch.full((xu.shape[0], 1), mu[0]); mu1 = torch.full((xu.shape[0], 1), mu[1])
    u1, u2, _ = model.fields(xu[:, 0:1], xu[:, 1:2], mu0, mu1)
    mu0p = torch.full((xp.shape[0], 1), mu[0]); mu1p = torch.full((xp.shape[0], 1), mu[1])
    _, _, p = model.fields(xp[:, 0:1], xp[:, 1:2], mu0p, mu1p)
    # pressure pin p(0,0)=0
    z = torch.zeros(1, 1)
    _, _, p0 = model.fields(z, z, torch.tensor([[mu[0]]]), torch.tensor([[mu[1]]]))
    U = np.zeros(fe["N_u"]); U[fe["pd0"]] = u1[:, 0].numpy(); U[fe["pd1"]] = u2[:, 0].numpy()
    P = (p[:, 0] - p0[0, 0]).numpy()
    return U, P


def evaluate(model, fe, test_params, u_fom, p_fom, n=None):
    idx = range(test_params.shape[0]) if n is None else range(0, test_params.shape[0], max(1, test_params.shape[0]//n))
    eu = []
    for i in idx:
        U, P = reconstruct(model, fe, test_params[i])
        l2u, _, _ = rel_errors(fe, U, P, u_fom[:, i], p_fom[:, i])
        eu.append(l2u)
    return float(np.mean(eu))


def full_report(model, fe, test_params, u_fom, p_fom):
    L2U, L2P, H1U = [], [], []
    for i in range(test_params.shape[0]):
        U, P = reconstruct(model, fe, test_params[i])
        a, b, c = rel_errors(fe, U, P, u_fom[:, i], p_fom[:, i])
        L2U.append(a); L2P.append(b); H1U.append(c)
    return np.array(L2U), np.array(L2P), np.array(H1U)


# --------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------
def train(mode, mu0, mu1, adam_iters, lbfgs_iters, n_coll, n_fourier, sigma,
          hidden, seed, eval_every):
    fe = load_fe()
    fom = np.load(DATA / "fom_solutions_test.npz")
    test_params = fom["test_params"]; u_fom = fom["u_fom_test"]; p_fom = fom["p_fom_test"]
    mu0_range = tuple(fe["mu0_range"]); mu1_range = tuple(fe["mu1_range"])
    fixed = (mu0, mu1) if mode == "single" else None

    model = PINN(mu0_range, mu1_range, hidden=hidden, n_fourier=n_fourier,
                 sigma=sigma, seed=seed)
    print(f"mode={mode} fixed_mu={fixed} fourier={n_fourier} sigma={sigma} "
          f"hidden={hidden} params={sum(p.numel() for p in model.parameters())}")

    def ev():
        if mode == "single":
            U, P = reconstruct(model, fe, (mu0, mu1))
            # nearest test index for reference truth (single-mu compares to FOM at that mu)
            j = int(np.argmin(np.sum((test_params - np.array([mu0, mu1]))**2, axis=1)))
            return rel_errors(fe, U, P, u_fom[:, j], p_fom[:, j])[0]
        return evaluate(model, fe, test_params, u_fom, p_fom, n=15)

    t0 = time.time()
    # ---- Adam warm-up ----
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for it in range(adam_iters):
        opt.zero_grad()
        x0, x1, m0, m1 = sample(n_coll, mu0_range, mu1_range, fixed)
        loss = residual(model, x0, x1, m0, m1)
        loss.backward(); opt.step()
        if it % eval_every == 0:
            print(f"  adam {it:5d}  loss {loss.item():.3e}  relL2(u) {ev():.3f}  [{time.time()-t0:.0f}s]", flush=True)

    # ---- L-BFGS (strong Wolfe) on a fixed large batch ----
    x0, x1, m0, m1 = sample(4 * n_coll, mu0_range, mu1_range, fixed)
    opt = torch.optim.LBFGS(model.parameters(), lr=1.0, max_iter=lbfgs_iters,
                            history_size=50, line_search_fn="strong_wolfe",
                            tolerance_grad=1e-9, tolerance_change=1e-12)
    state = {"it": 0}
    def closure():
        opt.zero_grad()
        loss = residual(model, x0, x1, m0, m1)
        loss.backward()
        state["it"] += 1
        if state["it"] % 50 == 0:
            print(f"  lbfgs {state['it']:5d}  loss {loss.item():.3e}  relL2(u) {ev():.3f}  [{time.time()-t0:.0f}s]", flush=True)
        return loss
    opt.step(closure)

    print(f"\nFINAL relL2(u) probe: {ev():.4f}  [{time.time()-t0:.0f}s]")
    if mode == "parametric":
        L2U, L2P, H1U = full_report(model, fe, test_params, u_fom, p_fom)
        print(f"FULL 150-pt: relL2(u) mean={L2U.mean():.4f} median={np.median(L2U):.4f} "
              f"max={L2U.max():.4f} | relL2(p) mean={L2P.mean():.4f} | relH1(u) mean={H1U.mean():.4f}")
        np.savez_compressed(DATA / "pinn_torch_test_errors.npz",
                            rel_l2_u=L2U, rel_l2_p=L2P, rel_h1_u=H1U, test_params=test_params)
    torch.save(model.state_dict(), DATA / "pinn_torch_model.pt")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["single", "parametric"], default="single")
    ap.add_argument("--mu0", type=float, default=1.0)
    ap.add_argument("--mu1", type=float, default=1.5)
    ap.add_argument("--adam", type=int, default=2000)
    ap.add_argument("--lbfgs", type=int, default=2000)
    ap.add_argument("--coll", type=int, default=3000)
    ap.add_argument("--fourier", type=int, default=24)
    ap.add_argument("--sigma", type=float, default=2.0)
    ap.add_argument("--eval_every", type=int, default=500)
    a = ap.parse_args()
    train(a.mode, a.mu0, a.mu1, a.adam, a.lbfgs, a.coll, a.fourier, a.sigma,
          (64, 64, 64, 64), 0, a.eval_every)
