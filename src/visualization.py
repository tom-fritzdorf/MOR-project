"""Visualisation utilities.

:class:`Visualizer` implements the nine plots specified for Task 1, but each
plot is a self-contained method so that downstream tasks can selectively reuse
them (Task 3 in particular will want plots 2, 3, 6, 7, 8 with PODNN data).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
import seaborn as sns

from .problem import NavierStokesProblem


def apply_seaborn_theme() -> None:
    sns.set_theme(style="whitegrid", palette="muted", font_scale=1.2)


class Visualizer:
    """Make the diagnostic plots and save them to ``config.plots_dir``."""

    def __init__(self, problem: NavierStokesProblem) -> None:
        apply_seaborn_theme()
        self.problem = problem
        coords = problem.mesh.coordinates()
        self.triang = mtri.Triangulation(coords[:, 0], coords[:, 1],
                                         problem.mesh.cells())

    # =====================================================================
    # Low-level helpers
    # =====================================================================
    def velocity_magnitude(self, u_vec: np.ndarray) -> np.ndarray:
        u_fn = self.problem.velocity_function(u_vec)
        vals = u_fn.compute_vertex_values(self.problem.mesh).reshape(2, -1)
        return np.sqrt(vals[0] ** 2 + vals[1] ** 2)

    def pressure_vertex_values(self, p_vec: np.ndarray) -> np.ndarray:
        p_fn = self.problem.pressure_function(p_vec)
        return p_fn.compute_vertex_values(self.problem.mesh)

    def _savepath(self, name: str) -> Path:
        return self.problem.config.plots_dir / name

    # =====================================================================
    # Plot 1: FOM showcase
    # =====================================================================
    def plot_fom_showcase(self, fom_solver_fn: Callable,
                          params: Sequence[Tuple[float, float]],
                          labels: Sequence[str],
                          filename: str = "01_fom_showcase.png") -> Path:
        """``fom_solver_fn(mu0, mu1, w_init=None)`` -> (u_vec, p_vec, w_next)."""
        fig, axes = plt.subplots(3, 2, figsize=(11, 13))
        w_init = None
        for row, ((mu0, mu1), label) in enumerate(zip(params, labels)):
            u_vec, p_vec, w_init = fom_solver_fn(mu0, mu1, w_init=w_init)
            umag = self.velocity_magnitude(u_vec)
            pval = self.pressure_vertex_values(p_vec)

            ax = axes[row, 0]
            cs = ax.tricontourf(self.triang, umag, levels=20, cmap="viridis")
            plt.colorbar(cs, ax=ax)
            ax.set_title(f"{label}: $|u|$  ($\\mu_0={mu0},\\,\\mu_1={mu1}$)")
            ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_aspect("equal")

            ax = axes[row, 1]
            vmax = float(np.max(np.abs(pval))) or 1.0
            cs = ax.tricontourf(self.triang, pval, levels=20, cmap="RdBu_r",
                                vmin=-vmax, vmax=vmax)
            plt.colorbar(cs, ax=ax)
            ax.set_title(f"{label}: $p$")
            ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_aspect("equal")

        fig.suptitle("FOM solution showcase", fontsize=15)
        fig.tight_layout()
        out = self._savepath(filename)
        fig.savefig(out, dpi=200, bbox_inches="tight")
        plt.close(fig)
        return out

    # =====================================================================
    # Plot 2: singular-value decay
    # =====================================================================
    def plot_singular_values(self, sigmas_u: np.ndarray, sigmas_p: np.ndarray,
                             threshold: float,
                             filename: str = "02_singular_values.png") -> Path:
        fig, ax = plt.subplots(figsize=(8, 5.5))
        s_u = sigmas_u / sigmas_u[0]
        s_p = sigmas_p / sigmas_p[0]
        ax.semilogy(np.arange(1, len(s_u) + 1), s_u, "o-", label="Velocity")
        ax.semilogy(np.arange(1, len(s_p) + 1), s_p, "s-", label="Pressure")
        ax.axhline(np.sqrt(max(1 - threshold, 1e-16)), ls="--", color="grey",
                   label=f"$\\sqrt{{1-{threshold:.4f}}}$ ref.")
        ax.set_xlabel("Mode index")
        ax.set_ylabel(r"$\sigma_i / \sigma_1$")
        ax.set_title("Normalised singular value decay")
        ax.legend()
        fig.tight_layout()
        out = self._savepath(filename)
        fig.savefig(out, dpi=200, bbox_inches="tight")
        plt.close(fig)
        return out

    # =====================================================================
    # Plot 3: cumulative energy
    # =====================================================================
    def plot_cumulative_energy(self, sigmas_u: np.ndarray, sigmas_p: np.ndarray,
                               r_u: int, r_p: int,
                               filename: str = "03_cumulative_energy.png") -> Path:
        eu = np.cumsum(sigmas_u ** 2) / np.sum(sigmas_u ** 2)
        ep = np.cumsum(sigmas_p ** 2) / np.sum(sigmas_p ** 2)
        fig, ax = plt.subplots(figsize=(8, 5.5))
        ax.plot(np.arange(1, len(eu) + 1), eu, "o-", label="Velocity")
        ax.plot(np.arange(1, len(ep) + 1), ep, "s-", label="Pressure")
        for lvl in (0.99, 0.999, 0.9999):
            ax.axhline(lvl, ls="--", color="grey", alpha=0.5)
            ax.text(len(eu) * 0.6, lvl + 0.0005, f"{lvl*100:.2f}%",
                    color="grey", fontsize=9)
        ax.axvline(r_u, ls=":", color="C0", alpha=0.7, label=f"$r_u={r_u}$")
        ax.axvline(r_p, ls=":", color="C1", alpha=0.7, label=f"$r_p={r_p}$")
        ax.set_xlabel("Number of modes")
        ax.set_ylabel("Cumulative energy")
        ax.set_title("Cumulative POD energy")
        ax.set_ylim(0.9, 1.001)
        ax.legend()
        fig.tight_layout()
        out = self._savepath(filename)
        fig.savefig(out, dpi=200, bbox_inches="tight")
        plt.close(fig)
        return out

    # =====================================================================
    # Plot 4: POD modes
    # =====================================================================
    def plot_pod_modes(self, Phi_u: np.ndarray, Phi_p: np.ndarray,
                       n_u: int = 4, n_p: int = 2,
                       filename: str = "04_pod_modes.png") -> Path:
        n_u = min(n_u, Phi_u.shape[1])
        n_p = min(n_p, Phi_p.shape[1])
        n_total = n_u + n_p
        ncols = min(n_total, 3)
        nrows = (n_total + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
        axes_flat = np.array(axes).ravel()
        for k in range(n_u):
            ax = axes_flat[k]
            umag = self.velocity_magnitude(Phi_u[:, k])
            cs = ax.tricontourf(self.triang, umag, levels=20, cmap="viridis")
            plt.colorbar(cs, ax=ax)
            ax.set_title(f"Velocity POD mode {k + 1}")
            ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_aspect("equal")
        for k in range(n_p):
            ax = axes_flat[n_u + k]
            pval = self.pressure_vertex_values(Phi_p[:, k])
            vmax = float(np.max(np.abs(pval))) or 1.0
            cs = ax.tricontourf(self.triang, pval, levels=20, cmap="RdBu_r",
                                vmin=-vmax, vmax=vmax)
            plt.colorbar(cs, ax=ax)
            ax.set_title(f"Pressure POD mode {k + 1}")
            ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_aspect("equal")
        for ax in axes_flat[n_total:]:
            ax.set_visible(False)
        fig.suptitle("Leading POD modes", fontsize=15)
        fig.tight_layout()
        out = self._savepath(filename)
        fig.savefig(out, dpi=200, bbox_inches="tight")
        plt.close(fig)
        return out

    # =====================================================================
    # Plot 5: ROM vs FOM
    # =====================================================================
    def plot_rom_vs_fom(self, test_params: np.ndarray,
                        fom_u: np.ndarray, rom_u: np.ndarray,
                        filename: str = "05_rom_vs_fom.png") -> Path:
        order = np.argsort(test_params[:, 0])
        picks = [int(order[0]), int(order[len(order) // 2]), int(order[-1])]
        fig, axes = plt.subplots(3, 3, figsize=(14, 12))
        for row, idx in enumerate(picks):
            umag_f = self.velocity_magnitude(fom_u[:, idx])
            umag_r = self.velocity_magnitude(rom_u[:, idx])
            err = umag_r - umag_f
            vmax = max(umag_f.max(), umag_r.max())
            emax = float(np.max(np.abs(err))) or 1.0
            mu0, mu1 = test_params[idx]
            cs0 = axes[row, 0].tricontourf(self.triang, umag_f, levels=20,
                                            cmap="viridis", vmin=0, vmax=vmax)
            plt.colorbar(cs0, ax=axes[row, 0])
            axes[row, 0].set_title(
                f"FOM $|u|$  ($\\mu_0={mu0:.2f},\\,\\mu_1={mu1:.2f}$)")
            cs1 = axes[row, 1].tricontourf(self.triang, umag_r, levels=20,
                                            cmap="viridis", vmin=0, vmax=vmax)
            plt.colorbar(cs1, ax=axes[row, 1])
            axes[row, 1].set_title("ROM $|u|$")
            cs2 = axes[row, 2].tricontourf(self.triang, err, levels=20,
                                            cmap="RdBu_r", vmin=-emax, vmax=emax)
            plt.colorbar(cs2, ax=axes[row, 2])
            axes[row, 2].set_title("Pointwise $|u|$ error")
            for ax in axes[row]:
                ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_aspect("equal")
        fig.suptitle("FOM vs ROM velocity comparison", fontsize=15)
        fig.tight_layout()
        out = self._savepath(filename)
        fig.savefig(out, dpi=200, bbox_inches="tight")
        plt.close(fig)
        return out

    # =====================================================================
    # Plot 6: error vs number of modes
    # =====================================================================
    def plot_error_vs_modes(self, Nrs: np.ndarray,
                            mean_u: np.ndarray, mean_p: np.ndarray,
                            filename: str = "06_error_vs_modes.png") -> Path:
        fig, ax = plt.subplots(figsize=(8, 5.5))
        ax.semilogy(Nrs, mean_u, "o-", label="Velocity (mean rel. $L^2$)")
        ax.semilogy(Nrs, mean_p, "s-", label="Pressure (mean rel. $L^2$)")
        ax.set_xlabel("Number of velocity modes")
        ax.set_ylabel("Mean relative $L^2$ error on test set")
        ax.set_title("ROM error vs basis size")
        ax.legend()
        fig.tight_layout()
        out = self._savepath(filename)
        fig.savefig(out, dpi=200, bbox_inches="tight")
        plt.close(fig)
        return out

    # =====================================================================
    # Plot 7: error distribution in parameter space
    # =====================================================================
    def plot_error_parameter_space(self, test_params: np.ndarray,
                                   rel_l2_u: np.ndarray,
                                   filename: str = "07_error_parameter_space.png"
                                   ) -> Path:
        fig, ax = plt.subplots(figsize=(8, 5.5))
        sc = ax.scatter(test_params[:, 0], test_params[:, 1],
                        c=rel_l2_u, cmap="viridis", s=120,
                        edgecolor="black", linewidth=0.5)
        plt.colorbar(sc, ax=ax, label="Relative $L^2$ velocity error")
        ax.set_xlabel(r"$\mu_0$ (viscosity)")
        ax.set_ylabel(r"$\mu_1$ (forcing freq.)")
        ax.set_title("Test-set error across parameter space")
        fig.tight_layout()
        out = self._savepath(filename)
        fig.savefig(out, dpi=200, bbox_inches="tight")
        plt.close(fig)
        return out

    # =====================================================================
    # Plot 8: speedup summary
    # =====================================================================
    def plot_speedup(self, fom_mean: float, rom_mean: float,
                     offline_total: float,
                     filename: str = "08_speedup.png") -> Path:
        fig, ax = plt.subplots(figsize=(8, 5.5))
        cats = ["FOM (online)", "ROM (online)", "ROM (offline)"]
        vals = [fom_mean, rom_mean, offline_total]
        bars = ax.bar(cats, vals, color=["#4C72B0", "#55A868", "#C44E52"])
        ax.set_ylabel("Mean wall time (s)")
        ax.set_yscale("log")
        speedup = fom_mean / max(rom_mean, 1e-12)
        ax.set_title(f"Wall-time comparison — ROM online speedup: {speedup:.1f}x")
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, v * 1.05,
                    f"{v:.3g}s", ha="center", fontsize=10)
        fig.tight_layout()
        out = self._savepath(filename)
        fig.savefig(out, dpi=200, bbox_inches="tight")
        plt.close(fig)
        return out

    # =====================================================================
    # Plot 9: Newton convergence
    # =====================================================================
    def plot_newton_convergence(self, fom_residuals: List[float],
                                rom_residuals: List[float],
                                filename: str = "09_newton_convergence.png"
                                ) -> Path:
        fig, ax = plt.subplots(figsize=(8, 5.5))
        ax.semilogy(np.arange(1, len(fom_residuals) + 1), fom_residuals,
                    "o-", label="FOM Newton")
        ax.semilogy(np.arange(1, len(rom_residuals) + 1), rom_residuals,
                    "s-", label="ROM Newton")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Residual norm")
        ax.set_title("Newton convergence (representative test case)")
        ax.legend()
        fig.tight_layout()
        out = self._savepath(filename)
        fig.savefig(out, dpi=200, bbox_inches="tight")
        plt.close(fig)
        return out
