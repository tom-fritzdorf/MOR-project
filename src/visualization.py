"""Visualisation utilities.

:class:`Visualizer` implements the eleven plots for Task 1 (nine core plots
plus the Newton-convergence trace and the inf-sup instability diagnostic),
each as a self-contained method so downstream tasks can selectively reuse
them (Task 3 in particular will want plots 2, 3, 6, 7, 8 with PODNN data).

Plots over tabular/scalar series (singular values, energy, error-vs-modes,
Newton residuals, mode comparison, speedup, error-in-parameter-space) go
through a tidy long-form ``pandas.DataFrame`` and ``seaborn`` plotting
functions (``lineplot``/``scatterplot``/``barplot``) so hue, style and
legends are handled consistently. Plots over the unstructured FE mesh (FOM
fields, POD mode shapes, FOM-vs-ROM contours) stay on ``tricontourf`` --
seaborn has no unstructured-triangulation plot type -- but share the same
seaborn theme and palette for visual consistency.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
import pandas as pd
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers the '3d' projection)

from .problem import NavierStokesProblem

_PALETTE = "deep"


def apply_seaborn_theme() -> None:
    sns.set_theme(context="notebook", style="whitegrid",
                  palette=_PALETTE, font_scale=1.15)


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

    def _finish(self, fig, out: Path, suptitle: bool = False) -> Path:
        for ax in fig.axes:
            sns.despine(ax=ax)
        fig.tight_layout()
        fig.savefig(out, dpi=200, bbox_inches="tight")
        plt.close(fig)
        return out

    # =====================================================================
    # Plot 0: computational mesh
    # =====================================================================
    def plot_mesh(self, filename: str = "00_mesh.png") -> Path:
        """Taylor-Hood FE mesh on the unit square (P2 velocity / P1 pressure)."""
        mesh = self.problem.mesh
        n_cells = mesh.num_cells()
        n_vertices = mesh.num_vertices()
        n = self.problem.config.mesh_n

        fig, ax = plt.subplots(figsize=(7, 7))
        ax.triplot(self.triang, color=sns.color_palette(_PALETTE)[0],
                  linewidth=0.6)
        ax.set_title(
            f"Computational mesh: UnitSquareMesh({n}$\\times${n})\n"
            f"{n_cells} triangles, {n_vertices} vertices "
            f"(Taylor-Hood $P_2/P_1$)"
        )
        ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_aspect("equal")
        out = self._savepath(filename)
        return self._finish(fig, out)

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
        out = self._savepath(filename)
        return self._finish(fig, out)

    # =====================================================================
    # Plot 2: singular-value decay
    # =====================================================================
    def plot_singular_values(self, sigmas_u: np.ndarray, sigmas_p: np.ndarray,
                             threshold: float,
                             filename: str = "02_singular_values.png") -> Path:
        s_u = sigmas_u / sigmas_u[0]
        s_p = sigmas_p / sigmas_p[0]
        df = pd.concat([
            pd.DataFrame({"mode": np.arange(1, len(s_u) + 1),
                          "sigma": s_u, "field": "Velocity"}),
            pd.DataFrame({"mode": np.arange(1, len(s_p) + 1),
                          "sigma": s_p, "field": "Pressure"}),
        ], ignore_index=True)

        fig, ax = plt.subplots(figsize=(8, 5.5))
        sns.lineplot(data=df, x="mode", y="sigma", hue="field", style="field",
                    markers=False, dashes=False, linewidth=1.5, ax=ax)
        ax.set_yscale("log")
        ax.set_xscale("log")
        ax.axhline(np.sqrt(max(1 - threshold, 1e-16)), ls="--", color="grey",
                   label=f"$\\sqrt{{1-{threshold:.4f}}}$ ref.")
        ax.set_xlabel("Mode index")
        ax.set_ylabel(r"$\sigma_i / \sigma_1$")
        ax.set_title("Normalised singular value decay")
        ax.legend(title=None)
        out = self._savepath(filename)
        return self._finish(fig, out)

    # =====================================================================
    # Plot 3: cumulative energy
    # =====================================================================
    def plot_cumulative_energy(self, sigmas_u: np.ndarray, sigmas_p: np.ndarray,
                               r_u: int, r_p: int,
                               filename: str = "03_cumulative_energy.png") -> Path:
        eu = np.cumsum(sigmas_u ** 2) / np.sum(sigmas_u ** 2)
        ep = np.cumsum(sigmas_p ** 2) / np.sum(sigmas_p ** 2)
        df = pd.concat([
            pd.DataFrame({"mode": np.arange(1, len(eu) + 1),
                          "energy": eu, "field": "Velocity"}),
            pd.DataFrame({"mode": np.arange(1, len(ep) + 1),
                          "energy": ep, "field": "Pressure"}),
        ], ignore_index=True)

        fig, ax = plt.subplots(figsize=(8, 5.5))
        sns.lineplot(data=df, x="mode", y="energy", hue="field", style="field",
                    markers=False, dashes=False, linewidth=1.5, ax=ax)
        for lvl in (0.99, 0.999, 0.9999):
            ax.axhline(lvl, ls="--", color="grey", alpha=0.5)
            ax.text(max(r_u, r_p) * 0.7, lvl + 0.0005, f"{lvl*100:.2f}%",
                    color="grey", fontsize=9)
        ax.axvline(r_u, ls=":", color=sns.color_palette(_PALETTE)[0],
                   alpha=0.8, label=f"$r_u={r_u}$")
        ax.axvline(r_p, ls=":", color=sns.color_palette(_PALETTE)[1],
                   alpha=0.8, label=f"$r_p={r_p}$")
        ax.set_xlabel("Number of modes")
        ax.set_ylabel("Cumulative energy")
        ax.set_title("Cumulative POD energy")
        ax.set_ylim(0.9, 1.001)
        ax.set_xlim(0, max(r_u, r_p) + 20)
        handles, lbls = ax.get_legend_handles_labels()
        ax.legend(handles, lbls, title=None)
        out = self._savepath(filename)
        return self._finish(fig, out)

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
        out = self._savepath(filename)
        return self._finish(fig, out)

    # =====================================================================
    # Plot 5: ROM vs FOM
    # =====================================================================
    def plot_rom_vs_fom(self, test_params: np.ndarray,
                        fom_u: np.ndarray, rom_u: np.ndarray,
                        filename: str = "05_rom_vs_fom.png",
                        label: str = "ROM") -> Path:
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
            axes[row, 1].set_title(f"{label} $|u|$")
            cs2 = axes[row, 2].tricontourf(self.triang, err, levels=20,
                                            cmap="RdBu_r", vmin=-emax, vmax=emax)
            plt.colorbar(cs2, ax=axes[row, 2])
            axes[row, 2].set_title("Pointwise $|u|$ error")
            for ax in axes[row]:
                ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_aspect("equal")
        fig.suptitle(f"FOM vs {label} velocity comparison", fontsize=15)
        out = self._savepath(filename)
        return self._finish(fig, out)

    # =====================================================================
    # Plot 6: error vs number of modes
    # =====================================================================
    def plot_error_vs_modes(self, Nrs: np.ndarray,
                            mean_u: np.ndarray, mean_p: np.ndarray,
                            filename: str = "06_error_vs_modes.png") -> Path:
        df = pd.concat([
            pd.DataFrame({"r_u": Nrs, "error": mean_u, "field": "Velocity (mean rel. $L^2$)"}),
            pd.DataFrame({"r_u": Nrs, "error": mean_p, "field": "Pressure (mean rel. $L^2$)"}),
        ], ignore_index=True)

        fig, ax = plt.subplots(figsize=(8, 5.5))
        sns.lineplot(data=df, x="r_u", y="error", hue="field", style="field",
                    markers=True, dashes=False, markersize=7, ax=ax)
        ax.set_yscale("log")
        ax.set_xlabel("Number of velocity modes")
        ax.set_ylabel("Mean relative $L^2$ error on test set")
        ax.set_title("ROM error vs basis size (supremizer-stabilised)")
        ax.legend(title=None)
        out = self._savepath(filename)
        return self._finish(fig, out)

    # =====================================================================
    # Plot 7: error distribution in parameter space
    # =====================================================================
    def plot_error_parameter_space(self, test_params: np.ndarray,
                                   rel_l2_u: np.ndarray,
                                   filename: str = "07_error_parameter_space.png"
                                   ) -> Path:
        df = pd.DataFrame({
            "mu0": test_params[:, 0], "mu1": test_params[:, 1],
            "error": rel_l2_u,
        })
        fig, ax = plt.subplots(figsize=(8, 5.5))
        sc = ax.scatter(df["mu0"], df["mu1"], c=df["error"], cmap="viridis",
                        s=130, edgecolor="black", linewidth=0.6)
        plt.colorbar(sc, ax=ax, label="Relative $L^2$ velocity error")
        ax.set_xlabel(r"$\mu_0$ (viscosity)")
        ax.set_ylabel(r"$\mu_1$ (forcing freq.)")
        ax.set_title("Test-set error across parameter space")
        out = self._savepath(filename)
        return self._finish(fig, out)

    # =====================================================================
    # Plot 8: speedup summary
    # =====================================================================
    def plot_speedup(self, fom_mean: float, rom_mean: float,
                     offline_total: float,
                     filename: str = "08_speedup.png",
                     label: str = "ROM") -> Path:
        df = pd.DataFrame({
            "stage": ["FOM (online)", f"{label} (online)", f"{label} (offline)"],
            "time": [fom_mean, rom_mean, offline_total],
        })
        fig, ax = plt.subplots(figsize=(8, 5.5))
        sns.barplot(data=df, x="stage", y="time", hue="stage",
                   palette=_PALETTE, legend=False, ax=ax)
        ax.set_xlabel(""); ax.set_ylabel("Mean wall time (s)")
        ax.set_yscale("log")
        speedup = fom_mean / max(rom_mean, 1e-12)
        ax.set_title(f"Wall-time comparison — {label} online speedup: {speedup:.1f}x")
        for bar, v in zip(ax.patches, df["time"]):
            ax.text(bar.get_x() + bar.get_width() / 2, v * 1.05,
                    f"{v:.3g}s", ha="center", fontsize=10)
        out = self._savepath(filename)
        return self._finish(fig, out)

    # =====================================================================
    # Plot 9: Newton convergence
    # =====================================================================
    def plot_newton_convergence(self, fom_residuals: List[float],
                                rom_residuals: List[float],
                                filename: str = "09_newton_convergence.png"
                                ) -> Path:
        df = pd.concat([
            pd.DataFrame({"iteration": np.arange(1, len(fom_residuals) + 1),
                          "residual": fom_residuals, "solver": "FOM Newton"}),
            pd.DataFrame({"iteration": np.arange(1, len(rom_residuals) + 1),
                          "residual": rom_residuals, "solver": "ROM Newton"}),
        ], ignore_index=True)

        fig, ax = plt.subplots(figsize=(8, 5.5))
        sns.lineplot(data=df, x="iteration", y="residual", hue="solver",
                    style="solver", markers=True, dashes=False,
                    markersize=7, ax=ax)
        ax.set_yscale("log")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Residual norm")
        ax.set_title("Newton convergence (representative test case)")
        ax.legend(title=None)
        out = self._savepath(filename)
        return self._finish(fig, out)

    # =====================================================================
    # Plot 10: mode comparison (errors + speedup across all mode counts)
    # =====================================================================
    def plot_mode_comparison(self, Nrs: np.ndarray,
                             sweep_u: np.ndarray, sweep_p: np.ndarray,
                             sweep_h1_u: np.ndarray, sweep_speedups: np.ndarray,
                             filename: str = "10_mode_comparison.png") -> Path:
        fig, axes = plt.subplots(2, 2, figsize=(13, 9))
        pal = sns.color_palette(_PALETTE)

        panels = [
            (axes[0, 0], sweep_u, "Velocity $L^2$ error", "o", pal[0], True),
            (axes[0, 1], sweep_p, "Pressure $L^2$ error", "s", pal[1], True),
            (axes[1, 0], sweep_h1_u, "Velocity $H^1$ error", "^", pal[2], True),
            (axes[1, 1], sweep_speedups, "Online speedup", "D", pal[3], False),
        ]
        for ax, y, title, marker, color, logy in panels:
            df = pd.DataFrame({"r_u": Nrs, "y": y})
            sns.lineplot(data=df, x="r_u", y="y", marker=marker, color=color,
                        markersize=7, linewidth=1.8, ax=ax)
            ax.set_xlabel("Velocity modes $r_u$")
            ax.set_ylabel("Mean relative error" if logy else "Speedup factor")
            ax.set_title(title)
            ax.set_xticks(Nrs)
            if logy:
                ax.set_yscale("log")

        fig.suptitle("ROM accuracy and speedup vs number of modes "
                     "(supremizer-stabilised)", fontsize=14)
        out = self._savepath(filename)
        return self._finish(fig, out)

    # =====================================================================
    # Plot 12: PODNN training curve (Task 2)
    # =====================================================================
    def plot_podnn_training_curve(self, train_hist: np.ndarray,
                                  val_hist: np.ndarray,
                                  filename: str = "12_podnn_training_curve.png"
                                  ) -> Path:
        epochs = np.arange(1, len(train_hist) + 1)
        df = pd.concat([
            pd.DataFrame({"epoch": epochs, "mse": train_hist, "split": "Train"}),
            pd.DataFrame({"epoch": epochs, "mse": val_hist, "split": "Validation"}),
        ], ignore_index=True)

        fig, ax = plt.subplots(figsize=(8, 5.5))
        sns.lineplot(data=df, x="epoch", y="mse", hue="split", style="split",
                    ax=ax)
        ax.set_yscale("log")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("MSE (standardized coefficients)")
        ax.set_title("PODNN training convergence")
        ax.legend(title=None)
        out = self._savepath(filename)
        return self._finish(fig, out)

    # =====================================================================
    # Plot 13: PODNN coefficient parity (Task 2)
    # =====================================================================
    def plot_podnn_coeff_parity(self, c_true: np.ndarray, c_pred: np.ndarray,
                                r_u: int,
                                filename: str = "13_podnn_coeff_parity.png"
                                ) -> Path:
        """Predicted-vs-true scatter for leading velocity/pressure modes.

        ``c_true``/``c_pred`` are ``(r_u+r_p, M)`` reduced coefficients
        (test set); the leading modes of each block carry the most energy
        and are the clearest visual check of regression quality.
        """
        modes = [(0, "Velocity mode 1"), (1, "Velocity mode 2"),
                 (2, "Velocity mode 3"),
                 (r_u, "Pressure mode 1"), (r_u + 1, "Pressure mode 2"),
                 (r_u + 2, "Pressure mode 3")]
        modes = [(i, lbl) for i, lbl in modes if i < c_true.shape[0]]

        fig, axes = plt.subplots(2, 3, figsize=(14, 9))
        for ax, (i, lbl) in zip(axes.ravel(), modes):
            x, y = c_true[i], c_pred[i]
            ax.scatter(x, y, s=60, edgecolor="black", linewidth=0.6,
                      color=sns.color_palette(_PALETTE)[0])
            lo, hi = min(x.min(), y.min()), max(x.max(), y.max())
            pad = 0.05 * (hi - lo or 1.0)
            ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad],
                   ls="--", color="grey", label="$y=x$")
            ax.set_xlabel("True coefficient")
            ax.set_ylabel("Predicted coefficient")
            ax.set_title(lbl)
        for ax in axes.ravel()[len(modes):]:
            ax.set_visible(False)
        fig.suptitle("PODNN coefficient parity (test set)", fontsize=15)
        out = self._savepath(filename)
        return self._finish(fig, out)

    # =====================================================================
    # Plot 17: PODNN error vs training-set size (Task 2)
    # =====================================================================
    def plot_error_vs_trainsize(self, sizes: np.ndarray,
                                mean_u: np.ndarray, mean_p: np.ndarray,
                                filename: str = "17_podnn_error_vs_trainsize.png"
                                ) -> Path:
        df = pd.concat([
            pd.DataFrame({"n_train": sizes, "error": mean_u, "field": "Velocity (mean rel. $L^2$)"}),
            pd.DataFrame({"n_train": sizes, "error": mean_p, "field": "Pressure (mean rel. $L^2$)"}),
        ], ignore_index=True)

        fig, ax = plt.subplots(figsize=(8, 5.5))
        sns.lineplot(data=df, x="n_train", y="error", hue="field", style="field",
                    markers=True, dashes=False, markersize=7, ax=ax)
        ax.set_yscale("log")
        ax.set_xlabel("Number of training snapshots")
        ax.set_ylabel("Mean relative $L^2$ error on test set")
        ax.set_title("PODNN error vs training-set size")
        ax.legend(title=None)
        out = self._savepath(filename)
        return self._finish(fig, out)

    # =====================================================================
    # Plot 11: inf-sup instability diagnostic (no supremizer stabilisation)
    # =====================================================================
    def plot_infsup_instability(self, r_values: np.ndarray,
                                mean_u: np.ndarray, mean_p: np.ndarray,
                                r_primary: int,
                                filename: str = "11_infsup_instability.png"
                                ) -> Path:
        """Show what happens if the velocity reduced space is truncated to
        *primary* POD modes only, with no supremizer enrichment.

        Primary velocity POD modes are linear combinations of FOM snapshots
        that individually satisfy the discrete divergence-free constraint
        ``B_h u_h(mu) = 0``; any such combination is itself divergence-free,
        so ``B_r ~ 0`` to machine precision and the reduced saddle-point
        Jacobian is singular. This is the inf-sup (Babuska-Brezzi) failure
        that supremizer enrichment is designed to fix -- contrast with
        plot 06/10, which use the supremizer-stabilised paired basis and
        converge cleanly.
        """
        df = pd.concat([
            pd.DataFrame({"r": r_values, "error": mean_u, "field": "Velocity (mean rel. $L^2$)"}),
            pd.DataFrame({"r": r_values, "error": mean_p, "field": "Pressure (mean rel. $L^2$)"}),
        ], ignore_index=True)

        fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

        df_u = df[df["field"].str.startswith("Velocity")]
        sns.lineplot(data=df_u, x="r", y="error", marker="o",
                    color=sns.color_palette(_PALETTE)[0], ax=axes[0])
        axes[0].set_yscale("log")
        axes[0].set_xlabel("Velocity modes $r$ (primary only, no supremizer)")
        axes[0].set_ylabel("Mean relative $L^2$ error")
        axes[0].set_title("Velocity error stays at 1.0")
        axes[0].set_xticks(r_values)

        df_p = df[df["field"].str.startswith("Pressure")]
        sns.lineplot(data=df_p, x="r", y="error", marker="s",
                    color=sns.color_palette(_PALETTE)[3], ax=axes[1])
        axes[1].set_yscale("log")
        axes[1].set_xlabel("Velocity modes $r$ (primary only, no supremizer)")
        axes[1].set_ylabel("Mean relative $L^2$ error")
        axes[1].set_title("Pressure error explodes ($\\sim 10^{14}$)")
        axes[1].set_xticks(r_values)

        fig.suptitle(
            f"Effect of supremizer enrichment on inf-sup stability "
            f"(all $r \\leq r_\\mathrm{{primary}}={r_primary}$ shown)",
            fontsize=13,
        )
        fig.text(
            0.5, -0.02,
            "Without supremizer modes, the primary POD velocity space is exactly "
            "divergence-free ($B_r \\approx 0$ to machine precision), decoupling "
            "velocity and pressure in the reduced Newton system: the update to $a$ "
            "never leaves zero while $b$ diverges. Compare to plots 06/10, where "
            "supremizer modes are grown in lockstep with primary modes and the ROM "
            "converges monotonically.",
            ha="center", va="top", fontsize=9.5, wrap=True, color="grey",
        )
        out = self._savepath(filename)
        return self._finish(fig, out)

    # =====================================================================
    # Plot 18: Task 3 -- ROM vs PODNN accuracy comparison
    # =====================================================================
    def plot_comparison_accuracy(self, rom_summary: Dict[str, float],
                                 podnn_summary: Dict[str, float],
                                 floor_summary: Dict[str, float],
                                 pinn_summary: Optional[Dict[str, float]] = None,
                                 filename: str = "18_comparison_accuracy.png"
                                 ) -> Path:
        """Grouped bars: mean rel. L2(u)/L2(p)/H1(u) for ROM vs PODNN (and PINN
        if given), with the POD-truncation floor (best either projection-based
        method's basis could achieve) marked as a reference line."""
        metrics = [("mean_l2_u", "$L^2(u)$"), ("mean_l2_p", "$L^2(p)$"),
                   ("mean_h1_u", "$H^1(u)$")]
        frames = [
            pd.DataFrame({"metric": [lbl for _, lbl in metrics],
                          "error": [rom_summary[k] for k, _ in metrics],
                          "method": "ROM (Galerkin)"}),
            pd.DataFrame({"metric": [lbl for _, lbl in metrics],
                          "error": [podnn_summary[k] for k, _ in metrics],
                          "method": "PODNN"}),
        ]
        if pinn_summary is not None:
            frames.append(pd.DataFrame({"metric": [lbl for _, lbl in metrics],
                          "error": [pinn_summary[k] for k, _ in metrics],
                          "method": "PINN"}))
        df = pd.concat(frames, ignore_index=True)

        fig, ax = plt.subplots(figsize=(8.5, 5.5))
        sns.barplot(data=df, x="metric", y="error", hue="method",
                   palette=_PALETTE, ax=ax)
        ax.set_yscale("log")
        ax.set_xlabel(""); ax.set_ylabel("Mean relative error on test set")
        _title = ("Task 3/4 -- ROM vs PODNN vs PINN accuracy (rel. to FOM)"
                  if pinn_summary is not None
                  else "Task 3 -- ROM vs PODNN accuracy (relative to FOM)")
        ax.set_title(_title)
        floor_l2u = floor_summary.get("mean_l2_u")
        if floor_l2u is not None:
            ax.axhline(floor_l2u, color="grey", linestyle="--", linewidth=1.3,
                       label="POD-truncation floor ($L^2(u)$)")
        ax.legend(title=None)
        out = self._savepath(filename)
        return self._finish(fig, out)

    # =====================================================================
    # Plot 19: Task 3 -- computational cost comparison
    # =====================================================================
    def plot_comparison_cost(self, fom_mean: float, rom_mean: float,
                             podnn_mean: float, shared_offline: float,
                             rom_increment: float, podnn_increment: float,
                             pinn_mean: Optional[float] = None,
                             pinn_train: Optional[float] = None,
                             filename: str = "19_comparison_cost.png") -> Path:
        """Two panels: (left) per-query online wall time, (right) offline
        cost decomposed into the shared snapshot+POD cost and each method's
        own increment, so the shared part is never double counted. If PINN
        args are given it is added -- with its offline shown as a *standalone*
        training bar (PINN needs no shared snapshots/POD)."""
        fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

        online_stage = ["FOM", "ROM", "PODNN"]
        online_time = [fom_mean, rom_mean, podnn_mean]
        if pinn_mean is not None:
            online_stage.append("PINN"); online_time.append(pinn_mean)
        df_online = pd.DataFrame({"stage": online_stage, "time": online_time})
        sns.barplot(data=df_online, x="stage", y="time", hue="stage",
                   palette=_PALETTE, legend=False, ax=axes[0])
        axes[0].set_yscale("log")
        axes[0].set_xlabel(""); axes[0].set_ylabel("Mean online wall time (s)")
        axes[0].set_title("Online cost per query")
        for bar, v in zip(axes[0].patches, df_online["time"]):
            axes[0].text(bar.get_x() + bar.get_width() / 2, v * 1.05,
                        f"{v:.3g}s", ha="center", fontsize=9.5)

        off_frames = [
            pd.DataFrame({"stage": ["Shared\n(snapshots+POD)"], "time": [shared_offline],
                          "part": "Shared"}),
            pd.DataFrame({"stage": ["ROM\nincrement"], "time": [rom_increment],
                          "part": "Method-specific"}),
            pd.DataFrame({"stage": ["PODNN\nincrement"], "time": [podnn_increment],
                          "part": "Method-specific"}),
        ]
        if pinn_train is not None:
            off_frames.append(pd.DataFrame({"stage": ["PINN\n(standalone)"],
                          "time": [pinn_train], "part": "No shared cost"}))
        df_offline = pd.concat(off_frames, ignore_index=True)
        sns.barplot(data=df_offline, x="stage", y="time", hue="part",
                   palette=_PALETTE, legend=(pinn_train is not None), ax=axes[1])
        axes[1].set_yscale("log")
        axes[1].set_xlabel(""); axes[1].set_ylabel("Wall time (s)")
        axes[1].set_title("Offline cost breakdown (shared cost counted once)")
        for bar, v in zip(axes[1].patches, df_offline["time"]):
            axes[1].text(bar.get_x() + bar.get_width() / 2, v * 1.05,
                        f"{v:.3g}s", ha="center", fontsize=9.5)

        fig.suptitle("Task 3/4 -- computational cost vs FOM", fontsize=14)
        out = self._savepath(filename)
        return self._finish(fig, out)

    # =====================================================================
    # Plot 20: Task 3 -- accuracy/speed trade-off + per-point error
    # =====================================================================
    def plot_comparison_tradeoff(self, rom_mean_time: float, rom_mean_err: float,
                                 podnn_mean_time: float, podnn_mean_err: float,
                                 test_params: np.ndarray,
                                 rom_err_per_point: np.ndarray,
                                 podnn_err_per_point: np.ndarray,
                                 pinn_mean_time: Optional[float] = None,
                                 pinn_mean_err: Optional[float] = None,
                                 pinn_err_per_point: Optional[np.ndarray] = None,
                                 filename: str = "20_comparison_tradeoff.png"
                                 ) -> Path:
        """(left) accuracy-vs-online-cost scatter (lower-left is better);
        (right) paired per-test-point rel. L2(u) error. PINN added if given."""
        fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
        pal = sns.color_palette(_PALETTE)

        axes[0].scatter([rom_mean_time], [rom_mean_err], s=180, marker="o",
                        color=pal[0], edgecolor="black", linewidth=0.8, label="ROM")
        axes[0].scatter([podnn_mean_time], [podnn_mean_err], s=180, marker="^",
                        color=pal[1], edgecolor="black", linewidth=0.8, label="PODNN")
        if pinn_mean_time is not None and pinn_mean_err is not None:
            axes[0].scatter([pinn_mean_time], [pinn_mean_err], s=180, marker="s",
                            color=pal[2], edgecolor="black", linewidth=0.8, label="PINN")
        axes[0].set_xscale("log"); axes[0].set_yscale("log")
        axes[0].set_xlabel("Mean online wall time (s)")
        axes[0].set_ylabel("Mean relative $L^2(u)$ error")
        axes[0].set_title("Accuracy vs. speed (lower-left is better)")
        axes[0].legend(title=None)

        idx = np.arange(len(test_params))
        frames = [
            pd.DataFrame({"test_point": idx, "error": rom_err_per_point, "method": "ROM"}),
            pd.DataFrame({"test_point": idx, "error": podnn_err_per_point, "method": "PODNN"}),
        ]
        if pinn_err_per_point is not None:
            frames.append(pd.DataFrame({"test_point": idx,
                          "error": pinn_err_per_point, "method": "PINN"}))
        df = pd.concat(frames, ignore_index=True)
        sns.lineplot(data=df, x="test_point", y="error", hue="method", style="method",
                    markers=True, dashes=False, markersize=3, linewidth=0.8, ax=axes[1])
        axes[1].set_yscale("log")
        axes[1].set_xlabel("Test point index (natural order)")
        axes[1].set_ylabel("Relative $L^2(u)$ error")
        _mlist = "ROM vs PODNN" + (" vs PINN" if pinn_err_per_point is not None else "")
        axes[1].set_title(f"Per-test-point error, {_mlist}")
        axes[1].legend(title=None)

        fig.suptitle("Task 3/4 -- accuracy/speed trade-off", fontsize=14)
        out = self._savepath(filename)
        return self._finish(fig, out)

    # =====================================================================
    # Plot 21: PINN training convergence (Task 4)
    # =====================================================================
    def plot_pinn_training_curve(self, hist_total: np.ndarray,
                                 hist_b: np.ndarray, hist_p: np.ndarray,
                                 filename: str = "21_pinn_training_curve.png"
                                 ) -> Path:
        """Three-curve training history: total, boundary MSE and physics MSE."""
        epochs = np.arange(1, len(hist_total) + 1)
        df = pd.concat([
            pd.DataFrame({"epoch": epochs, "loss": hist_total, "term": "Total"}),
            pd.DataFrame({"epoch": epochs, "loss": hist_b,     "term": "Boundary (MSE_b)"}),
            pd.DataFrame({"epoch": epochs, "loss": hist_p,     "term": "Physics (MSE_p)"}),
        ], ignore_index=True)
        df["loss"] = df["loss"].clip(lower=1e-30)

        fig, ax = plt.subplots(figsize=(9, 5.5))
        sns.lineplot(data=df, x="epoch", y="loss", hue="term", style="term", ax=ax)
        ax.set_yscale("log")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss (MSE)")
        ax.set_title("PINN training convergence")
        ax.legend(title=None)
        out = self._savepath(filename)
        return self._finish(fig, out)

    # =====================================================================
    # Plot 22: PINN vs FOM field comparison (Task 4)
    # =====================================================================
    def plot_pinn_vs_fom(self, test_params: np.ndarray,
                         fom_u: np.ndarray,
                         pinn_u_mag: np.ndarray,
                         filename: str = "22_pinn_vs_fom.png",
                         label: str = "PINN") -> Path:
        """Side-by-side FOM and PINN velocity-magnitude contours + error.

        ``pinn_u_mag`` is (N_vertices, M_test) magnitude interpolated to the
        mesh vertices.
        """
        order = np.argsort(test_params[:, 0])
        picks = [int(order[0]), int(order[len(order) // 2]), int(order[-1])]
        fig, axes = plt.subplots(3, 3, figsize=(14, 12))
        for row, idx in enumerate(picks):
            umag_f = self.velocity_magnitude(fom_u[:, idx])
            umag_r = pinn_u_mag[:, idx]
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
            axes[row, 1].set_title(f"{label} $|u|$")
            cs2 = axes[row, 2].tricontourf(self.triang, err, levels=20,
                                            cmap="RdBu_r", vmin=-emax, vmax=emax)
            plt.colorbar(cs2, ax=axes[row, 2])
            axes[row, 2].set_title("Pointwise $|u|$ error")
            for ax in axes[row]:
                ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_aspect("equal")
        fig.suptitle(f"FOM vs {label} velocity comparison", fontsize=15)
        out = self._savepath(filename)
        return self._finish(fig, out)

    # =====================================================================
    # Plot 23: PINN error in parameter space (Task 4)
    # =====================================================================
    def plot_pinn_error_parameter_space(self, test_params: np.ndarray,
                                        rel_l2_u: np.ndarray,
                                        filename: str = "23_pinn_error_parameter_space.png"
                                        ) -> Path:
        df = pd.DataFrame({
            "mu0": test_params[:, 0], "mu1": test_params[:, 1],
            "error": rel_l2_u,
        })
        fig, ax = plt.subplots(figsize=(8, 5.5))
        sc = ax.scatter(df["mu0"], df["mu1"], c=df["error"], cmap="viridis",
                        s=130, edgecolor="black", linewidth=0.6)
        plt.colorbar(sc, ax=ax, label="Relative $L^2$ velocity error")
        ax.set_xlabel(r"$\mu_0$ (viscosity)")
        ax.set_ylabel(r"$\mu_1$ (forcing freq.)")
        ax.set_title("PINN test-set error in parameter space")
        out = self._savepath(filename)
        return self._finish(fig, out)

    # =====================================================================
    # Plot 24: 4-method accuracy + cost summary (Task 4)
    # =====================================================================
    def plot_pinn_summary_comparison(self,
                                     rom_summary: Dict[str, float],
                                     podnn_summary: Dict[str, float],
                                     pinn_summary: Dict[str, float],
                                     fom_mean: float, rom_mean: float,
                                     podnn_mean: float, pinn_mean: float,
                                     filename: str = "24_pinn_summary_comparison.png"
                                     ) -> Path:
        """Two panels: (left) mean rel L2(u) accuracy for ROM/PODNN/PINN;
        (right) online wall-time for FOM/ROM/PODNN/PINN."""
        fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

        methods = ["ROM", "PODNN", "PINN"]
        errors = [rom_summary["mean_l2_u"], podnn_summary["mean_l2_u"],
                  pinn_summary["mean_l2_u"]]
        df_acc = pd.DataFrame({"Method": methods, "error": errors})
        sns.barplot(data=df_acc, x="Method", y="error",
                    hue="Method", palette=_PALETTE, legend=False, ax=axes[0])
        axes[0].set_yscale("log")
        axes[0].set_xlabel("")
        axes[0].set_ylabel("Mean relative $L^2(u)$ error")
        axes[0].set_title("Accuracy comparison (all methods)")
        for bar, v in zip(axes[0].patches, errors):
            axes[0].text(bar.get_x() + bar.get_width() / 2, v * 1.3,
                         f"{v:.2e}", ha="center", fontsize=9)

        methods_t = ["FOM", "ROM", "PODNN", "PINN"]
        times = [fom_mean, rom_mean, podnn_mean, pinn_mean]
        df_time = pd.DataFrame({"Method": methods_t, "Time (s)": times})
        sns.barplot(data=df_time, x="Method", y="Time (s)",
                    hue="Method", palette=_PALETTE, legend=False, ax=axes[1])
        axes[1].set_yscale("log")
        axes[1].set_xlabel("")
        axes[1].set_ylabel("Mean online wall time (s)")
        axes[1].set_title("Online cost per query")
        for bar, v in zip(axes[1].patches, times):
            axes[1].text(bar.get_x() + bar.get_width() / 2, v * 1.3,
                         f"{v:.3g}s", ha="center", fontsize=9)

        fig.suptitle("Task 4 -- PINN vs ROM vs PODNN vs FOM", fontsize=14)
        out = self._savepath(filename)
        return self._finish(fig, out)

    # =====================================================================
    # Plot 25: 3D error surfaces over the (mu0, mu1) test-parameter plane
    # =====================================================================
    def plot_error_surface_3d(self, test_params: np.ndarray,
                              rom_error: np.ndarray, podnn_error: np.ndarray,
                              metric_name: str = "rel $L^2(u)$ error",
                              filename: str = "25_error_surface_3d.png") -> Path:
        """Side-by-side 3D error surfaces (Delaunay-triangulated scatter) for
        ROM and PODNN over the held-out test set's (mu0, mu1) plane.

        Independent z-scale/colorbar per panel -- ROM and PODNN errors differ
        by roughly an order of magnitude on this basis, so a shared z-axis
        would flatten the ROM panel into visual noise; the accuracy bar chart
        (18_comparison_accuracy.png) is where the cross-method magnitude
        comparison lives, this plot is for spatial error pattern only.
        """
        mu0, mu1 = test_params[:, 0], test_params[:, 1]
        fig = plt.figure(figsize=(14, 6))
        for i, (label, err) in enumerate([("ROM", rom_error), ("PODNN", podnn_error)]):
            ax = fig.add_subplot(1, 2, i + 1, projection="3d")
            surf = ax.plot_trisurf(mu0, mu1, err, cmap="viridis",
                                   linewidth=0.1, antialiased=True)
            fig.colorbar(surf, ax=ax, shrink=0.6, pad=0.1)
            ax.set_xlabel("$\\mu_0$")
            ax.set_ylabel("$\\mu_1$")
            ax.set_zlabel(metric_name)
            ax.set_title(f"{label}: {metric_name}\n"
                        f"(mean {err.mean():.2e}, max {err.max():.2e})")
        fig.suptitle("Error surfaces over the test parameter space", fontsize=14)
        # Not routed through _finish(): sns.despine() assumes 2D-style spines
        # and errors on Axes3D objects.
        fig.tight_layout()
        out = self._savepath(filename)
        fig.savefig(out, dpi=200, bbox_inches="tight")
        plt.close(fig)
        return out
