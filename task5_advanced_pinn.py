#!/usr/bin/env python3
"""
Task 5 driver: Advanced Physics-Informed Neural Network (PINN) for steady Navier-Stokes.

This driver uses **Dynamic Weight Balancing** (Learning Rate Annealing) to automatically
scale the boundary loss weight during training. This mathematically prevents the PINN
from falling into the "trivial solution trap" (u=0) that we saw in Task 4.

== DOF reconstruction ==
FEniCS P2 VectorElement stores DOFs in interleaved order (u1/u2 alternating).
Confirmed by V.sub(0).dofmap().dofs() returning only even global indices.
We use parent_dofs_0/parent_dofs_1 to scatter predictions into correct slots.

Run with:
    python task5_advanced_pinn.py
"""

from __future__ import annotations

import csv
import time

import numpy as np

from src import Config, NavierStokesProblem, Visualizer, PINNModel
from src.pinn import sample_collocation_points
from src.analysis import ErrorAnalyzer

# -----------------------------------------------------------------------
# Hyper-parameters
# -----------------------------------------------------------------------
HIDDEN_SIZES = [64, 64, 64]

TOTAL_EPOCHS   = 10_000

LR         = 1e-3
LAMBDA_P   = 1.0           # physics weight (MSE_p already normalised to O(1))
LAMBDA_B_INIT = 1.0        # boundary weight initialized to 1, updated dynamically
ALPHA      = 0.9           # exponential moving average decay for lambda_b
H_FD       = 1e-3

N_INTERIOR = 2_000
N_BOUNDARY = 500
N_PIN      = 200
LOG_EVERY  = 500

SEED = 42


def _get_parent_dof_indices(problem: NavierStokesProblem):
    V = problem.V
    V0 = V.sub(0).collapse()
    parent_dofs_0 = np.array(V.sub(0).dofmap().dofs())
    parent_dofs_1 = np.array(V.sub(1).dofmap().dofs())
    dof_coords_u  = V0.tabulate_dof_coordinates()
    dof_coords_p  = problem.Q.tabulate_dof_coordinates()
    return parent_dofs_0, parent_dofs_1, dof_coords_u, dof_coords_p


def _pinn_dof_vectors(model, mu, parent_dofs_0, parent_dofs_1,
                      dof_coords_u, dof_coords_p, N_u, N_p):
    n_sc = dof_coords_u.shape[0]
    out_u = model.predict(dof_coords_u, np.tile(mu, (n_sc, 1)))
    out_p = model.predict(dof_coords_p, np.tile(mu, (N_p, 1)))
    U = np.zeros(N_u)
    U[parent_dofs_0] = out_u[:, 0]
    U[parent_dofs_1] = out_u[:, 1]
    return U, out_p[:, 2]


def main() -> None:
    print("=" * 64)
    print("Task 5 — Advanced PINN with Dynamic Weight Balancing")
    print("=" * 64)

    config = Config()
    problem = NavierStokesProblem(config)
    N_u, N_p = problem.N_u, problem.N_p
    print(f"Mesh: {config.mesh_n}x{config.mesh_n}  |  N_u={N_u}, N_p={N_p}")

    fom_test = np.load(config.data_dir / "fom_solutions_test.npz")
    u_fom_test  = fom_test["u_fom_test"]
    p_fom_test  = fom_test["p_fom_test"]
    test_params = fom_test["test_params"]
    M_test      = test_params.shape[0]

    timing_data    = np.load(config.data_dir / "timing_data.npz")
    fom_times_test = timing_data["fom_times_test"]
    fom_mean       = float(fom_times_test.mean())

    try:
        rom_err      = np.load(config.data_dir / "test_errors.npz")
        podnn_err    = np.load(config.data_dir / "podnn_test_errors.npz")
        rom_timing   = np.load(config.data_dir / "timing_data.npz")
        podnn_timing = np.load(config.data_dir / "podnn_timing.npz")
        has_task13   = True
    except FileNotFoundError:
        print("  [warn] Task 1/2/3 artefacts not found -- skipping plot 24.")
        has_task13 = False

    parent_dofs_0, parent_dofs_1, dof_coords_u, dof_coords_p = \
        _get_parent_dof_indices(problem)
    vert_coords = problem.mesh.coordinates().copy()
    N_vert = vert_coords.shape[0]

    assert np.all(parent_dofs_0 % 2 == 0), "Expected u0 DOFs at even indices"
    assert np.all(parent_dofs_1 % 2 == 1), "Expected u1 DOFs at odd indices"
    print(f"DOF layout: interleaved (u1=even, u2=odd) ✓  n_scalar={dof_coords_u.shape[0]}")
    print(f"Test parameters: {M_test}   Mesh vertices: {N_vert}")
    print(f"PINN: {[4] + HIDDEN_SIZES + [3]}")
    print(f"Training: {TOTAL_EPOCHS} dynamic epochs")
    print(f"  lambda_b updated dynamically, lambda_p={LAMBDA_P} (MSE_p normalised)")
    print(f"  N_int={N_INTERIOR}, N_bnd={N_BOUNDARY}, N_pin={N_PIN}")

    # ---------------------------------------------------------------
    # Train: Dynamic Weight Balancing
    # ---------------------------------------------------------------
    print(f"\n[Training] Dynamic PINN")
    t0 = time.time()
    model = PINNModel(mu0_range=config.mu0_range, mu1_range=config.mu1_range,
                      hidden_sizes=HIDDEN_SIZES, seed=SEED)
    
    hist_total, hist_b, hist_p, hist_lambda_b = model.train_dynamic(
        n_interior=N_INTERIOR, n_boundary=N_BOUNDARY, n_pin=N_PIN,
        epochs=TOTAL_EPOCHS, lr=LR, lambda_p=LAMBDA_P,
        lambda_b_init=LAMBDA_B_INIT, alpha=ALPHA, h=H_FD, seed=SEED,
        verbose=True, log_every=LOG_EVERY
    )
    
    pinn_train_time = time.time() - t0
    print(f"Training done in {pinn_train_time:.1f}s  "
          f"(final total={hist_total[-1]:.3e}, "
          f"MSE_b={hist_b[-1]:.3e}, MSE_p(norm)={hist_p[-1]:.3e}, lambda_b={hist_lambda_b[-1]:.3f})")

    model.save(config.data_dir / "pinn_model.npz")
    print(f"Model saved: {config.data_dir / 'pinn_model.npz'}")

    # ---------------------------------------------------------------
    # Evaluate on 15 test parameters
    # ---------------------------------------------------------------
    print("\nEvaluating on test set...")
    err_analyzer     = ErrorAnalyzer(problem)
    pinn_u_mag       = np.zeros((N_vert, M_test))
    U_pinn           = np.zeros((N_u, M_test))
    P_pinn           = np.zeros((N_p, M_test))
    pinn_times_test  = np.zeros(M_test)

    for i in range(M_test):
        mu = test_params[i]
        t_start = time.time()
        U_pinn[:, i], P_pinn[:, i] = _pinn_dof_vectors(
            model, mu, parent_dofs_0, parent_dofs_1,
            dof_coords_u, dof_coords_p, N_u, N_p)
        out_v = model.predict(vert_coords, np.tile(mu, (N_vert, 1)))
        pinn_u_mag[:, i] = np.sqrt(out_v[:, 0]**2 + out_v[:, 1]**2)
        pinn_times_test[i] = time.time() - t_start
        print(f"  test {i+1:2d}/{M_test}: mu=({mu[0]:.2f},{mu[1]:.2f})  "
              f"t={pinn_times_test[i]:.3f}s")

    report            = err_analyzer.batch_report(u_fom_test, p_fom_test, U_pinn, P_pinn)
    pinn_summary      = report.summary()
    pinn_predict_mean = float(pinn_times_test.mean())
    speedup           = fom_mean / max(pinn_predict_mean, 1e-12)

    print("\nPINN errors on test set:")
    print(f"  mean rel L2(u) = {pinn_summary['mean_l2_u']:.3e}   "
          f"max = {pinn_summary['max_l2_u']:.3e}")
    print(f"  mean rel L2(p) = {pinn_summary['mean_l2_p']:.3e}   "
          f"max = {pinn_summary['max_l2_p']:.3e}")
    print(f"  mean rel H1(u) = {pinn_summary['mean_h1_u']:.3e}   "
          f"max = {pinn_summary['max_h1_u']:.3e}")
    print(f"  FOM={fom_mean:.4f}s  PINN={pinn_predict_mean:.4f}s  "
          f"speedup={speedup:.1f}x")

    # ---------------------------------------------------------------
    # Persist artefacts
    # ---------------------------------------------------------------
    np.savez_compressed(config.data_dir / "pinn_training_history.npz",
                        hist_total=hist_total, hist_b=hist_b, hist_p=hist_p)
    np.savez_compressed(config.data_dir / "pinn_test_errors.npz",
                        rel_l2_u=report.rel_l2_u, rel_l2_p=report.rel_l2_p,
                        rel_h1_u=report.rel_h1_u, test_params=test_params)
    np.savez_compressed(config.data_dir / "pinn_timing.npz",
                        train_time=np.array(pinn_train_time),
                        predict_mean=np.array(pinn_predict_mean),
                        predict_times=pinn_times_test,
                        fom_mean=np.array(fom_mean), speedup=np.array(speedup))
    np.savez_compressed(config.data_dir / "pinn_vertex_umag.npz", pinn_u_mag=pinn_u_mag)

    table_path = config.data_dir / "pinn_comparison_table.csv"
    with open(table_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["idx", "mu0", "mu1", "fom_time_s",
                                           "pinn_time_s", "pinn_rel_l2_u",
                                           "pinn_rel_l2_p", "pinn_rel_h1_u"])
        w.writeheader()
        for i in range(M_test):
            w.writerow({"idx": i, "mu0": test_params[i,0], "mu1": test_params[i,1],
                         "fom_time_s": fom_times_test[i], "pinn_time_s": pinn_times_test[i],
                         "pinn_rel_l2_u": report.rel_l2_u[i],
                         "pinn_rel_l2_p": report.rel_l2_p[i],
                         "pinn_rel_h1_u": report.rel_h1_u[i]})
    print(f"CSV: {table_path}")

    # ---------------------------------------------------------------
    # Plots 21-24
    # ---------------------------------------------------------------
    print("\nGenerating plots...")
    vis = Visualizer(problem)
    vis.plot_pinn_training_curve(hist_total, hist_b, hist_p)
    vis.plot_pinn_vs_fom(test_params, u_fom_test, pinn_u_mag)
    vis.plot_pinn_error_parameter_space(test_params, report.rel_l2_u)
    n_plots = 3
    if has_task13:
        def _s(e):
            return {"mean_l2_u": float(e["rel_l2_u"].mean()),
                    "max_l2_u": float(e["rel_l2_u"].max()),
                    "mean_l2_p": float(e["rel_l2_p"].mean()),
                    "max_l2_p": float(e["rel_l2_p"].max()),
                    "mean_h1_u": float(e["rel_h1_u"].mean()),
                    "max_h1_u": float(e["rel_h1_u"].max())}
        vis.plot_pinn_summary_comparison(
            rom_summary=_s(rom_err), podnn_summary=_s(podnn_err),
            pinn_summary=pinn_summary, fom_mean=fom_mean,
            rom_mean=float(rom_timing["rom_times_test"].mean()),
            podnn_mean=float(podnn_timing["predict_mean"]),
            pinn_mean=pinn_predict_mean)
        n_plots = 4
    print(f"Saved {n_plots} plots to: {config.plots_dir}")

    # ---------------------------------------------------------------
    # Summary
    # ---------------------------------------------------------------
    print("\n" + "=" * 64)
    print("RESULT SUMMARY")
    print("=" * 64)
    print(f"PINN:                         {[4] + HIDDEN_SIZES + [3]}")
    print(f"Training:                     {TOTAL_EPOCHS} dynamic epochs")
    print(f"lambda_b / lambda_p:          Dynamic / {LAMBDA_P}  (MSE_p force-normalised)")
    print(f"DOF layout:                   interleaved ✓")
    print(f"Mean rel. L2(u):              {pinn_summary['mean_l2_u']:.3e}")
    print(f"Max  rel. L2(u):              {pinn_summary['max_l2_u']:.3e}")
    print(f"Mean rel. L2(p):              {pinn_summary['mean_l2_p']:.3e}")
    print(f"Mean rel. H1(u):              {pinn_summary['mean_h1_u']:.3e}")
    print(f"Mean FOM time:                {fom_mean:.4f} s")
    print(f"Mean PINN predict time:       {pinn_predict_mean:.4f} s  ({speedup:.1f}x speedup)")
    print(f"PINN training time:           {pinn_train_time:.2f} s")
    print(f"\nArtefacts: {config.data_dir}")
    print(f"Plots:     {config.plots_dir}")
    print("=" * 64)


if __name__ == "__main__":
    main()
