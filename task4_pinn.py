#!/usr/bin/env python3
"""
Task 4 driver: Physics-Informed Neural Network (PINN) for steady Navier-Stokes.

The PINN learns  (x, mu) -> (u1, u2, p)  directly from the PDE residual.

== DOF reconstruction ==
FEniCS P2 VectorElement stores DOFs in interleaved order (u1/u2 alternating).
Confirmed by V.sub(0).dofmap().dofs() returning only even global indices.
We use parent_dofs_0/parent_dofs_1 to scatter predictions into correct slots.

== Training strategy ==
Two-phase curriculum with properly scaled losses:
  Phase 1 (warmup):  lambda_p=0, lambda_b=1   -> drive MSE_b to ~1e-6
  Phase 2 (physics): lambda_p=1, lambda_b=100 -> keep boundary enforced while
           reducing the normalised physics residual.

The physics residual MSE_p is normalised by the mean squared forcing inside
PINNModel.train_step (see src/pinn.py), so it stays O(1) rather than O(2000).
lambda_b=100 then ensures the boundary term stays 100x heavier than physics.

Run with:
    python task4_pinn.py
"""

from __future__ import annotations

import csv
import time

import numpy as np

from src import Config, NavierStokesProblem, Visualizer, PINNModel
from src.pinn import sample_collocation_points
from src.analysis import ErrorAnalyzer

# -----------------------------------------------------------------------
# Hyper-parameters (Maxed out for brute-force test)
# -----------------------------------------------------------------------
HIDDEN_SIZES = [128, 128, 128, 128]

WARMUP_EPOCHS  = 4_000     # Phase 1: boundary only
PHYSICS_EPOCHS = 16_000    # Phase 2: full PINN
TOTAL_EPOCHS   = WARMUP_EPOCHS + PHYSICS_EPOCHS

LR_WARMUP  = 5e-4
LR_PHYSICS = 1e-3
LAMBDA_P   = 1.0           # physics weight (MSE_p already normalised to O(1))
LAMBDA_B   = 100.0         # boundary weight >> lambda_p to protect no-slip BC
H_FD       = 1e-3

N_INTERIOR = 8_000
N_BOUNDARY = 2_000
N_PIN      = 800
LOG_EVERY  = 1_000

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
    print("Task 4 — Physics-Informed Neural Network (PINN)")
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
    print(f"Training: {WARMUP_EPOCHS} warmup + {PHYSICS_EPOCHS} physics epochs")
    print(f"  lambda_b={LAMBDA_B}, lambda_p={LAMBDA_P} (MSE_p normalised to O(1))")
    print(f"  N_int={N_INTERIOR}, N_bnd={N_BOUNDARY}, N_pin={N_PIN}")

    # ---------------------------------------------------------------
    # Train: Phase 1 (boundary warmup)
    # ---------------------------------------------------------------
    model = PINNModel(mu0_range=config.mu0_range, mu1_range=config.mu1_range,
                      hidden_sizes=HIDDEN_SIZES, seed=SEED)
    rng_train = np.random.default_rng(SEED)
    hist_total = np.zeros(TOTAL_EPOCHS)
    hist_b     = np.zeros(TOTAL_EPOCHS)
    hist_p     = np.zeros(TOTAL_EPOCHS)

    print("\n[Phase 1] Boundary warmup")
    t0 = time.time()
    for ep in range(WARMUP_EPOCHS):
        coll = sample_collocation_points(N_INTERIOR, N_BOUNDARY, N_PIN,
                                         config.mu0_range, config.mu1_range, rng_train)
        total, lb, lp = model.train_step(coll, LR_WARMUP, lambda_p=0.0, h=H_FD,
                                          lambda_b=1.0)
        hist_total[ep] = lb; hist_b[ep] = lb; hist_p[ep] = 0.0
        if ep % LOG_EVERY == 0:
            print(f"  epoch {ep:5d}  MSE_b {lb:.3e}")
    t_warmup = time.time() - t0
    print(f"Phase 1 done in {t_warmup:.1f}s  (final MSE_b={hist_b[WARMUP_EPOCHS-1]:.3e})")

    # ---------------------------------------------------------------
    # Train: Phase 2 (full PINN, lambda_b=100 >> lambda_p=1)
    # ---------------------------------------------------------------
    print(f"\n[Phase 2] Full PINN  (lambda_b={LAMBDA_B}, lambda_p={LAMBDA_P})")
    t1 = time.time()
    for ep in range(PHYSICS_EPOCHS):
        coll = sample_collocation_points(N_INTERIOR, N_BOUNDARY, N_PIN,
                                         config.mu0_range, config.mu1_range, rng_train)
        total, lb, lp = model.train_step(coll, LR_PHYSICS, lambda_p=LAMBDA_P,
                                          h=H_FD, lambda_b=LAMBDA_B)
        idx = WARMUP_EPOCHS + ep
        hist_total[idx] = total; hist_b[idx] = lb; hist_p[idx] = lp
        if ep % LOG_EVERY == 0:
            print(f"  epoch {ep:5d}  total {total:.3e}  MSE_b {lb:.3e}  MSE_p {lp:.3e}")
    t_physics = time.time() - t1
    pinn_train_time = t_warmup + t_physics
    print(f"Phase 2 done in {t_physics:.1f}s")
    print(f"Total training: {pinn_train_time:.1f}s  "
          f"(final total={hist_total[-1]:.3e}, "
          f"MSE_b={hist_b[-1]:.3e}, MSE_p(norm)={hist_p[-1]:.3e})")

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
    print(f"Training:                     {WARMUP_EPOCHS} warmup + {PHYSICS_EPOCHS} physics")
    print(f"lambda_b / lambda_p:          {LAMBDA_B} / {LAMBDA_P}  (MSE_p force-normalised)")
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
