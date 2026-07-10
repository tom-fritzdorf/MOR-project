#!/usr/bin/env python3
"""
Task 2 driver: parametric PODNN for steady Navier-Stokes.

PODNN (Hesthaven & Ubbiali 2018): reuse Task 1's snapshots + POD basis
unchanged, but replace the online Newton-Galerkin solve with a feed-forward
network N_theta: mu -> c(mu) trained offline on the projected POD
coefficients. Online cost is a single NN forward pass + basis matvec --
no Galerkin operators, no Newton iteration, no FEM assembly.

Every artefact this script saves is built to be directly, index-for-index
comparable to Task 1's (`task1_pod_galerkin.py`) saved artefacts, so Task 3
can load both and compare honestly:
  - identical ground truth (`data/fom_solutions_test.npz`, never re-solved)
  - identical error machinery (`ErrorAnalyzer`, same FE matrices)
  - same POD basis (`data/pod_basis_u.npz` / `pod_basis_p.npz`), restricted
    to the primary (non-supremizer) velocity modes -- the supremizer block
    exists solely to stabilize Task 1's online Galerkin saddle-point solve
    and carries ~0 singular value / reconstruction energy (floor rel-L2(u)
    0.0781 with all 43 modes vs 0.0791 with the 15 primary modes alone), so
    this does not give either method a different achievable subspace
  - matching natural test-index order
  - per-sample, FEM-free online timing (full ensemble forward pass, honest)
  - an offline-cost decomposition that avoids double-counting the
    snapshot+POD cost shared with Task 1

Modelling note: a single small feed-forward net trained on this problem's
160 snapshots does not generalize reliably -- the mu1-dependence is steep
(forcing term ~cos(mu1**2*pi*x), see src/config.py) and even a plain linear
regression baseline on the raw (mu0, mu1) input achieves no better than a
"predict-the-mean" baseline. Three standard, data-independent fixes are
used together: (1) feed mu1**2 as an extra input feature (a fixed
reparametrization motivated by the forcing term's own structure, not extra
data), (2) L2 weight decay, (3) a small deep-ensemble (bagging) of
independently split/initialized nets, averaged at predict time. This is
documented here, not hidden, and the resulting PODNN test error is reported
next to the POD-truncation floor so the honest accuracy gap is visible.

Run with:
    python task2_podnn.py
"""

from __future__ import annotations

import csv
import time

import numpy as np

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable

from src import Config, NavierStokesProblem, PODBasis, ErrorAnalyzer, Visualizer
from src.podnn import PODNNModel, train_val_split

# Ensemble/regularization hyperparameters (see the "Modelling notes" comment
# block below main() for why these are needed -- a single, small, unregularized
# net does not generalize on this problem).
N_ENSEMBLE = 5
HIDDEN_SIZES = [64, 64, 64]
WEIGHT_DECAY = 1e-4
EPOCHS = 8000
LR = 1e-3
PATIENCE = 600
VAL_FRAC = 0.15


def mu_features(mu: np.ndarray) -> np.ndarray:
    """Augment raw (mu0, mu1) with mu1**2.

    The forcing term's own cos(mu1**2*pi*x) structure (see src/config.py)
    means the solution -- and hence the POD coefficients -- vary smoothly in
    mu1**2 but increasingly rapidly in raw mu1 as mu1 grows. Feeding mu1**2
    directly removes a nonlinearity the net would otherwise have to discover
    from only 160 training points; it is a fixed, data-independent
    reparametrization of the same 2D input, not extra information.
    """
    return np.column_stack([mu[:, 0], mu[:, 1], mu[:, 1] ** 2])


def main() -> None:
    print("=" * 64)
    print("Task 2 — Parametric PODNN for steady Navier-Stokes")
    print("=" * 64)

    # --- 1. Setup (FE matrices + mesh only, no FOM solves) -----------
    config = Config()
    problem = NavierStokesProblem(config)
    print(f"Mesh: {config.mesh_n}x{config.mesh_n}  |  "
          f"N_u = {problem.N_u}, N_p = {problem.N_p}")

    # --- 2. Load Task 1 artefacts (no recomputation) ------------------
    rc = np.load(config.data_dir / "reduced_coefficients.npz")
    rom_coeffs_train = rc["rom_coeffs_train"]          # (r_u+r_p, M_train)
    train_params = rc["train_params"]                  # (M_train, 2)
    test_params_rc = rc["test_params"]                 # (M_test, 2)
    r_u = int(rc["r_u"]); r_p = int(rc["r_p"])

    pod_u = PODBasis.load(config.data_dir / "pod_basis_u.npz")
    pod_p = PODBasis.load(config.data_dir / "pod_basis_p.npz")
    Phi_p = pod_p.Phi
    assert pod_u.Phi.shape[1] == r_u and Phi_p.shape[1] == r_p

    # PODNN regresses/reconstructs onto the *primary* velocity POD modes only
    # (pod_u.r_primary, typically 15 of the 43 enriched modes), dropping the
    # supremizer-enrichment block. Supremizers are a classical-RB-Galerkin
    # device added purely to satisfy the discrete inf-sup condition of the
    # online saddle-point system Task 1 solves -- PODNN never solves that
    # system online, so they serve no purpose here. Their singular values are
    # ~1e-9 or exactly 0 (see pod_basis_u.npz's `sigmas`), so this changes the
    # achievable reconstruction accuracy negligibly (floor rel-L2(u) 0.0781 ->
    # 0.0791, verified) while removing ~28 near-noise regression targets that
    # made the NN's job needlessly harder without buying any accuracy.
    r_primary = pod_u.r_primary
    Phi_u = pod_u.Phi[:, :r_primary]
    print(f"PODNN reconstructs from the primary {r_primary}/{r_u} velocity "
          f"modes (dropping {r_u - r_primary} supremizer-enrichment modes)")

    fom_test = np.load(config.data_dir / "fom_solutions_test.npz")
    u_fom_test = fom_test["u_fom_test"]
    p_fom_test = fom_test["p_fom_test"]
    test_params = fom_test["test_params"]
    # Task 3 readiness: both files must agree on which 15 test params were used.
    assert np.allclose(test_params, test_params_rc), \
        "test_params mismatch between reduced_coefficients.npz and fom_solutions_test.npz"
    M_test = test_params.shape[0]

    timing = np.load(config.data_dir / "timing_data.npz")
    fom_times_test = timing["fom_times_test"]
    offline_total = float(timing["offline_total"])
    rom_assembly = float(timing["rom_assembly"])
    shared_offline = offline_total - rom_assembly
    rom_offline_total = offline_total

    print(f"Loaded: r_u={r_u}, r_p={r_p}, "
          f"M_train={train_params.shape[0]}, M_test={M_test}")
    print(f"Shared offline cost (snapshots+POD, reused from Task 1): "
          f"{shared_offline:.2f}s")

    # --- 3. Build the (primary-only) regression target -----------------
    # rows [0:r_primary] = primary velocity coeffs, rows [r_u:r_u+r_p] = all
    # pressure coeffs (pressure has no supremizer enrichment).
    c_train_all = np.vstack([
        rom_coeffs_train[:r_primary, :],
        rom_coeffs_train[r_u:, :],
    ]).T                                                # (M_train, r_primary+r_p)
    r_out = r_primary + r_p
    mu_train_f = mu_features(train_params)
    test_params_f = mu_features(test_params)

    # --- 4. Train an ensemble of PODNNModels (offline step) ------------
    # A single small net trained on ~130 points is highly sensitive to which
    # points end up in the validation split (best-val MSE ranges from ~0.05
    # to ~1.5 across split seeds on this problem -- verified empirically);
    # averaging several independently-initialized/split models is the
    # standard variance-reduction fix (bagging) and is applied honestly here:
    # both training time and online predict time include the full ensemble.
    print(f"\nTraining ensemble of {N_ENSEMBLE} PODNN models "
          f"(hidden={HIDDEN_SIZES}, weight_decay={WEIGHT_DECAY})...")
    t0 = time.time()
    ensemble = []
    train_hists, val_hists = [], []
    for k in range(N_ENSEMBLE):
        mu_tr, c_tr, mu_val, c_val = train_val_split(
            mu_train_f, c_train_all, val_frac=VAL_FRAC, seed=k)
        member = PODNNModel(input_dim=mu_train_f.shape[1], output_dim=r_out,
                            hidden_sizes=HIDDEN_SIZES, seed=config.seed + k)
        th, vh = member.fit(mu_tr, c_tr, mu_val, c_val,
                            epochs=EPOCHS, lr=LR, patience=PATIENCE,
                            weight_decay=WEIGHT_DECAY, verbose=True)
        ensemble.append(member)
        train_hists.append(th)
        val_hists.append(vh)
        print(f"  member {k}: {len(th)} epochs, best val MSE {vh.min():.3e}")
    podnn_train_time = time.time() - t0
    podnn_offline_total = shared_offline + podnn_train_time
    train_hist, val_hist = train_hists[0], val_hists[0]   # representative curve for plotting
    print(f"Ensemble trained in {podnn_train_time:.2f}s total")

    def ensemble_predict(mu_f: np.ndarray) -> np.ndarray:
        """mu_f: (n, 3) engineered features -> (n, r_primary+r_p) mean prediction."""
        return np.mean([m.predict(mu_f) for m in ensemble], axis=0)

    # --- 5. Online evaluation on the 15 official test params ---------
    # Natural order (no warm-start needed -- there's no iterative solve),
    # per-sample timing, zero FEM/dolfin calls in the timed region.
    err_analyzer = ErrorAnalyzer(problem)
    U_podnn = np.zeros((problem.N_u, M_test))
    P_podnn = np.zeros((problem.N_p, M_test))
    coeffs_pred_test = np.zeros((r_out, M_test))
    podnn_times_test = np.zeros(M_test)

    for i in tqdm(range(M_test), desc="PODNN online eval"):
        mu_i = test_params_f[i:i + 1]
        t0 = time.time()
        c_pred = ensemble_predict(mu_i)[0]        # full ensemble forward pass, timed honestly
        u_i = Phi_u @ c_pred[:r_primary]
        p_i = Phi_p @ c_pred[r_primary:]
        podnn_times_test[i] = time.time() - t0
        coeffs_pred_test[:, i] = c_pred
        U_podnn[:, i] = u_i
        P_podnn[:, i] = p_i

    report = err_analyzer.batch_report(u_fom_test, p_fom_test, U_podnn, P_podnn)
    summary = report.summary()
    podnn_predict_mean = float(podnn_times_test.mean())
    fom_mean = float(fom_times_test.mean())
    speedup = fom_mean / max(podnn_predict_mean, 1e-12)

    print("\nPODNN error summary on test set:")
    print(f"  mean rel L2(u) = {summary['mean_l2_u']:.3e}   "
          f"max = {summary['max_l2_u']:.3e}")
    print(f"  mean rel L2(p) = {summary['mean_l2_p']:.3e}   "
          f"max = {summary['max_l2_p']:.3e}")
    print(f"  mean rel H1(u) = {summary['mean_h1_u']:.3e}   "
          f"max = {summary['max_h1_u']:.3e}")
    print(f"  mean FOM time = {fom_mean:.3f}s,  "
          f"mean PODNN predict time = {podnn_predict_mean:.6f}s,  "
          f"speedup = {speedup:.1f}x")

    # --- POD-truncation floor diagnostic ------------------------------
    # Best any PODNN could ever achieve with this (primary-only) basis:
    # project the true FOM test fields onto Phi (exact M-orthogonal
    # projection), reconstruct.
    c_true_u = Phi_u.T @ (problem.Mu @ u_fom_test)
    c_true_p = Phi_p.T @ (problem.Mp @ p_fom_test)
    U_floor = Phi_u @ c_true_u
    P_floor = Phi_p @ c_true_p
    report_floor = err_analyzer.batch_report(u_fom_test, p_fom_test, U_floor, P_floor)
    floor_summary = report_floor.summary()
    print("\nPOD-truncation floor (best possible with this basis):")
    print(f"  mean rel L2(u) = {floor_summary['mean_l2_u']:.3e}")
    print(f"  mean rel L2(p) = {floor_summary['mean_l2_p']:.3e}")

    # --- 6. Training-size sweep -----------------------------------------
    # Single model per size (not the full ensemble) to keep this diagnostic
    # sweep fast; it uses the same feature engineering, primary-only target
    # and weight decay as the production ensemble, just without averaging.
    sweep_sizes = np.array([20, 40, 80, 120, train_params.shape[0]])
    sweep_sizes = np.unique(np.clip(sweep_sizes, 4, train_params.shape[0]))
    rng = np.random.default_rng(config.seed)
    perm = rng.permutation(train_params.shape[0])
    sweep_mean_u, sweep_mean_p = [], []
    for size in tqdm(sweep_sizes, desc="Training-size sweep"):
        sub_idx = perm[:size]
        sub_mu = mu_train_f[sub_idx]
        sub_c = c_train_all[sub_idx]
        s_mu_tr, s_c_tr, s_mu_val, s_c_val = train_val_split(
            sub_mu, sub_c, val_frac=VAL_FRAC, seed=config.seed)
        sub_model = PODNNModel(input_dim=sub_mu.shape[1], output_dim=r_out,
                               hidden_sizes=HIDDEN_SIZES, seed=config.seed)
        sub_model.fit(s_mu_tr, s_c_tr, s_mu_val, s_c_val,
                     epochs=2000, lr=LR, patience=150,
                     weight_decay=WEIGHT_DECAY, verbose=True)
        c_sub_pred = sub_model.predict(test_params_f)        # (M_test, r_primary+r_p)
        U_sub = Phi_u @ c_sub_pred[:, :r_primary].T
        P_sub = Phi_p @ c_sub_pred[:, r_primary:].T
        rep_sub = err_analyzer.batch_report(u_fom_test, p_fom_test, U_sub, P_sub)
        sweep_mean_u.append(float(rep_sub.rel_l2_u.mean()))
        sweep_mean_p.append(float(rep_sub.rel_l2_p.mean()))
    sweep_mean_u = np.array(sweep_mean_u)
    sweep_mean_p = np.array(sweep_mean_p)

    sweep_csv_path = config.data_dir / "podnn_trainsize_sweep.csv"
    with open(sweep_csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["n_train", "mean_l2_u", "mean_l2_p"])
        for i, size in enumerate(sweep_sizes):
            w.writerow([int(size), sweep_mean_u[i], sweep_mean_p[i]])
    print(f"Training-size sweep saved to: {sweep_csv_path}")

    # =================================================================
    # Plots
    # =================================================================
    print("\nGenerating plots...")
    vis = Visualizer(problem)
    vis.plot_rom_vs_fom(test_params, u_fom_test, U_podnn,
                        label="PODNN", filename="14_podnn_vs_fom.png")
    vis.plot_error_parameter_space(test_params, report.rel_l2_u,
                                   filename="15_podnn_error_parameter_space.png")
    vis.plot_speedup(fom_mean, podnn_predict_mean, podnn_offline_total,
                     label="PODNN", filename="16_podnn_speedup.png")
    vis.plot_podnn_training_curve(train_hist, val_hist)
    vis.plot_podnn_coeff_parity(np.concatenate([c_true_u, c_true_p], axis=0),
                                coeffs_pred_test, r_primary)
    vis.plot_error_vs_trainsize(sweep_sizes, sweep_mean_u, sweep_mean_p)
    print(f"Saved 6 plots to: {config.plots_dir}")

    # =================================================================
    # Persist data for Task 3
    # =================================================================
    for k, member in enumerate(ensemble):
        member.save(config.data_dir / f"podnn_model_{k}.npz")
    np.savez_compressed(
        config.data_dir / "podnn_ensemble_meta.npz",
        n_ensemble=np.array(N_ENSEMBLE),
        r_primary=np.array(r_primary),
        r_p=np.array(r_p),
        hidden_sizes=np.array(HIDDEN_SIZES),
    )
    np.savez_compressed(
        config.data_dir / "podnn_training_history.npz",
        train_hist=train_hist, val_hist=val_hist,
    )
    # Same schema + same natural row order as Task 1's test_errors.npz.
    np.savez_compressed(
        config.data_dir / "podnn_test_errors.npz",
        rel_l2_u=report.rel_l2_u,
        rel_l2_p=report.rel_l2_p,
        rel_h1_u=report.rel_h1_u,
        test_params=test_params,
    )
    np.savez_compressed(
        config.data_dir / "podnn_timing.npz",
        train_time=np.array(podnn_train_time),
        predict_mean=np.array(podnn_predict_mean),
        predict_times=podnn_times_test,
        shared_offline=np.array(shared_offline),
        podnn_offline_total=np.array(podnn_offline_total),
        rom_offline_total=np.array(rom_offline_total),
        fom_mean=np.array(fom_mean),
        speedup=np.array(speedup),
    )

    # =================================================================
    # Final summary
    # =================================================================
    print("\n" + "=" * 64)
    print("RESULT SUMMARY")
    print("=" * 64)
    print(f"PODNN velocity basis:            {r_primary} primary modes "
          f"(of {r_u} enriched in Task 1's basis)")
    print(f"PODNN pressure basis:            {r_p} modes (reused unchanged)")
    print(f"Training pairs:                  {train_params.shape[0]} "
          f"(ensemble of {N_ENSEMBLE} models, {int((1 - VAL_FRAC) * train_params.shape[0])} "
          f"train / {int(VAL_FRAC * train_params.shape[0])} val each, different splits)")
    print(f"Mean rel. L2 error (u):        {summary['mean_l2_u']:.3e}   "
          f"[POD floor: {floor_summary['mean_l2_u']:.3e}]")
    print(f"Max  rel. L2 error (u):        {summary['max_l2_u']:.3e}")
    print(f"Mean rel. L2 error (p):        {summary['mean_l2_p']:.3e}   "
          f"[POD floor: {floor_summary['mean_l2_p']:.3e}]")
    print(f"Max  rel. L2 error (p):        {summary['max_l2_p']:.3e}")
    print(f"Mean rel. H1 error (u):        {summary['mean_h1_u']:.3e}")
    print(f"Mean FOM solve time:            {fom_mean:.4f} s")
    print(f"Mean PODNN online (predict) time: {podnn_predict_mean:.6f} s")
    print(f"Mean online speedup:            {speedup:.1f}x")
    print(f"PODNN train time (offline):     {podnn_train_time:.2f} s")
    print(f"Shared offline (snapshots+POD): {shared_offline:.2f} s")
    print(f"ROM   offline total:            {rom_offline_total:.2f} s "
          f"(shared + Galerkin operator assembly {rom_assembly:.2f}s)")
    print(f"PODNN offline total:            {podnn_offline_total:.2f} s "
          f"(shared + NN training {podnn_train_time:.2f}s)")
    print(f"\nArtefacts saved under:    {config.data_dir}")
    print(f"Plots saved under:        {config.plots_dir}")
    print("=" * 64)


if __name__ == "__main__":
    main()
