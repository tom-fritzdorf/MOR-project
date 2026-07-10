#!/usr/bin/env python3
"""
Task 1 driver: POD-Galerkin ROM for parametric steady Navier-Stokes.

This script wires together the classes under ``src/``. All heavy lifting
(weak forms, FOM solve, snapshot collection, supremizer enrichment, POD,
ROM assembly and Newton solve, error analysis, plots) lives in the package
so Tasks 2-4 can import the same components.

Run with:
    python task1_pod_galerkin.py
"""

from __future__ import annotations

import pickle
import time

import numpy as np

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable

from src import (
    Config,
    NavierStokesProblem,
    FOMSolver,
    ParameterSampler,
    SnapshotCollector,
    SupremizerEnricher,
    PODBasis,
    ROMOperators,
    ROMSolver,
    ErrorAnalyzer,
    Visualizer,
)


def main() -> None:
    print("=" * 64)
    print("Task 1 — POD-Galerkin ROM for parametric Navier-Stokes")
    print("=" * 64)

    # --- 1. Setup ----------------------------------------------------
    config = Config()
    problem = NavierStokesProblem(config)
    fom = FOMSolver(problem)
    print(f"Mesh: {config.mesh_n}x{config.mesh_n}  |  "
          f"N_u = {problem.N_u}, N_p = {problem.N_p}")

    # --- 2. Parameter sampling --------------------------------------
    sampler = ParameterSampler(config)
    train_params, test_params = sampler.both()
    print(f"Training params: {train_params.shape[0]}   "
          f"Test params: {test_params.shape[0]}")

    # --- 3. Snapshot collection -------------------------------------
    t_offline_start = time.time()
    snapshots = SnapshotCollector(problem, fom).collect(train_params, label="train")
    print(f"Mean FOM solve (training): {snapshots.times.mean():.3f}s  "
          f"(min {snapshots.times.min():.3f}, max {snapshots.times.max():.3f})")

    # --- 4. Supremizer snapshots ------------------------------------
    sup_enricher = SupremizerEnricher(problem)
    S_sup = sup_enricher.compute(snapshots.S_p)
    print(f"Computed {S_sup.shape[1]} supremizer snapshots")

    # --- 5. POD bases (separate POD on velocity and supremizers) ----
    # pod_p first: pod_u's supremizer truncation needs r_p as a floor (see
    # PODBasis.from_velocity_and_supremizers) to guarantee enough supremizer
    # directions for the discrete inf-sup condition.
    pod_p = PODBasis.from_snapshots(snapshots.S_p, problem.Mp,
                                    energy_threshold=config.energy_threshold,
                                    name="p")
    pod_u = PODBasis.from_velocity_and_supremizers(
        snapshots.S_u, S_sup, problem.Mu,
        energy_threshold=config.energy_threshold, name="u",
        min_sup=pod_p.r,
    )
    print(f"Velocity basis: r_primary={pod_u.r_primary}, "
          f"r_sup={pod_u.r_sup}, r_u_total={pod_u.r}")
    truncations = {lvl: (pod_u.modes_for_energy(lvl), pod_p.modes_for_energy(lvl))
                   for lvl in config.energy_levels}
    for lvl, (ru, rp) in truncations.items():
        print(f"  energy {lvl:.5f}: r_u={ru}, r_p={rp}")
    print(f"Chosen: r_u={pod_u.r}, r_p={pod_p.r}  "
          f"(energy threshold {config.energy_threshold})")

    # --- 6. ROM operators -------------------------------------------
    operators = ROMOperators.assemble(problem, pod_u.Phi, pod_p.Phi)
    rom = ROMSolver(problem, operators, pod_u.Phi, pod_p.Phi)
    print(f"ROM operators assembled in {operators.assembly_time:.2f}s "
          f"(A_r {operators.A_r.shape}, B_r {operators.B_r.shape}, "
          f"C_r {operators.C_r.shape})")
    t_offline = time.time() - t_offline_start

    # --- 7. ROM coefficients on training set (for Task 2 PODNN) -----
    # POD-NN targets are the M-orthogonal projection of the *true* FOM
    # snapshot onto the POD basis (Hesthaven & Ubbiali 2018) -- i.e. what the
    # reduced coordinates of the training data actually are -- rather than a
    # re-solved ROM Galerkin system. This is exact (no Newton needed) and
    # sidesteps cold-started ROM Newton robustness issues entirely.
    rom_coeffs_train = np.zeros((operators.r_u + operators.r_p, train_params.shape[0]))
    rom_coeffs_train[:operators.r_u, :] = pod_u.Phi.T @ (problem.Mu @ snapshots.S_u)
    rom_coeffs_train[operators.r_u:, :] = pod_p.Phi.T @ (problem.Mp @ snapshots.S_p)

    # --- 8. Test-set: FOM + ROM + errors ----------------------------
    err_analyzer = ErrorAnalyzer(problem)
    M_test = test_params.shape[0]
    fom_u_test = np.zeros((problem.N_u, M_test))
    fom_p_test = np.zeros((problem.N_p, M_test))
    rom_u_test = np.zeros((problem.N_u, M_test))
    rom_p_test = np.zeros((problem.N_p, M_test))
    fom_times_test = np.zeros(M_test)
    rom_times_test = np.zeros(M_test)
    rom_coeffs_test = np.zeros((operators.r_u + operators.r_p, M_test))

    # lexsort (not plain argsort) so ties in mu0 are broken by ascending
    # mu1 -- keeps both the FOM and ROM warm-starts walking mu1 smoothly.
    test_order = np.lexsort((test_params[:, 1], -test_params[:, 0]))
    w_prev = None
    a_prev, b_prev = None, None
    for idx in tqdm(test_order, desc="Test set FOM+ROM"):
        mu0, mu1 = test_params[idx]
        fom_res = fom.solve(mu0, mu1, w_init=w_prev)
        w_prev = fom_res.w
        rom_res = rom.solve(mu0, mu1, a_init=a_prev, b_init=b_prev)
        a_prev, b_prev = rom_res.a, rom_res.b
        u_rom, p_rom = rom.reconstruct(rom_res.a, rom_res.b)

        fom_u_test[:, idx] = fom_res.u
        fom_p_test[:, idx] = fom_res.p
        rom_u_test[:, idx] = u_rom
        rom_p_test[:, idx] = p_rom
        fom_times_test[idx] = fom_res.solve_time
        rom_times_test[idx] = rom_res.solve_time
        rom_coeffs_test[:operators.r_u, idx] = rom_res.a
        rom_coeffs_test[operators.r_u:, idx] = rom_res.b

    report = err_analyzer.batch_report(fom_u_test, fom_p_test,
                                       rom_u_test, rom_p_test)
    summary = report.summary()
    mean_speedup = fom_times_test.mean() / max(rom_times_test.mean(), 1e-12)
    print("\nError summary on test set:")
    print(f"  mean rel L2(u) = {summary['mean_l2_u']:.3e}   "
          f"max = {summary['max_l2_u']:.3e}")
    print(f"  mean rel L2(p) = {summary['mean_l2_p']:.3e}   "
          f"max = {summary['max_l2_p']:.3e}")
    print(f"  mean rel H1(u) = {summary['mean_h1_u']:.3e}   "
          f"max = {summary['max_h1_u']:.3e}")
    print(f"  mean FOM time = {fom_times_test.mean():.3f}s,  "
          f"mean ROM time = {rom_times_test.mean():.4f}s,  "
          f"speedup = {mean_speedup:.1f}x")

    # --- 9. Error-vs-modes sweep (paired primary+supremizer growth) -
    # Growing supremizer modes in lockstep with primary modes keeps the
    # reduced pair inf-sup stable at every truncation level (see
    # data/infsup_sweep.csv / plot 11 for what happens without this).
    max_N = min(pod_u.r_primary, pod_u.r_sup)
    Ns = np.arange(1, max_N + 1)
    Nrs, sweep_u, sweep_p, sweep_max_u, sweep_h1_u = [], [], [], [], []
    sweep_rom_times, sweep_speedups = [], []
    # Sorted purely to give each sub-ROM's Newton solves a smooth
    # warm-start walk in mu1; array writes below still use the original
    # index i, so result ordering/alignment is unaffected.
    sweep_iter_order = np.lexsort((test_params[:, 1], -test_params[:, 0]))
    for N in tqdm(Ns, desc="Error sweep (paired)"):
        u_idx = pod_u.paired_indices(int(N))
        p_idx = np.arange(min(int(N), operators.r_p))
        sub_pod_u = pod_u.truncate_by_indices(u_idx)
        sub_pod_p = pod_p.truncate_by_indices(p_idx)
        sub_ops = operators.truncate_indices(u_idx, p_idx)
        sub_rom = ROMSolver(problem, sub_ops, sub_pod_u.Phi, sub_pod_p.Phi)
        eu_list, ep_list, eh_list, t_list = [None] * M_test, [None] * M_test, [None] * M_test, [None] * M_test
        a_prev, b_prev = None, None
        for i in sweep_iter_order:
            mu0, mu1 = test_params[i]
            r = sub_rom.solve(mu0, mu1, a_init=a_prev, b_init=b_prev)
            a_prev, b_prev = r.a, r.b
            t_list[i] = r.solve_time
            u_r, p_r = sub_rom.reconstruct(r.a, r.b)
            eu, ep, eh = err_analyzer.relative_errors(
                fom_u_test[:, i], fom_p_test[:, i], u_r, p_r,
            )
            eu_list[i] = eu; ep_list[i] = ep; eh_list[i] = eh
        Nrs.append(u_idx.size)
        sweep_u.append(float(np.mean(eu_list)))
        sweep_p.append(float(np.mean(ep_list)))
        sweep_max_u.append(float(np.max(eu_list)))
        sweep_h1_u.append(float(np.mean(eh_list)))
        mean_t = float(np.mean(t_list))
        sweep_rom_times.append(mean_t)
        sweep_speedups.append(fom_times_test.mean() / max(mean_t, 1e-12))
    Nrs          = np.array(Nrs)
    sweep_u      = np.array(sweep_u)
    sweep_p      = np.array(sweep_p)
    sweep_max_u  = np.array(sweep_max_u)
    sweep_h1_u   = np.array(sweep_h1_u)
    sweep_rom_times  = np.array(sweep_rom_times)
    sweep_speedups   = np.array(sweep_speedups)

    # --- Print comparison table ---
    fom_mean_t = fom_times_test.mean()
    print(f"\nMode comparison table (FOM mean time = {fom_mean_t:.3f} s, "
          f"{M_test} test samples):")
    hdr = (f"{'r_u':>4} {'r_p':>4} {'L2(u)mean':>11} {'L2(u)max':>11} "
           f"{'L2(p)mean':>11} {'H1(u)mean':>11} {'ROMtime(s)':>11} {'Speedup':>9}")
    print(hdr)
    print("-" * len(hdr))
    for i, N in enumerate(Ns):
        Np_s = min(int(N), operators.r_p)
        print(f"{Nrs[i]:>4} {Np_s:>4} {sweep_u[i]:>11.3e} {sweep_max_u[i]:>11.3e} "
              f"{sweep_p[i]:>11.3e} {sweep_h1_u[i]:>11.3e} "
              f"{sweep_rom_times[i]:>11.4f} {sweep_speedups[i]:>8.1f}x")

    # --- Save comparison table as CSV ---
    import csv
    csv_path = config.data_dir / "mode_comparison.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["r_u", "r_p", "mean_l2_u", "max_l2_u",
                    "mean_l2_p", "mean_h1_u", "rom_time_s", "speedup"])
        for i, N in enumerate(Ns):
            Np_s = min(int(N), operators.r_p)
            w.writerow([int(Nrs[i]), Np_s, sweep_u[i], sweep_max_u[i],
                        sweep_p[i], sweep_h1_u[i],
                        sweep_rom_times[i], sweep_speedups[i]])
    print(f"Mode comparison table saved to: {csv_path}")

    # --- 9b. Inf-sup instability diagnostic (primary-only, no supremizer) --
    # Deliberately unstable by construction: for r <= r_primary the reduced
    # velocity space contains no supremizer modes, so it is exactly
    # divergence-free (built from FOM snapshots satisfying B_h u_h = 0) and
    # cannot see the pressure gradient at all -- B_r ~ 0 to machine
    # precision, the reduced saddle-point Jacobian is singular, and Newton
    # never moves away from a=0. Kept as a standalone labeled diagnostic
    # (plot 11) that mirrors the class notes' inf-sup / supremizer section,
    # rather than contaminating the main convergence sweep above.
    rs_instab = np.arange(1, pod_u.r_primary + 1)
    eu_instab, ep_instab = [], []
    for r in tqdm(rs_instab, desc="Inf-sup instability sweep"):
        Np_s = min(int(r), operators.r_p)
        sub_pod_u = pod_u.truncate(int(r))
        sub_pod_p = pod_p.truncate(Np_s)
        sub_ops = operators.truncate(int(r), Np_s)
        sub_rom = ROMSolver(problem, sub_ops, sub_pod_u.Phi, sub_pod_p.Phi)
        eu_list, ep_list = [], []
        for i in range(M_test):
            mu0, mu1 = test_params[i]
            res = sub_rom.solve(mu0, mu1)
            u_r, p_r = sub_rom.reconstruct(res.a, res.b)
            eu, ep, _ = err_analyzer.relative_errors(
                fom_u_test[:, i], fom_p_test[:, i], u_r, p_r,
            )
            eu_list.append(eu); ep_list.append(ep)
        eu_instab.append(float(np.mean(eu_list)))
        ep_instab.append(float(np.mean(ep_list)))
    eu_instab = np.array(eu_instab)
    ep_instab = np.array(ep_instab)

    infsup_csv_path = config.data_dir / "infsup_sweep.csv"
    with open(infsup_csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["r", "mean_l2_u", "mean_l2_p"])
        for i, r in enumerate(rs_instab):
            w.writerow([int(r), eu_instab[i], ep_instab[i]])
    print(f"Inf-sup instability sweep saved to: {infsup_csv_path}")

    # --- 10. Representative Newton residual histories (Plot 9) ------
    rep_idx = int(np.argsort(test_params[:, 0])[0])    # smallest mu0 (hardest)
    mu0_rep, mu1_rep = test_params[rep_idx]
    fom_track = fom.solve(mu0_rep, mu1_rep, track_residuals=True)
    rom_track = rom.solve(mu0_rep, mu1_rep)

    # =================================================================
    # Plots
    # =================================================================
    print("\nGenerating plots...")
    vis = Visualizer(problem)

    vis.plot_mesh()

    # Plot 1: need a small wrapper so the Visualizer can warm-start FOM solves.
    def fom_for_showcase(mu0, mu1, w_init=None):
        res = fom.solve(mu0, mu1, w_init=w_init)
        return res.u, res.p, res.w

    vis.plot_fom_showcase(
        fom_for_showcase,
        params=[(0.5, 1.5), (3.0, 2.0), (8.0, 2.5)],
        labels=["Low viscosity", "Medium viscosity", "High viscosity"],
    )
    vis.plot_singular_values(pod_u.sigmas, pod_p.sigmas, config.energy_threshold)
    vis.plot_cumulative_energy(pod_u.sigmas, pod_p.sigmas, pod_u.r, pod_p.r)
    vis.plot_pod_modes(pod_u.Phi, pod_p.Phi)
    vis.plot_rom_vs_fom(test_params, fom_u_test, rom_u_test)
    vis.plot_error_vs_modes(Nrs, sweep_u, sweep_p)
    vis.plot_mode_comparison(Nrs, sweep_u, sweep_p, sweep_h1_u, sweep_speedups)
    vis.plot_error_parameter_space(test_params, report.rel_l2_u)
    vis.plot_speedup(fom_times_test.mean(), rom_times_test.mean(), t_offline)
    vis.plot_newton_convergence(fom_track.residuals or [], rom_track.residuals)
    vis.plot_infsup_instability(rs_instab, eu_instab, ep_instab, pod_u.r_primary)
    print(f"Saved 11 plots to: {config.plots_dir}")

    # =================================================================
    # Persist data for Tasks 2-4
    # =================================================================
    snapshots.save(config.data_dir / "snapshot_data.npz")
    pod_u.save(config.data_dir / "pod_basis_u.npz")
    pod_p.save(config.data_dir / "pod_basis_p.npz")
    operators.save(config.data_dir / "rom_operators.npz")

    np.savez_compressed(
        config.data_dir / "reduced_coefficients.npz",
        rom_coeffs_train=rom_coeffs_train,
        rom_coeffs_test=rom_coeffs_test,
        train_params=train_params, test_params=test_params,
        r_u=np.array(operators.r_u), r_p=np.array(operators.r_p),
    )
    np.savez_compressed(
        config.data_dir / "fom_solutions_test.npz",
        u_fom_test=fom_u_test, p_fom_test=fom_p_test,
        test_params=test_params,
    )
    np.savez_compressed(
        config.data_dir / "timing_data.npz",
        fom_times_train=snapshots.times,
        fom_times_test=fom_times_test,
        rom_times_test=rom_times_test,
        offline_total=np.array(t_offline),
        rom_assembly=np.array(operators.assembly_time),
        mean_speedup=np.array(mean_speedup),
    )
    np.savez_compressed(
        config.data_dir / "error_sweep.npz",
        Nrs=Nrs, mean_u=sweep_u, mean_p=sweep_p,
        max_u=sweep_max_u, mean_h1_u=sweep_h1_u,
        rom_times=sweep_rom_times, speedups=sweep_speedups,
    )
    np.savez_compressed(
        config.data_dir / "test_errors.npz",
        rel_l2_u=report.rel_l2_u,
        rel_l2_p=report.rel_l2_p,
        rel_h1_u=report.rel_h1_u,
        test_params=test_params,
    )

    fe_info = problem.fe_info()
    fe_info["truncations"] = truncations
    fe_info["r_u_chosen"] = pod_u.r
    fe_info["r_p_chosen"] = pod_p.r
    with open(config.data_dir / "fe_info.pkl", "wb") as f:
        pickle.dump(fe_info, f)

    # =================================================================
    # Final summary
    # =================================================================
    print("\n" + "=" * 64)
    print("RESULT SUMMARY")
    print("=" * 64)
    print(f"POD modes:               velocity r_u = {pod_u.r},  pressure r_p = {pod_p.r}")
    print(f"FE DOFs:                 N_u = {problem.N_u},  N_p = {problem.N_p}")
    print(f"Mean rel. L2 error (u):  {summary['mean_l2_u']:.3e}")
    print(f"Max  rel. L2 error (u):  {summary['max_l2_u']:.3e}")
    print(f"Mean rel. L2 error (p):  {summary['mean_l2_p']:.3e}")
    print(f"Max  rel. L2 error (p):  {summary['max_l2_p']:.3e}")
    print(f"Mean rel. H1 error (u):  {summary['mean_h1_u']:.3e}")
    print(f"Mean FOM solve time:     {fom_times_test.mean():.4f} s")
    print(f"Mean ROM online time:    {rom_times_test.mean():.4f} s")
    print(f"Mean ROM offline time:   {t_offline:.2f} s "
          f"(snapshots + POD + assembly)")
    print(f"Mean online speedup:     {mean_speedup:.1f}x")
    print(f"\nArtefacts saved under:    {config.data_dir}")
    print(f"Plots saved under:        {config.plots_dir}")
    print("=" * 64)


if __name__ == "__main__":
    main()
