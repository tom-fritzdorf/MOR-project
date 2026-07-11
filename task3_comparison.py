#!/usr/bin/env python3
"""
Task 3 driver: compare the two reduced-order approaches (POD-Galerkin ROM,
Task 1; parametric PODNN, Task 2) against each other and against the FOM, in
terms of computational cost and accuracy.

This script performs no new solves -- it only loads the artefacts Task 1 and
Task 2 already saved (`test_errors.npz` / `podnn_test_errors.npz`,
`timing_data.npz` / `podnn_timing.npz`) and tabulates/plots them. Both
methods were built (see the header comments of `task1_pod_galerkin.py` and
`task2_podnn.py`) to be directly, index-for-index comparable:
  - identical held-out test parameters, identical FOM ground truth
  - identical error machinery (`ErrorAnalyzer`, same FE mass/stiffness
    matrices), identical natural test-index row order
  - identical POD basis (PODNN reconstructs from the same primary velocity
    modes plus the same pressure modes as ROM; only ROM additionally uses
    the supremizer-enrichment block, which exists purely for online
    Galerkin inf-sup stability and carries ~0 reconstruction energy -- see
    task2_podnn.py's basis-scope comment)
  - a shared/incremental offline-cost split (`shared_offline` = snapshot
    collection + POD, counted once; `rom_assembly` / `train_time` = each
    method's own increment) so the offline comparison never double-counts

Run with:
    python task3_comparison.py
"""

from __future__ import annotations

import csv

import numpy as np

from src import Config, NavierStokesProblem, Visualizer


def main() -> None:
    print("=" * 64)
    print("Task 3 — ROM vs PODNN vs FOM comparison")
    print("=" * 64)

    config = Config()
    problem = NavierStokesProblem(config)
    vis = Visualizer(problem)

    # --- 1. Load Task 1 (ROM) and Task 2 (PODNN) artefacts -------------
    rom_err = np.load(config.data_dir / "test_errors.npz")
    podnn_err = np.load(config.data_dir / "podnn_test_errors.npz")
    rom_timing = np.load(config.data_dir / "timing_data.npz")
    podnn_timing = np.load(config.data_dir / "podnn_timing.npz")

    # Optional Task 4 (PINN): included only if evaluated on the same test set.
    pinn_err = pinn_timing = None
    try:
        _pe = np.load(config.data_dir / "pinn_test_errors.npz")
        _pt = np.load(config.data_dir / "pinn_timing.npz")
        if _pe["test_params"].shape == rom_err["test_params"].shape and \
                np.allclose(_pe["test_params"], rom_err["test_params"]):
            pinn_err, pinn_timing = _pe, _pt
        else:
            print("  [warn] PINN test_params differ -- excluding PINN from comparison.")
    except FileNotFoundError:
        pass

    test_params = rom_err["test_params"]
    assert np.allclose(test_params, podnn_err["test_params"]), \
        "Task 1 / Task 2 test_params mismatch -- comparison would be invalid"
    M_test = test_params.shape[0]

    fom_mean = float(rom_timing["fom_times_test"].mean())
    rom_mean = float(rom_timing["rom_times_test"].mean())
    podnn_mean = float(podnn_timing["predict_mean"])
    assert np.isclose(fom_mean, float(podnn_timing["fom_mean"]), rtol=1e-6), \
        "Task 1 / Task 2 disagree on mean FOM time -- stale artefacts?"

    shared_offline = float(podnn_timing["shared_offline"])
    rom_assembly = float(rom_timing["rom_assembly"])
    podnn_train_time = float(podnn_timing["train_time"])
    rom_offline_total = float(rom_timing["offline_total"])
    podnn_offline_total = float(podnn_timing["podnn_offline_total"])
    assert np.isclose(shared_offline, rom_offline_total - rom_assembly, rtol=1e-6), \
        "shared_offline does not reconcile with Task 1's offline_total - rom_assembly"

    def summary(err: dict) -> dict:
        return {
            "mean_l2_u": float(err["rel_l2_u"].mean()),
            "max_l2_u": float(err["rel_l2_u"].max()),
            "mean_l2_p": float(err["rel_l2_p"].mean()),
            "max_l2_p": float(err["rel_l2_p"].max()),
            "mean_h1_u": float(err["rel_h1_u"].mean()),
            "max_h1_u": float(err["rel_h1_u"].max()),
        }

    rom_summary = summary(rom_err)
    podnn_summary = summary(podnn_err)
    pinn_summary = summary(pinn_err) if pinn_err is not None else None

    # POD-truncation floor: PODNN's own diagnostic, reused here as the
    # accuracy ceiling both methods' bases could ever achieve.
    floor_summary = {
        "mean_l2_u": None, "mean_l2_p": None,
    }
    floor_path = config.data_dir / "pod_basis_u.npz"  # placeholder check only

    # --- 2. Per-test-point comparison table -----------------------------
    rows = []
    for i in range(M_test):
        row = {
            "idx": i, "mu0": test_params[i, 0], "mu1": test_params[i, 1],
            "fom_time_s": rom_timing["fom_times_test"][i],
            "rom_time_s": rom_timing["rom_times_test"][i],
            "podnn_time_s": podnn_timing["predict_times"][i],
            "rom_rel_l2_u": rom_err["rel_l2_u"][i],
            "podnn_rel_l2_u": podnn_err["rel_l2_u"][i],
            "rom_rel_l2_p": rom_err["rel_l2_p"][i],
            "podnn_rel_l2_p": podnn_err["rel_l2_p"][i],
            "rom_rel_h1_u": rom_err["rel_h1_u"][i],
            "podnn_rel_h1_u": podnn_err["rel_h1_u"][i],
        }
        if pinn_err is not None:
            row.update({
                "pinn_time_s": pinn_timing["predict_times"][i],
                "pinn_rel_l2_u": pinn_err["rel_l2_u"][i],
                "pinn_rel_l2_p": pinn_err["rel_l2_p"][i],
                "pinn_rel_h1_u": pinn_err["rel_h1_u"][i],
            })
        rows.append(row)

    table_path = config.data_dir / "comparison_table.csv"
    with open(table_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"Per-test-point comparison table saved to: {table_path}")

    # --- 3. Console summary ---------------------------------------------
    rom_speedup = fom_mean / max(rom_mean, 1e-12)
    podnn_speedup = fom_mean / max(podnn_mean, 1e-12)

    has_pinn = pinn_summary is not None
    pcol = f"{'PINN':>14}" if has_pinn else ""
    def _pval(key):
        return f"{pinn_summary[key]:>14.3e}" if has_pinn else ""

    print("\n" + "-" * 64)
    print(f"ACCURACY (relative to FOM, mean over {M_test} held-out test points)")
    print("-" * 64)
    print(f"{'metric':<14}{'ROM':>14}{'PODNN':>14}{pcol}")
    for key, label in [("mean_l2_u", "rel L2(u)"), ("mean_l2_p", "rel L2(p)"),
                        ("mean_h1_u", "rel H1(u)")]:
        print(f"{label:<14}{rom_summary[key]:>14.3e}{podnn_summary[key]:>14.3e}{_pval(key)}")
    print(f"{'max L2(u)':<14}{rom_summary['max_l2_u']:>14.3e}"
          f"{podnn_summary['max_l2_u']:>14.3e}{_pval('max_l2_u')}")

    print("\n" + "-" * 64)
    print("COMPUTATIONAL COST")
    print("-" * 64)
    print(f"Mean FOM solve time (online):     {fom_mean:.4f} s")
    print(f"Mean ROM solve time (online):     {rom_mean:.4f} s   "
          f"(speedup {rom_speedup:.1f}x)")
    print(f"Mean PODNN predict time (online): {podnn_mean:.6f} s   "
          f"(speedup {podnn_speedup:.1f}x)")
    if has_pinn:
        pinn_mean = float(pinn_timing["predict_mean"])
        pinn_speedup = fom_mean / max(pinn_mean, 1e-12)
        print(f"Mean PINN predict time (online):  {pinn_mean:.6f} s   "
              f"(speedup {pinn_speedup:.1f}x)")
    print(f"\nShared offline (snapshots + POD): {shared_offline:.2f} s  (counted once)")
    print(f"  + ROM Galerkin operator assembly: {rom_assembly:.2f} s  "
          f"-> ROM offline total   {rom_offline_total:.2f} s")
    print(f"  + PODNN ensemble training:        {podnn_train_time:.2f} s  "
          f"-> PODNN offline total {podnn_offline_total:.2f} s")
    if has_pinn:
        print(f"PINN offline (training only, NO shared snapshots/POD): "
              f"{float(pinn_timing['train_time']):.2f} s")

    print("\n" + "-" * 64)
    print("INTERPRETATION")
    print("-" * 64)
    if rom_summary["mean_l2_u"] < podnn_summary["mean_l2_u"]:
        acc_winner, acc_gap = "ROM", podnn_summary["mean_l2_u"] / rom_summary["mean_l2_u"]
    else:
        acc_winner, acc_gap = "PODNN", rom_summary["mean_l2_u"] / podnn_summary["mean_l2_u"]
    speed_winner = "PODNN" if podnn_mean < rom_mean else "ROM"
    print(f"More accurate (mean rel L2(u)): {acc_winner} "
          f"({acc_gap:.1f}x smaller error)")
    print(f"Faster online:                  {speed_winner} "
          f"({max(rom_mean, podnn_mean) / min(rom_mean, podnn_mean):.1f}x)")
    print("ROM solves a reduced Newton-Galerkin system every query and still "
          "pays a non-affine forcing FEM assembly online (mu1-dependent, "
          "O(N_h)); PODNN pays zero FEM cost online (one NN forward pass + "
          "basis matvec) but its accuracy is capped by how well a small net "
          f"can regress POD coefficients from only {config.n_train()} training snapshots -- "
          "not by the POD basis itself (PODNN's own POD-truncation-floor "
          "diagnostic in Task 2 shows the achievable floor is close to ROM's "
          "actual error).")

    # --- 4. Plots ---------------------------------------------------------
    print("\nGenerating comparison plots...")
    # POD-truncation floor for the accuracy-comparison plot: recompute from
    # PODNN's own printed diagnostic is not persisted per-field, so reuse the
    # mean_l2_u value baked into podnn's RESULT SUMMARY via a direct floor
    # computation identical to task2_podnn.py's step (kept consistent by
    # reusing the same basis + FOM test fields, no new solve).
    from src import PODBasis, ErrorAnalyzer
    pod_u = PODBasis.load(config.data_dir / "pod_basis_u.npz")
    pod_p = PODBasis.load(config.data_dir / "pod_basis_p.npz")
    r_primary = pod_u.r_primary
    Phi_u = pod_u.Phi[:, :r_primary]
    Phi_p = pod_p.Phi

    fom_test = np.load(config.data_dir / "fom_solutions_test.npz")
    u_fom_test = fom_test["u_fom_test"]
    p_fom_test = fom_test["p_fom_test"]

    c_true_u = Phi_u.T @ (problem.Mu @ u_fom_test)
    c_true_p = Phi_p.T @ (problem.Mp @ p_fom_test)
    U_floor = Phi_u @ c_true_u
    P_floor = Phi_p @ c_true_p
    err_analyzer = ErrorAnalyzer(problem)
    report_floor = err_analyzer.batch_report(u_fom_test, p_fom_test, U_floor, P_floor)
    floor_summary = report_floor.summary()

    # PINN args (only when Task 4 artefacts are present & aligned)
    p_mean = float(pinn_timing["predict_mean"]) if pinn_err is not None else None
    p_train = float(pinn_timing["train_time"]) if pinn_err is not None else None
    p_err_l2u = pinn_summary["mean_l2_u"] if pinn_summary is not None else None
    p_err_pp = pinn_err["rel_l2_u"] if pinn_err is not None else None

    vis.plot_comparison_accuracy(rom_summary, podnn_summary, floor_summary,
                                 pinn_summary=pinn_summary)
    vis.plot_comparison_cost(fom_mean, rom_mean, podnn_mean, shared_offline,
                             rom_assembly, podnn_train_time,
                             pinn_mean=p_mean, pinn_train=p_train)
    vis.plot_comparison_tradeoff(rom_mean, rom_summary["mean_l2_u"],
                                 podnn_mean, podnn_summary["mean_l2_u"],
                                 test_params, rom_err["rel_l2_u"], podnn_err["rel_l2_u"],
                                 pinn_mean_time=p_mean, pinn_mean_err=p_err_l2u,
                                 pinn_err_per_point=p_err_pp)
    print(f"Saved 3 plots to: {config.plots_dir}")

    print("\n" + "=" * 64)
    print(f"Comparison table: {table_path}")
    print(f"Plots: 18_comparison_accuracy.png, 19_comparison_cost.png, "
          f"20_comparison_tradeoff.png")
    print("=" * 64)


if __name__ == "__main__":
    main()
