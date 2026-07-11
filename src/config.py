"""Project-wide configuration.

A single :class:`Config` dataclass holds every knob (mesh resolution, parameter
ranges, POD energy threshold, solver tolerances, paths). Downstream tasks
(PODNN, PINNs, comparison) should construct a Config with matching values to
reproduce the FOM/ROM setup exactly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Tuple

ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=False)
class Config:
    """Global hyperparameters and output paths."""

    # --- mesh / FE ---
    mesh_n: int = 22                       # MxN UnitSquareMesh resolution (2*22*22=968 triangles)
    velocity_degree: int = 2               # P2 velocity
    pressure_degree: int = 1               # P1 pressure  (Taylor-Hood)

    # --- parameter space ---
    mu0_range: Tuple[float, float] = (0.1, 10.0)
    mu1_range: Tuple[float, float] = (1.0, 3.0)
    # Training params are drawn by Latin Hypercube Sampling (see
    # ParameterSampler.train), not a tensor grid, so these two factors now
    # only set the *total* sample count n_train = n_train_mu0 * n_train_mu1.
    # LHS matches the uniform-random test distribution, avoiding the
    # edge-clustering magnitude bias a Chebyshev grid introduced.
    n_train_mu0: int = 25
    n_train_mu1: int = 60                  # 25*60 = 1500 LHS training params
    n_test: int = 150                      # disjoint from train, full-range uniform coverage
    seed: int = 42

    # --- POD ---
    energy_threshold: float = 0.9999
    energy_levels: Tuple[float, ...] = (0.99, 0.999, 0.9999, 0.99999)

    # --- solver tolerances ---
    fom_abs_tol: float = 1e-10
    fom_rel_tol: float = 1e-8
    fom_max_iter: int = 40
    rom_tol: float = 1e-6
    rom_max_iter: int = 50

    # --- I/O ---
    plots_dir: Path = field(default_factory=lambda: ROOT / "plots")
    data_dir: Path = field(default_factory=lambda: ROOT / "data")

    def __post_init__(self) -> None:
        self.plots_dir = Path(self.plots_dir)
        self.data_dir = Path(self.data_dir)
        self.plots_dir.mkdir(parents=True, exist_ok=True)
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def n_train(self) -> int:
        return self.n_train_mu0 * self.n_train_mu1
