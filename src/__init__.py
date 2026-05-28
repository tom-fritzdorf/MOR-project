"""MOR-project source package.

Public API used across Tasks 1-4:

    from src import (
        Config, NavierStokesProblem, FOMSolver,
        ParameterSampler, SnapshotCollector, SupremizerEnricher, PODBasis,
        ROMOperators, ROMSolver, ErrorAnalyzer, Visualizer,
    )
"""

from .config import Config
from .problem import NavierStokesProblem, ForcingExpression
from .fom import FOMSolver
from .pod import ParameterSampler, SnapshotCollector, SupremizerEnricher, PODBasis
from .rom import ROMOperators, ROMSolver
from .analysis import ErrorAnalyzer
from .visualization import Visualizer

__all__ = [
    "Config",
    "NavierStokesProblem",
    "ForcingExpression",
    "FOMSolver",
    "ParameterSampler",
    "SnapshotCollector",
    "SupremizerEnricher",
    "PODBasis",
    "ROMOperators",
    "ROMSolver",
    "ErrorAnalyzer",
    "Visualizer",
]
