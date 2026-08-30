"""Reusable search primitives for restricted Pin-FS-SVM solves."""

from .mip_start import MIPStartData, result_to_mip_start, validate_mip_start
from .progress import (
    SolverProgressRecord,
    first_incumbent_time,
    time_to_target_gap,
    validate_progress_trajectory,
)
from .kernel_engine import run_kernel_search
from .objectives import primal_integral, solver_progress_summary
from .restricted_solver import build_pin_fs_problem, solve_restricted_pin_fs
from .signals import LPRelaxationResult, solve_pin_fs_relaxation
from .states import (
    FeatureState,
    KernelSearchResult,
    PinFSProblemData,
    RestrictedSolveResult,
    SearchState,
)

__all__ = [
    "MIPStartData",
    "FeatureState",
    "KernelSearchResult",
    "LPRelaxationResult",
    "PinFSProblemData",
    "RestrictedSolveResult",
    "SearchState",
    "SolverProgressRecord",
    "build_pin_fs_problem",
    "first_incumbent_time",
    "primal_integral",
    "result_to_mip_start",
    "solve_restricted_pin_fs",
    "solve_pin_fs_relaxation",
    "solver_progress_summary",
    "run_kernel_search",
    "time_to_target_gap",
    "validate_mip_start",
    "validate_progress_trajectory",
]
