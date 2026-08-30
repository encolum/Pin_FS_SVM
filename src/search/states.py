"""Typed state shared by the restricted Pin-FS search components."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
from scipy.sparse import csr_matrix

from .progress import SolverProgressRecord

if TYPE_CHECKING:
    from src.models.corrected.base import SolverDiagnostics


@dataclass
class PinFSProblemData:
    """Solver-ready representation of the paper's Pin-FS-SVM formulation."""

    c: np.ndarray
    lower_bounds: np.ndarray
    upper_bounds: np.ndarray
    constraint_matrix: csr_matrix
    constraint_lower: np.ndarray
    constraint_upper: np.ndarray
    integrality: np.ndarray

    w_slice: slice
    b_index: int
    z_slice: slice
    xi_slice: slice
    v_slice: slice

    feature_budget: int
    allowed_features: frozenset[int]

    @property
    def number_of_variables(self) -> int:
        return int(self.c.size)

    @property
    def number_of_features(self) -> int:
        return int(self.w_slice.stop - self.w_slice.start)


@dataclass
class RestrictedSolveResult:
    """Complete result of a full- or restricted-kernel Pin-FS solve."""

    objective: float
    support: set[int]
    coefficients: np.ndarray
    intercept: float
    z: np.ndarray
    xi: np.ndarray
    v: np.ndarray
    diagnostics: SolverDiagnostics
    progress: list[SolverProgressRecord]
    solve_time: float
    kernel: set[int]
    mip_start_status: str | None


@dataclass(frozen=True)
class FeatureState:
    """Normalized train-only signals exposed to every kernel policy."""

    index: int
    in_kernel: bool
    is_selected: bool
    abs_coefficient: float

    fisher_score: float
    mutual_information: float

    mean_abs_correlation: float
    max_abs_correlation: float

    lp_activation: float
    lp_abs_coefficient: float

    slack_association: float
    selection_frequency: float

    inactive_iterations: int
    kernel_age: int

    l1_abs_coefficient: float = 0.0
    pin_abs_coefficient: float = 0.0
    support_redundancy: float = 0.0


@dataclass(frozen=True)
class SearchState:
    """Search-wide state visible to policies; it contains no held-out metrics."""

    iteration: int
    current_objective: float
    best_objective: float

    current_gap: float | None
    best_bound: float | None

    kernel_size: int
    feature_budget: int
    total_features: int

    stagnation_iterations: int
    elapsed_seconds: float
    remaining_seconds: float

    C: float
    tau: float
    improved_last_iteration: bool = False


@dataclass
class KernelSearchResult:
    """Complete outcome and audit history from one kernel-search route."""

    best_result: RestrictedSolveResult
    history: list[dict]
    final_kernel: set[int]
    total_runtime: float
    initial_kernel: set[int]
    method: str
    metadata: dict = field(default_factory=dict)
