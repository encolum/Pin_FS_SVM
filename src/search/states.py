"""Typed state shared by the restricted Pin-FS search components."""

from __future__ import annotations

from dataclasses import dataclass
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
