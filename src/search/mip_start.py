"""Validated full-vector MIP starts for CPLEX-backed Pin-FS solves."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .states import PinFSProblemData, RestrictedSolveResult


_EFFORT_LEVELS = {
    "auto",
    "check_feas",
    "solve_fixed",
    "solve_mip",
    "repair",
    "no_check",
}


@dataclass
class MIPStartData:
    """A complete decision vector to submit as a CPLEX MIP start."""

    values: np.ndarray
    effort_level: str = "auto"
    name: str = "external_start"

    def __post_init__(self) -> None:
        values = np.asarray(self.values, dtype=float)
        if values.ndim != 1:
            raise ValueError("MIP-start values must be a one-dimensional vector")
        if not np.isfinite(values).all():
            raise ValueError("MIP-start values must all be finite")
        effort = str(self.effort_level).strip().lower().replace("-", "_")
        if effort not in _EFFORT_LEVELS:
            choices = ", ".join(sorted(_EFFORT_LEVELS))
            raise ValueError(f"unsupported MIP-start effort level; choose one of: {choices}")
        name = str(self.name).strip()
        if not name:
            raise ValueError("MIP-start name must not be empty")
        self.values = values.copy()
        self.effort_level = effort
        self.name = name


def validate_mip_start(
    mip_start: MIPStartData,
    problem: PinFSProblemData,
    *,
    tolerance: float = 1e-7,
    check_constraints: bool = False,
) -> np.ndarray:
    """Validate size, variable bounds, binaries, and optionally all rows."""
    if not isinstance(mip_start, MIPStartData):
        raise TypeError("mip_start must be an MIPStartData instance")
    values = np.asarray(mip_start.values, dtype=float)
    if values.shape != problem.c.shape:
        raise ValueError(
            f"MIP-start length {values.size} does not match the model's "
            f"{problem.number_of_variables} variables"
        )
    if not np.isfinite(values).all():
        raise ValueError("MIP-start values must all be finite")
    below = np.isfinite(problem.lower_bounds) & (values < problem.lower_bounds - tolerance)
    above = np.isfinite(problem.upper_bounds) & (values > problem.upper_bounds + tolerance)
    if np.any(below | above):
        index = int(np.flatnonzero(below | above)[0])
        raise ValueError(f"MIP-start value at index {index} violates its variable bounds")

    binary_indices = np.flatnonzero(problem.integrality)
    if binary_indices.size:
        binary_values = values[binary_indices]
        rounded = np.rint(binary_values)
        invalid = (np.abs(binary_values - rounded) > tolerance) | ~np.isin(rounded, [0.0, 1.0])
        if np.any(invalid):
            index = int(binary_indices[np.flatnonzero(invalid)[0]])
            raise ValueError(f"MIP-start value at index {index} is not binary")

    if check_constraints:
        activity = np.asarray(problem.constraint_matrix @ values).reshape(-1)
        below_row = np.isfinite(problem.constraint_lower) & (
            activity < problem.constraint_lower - tolerance
        )
        above_row = np.isfinite(problem.constraint_upper) & (
            activity > problem.constraint_upper + tolerance
        )
        if np.any(below_row | above_row):
            row = int(np.flatnonzero(below_row | above_row)[0])
            raise ValueError(f"MIP start violates model constraint row {row}")
    return values.copy()


def result_to_mip_start(
    result: RestrictedSolveResult,
    problem: PinFSProblemData,
    *,
    effort_level: str = "auto",
    name: str = "restricted_solution",
    tolerance: float = 1e-7,
) -> MIPStartData:
    """Convert a feasible restricted solution into a validated full-model start."""
    w = np.asarray(result.coefficients, dtype=float)
    z = np.asarray(result.z, dtype=float)
    xi = np.asarray(result.xi, dtype=float)
    v = np.asarray(result.v, dtype=float)
    expected = {
        "coefficients": (w, problem.w_slice.stop - problem.w_slice.start),
        "z": (z, problem.z_slice.stop - problem.z_slice.start),
        "xi": (xi, problem.xi_slice.stop - problem.xi_slice.start),
        "v": (v, problem.v_slice.stop - problem.v_slice.start),
    }
    for label, (array, length) in expected.items():
        if array.shape != (length,):
            raise ValueError(f"result {label} has shape {array.shape}; expected ({length},)")
        if not np.isfinite(array).all():
            raise ValueError(f"result {label} contains NaN or infinite values")
    if not np.isfinite(result.intercept):
        raise ValueError("result intercept must be finite")
    if np.any(z + tolerance < np.abs(w)):
        raise ValueError("cannot create MIP start: z must dominate |w|")

    rounded_v = np.rint(v)
    if np.any(np.abs(v - rounded_v) > tolerance) or np.any(~np.isin(rounded_v, [0.0, 1.0])):
        raise ValueError("cannot create MIP start: v must be binary")
    if np.any((rounded_v == 0) & (np.abs(w) > tolerance)):
        raise ValueError("cannot create MIP start: v=0 requires w=0")
    if int(rounded_v.sum()) > problem.feature_budget:
        raise ValueError("cannot create MIP start: selected variables exceed feature budget B")

    values = np.zeros(problem.number_of_variables, dtype=float)
    values[problem.w_slice] = w
    values[problem.b_index] = float(result.intercept)
    values[problem.z_slice] = z
    values[problem.xi_slice] = xi
    values[problem.v_slice] = rounded_v
    mip_start = MIPStartData(values=values, effort_level=effort_level, name=name)
    validate_mip_start(mip_start, problem, tolerance=tolerance, check_constraints=True)
    return mip_start
