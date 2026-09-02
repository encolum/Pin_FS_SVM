"""Deterministic model-selection ordering shared by VeraPin experiments."""

from __future__ import annotations

from typing import Any

import numpy as np


def selection_tie_key(
    parameters: dict[str, Any], mean_selected_features: float
) -> tuple[Any, ...]:
    """Prefer sparsity, then smaller B, then deterministic parameter order."""
    budget = int(parameters.get("B", 0))
    parameter_order = tuple(
        (name, _sortable_parameter(parameters[name])) for name in sorted(parameters)
    )
    return (float(mean_selected_features), budget, parameter_order)


def _sortable_parameter(value: Any) -> tuple[int, Any]:
    if isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, bool):
        return (0, float(value))
    if isinstance(value, str):
        return (1, value)
    return (2, repr(value))
