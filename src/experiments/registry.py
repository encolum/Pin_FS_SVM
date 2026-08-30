"""Corrected model registry; no legacy class is reachable from the v2 runner."""

from __future__ import annotations

from typing import Any

from src.models.corrected.ablations import PinballCardinalitySVM, PinballL1SVM
from src.models.corrected.budgeted_milp_svm import BudgetedMILPSVM
from src.models.corrected.fisher_l1_svm import FisherL1SVM
from src.models.corrected.l1_svm import L1SVM
from src.models.corrected.l1_svm_rfe import L1SVMRFE
from src.models.corrected.l2_svm import L2SVM
from src.models.corrected.pin_fs_svm import PinFSSVM
from src.models.corrected.pin_svm import PinSVM


MODEL_NAMES = {
    "l1_svm", "l2_svm", "pin_svm", "pin_fs_svm", "budgeted_milp_svm",
    "fisher_l1_svm", "l1_svm_rfe", "pinball_l1_svm", "pinball_cardinality_svm",
}
MILP_MODEL_NAMES = {"pin_fs_svm", "budgeted_milp_svm", "pinball_cardinality_svm"}


def create_model(model_name: str, parameters: dict[str, Any], config: dict[str, Any], *, seed: int):
    solver = config.get("solver", {})
    common_solver = {
        "time_limit": solver.get("time_limit"),
        "backend": solver.get("backend", "scipy"),
        "threads": int(solver.get("threads", 1)),
    }
    bounds = config.get("coefficient_bounds", {})
    milp = {
        **common_solver,
        "mip_gap": solver.get("mip_gap"),
        "lower_bound": bounds.get("lower"),
        "upper_bound": bounds.get("upper"),
    }
    if model_name == "l1_svm":
        return L1SVM(**parameters, **common_solver)
    if model_name == "l2_svm":
        return L2SVM(**parameters, **common_solver)
    if model_name == "pin_svm":
        return PinSVM(**parameters, max_iter=int(solver.get("max_iter", 2000)), **common_solver)
    if model_name == "pin_fs_svm":
        return PinFSSVM(**parameters, **milp)
    if model_name == "budgeted_milp_svm":
        return BudgetedMILPSVM(**parameters, **milp)
    if model_name == "fisher_l1_svm":
        return FisherL1SVM(**parameters, **common_solver)
    if model_name == "l1_svm_rfe":
        return L1SVMRFE(**parameters, **common_solver)
    if model_name == "pinball_l1_svm":
        return PinballL1SVM(**parameters, **common_solver)
    if model_name == "pinball_cardinality_svm":
        return PinballCardinalitySVM(**parameters, **milp)
    raise KeyError(f"unknown model: {model_name}")
