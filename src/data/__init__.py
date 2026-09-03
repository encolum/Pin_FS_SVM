"""Validated benchmark loading, preprocessing, and deterministic corruptions."""

from .synthetic import generate_synthetic_instance
from .preprocessing import FittedPreprocessor, fit_preprocessor, transform_partition
from .data_loader import (
    load_benchmark_dataset,
    load_basehock, load_colon, load_gina, load_hiva, load_hill_valley, load_madelon,
)
from .benchmark_data import (
    audit_benchmark_datasets,
    load_solver_ready_benchmark, audit_solver_ready_benchmarks,
)

__all__ = [
    "generate_synthetic_instance",
    "load_benchmark_dataset",
    "load_basehock", "load_colon", "load_gina", "load_hiva", "load_hill_valley", "load_madelon",
    "audit_benchmark_datasets",
    "load_solver_ready_benchmark", "audit_solver_ready_benchmarks",
    "FittedPreprocessor", "fit_preprocessor", "transform_partition",
]
