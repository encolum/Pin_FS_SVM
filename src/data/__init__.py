"""Validated dataset loading, preprocessing, and deterministic corruptions."""

from .loaders import DATASET_SPECS, audit_datasets, load_dataset
from .synthetic import generate_synthetic_instance, generate_clean_synthetic_instance
from .preprocessing import FittedPreprocessor, fit_preprocessor, transform_partition
from .benchmark_loaders import (
    BENCHMARK_LOADERS, RawBenchmarkDataset, load_benchmark_dataset,
    load_basehock, load_colon, load_gina, load_hiva, load_hill_valley, load_madelon,
)
from .benchmark_validation import audit_benchmark_datasets
from .benchmark_adapter import (
    SolverReadyBenchmark, SolverReadyPartition, load_solver_ready_benchmark,
    audit_solver_ready_benchmarks,
)

__all__ = [
    "DATASET_SPECS", "audit_datasets", "generate_synthetic_instance", "load_dataset",
    "BENCHMARK_LOADERS", "RawBenchmarkDataset", "load_benchmark_dataset",
    "load_basehock", "load_colon", "load_gina", "load_hiva", "load_hill_valley", "load_madelon",
    "audit_benchmark_datasets",
    "SolverReadyBenchmark", "SolverReadyPartition", "load_solver_ready_benchmark",
    "audit_solver_ready_benchmarks",
    "generate_clean_synthetic_instance", "FittedPreprocessor", "fit_preprocessor", "transform_partition",
]
