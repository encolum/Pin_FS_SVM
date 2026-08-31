"""Validated dataset loading, preprocessing, and deterministic corruptions."""

from .loaders import DATASET_SPECS, audit_datasets, load_dataset
from .synthetic import generate_synthetic_instance
from .benchmark_loaders import (
    BENCHMARK_LOADERS, RawBenchmarkDataset, load_benchmark_dataset,
    load_basehock, load_colon, load_gina, load_hiva, load_hill_valley, load_madelon,
)
from .benchmark_validation import audit_benchmark_datasets

__all__ = [
    "DATASET_SPECS", "audit_datasets", "generate_synthetic_instance", "load_dataset",
    "BENCHMARK_LOADERS", "RawBenchmarkDataset", "load_benchmark_dataset",
    "load_basehock", "load_colon", "load_gina", "load_hiva", "load_hill_valley", "load_madelon",
    "audit_benchmark_datasets",
]
