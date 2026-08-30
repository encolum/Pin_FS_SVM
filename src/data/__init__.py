"""Validated dataset loading, preprocessing, and deterministic corruptions."""

from .loaders import DATASET_SPECS, audit_datasets, load_dataset
from .synthetic import generate_synthetic_instance

__all__ = ["DATASET_SPECS", "audit_datasets", "generate_synthetic_instance", "load_dataset"]
